# Reference

## Schemas

### `schemas/schemas.py`

Pydantic request/response models, driving both FastAPI validation and generated API docs.

`GenerateRequest`:
- `prompt: str`, required, `min_length=1`.
- `max_tokens: int | None`, `[1, 2048]`.
- `temperature: float | None`, `[0.0, 2.0]`.
- `idempotency_key: str | None`, `max_length=200` -- see `scheduler/idempotency.py`.
- `stop: list[str] | None`, `max_length=4` (at most 4 entries) -- a `field_validator` rejects any empty string in the list. Generation halts before emitting a matching sequence; see `ContinuousScheduler._dispatch_tokens`, `InferenceEngine.generate_batch`, and `streaming/stream_manager.stream_response`.

`BatchGenerateRequest`: `requests: list[GenerateRequest]`, `min_length=1` -- each item carries its own independent `stop`/`idempotency_key`/etc.

`BatchGenerateResponse`: `outputs: list[str]`, `batch_size: int`, `queue_latency_ms: float | None`, `token_throughput_per_sec: float | None`.

`HealthResponse`: `status: str`.

`ModelSwapRequest`: `model_name: str` (required), `drain_timeout_seconds: float | None` (`> 0`; defaults to `scheduler_config.model_swap_drain_timeout_seconds` if omitted).

`ModelSwapResponse`: `model_name: str` -- the model currently loaded and serving requests.

---

## Logging

### `logger/logger.py`

Sets up structured logging for console and file output.

`setup_logger(name, level="INFO", log_file="logs/app.log") -> logging.Logger`:
- Ensures the log file's parent directory exists.
- Returns the existing logger for `name` unmodified if it already has handlers (`logger.hasHandlers()`), preventing duplicate handlers on repeated calls.
- Otherwise sets the level (`getattr(logging, level.upper())`), attaches a `StreamHandler(sys.stdout)` and a `FileHandler(log_file)`, both using the formatter `%(asctime)s | %(levelname)s | %(name)s | %(message)s`.

Used throughout the application (every module that logs calls `setup_logger(__name__, level=logging_settings.log_level, log_file=logging_settings.log_file)`).

---

## Utility Helpers

### `utils/utils.py`

`Utils` (static methods):
- `load_config(config_path)`: loads YAML via `yaml.safe_load`.
- `_save_config(config, config_path)`: writes YAML via `yaml.dump`.
- `update_config(config_path, new_config)`: shallow-merges `new_config` into the existing file's keys and saves.
- `configure_multiprocessing()`: sets `torch.multiprocessing`'s sharing strategy to `"file_system"` and registers an `atexit` handler suppressing `resource_tracker` semaphore warnings -- called first thing in `api/server.py`'s `lifespan`, before any other `torch`-touching import, to avoid noisy/harmless shutdown warnings.
- `suppress_resource_tracker_warnings()`: the `atexit` handler installed by `configure_multiprocessing()`.

Used by `settings/settings.py` (to read `config.yaml`) and `api/server.py` (multiprocessing setup).

### `utils/stop_sequences.py`

`find_stop_index(text: str, stop_sequences: Sequence[str]) -> int | None`: returns the index in `text` where the *earliest*-occurring stop sequence begins (scanning every entry in `stop_sequences` and taking the minimum match index), or `None` if none appear. The single shared implementation used by the streaming path (`streaming/stream_manager.py`), the continuous scheduler (`scheduler/continuous_scheduler.py`), and the batch engine path (`engine/generator.py`) -- kept in `utils/` rather than `scheduler/` specifically so `engine/generator.py` can use it without importing anything from the `scheduler` package (see the "decoupling" note on `GenerationRequest` in [Inference Engine](Cache-and-Engine#inference-engine)).

### `utils/errors.py`

`INTERNAL_ERROR_MESSAGE = "Internal server error"`: the single generic message sent to clients for an unexpected, otherwise-unhandled failure (e.g. a CUDA/MPS OOM during generation). Deliberately generic -- raw exception text (stack-trace-flavored messages, memory sizes, env var hints, file paths, ...) must never reach a client. Full details are always logged server-side via `logger.exception`/`logger.warning` at the point of failure; this constant is only for what actually goes out over the wire, as either an SSE `error` event's `data` (`ContinuousScheduler._fail_single_request`, see [Scheduler Layer](Scheduler-Layer#scheduler-layer)) or an `HTTPException`'s `detail` (`/generate_batch`'s and `POST /api/model`'s `500` responses in `api/routes.py`, see [API Layer](API-Layer#api-layer)). Deliberately-authored, already-safe messages (e.g. the `409`/`503`/`504` conflict/timeout details elsewhere in `api/routes.py`) don't go through this constant -- they're safe to return as-is.

### `utils/device_cache.py`

Shared helpers for releasing PyTorch's CUDA/MPS allocator-cached (but currently unused) device memory, both reactively (after a failure, when idle) and proactively (before usage ever nears the device's ceiling). Used by `engine/model_loader.py` (`reload()`), `scheduler/model_swap.py`, `scheduler/batch_scheduler.py` (`run()`), and `scheduler/continuous_scheduler.py` (`_step()`).

`DEFAULT_MEMORY_PRESSURE_THRESHOLD = 0.7`: the fraction of a device's memory budget at which `maybe_empty_device_cache` proactively clears. Conservative on purpose -- the goal is to release memory *before* it ever gets close to the device's actual ceiling (observed as e.g. `MPS backend out of memory`), not to wait for a fixed step count or for a failure to happen first.

`empty_device_cache(device)`:
- Calls `torch.cuda.empty_cache()` when `device` starts with `"cuda"` and CUDA is available, or `torch.mps.empty_cache()` when `device == "mps"` and MPS is available. No-op on CPU or an unavailable backend.
- This is a single shared implementation -- previously `engine/model_loader.py` had its own private, duplicated copy of the same logic.

`device_memory_pressure(device) -> float | None`:
- Fraction of `device`'s memory budget currently held by PyTorch (allocated + cached), or `None` if it can't be determined (CPU, an unavailable backend, or a torch build missing these APIs -- any exception is caught and treated as "unknown").
- For MPS: `torch.mps.driver_allocated_memory()` (total GPU memory the process holds from the driver, including cached allocator pools -- the same figure that shows up as "MPS allocated" in an MPS OOM error) divided by `torch.mps.recommended_max_memory()` (the OS's recommended Metal working-set size).
- For CUDA: the analogous `torch.cuda.memory_reserved(idx)` divided by the device's `total_memory`.

`maybe_empty_device_cache(device, threshold=DEFAULT_MEMORY_PRESSURE_THRESHOLD) -> bool`:
- Clears `device`'s cached memory if `device_memory_pressure(device)` is at or above `threshold`; returns whether it actually cleared.
- Safe to call every scheduler step: the pressure check itself is just a metadata query (no device sync), so only the clear itself -- when actually triggered -- does real work. A device where pressure can't be measured is left alone here; it still gets cache clears from the event-driven call sites (idle, retry-on-failure) elsewhere.

---

## Low-Level Data Flow and Types

### Request payload

A POST to `/api/generate` sends JSON like:
```json
{
  "prompt": "Hello world",
  "max_tokens": 16,
  "temperature": 0.7,
  "stop": ["\nuser:"],
  "idempotency_key": "optional-client-supplied-key"
}
```
`stop` and `idempotency_key` are both optional. `idempotency_key`, when set, deduplicates retried requests via `scheduler.idempotency.idempotency_store`.

### Runtime objects

- `InferenceRequest` is the primary per-call state object (see `scheduler/request.py` above for its full field list).
- `request_queue`/`batch_request_queue` are `RequestQueue`-wrapped `asyncio.Queue[InferenceRequest]`.
- `req.queue` is an `asyncio.Queue[int | str | tuple]` for token IDs, the `"[DONE]"` sentinel, and the `("[ERROR]", message)` sentinel.
- `req.input_ids` is a `torch.Tensor` on the engine device; there is no per-request `attention_mask` or `past` field -- the KV cache lives in the scheduler's shared `PagedKVCache`, addressed via `req.block_table` (a `cache.paged_kv_cache.BlockTable`).

### Tensor shapes

- `tokenizer_service.encode(..., return_tensors=True)` returns `input_ids`/`attention_mask` shape `(1, seq_len)`.
- `_pad_batch()` returns `(batch, max_new_len)` for the new-tokens region.
- `PagedKVCache.gather_dense()` returns per-layer `(batch, num_kv_heads, max_past_len, head_dim)` key/value tensors, left-padded per row to the batch's longest real past length.
- `forward_step()` expects `input_ids`/`attention_mask` shape `(batch, seq_len)` (where `seq_len` is `past_width + max_new_len` for the attention mask); after gathering, `logits` is normalized to `(batch, vocab_size)`.
- `next_tokens` is `(batch, 1)`.

### SSE stream values

The SSE pipeline yields decoded text fragments, not token IDs, buffered to natural word/punctuation boundaries (see `streaming/stream_manager.py`). If the request's queue receives the `("[ERROR]", message)` sentinel, `stream_response` yields a single SSE `event: error` frame with `message` as its data and ends the stream instead of the normal `"[DONE]"` path. If a `stop` sequence matches, the stream ends immediately after yielding whatever real content preceded the match -- the matched text itself is never yielded.

---

## Module Reference

- `main.py`: server entrypoint (`uvicorn.run`), used by `make run`/`make run-prod`.
- `cli/main.py`: `ephemeris-serve` CLI -- `serve` (alternative server entrypoint with `--model`) and `start` (boxed-UI REPL chat client with `/model` command).
- `cli/logo.py`: precomputed block-art rendering of the project logo, used by the CLI's startup splash.
- `api/server.py`: FastAPI app creation, lifespan (model/tokenizer/scheduler startup, `app.state.scheduler`), shutdown.
- `api/routes.py`: `/generate`, `/generate_batch`, `/model` (GET/POST), `/metrics` route handlers.
- `engine/model_loader.py`: model loading, device placement, warmup, and runtime `reload()`.
- `engine/generator.py`: token sampling, forward pass, repetition penalty, batched generation, `invalidate_model_cache()`.
- `tokenizer/tokenizer_service.py`: tokenizer loading, encoding, decoding, and runtime `reload()`.
- `scheduler/request_queue.py`: `RequestQueue` wrapper around `asyncio.Queue`, with `empty()`.
- `scheduler/idempotency.py`: `IdempotencyStore` for `/generate`'s `idempotency_key`.
- `scheduler/request.py`: `InferenceRequest` -- per-call state, including `stop_sequences` and `block_table`.
- `scheduler/continuous_scheduler.py`: paged-KV-cache dynamic batching scheduler, with mixed prefill/decode and stop-sequence handling.
- `scheduler/model_swap.py`: runtime model hot-swap coordinator (drain, reload, invalidate caches).
- `scheduler/batch_scheduler.py`: non-streaming batch endpoint's background processing loop.
- `cache/paged_kv_cache.py`: `PagedKVCache`/`BlockTable` -- block-based KV cache storage shared across active requests.
- `streaming/stream_manager.py`: SSE token decoding, buffering, and stop-sequence enforcement.
- `metrics/metrics.py`: `BatchMetrics` rolling tracker, `metrics`/`streaming_metrics` singletons, `summarize_batch_response_metrics()`.
- `settings/settings.py`: YAML/env config loader (`model_settings`, `logging_settings`, `scheduler_settings`, `cache_settings`, `secret_settings`), `resolve_device()`.
- `settings/config.yaml`: default configuration values.
- `schemas/schemas.py`: request/response validation models.
- `logger/logger.py`: logger setup and handlers.
- `utils/utils.py`: YAML config utilities, multiprocessing configuration.
- `utils/stop_sequences.py`: shared `find_stop_index()` used by the streaming, scheduler, and engine paths.
- `utils/errors.py`: `INTERNAL_ERROR_MESSAGE` -- the generic client-facing message for unexpected failures.
- `utils/device_cache.py`: `empty_device_cache()`/`device_memory_pressure()`/`maybe_empty_device_cache()` -- reactive and proactive CUDA/MPS cache clearing.

---

## Appendix: Key Data Structures

### `InferenceRequest`
- `prompt`, `max_tokens`, `temperature`: request parameters (the latter two fall back to `model_settings` if not given).
- `stop_sequences`: list of strings; generation halts before emitting a match.
- `future`: completion future.
- `queue`: SSE token queue (`int | str | tuple` -- token ids, `"[DONE]"`, or `("[ERROR]", message)`).
- `enqueue_time`, `deadline`, `queue_latency_ms`: timeout/metrics bookkeeping.
- `input_ids`: tensor state, grows by one column per generated token.
- `generated_tokens`: output token history.
- `finished`: bool.
- `block_table`: this request's slice of the scheduler's shared `PagedKVCache` (see `cache/paged_kv_cache.py`).

### `GenerationRequest` (Protocol)
- Structural interface (`typing.Protocol`, defined in `engine/generator.py`) that `InferenceEngine.generate_batch()` depends on instead of a scheduler-owned wrapper type.
- Fields: `future`, `queue`, `temperature`, `max_tokens`, `generated_tokens`, `finished`, `stop_sequences`.
- `InferenceRequest` satisfies this protocol structurally; no wrapper object is created. `generate_batch()` tracks `(original_index, request)` tuples internally to preserve output ordering.

### `PagedKVCache` / `BlockTable` (`cache/paged_kv_cache.py`)
- `PagedKVCache`: pre-allocated, fixed-size-block key/value tensor pools (`key_pool`/`value_pool`, one entry per layer), grown by doubling when exhausted. `allocate()`/`append()`/`gather_dense()`/`free()`/`is_valid()` are the operations the scheduler uses each step.
- `BlockTable`: a request's own `block_ids: list[int]` (which blocks in the pool belong to it) and `length: int` (real, unpadded token count stored so far). Freed and reset to empty by `PagedKVCache.free()`.
- Pure PyTorch indexing, no fused kernel -- works identically on MPS/CPU/CUDA. `gather_dense()` still materializes a dense `(batch, heads, seq, head_dim)` tensor every step for the model's forward pass.

---

## How to Read the Code per Module

Same order as [Module Reference](#module-reference) above -- start at `main.py`/`cli/main.py` for entrypoints, then `api/` for HTTP surface, `scheduler/`+`cache/`+`engine/` for the generation core, `streaming/` for how tokens leave the process, and `settings/`+`schemas/`+`logger/`+`utils/` for cross-cutting concerns.
