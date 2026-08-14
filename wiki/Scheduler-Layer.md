# Scheduler Layer

## Scheduler Layer

The scheduler is the core of the system and manages request multiplexing, batching, and incremental generation.

### `scheduler/request_queue.py`

A minimal wrapper around `asyncio.Queue`.

Class `RequestQueue`:
- `self.queue = asyncio.Queue()`.
- `async def put(self, request)`.
- `async def get(self)`.
- `def empty(self) -> bool`: returns `self.queue.empty()` -- used by `scheduler/model_swap.py` to confirm no request is sitting in the queue, unpicked-up, before a swap proceeds.

Global singletons:
- `request_queue = RequestQueue()`, `batch_request_queue = RequestQueue()`.

These queues are the handoff point between the HTTP endpoints and the scheduler loops.

### `scheduler/idempotency.py`

An in-process, TTL-bounded store mapping a client-supplied `idempotency_key` to the `InferenceRequest` it was assigned. Used by `POST /api/generate` to deduplicate retried requests.

Class `IdempotencyStore`:
- `get(key)`: lazily purges expired entries, returns the live `InferenceRequest` for `key` if present, else `None`.
- `put(key, request)`: stores `(time.monotonic() + ttl_seconds, request)`.
- `discard(key)`: removes an entry.
- TTL defaults to `scheduler_settings.idempotency_key_ttl_seconds`.

Purging is lazy (on `get`/`put`), not a background task -- there is no persistence layer elsewhere in this repo, so the store is scoped to a single process's lifetime.

Global singleton:
- `idempotency_store = IdempotencyStore()`.

### `scheduler/request.py`

Defines the in-memory request state used by the scheduler and streaming layers.

`InferenceRequest.__init__(prompt, max_tokens, temperature, stop_sequences=None)`:
- `prompt: str`
- `max_tokens: int` -- falls back to `model_settings.max_length` if `None`
- `temperature: float` -- falls back to `model_settings.temperature` if `None`
- `stop_sequences: list[str]` -- `list(stop_sequences)` if given, else `[]`
- `future: asyncio.Future[str]` -- created from the currently running event loop (or a newly created one if none is running), used for external completion notification
- `queue: asyncio.Queue[int | str | tuple]` -- streams token IDs, the final `"[DONE]"` sentinel, and the `("[ERROR]", message)` sentinel
- `enqueue_time: float`, `deadline: float | None` -- timeout enforcement on both the streaming and batch paths
- `queue_latency_ms: float | None` -- set once the request is admitted into a scheduler
- `input_ids: Optional[torch.Tensor]` -- set when the request enters the scheduler; grows by one column per generated token
- `generated_tokens: list[int]` -- tokens produced so far
- `finished: bool`
- `block_table: BlockTable` (from `cache.paged_kv_cache`) -- this request's slice of the shared `PagedKVCache`; starts empty, which is the scheduler's "no cached prefill yet" signal

Note: there is no `attention_mask` or `past`/`DynamicCache` field on `InferenceRequest` -- the attention mask is rebuilt each step from `block_table`/`_prepare_batch`, and the KV cache lives in the scheduler's shared `PagedKVCache`, addressed per-request via `block_table`, not stored per-request as a `DynamicCache`.

There is no `ActiveRequest` wrapper type. `InferenceEngine.generate_batch()` operates directly on `InferenceRequest` objects (paired with their original index) via a structural `GenerationRequest` protocol -- see `engine/generator.py` and the Appendix.

### `scheduler/continuous_scheduler.py`

This module is responsible for continuous request consumption, dynamic batching (including mixing prefill and decode rows in the same step), and incremental token generation, backed by a block-based ("paged") KV cache.

Imports:
- `asyncio`, `time`, `torch`, `dataclass`, `List`, `Optional`
- `DynamicCache` from `transformers`
- `model_settings`, `logging_settings`, `cache_settings` from `settings.settings`
- `tokenizer_service` from `tokenizer.tokenizer_service`
- `request_queue` from `scheduler.request_queue`
- `InferenceRequest` from `scheduler.request`
- `find_stop_index` from `utils.stop_sequences`
- `empty_device_cache`, `maybe_empty_device_cache` from `utils.device_cache` (see [Utility Helpers](Reference#utility-helpers))
- `INTERNAL_ERROR_MESSAGE` from `utils.errors`
- `streaming_metrics` from `metrics.metrics` (see [Metrics](Streaming-and-Metrics#metrics))
- `PagedKVCache` from `cache.paged_kv_cache` (see [Cache Layer](Cache-and-Engine#cache-layer))
- `setup_logger` from `logger`

`_BatchInputs` dataclass -- one step's batched forward-pass inputs, built by `_prepare_batch()`:
- `input_ids`, `attention_mask`, `position_ids`, `logit_gather_indices`: `torch.Tensor`
- `past_key_values: Optional[DynamicCache]`
- `past_width: int` -- the shared past-region width fed into this step (same for every row)
- `new_lengths: List[int]` -- per-row count of real new tokens contributed this step (a row's whole prompt on its first step, or `1` while mid-decode)

`ContinuousScheduler.__init__(engine, tokenizer, max_batch_size=8, timeout=0.01)`:
- `engine`: `InferenceEngine` instance.
- `tokenizer`: `TokenizerService` instance.
- `max_batch_size`: default 8.
- `timeout`: default 0.01 seconds (queue-poll interval used by `_add_new_requests`; distinct from a request's `deadline`).
- `active_requests: list[InferenceRequest]` starts empty.
- `_paged_cache: Optional[PagedKVCache]` starts `None` -- constructed lazily.

#### `paged_cache` property

- Block-based KV cache storage shared by every active request. Built on first access (not in `__init__`, so it doesn't force the model to load before it's otherwise needed).
- Derives its shape from `self.engine.model.config`: `num_kv_heads` (`num_key_value_heads`, falling back to `num_attention_heads`), `head_dim` (`head_dim`, falling back to `hidden_size // num_attention_heads`), `num_layers` (`num_hidden_layers`), `block_size` from `cache_settings.kv_block_size`, and the engine's `dtype`/`device`.

#### `invalidate_paged_cache()`

- Sets `self._paged_cache = None`, forcing a rebuild (against whatever model is now loaded) on next access. Used after a runtime model swap (`scheduler/model_swap.py`) -- the old cache's tensor shapes are tied to the previous model's architecture. Must only be called once the caller has confirmed `active_requests` is empty, since an active request's `block_table` still points into the old cache.

#### `_pad_batch(tensors, padding_value)`

- Right-pads each 2D tensor to the batch's max width along dim 1 and concatenates along dim 0.
- Used for the "new tokens this step" region: real content starts at column 0 for every row, so once concatenated after that row's real past it stays one contiguous real range.

#### `_add_new_requests()`

- While `len(self.active_requests) < self.max_batch_size`: calls `await asyncio.wait_for(request_queue.get(), timeout=self.timeout)`; on timeout, returns.
- For each dequeued `InferenceRequest`:
  - Applies a chat prompt template if the tokenizer exposes `apply_chat_template()` (formats the prompt as a single `user` message, `add_generation_prompt=True`), otherwise uses the raw prompt. A template failure is caught and logged, falling back to the raw prompt.
  - If `req.deadline` has already passed (e.g. it sat behind a long queue backlog), calls `streaming_metrics.record_timeout_eviction()` and fails the request immediately via `_fail_request_timeout()` without ever scheduling it -- no forward-pass compute is spent on an already-dead request.
  - Encodes the resulting text via `self.tokenizer.tokenizer(..., return_tensors='pt')` and moves `input_ids` to `self.engine.device`.
  - `req.block_table` starts empty (from `InferenceRequest.__init__`) -- that's the "no KV cache yet" signal for the first step.
  - Records `req.queue_latency_ms = time.monotonic() - req.enqueue_time` and calls `streaming_metrics.record_queue_latency(...)` exactly once per admitted request.
  - Appends the request to `self.active_requests`.

#### `_prepare_batch() -> Optional[_BatchInputs]`

Builds one batched forward-pass input, **mixing prefill and decode rows in the same step**: a request with no cached past yet (`block_table.length == 0`) contributes its whole prompt as new input; a request already mid-decode contributes exactly its one most-recently-generated token (`input_ids[:, -1:]`). Both kinds of rows are batched together, so one request joining never forces every other active request to redo a full-sequence recompute. Returns `None` if there are no active requests.

Steps:
1. For each active request, if `block_table.length > 0` but `paged_cache.is_valid(block_table)` is `False`, logs a warning and frees the table -- the next step will fall back to a fresh prefill for that request.
2. Builds `new_token_tensors` (the per-row "new tokens this step", as above) and `new_lengths = [t.shape[1] for t in new_token_tensors]`; `max_new_len = max(new_lengths)`.
3. Calls `self.paged_cache.gather_dense([req.block_table for req in active_requests])` to get `(keys_per_layer, values_per_layer, past_lengths)` -- a left-padded, batched dense view of each row's real cached history. `past_width = max(past_lengths)`.
4. `input_ids`: the new-tokens region, right-padded to `max_new_len` via `_pad_batch` (padding value = the tokenizer's `pad_token_id`).
5. `attention_mask`: per row, `[past region, left-padded to past_width]` concatenated with `[new-tokens region, right-padded to max_new_len]` -- built directly from `past_len`/`new_len`, not assumed.
6. `position_ids`: derived from the attention mask itself (`cumsum(attention_mask, dim=1)`, sliced to the new-tokens region, `-1`, clamped to `>= 0`) -- correct regardless of which side padding sits on.
7. `logit_gather_indices`: per row, `new_len - 1` -- the column (within the logits tensor, which spans only the new-tokens region) holding the real last-new-token prediction, since the new-tokens region is right-padded and a shorter row's real content doesn't sit at the tensor's absolute end.
8. `past_key_values`: `None` if `past_width == 0`, else a `DynamicCache` built from the batched per-layer `(keys, values)` gathered in step 3.

#### `_dispatch_tokens(reqs, next_tokens, new_past, past_width, new_lengths)`

Streams sampled tokens back to clients and updates per-request state. `reqs` is index-aligned with `next_tokens`/`new_past`/`new_lengths` -- normally `self.active_requests`, but may be a single-request subset during an isolation retry (see `_retry_requests_individually()`). `next_tokens` has shape `(batch, 1)`.

Per request (index `idx`):
1. If `req.queue` is set, `put_nowait(token_id)` immediately (before the stop-sequence check below -- see `streaming/stream_manager.py` for how the client side still avoids ever seeing the stop text).
2. Appends `token_id` to `req.generated_tokens`.
3. **Stop-sequence check**: if `req.stop_sequences` is non-empty, decodes the *full* `req.generated_tokens` (a stop sequence can span multiple tokens and needn't align with token boundaries) and calls `find_stop_index(decoded, req.stop_sequences)`. If a match is found, `stop_text = decoded[:stop_idx]` (the text to actually return, with the stop sequence and anything after it trimmed).
4. Appends the new token to `req.input_ids`.
5. Extracts this step's newly computed K/V for row `idx` via `_extract_new_kv()` and either frees the request's `block_table` (if extraction failed) or appends the new K/V into the shared `paged_cache`.
6. Finishes the request (`_finish_request(req, final_text=stop_text)`) if `stop_text is not None`, or `token_id == self.engine.eos_token_id`, or `len(req.generated_tokens) >= req.max_tokens`.

Finished requests are removed from `self.active_requests` at the end.

#### `_extract_new_kv(new_past, idx, start_col, length, prompt)`

Pulls request `idx`'s newly computed `(key, value)` tensors out of `new_past`, using the exact real-content range `[start_col : start_col + length]` for that row (not just "the last N columns" -- with a mixed batch, a shorter-than-max row's real new tokens sit before some trailing padding). Returns `None` (logging a warning) if `new_past` is missing or structurally invalid, in which case the caller drops the request's cached state and lets it recompute from scratch on the next step.

#### `_finish_request(req, final_text=None)`

Shared finalization helper (used by `_dispatch_tokens`'s natural-completion branch and by `_evict_expired_requests()`):
- Sets `req.finished = True`.
- If `req.future` is not done, resolves it with `final_text` if given, else `tokenizer_service.decode(req.generated_tokens)`. `final_text` is what a stop-sequence match passes in, so the matched text (and anything after it) never reaches the caller.
- Enqueues `"[DONE]"` into `req.queue` if present.
- Frees the request's paged-cache blocks via `_free_block_table()`.

#### `_fail_request_timeout(req)`

Fails a request that timed out before generating any tokens: sets `asyncio.TimeoutError` on `req.future` if not already done, enqueues `("[ERROR]", "generation timed out")`, and frees its block table.

#### `_fail_single_request(req, exc)`

Fails exactly one request. Sets `exc` -- which may carry internal detail (stack-trace-flavored text, memory sizes, ...) -- as the exception on `req.future`, for internal bookkeeping only; pushes an `("[ERROR]", INTERNAL_ERROR_MESSAGE)` sentinel to `req.queue` instead of `str(exc)`, so the client-facing SSE message is always the generic, safe constant from `utils.errors`. Frees the request's block table. Called by `_retry_requests_individually()` for a request that still fails in its own batch of one; the caller is responsible for removing `req` from `self.active_requests`.

#### `_retry_requests_individually()`

Called when a generation step fails twice in a row (see `_step()`). A batched forward pass fails or succeeds as a unit, so at that point there's no way to tell from the exception alone which row is actually responsible. Retries every currently-active request in its own batch of one, via `_build_batch_inputs([req])` and `_forward_and_sample(single_batch, [req])`: a request that succeeds alone is dispatched normally via `_dispatch_tokens([req], ...)`, same as any other step; a request that fails even alone is failed via `_fail_single_request()`, removed from `self.active_requests`, and the device cache is cleared before moving on to the next request. This means a single poisoned request (bad cached state, a degenerate shape, ...) no longer takes every co-batched request down with it -- only requests that are still active when isolation begins are affected, since it always re-checks `self.active_requests` membership before retrying a given request.

#### `_evict_cancelled_requests()`

Filters `self.active_requests` to drop any request whose `future.cancelled()` is `True` (set by `api/routes.py`'s disconnect-polling task), freeing its block table and calling `streaming_metrics.record_cancelled_eviction()`. No finalization is attempted -- the future is already terminal, and there's no client left to read `req.queue`.

#### `_free_block_table(req)`

Returns a request's paged KV blocks to the pool, if it has any (`block_table.block_ids` non-empty). Guarded so this never forces the (lazily-constructed) paged cache -- and therefore the model -- to load just to free a table that was never populated.

#### `_evict_expired_requests()`

Evicts any request whose `deadline` has passed: calls `streaming_metrics.record_timeout_eviction()`, then if it already generated at least one token, finishes it with that partial output via `_finish_request()`; otherwise fails it via `_fail_request_timeout()`. The same counter is also incremented in `_add_new_requests()` when a request's deadline has already passed *before* it's ever scheduled (see below) -- both are the same underlying event (a request never got its full generation window), just caught at different points.

#### `_forward_and_sample(batch_inputs, reqs)`

Runs `engine.forward_step(input_ids, attention_mask, past_key_values, position_ids=..., logit_gather_indices=...)`, `engine.apply_repetition_penalty()`, and per-request `engine.sample()` (using each request's own `temperature`, and the global `model_settings.top_k`/`top_p`), returning `(next_tokens, new_past)`. `reqs` must be index-aligned with `batch_inputs`'s rows -- normally `self.active_requests`, but a single-request subset during an isolation retry. Extracted so `_step()` can retry the same call once on failure without duplicating it, and so `_retry_requests_individually()` can reuse it for a batch of one.

#### `_step()`

Coordinates a single scheduler iteration:
1. `_evict_cancelled_requests()` and `_evict_expired_requests()` run first, before any tensors are built.
2. If `self.active_requests` is empty after eviction, return.
3. `batch_inputs = self._prepare_batch()`; if `None`, return.
4. `self._forward_and_sample(batch_inputs, self.active_requests)`. On failure, log a warning, call `empty_device_cache(self.engine.device)`, then retry once -- the retry is only likely to help for an OOM-shaped failure, and retrying against the exact same (still-exhausted) device memory state that just failed would almost certainly fail again. If the retry also fails, log the exception in full and call `self._retry_requests_individually()`, which retries and dispatches (or fails) each request on its own; `_step()` itself skips its own dispatch call in that case (`next_tokens` stays `None`).
5. If `next_tokens is not None` (i.e. the whole-batch attempt succeeded): `await self._dispatch_tokens(self.active_requests, next_tokens, new_past, batch_inputs.past_width, batch_inputs.new_lengths)`.
6. Device memory is released proactively rather than only on failure/idle: if `self.active_requests` is now empty, calls `empty_device_cache(self.engine.device)` unconditionally; otherwise calls `maybe_empty_device_cache(self.engine.device)`, which only clears once usage crosses 70% of the device's memory budget (a cheap metadata query every step via `utils.device_cache.device_memory_pressure()`, not a device sync). This closes the gap where a single long-running request that never hits a failure or an idle gap could otherwise accumulate cached-but-unused memory all the way to the device's actual ceiling (observed as `MPS backend out of memory`).
7. Records `streaming_metrics.record_batch_size(active_count)` and `streaming_metrics.record_token_throughput(active_count, elapsed)` -- one token per active request per step.
8. Records `streaming_metrics.record_batch_occupancy(active_count, self.max_batch_size)`. Also records `streaming_metrics.record_cache_utilization(...)` -- but only if `self._paged_cache` has actually been constructed already (read directly, not via the `paged_cache` property), so this metric never itself forces the paged cache -- and therefore the model -- to build.

#### `run()`

Main scheduler loop: forever, `await self._add_new_requests()`; if `active_requests` is empty, sleep `0.01`s and continue; otherwise `await self._step()`. On any exception, logs the stack trace and sleeps 1 second before retrying. Started once during FastAPI startup (`api/server.py`) and canceled cleanly on shutdown.

### `scheduler/model_swap.py`

Hot-swaps the model an already-running server serves, without a process restart. Used by `POST /api/model` (`api/routes.py`) and the CLI's `/model` REPL command (`cli/main.py`).

Imports:
- `asyncio`, `gc`, `time`
- `engine` from `engine.generator`
- `model_loader` from `engine.model_loader`
- `tokenizer_service` from `tokenizer.tokenizer_service`
- `request_queue`, `batch_request_queue` from `scheduler.request_queue`
- `model_settings`, `logging_settings` from `settings.settings`
- `empty_device_cache` from `utils.device_cache`
- `setup_logger` from `logger`

Module-level `swap_lock = asyncio.Lock()`: held for the duration of a swap. Route handlers that accept new requests check `swap_lock.locked()` (a fast, non-blocking read -- not full acquisition, since a swap can take a while) and reject with `503` rather than letting requests pile up against a model that's about to disappear.

`swap_model(new_model_name, continuous_scheduler, drain_timeout) -> str`:
1. Acquires `swap_lock`.
2. **Drain**: loops (`asyncio.sleep(0.05)` between checks) while `continuous_scheduler.active_requests` is non-empty, or `request_queue`/`batch_request_queue` isn't empty. Raises `TimeoutError` if this doesn't resolve within `drain_timeout` seconds.
3. **Swap**: reloads the tokenizer first (`tokenizer_service.reload(new_model_name)`), then the model (`model_loader.reload(new_model_name)`). If either raises, the tokenizer is explicitly rolled back to the *previous* model name (`tokenizer_service.reload(previous_model_name)`) before re-raising -- this keeps the tokenizer paired with whichever model actually ended up loaded, since `model_loader.reload` itself already leaves the old model in place on failure.
4. **Invalidate caches**: on success, calls `engine.invalidate_model_cache()` and `continuous_scheduler.invalidate_paged_cache()` so both rebuild against the new model's architecture on next use.
5. **Reclaim old paged-cache memory**: calls `gc.collect()` then `empty_device_cache(model_settings.device)`. `model_loader.reload()` already ran its own `gc.collect()`/cache-empty for the old *model*, but at that point the old *paged cache*'s block pool (dropped in step 4, just above) was still referenced by `continuous_scheduler` -- so that memory wasn't actually reclaimed until this second, later clear.
6. Returns `new_model_name`.

Because the drain step blocks on real in-flight work and the reload step performs a real (potentially slow, network-bound) model/tokenizer load, this is a genuinely slow operation by design -- callers (the API route, the CLI) are expected to use a generous timeout.

### `scheduler/batch_scheduler.py`

This module implements the non-streaming batch endpoint's background processing loop and metrics collection.

Imports:
- `asyncio`, `time`, `List`
- `InferenceRequest` from `scheduler.request`
- `batch_request_queue` from `scheduler.request_queue`
- `logging_settings`, `model_settings` from `settings.settings`
- `tokenizer_service` from `tokenizer.tokenizer_service`
- `empty_device_cache` from `utils.device_cache`
- `metrics` from `metrics.metrics` (see [Metrics](Streaming-and-Metrics#metrics))

`BatchScheduler.__init__(engine, tokenizer, max_batch_size=8, queue_timeout=0.02, request_timeout=20.0)`.

`_collect_batch()`:
- Blocks on the first item via `await batch_request_queue.get()`, records its queue latency, then opportunistically drains up to `max_batch_size - 1` more items within the remaining `queue_timeout` window.

`process_batch(batch)`:
- Filters out requests that are already done/cancelled, or whose `deadline` has already passed (failed with `asyncio.TimeoutError` if so).
- Applies the chat template to each remaining prompt (same `apply_chat_template()` pattern as the continuous scheduler), falling back to raw prompts on failure.
- Tokenizes the batch with padding/truncation to `model_settings.max_length`.
- Calls `await self.engine.generate_batch(input_ids, attention_mask, valid_requests)`.
- Resolves each request's `future` with its output string (or an exception, if `generate_batch` itself raised).
- Records batch size and token throughput after `generate_batch()` returns.

Timeout configuration:
- `self.request_timeout` (constructor param, default from `scheduler_settings.batch_request_timeout_seconds`) is passed in explicitly by `api/server.py` at startup.
- `api/routes.py`'s `POST /generate_batch` sets each request's `deadline = enqueue_time + scheduler_settings.batch_request_timeout_seconds` -- the same settings value, so the constructor parameter and the enforced deadline don't drift independently.

`run()`: forever, collects and processes batches; sleeps briefly when the queue is empty; calls `empty_device_cache(self.engine.device)` once each batch is fully processed (PyTorch's allocator otherwise holds cached-but-unused memory rather than returning it to the system, which can accumulate across many batches on a long-running process); re-raises `asyncio.CancelledError` (logging first) rather than swallowing it, so the task actually stops on shutdown; on any other unexpected exception, fails every request in the in-flight batch and sleeps before retrying.
