# Operations and Roadmap

## Deployment and Runtime Notes

- The app can be started via `python main.py`, `make run`/`make run-prod`, or `ephemeris-serve serve` (the last also supports `--model` to pick a model without editing `config.yaml`).
- In production, remove `reload=True`/`--reload` and consider increasing `workers` -- note that `--workers > 1` only works correctly with `--model` because the model name is propagated via the `EPHEMERIS_SERVER_MODEL_NAME` env var, not an in-process mutation (uvicorn's worker processes are fresh Python processes that re-import `settings.settings`).
- The FastAPI app is created by `create_app()` in `api/server.py`; the live `ContinuousScheduler` instance is reachable at runtime via `app.state.scheduler`.
- The scheduler lifecycle is tied to FastAPI startup and shutdown.
- Hugging Face auth is optional; if `HF_KEY` is missing, the server uses anonymous access.
- A running server's model can be changed without a restart via `POST /api/model` (or the CLI's `/model <name>` REPL command) -- expect a pause of at least a few seconds (drain + reload) while it happens, and `503`s from `/generate`/`/generate_batch` for that window.

---

## Authentication

The `/api` routes are gated by an API key (`Authorization: Bearer <key>`) in two tiers -- `EPHEMERIS_SERVER_API_KEYS` for generation and read-only routes, `EPHEMERIS_SERVER_ADMIN_API_KEYS` additionally for `POST /api/model`. See [`api/auth.py`](API-Layer#apiauthpy) for the mechanism.

**With no keys configured, every route is open.** That is the local-development default, and the startup log warns about it. A public deployment with an unauthenticated `POST /api/model` lets any caller make the server download and load an arbitrary Hugging Face repo, so this is the first thing to set.

Keys are supplied through the systemd unit's `EnvironmentFile` (`/etc/ephemeris-serve/env`, root-owned `0600`). Rotation is add-then-remove: both keys are accepted while clients redeploy. There is no revocation without a restart and no expiry -- the keys are read once into `SecretSetting`.

Authentication deliberately lives in the app rather than in nginx: one mechanism covers requests whether or not they arrived through the proxy, and clients need one credential instead of two.

---

## Process Supervision

`deploy/systemd/ephemeris-serve.service` runs the server under systemd. Points that are load-bearing rather than boilerplate:

- `WorkingDirectory=/opt/ephemeris-serve` -- `settings/settings.py` reads `settings/config.yaml` and the logger writes `logs/app.log`, both *relative* paths. Started elsewhere, the server can't find its config.
- `TimeoutStartSec=900` -- the model is loaded (and on first boot, downloaded) during FastAPI's lifespan startup, before the socket accepts traffic; systemd must not read that as a hang.
- `TimeoutStopSec=120` -- draining in-flight generation and releasing device memory outlasts the 90s default.
- `StartLimitBurst=5`/`StartLimitIntervalSec=600` -- a bad config surfaces as a failed unit instead of an endless restart loop.
- `--workers 1` -- more workers would each hold their own model, and `POST /api/model` swaps only the process that receives it (see [Known Behavior](#known-behavior)).
- `MemoryDenyWriteExecute` is **not** set: PyTorch JIT-compiles at runtime and needs W+X pages. On NVIDIA hosts `PrivateDevices` must also stay off so `/dev/nvidia*` is reachable.

---

## Reverse Proxy

In production uvicorn binds loopback only and nginx is the single public listener. The config is `deploy/nginx/ephemeris-serve.conf`; `deploy/nginx/README.md` covers install, TLS, and verification. The proxy listens on `:8080` and forwards to `127.0.0.1:8000`.

nginx is there to terminate TLS (uvicorn serves plain HTTP), give clients one stable address independent of uvicorn's port, reject oversized bodies and rate-limit per client address before a request ever reaches Python, and absorb slow readers on the buffered JSON endpoints.

### SSE and buffering

The critical part is the `location = /api/generate` block:

```nginx
proxy_buffering off;
proxy_cache off;
gzip off;
add_header X-Accel-Buffering no;
proxy_read_timeout 300s;
```

Measured on nginx 1.31.3, this config delivers SSE frames with the same timing as a direct connection to uvicorn (10 frames, 2.71s spread, 0.30s median gap, identical both ways). Re-running with `proxy_buffering on` and no upstream hint header gave *the same* timing -- the widely-cited "nginx breaks SSE" failure did not reproduce on this version, because nginx forwards each chunked read rather than waiting to fill a buffer.

So these directives are insurance, not a fix for an observed break. They matter if gzip is widened to cover `text/event-stream`, a `proxy_cache` is added, or an nginx build coalesces small chunks. Independently, `sse_starlette` sets `X-Accel-Buffering: no` on every `EventSourceResponse` and nginx honors that from upstream, so buffering is already off for this endpoint regardless. The `add_header X-Accel-Buffering no` in the config is for whatever sits *in front of* nginx (a CDN, another proxy) and honors it.

`proxy_read_timeout` must comfortably exceed `streaming_request_timeout_seconds` from `settings/config.yaml` (60s by default), because an SSE connection is idle between tokens and nginx cannot distinguish "thinking" from "hung". The `/api/` location instead uses 600s, since `POST /api/model` drains in-flight work and then loads -- often downloads -- an entire model.

`proxy_ignore_client_abort off` (the default, set explicitly) matters for the same endpoint: the scheduler cancels a request's future when the client disconnects (see [Scheduler Layer](Scheduler-Layer#scheduler-layer)), so a disconnect must propagate upstream rather than leaving tokens being generated for nobody.

### Forwarded headers

`ephemeris-serve serve` passes `proxy_headers=True` with `forwarded_allow_ips="127.0.0.1"` to `uvicorn.run()` by default (as does `main.py` and `make run-prod`), so the app reads the real client address and scheme from `X-Forwarded-For`/`X-Forwarded-Proto` rather than seeing nginx's. Those headers are client-supplied and therefore spoofable: trusting them is only safe while nothing but the local proxy can reach uvicorn's port, which is why the allow-list is loopback-only and uvicorn should stay bound to `127.0.0.1` in production. `--no-proxy-headers` turns the trust off for a directly-exposed server.

### Limits

- `limit_req_zone` is keyed on `$binary_remote_addr` at 10r/s with `burst=20` -- a blunt guard against one client saturating the scheduler queue, not a quota system.
- `upstream ... keepalive 32` sizes reusable upstream connections; each open SSE stream holds one, so size it against expected concurrent streams rather than request rate.
- `POST /api/model` is still single-process (see [Known Behavior](#known-behavior)): behind this proxy with `--workers > 1`, nginx routes the swap to one worker and the rest keep serving the old model.

---

## Completed Improvements & Current Observations

Key improvements that have been implemented:
- **Request Cancellation**: Supported client-disconnect detection and request cancellation for batch generation, and enforced on the streaming path too -- `ContinuousScheduler` evicts cancelled requests every step instead of continuing to generate for a disconnected SSE client.
- **Request Timeouts**: Configurable deadlines for both the batch path (`batch_request_timeout_seconds`, `batch_generation_timeout_seconds`) and the streaming path (`streaming_request_timeout_seconds`), sourced from `scheduler_settings`. An expired streaming request is finished with partial output if any tokens were generated, otherwise failed with an SSE `error` event.
- **Runtime Metrics**: Queue latency, batch size, and token throughput tracking, exposed via `/api/metrics` as `{"batch": ..., "streaming": ...}` (separate singletons so `/generate_batch` responses aren't polluted by concurrent streaming traffic). The streaming singleton also tracks scheduler-specific gauges/counters not applicable to the batch path: active-batch occupancy, paged-KV-cache block utilization, and timeout/cancellation eviction counts.
- **Engine/Scheduler Decoupling**: `InferenceEngine.generate_batch()` depends on a local `GenerationRequest` `typing.Protocol` rather than importing a scheduler-owned wrapper type; `InferenceRequest` satisfies it structurally.
- **Idempotency & Retry (Streaming)**: `/api/generate` accepts an optional `idempotency_key` (dedup via `scheduler/idempotency.py`); `ContinuousScheduler._step()` retries a failed generation step once before falling back to per-request isolation (see below) rather than failing every active request outright.
- **Per-Request Retry Isolation**: a whole-batch forward pass that still fails after the single retry no longer fails every co-batched request together. `ContinuousScheduler._retry_requests_individually()` retries each currently-active request in its own batch of one (`_build_batch_inputs([req])` / `_forward_and_sample(single_batch, [req])`): a request that succeeds alone is dispatched normally, and only a request that still fails in isolation is failed via the new `_fail_single_request()` and removed from the pool. `_fail_active_batch()` (which unconditionally failed every active request) has been removed, since `_fail_single_request()` now covers the same job per-request.
- **Paged KV Cache with Mixed Prefill/Decode Batching**: the scheduler no longer processes a step as strictly "all prompts" or "all cached decode tokens" -- `cache/paged_kv_cache.PagedKVCache` (block-based storage, addressed per-request via `BlockTable`) lets a brand-new request's prefill share the same batched forward pass as requests already mid-decode, without forcing either kind of row to redo work it doesn't need to.
- **Apple Silicon (MPS) / CUDA Support**: `resolve_device()`'s `"auto"` mode now checks CUDA before MPS, then falls back to CPU. Models load with `float16` on MPS.
- **Stop Sequences**: `GenerateRequest.stop` (up to 4 strings) halts generation before emitting a match, checked identically on the streaming and batch paths (`utils.stop_sequences.find_stop_index`) and independently re-checked in `streaming/stream_manager.py` so the matched text can never leak into a client's SSE stream even if it was already pushed onto the token queue.
- **Runtime Model Hot-Swap**: `GET`/`POST /api/model` (backed by `scheduler/model_swap.py`) replace the loaded model/tokenizer without a process restart -- draining in-flight requests first, rolling the tokenizer back if the model half of the swap fails, and invalidating the engine's cached model reference and the scheduler's paged KV cache so both rebuild against the new model's architecture. `/generate`/`/generate_batch` reject with `503` for the duration.
- **CLI (`ephemeris-serve`)**: `serve` starts the server itself with `--model` selection (via a new `EPHEMERIS_SERVER_MODEL_NAME` env var, so it survives uvicorn's multi-worker respawning); `start` is a boxed-UI REPL chat client with a live-streaming bordered response box, a startup splash rendering the project's logo as block-art, a `/model` command for runtime model switching, a `/creativity` command for adjusting sampling temperature mid-session, and arrow-key line editing with cross-session persistent history (`readline`, `~/.ephemeris_history`).
- **Client-Safe Error Messages**: `utils/errors.INTERNAL_ERROR_MESSAGE` replaces raw exception text in every client-facing failure path (`/generate_batch`'s and `POST /api/model`'s `500` responses, and the streaming path's SSE `error` event via `ContinuousScheduler._fail_single_request`) -- the real exception is still logged in full server-side, just never put on the wire.
- **SSE Disconnect-Cancellation Race Fix**: `POST /generate` now cancels its `InferenceRequest.future` via `EventSourceResponse`'s own `client_close_handler_callable` instead of a second `request.is_disconnected()` poller racing (and always losing to) `sse_starlette`'s own disconnect listener for the one-shot ASGI `http.disconnect` message -- previously this meant a disconnected SSE client's request was never actually cancelled.
- **Proactive Device Memory Management**: `utils/device_cache.py` adds `device_memory_pressure()`/`maybe_empty_device_cache()`, checked every `ContinuousScheduler._step()` (a cheap metadata query, not a device sync) to clear cached-but-unused CUDA/MPS memory once usage crosses 70% of budget -- closing the gap where a single long-running request that never hit a failure or an idle gap could accumulate cached memory all the way to the device's ceiling (`MPS backend out of memory`). The retry-on-failure path also now clears the cache *before* retrying, rather than retrying against the same exhausted memory state.
- **Testability Refactors**: Extracted `/generate_batch`'s response-metrics assembly into `metrics.summarize_batch_response_metrics()`, and `/generate_batch`'s disconnect-cancellation polling into a shared `api.routes.cancel_futures_on_disconnect()`.
- **Python 3.12 Floor**: Bumped the minimum/CI-tested Python version to 3.12.

Future areas for improvement:
- **Per-request `top_k`/`top_p`**: both `ContinuousScheduler._forward_and_sample()` and `InferenceEngine.generate_batch()` currently pull `top_k`/`top_p` from the global `model_settings` rather than from the individual request/schema objects, even though `engine.sample()` itself already accepts per-call `top_k`/`top_p` values. `temperature` and `stop` are already per-request; extending `GenerateRequest`/`InferenceRequest` with optional `top_k`/`top_p` fields and threading them through would be a small, self-contained change.

---

## Extension Points

Implemented enhancements:
- Non-streaming batch endpoint support via `/api/generate_batch`.
- Explicit request cancellation and timeout handling for both batch and streaming inference.
- Metrics collection for queue latency, batch size, and token throughput, tracked separately for the batch and streaming paths; the streaming path additionally tracks active-batch occupancy, paged-KV-cache utilization, and timeout/cancellation eviction counts.
- Support for both SSE streaming and synchronous batch generation paths.
- Client idempotency keys for deduplicating retried streaming requests.
- Mixed prefill/decode batching via a paged KV cache, instead of a strictly "all-prefill or all-decode" step.
- Per-request `stop` sequences, enforced identically on both generation paths.
- Runtime model hot-swapping without a process restart.
- A CLI that can both launch the server (with model selection) and act as a full chat client (with live runtime model switching, adjustable creativity, and shell-like command history).
- Client-safe generic error messages for every unexpected-failure path, with full detail always preserved server-side in logs.
- Proactive (not just reactive) CUDA/MPS memory-pressure management during the streaming scheduler's hot loop.

Not yet implemented -- natural next steps:
- **Per-request `top_k`/`top_p`**: see the note under Completed Improvements above.
- **Concurrent model swaps across multiple server processes**: `POST /api/model` only affects the single process that receives the request -- a `--workers > 1` deployment would need each worker swapped independently (there's no cross-process coordination).

---

## Known Behavior

- The scheduler expects token IDs and attention masks to be padded consistently within a batched step; `_prepare_batch()` derives `position_ids` from the constructed attention mask rather than assuming a fixed padding side, so mixed past-lengths/padding within one step are handled correctly.
- Token generation is emitted as raw decoded text fragments, not full sentences; SSE consumers must reassemble fragments if they want complete text messages.
- **Isolation retry cost**: after two whole-batch forward-pass failures, `ContinuousScheduler._retry_requests_individually()` retries each currently-active request in its own forward pass, one at a time -- O(n) forward passes for an n-request batch. This only runs on the rare double-failure path (not the hot loop), so the added latency is traded for not failing every co-batched request over one bad one.
- **Stop-sequence detection re-decodes the full token history each step**: both `ContinuousScheduler._dispatch_tokens()` and `streaming/stream_manager.py` call `tokenizer_service.decode()` on the *entire* accumulated token list whenever `stop_sequences` is non-empty (not just the newest token), since a stop sequence can span multiple tokens and needn't align with token boundaries. This mirrors the existing multi-byte-character-safe decoding pattern already used for normal streaming, but means per-step cost grows with sequence length for requests that set `stop`.
- **Model swap is single-process**: `POST /api/model` only reloads the model in the process that handles the request; see Extension Points above.
- **`internal/unused_code/` is dead code, not part of the running app**: `batch_scheduler.py` and `sse.py` there are earlier, superseded prototypes (a simpler pre-paged-cache batch scheduler and an O(n²)-decode SSE token streamer) kept only for historical reference. Nothing in the live app imports from `internal/`; see `scheduler/batch_scheduler.py`, `scheduler/continuous_scheduler.py`, and `streaming/stream_manager.py` for the actual implementations.
