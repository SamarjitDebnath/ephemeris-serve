# API Layer

## API Layer

### `api/server.py`

This module constructs the FastAPI application and defines application lifecycle behavior.

Imports:
- `FastAPI` and `JSONResponse` from `fastapi`
- `asynccontextmanager` from `contextlib`
- `login` from `huggingface_hub`
- `asyncio`
- `router` from `api.routes`
- `HealthResponse` from `schemas.schemas`
- `logging_settings`, `model_settings`, `scheduler_settings`, `secret_settings` from `settings.settings`
- `Utils` from `utils.utils`
- `setup_logger` from `logger`

Lifecycle logic:
- `lifespan` is an async context manager used by FastAPI for startup and shutdown.
- On startup:
  - Log `Starting up the server...`.
  - Call `Utils.configure_multiprocessing()` early, before any heavy `torch`/model imports, to avoid semaphore/`resource_tracker` warnings.
  - Import `tokenizer_service`, `model_loader`, `BatchScheduler`, `ContinuousScheduler`, and `engine` (deferred until after multiprocessing configuration, for the same reason).
  - If `secret_settings.hf_key` is defined, call `login(token=secret_settings.hf_key)` to authenticate to the Hugging Face Hub; otherwise log a warning and continue with anonymous access.
  - Call `tokenizer_service.load()` to instantiate the tokenizer.
  - Call `model_loader.load()` to instantiate the model and move it to the configured device.
  - Call `model_loader.warmup()` to run a single forward pass and initialize internal model caches.
  - Instantiate `ContinuousScheduler(engine, tokenizer_service)` and store it as `app.state.scheduler` -- this is how route handlers (e.g. `POST /api/model`) reach the live scheduler instance, since it's otherwise only closed over by the background task.
  - Start the scheduler loop in the background with `asyncio.create_task(scheduler.run())`.
  - Instantiate `BatchScheduler(engine, tokenizer_service, request_timeout=scheduler_settings.batch_request_timeout_seconds)` and start its loop the same way.
- On shutdown:
  - Cancel both scheduler tasks.
  - Await each canceled task and ignore `asyncio.CancelledError`.

App creation:
- `create_app()` builds `FastAPI(title="Ephemeris Serve", version="0.1.0", lifespan=lifespan)`.
- Defines `/` root endpoint returning a JSON welcome message.
- Defines `/health` endpoint returning a `HealthResponse` model with `status: "healthy"`.
- Includes `api.routes` under prefix `/api`.
- Exposes `app = create_app()`.

Notes:
- All request and lifecycle logs are written using the configured logger.
- Startup and shutdown are tied to FastAPI's lifespan, ensuring model and scheduler lifecycle are managed automatically.

### `api/auth.py`

API-key authentication for the `/api` routes, applied as FastAPI dependencies rather than middleware, so the protected surface is visible in each route decorator.

Two tiers, both comma-separated lists read from the environment via `SecretSetting` (`settings/settings.py`, accepting `EPHEMERIS_SERVER_API_KEYS`/`EPHEMERIS_SERVER_ADMIN_API_KEYS` or the unprefixed names):
- `api_keys()` -- ordinary access: `/generate`, `/generate_batch`, `GET /model`, `/metrics`. Admin keys are concatenated in, so an admin key works everywhere.
- `admin_api_keys()` -- additionally authorizes `POST /model`, the one route that makes the server download and load an arbitrary Hugging Face repo (disk exhaustion, OOM, or a drain-and-reload stall are all reachable from it).

`auth_enabled()`: true once any key is configured. **With no keys configured every route is open** -- deliberate, so `make run` and the test suite need no setup -- and `api/server.py`'s lifespan logs a prominent warning in that state. It also warns when ordinary keys exist but no admin key does, since `POST /model` then rejects everything.

`_extract_bearer(authorization)`: parses `Authorization: Bearer <token>`; any other scheme, or an empty token, yields `None`.

`_matches(candidate, allowed)`: `hmac.compare_digest` against every entry, and deliberately does **not** short-circuit on the first match -- constant-time comparison stops response timing from leaking how much of a key was correct, and comparing every entry keeps the comparison count independent of which key matched.

`require_api_key(authorization)`: dependency raising `401` for a missing (`_MISSING_KEY_MESSAGE`) or unrecognized (`_INVALID_KEY_MESSAGE`) key; returns the presented key so `require_admin_api_key` can depend on it rather than re-reading the header. A rejected key is logged **without its value** -- a mistyped key is often a real key from another environment.

`require_admin_api_key(token)`: `403` if the presented key isn't in the admin list, and also `403` when ordinary keys exist but no admin key is configured -- refusing rather than silently letting every key swap the model.

`/health` and `/` live on the app, not the router, so they stay unauthenticated for proxy health checks and uptime monitors.

### `api/routes.py`

This module implements the public inference endpoints.

Imports:
- `APIRouter`, `HTTPException`, `Request` from `fastapi`
- `EventSourceResponse` from `sse_starlette.sse`
- `GenerateRequest`, `BatchGenerateRequest`, `BatchGenerateResponse`, `ModelSwapRequest`, `ModelSwapResponse` from `schemas.schemas`
- `InferenceRequest` from `scheduler.request`
- `idempotency_store` from `scheduler.idempotency`
- `stream_response` from `streaming.stream_manager`
- `request_queue`, `batch_request_queue` from `scheduler.request_queue`
- `swap_lock`, `swap_model_coordinated` from `scheduler.model_swap`; `model_state` from `scheduler`
- `logging_settings`, `model_settings`, `scheduler_settings` from `settings.settings`
- `INTERNAL_ERROR_MESSAGE` from `utils.errors` (see [Utility Helpers](Reference#utility-helpers))
- `metrics`, `streaming_metrics`, `summarize_batch_response_metrics` from `metrics.metrics` (see [Metrics](Streaming-and-Metrics#metrics))
- `setup_logger` from `logger`

`cancel_futures_on_disconnect(request, futures, poll_interval)`:
- Used by `/generate_batch` (a non-streaming JSON endpoint, so there's no SSE listener already reading the disconnect signal). `/generate` instead hooks `EventSourceResponse`'s own `client_close_handler_callable` -- see below.
- Polls `request.is_disconnected()` in a loop; once the client disconnects, cancels every future in `futures` that isn't already done.

`_replay_result(result)`:
- A single-frame async generator that yields an already-completed result string -- used to replay a cached idempotent response through the same `EventSourceResponse` shape as a live stream.

`POST /generate`:
- Rejects immediately with `503` if `swap_lock.locked()` (a model hot-swap is in progress -- see `scheduler/model_swap.py`).
- Logs the prompt, `max_tokens`, `temperature`, `idempotency_key`, and `stop`.
- If `idempotency_key` is set: checks `idempotency_store`. A completed, non-failed prior result is replayed via `_replay_result`; a still-in-flight prior request returns `409`; a failed prior attempt is discarded and this request proceeds fresh.
- Constructs `InferenceRequest(req.prompt, req.max_tokens, req.temperature, req.stop)` and sets its `deadline = enqueue_time + scheduler_settings.streaming_request_timeout_seconds`.
- If `idempotency_key` is set, stores the request in `idempotency_store` before enqueueing.
- Enqueues onto `request_queue` and returns `EventSourceResponse(stream_response(inference_request), client_close_handler_callable=_cancel_on_client_disconnect)`.
- `_cancel_on_client_disconnect(message)`: cancels `inference_request.future` if not already done. Passed as `EventSourceResponse`'s own `client_close_handler_callable` rather than run via `cancel_futures_on_disconnect` as a separate polling task (unlike `/generate_batch`, below) -- `sse_starlette`'s `EventSourceResponse` is itself the sole reader of the ASGI `receive` channel for this request, and its disconnect listener has no timeout, so a second poller calling `request.is_disconnected()` in parallel always loses the race for the one-shot `http.disconnect` message. That previously left the future never cancelled, so the scheduler kept generating tokens for a client that had already gone away. Hooking `EventSourceResponse`'s own listener sidesteps the race entirely.

`POST /generate_batch`:
- Rejects immediately with `503` if `swap_lock.locked()`.
- Builds one `InferenceRequest` per item in `batch_req.requests` (each with its own `stop`), each with `deadline = enqueue_time + scheduler_settings.batch_request_timeout_seconds`.
- Enqueues every request onto `batch_request_queue`, then `asyncio.gather`s their futures under an overall `asyncio.wait_for(..., timeout=scheduler_settings.batch_generation_timeout_seconds)`, with `cancel_futures_on_disconnect` running as a background task for the duration (this endpoint isn't SSE, so there's no `EventSourceResponse` listener to hook instead -- the race described above doesn't apply here).
- On overall timeout: cancels any still-pending futures and raises `504`.
- Per-result handling: a cancelled future raises `499`; any other exception is logged in full server-side and raises `500` with `detail=INTERNAL_ERROR_MESSAGE` (never the raw exception text); a non-`str` result raises `500` with the same generic detail, as a defensive check.
- Returns a `BatchGenerateResponse` with per-batch `queue_latency_ms`/`token_throughput_per_sec` from `metrics.summarize_batch_response_metrics()`.

`GET /model`:
- Returns `ModelSwapResponse(model_name=model_settings.model_name)` -- the model currently loaded and serving.

`POST /model`:
- Returns `409` if a swap is already in progress (`swap_lock.locked()`).
- Returns `503` if `request.app.state.scheduler` isn't set yet (server still starting up).
- Calls `scheduler.model_swap.swap_model_coordinated(swap_req.model_name, scheduler, drain_timeout)`, where `drain_timeout` is `swap_req.drain_timeout_seconds` or `scheduler_settings.model_swap_drain_timeout_seconds`. That swaps the handling worker first, then publishes the new target for the rest of the pool -- in that order, so a model this worker could not load is never advertised to the others.
- **A multi-worker swap is not atomic, and the response says so.** Each worker drains its own in-flight requests before reloading, and those drains finish at different times, so there is necessarily a window where different workers serve different models. `ModelSwapResponse` therefore carries `generation`, `converged_workers`, and `known_workers`; a client polls `GET /api/model` until the two counts match. All three are null when `scheduler_config.model_state_dir` is unset, which is the single-process case where they would be meaningless.
- Maps `TimeoutError` (drain took too long) to `504` with `detail=str(exc)` -- this message is deliberately authored (not raw exception text), so it's safe to return as-is. Any other exception (bad repo id, OOM, ...) is logged in full via `logger.exception` and mapped to `500` with `detail=INTERNAL_ERROR_MESSAGE`, never the raw exception text.
- Returns `ModelSwapResponse(model_name=new_name)` on success.

Client-facing errors never carry raw exception text: every `500` response above, and the SSE `error` event pushed by `ContinuousScheduler._fail_single_request()` (see [Scheduler Layer](Scheduler-Layer#scheduler-layer)), use the same `utils.errors.INTERNAL_ERROR_MESSAGE` constant. The real exception is always logged server-side (`logger.exception`/`logger.warning`) at the point of failure; only deliberately-authored, already-safe messages (e.g. the `409`/`503`/`504` details above) are ever sent verbatim.

`GET /metrics`:
- Returns `{"batch": metrics.snapshot(), "streaming": streaming_metrics.snapshot()}`.

Low-level notes:
- The streaming endpoint does not block on actual model generation; it returns a stream handle immediately.
- The batch endpoint is designed for synchronous batch workflows and returns full text output once generation completes.
- `GenerateRequest`, `BatchGenerateRequest`, and `ModelSwapRequest` validation occur before any request objects are created.
- `/generate` and `/generate_batch` only *check* `swap_lock.locked()` (a fast, non-blocking read); they never wait on the lock, so a swap in progress fails fast rather than queuing behind it.
