# Architecture

## Architecture Flow

```mermaid
flowchart TD
    A[Client] -->|POST /generate| GEN[FastAPI: /generate]
    A -->|POST /generate_batch| BATCH[FastAPI: /generate_batch]
    A -->|GET/POST /model| MODELAPI[FastAPI: /model]

    GEN --> C1[Create InferenceRequest incl. stop_sequences]
    C1 --> Q1[Enqueue to request_queue]
    Q1 --> SCHED1[ContinuousScheduler.run]
    SCHED1 --> ADD[_add_new_requests - chat template, deadline check]
    ADD --> PREP1[_prepare_batch - mixed prefill/decode via paged cache]
    PREP1 --> FWD1[InferenceEngine.forward_step]
    FWD1 --> PEN1[InferenceEngine.apply_repetition_penalty]
    PEN1 --> SMP1[InferenceEngine.sample]
    SMP1 --> DISP1[_dispatch_tokens - stop-sequence check, append K/V to paged cache]
    DISP1 --> PUT1[push token to req.queue / update state]
    PUT1 --> STREAM1[streaming.stream_response - stop-sequence trim]
    STREAM1 -->|SSE tokens| CLIENT1[Client]

    BATCH --> C2[Create InferenceRequest batch]
    C2 --> Q2[Enqueue to batch_request_queue]
    Q2 --> SCHED2[BatchScheduler.run]
    SCHED2 --> COLLECT[_collect_batch]
    COLLECT --> PROC[process_batch]
    PROC --> ENCODE[tokenizer.encode and build tensors]
    ENCODE --> GENB[InferenceEngine.generate_batch - stop-sequence check]
    GENB --> SETF[Set req.future results / push DONE to queues]
    SETF --> CLIENT2[Return BatchGenerateResponse]

    MODELAPI --> SWAP[scheduler.model_swap.swap_model]
    SWAP --> DRAIN[Wait: active_requests and both queues empty]
    DRAIN --> RELOAD[tokenizer_service.reload, then model_loader.reload]
    RELOAD --> INVAL[engine.invalidate_model_cache / scheduler.invalidate_paged_cache]
    INVAL --> CLIENT3[Return ModelSwapResponse]

    subgraph startup[Server startup]
        SRV[api/server.py lifespan]
    end
    SRV -->|create task, store on app.state.scheduler| SCHED1
    SRV -->|create task| SCHED2

    subgraph clientry[CLI entrypoints]
        SERVECMD["ephemeris-serve serve [--model]"] -.->|uvicorn.run| SRV
        STARTCMD[ephemeris-serve start - REPL] -.->|HTTP| GEN
        STARTCMD -.->|/model command| MODELAPI
    end
```

> The scheduler batches active requests and reuses each request's slice of a shared paged KV cache to avoid recomputing full prompts on every step -- including when a brand-new (prefill) request joins the same batched step as requests already mid-decode.

---

## Root Entry Point

### `main.py`

The launcher used during local development and by `make run`/`make run-prod`.

Implementation details:
- Imports `uvicorn` and `app` from `api.server`.
- Calls `uvicorn.run("main:app", host="0.0.0.0", port=8000, workers=1, reload=True)`.
- `workers=1` is explicitly chosen for development and to avoid model loading overhead on multiple workers.
- `reload=True` enables auto-reload on source changes and should be disabled in production.

### `ephemeris-serve serve` (`cli/main.py`)

An alternative entrypoint to `main.py`, installed via the `[project.scripts]` entry point. Runs the same `api.server:app`, but exposes `--host`, `--port`, `--workers`, `--reload`, and (unlike `main.py`) `--model` as CLI flags instead of hardcoded values. See [CLI Layer](CLI-and-Configuration#cli-layer) below.

---

## Internal Control Flow Summary

1. `api/routes.py` receives a validated request (checking `swap_lock` first) and creates `InferenceRequest`.
2. The request is enqueued into `request_queue`.
3. `ContinuousScheduler.run()` wakes up and calls `_add_new_requests()`, which applies the chat template and tokenizes.
4. `_prepare_batch()` builds one batched step's inputs, mixing any brand-new (prefill) requests with any already mid-decode, using the paged KV cache's `gather_dense()` for the past.
5. `InferenceEngine.forward_step()` executes the model and returns logits and the new `past_key_values`.
6. `apply_repetition_penalty()` modifies logits to reduce repetitions.
7. `InferenceEngine.sample()` selects a next token for each active request.
8. `_dispatch_tokens()` streams the token to the client's queue, checks it against any configured `stop` sequences, appends the newly computed K/V into the paged cache, and finishes requests that hit a stop sequence, EOS, or `max_tokens`.
9. `stream_response()` decodes accumulated tokens, independently re-checks `stop` sequences, and yields buffered text fragments as SSE frames.
10. The scheduler repeats until all active requests finish.

(A `POST /api/model` swap instead routes through `scheduler/model_swap.py`: drain `active_requests`/both queues, reload tokenizer then model, invalidate the engine's cached model reference and the scheduler's paged cache.)

---

## Request Lifecycle Summary

1. Client sends `POST /api/generate` with a prompt and optional `max_tokens`/`temperature`/`stop`/`idempotency_key`.
2. `api/routes.py` builds `InferenceRequest` and enqueues it onto `request_queue` (after checking `swap_lock` and any idempotency replay).
3. `ContinuousScheduler` asynchronously pulls queued requests, applies the chat template, and tokenizes.
4. Each step, the scheduler forms a batch mixing any new prefills with any requests already mid-decode, gathering each row's cached past from the shared `PagedKVCache`.
5. `InferenceEngine` computes logits, applies repetition penalty, and samples next tokens.
6. The scheduler dispatches each token to its request's streaming queue, checks it against `stop` sequences, and appends new K/V into the paged cache.
7. `stream_manager` decodes accumulated tokens, buffers to natural boundaries, independently enforces `stop` sequences, and emits fragments through SSE.
8. When a stop sequence matches, EOS is reached, or `max_tokens` is hit, the request completes and a `"[DONE]"` sentinel closes the stream (or a single `error` frame does, on failure/timeout).
