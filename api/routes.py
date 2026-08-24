import asyncio

from fastapi import APIRouter, Depends, HTTPException, Request, Response
from sse_starlette.sse import EventSourceResponse
from schemas.schemas import (
    BatchGenerateRequest,
    BatchGenerateResponse,
    GenerateRequest,
    ModelSwapRequest,
    ModelSwapResponse,
)
from api.auth import require_admin_api_key, require_api_key
from api.ratelimit import identity_for, limiter, rate_limited, release_after
from scheduler.request import InferenceRequest
from scheduler.idempotency import idempotency_store
from streaming.stream_manager import stream_response
from scheduler.request_queue import batch_request_queue, request_queue
from scheduler.model_swap import swap_lock, swap_model_coordinated
from scheduler import model_state
from settings.settings import logging_settings, model_settings, scheduler_settings
from utils.errors import INTERNAL_ERROR_MESSAGE
from logger import setup_logger
from metrics.metrics import metrics, streaming_metrics, summarize_batch_response_metrics
from metrics.prometheus import CONTENT_TYPE_LATEST, render_latest

logger = setup_logger(__name__, level=logging_settings.log_level, log_file=logging_settings.log_file)


router = APIRouter()


async def cancel_futures_on_disconnect(
    request: Request,
    futures: list[asyncio.Future],
    poll_interval: float = 0.1,
) -> None:
    """Cancel every future in ``futures`` once the client disconnects."""
    try:
        while not await request.is_disconnected():
            await asyncio.sleep(poll_interval)
    except asyncio.CancelledError:
        return

    logger.info("Client disconnected; cancelling %d pending request(s)", len(futures))
    for future in futures:
        if not future.done():
            future.cancel()


async def _replay_result(result: str):
    """Single-frame SSE stream replaying an already-completed result."""
    yield result


@router.post("/generate")
async def generate(
    req: GenerateRequest,
    request: Request,
    token: str | None = Depends(require_api_key),
):
    if swap_lock.locked():
        raise HTTPException(status_code=503, detail="Model swap in progress; try again shortly.")

    logger.info(
        "Received /generate request: prompt=%s max_tokens=%s temperature=%s "
        "top_k=%s top_p=%s idempotency_key=%s stop=%s",
        req.prompt,
        req.max_tokens,
        req.temperature,
        req.top_k,
        req.top_p,
        req.idempotency_key,
        req.stop,
    )

    if req.idempotency_key is not None:
        existing = idempotency_store.get(req.idempotency_key)
        if existing is not None:
            if existing.future.done() and existing.future.exception() is None:
                logger.info("Replaying cached result for idempotency_key=%s", req.idempotency_key)
                return EventSourceResponse(_replay_result(existing.future.result()))
            if existing.future.done():
                # Prior attempt failed -- discard and let this request retry fresh.
                idempotency_store.discard(req.idempotency_key)
            else:
                raise HTTPException(
                    status_code=409,
                    detail="A request with this idempotency key is already in progress.",
                )

    inference_request = InferenceRequest(
        prompt=req.prompt,
        max_tokens=req.max_tokens,
        temperature=req.temperature,
        stop_sequences=req.stop,
        top_k=req.top_k,
        top_p=req.top_p,
    )
    inference_request.deadline = (
        inference_request.enqueue_time + scheduler_settings.streaming_request_timeout_seconds
    )

    if req.idempotency_key is not None:
        idempotency_store.put(req.idempotency_key, inference_request)

    # Charged before the request is enqueued, so a throttled caller never
    # occupies a scheduler slot. The matching release is deferred to the end of
    # the SSE stream (see `release_after`) rather than to this handler's
    # return: generation is still running at that point, and freeing the
    # concurrency slot here would let a caller hold many streams open at once.
    identity = identity_for(token, request)
    await limiter.acquire(identity, cost=1)

    try:
        # Enqueue the request for the continuous scheduler
        await request_queue.put(inference_request)
    except Exception:
        await limiter.release(identity)
        raise
    logger.debug("Enqueued inference request and returning SSE stream")

    async def _cancel_on_client_disconnect(message: dict) -> None:
        # EventSourceResponse's own disconnect listener (below) is the sole
        # reader of the ASGI receive channel for this request -- a second
        # poller racing it via `request.is_disconnected()` (as this used to
        # do) loses that race every time, since sse_starlette's listener has
        # no timeout and always consumes the one-shot `http.disconnect`
        # message first. That left the future never cancelled, so the
        # scheduler kept generating tokens for a client that was long gone.
        # Hooking sse_starlette's own listener via `client_close_handler_callable`
        # sidesteps the race entirely.
        logger.info("Client disconnected; cancelling in-flight request")
        if not inference_request.future.done():
            inference_request.future.cancel()

    # Return an SSE stream of decoded tokens
    return EventSourceResponse(
        release_after(stream_response(inference_request), identity),
        client_close_handler_callable=_cancel_on_client_disconnect,
    )


@router.post(
    "/generate_batch",
    response_model=BatchGenerateResponse,
)
async def generate_batch(
    batch_req: BatchGenerateRequest,
    request: Request,
    token: str | None = Depends(require_api_key),
):
    if swap_lock.locked():
        raise HTTPException(status_code=503, detail="Model swap in progress; try again shortly.")

    logger.info(
        "Received /generate_batch request: batch_size=%s",
        len(batch_req.requests),
    )

    # Cost is the sub-request count, not 1: a batch of 32 is 32 generations'
    # worth of work, and charging it as a single request would let one caller
    # bypass the limit by batching. This is why the charge happens here rather
    # than in a route-level dependency -- the cost is only known once the body
    # has been parsed.
    async with rate_limited(request, token, cost=len(batch_req.requests)):
        batch_requests = []
        for item in batch_req.requests:
            batch_request = InferenceRequest(
                prompt=item.prompt,
                max_tokens=item.max_tokens,
                temperature=item.temperature,
                stop_sequences=item.stop,
                top_k=item.top_k,
                top_p=item.top_p,
            )
            batch_request.deadline = batch_request.enqueue_time + scheduler_settings.batch_request_timeout_seconds
            batch_requests.append(batch_request)

        cancel_task = asyncio.create_task(
            cancel_futures_on_disconnect(
                request, [req.future for req in batch_requests], poll_interval=0.01
            )
        )

        try:
            for batch_request in batch_requests:
                await batch_request_queue.put(batch_request)

            results = await asyncio.wait_for(
                asyncio.gather(*(req.future for req in batch_requests), return_exceptions=True),
                timeout=scheduler_settings.batch_generation_timeout_seconds,
            )
        except asyncio.TimeoutError:
            for batch_request in batch_requests:
                if not batch_request.future.done():
                    batch_request.future.cancel()
            raise HTTPException(status_code=504, detail="Batch generation timed out.")
        finally:
            cancel_task.cancel()

        outputs = []
        for result in results:
            if isinstance(result, asyncio.CancelledError):
                raise HTTPException(status_code=499, detail="Client disconnected during batch generation.")
            if isinstance(result, BaseException):
                # `result` may carry internal detail (stack-trace-flavored text,
                # memory sizes, ...) -- logged in full server-side, but never
                # returned to the client as-is.
                logger.error("Batch item failed: %s", result)
                raise HTTPException(status_code=500, detail=INTERNAL_ERROR_MESSAGE)
            if not isinstance(result, str):
                logger.error("Unexpected non-string batch result: %s", type(result).__name__)
                raise HTTPException(status_code=500, detail=INTERNAL_ERROR_MESSAGE)
            outputs.append(result)

        metrics_summary = summarize_batch_response_metrics(batch_requests)
        return BatchGenerateResponse(
            outputs=outputs,
            batch_size=len(outputs),
            queue_latency_ms=metrics_summary["queue_latency_ms"],
            token_throughput_per_sec=metrics_summary["token_throughput_per_sec"],
        )


@router.get(
    "/model",
    response_model=ModelSwapResponse,
    dependencies=[Depends(require_api_key)],
)
async def current_model():
    """What this worker is serving, plus how far the pool has converged.

    The model name is this worker's; the convergence counts are pool-wide. A
    client that just swapped polls this until `converged_workers` reaches
    `known_workers`.
    """
    converged, known = model_state.convergence()
    state = model_state.read_state()
    return ModelSwapResponse(
        model_name=model_settings.model_name,
        generation=state[1] if state is not None else None,
        converged_workers=converged if model_state.enabled() else None,
        known_workers=known if model_state.enabled() else None,
    )


@router.post(
    "/model",
    response_model=ModelSwapResponse,
    dependencies=[Depends(require_admin_api_key)],
)
async def switch_model(swap_req: ModelSwapRequest, request: Request):
    """Hot-swap the model this server is running, without a process restart.

    Waits for in-flight requests to drain (see `scheduler/model_swap.py`),
    then loads the new model in place. `/generate` and `/generate_batch`
    reject with 503 for the duration.
    """
    if swap_lock.locked():
        raise HTTPException(status_code=409, detail="A model swap is already in progress.")

    scheduler = getattr(request.app.state, "scheduler", None)
    if scheduler is None:
        raise HTTPException(status_code=503, detail="Scheduler not ready yet.")

    logger.info("Received /model swap request: model_name=%s", swap_req.model_name)
    drain_timeout = swap_req.drain_timeout_seconds or scheduler_settings.model_swap_drain_timeout_seconds

    try:
        new_name, generation = await swap_model_coordinated(
            swap_req.model_name, scheduler, drain_timeout
        )
    except TimeoutError as exc:
        # Deliberately-authored, safe message (not a raw internal exception) --
        # fine to return as-is.
        raise HTTPException(status_code=504, detail=str(exc))
    except Exception as exc:
        logger.exception("Model swap to '%s' failed: %s", swap_req.model_name, exc)
        raise HTTPException(status_code=500, detail=INTERNAL_ERROR_MESSAGE)

    converged, known = model_state.convergence()
    return ModelSwapResponse(
        model_name=new_name,
        generation=generation,
        converged_workers=converged if model_state.enabled() else None,
        known_workers=known if model_state.enabled() else None,
    )


@router.get("/metrics", dependencies=[Depends(require_api_key)])
async def metrics_endpoint():
    """Rolling averages as JSON. Unchanged -- the CLI consumes this shape.

    The Prometheus export at `GET /metrics` is a separate endpoint with raw
    counters and histograms; see `metrics/prometheus.py` for why the two are
    not derived from each other.
    """
    return {"batch": metrics.snapshot(), "streaming": streaming_metrics.snapshot()}


async def prometheus_endpoint(_: str | None = Depends(require_api_key)) -> Response:
    """Prometheus exposition format.

    Registered on the app (not this `/api` router) at `GET /metrics`, which is
    the path scrapers default to. Auth is configurable because a scraper
    usually does not send bearer tokens -- see `metrics_config.require_auth`.
    """
    return Response(content=render_latest(), media_type=CONTENT_TYPE_LATEST)


async def prometheus_endpoint_open() -> Response:
    """`prometheus_endpoint` without the key requirement."""
    return Response(content=render_latest(), media_type=CONTENT_TYPE_LATEST)
