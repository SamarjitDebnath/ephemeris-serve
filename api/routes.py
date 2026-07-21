import asyncio

from fastapi import APIRouter, HTTPException, Request
from sse_starlette.sse import EventSourceResponse
from schemas.schemas import (
    BatchGenerateRequest,
    BatchGenerateResponse,
    GenerateRequest,
)
from scheduler.request import InferenceRequest
from streaming.stream_manager import stream_response
from scheduler.request_queue import batch_request_queue, request_queue
from settings.settings import logging_settings
from logger import setup_logger
from metrics.metrics import metrics

logger = setup_logger(__name__, level=logging_settings.log_level, log_file=logging_settings.log_file)


router = APIRouter()


@router.post("/generate")
async def generate(req: GenerateRequest, request: Request):
    logger.info(
        "Received /generate request: prompt=%s max_tokens=%s temperature=%s",
        req.prompt,
        req.max_tokens,
        req.temperature,
    )
    inference_request = InferenceRequest(
        req.prompt,
        req.max_tokens,
        req.temperature
    )

    # Enqueue the request for the continuous scheduler
    await request_queue.put(inference_request)
    logger.debug("Enqueued inference request and returning SSE stream")

    # TODO: Consider creating a separate function to handle the cancellation logic for better testability and maintainability.
    async def cancel_on_disconnect() -> None:
        """Monitor for client disconnection and cancel the request."""
        try:
            while not await request.is_disconnected():
                await asyncio.sleep(0.1)
        except asyncio.CancelledError:
            return
        
        # Client disconnected; cancel the inference request
        if not inference_request.future.done():
            logger.info("Client disconnected; cancelling inference request")
            inference_request.future.cancel()

    cancel_task = asyncio.create_task(cancel_on_disconnect())
    
    try:
        # Return an SSE stream of decoded tokens
        return EventSourceResponse(stream_response(inference_request))
    finally:
        cancel_task.cancel()
        try:
            await cancel_task
        except asyncio.CancelledError:
            pass


@router.post("/generate_batch", response_model=BatchGenerateResponse)
async def generate_batch(batch_req: BatchGenerateRequest, request: Request):
    logger.info(
        "Received /generate_batch request: batch_size=%s",
        len(batch_req.requests),
    )

    batch_requests = []
    for item in batch_req.requests:
        batch_request = InferenceRequest(item.prompt, item.max_tokens, item.temperature)
        batch_request.deadline = batch_request.enqueue_time + 20.0
        batch_requests.append(batch_request)

    # TODO: Consider creating a separate function to handle the cancellation logic for better testability and maintainability.
    async def cancel_on_disconnect() -> None:
        try:
            while not await request.is_disconnected():
                await asyncio.sleep(0.01)
        except asyncio.CancelledError:
            return
        for batch_request in batch_requests:
            if not batch_request.future.done():
                batch_request.future.cancel()

    cancel_task = asyncio.create_task(cancel_on_disconnect())

    try:
        for batch_request in batch_requests:
            await batch_request_queue.put(batch_request)

        results = await asyncio.wait_for(
            asyncio.gather(*(req.future for req in batch_requests), return_exceptions=True),
            timeout=25.0,
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
            raise HTTPException(status_code=500, detail=str(result))
        if not isinstance(result, str):
            raise HTTPException(status_code=500, detail=f"Unexpected non-string batch result: {type(result).__name__}")
        outputs.append(result)

    # TODO: Consider moving the metrics calculation to a separate function or class for better testability and maintainability.
    queue_latency_values = [getattr(req, "queue_latency_ms", None) for req in batch_requests]
    valid_queue_values = [value for value in queue_latency_values if value is not None]
    queue_latency_ms = (
        (sum(valid_queue_values) / len(valid_queue_values)) * 1000.0
        if valid_queue_values
        else None
    )
    token_throughput_per_sec = metrics.snapshot()["average_token_throughput_per_sec"]
    return BatchGenerateResponse(
        outputs=outputs,
        batch_size=len(outputs),
        queue_latency_ms=queue_latency_ms,
        token_throughput_per_sec=token_throughput_per_sec,
    )


@router.get("/metrics")
async def metrics_endpoint():
    return metrics.snapshot()
