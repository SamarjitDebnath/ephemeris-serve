"""Hot-swaps the model an already-running server serves, without a process
restart -- e.g. from `POST /api/model` (see `api/routes.py`) or the CLI's
`/model` REPL command (see `cli/main.py`).

The engine, scheduler, and paged KV cache are all built around exactly one
loaded model at a time, so a swap must: stop new requests from being
accepted, wait for whatever is already in flight to finish, replace the
model and tokenizer, then invalidate the caches that were sized for the old
model's architecture.
"""
import asyncio
import gc
import time

from engine.generator import engine
from engine.model_loader import model_loader
from tokenizer.tokenizer_service import tokenizer_service
from scheduler.request_queue import request_queue, batch_request_queue
from settings.settings import model_settings, logging_settings
from utils.device_cache import empty_device_cache
from logger import setup_logger

logger = setup_logger(__name__, level=logging_settings.log_level, log_file=logging_settings.log_file)

# Held for the duration of a swap. Route handlers that accept new requests
# check `swap_lock.locked()` (not full acquisition -- a swap can take a
# while) and reject with 503 rather than letting requests pile up against a
# model that's about to disappear.
swap_lock = asyncio.Lock()


async def swap_model(new_model_name: str, continuous_scheduler, drain_timeout: float) -> str:
    """Drain in-flight requests, then replace the loaded model + tokenizer.

    Returns the (new) model name on success. Raises `TimeoutError` if
    in-flight requests don't finish within `drain_timeout` seconds, or
    whatever `AutoModelForCausalLM`/`AutoTokenizer` raise for a bad repo id
    -- in both failure cases the previously-loaded model is left untouched
    and still serving.
    """
    async with swap_lock:
        deadline = time.monotonic() + drain_timeout
        while continuous_scheduler.active_requests or not request_queue.empty() or not batch_request_queue.empty():
            if time.monotonic() >= deadline:
                raise TimeoutError(
                    f"Timed out after {drain_timeout}s waiting for in-flight requests "
                    "to finish before swapping models."
                )
            await asyncio.sleep(0.05)

        previous_model_name = model_settings.model_name
        logger.info("Swapping model: '%s' -> '%s'", previous_model_name, new_model_name)
        try:
            tokenizer_service.reload(new_model_name)
            model_loader.reload(new_model_name)
        except Exception:
            logger.exception(
                "Model swap to '%s' failed; rolling tokenizer back to '%s'",
                new_model_name,
                previous_model_name,
            )
            # Keep the tokenizer paired with whichever model actually ended up
            # loaded -- `model_loader.reload` leaves the old model in place on
            # failure, so the tokenizer must follow it back.
            tokenizer_service.reload(previous_model_name)
            raise

        engine.invalidate_model_cache()
        continuous_scheduler.invalidate_paged_cache()

        # `model_loader.reload()` already ran its own gc.collect()/cache-empty
        # for the old *model*, but at that point the old *paged cache*'s block
        # pool (dropped just above) was still referenced -- so its memory was
        # never actually reclaimed until now.
        gc.collect()
        empty_device_cache(model_settings.device)

        logger.info("Model swap complete: now serving '%s'", new_model_name)
        return new_model_name
