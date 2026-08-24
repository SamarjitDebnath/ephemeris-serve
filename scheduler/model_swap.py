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
from scheduler import model_state
from settings.settings import model_settings, logging_settings, scheduler_settings
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


async def swap_model_coordinated(new_model_name: str, continuous_scheduler, drain_timeout: float):
    """Swap this worker, then publish the new target for the rest of the pool.

    Returns `(model_name, generation)`; `generation` is None when coordination
    is disabled, in which case this is exactly `swap_model`.

    The local swap happens first on purpose. Publishing a target this worker
    then failed to load would leave every other worker chasing a broken model
    name, turning one failed request into a pool-wide outage.
    """
    name = await swap_model(new_model_name, continuous_scheduler, drain_timeout)
    generation = model_state.publish_target(name)
    if generation is not None:
        model_state.publish_worker_generation(generation, name)
        logger.info("Published model target '%s' as generation %d", name, generation)
    return name, generation


async def follow_model_state(continuous_scheduler) -> None:
    """Converge this worker on the published target, if it has fallen behind.

    Called from the scheduler's idle branch, which is the one place that
    already knows this worker has nothing in flight -- exactly the
    precondition a swap needs. A worker still generating simply checks again
    on its next idle tick.
    """
    if not model_state.enabled():
        return
    state = model_state.read_state()
    if state is None:
        return
    target_name, target_generation = state

    local = getattr(continuous_scheduler, "_model_generation", 0)
    if target_generation <= local:
        return

    if target_name == model_settings.model_name:
        # Already serving it (this worker performed the swap, or started with
        # it configured). Record convergence without reloading.
        continuous_scheduler._model_generation = target_generation
        model_state.publish_worker_generation(target_generation, target_name)
        return

    logger.info(
        "Following model swap to '%s' (generation %d -> %d)",
        target_name,
        local,
        target_generation,
    )
    try:
        # The same drain window the request-handling worker gets. A zero
        # timeout would raise on the first tick where the queue happens to be
        # non-empty, even though the batch itself is idle.
        await swap_model(
            target_name,
            continuous_scheduler,
            drain_timeout=scheduler_settings.model_swap_drain_timeout_seconds,
        )
    except Exception:
        # Logged by `swap_model`; leave the local generation behind so the
        # next idle tick retries rather than silently reporting convergence.
        logger.exception("Failed to follow model swap to '%s'", target_name)
        return
    continuous_scheduler._model_generation = target_generation
    model_state.publish_worker_generation(target_generation, target_name)
