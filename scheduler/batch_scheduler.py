import asyncio
import time
from typing import List

from logger import setup_logger
from scheduler.request import InferenceRequest
from scheduler.request_queue import batch_request_queue
from settings.settings import logging_settings, model_settings
from tokenizer.tokenizer_service import tokenizer_service
from utils.device_cache import empty_device_cache
from metrics.metrics import metrics

logger = setup_logger(__name__, level=logging_settings.log_level, log_file=logging_settings.log_file)


class BatchScheduler:
    def __init__(
        self,
        engine,
        tokenizer,
        max_batch_size: int = 8,
        queue_timeout: float = 0.02,
        request_timeout: float = 20.0,
    ):
        self.engine = engine
        self.tokenizer = tokenizer
        self.max_batch_size = max_batch_size
        self.queue_timeout = queue_timeout
        self.request_timeout = request_timeout

    async def _collect_batch(self) -> List[InferenceRequest]:
        first_request = await batch_request_queue.get()
        batch = [first_request]
        batch_start = time.monotonic()
        metrics.record_queue_latency(batch_start - first_request.enqueue_time)

        while len(batch) < self.max_batch_size:
            try:
                remaining = self.queue_timeout - (time.monotonic() - batch_start)
                if remaining <= 0:
                    break
                request = await asyncio.wait_for(batch_request_queue.get(), timeout=remaining)
                batch.append(request)
                metrics.record_queue_latency(time.monotonic() - request.enqueue_time)
            except asyncio.TimeoutError:
                break

        return batch

    async def process_batch(self, batch: List[InferenceRequest]) -> None:
        valid_requests: List[InferenceRequest] = []
        for req in batch:
            if req.future.done() or req.future.cancelled():
                continue
            deadline = getattr(req, "deadline", None)
            if deadline is not None and deadline <= time.monotonic():
                if not req.future.done():
                    req.future.set_exception(asyncio.TimeoutError("Batch generation timed out while waiting for scheduled execution."))
                continue
            valid_requests.append(req)

        if not valid_requests:
            return

        for req in valid_requests:
            req.queue_latency_ms = time.monotonic() - req.enqueue_time

        prompts = [req.prompt for req in valid_requests]
        
        # Apply chat template if available
        tokenizer_obj = self.tokenizer.tokenizer
        if hasattr(tokenizer_obj, 'apply_chat_template'):
            try:
                formatted_prompts = []
                for prompt in prompts:
                    messages = [{"role": "user", "content": prompt}]
                    formatted = tokenizer_obj.apply_chat_template(
                        messages,
                        tokenize=False,
                        add_generation_prompt=True
                    )
                    formatted_prompts.append(formatted)
                prompts = formatted_prompts
                logger.info("Applied chat template to %d batch prompts", len(prompts))
            except Exception as e:
                logger.warning("Failed to apply chat template to batch: %s, using raw prompts", e)
        
        encoded = self.tokenizer.tokenizer(
            prompts,
            return_tensors="pt",
            padding=True,
            truncation=True,
            max_length=model_settings.max_length,
        )

        input_ids = encoded["input_ids"]
        attention_mask = encoded["attention_mask"]
        
        # Track initial sequence length for each request (for KV-cached attention masks)
        for req, inp in zip(valid_requests, input_ids):
            setattr(req, 'seq_length', inp.shape[0])

        start_time = time.monotonic()
        try:
            outputs = await self.engine.generate_batch(input_ids, attention_mask, valid_requests)
        except Exception as exc:
            logger.exception("Batch generation failed: %s", exc)
            for req in valid_requests:
                if not req.future.done() and not req.future.cancelled():
                    req.future.set_exception(exc)
            return

        duration = time.monotonic() - start_time
        token_count = sum(len(req.generated_tokens) for req in valid_requests)
        metrics.record_batch_size(len(valid_requests))
        metrics.record_token_throughput(token_count, duration)

        for req, output in zip(valid_requests, outputs):
            if req.future.done() or req.future.cancelled():
                continue
            if output is None:
                req.future.set_result("")
            else:
                req.future.set_result(output)

        logger.info(
            "Processed batch of %d requests in %.2fms, tokens=%d, throughput=%.2ftoks/s",
            len(valid_requests),
            duration * 1000.0,
            token_count,
            token_count / duration if duration > 0 else 0.0,
        )

    async def run(self) -> None:
        while True:
            batch = []
            try:
                batch = await self._collect_batch()
                if not batch:
                    await asyncio.sleep(0.01)
                    continue
                await self.process_batch(batch)
                # Release cached-but-unused device memory once this batch is
                # fully done -- PyTorch's allocator otherwise holds onto it
                # rather than returning it to the system, which can
                # accumulate across many batches on a long-running process.
                empty_device_cache(self.engine.device)
            except asyncio.CancelledError:
                logger.info("Batch scheduler task cancelled")
                raise
            except Exception as exc:
                logger.exception("Unexpected batch scheduler error: %s", exc)
                for req in batch:
                    if not req.future.done() and not req.future.cancelled():
                        req.future.set_exception(exc)
                await asyncio.sleep(0.1)
