import asyncio
import time
import torch
from typing import Optional
from cache.paged_kv_cache import BlockTable
from settings.settings import model_settings


class InferenceRequest:
    def __init__(self, prompt, max_tokens, temperature, stop_sequences=None):
        self.prompt = prompt
        self.max_tokens = max_tokens if max_tokens is not None else model_settings.max_length
        self.temperature = temperature if temperature is not None else model_settings.temperature
        self.stop_sequences: list[str] = list(stop_sequences) if stop_sequences else []

        try:
            loop = asyncio.get_running_loop()
        except RuntimeError:
            loop = asyncio.new_event_loop()
            asyncio.set_event_loop(loop)
        self.future = loop.create_future()
        # Streaming queue for token-wise output
        self.queue = asyncio.Queue()
        self.enqueue_time = time.monotonic()
        self.deadline: float | None = None
        self.queue_latency_ms: float | None = None
        # State for continuous generation
        self.input_ids: Optional[torch.Tensor] = None  # will be set when added to scheduler
        self.generated_tokens: list[int] = []
        self.finished = False
        self.block_table = BlockTable()  # paged KV cache state (see cache/paged_kv_cache.py)
