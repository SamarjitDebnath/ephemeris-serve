import asyncio
import time
import torch
from typing import Optional
from cache.paged_kv_cache import BlockTable
from settings.settings import model_settings, scheduler_settings


#: Scheduling classes. Lower is served first, all else equal.
SHORT_REQUEST_CLASS = 0
GENERAL_REQUEST_CLASS = 1


class InferenceRequest:
    def __init__(
        self,
        prompt,
        max_tokens,
        temperature,
        stop_sequences=None,
        top_k=None,
        top_p=None,
    ):
        self.prompt = prompt
        self.max_tokens = max_tokens if max_tokens is not None else model_settings.max_length
        self.temperature = temperature if temperature is not None else model_settings.temperature
        self.stop_sequences: list[str] = list(stop_sequences) if stop_sequences else []
        # Longest stop sequence in *characters*. Both the scheduler and the
        # stream manager size their bounded search window from this, so it is
        # computed once here rather than on every generated token.
        self.max_stop_length: int = max((len(s) for s in self.stop_sequences), default=0)
        # Resolved here rather than at the sampling site so the rest of the
        # code never has to ask whether a value was supplied -- the same
        # property that keeps `temperature` clean downstream. Note the
        # `is not None` test: `top_k=0` is a meaningful value (filtering off),
        # not an absent one.
        self.top_k = top_k if top_k is not None else model_settings.top_k
        self.top_p = top_p if top_p is not None else model_settings.top_p

        # Scheduling class, from the only cost signal available before the
        # request runs. 0 = short lane, 1 = general. See
        # `scheduler/request_queue.PriorityRequestQueue`.
        self.priority_class: int = (
            SHORT_REQUEST_CLASS
            if self.max_tokens <= scheduler_settings.short_request_max_tokens
            else GENERAL_REQUEST_CLASS
        )

        try:
            loop = asyncio.get_running_loop()
        except RuntimeError:
            loop = asyncio.new_event_loop()
            asyncio.set_event_loop(loop)
        self.future = loop.create_future()
        # Streaming queue for token-wise output.
        #
        # Deliberately *not* annotated `Optional` to match
        # `engine.generator.GenerationRequest.queue`. Protocol attributes are
        # invariant, so the two types not matching makes Pyright complain
        # wherever an `InferenceRequest` is passed to `generate_batch` -- but
        # widening it here is the worse trade: every `req.queue.put_nowait(...)`
        # in the scheduler and stream manager would then need narrowing, which
        # moves five type errors out of the tests and into the hot path.
        self.queue = asyncio.Queue()
        self.enqueue_time = time.monotonic()
        self.deadline: float | None = None
        self.queue_latency_ms: float | None = None
        # State for continuous generation
        self.input_ids: Optional[torch.Tensor] = None  # will be set when added to scheduler
        self.generated_tokens: list[int] = []
        self.finished = False
        self.block_table = BlockTable()  # paged KV cache state (see cache/paged_kv_cache.py)
