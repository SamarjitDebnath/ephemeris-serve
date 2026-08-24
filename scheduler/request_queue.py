import asyncio
import heapq
import itertools
import time

from settings.settings import scheduler_settings


class RequestQueue:
    """Plain FIFO queue.

    Still used for `batch_request_queue`: the batch path collects a whole batch
    and returns it in one response, so reordering within it buys nothing and
    would only make the response order surprising.
    """

    def __init__(self):
        self.queue = asyncio.Queue()

    async def put(self, request):
        await self.queue.put(request)

    async def get(self):
        return await self.queue.get()

    def empty(self) -> bool:
        return self.queue.empty()


class PriorityRequestQueue:
    """Cost-classed queue with aging, for the continuous scheduler.

    Continuous batching means a long request does not block the queue outright
    -- slots free as individual requests finish. The failure it does cause is
    *slot occupancy*: with `max_batch_size` long generations admitted, a short
    request that arrives behind them waits for one of them to **finish**, not
    for its turn. At the default `streaming_request_timeout_seconds` those
    short requests are evicted having never run, so the server drops work it
    had the capacity to do cheaply.

    Ordering is `(effective_priority, sequence)`:

    * **Class** comes from `max_tokens` at construction. Short requests sort
      ahead of long ones.
    * **Aging** subtracts elapsed wait time, so a long request is promoted the
      longer it waits. Without this, fixing one starvation creates another --
      continuous short traffic would keep long requests queued forever.
    * **Sequence** breaks ties in arrival order, which keeps behavior identical
      to the old FIFO within a class.

    The interface is deliberately the same as `RequestQueue`: `get()` is a real
    coroutine that blocks when empty (the scheduler awaits it under
    `asyncio.wait_for`, so polling would break that), and `empty()` covers every
    class at once (`scheduler/model_swap.py` drains on it).
    """

    def __init__(self):
        self._heap: list = []
        self._counter = itertools.count()
        self._condition = asyncio.Condition()

    @staticmethod
    def _sort_key(request) -> float:
        """Aging priority, computed once at push time.

        The obvious implementation recomputes `base - waited / aging` on every
        pop, because aging depends on elapsed time. It does not need to. For
        two queued requests compared at the same instant:

            base_i - (now - t_i)/a   vs   base_j - (now - t_j)/a

        the `now/a` term is common to both sides, so it cancels. Their relative
        order is decided entirely by `base + t/a`, which is fixed the moment a
        request arrives and never changes. That makes a plain heap correct --
        no re-scoring, no re-heapify, O(log n) per operation.

        Lower sorts first. A long request enqueued `aging_seconds` earlier ties
        with a short one that just arrived, which is exactly what that setting
        is meant to mean.
        """
        base = float(getattr(request, "priority_class", 0))
        aging_seconds = scheduler_settings.priority_aging_seconds
        if aging_seconds <= 0:
            return base
        return base + getattr(request, "enqueue_time", time.monotonic()) / aging_seconds

    async def put(self, request):
        async with self._condition:
            heapq.heappush(self._heap, (self._sort_key(request), next(self._counter), request))
            self._condition.notify()

    async def get(self):
        async with self._condition:
            while not self._heap:
                await self._condition.wait()
            _, _, request = heapq.heappop(self._heap)
            return request

    def empty(self) -> bool:
        return not self._heap

    def qsize(self) -> int:
        return len(self._heap)

    def count_by_class(self) -> dict:
        """Queue depth per priority class, for metrics."""
        counts: dict = {}
        for _, _, request in self._heap:
            key = getattr(request, "priority_class", 0)
            counts[key] = counts.get(key, 0) + 1
        return counts


request_queue = PriorityRequestQueue()
batch_request_queue = RequestQueue()
