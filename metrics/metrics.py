from __future__ import annotations

from collections import deque
from statistics import mean
from typing import Deque, Dict, Sequence


class BatchMetrics:
    def __init__(self, max_samples: int = 1000):
        self.queue_latencies: Deque[float] = deque(maxlen=max_samples)
        self.batch_sizes: Deque[float] = deque(maxlen=max_samples)
        self.token_throughputs: Deque[float] = deque(maxlen=max_samples)
        # Continuous-scheduler-only gauges/counters (unused by the plain
        # batch path): how full the active batch and paged KV cache are
        # each step, and how often requests get evicted before finishing
        # normally.
        self.batch_occupancies: Deque[float] = deque(maxlen=max_samples)
        self.cache_utilizations: Deque[float] = deque(maxlen=max_samples)
        self.timeout_evictions: int = 0
        self.cancelled_evictions: int = 0

    def record_queue_latency(self, latency_seconds: float) -> None:
        if latency_seconds < 0:
            return
        self.queue_latencies.append(latency_seconds)

    def record_batch_size(self, batch_size: int) -> None:
        if batch_size <= 0:
            return
        self.batch_sizes.append(float(batch_size))

    def record_token_throughput(self, tokens: int, elapsed_seconds: float) -> None:
        if elapsed_seconds <= 0 or tokens < 0:
            return
        self.token_throughputs.append(tokens / elapsed_seconds)

    def record_batch_occupancy(self, active_count: int, max_batch_size: int) -> None:
        if max_batch_size <= 0:
            return
        self.batch_occupancies.append(active_count / max_batch_size)

    def record_cache_utilization(self, used_blocks: int, capacity_blocks: int) -> None:
        if capacity_blocks <= 0:
            return
        self.cache_utilizations.append(used_blocks / capacity_blocks)

    def record_timeout_eviction(self) -> None:
        self.timeout_evictions += 1

    def record_cancelled_eviction(self) -> None:
        self.cancelled_evictions += 1

    def _average(self, values: deque[float]) -> float | None:
        return mean(values) if values else None

    def snapshot(self) -> Dict[str, float | None]:
        average_queue_latency = self._average(self.queue_latencies)
        average_batch_size = self._average(self.batch_sizes)
        average_token_throughput = self._average(self.token_throughputs)
        average_batch_occupancy = self._average(self.batch_occupancies)
        average_cache_utilization = self._average(self.cache_utilizations)
        return {
            "average_queue_latency_ms": average_queue_latency * 1000.0 if average_queue_latency is not None else None,
            "average_batch_size": average_batch_size,
            "average_token_throughput_per_sec": average_token_throughput,
            "queue_latency_samples": len(self.queue_latencies),
            "batch_size_samples": len(self.batch_sizes),
            "throughput_samples": len(self.token_throughputs),
            "average_batch_occupancy": average_batch_occupancy,
            "average_cache_utilization": average_cache_utilization,
            "timeout_evictions": self.timeout_evictions,
            "cancelled_evictions": self.cancelled_evictions,
        }


metrics = BatchMetrics()
streaming_metrics = BatchMetrics()


def summarize_batch_response_metrics(batch_requests: Sequence[object]) -> Dict[str, float | None]:
    """Assemble the queue-latency/throughput fields returned by /generate_batch.

    Queue latency is averaged over the batch's own requests; throughput is
    pulled from the rolling batch-path average since it's a per-engine-call
    measurement, not something computed per response.
    """
    queue_latency_values = [
        getattr(req, "queue_latency_ms", None) for req in batch_requests
    ]
    valid_queue_values = [value for value in queue_latency_values if value is not None]
    queue_latency_ms = (
        (sum(valid_queue_values) / len(valid_queue_values)) * 1000.0
        if valid_queue_values
        else None
    )
    token_throughput_per_sec = metrics.snapshot()["average_token_throughput_per_sec"]
    return {
        "queue_latency_ms": queue_latency_ms,
        "token_throughput_per_sec": token_throughput_per_sec,
    }
