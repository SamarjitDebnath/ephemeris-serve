from __future__ import annotations

from collections import deque
from statistics import mean
from typing import Deque, Dict, Sequence

from metrics.prometheus import prometheus_metrics


class BatchMetrics:
    def __init__(self, max_samples: int = 1000, path: str = "batch"):
        self._max_samples = max_samples
        # Which singleton this is ("batch" or "streaming"), used as the
        # Prometheus label that mirrors the JSON endpoint's two top-level keys.
        self._path = path
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
        # Blocks handed back to the allocator by `PagedKVCache.trim_tail()`.
        # A monotone counter, not a gauge: paired with the utilization
        # samples above it is what makes idle reclamation visible in
        # production -- capacity coming down is otherwise invisible.
        self.kv_blocks_reclaimed: int = 0
        # Queue latency split by scheduling class (see
        # scheduler/request_queue.PriorityRequestQueue). Without this split the
        # fairness change is unmeasurable: a improvement for short requests and
        # a regression for long ones average out to nothing in the aggregate
        # number above.
        self.queue_latencies_by_class: Dict[int, Deque[float]] = {}

    def record_queue_latency(self, latency_seconds: float) -> None:
        if latency_seconds < 0:
            return
        self.queue_latencies.append(latency_seconds)
        prometheus_metrics.queue_latency.labels(self._path).observe(latency_seconds)

    def record_batch_size(self, batch_size: int) -> None:
        if batch_size <= 0:
            return
        self.batch_sizes.append(float(batch_size))
        prometheus_metrics.batch_size.labels(self._path).set(batch_size)

    def record_token_throughput(self, tokens: int, elapsed_seconds: float) -> None:
        if elapsed_seconds <= 0 or tokens < 0:
            return
        throughput = tokens / elapsed_seconds
        self.token_throughputs.append(throughput)
        prometheus_metrics.token_throughput.labels(self._path).observe(throughput)

    def record_batch_occupancy(self, active_count: int, max_batch_size: int) -> None:
        if max_batch_size <= 0:
            return
        occupancy = active_count / max_batch_size
        self.batch_occupancies.append(occupancy)
        prometheus_metrics.batch_occupancy.labels(self._path).set(occupancy)

    def record_cache_utilization(self, used_blocks: int, capacity_blocks: int) -> None:
        if capacity_blocks <= 0:
            return
        utilization = used_blocks / capacity_blocks
        self.cache_utilizations.append(utilization)
        prometheus_metrics.cache_utilization.labels(self._path).set(utilization)

    def record_queue_latency_by_class(self, latency_seconds: float, priority_class: int) -> None:
        if latency_seconds < 0:
            return
        bucket = self.queue_latencies_by_class.get(priority_class)
        if bucket is None:
            bucket = deque(maxlen=self._max_samples)
            self.queue_latencies_by_class[priority_class] = bucket
        bucket.append(latency_seconds)

    def record_kv_blocks_reclaimed(self, blocks: int) -> None:
        if blocks <= 0:
            return
        self.kv_blocks_reclaimed += blocks
        prometheus_metrics.kv_blocks_reclaimed.labels(self._path).inc(blocks)

    def record_timeout_eviction(self) -> None:
        self.timeout_evictions += 1
        prometheus_metrics.timeout_evictions.labels(self._path).inc()

    def record_cancelled_eviction(self) -> None:
        self.cancelled_evictions += 1
        prometheus_metrics.cancelled_evictions.labels(self._path).inc()

    def _average(self, values: deque[float]) -> float | None:
        return mean(values) if values else None

    def snapshot(self) -> Dict[str, "float | int | None | Dict[str, float]"]:
        """Rolling averages as plain JSON.

        The value type is not uniform: most entries are floats, the eviction
        counters are ints, and the per-class breakdown is a nested mapping.
        Declared honestly rather than as `float | None`, which it stopped being
        when the breakdown was added.
        """
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
            "kv_blocks_reclaimed": self.kv_blocks_reclaimed,
            "average_queue_latency_ms_by_class": {
                str(priority_class): average * 1000.0
                for priority_class, average in (
                    (key, self._average(samples))
                    for key, samples in sorted(self.queue_latencies_by_class.items())
                )
                if average is not None
            },
        }


metrics = BatchMetrics(path="batch")
streaming_metrics = BatchMetrics(path="streaming")


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
    snapshot_value = metrics.snapshot()["average_token_throughput_per_sec"]
    # `snapshot()` returns a heterogeneous mapping (see its docstring); this
    # particular key is always a float or None.
    token_throughput_per_sec = snapshot_value if isinstance(snapshot_value, float) else None
    return {
        "queue_latency_ms": queue_latency_ms,
        "token_throughput_per_sec": token_throughput_per_sec,
    }
