"""Prometheus metric definitions, mirroring `metrics/metrics.py`.

The instinct when asked to persist metrics is to write `snapshot()` to disk on
a timer. That is the wrong direction. `BatchMetrics` computes *averages over
the last 1000 samples*, so the window it covers depends on traffic -- at 10
req/s it spans 100 seconds, at 1000 req/s it spans one. Persisting a
pre-averaged number over a variable window throws away exactly the freedom a
time-series database exists to provide.

So this module emits raw primitives -- counters, gauges, histograms -- and lets
the scrape do the aggregating. It sits *alongside* the deques rather than
deriving from them: `GET /api/metrics` and its JSON stay exactly as they were,
because the CLI consumes them.

`prometheus_client` is an optional dependency. Everything here degrades to a
no-op when it is missing, so the server runs unchanged without it.

**Multi-process.** Under `--workers N` each worker has its own metrics.
`prometheus_client` handles this with `PROMETHEUS_MULTIPROC_DIR`: each process
writes mmap files and the exporter aggregates them at scrape time. Counters
aggregate correctly on their own; **gauges do not** -- each needs an explicit
mode, and picking the wrong one silently produces plausible, incorrect numbers.
The modes chosen here are documented per gauge below.

One documented footgun: `PROMETHEUS_MULTIPROC_DIR` must be emptied at startup.
Files left by a previous run are otherwise picked up and inflate counters.
"""
import os
import shutil
from typing import Optional

from logger import setup_logger
from settings.settings import logging_settings

logger = setup_logger(__name__, level=logging_settings.log_level, log_file=logging_settings.log_file)

try:  # pragma: no cover - optional dependency
    from prometheus_client import CollectorRegistry, Counter, Gauge, Histogram, generate_latest
    from prometheus_client import multiprocess as _multiprocess
    from prometheus_client import CONTENT_TYPE_LATEST

    _AVAILABLE = True
except ImportError:  # pragma: no cover - extra not installed
    CollectorRegistry = None
    Counter = Gauge = Histogram = None
    generate_latest = None
    _multiprocess = None
    CONTENT_TYPE_LATEST = "text/plain; version=0.0.4; charset=utf-8"
    _AVAILABLE = False


MULTIPROC_ENV = "PROMETHEUS_MULTIPROC_DIR"

#: Buckets in seconds. Chosen to straddle the range actually observed -- queue
#: latency is sub-millisecond when idle and seconds deep under load. A
#: histogram whose buckets all saturate is worse than no histogram, because it
#: still looks like data.
_LATENCY_BUCKETS = (0.001, 0.005, 0.01, 0.05, 0.1, 0.5, 1.0, 5.0, 10.0, 30.0, 60.0)
_THROUGHPUT_BUCKETS = (1.0, 5.0, 10.0, 25.0, 50.0, 100.0, 250.0, 500.0, 1000.0)


class _NullMetric:
    """Stand-in used when `prometheus_client` is absent."""

    def labels(self, *args, **kwargs):
        return self

    def inc(self, amount: float = 1.0) -> None:
        pass

    def observe(self, amount: float) -> None:
        pass

    def set(self, value: float) -> None:
        pass


class PrometheusMetrics:
    """The metric objects, or null stand-ins when the extra is not installed."""

    def __init__(self) -> None:
        self.available = _AVAILABLE
        if not _AVAILABLE or Histogram is None or Gauge is None or Counter is None:
            # The `or ... is None` arms are redundant at runtime (`_AVAILABLE`
            # already implies them) but they are what narrows the optional
            # imports for a type checker below.
            self.queue_latency = _NullMetric()
            self.token_throughput = _NullMetric()
            self.batch_size = _NullMetric()
            self.batch_occupancy = _NullMetric()
            self.cache_utilization = _NullMetric()
            self.timeout_evictions = _NullMetric()
            self.cancelled_evictions = _NullMetric()
            self.kv_blocks_reclaimed = _NullMetric()
            return

        # `path` labels which of the two `BatchMetrics` singletons a sample
        # came from ("batch" or "streaming"), which is the distinction the JSON
        # endpoint encodes as two top-level keys.
        self.queue_latency = Histogram(
            "ephemeris_queue_latency_seconds",
            "Time a request waited in the queue before admission.",
            ["path"],
            buckets=_LATENCY_BUCKETS,
        )
        self.token_throughput = Histogram(
            "ephemeris_token_throughput_per_second",
            "Tokens per second observed per scheduler step.",
            ["path"],
            buckets=_THROUGHPUT_BUCKETS,
        )
        # `multiprocess_mode="livesum"`: these are per-worker instantaneous
        # values, and the pool-wide quantity that means anything is their sum
        # across *live* workers. "all" would keep counting dead ones.
        self.batch_size = Gauge(
            "ephemeris_batch_size",
            "Requests in the most recent batch.",
            ["path"],
            multiprocess_mode="livesum",
        )
        # ...whereas occupancy and utilization are ratios. Summing ratios is
        # meaningless, so take the max across live workers: it answers "is any
        # worker saturated", which is the question worth alerting on.
        self.batch_occupancy = Gauge(
            "ephemeris_batch_occupancy_ratio",
            "Active requests as a fraction of max_batch_size.",
            ["path"],
            multiprocess_mode="livemax",
        )
        self.cache_utilization = Gauge(
            "ephemeris_kv_cache_utilization_ratio",
            "Used KV blocks as a fraction of pool capacity.",
            ["path"],
            multiprocess_mode="livemax",
        )
        self.timeout_evictions = Counter(
            "ephemeris_timeout_evictions_total",
            "Requests evicted for exceeding their deadline.",
            ["path"],
        )
        self.cancelled_evictions = Counter(
            "ephemeris_cancelled_evictions_total",
            "Requests evicted because the client disconnected.",
            ["path"],
        )
        self.kv_blocks_reclaimed = Counter(
            "ephemeris_kv_blocks_reclaimed_total",
            "KV cache blocks returned to the allocator by an idle trim.",
            ["path"],
        )


prometheus_metrics = PrometheusMetrics()


def prepare_multiprocess_dir() -> Optional[str]:
    """Empty and return `PROMETHEUS_MULTIPROC_DIR`, if it is set.

    Called once at startup. Files left behind by a previous run are counted by
    the exporter and inflate every counter -- a documented `prometheus_client`
    footgun that produces wrong numbers rather than an error.
    """
    directory = os.environ.get(MULTIPROC_ENV)
    if not directory:
        return None
    try:
        shutil.rmtree(directory, ignore_errors=True)
        os.makedirs(directory, exist_ok=True)
    except OSError as exc:
        logger.warning("Could not reset %s=%s: %s", MULTIPROC_ENV, directory, exc)
        return None
    return directory


def render_latest() -> bytes:
    """The exposition-format payload for a scrape.

    In multiprocess mode a fresh registry is built per scrape and populated
    from every worker's mmap files; otherwise the default registry is used.
    """
    if not _AVAILABLE or generate_latest is None:
        return b"# prometheus_client is not installed; install the 'metrics' extra\n"
    if os.environ.get(MULTIPROC_ENV) and CollectorRegistry is not None and _multiprocess is not None:
        registry = CollectorRegistry()
        _multiprocess.MultiProcessCollector(registry)
        return generate_latest(registry)
    return generate_latest()
