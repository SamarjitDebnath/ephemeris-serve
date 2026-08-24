"""Latency and performance benchmarks for the inference server.

The prompt-driven benchmarks here are declarative: cases live in
`tests/scenarios.yaml` (see TESTING.md), `tests/conftest.py` expands them into
one pytest case each, and `tests/report.py` renders the measurements as
markdown. Adding a benchmark does not require editing this file.

The tests that remain hand-written are the ones that are not a
prompt-plus-threshold shape -- queue latency, degradation over time, and
memory growth -- where the assertion is about a trend rather than a number.
"""
import asyncio
import time
import statistics
import pytest
from typing import List, Sequence
from dataclasses import dataclass
from unittest.mock import Mock

from logger import setup_logger
from tests import report as report_module
from tests.scenarios import Scenario

logger = setup_logger(__name__, level="INFO")


@dataclass
class LatencyMetrics:
    """Container for latency metrics"""
    min_ms: float
    max_ms: float
    mean_ms: float
    median_ms: float
    stdev_ms: float
    p95_ms: float
    p99_ms: float


def measure_latency(func, *args, **kwargs) -> float:
    """Measure execution time in milliseconds"""
    start = time.perf_counter()
    result = func(*args, **kwargs)
    end = time.perf_counter()
    return (end - start) * 1000  # Convert to ms


async def measure_latency_async(coro) -> float:
    """Measure async execution time in milliseconds"""
    start = time.perf_counter()
    await coro
    end = time.perf_counter()
    return (end - start) * 1000  # Convert to ms


def analyze_latencies(latencies: List[float]) -> LatencyMetrics:
    """Analyze a list of latency measurements"""
    latencies_sorted = sorted(latencies)
    n = len(latencies_sorted)
    p95_idx = int(n * 0.95)
    p99_idx = int(n * 0.99)

    return LatencyMetrics(
        min_ms=min(latencies),
        max_ms=max(latencies),
        mean_ms=statistics.mean(latencies),
        median_ms=statistics.median(latencies),
        stdev_ms=statistics.stdev(latencies) if n > 1 else 0.0,
        p95_ms=latencies_sorted[p95_idx],
        p99_ms=latencies_sorted[p99_idx],
    )


def print_latency_report(name: str, metrics: LatencyMetrics):
    """Log a formatted latency report"""
    logger.info("%s", "\n" + "=" * 60)
    logger.info("Latency Report: %s", name)
    logger.info("%s", "=" * 60)
    logger.info("  Min:     %.2f ms", metrics.min_ms)
    logger.info("  Max:     %.2f ms", metrics.max_ms)
    logger.info("  Mean:    %.2f ms", metrics.mean_ms)
    logger.info("  Median:  %.2f ms", metrics.median_ms)
    logger.info("  StdDev:  %.2f ms", metrics.stdev_ms)
    logger.info("  P95:     %.2f ms", metrics.p95_ms)
    logger.info("  P99:     %.2f ms", metrics.p99_ms)
    logger.info("%s", "=" * 60)


class TestLatencyBenchmarks:
    """Benchmark tests for inference latency.

    The prompt-plus-threshold benchmarks that used to live here are now
    scenarios (`tests/scenarios.yaml`, run by `TestScenarios` below). What
    remains is the measurements that are about a trend rather than a number.
    """

    # Thin aliases: these were methods before the helpers were lifted to module
    # scope for `TestScenarios` to share, and the math behind them is unchanged.
    measure_latency = staticmethod(measure_latency)
    measure_latency_async = staticmethod(measure_latency_async)
    analyze_latencies = staticmethod(analyze_latencies)
    print_latency_report = staticmethod(print_latency_report)

    @pytest.mark.asyncio
    async def test_batch_scheduler_queue_latency(self):
        try:
            from scheduler.batch_scheduler import BatchScheduler
            from scheduler.request import InferenceRequest
            from scheduler.request_queue import batch_request_queue
            from metrics.metrics import metrics
        except ImportError:
            pytest.skip("Batch scheduler module not available")

        metrics.queue_latencies.clear()
        # Drain the queue before using it for this test.
        while not batch_request_queue.queue.empty():
            batch_request_queue.queue.get_nowait()

        first = InferenceRequest(prompt="latency prompt 1", max_tokens=1, temperature=0.5)
        first.enqueue_time = time.monotonic() - 0.08
        second = InferenceRequest(prompt="latency prompt 2", max_tokens=1, temperature=0.5)
        second.enqueue_time = time.monotonic() - 0.04

        await batch_request_queue.put(first)
        await batch_request_queue.put(second)

        scheduler = BatchScheduler(Mock(), Mock(), max_batch_size=2, queue_timeout=0.5)
        batch = await scheduler._collect_batch()

        assert len(batch) == 2
        assert len(metrics.queue_latencies) >= 2
        assert all(latency >= 0 for latency in metrics.queue_latencies)
        assert batch[0] is first
        assert batch[1] is second

    def test_repeated_inference_stability(self, test_prompt):
        """
        Test latency stability over repeated calls.
        
        Ensures that latency doesn't degrade with repeated usage.
        """
        try:
            from tokenizer.tokenizer_service import tokenizer_service
        except ImportError:
            pytest.skip("Tokenizer service not available")

        # Warm up the tokenizer to avoid measuring any one-time initialization costs.
        for _ in range(5):
            tokenizer_service.encode(test_prompt)

        latencies = []
        num_runs = 100

        for _ in range(num_runs):
            latency = self.measure_latency(
                tokenizer_service.encode,
                test_prompt
            )
            latencies.append(latency)

        # Split into early and late runs to check for degradation
        early = latencies[:25]
        late = latencies[-25:]

        early_metrics = self.analyze_latencies(early)
        late_metrics = self.analyze_latencies(late)

        logger.info("Latency Stability Test (100 runs)")
        logger.info("  Early (runs 1-25):  mean=%.2fms", early_metrics.mean_ms)
        logger.info("  Late (runs 76-100): mean=%.2fms", late_metrics.mean_ms)
        logger.info("  Degradation:        %.2fms", late_metrics.mean_ms - early_metrics.mean_ms)

        # Assertion: latency should not degrade by more than 20%, with a sane lower bound
        # to account for sub-millisecond scheduling jitter on modern systems.
        degradation = late_metrics.mean_ms - early_metrics.mean_ms
        max_degradation = max(early_metrics.mean_ms * 0.2, 0.5)
        epsilon = 0.001  # Small tolerance for floating point comparison
        assert degradation <= max_degradation + epsilon, \
            f"Latency degradation too high: {degradation:.2f}ms (expected <= {max_degradation:.2f}ms)"


class TestScenarios:
    """One test per entry in the scenario file -- see `tests/scenarios.yaml`.

    `tests/conftest.py` parametrizes the `scenario` fixture, so the ids here
    are the scenario names: a failure reads `test_scenario[batch_throughput]`
    and points straight at the YAML that produced it.

    Two shapes of scenario, distinguished by which block the entry carries:

    - `expect:` -- a behavioral check. Deterministic, so it always asserts.
    - otherwise -- a latency measurement. Always recorded in the report;
      asserted only when the scenario opts in (see
      `Scenario.should_enforce_thresholds`), because wall-clock thresholds on
      a shared CI runner flake, and a flaky red build trains people to ignore
      the suite.
    """

    def test_scenario(self, scenario: Scenario, scenario_report):
        if scenario.expect:
            self._run_behavior(scenario, scenario_report)
        else:
            self._run_latency(scenario, scenario_report)

    # ------------------------------------------------------------------ #

    def _run_behavior(self, scenario: Scenario, scenario_report) -> None:
        """Assert a scenario's `expect:` block against stop-sequence truncation.

        This exercises `utils.stop_sequences.find_stop_index`, the same
        function the server uses both to halt generation and to trim the stop
        text (and everything after it) off the response -- so the check is real
        without needing a live model.
        """
        from utils.stop_sequences import find_stop_index

        text = scenario.prompts[0]
        stop_index = find_stop_index(text, scenario.stop)
        result_text = text if stop_index is None else text[:stop_index]

        failures: List[str] = []
        needle = scenario.expect.get("contains")
        if needle is not None and needle not in result_text:
            failures.append(f"expected {needle!r} in output, got {result_text!r}")
        forbidden = scenario.expect.get("not_contains")
        if forbidden is not None and forbidden in result_text:
            failures.append(f"expected {forbidden!r} to be absent, got {result_text!r}")

        scenario_report.record(
            report_module.ScenarioResult(
                name=scenario.name,
                status=report_module.status_for(failures, asserted=True),
                tags=scenario.tags,
                failures=failures,
                note=f"stop={list(scenario.stop)} -> {result_text!r}",
            )
        )
        assert not failures, "; ".join(failures)

    def _run_latency(self, scenario: Scenario, scenario_report) -> None:
        try:
            from tokenizer.tokenizer_service import tokenizer_service
        except ImportError:
            scenario_report.record(
                report_module.ScenarioResult(
                    name=scenario.name,
                    status=report_module.STATUS_SKIPPED,
                    tags=scenario.tags,
                    thresholds=scenario.thresholds,
                    note="tokenizer service not available",
                )
            )
            pytest.skip("Tokenizer service not available")

        # Warm up so one-time initialization is not folded into the numbers.
        tokenizer_service.encode(scenario.prompts[0])

        latencies = [
            self._measure_one_pass(scenario, tokenizer_service)
            for _ in range(scenario.iterations)
        ]
        metrics = analyze_latencies(latencies)
        print_latency_report(scenario.name, metrics)

        enforced = scenario.should_enforce_thresholds()
        failures: List[str] = []
        if enforced:
            measured = report_module.metrics_as_dict(metrics)
            for key, limit in sorted(scenario.thresholds.items()):
                actual = measured[key]
                if actual > limit:
                    failures.append(f"{key} was {actual:.2f}ms, threshold {limit:.2f}ms")

        note = ""
        if scenario.thresholds and not enforced:
            note = "thresholds recorded but not asserted on this run"
        elif scenario.is_measure_only:
            note = "no thresholds declared -- measure only"

        scenario_report.record(
            report_module.ScenarioResult(
                name=scenario.name,
                status=report_module.status_for(failures, asserted=enforced),
                tags=scenario.tags,
                metrics=report_module.metrics_as_dict(metrics),
                thresholds=scenario.thresholds,
                enforced=enforced,
                failures=failures,
                note=note,
            )
        )
        assert not failures, "; ".join(failures)

    def _measure_one_pass(self, scenario: Scenario, tokenizer_service) -> float:
        """Time one pass over the scenario's prompts.

        `concurrency: 1` runs them in sequence; anything higher fans them out
        over threads, which is what the old `test_concurrent_request_latency`
        measured.
        """
        if scenario.concurrency <= 1:
            return measure_latency(self._encode_all, tokenizer_service, scenario.prompts)
        return asyncio.run(
            measure_latency_async(self._encode_all_concurrently(tokenizer_service, scenario))
        )

    @staticmethod
    def _encode_all(tokenizer_service, prompts: Sequence[str]) -> None:
        for prompt in prompts:
            tokenizer_service.encode(prompt)

    @staticmethod
    async def _encode_all_concurrently(tokenizer_service, scenario: Scenario) -> None:
        semaphore = asyncio.Semaphore(scenario.concurrency)

        async def encode(prompt: str) -> None:
            async with semaphore:
                await asyncio.to_thread(tokenizer_service.encode, prompt)

        await asyncio.gather(*(encode(prompt) for prompt in scenario.prompts))


class TestLoadPatterns:
    """Test various load patterns and stress scenarios"""

    def test_memory_under_load(self, test_prompt):
        """
        Test memory usage under load.
        
        Runs multiple sequential tokenizations to check for memory leaks.
        """
        try:
            from tokenizer.tokenizer_service import tokenizer_service
            import psutil
            import os
        except ImportError:
            pytest.skip("Required modules not available")

        process = psutil.Process(os.getpid())
        initial_memory = process.memory_info().rss / 1024 / 1024  # MB

        # Run many iterations
        for _ in range(500):
            tokenizer_service.encode(test_prompt)

        final_memory = process.memory_info().rss / 1024 / 1024  # MB
        memory_increase = final_memory - initial_memory

        logger.info("Memory Test (500 iterations)")
        logger.info("  Initial Memory: %.2f MB", initial_memory)
        logger.info("  Final Memory:   %.2f MB", final_memory)
        logger.info("  Increase:       %.2f MB", memory_increase)

        # Assertion: memory increase should be < 100MB
        assert memory_increase < 100.0, \
            f"Memory increased too much: {memory_increase:.2f}MB (expected < 100MB)"
