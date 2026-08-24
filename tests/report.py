"""Markdown report for the declarative scenario suite.

The old latency benchmarks logged their numbers and threw them away: nothing
machine-readable, nothing persisted, nothing diffable between runs. This module
collects one record per scenario and renders them as markdown, so a run's
results can be attached to a PR or checked against last week's.

The report is written from `pytest_sessionfinish` (see `tests/conftest.py`),
not at test teardown, so an interrupted or crashed run still emits whatever was
collected before it died.
"""
from __future__ import annotations

import os
import platform
import subprocess
import time
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Dict, List, Mapping, Optional, Sequence, Tuple

#: Columns in the results table, in order, mapped to their `LatencyMetrics`
#: field. `median_ms` is p50 -- named for the statistic, displayed as the
#: percentile it is.
_METRIC_COLUMNS: Tuple[Tuple[str, str], ...] = (
    ("p50 (ms)", "median_ms"),
    ("p95 (ms)", "p95_ms"),
    ("p99 (ms)", "p99_ms"),
)

STATUS_PASSED = "passed"
STATUS_FAILED = "failed"
STATUS_MEASURE_ONLY = "measure-only"
STATUS_SKIPPED = "skipped"


@dataclass
class ScenarioResult:
    """One scenario's outcome, as it will appear in the report."""

    name: str
    status: str
    tags: Tuple[str, ...] = ()
    metrics: Optional[Mapping[str, float]] = None
    thresholds: Mapping[str, float] = field(default_factory=dict)
    enforced: bool = False
    failures: List[str] = field(default_factory=list)
    note: str = ""


@dataclass
class RunContext:
    """What the run was, for the report header."""

    model: str = "-"
    device: str = "-"
    scenario_file: str = "-"
    started_at: float = field(default_factory=time.time)


class ReportCollector:
    """Accumulates `ScenarioResult`s for the life of a pytest session."""

    def __init__(self) -> None:
        self.context = RunContext()
        self._results: Dict[str, ScenarioResult] = {}

    def reset(self) -> None:
        self.context = RunContext()
        self._results = {}

    def record(self, result: ScenarioResult) -> None:
        # Keyed by name so a re-run of the same scenario within one session
        # (`--lf`, a rerun plugin) replaces rather than duplicates its row.
        self._results[result.name] = result

    @property
    def results(self) -> List[ScenarioResult]:
        return list(self._results.values())

    def counts(self) -> Dict[str, int]:
        tally: Dict[str, int] = {}
        for result in self._results.values():
            tally[result.status] = tally.get(result.status, 0) + 1
        return tally


#: Session-wide singleton. `pytest_sessionfinish` needs to reach the collected
#: results from outside any fixture's scope.
collector = ReportCollector()


def git_sha() -> str:
    """Short HEAD sha, or `unknown` outside a git checkout."""
    try:
        completed = subprocess.run(
            ["git", "rev-parse", "--short", "HEAD"],
            capture_output=True,
            text=True,
            timeout=5,
            check=False,
        )
    except (OSError, subprocess.SubprocessError):
        return "unknown"
    return completed.stdout.strip() or "unknown"


def detect_device() -> str:
    """The device the server would pick, without importing the engine."""
    try:
        from settings.settings import model_settings, resolve_device

        return resolve_device(model_settings.device)
    except Exception:  # pragma: no cover - settings unavailable
        return "unknown"


def _format_ms(value: Optional[float]) -> str:
    return "-" if value is None else f"{value:.2f}"


def _threshold_cell(result: ScenarioResult) -> str:
    if not result.thresholds:
        return "-"
    suffix = "" if result.enforced else " *(recorded only)*"
    body = "<br>".join(
        f"{key} ≤ {limit:.2f}" for key, limit in sorted(result.thresholds.items())
    )
    return body + suffix


def _margin_cell(result: ScenarioResult) -> str:
    if not result.thresholds or not result.metrics:
        return "-"
    parts = []
    for key, limit in sorted(result.thresholds.items()):
        actual = result.metrics.get(key)
        if actual is None:
            parts.append(f"{key}: -")
            continue
        # Positive margin = headroom left under the threshold.
        parts.append(f"{limit - actual:+.2f}")
    return "<br>".join(parts)


def render_markdown(report: ReportCollector, *, finished_at: Optional[float] = None) -> str:
    """Render `report` as a standalone markdown document."""
    finished_at = time.time() if finished_at is None else finished_at
    results = sorted(report.results, key=lambda item: item.name)
    tally = report.counts()
    summary = ", ".join(f"{count} {status}" for status, count in sorted(tally.items())) or "no scenarios ran"

    lines: List[str] = ["# Ephemeris scenario report", ""]

    lines += [
        "| | |",
        "| --- | --- |",
        f"| Generated | {datetime.now(timezone.utc).strftime('%Y-%m-%d %H:%M:%S UTC')} |",
        f"| Scenario file | `{report.context.scenario_file}` |",
        f"| Model | `{report.context.model}` |",
        f"| Device | `{report.context.device}` |",
        f"| Platform | {platform.platform()} |",
        f"| Git SHA | `{git_sha()}` |",
        f"| CI | {'yes' if os.environ.get('CI') else 'no'} |",
        f"| Duration | {finished_at - report.context.started_at:.1f}s |",
        f"| Result | {summary} |",
        "",
    ]

    if not results:
        lines += [
            "No scenarios were collected. If this is unexpected, check the",
            "`scenarios:` list in the scenario file named above.",
            "",
        ]
        return "\n".join(lines)

    header = ["Scenario", "Status", *[label for label, _ in _METRIC_COLUMNS], "Threshold", "Margin", "Tags"]
    lines += [
        "## Scenarios",
        "",
        "| " + " | ".join(header) + " |",
        "| " + " | ".join("---" for _ in header) + " |",
    ]
    for result in results:
        cells = [
            result.name,
            result.status,
            *[
                _format_ms(result.metrics.get(field_name) if result.metrics else None)
                for _, field_name in _METRIC_COLUMNS
            ],
            _threshold_cell(result),
            _margin_cell(result),
            ", ".join(result.tags) or "-",
        ]
        lines.append("| " + " | ".join(cells) + " |")
    lines.append("")

    lines += [
        "Margin is `threshold - measured`: positive is headroom, negative is an",
        "overrun. A threshold marked *(recorded only)* was measured but not",
        "asserted -- see the `enforce` / `enforce_on_ci` keys in TESTING.md.",
        "",
    ]

    failed = [result for result in results if result.failures]
    if failed:
        lines += ["## Failures", ""]
        for result in failed:
            lines.append(f"### {result.name}")
            lines.append("")
            for message in result.failures:
                lines.append(f"- {message}")
            lines.append("")

    notes = [result for result in results if result.note]
    if notes:
        lines += ["## Notes", ""]
        for result in notes:
            lines.append(f"- **{result.name}**: {result.note}")
        lines.append("")

    return "\n".join(lines)


def write_report(path: Path, report: ReportCollector) -> Path:
    """Write the rendered report to `path`, creating parent directories."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(render_markdown(report), encoding="utf-8")
    return path


def metrics_as_dict(metrics: object) -> Dict[str, float]:
    """Flatten a `LatencyMetrics` dataclass into plain floats.

    Kept here rather than importing `LatencyMetrics` from `tests/test_latency.py`
    -- the test module imports this one, and the report has no reason to depend
    on the benchmark module in return.
    """
    from dataclasses import asdict, is_dataclass

    if is_dataclass(metrics) and not isinstance(metrics, type):
        return {key: float(value) for key, value in asdict(metrics).items()}
    if isinstance(metrics, Mapping):
        return {str(key): float(value) for key, value in metrics.items()}
    raise TypeError(f"Cannot read metrics from {type(metrics).__name__}")


def status_for(failures: Sequence[str], *, asserted: bool) -> str:
    """Pick a row status: failures win, then whether anything was asserted."""
    if failures:
        return STATUS_FAILED
    return STATUS_PASSED if asserted else STATUS_MEASURE_ONLY
