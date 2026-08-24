"""Declarative test scenarios: a YAML/JSON file in, `Scenario` objects out.

The point of this module is that adding a benchmark or a behavioral check
should not require writing Python. `tests/scenarios.yaml` holds a `defaults:`
block (the run configuration -- model, host, generation parameters) and a
`scenarios:` list, and `tests/conftest.py` expands that list into one pytest
case per entry.

One loader handles both formats: JSON is a subset of YAML, so `yaml.safe_load`
parses a `.json` scenario file with no format branching. `pyyaml` is already a
dependency of the client distribution.

Validation is strict and fails loudly on unknown keys. A silently ignored typo
in a threshold name is worse than having no threshold at all -- the suite would
report green while measuring nothing.
"""
from __future__ import annotations

import os
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Mapping, Optional, Sequence, Tuple

import yaml

#: Env var pointing at an alternate scenario file, so a different set can be
#: run without editing the repo (`EPHEMERIS_TEST_SCENARIOS=/tmp/mine.yaml`).
SCENARIO_PATH_ENV = "EPHEMERIS_TEST_SCENARIOS"

DEFAULT_SCENARIO_PATH = Path(__file__).resolve().parent / "scenarios.yaml"

#: Keys accepted in the top-level `defaults:` block. Every one of these is also
#: overridable per scenario.
DEFAULT_KEYS = frozenset({
    "model",
    "base_url",
    "max_tokens",
    "temperature",
    "timeout_seconds",
    "iterations",
    "concurrency",
})

#: Keys accepted on a scenario, on top of `DEFAULT_KEYS`.
SCENARIO_ONLY_KEYS = frozenset({
    "name",
    "prompt",
    "prompts",
    "stop",
    "tags",
    "thresholds",
    "expect",
    "enforce",
    "enforce_on_ci",
})

SCENARIO_KEYS = DEFAULT_KEYS | SCENARIO_ONLY_KEYS

#: Threshold names, matching `LatencyMetrics`' fields in `tests/test_latency.py`.
THRESHOLD_KEYS = frozenset({
    "min_ms", "max_ms", "mean_ms", "median_ms", "stdev_ms", "p95_ms", "p99_ms",
})

#: Behavioral expectations. Unlike thresholds these are deterministic, so they
#: are always asserted -- CI included.
EXPECT_KEYS = frozenset({"contains", "not_contains"})

BUILTIN_DEFAULTS: Dict[str, Any] = {
    "model": "TinyLlama/TinyLlama-1.1B-Chat-v1.0",
    "base_url": "http://127.0.0.1:8000",
    "max_tokens": 64,
    "temperature": 0.7,
    "timeout_seconds": 30.0,
    "iterations": 20,
    "concurrency": 1,
}


class ScenarioError(ValueError):
    """A scenario file is malformed. Raised at load time, never swallowed."""


@dataclass(frozen=True)
class Scenario:
    """One resolved scenario: the file's entry with `defaults:` merged in."""

    name: str
    prompts: Tuple[str, ...]
    model: str
    base_url: str
    max_tokens: int
    temperature: float
    timeout_seconds: float
    iterations: int
    concurrency: int
    stop: Tuple[str, ...] = ()
    tags: Tuple[str, ...] = ()
    thresholds: Mapping[str, float] = field(default_factory=dict)
    expect: Mapping[str, str] = field(default_factory=dict)
    enforce: bool = False
    enforce_on_ci: bool = False

    @property
    def is_measure_only(self) -> bool:
        """True when nothing about this scenario can fail on timing alone."""
        return not self.thresholds

    def should_enforce_thresholds(self) -> bool:
        """Whether this run should *assert* on the thresholds it records.

        Measurements are always recorded. Assertions are opt-in, because CI
        runs on shared runners with no GPU: wall-clock thresholds there will
        flake, and a flaky red build trains people to ignore the suite.
        """
        if not self.thresholds or not self.enforce:
            return False
        if is_ci():
            return self.enforce_on_ci
        return True


def is_ci() -> bool:
    """GitHub Actions, GitLab, CircleCI and friends all set `CI`."""
    return bool(os.environ.get("CI"))


def scenario_path() -> Path:
    """The scenario file to load: `$EPHEMERIS_TEST_SCENARIOS`, else the default."""
    override = os.environ.get(SCENARIO_PATH_ENV)
    return Path(override).expanduser() if override else DEFAULT_SCENARIO_PATH


def load_scenarios(path: Optional[Path] = None) -> List[Scenario]:
    """Parse and validate a scenario file. Raises `ScenarioError` on any problem."""
    path = Path(path) if path is not None else scenario_path()
    if not path.is_file():
        raise ScenarioError(f"Scenario file not found: {path}")

    try:
        document = yaml.safe_load(path.read_text(encoding="utf-8"))
    except yaml.YAMLError as exc:
        raise ScenarioError(f"{path}: could not be parsed as YAML/JSON: {exc}") from exc

    if document is None:
        raise ScenarioError(f"{path}: file is empty")
    if not isinstance(document, dict):
        raise ScenarioError(f"{path}: top level must be a mapping, got {type(document).__name__}")

    unknown_top = set(document) - {"defaults", "scenarios"}
    if unknown_top:
        raise ScenarioError(
            f"{path}: unknown top-level key(s) {sorted(unknown_top)}; expected 'defaults' and 'scenarios'"
        )

    defaults = document.get("defaults") or {}
    if not isinstance(defaults, dict):
        raise ScenarioError(f"{path}: 'defaults' must be a mapping")
    unknown_defaults = set(defaults) - DEFAULT_KEYS
    if unknown_defaults:
        raise ScenarioError(
            f"{path}: unknown key(s) in 'defaults': {sorted(unknown_defaults)}; "
            f"allowed: {sorted(DEFAULT_KEYS)}"
        )

    merged_defaults = {**BUILTIN_DEFAULTS, **defaults}

    entries = document.get("scenarios") or []
    if not isinstance(entries, list):
        raise ScenarioError(f"{path}: 'scenarios' must be a list")

    scenarios = [_build_scenario(entry, merged_defaults, path, index) for index, entry in enumerate(entries)]

    seen: set[str] = set()
    for scenario in scenarios:
        if scenario.name in seen:
            # Duplicate names would collide as pytest ids and silently shadow
            # each other in the report.
            raise ScenarioError(f"{path}: duplicate scenario name {scenario.name!r}")
        seen.add(scenario.name)

    return scenarios


def _build_scenario(entry: Any, defaults: Mapping[str, Any], path: Path, index: int) -> Scenario:
    where = f"{path}: scenarios[{index}]"
    if not isinstance(entry, dict):
        raise ScenarioError(f"{where}: must be a mapping, got {type(entry).__name__}")

    name = entry.get("name")
    if not isinstance(name, str) or not name.strip():
        raise ScenarioError(f"{where}: 'name' is required and must be a non-empty string")
    where = f"{path}: scenario {name!r}"

    unknown = set(entry) - SCENARIO_KEYS
    if unknown:
        raise ScenarioError(
            f"{where}: unknown key(s) {sorted(unknown)}; allowed: {sorted(SCENARIO_KEYS)}"
        )

    if "prompt" in entry and "prompts" in entry:
        raise ScenarioError(f"{where}: set either 'prompt' or 'prompts', not both")
    if "prompt" in entry:
        prompt = entry["prompt"]
        if not isinstance(prompt, str) or not prompt:
            raise ScenarioError(f"{where}: 'prompt' must be a non-empty string")
        prompts: Tuple[str, ...] = (prompt,)
    elif "prompts" in entry:
        prompts = tuple(_string_list(entry["prompts"], f"{where}: 'prompts'"))
        if not prompts:
            raise ScenarioError(f"{where}: 'prompts' must contain at least one entry")
    else:
        raise ScenarioError(f"{where}: one of 'prompt' or 'prompts' is required")

    def resolved(key: str) -> Any:
        return entry[key] if key in entry else defaults[key]

    concurrency = _positive_int(resolved("concurrency"), f"{where}: 'concurrency'")
    iterations = _positive_int(resolved("iterations"), f"{where}: 'iterations'")
    max_tokens = _positive_int(resolved("max_tokens"), f"{where}: 'max_tokens'")

    return Scenario(
        name=name,
        prompts=prompts,
        model=_string(resolved("model"), f"{where}: 'model'"),
        base_url=_string(resolved("base_url"), f"{where}: 'base_url'"),
        max_tokens=max_tokens,
        temperature=_number(resolved("temperature"), f"{where}: 'temperature'"),
        timeout_seconds=_number(resolved("timeout_seconds"), f"{where}: 'timeout_seconds'"),
        iterations=iterations,
        concurrency=concurrency,
        stop=tuple(_string_list(entry.get("stop", []), f"{where}: 'stop'")),
        tags=tuple(_string_list(entry.get("tags", []), f"{where}: 'tags'")),
        thresholds=_mapping_of_numbers(entry.get("thresholds", {}), THRESHOLD_KEYS, f"{where}: 'thresholds'"),
        expect=_mapping_of_strings(entry.get("expect", {}), EXPECT_KEYS, f"{where}: 'expect'"),
        enforce=_boolean(entry.get("enforce", False), f"{where}: 'enforce'"),
        enforce_on_ci=_boolean(entry.get("enforce_on_ci", False), f"{where}: 'enforce_on_ci'"),
    )


def _string(value: Any, where: str) -> str:
    if not isinstance(value, str) or not value:
        raise ScenarioError(f"{where}: must be a non-empty string, got {value!r}")
    return value


def _string_list(value: Any, where: str) -> List[str]:
    if not isinstance(value, list):
        raise ScenarioError(f"{where}: must be a list, got {type(value).__name__}")
    for item in value:
        if not isinstance(item, str):
            raise ScenarioError(f"{where}: entries must be strings, got {item!r}")
    return list(value)


def _number(value: Any, where: str) -> float:
    # `bool` is an `int` subclass; a stray `true` where a number belongs is a
    # mistake worth reporting rather than reading as 1.
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise ScenarioError(f"{where}: must be a number, got {value!r}")
    return float(value)


def _positive_int(value: Any, where: str) -> int:
    if isinstance(value, bool) or not isinstance(value, int) or value <= 0:
        raise ScenarioError(f"{where}: must be a positive integer, got {value!r}")
    return value


def _boolean(value: Any, where: str) -> bool:
    if not isinstance(value, bool):
        raise ScenarioError(f"{where}: must be true or false, got {value!r}")
    return value


def _mapping_of_numbers(value: Any, allowed: frozenset, where: str) -> Dict[str, float]:
    if not isinstance(value, dict):
        raise ScenarioError(f"{where}: must be a mapping, got {type(value).__name__}")
    unknown = set(value) - allowed
    if unknown:
        raise ScenarioError(f"{where}: unknown key(s) {sorted(unknown)}; allowed: {sorted(allowed)}")
    return {key: _number(item, f"{where}.{key}") for key, item in value.items()}


def _mapping_of_strings(value: Any, allowed: frozenset, where: str) -> Dict[str, str]:
    if not isinstance(value, dict):
        raise ScenarioError(f"{where}: must be a mapping, got {type(value).__name__}")
    unknown = set(value) - allowed
    if unknown:
        raise ScenarioError(f"{where}: unknown key(s) {sorted(unknown)}; allowed: {sorted(allowed)}")
    return {key: _string(item, f"{where}.{key}") for key, item in value.items()}


def tags_of(scenarios: Sequence[Scenario]) -> List[str]:
    """Every distinct tag across `scenarios`, sorted. Used for marker mapping."""
    return sorted({tag for scenario in scenarios for tag in scenario.tags})
