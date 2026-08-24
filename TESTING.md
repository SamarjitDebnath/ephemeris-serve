# Testing Guide

## Overview

This project includes a comprehensive test suite with:
- **Unit tests** for core modules
- **Integration tests** for API endpoints and components
- **Latency benchmarks** for performance monitoring
- **Load pattern tests** for stress testing

## Quick Start

### Install Dependencies

```bash
make dev
```

This installs the server, all dev dependencies (pytest, black, isort, pylint, flake8, psutil), and the `ephemeris-cli` chat client — the unit tests import both distributions.

### Run All Tests

```bash
make test-all
```

### Run Specific Test Suites

```bash
# Unit tests only
make test-unit

# Latency benchmarks only
make test-latency

# Integration tests only
pytest tests/test_integration.py -v
```

## Test Structure

```
tests/
├── __init__.py
├── conftest.py           # Pytest configuration, shared fixtures, scenario parametrization
├── scenarios.yaml        # Declarative test cases -- add a benchmark without writing Python
├── scenarios.py          # Scenario loader and strict schema validation
├── report.py             # Markdown report renderer
├── test_unit.py          # Unit tests for core modules
├── test_integration.py   # Integration tests for API and components
└── test_latency.py       # Latency benchmarks and performance tests
```

## Test Categories

### Unit Tests (`test_unit.py`)

Tests for individual components:
- **Scheduler**: Request queue and request structures
- **Tokenizer**: Encoding and decoding functionality
- **API**: Routes and server initialization
- **Settings**: Configuration loading
- **Logger**: Logging setup and methods

Run with:
```bash
make test-unit
```

### Integration Tests (`test_integration.py`)

Tests for component interactions:
- FastAPI app initialization
- Scheduler request handling
- Error resilience

Run with:
```bash
pytest tests/test_integration.py -v
```

### Latency Benchmarks (`test_latency.py`)

Most benchmarks here are **declarative**: the cases live in `tests/scenarios.yaml`,
not in Python. See [Declarative Scenarios](#declarative-scenarios) below.

What stays hand-written in `test_latency.py` is the handful of tests whose
assertion is about a *trend* rather than a number, which does not fit the
scenario schema:

#### **Queue Latency** (`test_batch_scheduler_queue_latency`)
- Asserts the batch scheduler records a queue-latency sample per request

#### **Latency Stability** (`test_repeated_inference_stability`)
- Checks for latency degradation over time
- 100 consecutive runs
- **Target**: < 20% degradation from start to end

#### **Memory Under Load** (`test_memory_under_load`)
- Tracks memory usage over 500 iterations
- **Target**: < 100MB memory increase

Run with:
```bash
make test-latency
```

## Declarative Scenarios

Adding a benchmark or a behavioral check should not mean editing Python. Cases
live in `tests/scenarios.yaml`; `conftest.py` expands the list into one pytest
case per entry, named after its `name:` field — a failure reads as
`test_scenario[short_prompt_latency]` and points straight at the YAML entry
that produced it.

```bash
make test-report                              # run everything, write reports/test_report.md
pytest tests/ --report-md=reports/mine.md     # same, explicit path
pytest tests/test_latency.py -m "not slow"    # tags work as pytest markers
EPHEMERIS_TEST_SCENARIOS=/tmp/other.yaml pytest tests/test_latency.py
```

JSON works everywhere YAML does — JSON is a YAML subset, so the same loader
reads a `.json` scenario file with no extra flag.

### Schema

```yaml
defaults:                 # every key here is also overridable per scenario
  model: TinyLlama/TinyLlama-1.1B-Chat-v1.0
  base_url: http://127.0.0.1:8000
  max_tokens: 64
  temperature: 0.7
  timeout_seconds: 30
  iterations: 50          # measurement passes per scenario
  concurrency: 1          # >1 fans the prompts out over threads

scenarios:
  - name: short_prompt_latency   # required, unique -- becomes the pytest id
    prompt: "What is machine learning?"
    tags: [latency, benchmark]
    enforce: true                # assert the thresholds, don't just record them
    thresholds:
      mean_ms: 20
      p95_ms: 50

  - name: batch_throughput
    prompts:                     # `prompts:` and `prompt:` are mutually exclusive
      - "Explain deep learning in simple terms"
      - "How does a neural network work?"
    iterations: 10
    tags: [latency, benchmark, slow]
    # No `thresholds:` -- measure-only. Recorded in the report, never asserted.

  - name: stop_sequence_honored
    prompt: "Count: 1 2 3 4 5"
    stop: ["4"]
    tags: [behavior]
    expect:
      not_contains: "5"
```

| Key | Meaning |
| --- | --- |
| `name` | Required, unique. Becomes the pytest id and the report row. |
| `prompt` / `prompts` | The input. Exactly one of the two. |
| `tags` | Applied as pytest markers, so `-m "not slow"` selects on them. Any tag is registered automatically — no `pytest.ini` edit needed. |
| `thresholds` | `min_ms`, `max_ms`, `mean_ms`, `median_ms`, `stdev_ms`, `p95_ms`, `p99_ms`. Omit the block for a measure-only scenario. |
| `enforce` | `false` by default. Thresholds are always *recorded*; this makes them *assert*. |
| `enforce_on_ci` | `false` by default. Required on top of `enforce` for the assertion to run when `CI` is set. |
| `expect` | `contains` / `not_contains`, checked against stop-sequence truncation. Deterministic, so always asserted — CI included. |
| `stop` | Stop sequences, used by `expect` scenarios. |
| `iterations`, `concurrency`, `model`, `base_url`, `max_tokens`, `temperature`, `timeout_seconds` | Fall back to `defaults:`. |

Validation is **strict**: an unknown key anywhere in the file aborts the run
with the offending scenario and key named. A silently ignored typo in a
threshold name is worse than no threshold at all — the suite would report green
while measuring nothing.

### Why thresholds are opt-in on CI

CI runs on shared `ubuntu-latest` runners with no GPU. Wall-clock thresholds
there will flake, and a flaky red build trains people to ignore the suite. So
measurements are recorded unconditionally, assertions need `enforce: true`, and
on CI they additionally need `enforce_on_ci: true`. Behavioral `expect:` blocks
are deterministic and always assert.

### The markdown report

`--report-md=PATH` (default `reports/test_report.md`, gitignored) writes a run
header (model, device, platform, git SHA, pass/fail tally), a per-scenario table
with p50/p95/p99, the threshold, and the margin left under it, then a failures
section. It is written from `pytest_sessionfinish`, so an interrupted or crashed
run still leaves behind whatever was collected. CI uploads it as a build
artifact, so the numbers are readable from a PR without a local run.

## Fixtures

Defined in `conftest.py`:

- `event_loop`: Async event loop for async tests
- `project_root`: Path to project root directory
- `test_prompt`: Single test prompt string
- `test_prompts`: List of 5 test prompts for batch testing
- `scenario`: One entry from `tests/scenarios.yaml`, parametrized over the whole file
- `scenario_report`: The session's result collector, for the markdown report

Use in tests:
```python
def test_something(test_prompt, test_prompts):
    # test_prompt is a single string
    # test_prompts is a list of strings
    pass
```

## Code Quality Tools

### Format Code

```bash
make format
```

Runs:
- `black` for code formatting
- `isort` for import sorting

### Lint Code

```bash
make lint
```

Runs:
- `pylint` for code analysis
- `flake8` for style guide enforcement

### Check Without Modifying

```bash
make check
```

Runs format check + lint without modifying files.

## Running Tests in CI/CD

Tests run automatically on:
- Push to `main` or `develop` branches
- Pull requests to `main` or `develop` branches

See `.github/workflows/tests.yml` for the GitHub Actions workflow.

## Performance Targets

Prompt-driven targets are declared in `tests/scenarios.yaml` under each
scenario's `thresholds:` block — that file is the source of truth, not this
table. The two targets below are asserted in Python because they are trends,
not single numbers:

| Test | Target | Description |
|------|--------|-------------|
| Latency Stability | < 20% degradation | 100 runs comparison |
| Memory Under Load | < 100MB increase | 500 iterations |

## Debugging Tests

### Run with Verbose Output

```bash
pytest tests/ -v -s
```

The `-s` flag shows print statements.

### Run Specific Test

```bash
pytest tests/test_latency.py::TestLatencyBenchmarks::test_tokenizer_latency -v -s
```

### Run with Code Coverage

```bash
pytest tests/ --cov=. --cov-report=html
```

View HTML report:
```bash
open htmlcov/index.html
```

## Adding New Tests

For a benchmark or a behavioral check on a prompt, **add an entry to
`tests/scenarios.yaml`** — no Python required. See
[Declarative Scenarios](#declarative-scenarios) for the schema.

For anything else:

1. Create test function with `test_` prefix in appropriate file
2. Use fixtures from `conftest.py`
3. Add docstring explaining what's tested
4. For latency tests, include performance target

Example:
```python
def test_my_feature(test_prompt):
    """Test my feature with a single prompt"""
    # Arrange
    my_obj = MyClass()
    
    # Act
    result = my_obj.do_something(test_prompt)
    
    # Assert
    assert result is not None
```

## Makefile Commands

```bash
make help          # Show all available commands
make install       # Install project (no dev dependencies)
make dev           # Install with dev dependencies + the chat client
make run           # Run server (dev mode)
make run-prod      # Run server (production mode)
make test-unit     # Run unit tests
make test-latency  # Run latency benchmarks
make test-all      # Run all tests
make test-report   # Run the suite and write reports/test_report.md
make format        # Format code
make lint          # Lint code
make check         # Check without modifying
make logs          # Tail application logs
make clean         # Remove cache and artifacts
```

## Notes

- Tests skip gracefully if required modules (like `tokenizer_service`) are not available
- Latency tests include detailed performance reports with min/max/mean/median/p95/p99 metrics
- Memory tests require `psutil` (included in dev dependencies)
- GitHub Actions runs on Python 3.11 and 3.12
