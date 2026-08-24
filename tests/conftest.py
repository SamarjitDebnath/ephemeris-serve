"""Pytest configuration and shared fixtures"""
import sys
from pathlib import Path

# Fallback for checkouts where the client was never installed.
#
# The chat client is a separate distribution (packages/ephemeris-cli) with its
# own pyproject and dependency set, so it is not importable from the repo root
# the way the server packages are. `make dev` and CI both install it
# (`uv pip install -e packages/ephemeris-cli`), which is the primary mechanism
# and the one static analysis can see -- see `pyrightconfig.json`. This
# insertion only keeps the suite green for a contributor who skipped that step;
# the unit-test job in CI has no `continue-on-error`, so an unresolved import
# there is a hard red build.
_CLI_PACKAGE_ROOT = Path(__file__).resolve().parent.parent / "packages" / "ephemeris-cli"
if str(_CLI_PACKAGE_ROOT) not in sys.path:
    sys.path.insert(0, str(_CLI_PACKAGE_ROOT))

from utils.utils import Utils

# Prevent multiprocessing/torch semaphore leaks at shutdown
Utils.configure_multiprocessing()

import asyncio
import pytest
from pathlib import Path

from tests import report as report_module
from tests.scenarios import Scenario, ScenarioError, load_scenarios, scenario_path

# Parsed once per session. `pytest_configure` needs the tag list before
# collection starts, and `pytest_generate_tests` needs the scenarios
# themselves during it; re-reading the file for each would be wasteful and
# could disagree if it changed mid-run.
_SCENARIOS: "list[Scenario] | None" = None


def _scenarios() -> "list[Scenario]":
    global _SCENARIOS
    if _SCENARIOS is None:
        try:
            _SCENARIOS = load_scenarios()
        except ScenarioError as exc:
            # `UsageError` rather than letting `ScenarioError` escape: raised
            # from `pytest_configure` an ordinary exception surfaces as an
            # INTERNALERROR dump, which buries the one line that says which
            # key in which scenario is wrong.
            raise pytest.UsageError(str(exc)) from exc
    return _SCENARIOS


def pytest_addoption(parser):
    parser.addoption(
        "--report-md",
        action="store",
        default="reports/test_report.md",
        metavar="PATH",
        help="Where to write the markdown scenario report (default: reports/test_report.md).",
    )


def pytest_configure(config):
    """Register every scenario tag as a marker and stamp the report header.

    `pytest.ini` runs with `--strict-markers`, so a tag that is not a
    registered marker would be a hard collection error. Registering them from
    the scenario file is what lets a contributor add a tag in YAML -- and
    select it with `-m` -- without touching `pytest.ini`.
    """
    scenarios = _scenarios()
    for tag in sorted({tag for scenario in scenarios for tag in scenario.tags}):
        config.addinivalue_line("markers", f"{tag}: scenario tag from the scenario file")

    report_module.collector.reset()
    report_module.collector.context.scenario_file = str(scenario_path())
    report_module.collector.context.device = report_module.detect_device()
    if scenarios:
        report_module.collector.context.model = scenarios[0].model


def pytest_generate_tests(metafunc):
    """Expand the scenario file into one test case per entry.

    `name` becomes the pytest id, so a failure reads as
    `test_scenario[short_prompt_latency]` and points straight at the YAML
    entry that produced it.
    """
    if "scenario" not in metafunc.fixturenames:
        return
    params = [
        pytest.param(
            scenario,
            id=scenario.name,
            marks=[getattr(pytest.mark, tag) for tag in scenario.tags],
        )
        for scenario in _scenarios()
    ]
    metafunc.parametrize("scenario", params)


@pytest.fixture(scope="session")
def scenario_report():
    """The session's result collector -- scenario tests record into this."""
    return report_module.collector


def pytest_sessionfinish(session, exitstatus):
    """Write the markdown report.

    Deliberately here rather than in a fixture teardown: this hook still runs
    when the session is interrupted, so a `Ctrl-C`'d or crashed run leaves a
    partial report behind instead of nothing.
    """
    if not report_module.collector.results:
        return
    destination = Path(session.config.getoption("--report-md"))
    written = report_module.write_report(destination, report_module.collector)
    reporter = session.config.pluginmanager.get_plugin("terminalreporter")
    if reporter is not None:
        reporter.write_line(f"scenario report written to {written}")


@pytest.fixture(scope="session")
def event_loop():
    """Create an event loop for async tests"""
    try:
        loop = asyncio.get_running_loop()
    except RuntimeError:
        loop = asyncio.new_event_loop()
        asyncio.set_event_loop(loop)
    yield loop
    loop.close()


@pytest.fixture
def project_root():
    """Return the project root directory"""
    return Path(__file__).parent.parent


@pytest.fixture
def test_prompt():
    """Return a test prompt"""
    return "What is machine learning?"


@pytest.fixture
def test_prompts():
    """Return multiple test prompts for batch testing"""
    return [
        "What is machine learning?",
        "Explain deep learning in simple terms",
        "How does a neural network work?",
        "What is a transformer model?",
        "Describe gradient descent",
    ]


try:
    from ephemeris_cli.config import env_file_paths as _ORIGINAL_ENV_FILE_PATHS
except ImportError:  # pragma: no cover - client package not on the path
    _ORIGINAL_ENV_FILE_PATHS = None


@pytest.fixture(autouse=True)
def isolate_client_env_files(monkeypatch):
    """Keep real `.env` files off the client-config tests.

    The client reads `.env` from the package directory, `~/.config`, and the
    current directory. A developer with any of those populated would otherwise
    see different results than CI. Tests that exercise `.env` handling
    monkeypatch `env_file_paths` themselves, which overrides this.
    """
    try:
        from ephemeris_cli import config as client_config
    except ImportError:  # client package not on the path
        return
    monkeypatch.setattr(client_config, "env_file_paths", lambda: [])


@pytest.fixture
def real_env_file_paths():
    """The unpatched `env_file_paths`, for tests that assert on its result.

    `isolate_client_env_files` above replaces it for every test; a test that is
    specifically about which paths get searched needs the original back.
    """
    from ephemeris_cli import config as client_config

    return client_config.env_file_paths.__wrapped__ if hasattr(
        client_config.env_file_paths, "__wrapped__"
    ) else _ORIGINAL_ENV_FILE_PATHS
