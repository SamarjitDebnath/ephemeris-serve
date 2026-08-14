"""Pytest configuration and shared fixtures"""
import sys
from pathlib import Path

# The chat client is a separate distribution (packages/ephemeris-cli) with its
# own pyproject and dependency set, so it is not importable from the repo root
# the way the server packages are. Put it on sys.path here so this monorepo's
# tests can exercise both sides without installing either one.
_CLI_PACKAGE_ROOT = Path(__file__).resolve().parent.parent / "packages" / "ephemeris-cli"
if str(_CLI_PACKAGE_ROOT) not in sys.path:
    sys.path.insert(0, str(_CLI_PACKAGE_ROOT))

from utils.utils import Utils

# Prevent multiprocessing/torch semaphore leaks at shutdown
Utils.configure_multiprocessing()

import asyncio
import pytest
from pathlib import Path


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
