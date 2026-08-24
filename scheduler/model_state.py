"""Shared model-swap state for a multi-worker deployment.

`POST /api/model` reaches exactly one worker, but everything the swap touches
is per-process: `swap_lock`, `model_settings.model_name`, and the engine /
tokenizer / paged-cache singletons. With `--workers 4` that leaves three
workers serving the old model with no error to the client.

The coordination here is deliberately **not** a distributed transaction. Each
worker has to drain its own in-flight requests before reloading, and those
drains finish at different times, so a window where different workers serve
different models is unavoidable. What this module provides instead is eventual
convergence with an observable state:

1. every worker converges on the requested model,
2. the window is bounded by each worker's own drain,
3. a client can tell whether convergence has finished.

The mechanism is a JSON file plus a monotone generation counter, written under
an OS file lock::

    {"model_name": "Qwen/Qwen2.5-0.5B", "generation": 7, "updated_at": ...}

Each worker additionally writes ``worker-<pid>.json`` recording the generation
it has actually reached, which is what makes convergence reportable.

All `fcntl` use is confined to this module -- it is the only platform-specific
code in the feature, and a Windows port would replace it here and nowhere else.
If the state directory is unset or unwritable, every function degrades to the
single-process behavior the server had before, rather than failing.
"""
import json
import os
import time
from pathlib import Path
from typing import Optional, Tuple

from logger import setup_logger
from settings.settings import logging_settings, scheduler_settings

logger = setup_logger(__name__, level=logging_settings.log_level, log_file=logging_settings.log_file)

_STATE_FILENAME = "model_state.json"
_WORKER_PREFIX = "worker-"
#: Worker files older than this are treated as belonging to dead processes.
_WORKER_STALE_SECONDS = 300.0

try:  # pragma: no cover - platform dependent
    import fcntl
except ImportError:  # pragma: no cover - Windows
    # Coordination still works without it -- the generation counter and the
    # convergence files do the real work. Losing the lock only widens the race
    # on a concurrent `publish_target`, which is a rare admin action.
    fcntl = None


def state_dir() -> Optional[Path]:
    """The configured state directory, or None when coordination is off."""
    configured = getattr(scheduler_settings, "model_state_dir", None)
    if not configured:
        return None
    return Path(configured).expanduser()


def enabled() -> bool:
    return state_dir() is not None


def _state_path() -> Optional[Path]:
    directory = state_dir()
    return None if directory is None else directory / _STATE_FILENAME


def _ensure_dir() -> Optional[Path]:
    directory = state_dir()
    if directory is None:
        return None
    try:
        directory.mkdir(parents=True, exist_ok=True)
    except OSError as exc:
        logger.warning("Model state dir %s is not usable (%s); coordination disabled", directory, exc)
        return None
    return directory


class _Locked:
    """`flock` around an open file handle, a no-op where flock is missing."""

    def __init__(self, handle):
        self.handle = handle

    def __enter__(self):
        # Tested against the module rather than `_HAVE_FLOCK` so a type
        # checker can narrow it -- the boolean carries the same information
        # but not to Pyright.
        if fcntl is not None:
            fcntl.flock(self.handle.fileno(), fcntl.LOCK_EX)
        return self.handle

    def __exit__(self, *_exc_info):
        if fcntl is not None:
            fcntl.flock(self.handle.fileno(), fcntl.LOCK_UN)
        return False


def read_state() -> Optional[Tuple[str, int]]:
    """`(model_name, generation)`, or None if unset/unreadable.

    A corrupt or truncated file is ignored with a warning rather than raised:
    every worker reads this on a timer, so an exception here would take down
    the whole pool at once over one bad write.
    """
    path = _state_path()
    if path is None or not path.is_file():
        return None
    try:
        with path.open("r", encoding="utf-8") as handle:
            with _Locked(handle):
                document = json.load(handle)
        return str(document["model_name"]), int(document["generation"])
    except (OSError, ValueError, KeyError, TypeError) as exc:
        logger.warning("Ignoring unreadable model state file %s: %s", path, exc)
        return None


def publish_target(model_name: str) -> Optional[int]:
    """Record `model_name` as the target and bump the generation.

    Returns the new generation, or None when coordination is disabled.
    """
    directory = _ensure_dir()
    if directory is None:
        return None
    path = directory / _STATE_FILENAME
    try:
        # Opened "a+" so the file is created if absent, without truncating an
        # existing one before the lock is held.
        with path.open("a+", encoding="utf-8") as handle:
            with _Locked(handle):
                handle.seek(0)
                raw = handle.read()
                try:
                    current = json.loads(raw) if raw.strip() else {}
                except ValueError:
                    current = {}
                generation = int(current.get("generation", 0)) + 1
                handle.seek(0)
                handle.truncate()
                json.dump(
                    {"model_name": model_name, "generation": generation, "updated_at": time.time()},
                    handle,
                )
                handle.flush()
                os.fsync(handle.fileno())
        return generation
    except OSError as exc:
        logger.warning("Could not publish model state to %s: %s", path, exc)
        return None


def publish_worker_generation(generation: int, model_name: str) -> None:
    """Record what this worker has actually converged on."""
    directory = _ensure_dir()
    if directory is None:
        return
    path = directory / f"{_WORKER_PREFIX}{os.getpid()}.json"
    try:
        path.write_text(
            json.dumps(
                {
                    "pid": os.getpid(),
                    "generation": generation,
                    "model_name": model_name,
                    "updated_at": time.time(),
                }
            ),
            encoding="utf-8",
        )
    except OSError as exc:
        logger.warning("Could not publish worker generation to %s: %s", path, exc)


def convergence() -> Tuple[int, int]:
    """`(converged_workers, known_workers)` for the current target generation.

    Stale files from workers that died are reaped rather than counted -- one
    crashed worker must not leave the pool permanently reported as unconverged.
    """
    directory = state_dir()
    state = read_state()
    if directory is None or state is None or not directory.is_dir():
        return (0, 0)
    _, target = state

    converged = 0
    known = 0
    now = time.time()
    for path in directory.glob(f"{_WORKER_PREFIX}*.json"):
        try:
            document = json.loads(path.read_text(encoding="utf-8"))
            updated_at = float(document.get("updated_at", 0.0))
            if now - updated_at > _WORKER_STALE_SECONDS:
                path.unlink(missing_ok=True)
                continue
            known += 1
            if int(document.get("generation", -1)) >= target:
                converged += 1
        except (OSError, ValueError, TypeError):
            # Half-written file from a worker mid-publish; skip it this round.
            continue
    return (converged, known)
