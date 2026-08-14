"""Client-side configuration for the `ephemeris` CLI.

The REPL client is a plain HTTP client and deliberately does not import
`settings.settings` -- that module pulls in `torch` and reads
`settings/config.yaml` by a repo-relative path, neither of which is available
when the CLI is installed as a console script and run from an arbitrary
directory. This module is the client's own, dependency-light equivalent: it
resolves which server to talk to, from configuration rather than from a
hardcoded address.

Everything here is scoped to the client. It has its own ``.env`` files, its
own YAML files, and its own ``EPHEMERIS_CLIENT_*`` variables -- it never reads
the server's ``.env`` keys even when both happen to live in one directory,
because only ``EPHEMERIS_CLIENT_*`` names are recognized.

Resolution order, highest priority first:

1. ``--url``/``--api-key`` (or ``--host``/``--port``) command-line options
2. real ``EPHEMERIS_CLIENT_*`` environment variables
3. ``.env`` files (see :func:`env_file_paths`)
4. the file named by ``EPHEMERIS_CLIENT_CONFIG``, if set
5. the user-level file (``$XDG_CONFIG_HOME``/``~/.config``)
6. the packaged default, ``ephemeris_cli/client_config.yaml``
"""
from __future__ import annotations

import os
from pathlib import Path
from typing import Any, NamedTuple
from urllib.parse import urlsplit, urlunsplit

import yaml

PACKAGED_CONFIG_PATH = Path(__file__).with_name("client_config.yaml")

CONFIG_PATH_ENV_VAR = "EPHEMERIS_CLIENT_CONFIG"
SERVER_URL_ENV_VAR = "EPHEMERIS_CLIENT_URL"
API_KEY_ENV_VAR = "EPHEMERIS_CLIENT_API_KEY"
ENV_FILE_ENV_VAR = "EPHEMERIS_CLIENT_ENV"

# `.env` entries are only honored for names this client owns. A `.env` that
# also holds the server's HF_KEY/EPHEMERIS_SERVER_* entries is therefore safe
# to sit next to -- those keys are read by the server's settings module, never
# by this one.
_RECOGNIZED_ENV_KEYS = (SERVER_URL_ENV_VAR, API_KEY_ENV_VAR, CONFIG_PATH_ENV_VAR)

# Last-resort values, used only if the packaged config file is missing or
# unreadable (e.g. a partial install). Keeping them here rather than in the
# CLI's option defaults means there is still exactly one place in Python that
# knows an address at all.
_FALLBACK_BASE_URL = "http://127.0.0.1:8080"
_FALLBACK_TIMEOUT_SECONDS = 120.0


class ClientConfigError(Exception):
    """A configuration file exists but could not be used."""


class ResolvedBaseUrl(NamedTuple):
    """A server address plus a human-readable note on where it came from."""

    url: str
    source: str


class ResolvedValue(NamedTuple):
    """Any other resolved setting, plus which layer supplied it."""

    value: str
    source: str


def package_env_path() -> Path:
    """The client package's own ``.env``, alongside its ``pyproject.toml``.

    Present when working in the monorepo checkout (``packages/ephemeris-cli/.env``)
    and simply absent in a wheel install, where the package sits in
    site-packages -- so this costs nothing for installed users while giving
    the client a `.env` of its own, separate from the server's.
    """
    return Path(__file__).resolve().parent.parent / ".env"


def env_file_paths() -> list[Path]:
    """Every ``.env`` this client will read, lowest priority first.

    1. the client package's own ``.env`` (monorepo checkouts)
    2. ``~/.config/ephemeris/.env`` (or the pre-rename ``ephemeris-serve`` directory)
    3. ``./.env`` in the current directory
    4. the file named by ``EPHEMERIS_CLIENT_ENV``
    """
    paths = [package_env_path(), user_config_dir() / ".env", Path.cwd() / ".env"]

    explicit = os.environ.get(ENV_FILE_ENV_VAR)
    if explicit:
        paths.append(Path(explicit).expanduser())
    return paths


def _parse_env_file(path: Path) -> dict[str, str]:
    """Minimal ``.env`` parser for the handful of keys this client owns.

    Deliberately not python-dotenv: this distribution stays at three
    dependencies, and the format actually in use here is ``KEY=value`` with
    comments and optional quoting.
    """
    if not path.is_file():
        return {}

    values: dict[str, str] = {}
    try:
        lines = path.read_text().splitlines()
    except OSError as exc:
        raise ClientConfigError(f"Could not read {path}: {exc}") from exc

    for raw in lines:
        line = raw.strip()
        if not line or line.startswith("#"):
            continue
        if line.startswith("export "):
            line = line[len("export "):].lstrip()
        key, sep, value = line.partition("=")
        if not sep:
            continue
        key = key.strip()
        if key not in _RECOGNIZED_ENV_KEYS:
            continue  # not ours -- e.g. the server's HF_KEY sitting in the same file
        value = value.strip()
        if len(value) >= 2 and value[0] == value[-1] and value[0] in ("'", '"'):
            value = value[1:-1]
        values[key] = value
    return values


def load_env_files() -> dict[str, str]:
    """Merge every ``.env``, later files winning, excluding the real environment."""
    merged: dict[str, str] = {}
    for path in env_file_paths():
        merged.update(_parse_env_file(path))
    return merged


def env_value(name: str) -> str | None:
    """Read ``name`` from the real environment, falling back to the ``.env`` files.

    The process environment wins, matching every other dotenv implementation:
    a value exported for one command must override a file checked in months ago.
    """
    from_process = os.environ.get(name)
    if from_process:
        return from_process
    return load_env_files().get(name) or None


def _config_home() -> Path:
    xdg = os.environ.get("XDG_CONFIG_HOME")
    return Path(xdg) if xdg else Path.home() / ".config"


def legacy_user_config_dir() -> Path:
    """The pre-rename config directory, still read when the current one is absent.

    The client shipped as `ephemeris-serve` before it became its own
    distribution; an upgrade must not silently ignore a config written then.
    """
    return _config_home() / "ephemeris-serve"


def user_config_dir() -> Path:
    """The client's config directory, falling back to the pre-rename one.

    Returns the legacy directory only when it exists and the current one does
    not, so a user who has migrated never has the old path resurface.
    """
    current = _config_home() / "ephemeris"
    if not current.exists() and legacy_user_config_dir().exists():
        return legacy_user_config_dir()
    return current


def user_config_path() -> Path:
    """Return the user-level client config path (need not exist)."""
    return user_config_dir() / "client.yaml"


def _read_defaults(path: Path) -> dict[str, Any]:
    """Read one config file's ``client_config.defaults`` mapping.

    Returns an empty mapping for a file that does not exist, so a missing
    user-level override is simply "nothing set here" rather than an error.
    """
    if not path.is_file():
        return {}
    try:
        loaded = yaml.safe_load(path.read_text()) or {}
    except (OSError, yaml.YAMLError) as exc:
        raise ClientConfigError(f"Could not read client config {path}: {exc}") from exc

    if not isinstance(loaded, dict):
        raise ClientConfigError(f"Client config {path} must be a YAML mapping.")

    defaults = loaded.get("client_config", {}).get("defaults", {})
    if not isinstance(defaults, dict):
        raise ClientConfigError(f"Client config {path}: 'client_config.defaults' must be a mapping.")
    return defaults


def load_config() -> dict[str, Any]:
    """Merge every config file into one mapping, later files winning.

    Files are layered packaged -> user-level -> ``EPHEMERIS_CLIENT_CONFIG``, so
    an override file only has to name the keys it actually changes.
    """
    paths = [PACKAGED_CONFIG_PATH, user_config_path()]

    explicit = env_value(CONFIG_PATH_ENV_VAR)
    if explicit:
        explicit_path = Path(explicit).expanduser()
        if not explicit_path.is_file():
            raise ClientConfigError(
                f"{CONFIG_PATH_ENV_VAR} points at {explicit_path}, which does not exist."
            )
        paths.append(explicit_path)

    merged: dict[str, Any] = {}
    for path in paths:
        merged.update(_read_defaults(path))
    return merged


def normalize_base_url(value: str) -> str:
    """Return ``value`` as a scheme-qualified base URL with no trailing slash.

    A bare ``host`` or ``host:port`` is accepted and assumed to be plain HTTP,
    so an operator can write ``ephemeris.example.com`` in a config file without
    it being parsed as a relative path.
    """
    candidate = value.strip()
    if not candidate:
        raise ClientConfigError("Server URL is empty.")

    if "://" not in candidate:
        candidate = f"http://{candidate}"

    parts = urlsplit(candidate)
    if parts.scheme not in ("http", "https"):
        raise ClientConfigError(f"Server URL {value!r} must use http or https, not {parts.scheme!r}.")
    if not parts.netloc:
        raise ClientConfigError(f"Server URL {value!r} is missing a host.")

    # Keep any path prefix (a proxy may mount the API under a subpath) but drop
    # a trailing slash, since every request path the CLI builds starts with one.
    return urlunsplit((parts.scheme, parts.netloc, parts.path.rstrip("/"), "", ""))


def resolve_base_url(
    url: str | None = None,
    host: str | None = None,
    port: int | None = None,
    config: dict[str, Any] | None = None,
) -> ResolvedBaseUrl:
    """Resolve the server address from options, environment, and config files."""
    if url and (host or port):
        raise ClientConfigError("Use either --url or --host/--port, not both.")

    if url:
        return ResolvedBaseUrl(normalize_base_url(url), "--url")

    if host or port:
        config = load_config() if config is None else config
        # Fill in whichever half wasn't given from the configured address, so
        # `--port 9000` alone still points at the configured host.
        configured = urlsplit(normalize_base_url(str(config.get("base_url") or _FALLBACK_BASE_URL)))
        resolved_host = host or configured.hostname or "127.0.0.1"
        resolved_port = port or configured.port
        netloc = f"{resolved_host}:{resolved_port}" if resolved_port else resolved_host
        return ResolvedBaseUrl(
            urlunsplit((configured.scheme, netloc, configured.path, "", "")),
            "--host/--port",
        )

    from_env = os.environ.get(SERVER_URL_ENV_VAR)
    if from_env:
        return ResolvedBaseUrl(normalize_base_url(from_env), f"${SERVER_URL_ENV_VAR}")

    from_dotenv = load_env_files().get(SERVER_URL_ENV_VAR)
    if from_dotenv:
        return ResolvedBaseUrl(normalize_base_url(from_dotenv), f"{SERVER_URL_ENV_VAR} (.env)")

    config = load_config() if config is None else config
    configured = config.get("base_url")
    if configured:
        return ResolvedBaseUrl(normalize_base_url(str(configured)), "client config")

    return ResolvedBaseUrl(_FALLBACK_BASE_URL, "built-in fallback")


def resolve_api_key(api_key: str | None = None, config: dict[str, Any] | None = None) -> ResolvedValue | None:
    """Resolve the API key sent to the server, or ``None`` if none is set.

    Order is ``--api-key``, then ``$EPHEMERIS_CLIENT_API_KEY``, then the config
    files' ``api_key``. The environment variable is listed above the file
    deliberately: a credential is better kept out of a file on disk, and a
    per-shell or systemd-supplied value should win over a stale checked-in one.

    A server with no keys configured accepts unauthenticated requests, so
    ``None`` is a valid outcome rather than an error.
    """
    if api_key:
        return ResolvedValue(api_key, "--api-key")

    from_env = os.environ.get(API_KEY_ENV_VAR)
    if from_env:
        return ResolvedValue(from_env, f"${API_KEY_ENV_VAR}")

    from_dotenv = load_env_files().get(API_KEY_ENV_VAR)
    if from_dotenv:
        return ResolvedValue(from_dotenv, f"{API_KEY_ENV_VAR} (.env)")

    config = load_config() if config is None else config
    configured = config.get("api_key")
    if configured:
        return ResolvedValue(str(configured), "client config")

    return None


def mask_secret(value: str) -> str:
    """Render a key for display without disclosing it.

    Shows enough leading characters to tell two keys apart when debugging,
    which is the whole point of showing it at all in `ephemeris config`.
    """
    if len(value) <= 8:
        return "*" * len(value)
    return f"{value[:4]}{'*' * (len(value) - 8)}{value[-4:]}"


def auth_headers(api_key: str | None) -> dict[str, str]:
    """Headers carrying ``api_key``, or an empty mapping if there is none."""
    return {"Authorization": f"Bearer {api_key}"} if api_key else {}


def resolve_timeout(timeout: float | None = None, config: dict[str, Any] | None = None) -> float:
    """Resolve the per-request HTTP timeout: option, then config, then fallback."""
    if timeout is not None:
        return timeout

    config = load_config() if config is None else config
    configured = config.get("timeout_seconds")
    if configured is None:
        return _FALLBACK_TIMEOUT_SECONDS
    try:
        return float(configured)
    except (TypeError, ValueError) as exc:
        raise ClientConfigError(f"Client config: 'timeout_seconds' must be a number, got {configured!r}.") from exc
