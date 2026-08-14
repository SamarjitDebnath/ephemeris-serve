"""API-key authentication for the `/api` routes.

Two tiers of key, both read from the environment (see `SecretSetting` in
`settings/settings.py`):

* ``EPHEMERIS_API_KEYS`` -- ordinary access: generation and read-only routes.
* ``EPHEMERIS_ADMIN_API_KEYS`` -- additionally allowed to swap the model.
  ``POST /api/model`` is the one route that makes the server download and load
  an arbitrary Hugging Face repo, so it is gated separately from generation.
  An admin key satisfies the ordinary tier too.

Each variable holds a comma-separated list, so keys can be rotated by adding
the new one, redeploying clients, then removing the old.

**If no keys are configured at all, authentication is disabled** and every
route stays open, which is what keeps `make run` and the test suite working
with no setup. `api/server.py` logs a prominent warning at startup in that
case. Any deployment reachable from outside localhost must set at least
``EPHEMERIS_API_KEYS``; see `deploy/README.md`.
"""
import hmac

from fastapi import Depends, Header, HTTPException

from logger import setup_logger
from settings.settings import logging_settings, secret_settings

logger = setup_logger(__name__, level=logging_settings.log_level, log_file=logging_settings.log_file)

_MISSING_KEY_MESSAGE = "Missing API key. Send it as 'Authorization: Bearer <key>'."
_INVALID_KEY_MESSAGE = "Invalid API key."
_ADMIN_REQUIRED_MESSAGE = "This endpoint requires an admin API key."


def _parse_keys(raw: str | None) -> tuple[str, ...]:
    """Split a comma-separated key list, dropping blanks and surrounding space."""
    if not raw:
        return ()
    return tuple(key.strip() for key in raw.split(",") if key.strip())


def api_keys() -> tuple[str, ...]:
    """Ordinary-tier keys. Admin keys are included, so they work everywhere."""
    return _parse_keys(secret_settings.api_keys) + admin_api_keys()


def admin_api_keys() -> tuple[str, ...]:
    """Keys allowed to perform a model swap."""
    return _parse_keys(secret_settings.admin_api_keys)


def auth_enabled() -> bool:
    """True once any key is configured; until then every route is open."""
    return bool(api_keys())


def _extract_bearer(authorization: str | None) -> str | None:
    """Pull the token out of an ``Authorization: Bearer <token>`` header."""
    if not authorization:
        return None
    scheme, _, token = authorization.partition(" ")
    if scheme.lower() != "bearer" or not token.strip():
        return None
    return token.strip()


def _matches(candidate: str, allowed: tuple[str, ...]) -> bool:
    """Constant-time membership test.

    `hmac.compare_digest` avoids leaking how much of a key was correct through
    response timing. Every entry is compared even after a match, so the number
    of comparisons doesn't vary either.
    """
    matched = False
    for key in allowed:
        if hmac.compare_digest(candidate, key):
            matched = True
    return matched


def require_api_key(authorization: str | None = Header(default=None)) -> str | None:
    """FastAPI dependency: allow the request if it carries a valid key.

    Returns the presented key (or ``None`` when auth is disabled) so a route
    can depend on it without re-reading the header.
    """
    if not auth_enabled():
        return None

    token = _extract_bearer(authorization)
    if token is None:
        raise HTTPException(status_code=401, detail=_MISSING_KEY_MESSAGE)
    if not _matches(token, api_keys()):
        # Logged without the presented value: a mistyped key is often a real
        # key belonging to another environment.
        logger.warning("Rejected request carrying an unrecognized API key")
        raise HTTPException(status_code=401, detail=_INVALID_KEY_MESSAGE)
    return token


def require_admin_api_key(token: str | None = Depends(require_api_key)) -> str | None:
    """FastAPI dependency for routes that may change what the server runs."""
    if not auth_enabled():
        return None

    admin_keys = admin_api_keys()
    if not admin_keys:
        # Ordinary keys are configured but no admin key is. Refuse rather than
        # silently letting every key swap the model.
        logger.warning("Model swap attempted with no EPHEMERIS_ADMIN_API_KEYS configured")
        raise HTTPException(status_code=403, detail=_ADMIN_REQUIRED_MESSAGE)

    if token is None or not _matches(token, admin_keys):
        logger.warning("Rejected model swap attempted with a non-admin API key")
        raise HTTPException(status_code=403, detail=_ADMIN_REQUIRED_MESSAGE)
    return token
