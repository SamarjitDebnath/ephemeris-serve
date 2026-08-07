import time

from scheduler.request import InferenceRequest
from settings.settings import scheduler_settings


class IdempotencyStore:
    """In-process, TTL-bounded map of client idempotency keys to requests.

    Purging is lazy (on `get`/`put`) rather than a background task, keeping
    this self-contained -- there's no persistence layer elsewhere in this
    repo to build on, so this stays scoped to a single process's lifetime.
    """

    def __init__(self, ttl_seconds: float | None = None):
        self._ttl_seconds = ttl_seconds if ttl_seconds is not None else scheduler_settings.idempotency_key_ttl_seconds
        self._entries: dict[str, tuple[float, InferenceRequest]] = {}

    def _purge_expired(self) -> None:
        now = time.monotonic()
        expired = [key for key, (expiry, _) in self._entries.items() if expiry <= now]
        for key in expired:
            del self._entries[key]

    def get(self, key: str) -> InferenceRequest | None:
        self._purge_expired()
        entry = self._entries.get(key)
        return entry[1] if entry is not None else None

    def put(self, key: str, request: InferenceRequest) -> None:
        self._purge_expired()
        self._entries[key] = (time.monotonic() + self._ttl_seconds, request)

    def discard(self, key: str) -> None:
        self._entries.pop(key, None)


idempotency_store = IdempotencyStore()
