"""Per-identity rate limiting for the generation routes.

Complements, rather than replaces, the per-address limit in
`deploy/nginx/ephemeris-serve.conf`. That one is a good outer layer against
connection floods that never need to reach Python, but it has three gaps this
module closes:

* it only exists behind the bundled nginx, so `make run` / `make run-prod` and
  any other front end are unprotected;
* it keys on IP, and for a multi-tenant deployment the tenant is the API key --
  two tenants behind one NAT would share a bucket, one tenant across ten
  addresses would get ten;
* it counts requests, and a `/generate_batch` carrying 32 sub-requests is one
  request to nginx. Cost has to be charged where it is known.

Two limits per identity, both needed:

* **Rate** -- a token bucket, which permits a burst without permitting a
  sustained rate. That matches how generation traffic actually arrives.
* **Concurrency** -- a cap on in-flight requests. Rate alone does not bound how
  many long generations one tenant holds open at once, and it is the concurrent
  ones that occupy `max_batch_size` slots and KV blocks.

**Per-worker caveat.** The buckets live in process memory, so with
`--workers N` the effective limit is N times the configured one and which
worker a request lands on is uvicorn's choice. This is adequate for abuse
prevention, not for billing-grade quota. Closing it needs state shared between
workers -- the same gap `scheduler/model_state.py` and the Prometheus
multiprocess exporter address; see `internal/to_execute.md`.
"""
import asyncio
import time
from contextlib import asynccontextmanager
from dataclasses import dataclass, field
from typing import Dict, Optional

from fastapi import HTTPException, Request

from logger import setup_logger
from settings.settings import logging_settings, rate_limit_settings

logger = setup_logger(__name__, level=logging_settings.log_level, log_file=logging_settings.log_file)

#: Buckets for identities not seen for this long are dropped, so a deployment
#: that rotates keys does not grow the table without bound.
_IDLE_EVICTION_SECONDS = 3600.0


@dataclass
class _Bucket:
    """One identity's token bucket plus its in-flight count."""

    tokens: float
    last_refill: float
    in_flight: int = 0
    last_seen: float = field(default_factory=time.monotonic)

    def refill(self, now: float, rate: float, capacity: float) -> None:
        elapsed = now - self.last_refill
        if elapsed > 0:
            self.tokens = min(capacity, self.tokens + elapsed * rate)
            self.last_refill = now

    def retry_after(self, cost: float, rate: float) -> float:
        """Seconds until `cost` tokens are available. Never zero -- a
        `Retry-After: 0` invites an immediate retry that fails again."""
        if rate <= 0:
            return 1.0
        deficit = max(cost - self.tokens, 0.0)
        return max(deficit / rate, 0.001)


class RateLimiter:
    """Token buckets keyed by identity. One instance per process."""

    def __init__(self) -> None:
        self._buckets: Dict[str, _Bucket] = {}
        self._lock = asyncio.Lock()
        self._last_eviction = time.monotonic()

    def reset(self) -> None:
        """Drop all state. For tests."""
        self._buckets = {}
        self._last_eviction = time.monotonic()

    def _evict_idle(self, now: float) -> None:
        if now - self._last_eviction < _IDLE_EVICTION_SECONDS:
            return
        self._last_eviction = now
        stale = [
            key
            for key, bucket in self._buckets.items()
            if bucket.in_flight == 0 and now - bucket.last_seen > _IDLE_EVICTION_SECONDS
        ]
        for key in stale:
            del self._buckets[key]
        if stale:
            logger.debug("Rate limiter evicted %d idle buckets", len(stale))

    async def acquire(self, identity: str, cost: int = 1) -> None:
        """Charge `cost` to `identity`, or raise 429.

        Also claims one concurrency slot, which `release` must hand back --
        including on the error path. A limiter that leaks a slot on failure
        eventually locks a tenant out permanently.
        """
        if not rate_limit_settings.enabled:
            return

        rate = rate_limit_settings.requests_per_second
        capacity = float(rate_limit_settings.burst)
        max_concurrent = rate_limit_settings.max_concurrent_requests

        async with self._lock:
            now = time.monotonic()
            self._evict_idle(now)
            bucket = self._buckets.get(identity)
            if bucket is None:
                bucket = _Bucket(tokens=capacity, last_refill=now)
                self._buckets[identity] = bucket

            bucket.last_seen = now
            bucket.refill(now, rate, capacity)

            if max_concurrent > 0 and bucket.in_flight >= max_concurrent:
                logger.warning("Rate limit: concurrency cap reached for one identity")
                raise HTTPException(
                    status_code=429,
                    detail=f"Too many concurrent requests (limit {max_concurrent}).",
                    headers={"Retry-After": "1"},
                )

            if bucket.tokens < cost:
                retry_after = bucket.retry_after(cost, rate)
                logger.warning("Rate limit: request rate exceeded for one identity")
                raise HTTPException(
                    status_code=429,
                    detail="Rate limit exceeded.",
                    # Ceil to whole seconds: Retry-After is defined in seconds,
                    # and rounding down would invite a retry that fails again.
                    headers={"Retry-After": str(max(1, int(retry_after + 0.999)))},
                )

            bucket.tokens -= cost
            bucket.in_flight += 1

    async def release(self, identity: str) -> None:
        """Hand back the concurrency slot claimed by `acquire`."""
        if not rate_limit_settings.enabled:
            return
        async with self._lock:
            bucket = self._buckets.get(identity)
            if bucket is not None and bucket.in_flight > 0:
                bucket.in_flight -= 1


limiter = RateLimiter()


def identity_for(token: Optional[str], request: Request) -> str:
    """Who to charge: the API key when there is one, else the client address.

    `require_api_key` already returns the presented key precisely so a route can
    reuse it without re-reading the header. The IP fallback keeps local
    development limited without requiring keys to be configured.
    """
    if token:
        return f"key:{token}"
    client = request.client
    return f"ip:{client.host}" if client is not None else "ip:unknown"


@asynccontextmanager
async def rate_limited(request: Request, token: Optional[str], cost: int = 1):
    """Charge `cost` for the duration of the block, then release.

    Used by `/generate_batch`, whose handler is still running when the work
    completes. `/generate` cannot use this -- its handler returns while
    generation is still going -- and uses `release_after` instead.
    """
    identity = identity_for(token, request)
    await limiter.acquire(identity, cost=cost)
    try:
        yield identity
    finally:
        await limiter.release(identity)


async def release_after(generator, identity: str):
    """Wrap a streaming generator so the concurrency slot comes back when the
    stream ends -- normally, by client disconnect, or by exception.

    `/generate` returns an SSE response whose body is produced long after the
    handler returns, so releasing in the handler would free the slot while the
    generation it accounts for is still occupying a batch slot.
    """
    try:
        async for item in generator:
            yield item
    finally:
        await limiter.release(identity)
