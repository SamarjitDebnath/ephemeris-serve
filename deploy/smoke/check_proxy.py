"""Assertions for the nginx smoke test.

Run by `deploy/smoke/smoke_test.sh` once nginx and the stub upstream are up.
Every check returns pass/fail and the script exits non-zero if any failed, so
this is usable in CI rather than something a human has to eyeball.
"""
from __future__ import annotations

import json
import os
import statistics
import sys
import time

import httpx

DIRECT = f"http://127.0.0.1:{os.environ.get('UPSTREAM_PORT', '18000')}"
PROXIED = f"http://127.0.0.1:{os.environ.get('PROXY_PORT', '18080')}"

results: list[tuple[bool, str, str]] = []


def check(name: str, passed: bool, detail: str = "") -> None:
    results.append((passed, name, detail))
    print(f"  {'PASS' if passed else 'FAIL'}  {name}" + (f"  --  {detail}" if detail else ""))


def stream_arrivals(base: str, path: str = "/api/generate") -> tuple[int, list[float]]:
    """Return (status, arrival times) for each SSE data frame."""
    start = time.monotonic()
    arrivals: list[float] = []
    with httpx.Client(timeout=90.0) as client:
        with client.stream("POST", base + path, headers={"Accept": "text/event-stream"}) as r:
            status = r.status_code
            for line in r.iter_lines():
                if line.startswith("data:"):
                    arrivals.append(time.monotonic() - start)
    return status, arrivals


def spread(arrivals: list[float]) -> float:
    return arrivals[-1] - arrivals[0] if len(arrivals) > 1 else 0.0


print("health")
with httpx.Client(timeout=10.0) as c:
    check("direct /health is 200", c.get(f"{DIRECT}/health").status_code == 200)
    check("proxied /health is 200", c.get(f"{PROXIED}/health").status_code == 200)

print("\nSSE streaming")
_, direct = stream_arrivals(DIRECT)
_, proxied = stream_arrivals(PROXIED)
_, no_hint = stream_arrivals(PROXIED, "/api/generate_no_hint")

check("proxy delivers every frame", len(proxied) == len(direct) == 10,
      f"direct={len(direct)} proxied={len(proxied)}")
# Buffering would collapse the arrivals into one burst at the end.
check("proxied stream is not buffered", spread(proxied) > 1.0,
      f"spread={spread(proxied):.2f}s")
# The proxy must not add meaningful latency on top of the upstream's own pacing.
check("proxy adds no material delay", abs(spread(proxied) - spread(direct)) < 1.0,
      f"direct={spread(direct):.2f}s proxied={spread(proxied):.2f}s")
# sse_starlette sets X-Accel-Buffering itself; this route omits it, so a pass
# here means `proxy_buffering off` is carrying the config on its own.
check("unbuffered without the upstream hint header", spread(no_hint) > 1.0,
      f"spread={spread(no_hint):.2f}s")
if len(proxied) > 1:
    gaps = [b - a for a, b in zip(proxied, proxied[1:])]
    print(f"        median inter-frame gap: {statistics.median(gaps):.2f}s")

print("\nheaders")
with httpx.Client(timeout=10.0) as c:
    body = c.get(f"{PROXIED}/api/metrics", headers={"Authorization": "Bearer smoke-key"}).json()
print("        " + json.dumps(body))
check("Authorization survives the proxy", body.get("authorization") == "Bearer smoke-key")
check("X-Forwarded-For is set", bool(body.get("x_forwarded_for")))
check("X-Forwarded-Proto is set", body.get("x_forwarded_proto") in ("http", "https"))

print("\nbody cap")
with httpx.Client(timeout=30.0) as c:
    r = c.post(f"{PROXIED}/api/generate_batch", content=b"x" * 2_000_000,
               headers={"Content-Type": "application/json"})
check("2MB body rejected with 413", r.status_code == 413, f"got {r.status_code}")

print("\nrate limit")
codes: list[int] = []
started = time.monotonic()
with httpx.Client(timeout=10.0) as c:
    for _ in range(40):
        codes.append(c.get(f"{PROXIED}/api/metrics").status_code)
elapsed = time.monotonic() - started
allowed = codes.count(200)
limited = codes.count(503)
print(f"        {allowed} allowed, {limited} limited, over {elapsed:.2f}s")
# Deliberately not asserting an exact split. `limit_req` is a leaky bucket:
# it refills at 10r/s while the burst drains, so a request can succeed *after*
# earlier ones were rejected. Only the envelope is stable.
ceiling = 20 + 10 * elapsed + 5  # burst + refill over the run + margin
check("rate limit engages", limited > 0, f"{limited} rejected")
check("allowed count within burst+refill envelope", allowed <= ceiling,
      f"allowed={allowed} ceiling={ceiling:.0f}")
check("limiter is not rejecting everything", allowed >= 20,
      f"allowed={allowed}, burst is 20")

failed = [name for ok, name, _ in results if not ok]
print("\n" + "=" * 60)
print(f"{len(results) - len(failed)}/{len(results)} checks passed")
if failed:
    print("FAILED: " + "; ".join(failed))
    sys.exit(1)
print("nginx config verified")
