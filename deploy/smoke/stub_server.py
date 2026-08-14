"""Stand-in upstream for the nginx smoke test.

Exposes the same routes as the real server but loads no model, so the proxy
can be exercised in seconds instead of waiting on a multi-gigabyte download.
Only the proxy's behavior is under test here -- buffering, headers, limits --
none of which depends on real generation.
"""
import asyncio

from fastapi import FastAPI, Request
from sse_starlette.sse import EventSourceResponse
from starlette.responses import StreamingResponse

app = FastAPI()

TOKENS = [f"tok{i} " for i in range(10)]
DELAY = 0.3  # slow enough that buffering would be obvious in the arrival times


@app.get("/health")
async def health():
    return {"status": "healthy"}


@app.post("/api/generate")
async def generate(request: Request):
    """SSE via sse_starlette, exactly as the real server does it."""

    async def gen():
        for tok in TOKENS:
            await asyncio.sleep(DELAY)
            yield tok

    return EventSourceResponse(gen())


@app.post("/api/generate_no_hint")
async def generate_no_hint(request: Request):
    """The same stream without `X-Accel-Buffering: no`.

    sse_starlette sets that header itself, and nginx honors it from upstream --
    which would mask a missing `proxy_buffering off`. This route removes that
    safety net so the nginx config is tested on its own merits.
    """

    async def gen():
        for tok in TOKENS:
            await asyncio.sleep(DELAY)
            yield f"data: {tok}\n\n"

    return StreamingResponse(gen(), media_type="text/event-stream")


@app.get("/api/metrics")
async def metrics(request: Request):
    """Echo what the app sees, so forwarded headers can be asserted on."""
    return {
        "client_host": request.client.host if request.client else None,
        "scheme": request.url.scheme,
        "x_forwarded_for": request.headers.get("x-forwarded-for"),
        "x_forwarded_proto": request.headers.get("x-forwarded-proto"),
        "authorization": request.headers.get("authorization"),
    }


@app.post("/api/generate_batch")
async def generate_batch(request: Request):
    return {"outputs": [], "batch_size": 0}
