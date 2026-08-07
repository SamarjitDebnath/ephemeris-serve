INTERNAL_ERROR_MESSAGE = "Internal server error"
"""Generic message sent to clients for an unexpected, otherwise-unhandled
failure (e.g. a CUDA/MPS OOM during generation).

Deliberately generic: raw exception text (stack-trace-flavored messages,
memory sizes, env var hints, file paths, ...) must never reach a client.
Full details are always logged server-side via `logger.exception`/
`logger.warning` at the point of failure -- this constant is only for what
gets sent back over the wire (an SSE `error` event's `data`, or an
`HTTPException`'s `detail`).
"""
