# Streaming and Metrics

## Streaming

### `streaming/stream_manager.py`

Converts raw token IDs from a request's queue into decoded SSE stream payloads, buffering to avoid noisy subword fragments and independently enforcing stop sequences so they never reach the client.

Imports:
- `asyncio`, `AsyncGenerator`, `Union`
- `InferenceRequest` from `scheduler.request`
- `tokenizer_service` from `tokenizer.tokenizer_service`
- `find_stop_index` from `utils.stop_sequences`
- `logging_settings` from `settings.settings`
- `setup_logger` from `logger`

`stream_response(req: InferenceRequest) -> AsyncGenerator[str | dict, None]`:
- Loops reading `req.queue` until the sentinel `"[DONE]"`.
- A `("[ERROR]", message)` tuple sentinel (pushed when a request is evicted for timing out or failing server-side) yields a single `{"event": "error", "data": message}` frame and ends the stream instead of following the `"[DONE]"` path.
- Lazily loads the tokenizer if needed; skips special token IDs (via `tokenizer_service.tokenizer.all_special_ids`).
- Accumulates every non-special token ID into a running `tokens` list and re-decodes the *entire* sequence each time (`tokenizer_service.decode(tokens)`) -- this, not per-token decoding, is what correctly handles multi-byte characters split across token boundaries. Trailing Unicode replacement characters (`�`, an incomplete UTF-8 sequence) are stripped from the decoded text before anything else happens.
- **Stop-sequence check**, run against the full decoded text before any buffering decision: if `req.stop_sequences` is non-empty and `find_stop_index()` finds a match, yields only the not-yet-emitted text up to the match (`clean_text[len(yielded_text):stop_idx]`, if non-empty) and returns -- ending the stream without ever yielding the stop sequence itself or anything after it, even if it spans the last few tokens the scheduler already pushed onto the queue before it (itself) noticed the same match and stopped generating.
- If no stop match: computes `delta = clean_text[len(yielded_text):]` and only actually yields it (`should_emit`) once the delta ends in whitespace, ends in `.,;:!?`, or has reached a 16-character buffer threshold -- otherwise the delta is held back (logged, not yielded) until a later token completes a natural boundary.

Relationship with API:
- `api/routes.py` returns `EventSourceResponse(stream_response(request))`; the client receives each yielded text fragment as one SSE `data:` frame (the SSE library adds one mandatory leading space after `data:`, which is exactly the "single leading space" a spec-compliant consumer must strip -- see the CLI's `_stream_reply` for the client-side half of this).

---

## Metrics

### `metrics/metrics.py`

Rolling in-memory metrics tracking, exposed via `GET /api/metrics` (`api/routes.py`).

`BatchMetrics(max_samples=1000)`:
- `queue_latencies`/`batch_sizes`/`token_throughputs`: `collections.deque(maxlen=max_samples)` -- a fixed-size rolling window, so metrics reflect recent activity rather than an ever-growing all-time average.
- `record_queue_latency(latency_seconds)`: appends if `>= 0`.
- `record_batch_size(batch_size)`: appends if `> 0`.
- `record_token_throughput(tokens, elapsed_seconds)`: appends `tokens / elapsed_seconds` if `elapsed_seconds > 0` and `tokens >= 0`.
- `batch_occupancies`/`cache_utilizations`: fed only by the continuous scheduler -- `record_batch_occupancy(active_count, max_batch_size)` appends `active_count / max_batch_size` (skipped if `max_batch_size <= 0`); `record_cache_utilization(used_blocks, capacity_blocks)` appends `used_blocks / capacity_blocks` (skipped if `capacity_blocks <= 0`).
- `timeout_evictions`/`cancelled_evictions`: plain counters (not deques), also fed only by the continuous scheduler -- `record_timeout_eviction()`/`record_cancelled_eviction()` each increment by 1.
- `snapshot()`: returns `average_queue_latency_ms` (mean of `queue_latencies`, converted to ms, or `None` if empty), `average_batch_size`, `average_token_throughput_per_sec`, `average_batch_occupancy`, `average_cache_utilization`, a `_samples` count for each of the five deques, and the two raw eviction counters (`timeout_evictions`, `cancelled_evictions`).

Two independent singletons -- `metrics` (batch path) and `streaming_metrics` (streaming path) -- so `/generate_batch` responses aren't polluted by concurrent SSE traffic and vice versa. `metrics` is fed by `scheduler/batch_scheduler.py`; `streaming_metrics` is fed by `scheduler/continuous_scheduler.py`. Since `batch_occupancies`/`cache_utilizations`/the eviction counters are only ever recorded on `streaming_metrics`, they stay at their empty/zero defaults in the batch path's `metrics.snapshot()`. `GET /api/metrics` returns `{"batch": metrics.snapshot(), "streaming": streaming_metrics.snapshot()}`.

`summarize_batch_response_metrics(batch_requests) -> {"queue_latency_ms": ..., "token_throughput_per_sec": ...}`:
- Used by `/generate_batch` to assemble its per-response metrics fields. Queue latency is averaged over *this batch's own* requests (`req.queue_latency_ms` for each, converted to ms) -- a per-response figure, unlike throughput. `token_throughput_per_sec` is instead pulled from the rolling batch-path average (`metrics.snapshot()["average_token_throughput_per_sec"]`), since throughput is a per-engine-call measurement rather than something meaningfully computed per individual response.
