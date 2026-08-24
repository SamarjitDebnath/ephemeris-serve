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
- Accumulates every non-special token ID into a running `tokens` list and decodes **incrementally**: `prefix_text = decode(tokens[prefix_offset:read_offset])` and `new_text = decode(tokens[prefix_offset:])`, with the delta taken as their difference. Two overlapping decodes rather than one per token, because `decode(a + b)` is not `decode(a) + decode(b)` for byte-level BPE -- leading-space handling and multi-byte sequences straddling a token boundary both differ at the seam. Both spans stay small because `prefix_offset` advances whenever a delta is emitted, so a long stream no longer costs a full-history decode per token. A trailing Unicode replacement character (`�`) means the token completed only part of a character: everything is held until the next token finishes it, which is the same signal the old strip-trailing-`�` loop used, now folded into the algorithm.
- **Stop-sequence check**, run against the accumulated decoded text before any buffering decision: if `req.stop_sequences` is non-empty and `find_stop_index()` finds a match, yields only the not-yet-emitted text up to the match (if non-empty) and returns -- ending the stream without ever yielding the stop sequence itself or anything after it, even if it spans the last few tokens the scheduler already pushed onto the queue before it noticed the same match and stopped generating. Only the region a *new* match could occupy is searched (`len(decoded_text) - len(new_delta) - req.max_stop_length + 1` onward): a sequence ending inside the new delta starts at most `max_stop_length - 1` characters before it, and anything earlier would already have matched on a previous token and returned.
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
- `snapshot()`: returns `average_queue_latency_ms` (mean of `queue_latencies`, converted to ms, or `None` if empty), `average_batch_size`, `average_token_throughput_per_sec`, `average_batch_occupancy`, `average_cache_utilization`, a `_samples` count for each of the five deques, and the two raw eviction counters (`timeout_evictions`, `cancelled_evictions`). Plus `kv_blocks_reclaimed` (blocks returned to the allocator by an idle paged-cache trim) and `average_queue_latency_ms_by_class`, a per-scheduling-class breakdown -- without the split, a fairness improvement for short requests and a regression for long ones average out to nothing in the aggregate figure.

Two independent singletons -- `metrics` (batch path) and `streaming_metrics` (streaming path) -- so `/generate_batch` responses aren't polluted by concurrent SSE traffic and vice versa. `metrics` is fed by `scheduler/batch_scheduler.py`; `streaming_metrics` is fed by `scheduler/continuous_scheduler.py`. Since `batch_occupancies`/`cache_utilizations`/the eviction counters are only ever recorded on `streaming_metrics`, they stay at their empty/zero defaults in the batch path's `metrics.snapshot()`. `GET /api/metrics` returns `{"batch": metrics.snapshot(), "streaming": streaming_metrics.snapshot()}`.

`summarize_batch_response_metrics(batch_requests) -> {"queue_latency_ms": ..., "token_throughput_per_sec": ...}`:
- Used by `/generate_batch` to assemble its per-response metrics fields. Queue latency is averaged over *this batch's own* requests (`req.queue_latency_ms` for each, converted to ms) -- a per-response figure, unlike throughput. `token_throughput_per_sec` is instead pulled from the rolling batch-path average (`metrics.snapshot()["average_token_throughput_per_sec"]`), since throughput is a per-engine-call measurement rather than something meaningfully computed per individual response.

### `metrics/prometheus.py`

Optional Prometheus export, enabled by `metrics_config.prometheus_enabled` and the `metrics` extra (`pip install -e ".[metrics]"`). Served at `GET /metrics` in the exposition format, registered on the app rather than the `/api` router because that is the path scrapers default to; when disabled the route does not exist, so a scraper sees a clean 404 rather than an endpoint reporting itself unavailable.

Emitted **alongside** the deques above, not derived from them. `snapshot()` averages over the last 1000 samples -- a window whose *duration* varies with traffic, spanning 100 seconds at 10 req/s and one second at 1000 -- and handing a time-series backend a pre-averaged number over a variable window discards exactly the freedom it exists to provide. `GET /api/metrics` and its JSON are untouched, because the CLI consumes that shape.

Every metric is labelled `path="batch"` or `path="streaming"`, mirroring the two top-level keys in the JSON endpoint. Histograms cover queue latency and token throughput; gauges cover batch size, batch occupancy and cache utilization; counters cover the two eviction paths and reclaimed KV blocks.

#### What a scrape looks like

![Prometheus graphing ephemeris_batch_occupancy_ratio scraped from a running Ephemeris Serve instance](https://raw.githubusercontent.com/SamarjitDebnath/ephemeris-serve/main/docs/assets/images/prometheus-metrics.png)

`ephemeris_batch_occupancy_ratio` scraped every two seconds while a burst-and-idle
load ran against the server. Each spike is the batch filling to `max_batch_size`
(occupancy `1.0`); each trough is the scheduler draining as that burst finishes,
before the next one arrives. The series is labelled
`{instance="127.0.0.1:8000", job="ephemeris", path="streaming"}` -- `path`
distinguishing the streaming scheduler from the batch one, exactly as it does in
the JSON endpoint's two top-level keys.

The image is served from the repository rather than the wiki: `scripts/sync_wiki.sh`
publishes `wiki/*.md` only, so assets live in `docs/assets/images/` and are linked
by raw URL, keeping one copy rather than two that can drift.

Under `--workers N` each worker keeps its own metrics, handled by `PROMETHEUS_MULTIPROC_DIR`. Counters aggregate across workers on their own; **gauges do not**, and each carries an explicit mode -- `livesum` for `ephemeris_batch_size` (a per-worker count whose pool-wide meaning is the sum over live workers), `livemax` for the two ratios (summing ratios is meaningless; the max answers "is any worker saturated"). Picking the wrong mode produces plausible, incorrect numbers rather than an error. The directory is cleared at startup by `prepare_multiprocess_dir()`, because files left by a previous run are picked up by the exporter and silently inflate every counter. See `deploy/prometheus/README.md` for scrape configuration and per-metric interpretation.
