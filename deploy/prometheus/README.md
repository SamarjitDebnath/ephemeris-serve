# Prometheus scraping

The server exposes raw counters, gauges, and histograms at `GET /metrics` in
the Prometheus exposition format. This is separate from `GET /api/metrics`,
which returns rolling averages as JSON and is what the CLI consumes. Neither is
derived from the other — see `metrics/prometheus.py` for why.

## Enabling

```bash
pip install -e ".[metrics]"
```

```yaml
# settings/config.yaml
metrics_config:
   defaults:
      prometheus_enabled: true
      require_auth: true
```

When `prometheus_enabled` is false the route is not registered at all, so a
scraper sees a clean 404 rather than an endpoint that reports being disabled.

## Authentication

`require_auth: true` keeps `/metrics` behind the ordinary API key. Prometheus
supports this:

```yaml
scrape_configs:
  - job_name: ephemeris
    authorization:
      type: Bearer
      credentials: <an EPHEMERIS_SERVER_API_KEYS value>
    static_configs:
      - targets: ["127.0.0.1:8000"]
```

The more common pattern is `require_auth: false` with the port simply not
routed publicly. Either is fine; what is not fine is picking one by accident.
Metrics leak traffic volume and model identity.

## Multiple workers

`make run-prod` runs `--workers 4`, and each worker keeps its own metrics.
Without multiprocess mode a scrape returns whichever worker answered.

```bash
export PROMETHEUS_MULTIPROC_DIR=/var/lib/ephemeris/prom
```

Each worker then writes mmap files there and the exporter aggregates them at
scrape time.

**The directory must be empty at startup.** Files left by a previous run are
picked up by the exporter and inflate every counter — it produces wrong
numbers, not an error. `api/server.py` clears it during lifespan, so this is
handled as long as the variable is set before the server starts.

Counters aggregate across workers on their own. Gauges do not, and each needs
an explicit mode:

| Metric | Mode | Why |
| --- | --- | --- |
| `ephemeris_batch_size` | `livesum` | A per-worker count; the pool-wide figure is the sum over live workers. |
| `ephemeris_batch_occupancy_ratio` | `livemax` | A ratio. Summing ratios is meaningless; the max answers "is any worker saturated". |
| `ephemeris_kv_cache_utilization_ratio` | `livemax` | Same reasoning. |

`live*` rather than plain `sum`/`max` so dead workers stop contributing.

## Metrics

| Metric | Type | Reading it |
| --- | --- | --- |
| `ephemeris_queue_latency_seconds` | histogram | Time between enqueue and admission. Climbing p99 with flat batch occupancy means the queue is starved of slots, not of throughput. |
| `ephemeris_token_throughput_per_second` | histogram | Tokens per second per scheduler step. |
| `ephemeris_batch_size` | gauge | Requests in the last batch. |
| `ephemeris_batch_occupancy_ratio` | gauge | Sustained near 1.0 means `max_batch_size` is the binding constraint. |
| `ephemeris_kv_cache_utilization_ratio` | gauge | Sustained near 1.0 means the block pool is about to grow; near 0 after a busy period means an idle trim is due. |
| `ephemeris_timeout_evictions_total` | counter | Requests dropped at their deadline. Any sustained rate here is user-visible failure. |
| `ephemeris_cancelled_evictions_total` | counter | Clients that disconnected mid-generation. |
| `ephemeris_kv_blocks_reclaimed_total` | counter | Blocks returned by an idle trim. Flat while memory grows means the trim never fires — check the dwell and hysteresis settings. |

All are labelled `path="batch"` or `path="streaming"`, matching the two
top-level keys in the JSON endpoint.
