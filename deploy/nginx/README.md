# nginx reverse proxy

`ephemeris-serve.conf` fronts uvicorn with nginx: uvicorn binds loopback only, and nginx is the single public listener.

## Why a proxy

- **TLS termination** — uvicorn serves plain HTTP; nginx holds the certificate.
- **A stable public address** — the CLI and other clients point at the proxy, not at whichever port uvicorn happens to use.
- **Request limits before Python** — oversized bodies and a per-address rate limit are rejected at the proxy rather than occupying the scheduler queue.
- **Slow-client absorption** — nginx buffers slow readers on the ordinary JSON endpoints, freeing uvicorn workers sooner.

## SSE and buffering

`POST /api/generate` streams Server-Sent Events, so the `/api/generate` location disables response buffering:

```nginx
proxy_buffering off;
proxy_cache off;
gzip off;
add_header X-Accel-Buffering no;
proxy_read_timeout 300s;
```

**Measured, not assumed:** on nginx 1.31.3 this config delivers SSE frames with the same timing as a direct connection to uvicorn — 10 frames, 2.71s spread, 0.30s median gap, identical both ways. Re-running with `proxy_buffering on` and no upstream hint header produced *the same* timing, so the common "nginx breaks SSE" failure did not reproduce on this version: nginx forwards each chunked read rather than waiting to fill a buffer. Treat these directives as insurance rather than a fix for an observed break. They earn their place if gzip is ever widened to cover `text/event-stream`, a `proxy_cache` is added, or an older/differently-built nginx coalesces small chunks.

Note also that `sse_starlette` sets `X-Accel-Buffering: no` on every `EventSourceResponse`, and nginx honors that from upstream — so buffering is already disabled for this endpoint even without the directives above.

`proxy_read_timeout` must comfortably exceed `streaming_request_timeout_seconds` in `settings/config.yaml` (60s by default), since an SSE connection is idle between tokens. The `/api/` location uses a 600s timeout because `POST /api/model` drains in-flight work and then loads — often downloads — an entire model.

## Install

```bash
# Debian / Ubuntu
sudo cp deploy/nginx/ephemeris-serve.conf /etc/nginx/sites-available/
sudo ln -s /etc/nginx/sites-available/ephemeris-serve.conf /etc/nginx/sites-enabled/

# RHEL / Alpine
sudo cp deploy/nginx/ephemeris-serve.conf /etc/nginx/conf.d/

# macOS (Homebrew)
cp deploy/nginx/ephemeris-serve.conf /opt/homebrew/etc/nginx/servers/

sudo nginx -t && sudo nginx -s reload
```

`nginx -t` validates the file in place; running it directly against this file (`nginx -t -c deploy/nginx/ephemeris-serve.conf`) fails, because a site config has no surrounding `http { }` block of its own.

Then start the app on loopback:

```bash
make run-prod
# or
ephemeris-serve serve --host 127.0.0.1 --port 8000
```

`serve` passes `--proxy-headers` with `--forwarded-allow-ips 127.0.0.1` by default, so the app sees the real client address and scheme from `X-Forwarded-For`/`X-Forwarded-Proto` instead of nginx's own. Keep uvicorn bound to `127.0.0.1` in production — the trust in those headers is only safe while nothing but the local proxy can reach that port.

Verify end to end:

```bash
curl http://127.0.0.1:8080/health                                          # {"status":"healthy"}

# tokens should arrive progressively, not all at once when generation ends
curl -N -H "Accept: text/event-stream" -H "Content-Type: application/json" \
  -H "Authorization: Bearer <key>" \
  -X POST -d '{"prompt":"Hello","max_tokens":50}' \
  http://127.0.0.1:8080/api/generate
```

## Smoke test

```bash
make smoke-nginx
```

Starts a stub upstream (no model, so it runs in seconds) plus a private nginx instance running this exact config in a temp prefix, asserts the proxy's behavior, then tears both down. Exits non-zero on any failure, so it works in CI. Needs `nginx` on PATH and ports 8000/8080 free; it never touches a system nginx install.

The stub deliberately exposes `/api/generate_no_hint`, an SSE route *without* the `X-Accel-Buffering: no` header that `sse_starlette` normally sets — nginx honors that header from upstream, which would otherwise mask a missing `proxy_buffering off` and let the config pass on a technicality.

The proxy's other behaviors, each verified against this config:

| Check | Expected |
| --- | --- |
| 2MB request body | `413` (`client_max_body_size 1m`) |
| 40 rapid requests from one address | roughly the first 20 pass, then `503`s (`rate=10r/s`, `burst=20`) |
| `Authorization` header | reaches the app unchanged |
| `X-Forwarded-For` / `X-Forwarded-Proto` | set, and visible to the app via `--proxy-headers` |

## Pointing the CLI at it

The CLI reads its address from configuration, not a hardcoded default — the packaged `packages/ephemeris-cli/ephemeris_cli/client_config.yaml` already points at the proxy's `http://127.0.0.1:8080`. For a real deployment, set it once per machine:

```bash
mkdir -p ~/.config/ephemeris
cat > ~/.config/ephemeris/client.yaml <<'YAML'
client_config:
  defaults:
    base_url: "https://ephemeris.example.com"
YAML

ephemeris config   # shows the resolved address and where it came from
```

## TLS

The bottom of `ephemeris-serve.conf` carries a commented HTTP→HTTPS redirect plus a TLS `server` block. Obtain a certificate (certbot or otherwise), uncomment those, and move the `location` blocks in unchanged.

## Notes and limits

- `limit_req_zone` is keyed on `$binary_remote_addr` at 10r/s with `burst=20`. Tune it to your traffic; it is a blunt guard, not a quota system. Note the cutoff is not clean: `limit_req` is a leaky bucket that refills at the configured rate while the burst drains, so a client over the limit sees *intermittent* successes among the `503`s rather than a hard stop. A measured run of 40 rapid requests returned 22 allowed / 18 limited, with one success appearing well after the first rejections.
- `upstream ... keepalive 32` sizes reusable upstream connections. Each open SSE stream holds one, so raise it if you expect more concurrent streams than that.
- `POST /api/model` only swaps the model in the process that handles the request. With `--workers > 1` behind this proxy, nginx will route the request to one worker and the others keep serving the old model.

## Rate limiting: two layers

`limit_req_zone $binary_remote_addr` in `ephemeris-serve.conf` limits by
**client address**, at the edge, before a request reaches Python. It is the
right place to absorb connection floods.

`rate_limit_config` in `settings/config.yaml` limits by **API key** (falling
back to client address when auth is disabled), inside the app. It is the right
place for multi-tenant quota, because the tenant is the key, not the address --
two tenants behind one NAT share an address, one tenant across ten machines
does not. It also charges `/api/generate_batch` its true cost: a batch of 32
sub-requests spends 32 tokens, where nginx sees one request.

Keep both. Neither replaces the other, and the app-side limit is the only one
that exists when the server runs without this nginx in front of it.

Note the app-side limit is **per worker process**. `make run-prod` runs
`--workers 4`, so the effective limit is four times the configured numbers.
