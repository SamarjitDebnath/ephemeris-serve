# Deploying Ephemeris Serve

Three pieces, in this order:

1. **systemd** (`systemd/ephemeris-serve.service`) — keeps the process running, supplies its environment.
2. **API keys** (`api/auth.py`) — nothing is authenticated by default.
3. **nginx** (`nginx/`) — TLS, the only public listener, SSE-safe proxying.

Then point clients at it.

## 0. Before you start

Capacity is one model on one device, batching up to 8 concurrent requests. `--workers > 1` is **not** a scaling knob here: `POST /api/model` swaps the model in one process only, so workers would diverge. Size expectations accordingly, and run one worker.

## 1. Install and supervise

```bash
sudo useradd --system --home /opt/ephemeris-serve ephemeris
sudo mkdir -p /opt/ephemeris-serve /var/lib/ephemeris-serve/huggingface /etc/ephemeris-serve
sudo chown -R ephemeris:ephemeris /opt/ephemeris-serve /var/lib/ephemeris-serve

# deploy the code, then build its venv
sudo -u ephemeris git clone https://github.com/SamarjitDebnath/ephemeris-serve /opt/ephemeris-serve
cd /opt/ephemeris-serve && sudo -u ephemeris uv sync

sudo cp deploy/systemd/ephemeris-serve.service /etc/systemd/system/
sudo systemctl daemon-reload
```

`WorkingDirectory=/opt/ephemeris-serve` in the unit is load-bearing: `settings/settings.py` reads `settings/config.yaml` and the logger writes `logs/app.log`, both **relative** paths. Started from anywhere else the server won't find its config.

`TimeoutStartSec=900` covers the first boot, when the model is downloaded before the socket accepts traffic.

## 2. API keys

Generate them:

```bash
python -c "import secrets; print(secrets.token_urlsafe(32))"
```

Write the environment file (root-owned, `0600` — it holds every credential):

```bash
sudo install -m 600 /dev/null /etc/ephemeris-serve/env
sudo tee /etc/ephemeris-serve/env >/dev/null <<'ENV'
HF_KEY=hf_...
EPHEMERIS_SERVER_API_KEYS=client-key-one,client-key-two
EPHEMERIS_SERVER_ADMIN_API_KEYS=admin-key
ENV

sudo systemctl enable --now ephemeris-serve
journalctl -u ephemeris-serve -f
```

Two tiers:

| Tier | Variable | Covers |
| --- | --- | --- |
| Ordinary | `EPHEMERIS_SERVER_API_KEYS` | `/api/generate`, `/api/generate_batch`, `GET /api/model`, `/api/metrics` |
| Admin | `EPHEMERIS_SERVER_ADMIN_API_KEYS` | additionally `POST /api/model` |

`POST /api/model` is separated because it makes the server download and load an arbitrary Hugging Face repo — the one route where an ordinary client could fill the disk, exhaust memory, or park the server in a drain-and-reload cycle. Admin keys satisfy the ordinary tier too, so one admin key is enough for an operator.

Rotation: add the new key to the list, redeploy clients, then remove the old one. Both are accepted meanwhile.

`/health` and `/` stay unauthenticated so the proxy and any uptime monitor can reach them.

> **Authentication is off until you set a key.** With both variables empty every `/api` route is open, which is what keeps `make run` and the tests working with no setup. The server logs a warning at startup in that state — check `journalctl` for it after the first boot, because an open `POST /api/model` on a public address is the worst failure mode this project has.

Verify:

```bash
curl -s -o /dev/null -w '%{http_code}\n' http://127.0.0.1:8000/api/metrics                                  # 401
curl -s -o /dev/null -w '%{http_code}\n' -H 'Authorization: Bearer client-key-one' \
  http://127.0.0.1:8000/api/metrics                                                                         # 200
curl -s -H 'Authorization: Bearer client-key-one' -X POST -H 'Content-Type: application/json' \
  -d '{"model_name":"distilgpt2"}' http://127.0.0.1:8000/api/model                                          # 403
```

## 3. nginx

See [`nginx/README.md`](nginx/README.md) for install, the SSE buffering rules, and the TLS template. In short: it terminates TLS, listens publicly, forwards to `127.0.0.1:8000`, and must keep `proxy_buffering off` on `/api/generate`.

Authentication is **not** duplicated in nginx. One mechanism, checked in the app, applies whether a request arrives through the proxy or not — two overlapping schemes would mean two credentials for every client and a gap the day someone reaches uvicorn directly.

## 4. Point clients at it

```bash
mkdir -p ~/.config/ephemeris
cat > ~/.config/ephemeris/client.yaml <<'YAML'
client_config:
  defaults:
    base_url: "https://ephemeris.example.com"
YAML
chmod 600 ~/.config/ephemeris/client.yaml

export EPHEMERIS_CLIENT_API_KEY=client-key-one   # prefer the env var over the file
ephemeris config                    # shows the address, the masked key, and where each came from
ephemeris
```

The key can also live as `api_key` in that YAML file, but the environment variable wins and keeps the credential off disk. `ephemeris start` probes one authenticated route before opening the REPL, so a wrong or missing key fails immediately with a clear message rather than on the first chat turn.

## What is still missing

- **No per-key rate limiting or quotas.** nginx limits by client address only; one key can spend as much as it likes from many addresses.
- **No audit trail of who generated what.** Rejections are logged; successful requests are not attributed to a key.
- **No multi-host story.** A second server means a second model in memory and independent `POST /api/model` state.
- **Keys are compared in-process from an env var.** No revocation without a restart, no expiry.
