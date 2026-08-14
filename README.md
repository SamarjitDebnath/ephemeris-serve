<p align="center">
  <img src="docs/assets/images/ephemeris-serve-logo.png" alt="Ephemeris Serve logo" width="200" style="border-radius: 24px;">
</p>

Ephemeris Serve
=====================

Ephemeris Serve is an LLM inference server with continuous scheduling, dynamic batching, and SSE streaming for token-level outputs.

**[Project site](https://samarjitdebnath.github.io/ephemeris-serve/) · [API reference](https://samarjitdebnath.github.io/ephemeris-serve/api.html) · [Technical documentation (wiki)](https://github.com/SamarjitDebnath/ephemeris-serve/wiki)**

Key features
------------
- Continuous scheduling and request queue for efficient CPU/GPU utilization
- Dynamic batching with both streaming and non-streaming batch inference paths
- Explicit request cancellation, timeouts, and cache-state fallback handling
- SSE (Server-Sent Events) streaming of decoded tokens for low-latency client rendering
- Metrics collection for queue latency, batch size, and token throughput
- Pluggable tokenizer and model loader (Hugging Face compatible)
- Health check and lightweight FastAPI-based HTTP API

Quick start
-----------
Prerequisites: Python 3.11+, a supported GPU (optional), and access to the Hugging Face Hub if you want authenticated model downloads.

Install dependencies:

```bash
python -m venv .venv
source .venv/bin/activate
pip install -U pip
pip install -e .
```

Run the server for development:

```bash
python main.py
# or with uvicorn directly
uvicorn main:app --host 0.0.0.0 --port 8000 --reload
```

Makefile
--------
This repository includes a `Makefile` with convenient targets for setup, running, testing, and maintenance. Common commands:

```bash
# Install dependencies with development extras
make dev

# Run server (development)
make run

# Run server (production)
make run-prod

# Run unit tests
make test-unit

# Format code
make format

# Lint the codebase
make lint

# Tail application logs
make logs

# E2E: sync deps, install the CLI entry point, and start the server
make build-ephemeris
```

Two distributions
-----------------
This is a monorepo containing **two independently installable packages**. The chat client is normally used from a different machine than the server, so it does not depend on any of the server's stack:

| Distribution | Source | Command | Installs |
| --- | --- | --- | --- |
| `ephemeris-serve` | repo root | `ephemeris-serve` | fastapi, uvicorn, torch, transformers, ... |
| `ephemeris-cli` | `packages/ephemeris-cli/` | `ephemeris` | click, httpx, pyyaml |

```bash
pip install -e .                        # server
pip install -e packages/ephemeris-cli   # chat client (no torch)
```

They share no code, no config files, and no environment variables. Each side owns its own prefix — `EPHEMERIS_SERVER_*` for the server, `EPHEMERIS_CLIENT_*` for the client — and its own `.env`: the server's at the repo root, the client's at `packages/ephemeris-cli/.env`. The client only ever reads `EPHEMERIS_CLIENT_*` entries, so even a single shared `.env` cannot leak one side's configuration into the other. A test enforces the boundary by importing the client in a clean interpreter and asserting no server module is reachable.

CLI chat client
----------------
`ephemeris start` (from the `ephemeris-cli` distribution) is a `click`-based REPL that talks to an already-running server over `/api/generate`'s SSE stream -- it never loads a model. The server is started by the other distribution's `ephemeris-serve` command.

```bash
# start the server in one terminal
ephemeris-serve serve
ephemeris-serve serve --model Qwen/Qwen2.5-0.5B   # pick a model without editing config.yaml

# chat with it from another terminal (or another machine)
ephemeris start
ephemeris start --url https://ephemeris.example.com --max-tokens 128 --creativity creative
```

The client's server address is **not hardcoded** — it is resolved from configuration, highest priority first:

1. `--url`, or `--host`/`--port` to override just one part of the configured address
2. the `EPHEMERIS_CLIENT_URL` environment variable
3. the client's own `.env` (`packages/ephemeris-cli/.env`, `~/.config/ephemeris/.env`, `./.env`, or `$EPHEMERIS_CLIENT_ENV`)
4. the file named by `EPHEMERIS_CLIENT_CONFIG`
5. `~/.config/ephemeris/client.yaml` (honors `XDG_CONFIG_HOME`)
6. the packaged default, `packages/ephemeris-cli/ephemeris_cli/client_config.yaml`

Set a permanent address once per machine:

```bash
mkdir -p ~/.config/ephemeris
cat > ~/.config/ephemeris/client.yaml <<'YAML'
client_config:
  defaults:
    base_url: "https://ephemeris.example.com"
    timeout_seconds: 120.0
YAML
```

`ephemeris config` prints the resolved address, which layer supplied it, and every file consulted.

Type a message and press Enter; `/exit`, `/quit`, or Ctrl-D ends the session. Each turn is an independent single-shot request -- the server doesn't retain conversation history between turns.

`--creativity` picks a friendly sampling-temperature preset (`deterministic`, `balanced`, `creative`, `high-freedom`) instead of a raw float; `--temperature <value>` is still available when you want exact control. Both the model and the creativity level can also be changed mid-session, without restarting the REPL: `/model [name]` views or hot-swaps the loaded model, and `/creativity [preset|number]` views or changes the sampling temperature for the next turn onward.
<p align="center">
  <img src="docs/assets/images/chat-cli1.png" alt="Ephemeris Serve CLI startup splash" width="700"><br>
  <img src="docs/assets/images/chat-cli2.png" alt="Ephemeris Serve CLI /model command" width="700"><br>
  <img src="docs/assets/images/chat-cli3.png" alt="Ephemeris Serve CLI chat session" width="700">
</p>

Configuration
-------------
- Server settings are under the `settings` package and `config.yaml`.
- CLI client settings (which server to talk to) are separate — see `packages/ephemeris-cli/ephemeris_cli/client_config.yaml` and the resolution order above. The client deliberately does not import the server's settings module, so it stays installable and runnable anywhere.
- Secrets (e.g., Hugging Face token) are read from `settings/secret_settings` (see `settings/settings.py`).
- For production, disable `--reload` and tune `workers` in your process manager (or container runtime).

Authentication
--------------
The `/api` routes accept an API key as `Authorization: Bearer <key>`, in two tiers:

| Tier | Environment variable | Covers |
| --- | --- | --- |
| Ordinary | `EPHEMERIS_SERVER_API_KEYS` | `/api/generate`, `/api/generate_batch`, `GET /api/model`, `/api/metrics` |
| Admin | `EPHEMERIS_SERVER_ADMIN_API_KEYS` | additionally `POST /api/model` |

Both are comma-separated lists, so keys rotate by adding the new one before removing the old. `POST /api/model` is gated separately because it makes the server download and load an arbitrary Hugging Face repo. `/health` and `/` stay open.

**With both variables empty, authentication is disabled and every route is open** — that is the local-development default and keeps `make run` and the test suite working with no setup. The server logs a warning at startup when it is in that state. Any deployment reachable from outside localhost must set at least `EPHEMERIS_SERVER_API_KEYS`; see [`deploy/README.md`](deploy/README.md).

Clients send the key via `EPHEMERIS_CLIENT_API_KEY`, `api_key` in the client config, or `--api-key` (in that order of precedence — the environment variable wins, keeping the credential off disk).

Deployment
----------
[`deploy/README.md`](deploy/README.md) is the full walkthrough: systemd unit, API keys, nginx, then pointing clients at the result.

In production, uvicorn binds loopback and nginx is the only public listener — it terminates TLS, enforces body-size and rate limits before a request reaches Python, and gives clients one stable address. The config lives in [`deploy/nginx/ephemeris-serve.conf`](deploy/nginx/ephemeris-serve.conf); [`deploy/nginx/README.md`](deploy/nginx/README.md) covers install, TLS, and verification. [`deploy/systemd/ephemeris-serve.service`](deploy/systemd/ephemeris-serve.service) supervises the process.

```bash
sudo cp deploy/nginx/ephemeris-serve.conf /etc/nginx/sites-available/
sudo ln -s /etc/nginx/sites-available/ephemeris-serve.conf /etc/nginx/sites-enabled/
sudo nginx -t && sudo nginx -s reload

make run-prod   # uvicorn on 127.0.0.1:8000, proxy headers trusted
make smoke-nginx  # verify the proxy config (needs nginx; no model required)
```

The proxy listens on `:8080` and forwards to uvicorn on `127.0.0.1:8000`. `POST /api/generate` gets `proxy_buffering off` so SSE frames are never held back — verified against nginx 1.31.3, where proxied frames arrive with the same timing as a direct connection. `make run-prod` and `ephemeris-serve serve` pass `--proxy-headers --forwarded-allow-ips 127.0.0.1`, so the app reads the real client address and scheme from the proxy's forwarded headers.

API
---
Base URL: `http://<host>:8000`

Every `/api` route below requires `Authorization: Bearer <key>` once API keys are configured (see Authentication above); `/` and `/health` never do.

- GET `/` — root welcome message
- GET `/health` — returns JSON health status
- POST `/api/generate` — stream model output via SSE
- POST `/api/generate_batch` — non-streaming batched generation, returns full text output for multiple requests
- GET `/api/metrics` — exposes queue latency, batch size, and throughput metrics

Generate request schema (JSON):

```json
{
  "prompt": "Your prompt text here",
  "max_tokens": 64,
  "temperature": 0.7
}
```

Example (curl SSE stream):

```bash
curl -N -H "Accept: text/event-stream" \
  -H "Content-Type: application/json" \
  -X POST \
  -d '{"prompt":"Hello world","max_tokens":50,"temperature":0.7}' \
  http://localhost:8000/api/generate
```

Example (curl batch generation):

```bash
curl -H "Content-Type: application/json" \
  -X POST \
  -d '{"requests": [{"prompt": "Hello world", "max_tokens": 50, "temperature": 0.7}]}' \
  http://localhost:8000/api/generate_batch
```

Example (curl metrics):

```bash
curl http://localhost:8000/api/metrics
```

Notes:
- `/api/generate` enqueues an `InferenceRequest` and returns a streaming `text/event-stream` response of decoded tokens.
- `/api/generate_batch` accepts one or more batch requests and returns aggregated text output once generation completes.
- `/api/metrics` returns server metrics for queue latency, batch size, and throughput.

Development
-----------
- Run tests with `pytest`.
- Use the development extras from `pyproject.toml` for linting and formatting.

Testing
-------
Run the test suite:

```bash
pytest -q
```

Project layout
--------------
- `api/` — FastAPI routes and server factory (`api/server.py`, `api/routes.py`)
- `engine/` — model loading and generation orchestration (`model_loader.py`, `generator.py`)
- `scheduler/` — continuous scheduler and request queue
- `streaming/` — SSE stream helpers and token streaming
- `tokenizer/` — tokenizer service abstraction
- `settings/` — configuration and secrets
- `schemas/` — Pydantic request/response models
- `tests/` — unit and integration tests
- `cli/` — `ephemeris-serve` CLI (`main.py`), its client configuration (`config.py`, `client_config.yaml`), and the splash logo
- `deploy/` — deployment configuration: `deploy/nginx/` (reverse proxy), `deploy/systemd/` (service unit), and `deploy/README.md` (the walkthrough)
- `docs/` — microsite (landing page + API reference), served directly by GitHub Pages from `main` / `docs`
- `wiki/` — source of truth for the [GitHub wiki](https://github.com/SamarjitDebnath/ephemeris-serve/wiki) (see below)

Documentation
-------------
The in-depth technical documentation lives in the [GitHub wiki](https://github.com/SamarjitDebnath/ephemeris-serve/wiki). The wiki is **generated from `wiki/` in this repository**, so it is version-controlled and reviewable in pull requests like any other file.

To change a wiki page, edit the matching file under `wiki/` and publish it:

```bash
make wiki-sync
```

Edits made directly in the GitHub wiki UI are overwritten by the next sync. The first sync requires the wiki to exist: enable **Settings → Features → Wikis**, then create any one page in the wiki UI so GitHub provisions the `.wiki.git` repository.

Contributing
------------
- Open an issue or submit a PR with tests and concise changes.
- Follow existing code style; use `black` / `isort` for formatting.

License
-------
Refer to `pyproject.toml` for package metadata and the `LICENSE` file for the applicable license terms.

Useful files
------------
- [main.py](main.py) — development entrypoint
- [api/server.py](api/server.py) — FastAPI app factory and lifespan hooks
- [api/routes.py](api/routes.py) — primary API routes (including `/api/generate`)
- [schemas/schemas.py](schemas/schemas.py) — request/response models
- [pyproject.toml](pyproject.toml) — dependencies and packaging
