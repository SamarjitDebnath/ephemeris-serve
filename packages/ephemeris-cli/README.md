# ephemeris-cli

Chat client for an [Ephemeris Serve](https://github.com/SamarjitDebnath/ephemeris-serve) inference server.

This is a **standalone HTTP client**. It never loads a model and imports nothing from the server, so it installs without torch, transformers, fastapi, or uvicorn — three small dependencies (`click`, `httpx`, `pyyaml`) and nothing else. It lives in the server's repository for convenience, but is versioned and installed as its own distribution.

## Install

```bash
pip install ephemeris-cli
# or, from a checkout of the monorepo:
pip install -e packages/ephemeris-cli
```

## Use

```bash
ephemeris start
ephemeris start --url https://ephemeris.example.com --max-tokens 128 --creativity creative
ephemeris config     # show resolved settings and where each came from
```

In the REPL: type a message and press Enter. `/model [name]` views or hot-swaps the server's model, `/creativity [preset|number]` adjusts sampling temperature for the next turn, `/exit` or Ctrl-D quits. Arrow keys edit the line, and history persists in `~/.ephemeris_history`.

## Configuration

The server address is never hardcoded. Resolution order, highest priority first:

1. `--url`, or `--host`/`--port` to override one part of the configured address
2. `EPHEMERIS_CLIENT_URL` in the real environment
3. a `.env` file (see below)
4. the file named by `EPHEMERIS_CLIENT_CONFIG`
5. `~/.config/ephemeris/client.yaml` (honors `XDG_CONFIG_HOME`)
6. the packaged default, `ephemeris_cli/client_config.yaml`

### `.env`

This distribution reads **its own** `.env`, never the server's configuration. `.env` files are consulted in this order, later ones winning:

1. `packages/ephemeris-cli/.env` — the package's own, for monorepo checkouts
2. `~/.config/ephemeris/.env`
3. `./.env` in the directory you run `ephemeris` from
4. the file named by `EPHEMERIS_CLIENT_ENV`

Only `EPHEMERIS_CLIENT_URL`, `EPHEMERIS_CLIENT_API_KEY` and `EPHEMERIS_CLIENT_CONFIG` are read. Every other entry is ignored — including the server's `HF_KEY` and `EPHEMERIS_SERVER_*` keys — so running the client from the server's repo root reads nothing of the server's, even though both find the same file.

```bash
cp .env-sample .env && chmod 600 .env
```

Real environment variables always beat `.env`, and command-line options beat both. `ephemeris config` prints every file consulted and which layer supplied each value.

```bash
mkdir -p ~/.config/ephemeris
cat > ~/.config/ephemeris/client.yaml <<'YAML'
client_config:
  defaults:
    base_url: "https://ephemeris.example.com"
    timeout_seconds: 120.0
YAML
chmod 600 ~/.config/ephemeris/client.yaml
```

### Upgrading from `ephemeris-serve`

The client was part of the server distribution and called `ephemeris-serve` before this split. Its two on-disk locations were renamed with it, and the old ones are still read when the new ones are absent, so nothing is lost on upgrade:

| Now | Previously |
| --- | --- |
| `~/.ephemeris_history` | `~/.ephemeris_serve_history` |
| `~/.config/ephemeris/` | `~/.config/ephemeris-serve/` |

Once a file exists at the new path, the old one is ignored. Move them at your leisure — or not at all.

## Authentication

If the server has API keys configured, send yours as `EPHEMERIS_CLIENT_API_KEY`, as `api_key` in the config file, or with `--api-key`. The environment variable wins, which keeps the credential off disk.

```bash
export EPHEMERIS_CLIENT_API_KEY=...
ephemeris start
```

`ephemeris start` probes one authenticated route before opening the REPL, so a missing or wrong key fails immediately and names which layer supplied it. Keys are shown masked, never in full.

Every variable this client reads is prefixed `EPHEMERIS_CLIENT_`. The server owns `EPHEMERIS_SERVER_*`, so the two never collide on a machine running both.
