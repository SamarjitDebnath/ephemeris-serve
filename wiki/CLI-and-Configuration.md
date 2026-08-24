# CLI and Configuration

## CLI Layer

The repository ships **two distributions with two separate command-line entry points**, because the chat client is meant to be installed on machines that will never run a model:

| Distribution | Source | Command | Dependencies |
| --- | --- | --- | --- |
| `ephemeris-serve` | repo root (`api/`, `engine/`, `scheduler/`, ...) | `ephemeris-serve` | fastapi, uvicorn, torch, transformers, ... |
| `ephemeris-cli` | `packages/ephemeris-cli/` | `ephemeris start` | click, httpx, pyyaml |

They share no code, no configuration files, and no environment variables. The client speaks HTTP to the server and imports nothing from it -- enforced by a test that imports the client package in a clean interpreter and asserts none of `torch`/`transformers`/`fastapi`/`uvicorn`/`settings`/`api` ended up in `sys.modules`.

### `api/cli.py` (server distribution)

`click`-based CLI, installed as the `ephemeris-serve` console script. One subcommand:

`ephemeris-serve serve` -- **runs the server itself**:
- Options: `--model` (HF repo id, overrides `settings/config.yaml`'s `model_name` for this run), `--host` (default `0.0.0.0`), `--port` (default `8000`), `--workers` (default `1`), `--reload/--no-reload` (default off), `--proxy-headers/--no-proxy-headers` (default on), `--forwarded-allow-ips` (default `127.0.0.1`).
- If `--model` is given, sets `os.environ["EPHEMERIS_SERVER_MODEL_NAME"]` *before* calling `uvicorn.run(...)` -- an env var rather than an in-process settings mutation, so it's correctly inherited even when `--workers > 1` makes uvicorn spawn fresh worker processes that re-import `settings.settings` from scratch (see `ModelSetting` in [Configuration](#configuration)).
- `--proxy-headers`/`--forwarded-allow-ips` are passed straight to `uvicorn.run()`, so the app reads the real client address and scheme from `X-Forwarded-For`/`X-Forwarded-Proto` when it sits behind the nginx reverse proxy (see [Reverse Proxy](Operations#reverse-proxy) below). Trusting those headers is only safe while nothing but the local proxy can reach the port, hence the loopback-only default allow-list.
- Calls `uvicorn.run("api.server:app", host=host, port=port, workers=workers, reload=reload, proxy_headers=..., forwarded_allow_ips=...)` -- functionally equivalent to `python main.py`, but with these as CLI flags instead of hardcoded values.

### `packages/ephemeris-cli/ephemeris_cli/main.py` (client distribution)

`click`-based CLI, installed as the `ephemeris start` console script. Two subcommands under one `@click.group()`.

`ephemeris start` -- **REPL chat client** against an already-running server (does not load a model itself; talks to `/api/generate`'s SSE stream over HTTP):
- Options: `--url` (full base URL), `--host`/`--port` (override just one part of the configured address; mutually exclusive with `--url`), `--max-tokens`, `--temperature`, `--timeout` (per-request HTTP timeout), `--stop` (repeatable; default `("\nuser:", "\nUser:")` -- guards against models that don't reliably emit EOS at the turn boundary and keep generating a hallucinated next turn; pass `--stop ''` once to disable).
- The server address has no hardcoded default in Python: `start()` calls `ephemeris_cli.config.resolve_base_url()`/`resolve_timeout()`, which layer command-line options over `$EPHEMERIS_CLIENT_URL` over the config files (see [`ephemeris_cli/config.py`](#packagesephemeris-cliephemeris_cliconfigpy-client-distribution) below). `--timeout` likewise falls back to the config file's `timeout_seconds`.
- On start: checks `/health`, raising a `click.ClickException` if the server isn't reachable -- the message names the resolved address *and which layer supplied it*, since a connection failure is as often a misconfigured address as a stopped server.

- Prints `_print_splash()` (see below), then `_print_welcome(base_url, base_url_source, temperature_label)`, then loads REPL command history (see below), then enters the REPL loop.
- `_init_readline()`/`_save_readline_history()`: history setup. macOS ships Python linked against **libedit** rather than GNU readline, and the two take incompatible `parse_and_bind` syntax -- a GNU-style binding string is silently ignored by libedit -- so up/down are bound explicitly for both flavors. A refused binding is swallowed rather than stopping the REPL.
- `_readline_safe_prompt(text)`: styles the prompt with the ANSI runs fenced in `\001`/`\002`. readline counts prompt characters to track the cursor; unfenced escapes are invisible on screen but not to that count, which misplaces the cursor once a recalled line wraps.
- Arrow-key line editing and history: at import time, `packages/ephemeris-cli/ephemeris_cli/main.py` tries `import readline` (wrapped in `try`/`except ImportError`, since it isn't available on Windows without a third-party `pyreadline3` install; `readline` is set to `None` if unavailable). Merely importing it is enough to give `click.prompt`'s underlying `input()` proper left/right cursor movement and up/down history recall -- without it, arrow keys just insert raw terminal escape sequences into the line instead of editing it. If `readline` loaded successfully, `start()` calls `readline.set_history_length(1000)` and `readline.read_history_file(_HISTORY_FILE)` (`~/.ephemeris_history`, ignoring `FileNotFoundError`/`OSError` on first run) right before the REPL loop, and `readline.write_history_file(_HISTORY_FILE)` on the way out, so command history persists across sessions like a shell's.
- REPL loop: reads a line via `click.prompt`; `/exit`/`/quit`/Ctrl-D/EOF ends the session; `/model` or `/model <name>` is routed to `_handle_model_command`; `/creativity` or `/creativity <preset|number>` is routed to `_handle_creativity_command`; anything else is sent as a prompt.
- Per turn: builds the JSON payload (`prompt`, optional `max_tokens`/`temperature`/`stop`), opens a `_StreamingBox("assistant", ...)`, and calls `_stream_reply(client, payload, box)`, feeding the box on success or an error message on `RuntimeError`/`httpx.HTTPError`, always closing the box in a `finally`.

`ephemeris config` -- **prints the resolved client configuration**: the effective `base_url` and where it came from, the effective `timeout_seconds`, every config file consulted (marking which exist), and the relevant environment variables. Purely diagnostic; it makes no network call.

`_stream_reply(client, payload, box)`:
- POSTs to `/api/generate` with `Accept: text/event-stream`; raises `RuntimeError` on a `>=400` status.
- Manually parses the SSE frame format (`event:`/`data:` lines, blank line as frame terminator, `:`-prefixed lines as keep-alive comments to skip) rather than using a client-side SSE library.
- For each `data:` line, strips *exactly one* leading space (the SSE-spec-mandated separator after `data:`) -- not `.strip()`, which would also eat a delta's own meaningful leading space (a word boundary) and run consecutive streamed words together.
- An `event: error` frame's `data:` is raised as a `RuntimeError` instead of being fed to the box.

`_handle_model_command(client, arg)` -- the REPL's `/model` command:
- No argument: `GET /api/model`, prints the current model name.
- With an argument: `POST /api/model` with `{"model_name": arg}`, using a `timeout=600.0` override on that one request (a swap drains in-flight work and may download/load a large model -- both can easily outlast a normal per-turn chat timeout). Prints the resulting model name on success, or the server's error `detail` (or a connection error) on failure.

`_handle_creativity_command(arg, current_temperature, current_label)` -- the REPL's `/creativity` command:
- No argument: prints the current creativity label.
- With an argument: resolves it against `_CREATIVITY_PRESETS` (`deterministic`/`balanced`/`creative`/`high-freedom`) first, falling back to parsing it as a raw float in `[0.0, 2.0]`; prints an error and leaves the current setting unchanged if neither matches. Purely client-side -- unlike `/model`, there's no server round-trip, since temperature is just a per-request payload field -- so it takes effect starting with the next turn's request.

Box-drawing helpers (all colored via `click.secho`, magenta borders unless noted):
- `_box_width()`: `max(min(terminal_columns - 4, 76), 20)` -- the content width used by every box.
- `_box_indent(width)`: left padding that centers a box in the terminal. A box occupies `width + 4` columns (two borders plus a space of padding each side), and boxes cap at 76 columns, so without this they would sit hard against the left edge on a wide terminal while the splash's logo is centered.
- `_box_top(label, width, border_color)` / `_box_bottom(width, border_color)` / `_box_row(text, width, border_color, fg, bold)`: draw one border/content line each; a top+bottom pair plus N rows always total the same fixed character width, so boxes stay aligned regardless of content.
- `_StreamingBox`: a box that grows as text is fed into it (`.feed(text)`), word-wrapping to its width as content accumulates, and closes its bottom border on `.close()`. Used to render the assistant's reply live as SSE deltas arrive, rather than buffering the whole response before drawing anything.
- `_print_welcome(base_url, base_url_source, temperature_label)`: the boxed connection-status panel shown once the REPL is ready -- title/version, tagline, `Connected to <base_url> (from <source>)`, the resolved `Creativity: <temperature_label>`, and usage hints (including `/model` and `/creativity`).
- `_print_splash()`: shown once, before connecting -- a small "✦ Welcome to Ephemeris Serve!" box, the block-art logo (`LOGO_LINES`, from `packages/ephemeris-cli/ephemeris_cli/logo.py`) centered in the terminal, the title, and a "Press Enter to continue" gate (blocks on a bare `input()`). Skipped entirely (falls straight through) if the terminal is narrower than the logo's fixed width, since a wrapped block-art render would just be noise.

### `packages/ephemeris-cli/ephemeris_cli/config.py` (client distribution)

Client-side configuration for the CLI: which server to talk to, and for how long to wait. It lives in the client distribution and never imports `settings/settings.py` -- that module imports `torch` and reads `settings/config.yaml` by a repo-relative path, neither of which holds when the CLI is installed as a console script and run from an arbitrary directory. `packages/ephemeris-cli/ephemeris_cli/config.py` depends only on `yaml` and the standard library.

Resolution order for the server address, highest priority first:
1. `--url`, or `--host`/`--port`
2. the `EPHEMERIS_CLIENT_URL` environment variable
3. the client's own `.env` files
4. the file named by `EPHEMERIS_CLIENT_CONFIG`
5. the user-level file: `$XDG_CONFIG_HOME/ephemeris/client.yaml`, else `~/.config/ephemeris/client.yaml`
6. the packaged default, `packages/ephemeris-cli/ephemeris_cli/client_config.yaml`

`user_config_dir()`: the client's config directory, `$XDG_CONFIG_HOME/ephemeris` (else `~/.config/ephemeris`), falling back to the pre-rename `ephemeris-serve` directory **only** when that one exists and the current one does not -- so an upgrade keeps working and a migrated user never sees the old path resurface. `legacy_user_config_dir()` names the old location. The history file behaves the same way (`_HISTORY_FILE`/`_LEGACY_HISTORY_FILE` in `main.py`).

`env_file_paths()`: the `.env` files consulted, lowest priority first -- `packages/ephemeris-cli/.env` (the package's own, resolved relative to the module and simply absent in a wheel install), `~/.config/ephemeris/.env`, `./.env` in the invocation directory, and the file named by `EPHEMERIS_CLIENT_ENV`.

`_parse_env_file(path)`: a ~25-line `KEY=value` parser (comments, blank lines, `export ` prefixes, optional quoting) rather than a `python-dotenv` dependency -- this distribution stays at three dependencies. Crucially it **only accepts `EPHEMERIS_CLIENT_*` keys**: a `.env` holding the server's `HF_KEY`/`EPHEMERIS_SERVER_*` contributes nothing, so running the client from the server's repo root -- where `./.env` *is* the server's file -- cannot leak one scope into the other.

`env_value(name)`: the real process environment first, `.env` second, matching every other dotenv implementation -- a value exported for one command must beat a file written months ago.

`load_config()`: merges every config file into one mapping, layered packaged → user-level → `$EPHEMERIS_CLIENT_CONFIG`, later files winning. Each file contributes its `client_config.defaults` mapping, so an override file only has to name the keys it actually changes. A file that doesn't exist contributes nothing (a missing user-level override is "nothing set here", not an error); a file that exists but can't be parsed, or has the wrong shape, raises `ClientConfigError`.

`normalize_base_url(value)`: returns a scheme-qualified base URL with no trailing slash. A bare `host` or `host:port` is assumed to be plain HTTP, so an operator can write `ephemeris.example.com` in a config file without it parsing as a relative path. Rejects any scheme other than `http`/`https`, and a URL with no host. A path prefix is preserved (a proxy may mount the API under a subpath) but the trailing slash is dropped, since every request path the CLI builds already starts with one.

`resolve_base_url(url, host, port, config) -> ResolvedBaseUrl`: applies the order above and returns a `(url, source)` `NamedTuple` -- the `source` string ("`--url`", "`$EPHEMERIS_CLIENT_URL`", "client config", ...) is shown in the welcome box and in the connection-failure message, so a wrong address is traceable to the layer that set it. `--url` combined with `--host`/`--port` is rejected rather than silently resolved. Given only `--host` or only `--port`, the missing half is taken from the configured address, so `--port 9000` alone still points at the configured host and scheme.

`resolve_timeout(timeout, config) -> float`: the `--timeout` option, else the config files' `timeout_seconds`, else a built-in fallback.

`_FALLBACK_BASE_URL`/`_FALLBACK_TIMEOUT_SECONDS` are last-resort values used only if the packaged config file is missing or unreadable (e.g. a partial install) -- keeping them here rather than in the CLI's option defaults means exactly one module in Python knows an address at all.

### `packages/ephemeris-cli/ephemeris_cli/client_config.yaml`

The packaged client defaults, shipped inside the `ephemeris_cli` package (`[tool.setuptools.package-data]` includes `*.yaml`) and located via `Path(__file__).with_name(...)` -- no repo-relative path, no server-side import.

- `base_url`: `http://127.0.0.1:8080` -- the nginx reverse proxy's port (see [Reverse Proxy](Operations#reverse-proxy)), not uvicorn's `8000`, so the default path exercises the same route real clients take. Bypass the proxy with `http://127.0.0.1:8000`.
- `timeout_seconds`: `120.0`.

This file is overwritten on reinstall/upgrade; a real deployment sets its address in the user-level file instead.

### `packages/ephemeris-cli/ephemeris_cli/logo.py`

`LOGO_LINES: list[str]` -- a **precomputed** (not regenerated at runtime) 36-column-wide, 18-row block-character rendering of the Ephemeris Serve logo, built from the vector geometry in `docs/assets/images/ephemeris-serve-logo.png` (an astronomical-instrument motif: graduated scale ring, tilted elliptical orbit, position markers, crosshair), rasterized onto a 36x36 grid with a wider stroke threshold than a literal 1:1 trace, for a smaller, bolder mark. Packed two rows per output line using Unicode half-block characters (`▀`/`▄`/`█`) for double vertical resolution -- hence 18 output lines for a 36-row grid. Originally a 48-wide/24-line rendering, shrunk so it fits more terminals without wrapping (see `_print_splash()`'s narrow-terminal skip, above).

It's precomputed rather than parsed from the SVG (or decoded from the sibling `.png`) at CLI startup for two reasons: decoding a large raster image in pure Python with no imaging library would be slow, and there's no need to re-derive a fixed piece of art on every invocation. There is currently no dependency (e.g. Pillow) added to the project for image handling.

---

## Configuration

### `settings/settings.py`

Loads runtime configuration from `settings/config.yaml` and environment variables.

Imports:
- `os`, `torch`
- `*` from `utils.utils`
- `BaseSettings` from `pydantic_settings`

`resolve_device(configured: str) -> str`:
- If `configured != "auto"`, returned as-is (so an operator can always pin a specific device, e.g. `"cpu"`, `"cuda:1"`, `"mps"`).
- If `"auto"`: checks **CUDA first**, then MPS, then falls back to CPU -- `torch.cuda.is_available()` → `"cuda"`; else `torch.backends.mps.is_available()` → `"mps"`; else `"cpu"`.

`ModelSetting`:
- Reads `model_config.defaults` from `settings/config.yaml`.
- `model_name`: `os.environ.get("EPHEMERIS_SERVER_MODEL_NAME")` if set, else the YAML default. This is how `ephemeris-serve serve --model` picks a model without editing the YAML file, and it's read via an env var (not a later in-process mutation) specifically so it survives uvicorn spawning fresh worker processes when `--workers > 1`.
- `device`: resolved via `resolve_device(config["device"])`.
- `max_length`, `temperature`, `top_k`, `top_p`, `repetition_penalty`, `num_return_sequences`: passed through from YAML. `temperature`, `top_k` and `top_p` are per-request *defaults* -- a caller can override each on `GenerateRequest`, and `InferenceRequest` resolves the fallback at construction. `repetition_penalty` stays global on both sampling paths.

Note: `model_settings.model_name` is also updated at runtime by `ModelLoader.reload()` (see [Model and Tokenizer](Model-and-Tokenizer#model-and-tokenizer)) once a `POST /api/model` hot-swap actually succeeds -- so `model_settings.model_name` always reflects whatever model is *currently* loaded, whether that was decided at process start (YAML or env var) or by a later runtime swap.

`LoggingSetting`: reads `logging_config.defaults` -- `log_level`, `log_file`.

`SchedulerSetting`: reads `scheduler_config.defaults` -- `streaming_request_timeout_seconds`, `batch_request_timeout_seconds`, `batch_generation_timeout_seconds`, `idempotency_key_ttl_seconds`, `model_swap_drain_timeout_seconds` (default `30.0` if absent from YAML, via `config.get(...)`). The single source of truth for every timeout used across `api/routes.py`, `scheduler/batch_scheduler.py`, and `scheduler/model_swap.py`. Also carries:
- `stop_window_slack_tokens` (default `16`): extra tokens decoded beyond the longest stop sequence when checking for a match. Sized in tokens against a character length, which errs long -- every token decodes to at least one character.
- `short_request_max_tokens` / `short_lane_reserved_slots` / `priority_aging_seconds`: request fairness. A request asking for at most `short_request_max_tokens` takes the short lane, which has slots reserved for it; aging promotes a waiting long request after `priority_aging_seconds` so the reservation cannot starve it in turn. `api/server.py` warns at startup if the aging window is not comfortably below `streaming_request_timeout_seconds`, since a request evicted before it is promoted makes the feature silently inert.
- `model_state_dir` / `model_state_poll_seconds`: cross-worker model-swap coordination. Empty means single-process behavior.

`RateLimitSetting`: reads `rate_limit_config.defaults` -- `enabled` (default `false`), `requests_per_second`, `burst`, `max_concurrent_requests` (`0` disables the concurrency cap). Consumed by `api/ratelimit.py`. Per worker process, so `--workers N` multiplies the effective limit by N.

`MetricsSetting`: reads `metrics_config.defaults` -- `prometheus_enabled` (default `false`), `require_auth`. Consumed by `metrics/prometheus.py` and the `/metrics` route registration in `api/server.py`.

`CacheSetting`: reads `cache_config.defaults` -- `kv_block_size`, consumed by `ContinuousScheduler.paged_cache` when constructing the `PagedKVCache`, plus `kv_pool_trim_idle_seconds`, `kv_pool_peak_decay` and `kv_pool_peak_slack`, which govern idle block-pool reclamation (see `ContinuousScheduler._maybe_trim_kv_pool`).

`SecretSetting(BaseSettings)`: `hf_key: str | None = ""`, read from `.env` by default.

Global instances: `model_settings`, `logging_settings`, `scheduler_settings`, `cache_settings`, `rate_limit_settings`, `metrics_settings`, `secret_settings`.

### `settings/config.yaml`

Current defaults:
- `model_name: TinyLlama/TinyLlama-1.1B-Chat-v1.0` (documented alternatives in a comment: `openai-community/gpt2-medium`, `distilgpt2`, `gpt2-large`, `Qwen/Qwen2.5-0.5B`)
- `device: "auto"` -- CUDA, then MPS, then CPU
- `max_length: 1024`
- `temperature: 0.7`
- `top_k: 8`
- `top_p: 0.9`
- `repetition_penalty: 1.2`
- `num_return_sequences: 1`
- `log_level: "DEBUG"`
- `log_file: "logs/app.log"`
- `streaming_request_timeout_seconds: 60.0`
- `batch_request_timeout_seconds: 20.0`
- `batch_generation_timeout_seconds: 25.0`
- `idempotency_key_ttl_seconds: 300.0`
- `model_swap_drain_timeout_seconds: 30.0`
- `stop_window_slack_tokens: 16`
- `short_request_max_tokens: 64`
- `short_lane_reserved_slots: 2`
- `priority_aging_seconds: 10.0`
- `model_state_dir: ""` (cross-worker swap coordination off)
- `model_state_poll_seconds: 2.0`
- `rate_limit_config.enabled: false`, `requests_per_second: 5.0`, `burst: 20`, `max_concurrent_requests: 8`
- `metrics_config.prometheus_enabled: false`, `require_auth: true`
- `kv_block_size: 16`
- `kv_pool_trim_idle_seconds: 60.0`, `kv_pool_peak_decay: 0.9`, `kv_pool_peak_slack: 1.5`

Notes:
- The model name is easily replaceable with any compatible causal language model -- at process start via this file or `EPHEMERIS_SERVER_MODEL_NAME`/`ephemeris-serve serve --model`, or at runtime via `POST /api/model` / the CLI's `/model` command.
- The device field supports `"auto"`, `"cpu"`, `"cuda"`, `"cuda:1"`, or `"mps"`.
