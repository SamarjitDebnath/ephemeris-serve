# CLI and Configuration

## CLI Layer

### `cli/main.py`

`click`-based CLI, installed as the `ephemeris-serve` console script (`[project.scripts]` in `pyproject.toml`). Two subcommands under one `@click.group()`:

`ephemeris-serve serve` -- **runs the server itself**:
- Options: `--model` (HF repo id, overrides `settings/config.yaml`'s `model_name` for this run), `--host` (default `0.0.0.0`), `--port` (default `8000`), `--workers` (default `1`), `--reload/--no-reload` (default off).
- If `--model` is given, sets `os.environ["EPHEMERIS_MODEL_NAME"]` *before* calling `uvicorn.run(...)` -- an env var rather than an in-process settings mutation, so it's correctly inherited even when `--workers > 1` makes uvicorn spawn fresh worker processes that re-import `settings.settings` from scratch (see `ModelSetting` in [Configuration](#configuration)).
- Calls `uvicorn.run("api.server:app", host=host, port=port, workers=workers, reload=reload)` -- functionally equivalent to `python main.py`, but with these as CLI flags instead of hardcoded values.

`ephemeris-serve start` -- **REPL chat client** against an already-running server (does not load a model itself; talks to `/api/generate`'s SSE stream over HTTP):
- Options: `--host` (default `127.0.0.1`), `--port` (default `8000`), `--max-tokens`, `--temperature`, `--timeout` (default `120.0`s, per-request HTTP timeout), `--stop` (repeatable; default `("\nuser:", "\nUser:")` -- guards against models that don't reliably emit EOS at the turn boundary and keep generating a hallucinated next turn; pass `--stop ''` once to disable).
- On start: checks `/health`, raising a `click.ClickException` with a helpful message if the server isn't reachable.
- Prints `_print_splash()` (see below), then `_print_welcome(base_url)`, then loads REPL command history (see below), then enters the REPL loop.
- Arrow-key line editing and history: at import time, `cli/main.py` tries `import readline` (wrapped in `try`/`except ImportError`, since it isn't available on Windows without a third-party `pyreadline3` install; `readline` is set to `None` if unavailable). Merely importing it is enough to give `click.prompt`'s underlying `input()` proper left/right cursor movement and up/down history recall -- without it, arrow keys just insert raw terminal escape sequences into the line instead of editing it. If `readline` loaded successfully, `start()` calls `readline.set_history_length(1000)` and `readline.read_history_file(_HISTORY_FILE)` (`~/.ephemeris_serve_history`, ignoring `FileNotFoundError`/`OSError` on first run) right before the REPL loop, and `readline.write_history_file(_HISTORY_FILE)` on the way out, so command history persists across sessions like a shell's.
- REPL loop: reads a line via `click.prompt`; `/exit`/`/quit`/Ctrl-D/EOF ends the session; `/model` or `/model <name>` is routed to `_handle_model_command`; `/creativity` or `/creativity <preset|number>` is routed to `_handle_creativity_command`; anything else is sent as a prompt.
- Per turn: builds the JSON payload (`prompt`, optional `max_tokens`/`temperature`/`stop`), opens a `_StreamingBox("assistant", ...)`, and calls `_stream_reply(client, payload, box)`, feeding the box on success or an error message on `RuntimeError`/`httpx.HTTPError`, always closing the box in a `finally`.

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
- `_box_top(label, width, border_color)` / `_box_bottom(width, border_color)` / `_box_row(text, width, border_color, fg, bold)`: draw one border/content line each; a top+bottom pair plus N rows always total the same fixed character width, so boxes stay aligned regardless of content.
- `_StreamingBox`: a box that grows as text is fed into it (`.feed(text)`), word-wrapping to its width as content accumulates, and closes its bottom border on `.close()`. Used to render the assistant's reply live as SSE deltas arrive, rather than buffering the whole response before drawing anything.
- `_print_welcome(base_url, temperature_label)`: the boxed connection-status panel shown once the REPL is ready -- title/version, tagline, `Connected to <base_url>`, the resolved `Creativity: <temperature_label>`, and usage hints (including `/model` and `/creativity`).
- `_print_splash()`: shown once, before connecting -- a small "✦ Welcome to Ephemeris Serve!" box, the block-art logo (`LOGO_LINES`, from `cli/logo.py`) centered in the terminal, the title, and a "Press Enter to continue" gate (blocks on a bare `input()`). Skipped entirely (falls straight through) if the terminal is narrower than the logo's fixed width, since a wrapped block-art render would just be noise.

### `cli/logo.py`

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
- `model_name`: `os.environ.get("EPHEMERIS_MODEL_NAME")` if set, else the YAML default. This is how `ephemeris-serve serve --model` picks a model without editing the YAML file, and it's read via an env var (not a later in-process mutation) specifically so it survives uvicorn spawning fresh worker processes when `--workers > 1`.
- `device`: resolved via `resolve_device(config["device"])`.
- `max_length`, `temperature`, `top_k`, `top_p`, `repetition_penalty`, `num_return_sequences`: passed through from YAML.

Note: `model_settings.model_name` is also updated at runtime by `ModelLoader.reload()` (see [Model and Tokenizer](Model-and-Tokenizer#model-and-tokenizer)) once a `POST /api/model` hot-swap actually succeeds -- so `model_settings.model_name` always reflects whatever model is *currently* loaded, whether that was decided at process start (YAML or env var) or by a later runtime swap.

`LoggingSetting`: reads `logging_config.defaults` -- `log_level`, `log_file`.

`SchedulerSetting`: reads `scheduler_config.defaults` -- `streaming_request_timeout_seconds`, `batch_request_timeout_seconds`, `batch_generation_timeout_seconds`, `idempotency_key_ttl_seconds`, `model_swap_drain_timeout_seconds` (default `30.0` if absent from YAML, via `config.get(...)`). The single source of truth for every timeout used across `api/routes.py`, `scheduler/batch_scheduler.py`, and `scheduler/model_swap.py`.

`CacheSetting`: reads `cache_config.defaults` -- `kv_block_size`, consumed by `ContinuousScheduler.paged_cache` when constructing the `PagedKVCache`.

`SecretSetting(BaseSettings)`: `hf_key: str | None = ""`, read from `.env` by default.

Global instances: `model_settings`, `logging_settings`, `scheduler_settings`, `cache_settings`, `secret_settings`.

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
- `kv_block_size: 16`

Notes:
- The model name is easily replaceable with any compatible causal language model -- at process start via this file or `EPHEMERIS_MODEL_NAME`/`ephemeris-serve serve --model`, or at runtime via `POST /api/model` / the CLI's `/model` command.
- The device field supports `"auto"`, `"cpu"`, `"cuda"`, `"cuda:1"`, or `"mps"`.
