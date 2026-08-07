"""Command-line interface for Ephemeris Serve.

`ephemeris-serve serve` runs the inference server itself (model load,
scheduler, HTTP API). `ephemeris-serve start` is a separate REPL chat client
-- it does not run the model, it talks to an already-running server over the
`/api/generate` SSE endpoint (see `api/routes.py`).
"""
import os
import shutil
import textwrap
from importlib.metadata import PackageNotFoundError, version
from pathlib import Path

import click
import httpx

from cli.logo import LOGO_LINES

try:
    # Importing readline is enough to give click.prompt's underlying input()
    # proper arrow-key line editing (left/right cursor movement, up/down
    # history) -- without it, arrow keys just insert raw escape sequences.
    # Not available on Windows without a third-party pyreadline3 install.
    import readline
except ImportError:  # pragma: no cover
    readline = None

_HISTORY_FILE = Path.home() / ".ephemeris_serve_history"
_HISTORY_LENGTH = 1000

_TITLE = "EPHEMERIS SERVE"
_TAGLINE = "continuous scheduling · dynamic batching · SSE streaming"
# Local fallback only -- shown if a server error response can't even be
# parsed as JSON. The CLI is a plain HTTP client and deliberately doesn't
# import server-side modules (see utils/errors.py's INTERNAL_ERROR_MESSAGE,
# which is what the server actually sends for an unexpected failure).
_INTERNAL_ERROR_MESSAGE = "Internal server error"

# Friendly stand-ins for a raw --temperature float, in ascending order (used
# for --creativity's Click.Choice order and the "creativity" help text).
# Values chosen to span the schema's full [0.0, 2.0] range: 0.0 is greedy/
# fully repeatable, "balanced" matches settings/config.yaml's own default.
_CREATIVITY_PRESETS: dict[str, float] = {
    "deterministic": 0.0,
    "balanced": 0.7,
    "creative": 1.0,
    "high-freedom": 1.5,
}


def _extract_detail(response: httpx.Response) -> str:
    """Pull a safe `detail` string out of an error response body.

    The server never sends raw exception text for an unexpected failure (see
    `utils/errors.py`), so whatever `detail` it provides is always safe to
    show as-is. Falls back to a generic message if the body isn't the
    expected JSON shape at all.
    """
    try:
        detail = response.json().get("detail")
    except ValueError:
        return _INTERNAL_ERROR_MESSAGE
    return detail if isinstance(detail, str) and detail else _INTERNAL_ERROR_MESSAGE


def _box_width() -> int:
    """Content width (excluding borders/padding) for the current terminal."""
    term_width = shutil.get_terminal_size(fallback=(80, 24)).columns
    return max(min(term_width - 4, 76), 20)


def _box_top(label: str, width: int, border_color: str = "magenta") -> None:
    label_part = f"─ {label} " if label else "─ "
    fill = "─" * max(0, width + 2 - len(label_part))
    click.secho("╭" + label_part + fill + "╮", fg=border_color)


def _box_bottom(width: int, border_color: str = "magenta") -> None:
    click.secho("╰" + "─" * (width + 2) + "╯", fg=border_color)


def _box_row(text: str, width: int, border_color: str = "magenta", fg: str | None = None, bold: bool = False) -> None:
    click.secho("│ ", fg=border_color, nl=False)
    click.secho(text.ljust(width), fg=fg, bold=bold, nl=False)
    click.secho(" │", fg=border_color)


class _StreamingBox:
    """A bordered box that grows as text is fed into it, wrapping to `width`.

    Used to render the assistant's reply as it streams in, rather than
    buffering the whole response before drawing the box.
    """

    def __init__(self, label: str, width: int, border_color: str = "cyan"):
        self.width = width
        self.border_color = border_color
        self._label = label
        self._pending = ""
        self._opened = False

    def _open(self) -> None:
        _box_top(self._label, self.width, self.border_color)
        self._opened = True

    def _print_line(self, line: str) -> None:
        _box_row(line, self.width, self.border_color)

    def feed(self, text: str) -> None:
        if not self._opened:
            self._open()
        self._pending += text
        while True:
            newline_at = self._pending.find("\n")
            if newline_at != -1:
                segment, self._pending = self._pending[:newline_at], self._pending[newline_at + 1:]
                for line in textwrap.wrap(segment, self.width) or [""]:
                    self._print_line(line)
                continue
            if len(self._pending) <= self.width:
                return
            wrap_at = self._pending.rfind(" ", 0, self.width + 1)
            if wrap_at <= 0:
                wrap_at = self.width
            self._print_line(self._pending[:wrap_at])
            self._pending = self._pending[wrap_at:].lstrip(" ")

    def close(self) -> None:
        if not self._opened:
            self._open()
        for line in textwrap.wrap(self._pending, self.width) or [""]:
            self._print_line(line)
        self._pending = ""
        _box_bottom(self.width, self.border_color)


def _print_splash() -> None:
    """Print the startup splash: a small welcome box, the block-art logo
    (rendered from `asset/images/ephemeris-serve-logo.svg`, see `cli/logo.py`),
    and a "press Enter to continue" gate.

    Skipped (falls straight through) on terminals too narrow to fit the logo
    without wrapping, since a wrapped block-art render is just noise.
    """
    required_width = len(LOGO_LINES[0])
    term_width = shutil.get_terminal_size(fallback=(80, 24)).columns
    if term_width < required_width + 4:
        return

    box_width = _box_width()
    click.echo()
    _box_top("", box_width)
    _box_row(f"✦ Welcome to {_TITLE.title()}!", box_width, bold=True)
    _box_bottom(box_width)
    click.echo()

    for line in LOGO_LINES:
        click.secho(line.center(term_width - 1), fg="cyan", bold=True)
    click.echo()
    click.secho(_TITLE.center(term_width - 1), fg="cyan", bold=True)
    click.echo()

    click.secho("Press Enter to continue", fg="bright_black")
    try:
        input()
    except (EOFError, KeyboardInterrupt):
        pass


def _describe_temperature(creativity: str | None, temperature: float | None) -> str:
    """Human-readable label for the resolved sampling temperature."""
    if temperature is not None:
        return f"custom (temperature {temperature})"
    if creativity is not None:
        return f"{creativity} (temperature {_CREATIVITY_PRESETS[creativity]})"
    return "server default"


def _print_welcome(base_url: str, temperature_label: str) -> None:
    try:
        pkg_version = f"v{version('ephemeris-serve')}"
    except PackageNotFoundError:
        pkg_version = ""

    width = _box_width()
    click.echo()
    _box_top(f"✦ {_TITLE} {pkg_version}".rstrip(), width)
    _box_row("", width)
    _box_row(_TAGLINE, width, fg="bright_black")
    _box_row("", width)
    _box_row(f"Connected to {base_url}", width, fg="green")
    _box_row(f"Creativity: {temperature_label}", width, fg="green")
    _box_row("", width)
    _box_row("Type your message and press Enter.", width)
    _box_row("Use /model [name] to view or switch the loaded model.", width)
    _box_row("Use /creativity [preset|number] to view or change creativity.", width)
    _box_row("Use /exit or Ctrl-D to quit.", width)
    _box_row("Use Ctrl-C while a reply is streaming to cancel that request.", width)
    _box_row("", width)
    _box_bottom(width)
    click.echo()


def _handle_model_command(client: httpx.Client, arg: str) -> None:
    """Handle the REPL's `/model` command: view or hot-swap the server's loaded model."""
    if not arg:
        try:
            resp = client.get("/api/model")
            resp.raise_for_status()
        except httpx.HTTPStatusError as exc:
            click.secho(f"[server error {exc.response.status_code}] {_extract_detail(exc.response)}", fg="red")
            return
        except httpx.HTTPError as exc:
            click.secho(f"[connection error] {exc}", fg="red")
            return
        click.secho(f"Current model: {resp.json()['model_name']}", fg="cyan")
        return

    click.secho(f"Swapping model to {arg} ...", fg="cyan")
    try:
        # Overrides the client's --timeout: a swap drains in-flight requests
        # and may download/load a large model, both of which can easily
        # outlast a normal per-turn chat timeout.
        resp = client.post("/api/model", json={"model_name": arg}, timeout=600.0)
        resp.raise_for_status()
    except httpx.HTTPStatusError as exc:
        click.secho(f"[server error {exc.response.status_code}] {_extract_detail(exc.response)}", fg="red")
        return
    except httpx.HTTPError as exc:
        click.secho(f"[connection error] {exc}", fg="red")
        return
    click.secho(f"Now serving: {resp.json()['model_name']}", fg="green", bold=True)


def _handle_creativity_command(
    arg: str, current_temperature: float | None, current_label: str
) -> tuple[float | None, str]:
    """Handle the REPL's `/creativity` command: view or change the sampling
    temperature for subsequent turns. Purely client-side (unlike `/model`,
    there's no server round-trip -- temperature is just a per-request field),
    so it takes effect on the very next message.

    Returns the (possibly updated) `(effective_temperature, label)` pair;
    the caller is expected to reassign its own locals from it.
    """
    if not arg:
        click.secho(f"Current creativity: {current_label}", fg="cyan")
        return current_temperature, current_label

    preset_key = arg.strip().lower()
    if preset_key in _CREATIVITY_PRESETS:
        new_temperature = _CREATIVITY_PRESETS[preset_key]
        new_label = f"{preset_key} (temperature {new_temperature})"
    else:
        try:
            new_temperature = float(arg)
        except ValueError:
            click.secho(
                f"[error] '{arg}' isn't a creativity preset or a number. "
                f"Choices: {', '.join(_CREATIVITY_PRESETS)}, or any number 0.0-2.0.",
                fg="red",
            )
            return current_temperature, current_label
        if not (0.0 <= new_temperature <= 2.0):
            click.secho("[error] Temperature must be between 0.0 and 2.0.", fg="red")
            return current_temperature, current_label
        new_label = f"custom (temperature {new_temperature})"

    click.secho(f"Creativity set to: {new_label}", fg="green", bold=True)
    return new_temperature, new_label


@click.group()
@click.version_option(package_name="ephemeris-serve")
def cli():
    """Ephemeris Serve command-line interface."""


@cli.command()
@click.option(
    "--model",
    "model_name",
    default=None,
    help=(
        "Hugging Face model repo id to load, overriding settings/config.yaml's "
        "model_name for this run (e.g. TinyLlama/TinyLlama-1.1B-Chat-v1.0, "
        "Qwen/Qwen2.5-0.5B, distilgpt2, gpt2-large)."
    ),
)
@click.option("--host", default="0.0.0.0", show_default=True, help="Host to bind the server to.")
@click.option("--port", default=8000, show_default=True, type=int, help="Port to bind the server to.")
@click.option("--workers", default=1, show_default=True, type=int, help="Number of uvicorn worker processes.")
@click.option(
    "--reload/--no-reload",
    default=False,
    show_default=True,
    help="Enable uvicorn autoreload (development only; ignored when --workers > 1).",
)
def serve(model_name, host, port, workers, reload):
    """Start the Ephemeris Serve inference server (loads the model, runs the HTTP API)."""
    import uvicorn

    if model_name:
        # Passed via env var, not an in-process settings mutation, so it's
        # correctly picked up even when uvicorn spawns fresh worker processes
        # for --workers > 1 (see settings/settings.py:ModelSetting).
        os.environ["EPHEMERIS_MODEL_NAME"] = model_name

    click.echo()
    click.secho(f"Starting Ephemeris Serve on {host}:{port}", fg="green", bold=True)
    click.secho(f"Model: {model_name or '(from settings/config.yaml)'}", fg="cyan")
    click.echo("Once it's up, run 'ephemeris-serve start' in another terminal to chat.\n")

    uvicorn.run("api.server:app", host=host, port=port, workers=workers, reload=reload)


@cli.command()
@click.option("--host", default="127.0.0.1", show_default=True, help="Server host to connect to.")
@click.option("--port", default=8000, show_default=True, type=int, help="Server port to connect to.")
@click.option("--max-tokens", "max_tokens", default=None, type=int, help="Max tokens to generate per turn.")
@click.option(
    "--creativity",
    type=click.Choice(list(_CREATIVITY_PRESETS), case_sensitive=False),
    default=None,
    help=(
        "Friendly sampling-temperature preset: "
        + ", ".join(f"{name}={value}" for name, value in _CREATIVITY_PRESETS.items())
        + ". Ignored if --temperature is also given."
    ),
)
@click.option(
    "--temperature",
    default=None,
    type=float,
    help="Exact sampling temperature (0.0-2.0), for precise control. Overrides --creativity.",
)
@click.option("--timeout", default=120.0, show_default=True, type=float, help="Per-request HTTP timeout in seconds.")
@click.option(
    "--stop",
    "stop_sequences",
    multiple=True,
    default=("\nuser:", "\nUser:"),
    show_default=True,
    help=(
        "Stop sequence(s): the reply is trimmed before the first one that appears. "
        "Repeatable. Pass --stop '' once to disable stop sequences entirely."
    ),
)
def start(host, port, max_tokens, creativity, temperature, timeout, stop_sequences):
    """Start an interactive REPL that chats with a running Ephemeris Serve server."""
    base_url = f"http://{host}:{port}"
    # Some models don't reliably emit EOS at the turn boundary and will keep
    # generating a hallucinated next turn (e.g. "user: ..."); the default
    # stop sequences catch the common case without the caller opting in.
    stop_list = [s for s in stop_sequences if s]
    effective_temperature = temperature if temperature is not None else _CREATIVITY_PRESETS.get(creativity)
    temperature_label = _describe_temperature(creativity, temperature)

    _print_splash()

    with httpx.Client(base_url=base_url, timeout=timeout) as client:
        try:
            client.get("/health").raise_for_status()
        except httpx.HTTPError as exc:
            raise click.ClickException(
                f"Could not reach Ephemeris Serve at {base_url} ({exc}). "
                "Is the server running? Try 'make build-ephemeris' or 'make run' first."
            )

        _print_welcome(base_url, temperature_label)

        if readline is not None:
            readline.set_history_length(_HISTORY_LENGTH)
            try:
                readline.read_history_file(_HISTORY_FILE)
            except (FileNotFoundError, OSError):
                pass

        while True:
            try:
                prompt = click.prompt(click.style("you", fg="yellow", bold=True), prompt_suffix="> ")
            except (EOFError, click.exceptions.Abort):
                click.echo()
                break

            prompt = prompt.strip()
            if not prompt:
                continue
            if prompt in ("/exit", "/quit"):
                break
            if prompt == "/model" or prompt.startswith("/model "):
                _handle_model_command(client, prompt[len("/model"):].strip())
                click.echo()
                continue
            if prompt == "/creativity" or prompt.startswith("/creativity "):
                effective_temperature, temperature_label = _handle_creativity_command(
                    prompt[len("/creativity"):].strip(), effective_temperature, temperature_label
                )
                click.echo()
                continue

            payload = {"prompt": prompt}
            if max_tokens is not None:
                payload["max_tokens"] = max_tokens
            if effective_temperature is not None:
                payload["temperature"] = effective_temperature
            if stop_list:
                payload["stop"] = stop_list

            click.echo()
            box = _StreamingBox("assistant", _box_width(), border_color="cyan")
            error_message = None
            try:
                _stream_reply(client, payload, box)
            except RuntimeError as exc:
                # _stream_reply already builds a safe, fully-formatted
                # "[server error ...] ..." message -- never raw exception text.
                error_message = str(exc)
            except httpx.HTTPError as exc:
                error_message = f"[connection error] {exc}"
            except KeyboardInterrupt:
                # Closing _stream_reply's `with client.stream(...)` block (as
                # this exception unwinds) tears down the HTTP connection, which
                # the server detects as a disconnect and cancels the in-flight
                # request (see cancel_futures_on_disconnect in api/routes.py) --
                # so this only kills the current request, not the whole REPL.
                error_message = "[cancelled]"
            finally:
                # Close the box on whatever content actually streamed before
                # the failure, then print the error as its own line below --
                # feeding it into the box would run it straight into the last
                # partial line of generated text with no visual separation,
                # making a server-side error look like part of the reply.
                box.close()
            if error_message:
                click.secho(error_message, fg="yellow" if error_message == "[cancelled]" else "red")
            click.echo()

    if readline is not None:
        try:
            readline.write_history_file(_HISTORY_FILE)
        except OSError:
            pass

    click.echo("Goodbye.")


def _stream_reply(client: httpx.Client, payload: dict, box: _StreamingBox) -> None:
    """POST `payload` to `/api/generate` and feed each SSE delta into `box` as it arrives.

    Mirrors `streaming/stream_manager.py`'s SSE shape: plain `data:` frames are
    text deltas to render verbatim; an `event: error` frame carries a single
    error message in its `data:` field instead.
    """
    with client.stream(
        "POST",
        "/api/generate",
        json=payload,
        headers={"Accept": "text/event-stream"},
    ) as response:
        if response.status_code >= 400:
            response.read()
            raise RuntimeError(f"[server error {response.status_code}] {_extract_detail(response)}")

        event_type = "message"
        data_lines: list[str] = []

        for line in response.iter_lines():
            if line == "":
                if data_lines:
                    data = "\n".join(data_lines)
                    if event_type == "error":
                        # `data` is already a safe, generic message -- the server
                        # never puts raw exception text in an SSE `error` event
                        # (see utils/errors.py).
                        raise RuntimeError(f"[server error] {data}")
                    box.feed(data)
                event_type = "message"
                data_lines = []
                continue
            if line.startswith(":"):
                continue  # SSE keep-alive comment
            if line.startswith("event:"):
                event_type = line[len("event:"):].strip()
            elif line.startswith("data:"):
                value = line[len("data:"):]
                # SSE spec: strip exactly one leading space (the "data: " separator),
                # not the delta's own leading space -- that space is a real word boundary.
                if value.startswith(" "):
                    value = value[1:]
                data_lines.append(value)


if __name__ == "__main__":
    cli()
