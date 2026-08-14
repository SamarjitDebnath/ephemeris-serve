"""Command-line interface for `ephemeris`, the Ephemeris Serve chat client.

A standalone HTTP client for a running Ephemeris Serve server: it never
loads a model and imports nothing from the server, talking to it only over
`/api/generate`'s SSE endpoint. It is packaged and versioned separately
(`packages/ephemeris-cli`) so it installs without the server's torch and
transformers dependencies.

The server itself is started by the other distribution's `ephemeris-serve`
command.
"""
import os
import shutil
import sys
import textwrap
from importlib.metadata import PackageNotFoundError, version
from pathlib import Path

import click
import httpx

from ephemeris_cli.config import (
    API_KEY_ENV_VAR,
    ENV_FILE_ENV_VAR,
    env_file_paths,
    CONFIG_PATH_ENV_VAR,
    PACKAGED_CONFIG_PATH,
    SERVER_URL_ENV_VAR,
    ClientConfigError,
    auth_headers,
    load_config,
    mask_secret,
    resolve_api_key,
    resolve_base_url,
    resolve_timeout,
    user_config_path,
)
from ephemeris_cli.logo import LOGO_LINES

try:
    # gnureadline is the real GNU readline, statically linked. Preferred over
    # the stdlib module because on macOS that one is BSD **libedit**, which
    # cannot render a coloured prompt and count its width at the same time:
    # inline escapes are counted as columns (so a recalled line wraps eight
    # columns early and backspace can no longer erase across the bad row
    # boundary), while \001/\002-fenced escapes are hoisted to the front of the
    # prompt, putting the reset before the text and rendering it unstyled.
    # GNU readline handles the fences correctly, so the prompt is both yellow
    # and editable. Declared as a macOS-only dependency in pyproject.toml.
    import gnureadline as readline
except ImportError:  # pragma: no cover - Linux already has GNU readline
    try:
        # Importing readline is enough to give input() proper arrow-key line
        # editing (left/right cursor movement, up/down history) -- without it,
        # arrow keys just insert raw escape sequences. Not available on
        # Windows without a third-party pyreadline3 install.
        import readline
    except ImportError:
        readline = None

_HISTORY_FILE = Path.home() / ".ephemeris_history"
# The CLI was called ephemeris-serve before the client became its own
# distribution. Read the old file when the new one doesn't exist yet, so an
# upgrade doesn't silently drop a user's shell history.
_LEGACY_HISTORY_FILE = Path.home() / ".ephemeris_serve_history"
_HISTORY_LENGTH = 1000

# libedit stamps this on the first line of the history files it writes and
# escapes every space, tab, newline and backslash as a `\` plus three octal
# digits. GNU readline writes one raw line per entry and knows nothing about
# either convention, so handing it a libedit file yields a literal
# `_HiStOrY_V2_` entry followed by lines full of `\040`.
_LIBEDIT_HISTORY_HEADER = "_HiStOrY_V2_"

# Commands that only ever end a session. Recalling one is never what the up
# arrow was pressed for, and they otherwise dominate the file.
_UNMEMORABLE_INPUTS = frozenset({"/exit", "/quit"})


def _unescape_libedit(line: str) -> str:
    r"""Decode libedit's `\NNN` octal escapes (`\040` is a space)."""
    out: list[str] = []
    index = 0
    while index < len(line):
        char = line[index]
        if char == "\\" and line[index + 1:index + 4].isdigit() and len(line[index + 1:index + 4]) == 3:
            try:
                out.append(chr(int(line[index + 1:index + 4], 8)))
            except ValueError:
                out.append(char)
                index += 1
                continue
            index += 4
            continue
        out.append(char)
        index += 1
    return "".join(out)


def _read_history_entries(path: Path) -> list[str]:
    """Read `path` as history, in whichever flavor's format it was written.

    Both formats are parsed here rather than left to `readline.read_history_file`
    because the flavor that wrote the file is not necessarily the one reading
    it: a macOS install that gains `gnureadline` (or loses it) switches
    mid-stream, and each library reads the other's file as garbage instead of
    failing. Entries are returned oldest first, already cleaned of the session
    noise there is no point recalling.
    """
    try:
        lines = path.read_text(errors="replace").splitlines()
    except (FileNotFoundError, OSError):
        return []

    if lines and lines[0].strip() == _LIBEDIT_HISTORY_HEADER:
        lines = [_unescape_libedit(line) for line in lines[1:]]

    entries: list[str] = []
    for line in lines:
        entry = line.strip()
        if not entry or entry in _UNMEMORABLE_INPUTS:
            continue
        # Consecutive repeats make the up arrow feel stuck: pressing it twice
        # should reach the previous *different* message.
        if entries and entries[-1] == entry:
            continue
        entries.append(entry)
    return entries[-_HISTORY_LENGTH:]


def _init_readline() -> None:
    """Load persistent history and make up/down recall it.

    macOS ships Python linked against **libedit**, not GNU readline, and the
    two take different `parse_and_bind` syntax -- a GNU-style binding string is
    silently ignored by libedit. Both are bound explicitly here so arrow-key
    history behaves the same on macOS and Linux rather than depending on which
    library happens to be underneath.
    """
    if readline is None:  # Windows without pyreadline3
        return

    is_libedit = _readline_is_libedit()
    try:
        if is_libedit:
            readline.parse_and_bind("bind ^[[A ed-search-prev-history")
            readline.parse_and_bind("bind ^[[B ed-search-next-history")
            readline.parse_and_bind("bind ^R em-inc-search-prev")
        else:
            readline.parse_and_bind(r'"\e[A": previous-history')
            readline.parse_and_bind(r'"\e[B": next-history')
    except Exception:  # pragma: no cover - a refused binding must not stop the REPL
        pass

    readline.set_history_length(_HISTORY_LENGTH)
    source = _HISTORY_FILE if _HISTORY_FILE.is_file() else _LEGACY_HISTORY_FILE
    readline.clear_history()
    for entry in _read_history_entries(source):
        readline.add_history(entry)


def _current_history_entries() -> list[str]:
    """Everything readline currently holds, oldest first."""
    return [
        readline.get_history_item(index)
        for index in range(1, readline.get_current_history_length() + 1)
    ]


def _save_readline_history() -> None:
    """Write history back out, minus the entries not worth recalling.

    readline appends every accepted line, `/exit` included, so the file is
    filtered on the way out as well as on the way in -- otherwise the junk is
    simply rewritten each session. Written directly rather than via
    `write_history_file` so the format is always the plain one-line-per-entry
    form both libedit and GNU readline can be handed back (see
    `_read_history_entries`).
    """
    if readline is None:
        return

    entries: list[str] = []
    for item in _current_history_entries():
        entry = (item or "").strip()
        if not entry or entry in _UNMEMORABLE_INPUTS:
            continue
        if entries and entries[-1] == entry:
            continue
        entries.append(entry)

    try:
        _HISTORY_FILE.write_text("".join(f"{entry}\n" for entry in entries[-_HISTORY_LENGTH:]))
        _HISTORY_FILE.chmod(0o600)
    except OSError:
        pass

_TITLE = "EPHEMERIS"
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


def _readline_safe_prompt(text: str) -> str:
    """Style `text` with the ANSI runs marked as zero-width for readline.

    readline counts the prompt's characters to know where the cursor sits. Raw
    escape sequences are invisible on screen but not to that count, so a
    coloured prompt makes it misplace the cursor once a recalled line is long
    enough to wrap. \001/\002 fence the non-printing runs.

    The markers are only meaningful to readline, so they are emitted only when
    the prompt is handed straight to `input()` (see `_read_prompt`). Colour is
    dropped entirely when stdout isn't a terminal, so a piped session doesn't
    collect escape sequences.

    Under libedit (macOS without `gnureadline`) the prompt is left uncoloured.
    Neither form works there: fenced escapes are hoisted to the front of the
    prompt and render it unstyled anyway, and inline escapes are counted as
    columns, so a recalled line wraps early and backspace stops erasing across
    the row boundary. A plain prompt that edits correctly beats a yellow one
    that doesn't. With no readline at all there is nothing to miscount, so the
    colour is safe.
    """
    if not _stdout_is_tty():
        return text
    if readline is None:
        return click.style(text, fg="yellow", bold=True)
    if _readline_is_libedit():
        return text
    return "\001\033[33m\033[1m\002" + text + "\001\033[0m\002"


def _readline_is_libedit() -> bool:
    """True when `readline` is the BSD libedit emulation (macOS's default)."""
    return readline is not None and "libedit" in (readline.__doc__ or "")


def _stdout_is_tty() -> bool:
    try:
        return sys.stdout.isatty()
    except (AttributeError, ValueError):  # pragma: no cover - detached stdout
        return False


def _read_prompt(text: str) -> str:
    """Read one REPL line behind a coloured `text` prompt.

    Deliberately `input()` rather than `click.prompt`: click echoes the prompt
    itself and passes only a space to `input()`, so readline never sees the
    prompt at all -- neither its width nor the \001/\002 fences that tell it
    which parts don't occupy a column. Passing the styled string straight to
    `input()` is what makes the colour render *and* keeps the cursor in the
    right place on recalled lines that wrap.
    """
    return input(_readline_safe_prompt(text))


# Widest a box's content may get, whatever the terminal. Wide enough that a
# full server URL or a long error line fits on one row, still narrow enough
# that streamed prose stays readable rather than running edge to edge.
_MAX_BOX_WIDTH = 100


def _box_width() -> int:
    """Content width (excluding borders/padding) for the current terminal."""
    term_width = shutil.get_terminal_size(fallback=(80, 24)).columns
    return max(min(term_width - 4, _MAX_BOX_WIDTH), 20)


def _box_indent(width: int) -> str:
    """Left padding that centers a box of `width` content columns.

    A box occupies `width + 4` columns: two border characters and one space of
    padding on each side. Boxes are capped at `_MAX_BOX_WIDTH` columns, so on a
    wider terminal they would otherwise sit hard against the left edge while
    the startup splash's logo is centered -- this keeps the two aligned.
    """
    term_width = shutil.get_terminal_size(fallback=(80, 24)).columns
    return " " * max(0, (term_width - (width + 4)) // 2)


def _box_top(label: str, width: int, border_color: str = "magenta") -> None:
    label_part = f"─ {label} " if label else "─ "
    # A label longer than the box would push the top border past every other
    # row's right edge, so it is cut to fit rather than allowed to widen the box.
    label_part = label_part[: width + 2]
    fill = "─" * max(0, width + 2 - len(label_part))
    click.secho(_box_indent(width) + "╭" + label_part + fill + "╮", fg=border_color)


def _box_bottom(width: int, border_color: str = "magenta") -> None:
    click.secho(_box_indent(width) + "╰" + "─" * (width + 2) + "╯", fg=border_color)


def _box_row(text: str, width: int, border_color: str = "magenta", fg: str | None = None, bold: bool = False) -> None:
    """Print `text` as one or more bordered rows, wrapped to `width`.

    Wrapping happens here rather than at the call sites so that no caller can
    push a line through the right border: a long URL, a long model name, or a
    long error string all stay inside the box. Long words are broken because a
    single unbreakable token (a URL) would otherwise overflow anyway. Callers
    that have already wrapped their own text (`_StreamingBox`) pass lines that
    fit and are printed verbatim -- wrapping a short line would also strip its
    leading whitespace, silently reindenting streamed code.
    """
    lines = [text] if len(text) <= width else (
        textwrap.wrap(text, width, break_long_words=True, break_on_hyphens=False) or [""]
    )
    for line in lines:
        click.secho(_box_indent(width) + "│ ", fg=border_color, nl=False)
        click.secho(line.ljust(width), fg=fg, bold=bold, nl=False)
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
    (rendered from the project logo, see `ephemeris_cli/logo.py`),
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


def _print_welcome(base_url: str, base_url_source: str, resolved_key, temperature_label: str) -> None:
    try:
        pkg_version = f"v{version('ephemeris-cli')}"
    except PackageNotFoundError:
        pkg_version = ""

    width = _box_width()
    click.echo()
    _box_top(f"✦ {_TITLE} {pkg_version}".rstrip(), width)
    _box_row("", width)
    _box_row(_TAGLINE, width, fg="bright_black")
    _box_row("", width)
    _box_row(f"Connected to {base_url} (from {base_url_source})", width, fg="green")
    if resolved_key is not None:
        # Only where the key came from, never any of the key itself. This box
        # is printed on every start -- including screen shares and pasted
        # terminal output -- so even a partial key is worth not putting here.
        # `ephemeris config` still shows a masked value for debugging.
        _box_row(f"API key: set (from {resolved_key.source})", width, fg="green")
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
@click.version_option(package_name="ephemeris-cli")
def cli():
    """Ephemeris -- chat client for a running Ephemeris Serve server."""


@cli.command()
@click.option(
    "--url",
    "base_url_option",
    default=None,
    help=(
        "Full base URL of the server to connect to, e.g. https://ephemeris.example.com. "
        f"Overrides ${SERVER_URL_ENV_VAR} and the client config file. Cannot be combined "
        "with --host/--port."
    ),
)
@click.option(
    "--host",
    default=None,
    help="Server host to connect to, overriding the host in the configured URL.",
)
@click.option(
    "--port",
    default=None,
    type=int,
    help="Server port to connect to, overriding the port in the configured URL.",
)
@click.option(
    "--api-key",
    "api_key_option",
    default=None,
    help=(
        f"API key to send as 'Authorization: Bearer <key>'. Overrides ${API_KEY_ENV_VAR} "
        "and the client config file. Prefer the environment variable, so the key "
        "doesn't land in shell history."
    ),
)
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
@click.option(
    "--timeout",
    default=None,
    type=float,
    help="Per-request HTTP timeout in seconds, overriding the client config's timeout_seconds.",
)
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
def start(base_url_option, host, port, api_key_option, max_tokens, creativity, temperature, timeout, stop_sequences):
    """Start an interactive REPL that chats with a running Ephemeris Serve server."""
    # The address is never hardcoded here: it comes from --url/--host/--port,
    # then $EPHEMERIS_SERVER_URL, then the client config files (see
    # cli/config.py and cli/client_config.yaml).
    try:
        config = load_config()
        base_url, base_url_source = resolve_base_url(base_url_option, host, port, config)
        timeout = resolve_timeout(timeout, config)
        resolved_key = resolve_api_key(api_key_option, config)
    except ClientConfigError as exc:
        raise click.ClickException(str(exc))

    # Some models don't reliably emit EOS at the turn boundary and will keep
    # generating a hallucinated next turn (e.g. "user: ..."); the default
    # stop sequences catch the common case without the caller opting in.
    stop_list = [s for s in stop_sequences if s]
    effective_temperature = temperature if temperature is not None else _CREATIVITY_PRESETS.get(creativity)
    temperature_label = _describe_temperature(creativity, temperature)

    _print_splash()

    # The key rides on the client, so every request -- chat turns, /health,
    # and the /model command -- carries it without each call site remembering.
    with httpx.Client(
        base_url=base_url,
        timeout=timeout,
        headers=auth_headers(resolved_key.value if resolved_key else None),
    ) as client:
        try:
            client.get("/health").raise_for_status()
        except httpx.HTTPError as exc:
            raise click.ClickException(
                f"Could not reach Ephemeris Serve at {base_url}, resolved from {base_url_source} ({exc}). "
                "Is the server running, and is that the right address? Start one with "
                "'make build-ephemeris' or 'make run', point this run elsewhere with "
                f"--url, or set a permanent address in {user_config_path()} "
                f"(see 'ephemeris config')."
            )

        # /health is unauthenticated, so reaching it proves nothing about the
        # key. Probe one authenticated route now, so a missing or wrong key
        # fails here with a clear message instead of on the first chat turn.
        auth_probe = client.get("/api/model")
        if auth_probe.status_code in (401, 403):
            hint = (
                f"Set it with --api-key, ${API_KEY_ENV_VAR}, or 'api_key' in {user_config_path()}."
                if resolved_key is None
                else f"The key from {resolved_key.source} was rejected."
            )
            raise click.ClickException(f"{base_url} requires an API key. {_extract_detail(auth_probe)} {hint}")

        _print_welcome(base_url, base_url_source, resolved_key, temperature_label)

        _init_readline()

        while True:
            try:
                prompt = _read_prompt("you> ")
            except (EOFError, KeyboardInterrupt, click.exceptions.Abort):
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

    _save_readline_history()

    click.echo("Goodbye.")


@cli.command()
def config():
    """Show the resolved client configuration and where each value came from."""
    try:
        loaded = load_config()
        resolved_url = resolve_base_url(config=loaded)
        resolved_timeout = resolve_timeout(config=loaded)
        resolved_key = resolve_api_key(config=loaded)
    except ClientConfigError as exc:
        raise click.ClickException(str(exc))

    user_path = user_config_path()
    explicit_path = os.environ.get(CONFIG_PATH_ENV_VAR)

    click.echo()
    click.secho("Resolved settings", fg="cyan", bold=True)
    click.echo(f"  base_url        {resolved_url.url}  ({resolved_url.source})")
    click.echo(f"  timeout_seconds {resolved_timeout}")
    if resolved_key is None:
        click.echo("  api_key         (unset -- requests are sent unauthenticated)")
    else:
        # Masked: enough to tell two keys apart while debugging, not enough to use.
        click.echo(f"  api_key         {mask_secret(resolved_key.value)}  ({resolved_key.source})")

    click.echo()
    click.secho("Config files, lowest priority first", fg="cyan", bold=True)
    click.echo(f"  packaged   {PACKAGED_CONFIG_PATH}")
    click.echo(f"  user       {user_path}{'' if user_path.is_file() else '  (not created)'}")
    click.echo(f"  ${CONFIG_PATH_ENV_VAR}  {explicit_path or '(unset)'}")

    click.echo()
    click.secho(".env files, lowest priority first", fg="cyan", bold=True)
    for path in env_file_paths():
        click.echo(f"  {path}{'' if path.is_file() else '  (not present)'}")
    click.echo("  (only EPHEMERIS_CLIENT_* entries are read; anything else is ignored)")

    click.echo()
    click.secho("Environment", fg="cyan", bold=True)
    click.echo(f"  ${SERVER_URL_ENV_VAR}  {os.environ.get(SERVER_URL_ENV_VAR) or '(unset)'}")
    click.echo(f"  ${API_KEY_ENV_VAR}     {'(set)' if os.environ.get(API_KEY_ENV_VAR) else '(unset)'}")
    click.echo(f"  ${ENV_FILE_ENV_VAR}         {os.environ.get(ENV_FILE_ENV_VAR) or '(unset)'}")

    click.echo()
    click.echo("To set a permanent address, create the user file above with:")
    click.echo()
    click.secho("  client_config:\n    defaults:\n      base_url: \"https://your-host\"", fg="bright_black")
    click.echo()


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
