"""Command-line entry point for the Ephemeris Serve **server**.

Installed as the `ephemeris-serve` console script by this repository's root
`pyproject.toml`. It loads the model and runs the HTTP API.

The chat client is a separate distribution (`packages/ephemeris-cli`, the
`ephemeris` command) with its own dependencies -- it speaks HTTP to a running
server and shares no code, no configuration files, and no environment
variables with this side.
"""
import os

import click


@click.group()
@click.version_option(package_name="ephemeris-serve")
def cli():
    """Ephemeris Serve inference server."""


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
@click.option(
    "--proxy-headers/--no-proxy-headers",
    "proxy_headers",
    default=True,
    show_default=True,
    help=(
        "Trust X-Forwarded-For/-Proto from a reverse proxy (see deploy/nginx). "
        "Disable if the server is exposed directly to untrusted clients, which "
        "could then spoof those headers."
    ),
)
@click.option(
    "--forwarded-allow-ips",
    "forwarded_allow_ips",
    default="127.0.0.1",
    show_default=True,
    help=(
        "Comma-separated addresses whose forwarded headers are trusted -- the "
        "proxy's own address. Use '*' only when nothing but a trusted proxy can reach this port."
    ),
)
def serve(model_name, host, port, workers, reload, proxy_headers, forwarded_allow_ips):
    """Start the Ephemeris Serve inference server (loads the model, runs the HTTP API)."""
    import uvicorn

    if model_name:
        # Passed via env var, not an in-process settings mutation, so it's
        # correctly picked up even when uvicorn spawns fresh worker processes
        # for --workers > 1 (see settings/settings.py:ModelSetting).
        os.environ["EPHEMERIS_SERVER_MODEL_NAME"] = model_name

    click.echo()
    click.secho(f"Starting Ephemeris Serve on {host}:{port}", fg="green", bold=True)
    click.secho(f"Model: {model_name or '(from settings/config.yaml)'}", fg="cyan")
    click.echo("Once it's up, run 'ephemeris' in another terminal to chat.\n")

    uvicorn.run(
        "api.server:app",
        host=host,
        port=port,
        workers=workers,
        reload=reload,
        # Behind the nginx config in deploy/nginx, every request arrives from
        # the proxy; without these, the app would see nginx's address and
        # scheme instead of the real client's.
        proxy_headers=proxy_headers,
        forwarded_allow_ips=forwarded_allow_ips,
    )


if __name__ == "__main__":
    cli()
