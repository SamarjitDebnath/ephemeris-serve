<p align="center">
  <img src="https://raw.githubusercontent.com/SamarjitDebnath/ephemeris-serve/main/docs/assets/images/ephemeris-serve-logo.png" alt="Ephemeris Serve logo" width="160">
</p>

# Ephemeris Serve — Technical Documentation

Continuous-batching inference server for Hugging Face causal language models,
with a paged KV cache, SSE token streaming, runtime model hot-swap, and a
`click`-based CLI.

| Page | Covers |
| --- | --- |
| [Architecture](Architecture) | End-to-end request flow, entrypoints, control-flow and lifecycle summaries |
| [API Layer](API-Layer) | `api/server.py`, `api/routes.py` — every HTTP endpoint |
| [Model and Tokenizer](Model-and-Tokenizer) | `engine/model_loader.py`, `tokenizer/tokenizer_service.py` |
| [Scheduler Layer](Scheduler-Layer) | Queues, idempotency, continuous scheduler, batch scheduler, model swap |
| [Cache and Inference Engine](Cache-and-Engine) | `cache/paged_kv_cache.py`, `engine/generator.py` |
| [Streaming and Metrics](Streaming-and-Metrics) | SSE stream manager, rolling metrics |
| [CLI and Configuration](CLI-and-Configuration) | `ephemeris-serve` CLI, settings and `config.yaml` |
| [Reference](Reference) | Schemas, logging, utils, tensor shapes, module index, data structures |
| [Operations and Roadmap](Operations) | Deployment notes, implemented work, extension points, known behavior |

---

## Overview

This repository implements a lightweight FastAPI-based inference server for autoregressive language models using the Hugging Face `transformers` ecosystem.

The system is architected around a continuous token scheduler that batches prompt requests, reuses a per-request slice of a shared paged KV cache, and streams decoded text back to the client through SSE. A separate distribution (`packages/ephemeris-cli`, the `ephemeris` command) provides a `click`-based REPL chat client that speaks HTTP to a running server and installs without the server's dependencies.

Key capabilities:
- HTTP endpoint `/api/generate` for streaming, prompt-based generation
- HTTP endpoint `/api/generate_batch` for non-streaming batch generation with request timeout and cancellation support
- HTTP endpoint `/api/model` (`GET`/`POST`) to inspect or hot-swap the loaded model without a process restart
- HTTP endpoint `/api/metrics` for runtime queue and batch metrics
- Server-sent events (SSE) token streaming using `EventSourceResponse`
- Automatic chat prompt formatting when tokenizer supports `apply_chat_template()`, with raw-prompt fallback for base models
- Central request queues and a continuous scheduler for asynchronous generation, using a block-based ("paged") KV cache that supports mixing prefill and decode rows in the same batched step
- Per-request `stop` sequences, checked on both the streaming and batch generation paths
- Buffered token streaming that emits only at whitespace/punctuation boundaries or after a short buffer threshold
- Pytorch model execution with configurable temperature, top-k, top-p, and repetition penalty
- Configurable model and logging settings via YAML, environment variables, and `.env`
- A CLI (`ephemeris-serve`) that can start the server (`serve`, with `--model` selection) and chat with a running one (`start`, a boxed-UI REPL with a `/model` command for runtime model switching)
