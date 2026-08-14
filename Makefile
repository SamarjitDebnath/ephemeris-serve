.PHONY: help install dev sync run test test-latency test-unit test-all format lint clean logs build-ephemeris wiki-sync

help:
	@echo "LLM Inference Server - Available Commands"
	@echo ""
	@echo "Setup & Installation:"
	@echo "  make install      - Install dependencies (production only)"
	@echo "  make dev          - Install with dev dependencies"
	@echo "  make sync         - Sync dependencies from uv.lock"
	@echo ""
	@echo "Running:"
	@echo "  make run          - Run the server (development mode)"
	@echo "  make run-prod     - Run the server (production mode)"
	@echo "  make build-ephemeris - E2E: sync deps, install the CLI, start the server"
	@echo ""
	@echo "Testing:"
	@echo "  make test-unit    - Run unit tests"
	@echo "  make test-latency - Run latency benchmarks"
	@echo "  make test-all     - Run all tests"
	@echo ""
	@echo "Code Quality:"
	@echo "  make format       - Format code with black & isort"
	@echo "  make lint         - Lint code with pylint & flake8"
	@echo "  make check        - Run format check + lint (no modifications)"
	@echo ""
	@echo "Utilities:"
	@echo "  make logs         - Tail application logs"
	@echo "  make wiki-sync    - Publish wiki/ to the GitHub wiki"
	@echo "  make clean        - Remove cache, logs, and build artifacts"

install:
	uv pip install -e .

dev: install
	uv pip install -e ".[dev]"

sync:
	uv sync

run:
	uv run python main.py

run-prod:
	uv run python -m uvicorn main:app --host 0.0.0.0 --port 8000 --workers 4

build-ephemeris:
	@echo "==> [1/2] Installing Ephemeris Serve and dependencies (uv sync)"
	uv sync
	@echo "==> [2/2] Starting the server -- once it's up, run 'ephemeris-serve start' in another terminal to chat"
	uv run python main.py

test-unit:
	uv run pytest tests/ -v

test-latency:
	uv run pytest tests/test_latency.py -v -s

test-all: test-unit test-latency
	@echo "✓ All tests completed"

format:
	uv run black . --exclude="venv|.venv"
	uv run isort . --skip-glob="venv|.venv"
	@echo "✓ Code formatted"

lint:
	uv run pylint **/*.py --disable=C0111,C0103,R0913 || true
	uv run flake8 . --count --statistics --exclude=venv,.venv || true

check: format lint
	@echo "✓ Code quality checks passed"

logs:
	tail -f logs/app.log

wiki-sync:
	bash scripts/sync_wiki.sh

clean:
	find . -type d -name __pycache__ -exec rm -rf {} + 2>/dev/null || true
	find . -type d -name .pytest_cache -exec rm -rf {} + 2>/dev/null || true
	find . -type d -name .coverage -exec rm -rf {} + 2>/dev/null || true
	find . -type d -name htmlcov -exec rm -rf {} + 2>/dev/null || true
	find . -type d -name *.egg-info -exec rm -rf {} + 2>/dev/null || true
	rm -f .coverage
	@echo "✓ Cleaned up cache and artifacts"
