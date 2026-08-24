from fastapi import FastAPI
from fastapi.responses import JSONResponse
from contextlib import asynccontextmanager
from huggingface_hub import login
import asyncio

from settings.settings import logging_settings, model_settings, scheduler_settings, secret_settings
from schemas.schemas import HealthResponse
from logger import setup_logger
from utils.utils import Utils
from api.routes import router

# Heavy imports that touch torch/multiprocessing are deferred until the
# application `lifespan` so `Utils.configure_multiprocessing()` can run
# first inside the worker process and prevent semaphore/resource_tracker warnings.


logger = setup_logger(__name__, level=logging_settings.log_level, log_file=logging_settings.log_file)

@asynccontextmanager
async def lifespan(app: FastAPI):
    logger.info("Starting up the server...")
    
    # Configure multiprocessing/torch early in the worker process
    Utils.configure_multiprocessing()

    # Prometheus multiprocess mode leaves mmap files behind; files from a
    # previous run are picked up by the exporter and inflate every counter.
    # Clearing at startup is the documented remedy.
    from metrics.prometheus import prepare_multiprocess_dir, prometheus_metrics
    from settings.settings import metrics_settings as _metrics_settings

    if _metrics_settings.prometheus_enabled:
        if not prometheus_metrics.available:
            logger.warning(
                "metrics_config.prometheus_enabled is true but prometheus-client is not "
                'installed; GET /metrics will report that. Install with: pip install -e ".[metrics]"'
            )
        multiproc_dir = prepare_multiprocess_dir()
        if multiproc_dir:
            logger.info("Prometheus multiprocess mode using %s", multiproc_dir)

    # Defer heavy imports until after multiprocessing configuration
    from tokenizer.tokenizer_service import tokenizer_service
    from engine.model_loader import model_loader
    from scheduler.batch_scheduler import BatchScheduler
    from scheduler.continuous_scheduler import ContinuousScheduler
    from engine.generator import engine

    # API-key authentication (see api/auth.py). Absent keys means every /api
    # route is open, which is deliberate for local development but must never
    # be the case for a deployment reachable from outside localhost -- so it
    # is warned about loudly rather than silently allowed.
    from api.auth import admin_api_keys, auth_enabled

    if not auth_enabled():
        logger.warning(
            "No API keys configured (EPHEMERIS_API_KEYS): every /api route is UNAUTHENTICATED, "
            "including POST /api/model, which loads arbitrary Hugging Face models. "
            "Do not expose this server beyond localhost."
        )
    elif not admin_api_keys():
        logger.warning(
            "EPHEMERIS_API_KEYS is set but EPHEMERIS_ADMIN_API_KEYS is not: "
            "POST /api/model will reject every request."
        )
    else:
        logger.info("API-key authentication enabled.")

    # Rate limiting (see api/ratelimit.py). Off by default like auth; the
    # per-worker caveat is stated here because the configured numbers are not
    # the effective ones under `--workers`.
    from settings.settings import rate_limit_settings

    if rate_limit_settings.enabled:
        logger.info(
            "Rate limiting enabled: %.1f req/s, burst %d, max %d concurrent per identity "
            "(per worker process -- multiply by the worker count for the effective limit).",
            rate_limit_settings.requests_per_second,
            rate_limit_settings.burst,
            rate_limit_settings.max_concurrent_requests,
        )
    else:
        logger.warning(
            "Rate limiting is disabled: one client can saturate the scheduler. "
            "Enable rate_limit_config.enabled for any deployment reachable beyond localhost."
        )

    # Aging only helps if a long request gets promoted *before* the eviction
    # that would otherwise drop it. The two settings are coupled, and getting
    # this wrong produces a fairness feature that silently never fires.
    if scheduler_settings.priority_aging_seconds >= scheduler_settings.streaming_request_timeout_seconds / 2:
        logger.warning(
            "priority_aging_seconds (%.1fs) is not comfortably below "
            "streaming_request_timeout_seconds (%.1fs): a waiting long request may be "
            "evicted before aging promotes it.",
            scheduler_settings.priority_aging_seconds,
            scheduler_settings.streaming_request_timeout_seconds,
        )

    # Hugging Face authentication
    if not secret_settings.hf_key:
        logger.warning("Token for Hugging Face Hub not found. Using anonymous access.")
    else:
        login(token=secret_settings.hf_key)
        logger.info("HuggingFace authentication successful.")

    # Load tokenizer and model
    logger.info("Loading tokenizer and model...")
    logger.info("Resolved compute device: %s", model_settings.device)
    tokenizer_service.load()
    model_loader.load()
    logger.info("Tokenizer and model loaded successfully.")

    # Warmup the model
    logger.info("Warming up the model...")
    model_loader.warmup()
    logger.info("Model warmup completed.")

    # setting up schedulers and include API router
    logger.info("Setting up continuous and batch schedulers...")
    scheduler = ContinuousScheduler(engine, tokenizer_service)
    # Exposed so route handlers (e.g. POST /api/model) can reach the live
    # scheduler instance -- it's otherwise only closed over by the task below.
    app.state.scheduler = scheduler
    scheduler_task = asyncio.create_task(scheduler.run())
    batch_scheduler = BatchScheduler(
        engine,
        tokenizer_service,
        request_timeout=scheduler_settings.batch_request_timeout_seconds,
    )
    batch_scheduler_task = asyncio.create_task(batch_scheduler.run())
    logger.info("Schedulers setup completed...")

    yield
    
    logger.info("Stopping schedulers...")
    scheduler_task.cancel()
    batch_scheduler_task.cancel()
    try:
        await scheduler_task
    except asyncio.CancelledError:
        pass
    try:
        await batch_scheduler_task
    except asyncio.CancelledError:
        pass

    logger.info("Shutting down the server...")

def create_app() -> FastAPI:
    app = FastAPI(title="Ephemeris Serve", version="0.1.0", lifespan=lifespan)
    app.include_router(router, prefix="/api")

    # Prometheus scrape endpoint, mounted at the path scrapers default to
    # rather than under /api. Registered only when enabled so the route does
    # not exist at all otherwise -- an endpoint returning "not installed" is
    # worse than a 404, which a scraper reports as a clean target-down.
    from settings.settings import metrics_settings

    if metrics_settings.prometheus_enabled:
        from api.routes import prometheus_endpoint, prometheus_endpoint_open

        handler = prometheus_endpoint if metrics_settings.require_auth else prometheus_endpoint_open
        app.add_api_route("/metrics", handler, methods=["GET"], include_in_schema=False)

    @app.get("/")
    def root():
        return {"message": "Welcome to Ephemeris Serve!"}

    @app.get("/health", response_model=HealthResponse)
    def health():
        logger.info("Health check endpoint called")
        return JSONResponse(status_code=200, content={"status": "healthy"})

    # Routers are included during lifespan after heavy imports

    return app

app = create_app()