from fastapi import FastAPI
from fastapi.responses import JSONResponse
from contextlib import asynccontextmanager
from huggingface_hub import login
import asyncio

from settings.settings import logging_settings, model_settings, secret_settings
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

    # Defer heavy imports until after multiprocessing configuration
    from tokenizer.tokenizer_service import tokenizer_service
    from engine.model_loader import model_loader
    from scheduler.batch_scheduler import BatchScheduler
    from scheduler.continuous_scheduler import ContinuousScheduler
    from engine.generator import engine

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
    scheduler_task = asyncio.create_task(scheduler.run())
    batch_scheduler = BatchScheduler(engine, tokenizer_service)
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
    app = FastAPI(title="LLM Inference Server", version="0.1.0", lifespan=lifespan)
    app.include_router(router, prefix="/api")

    @app.get("/")
    def root():
        return {"message": "Welcome to the LLM Inference Server!"}

    @app.get("/health", response_model=HealthResponse)
    def health():
        logger.info("Health check endpoint called")
        return JSONResponse(status_code=200, content={"status": "healthy"})

    # Routers are included during lifespan after heavy imports

    return app

app = create_app()