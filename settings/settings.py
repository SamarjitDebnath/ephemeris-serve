import os

import torch
from utils.utils import *
from pydantic_settings import BaseSettings


def resolve_device(configured: str) -> str:
    """Return the best available compute device.

    If ``configured`` is ``"auto"``, auto-detect in priority order:
    1. CUDA - NVIDIA GPU           (torch.cuda.is_available)
    2. MPS  - Apple Silicon / Metal (torch.backends.mps.is_available)
    3. CPU  - universal fallback

    Any other value is returned as-is so the operator can always pin a
    specific device (e.g. ``"cpu"``, ``"cuda:1"``, ``"mps"``).
    """
    if configured != "auto":
        return configured
    if torch.cuda.is_available():
        return "cuda"
    if torch.backends.mps.is_available():
        return "mps"
    return "cpu"


class ModelSetting:
    def __init__(self, config_path: str = "settings/config.yaml"):
        config = Utils.load_config(config_path)["model_config"]["defaults"]

        # EPHEMERIS_MODEL_NAME lets a caller (e.g. `ephemeris-serve serve --model`)
        # pick the model for this run without editing config.yaml -- read via env
        # var, not an in-process singleton mutation, so it survives uvicorn
        # spawning fresh worker processes that re-import settings from scratch.
        self.model_name = os.environ.get("EPHEMERIS_MODEL_NAME") or config["model_name"]
        self.device = resolve_device(config["device"])
        self.max_length = config["max_length"]
        self.temperature = config["temperature"]
        self.top_k = config["top_k"]
        self.top_p = config["top_p"]
        self.repetition_penalty = config["repetition_penalty"]
        self.num_return_sequences = config["num_return_sequences"]

class LoggingSetting:
    def __init__(self, config_path: str = "settings/config.yaml"):
        config = Utils.load_config(config_path)["logging_config"]["defaults"]

        self.log_level = config["log_level"]
        self.log_file = config["log_file"]


class SchedulerSetting:
    def __init__(self, config_path: str = "settings/config.yaml"):
        config = Utils.load_config(config_path)["scheduler_config"]["defaults"]

        self.streaming_request_timeout_seconds = config["streaming_request_timeout_seconds"]
        self.batch_request_timeout_seconds = config["batch_request_timeout_seconds"]
        self.batch_generation_timeout_seconds = config["batch_generation_timeout_seconds"]
        self.idempotency_key_ttl_seconds = config["idempotency_key_ttl_seconds"]
        self.model_swap_drain_timeout_seconds = config.get("model_swap_drain_timeout_seconds", 30.0)


class CacheSetting:
    def __init__(self, config_path: str = "settings/config.yaml"):
        config = Utils.load_config(config_path)["cache_config"]["defaults"]

        self.kv_block_size = config["kv_block_size"]


class SecretSetting(BaseSettings):
    hf_key: str | None = ""

    model_config = {
        "env_file": ".env",
        "env_file_encoding": "utf-8",
    }

model_settings = ModelSetting()
logging_settings = LoggingSetting()
scheduler_settings = SchedulerSetting()
cache_settings = CacheSetting()
secret_settings = SecretSetting()
