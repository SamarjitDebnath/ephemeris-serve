from pydantic import BaseModel, Field, field_validator

class GenerateRequest(BaseModel):
    prompt: str = Field(..., min_length=1, description="The input text for the model")

    max_tokens: int | None = Field(
        default=None,
        ge=1,
        le=2048,
        description="Must be between 1 and 2048"
    )

    temperature: float | None = Field(
        default=None,
        ge=0.0,
        le=2.0,
        description="Standard LLM temperature range is usually 0.0 to 2.0"
    )

    # `sample()` already treats 0 / 1.0 as "filter disabled" (see
    # engine/generator.py), so those are the permissive ends of these ranges.
    # `top_p: 0.0` is excluded deliberately -- it would filter every token and
    # surface as an empty distribution mid-generation rather than a 422 here.
    top_k: int | None = Field(
        default=None,
        ge=0,
        description="Keep only the k highest-probability tokens; 0 disables top-k filtering",
    )

    top_p: float | None = Field(
        default=None,
        gt=0.0,
        le=1.0,
        description="Nucleus sampling mass; 1.0 disables top-p filtering",
    )

    idempotency_key: str | None = Field(
        default=None,
        max_length=200,
        description="Optional client-supplied key to deduplicate retried /generate requests",
    )

    stop: list[str] | None = Field(
        default=None,
        max_length=4,
        description="Up to 4 strings; generation halts before emitting any of them.",
    )

    @field_validator("stop")
    @classmethod
    def _reject_empty_stop_strings(cls, value: list[str] | None) -> list[str] | None:
        if value is None:
            return value
        if any(not s for s in value):
            raise ValueError("stop sequences must be non-empty strings")
        return value

class BatchGenerateRequest(BaseModel):
    requests: list[GenerateRequest] = Field(..., min_length=1, description="A list of generation requests to batch")

class BatchGenerateResponse(BaseModel):
    outputs: list[str] = Field(..., description="Decoded outputs for each item in the batch")
    batch_size: int = Field(..., description="Number of requests processed in the batch")
    queue_latency_ms: float | None = Field(
        default=None,
        description="Average queue latency in milliseconds for this batch; null if no queue latency data is available"
    )
    token_throughput_per_sec: float | None = Field(
        default=None,
        description="Token throughput measured in tokens per second; null if no throughput data is available"
    )

class HealthResponse(BaseModel):
    status: str = Field(..., description="The health status of the server")

class ModelSwapRequest(BaseModel):
    model_name: str = Field(..., min_length=1, description="Hugging Face model repo id to load")
    drain_timeout_seconds: float | None = Field(
        default=None,
        gt=0,
        description=(
            "Max seconds to wait for in-flight requests to finish before swapping; "
            "defaults to scheduler_config.model_swap_drain_timeout_seconds"
        ),
    )

class ModelSwapResponse(BaseModel):
    model_name: str = Field(..., description="The model currently loaded and serving requests")

    # A swap cannot be atomic across workers: each drains its own in-flight
    # requests before reloading, and those drains finish at different times.
    # Rather than hide that, the response reports it, so a client can poll to
    # completion instead of guessing. All three are null when cross-worker
    # coordination is disabled (`scheduler_config.model_state_dir` unset),
    # which is the single-process case where they would be meaningless.
    generation: int | None = Field(
        default=None,
        description="Monotone counter for the published model target; null when coordination is disabled",
    )
    converged_workers: int | None = Field(
        default=None,
        description="Workers that have reached this generation; null when coordination is disabled",
    )
    known_workers: int | None = Field(
        default=None,
        description="Workers that have reported any generation; null when coordination is disabled",
    )
