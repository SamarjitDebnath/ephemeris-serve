import torch

# Proactively clear once PyTorch's allocated+cached memory reaches this
# fraction of the device's budget. Conservative on purpose: the point is to
# release memory *before* it ever gets close to the device's actual ceiling
# (observed as e.g. `MPS backend out of memory`), not to wait for a fixed
# step count or for a failure to happen first.
DEFAULT_MEMORY_PRESSURE_THRESHOLD = 0.7


def empty_device_cache(device: str) -> None:
    """Release cached-but-unused memory back to the system for `device`.

    PyTorch's CUDA/MPS allocators cache freed tensor memory for reuse rather
    than returning it to the OS immediately -- usually a throughput win, but
    over a long-running server process the cached (not necessarily live)
    portion can accumulate until the device's memory ceiling is hit, even
    though most of it isn't backing any tensor still in use. Called after a
    failed generation step (before retrying), whenever the scheduler goes
    idle, and proactively via `maybe_empty_device_cache` below.
    """
    if device.startswith("cuda") and torch.cuda.is_available():
        torch.cuda.empty_cache()
    elif device == "mps" and torch.backends.mps.is_available():
        torch.mps.empty_cache()


def device_memory_pressure(device: str) -> float | None:
    """Fraction of `device`'s memory budget currently held by PyTorch
    (allocated + cached), or `None` if it can't be determined -- CPU, an
    unavailable backend, or a torch build missing these APIs.

    For MPS, "budget" is `torch.mps.recommended_max_memory()` (the OS's
    recommended Metal working-set size) and "held" is
    `torch.mps.driver_allocated_memory()` (total GPU memory the process has
    from the driver, including cached allocator pools) -- the same figure
    that shows up as "MPS allocated" in an MPS OOM error. For CUDA, the
    analogous `memory_reserved()` / device total memory.
    """
    try:
        if device.startswith("cuda") and torch.cuda.is_available():
            idx = torch.cuda.current_device()
            total = torch.cuda.get_device_properties(idx).total_memory
            if not total:
                return None
            return torch.cuda.memory_reserved(idx) / total
        if device == "mps" and torch.backends.mps.is_available():
            budget = torch.mps.recommended_max_memory()
            if not budget:
                return None
            return torch.mps.driver_allocated_memory() / budget
    except Exception:
        return None
    return None


def maybe_empty_device_cache(device: str, threshold: float = DEFAULT_MEMORY_PRESSURE_THRESHOLD) -> bool:
    """Clear `device`'s cached memory if current usage is at/above `threshold`.

    Returns whether it actually cleared. Safe to call every scheduler step:
    the pressure check itself is just a metadata query (no device sync), so
    only the clear itself -- when actually triggered -- does real work.
    Devices where pressure can't be measured (see `device_memory_pressure`)
    are left alone here; they still get cache clears from the event-driven
    call sites (idle, retry-on-failure).
    """
    pressure = device_memory_pressure(device)
    if pressure is not None and pressure >= threshold:
        empty_device_cache(device)
        return True
    return False
