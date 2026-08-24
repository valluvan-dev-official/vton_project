"""
GPU pipeline arbiter — mutual-exclusion for heavy models sharing one GPU.

Why this exists: the GPU worker holds a single 22-24GB card, and
GPUInferenceEngine (IDM-VTON, ~21GB resident) is pre-loaded eagerly at
worker startup and never unloaded. When HandbagGripEngine (a second,
independent SDXL pipeline) tried to load on top of that for a "handbag"
accessory job, there wasn't enough free VRAM left and the load failed with
CUDA OutOfMemoryError — see the incident this module was added to fix.

Garment try-on and handbag try-on jobs never need to run at the same
instant on this worker (Celery processes one task at a time per GPU
process), so the fix is to swap: whichever pipeline is about to be used
evicts whatever else currently holds the GPU first. This trades latency
(reloading a pipeline costs 1-4 minutes) for correctness — there is
currently no code path where both pipelines are needed simultaneously.

Usage: each heavy engine calls `get_arbiter().register(name, unload_fn)`
once (idempotent — re-registering just overwrites the callback), then
`get_arbiter().acquire(name)` right before it loads/uses its pipeline. The
arbiter calls the OTHER currently-loaded pipeline's unload_fn (if any) the
first time a different name acquires the GPU.

If a real second GPU or a memory-fits-both configuration is set up later,
this becomes a no-op by simply not calling acquire()/register() — nothing
elsewhere needs to change.
"""
import logging
import threading

logger = logging.getLogger(__name__)


class GPUPipelineArbiter:
    def __init__(self):
        self._lock = threading.Lock()
        self._current_owner: str | None = None
        self._unload_fns: dict[str, callable] = {}

    def register(self, name: str, unload_fn) -> None:
        self._unload_fns[name] = unload_fn

    def acquire(self, name: str) -> None:
        """Block until `name` is the sole resident pipeline on the GPU —
        evicting whatever else is currently loaded first, if anything."""
        with self._lock:
            if self._current_owner == name:
                return
            if self._current_owner is not None:
                unload = self._unload_fns.get(self._current_owner)
                if unload is not None:
                    logger.info(
                        "GPUPipelineArbiter: evicting '%s' to free VRAM for '%s'...",
                        self._current_owner, name,
                    )
                    unload()
                    logger.info("GPUPipelineArbiter: '%s' evicted.", self._current_owner)
            self._current_owner = name


_arbiter: GPUPipelineArbiter | None = None


def get_arbiter() -> GPUPipelineArbiter:
    global _arbiter
    if _arbiter is None:
        _arbiter = GPUPipelineArbiter()
    return _arbiter
