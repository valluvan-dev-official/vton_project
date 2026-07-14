"""
gpu_inference_service.py — Singleton wrapper around GPUInferenceEngine.

This service owns the GPUInferenceEngine lifecycle:
  - The engine is constructed ONCE (lazy, on first call) and held for the
    lifetime of the Celery worker process.
  - Subsequent requests reuse the already-loaded GPU models — no reload.
  - Thread-safety: a threading.Lock guards the lazy-init path; once the
    engine is alive the lock is never held during inference.

Configuration (read from .env / environment):
  DEVICE      — "cuda" (default on GPU EC2) or "cpu" for testing
  WEIGHTS_DIR — absolute path to the directory containing viton512.ckpt
                and warp_viton.pth (mounted into the container at /app/ml/weights)
  WORKSPACE   — working directory for repos / temp files
                (default /tmp/vton_workspace)

Usage (inside a Celery task):
    from app.services.gpu_inference_service import get_gpu_engine

    engine = get_gpu_engine()
    engine.run(person_path, garment_path, output_path, job_id=job_id)
"""
from __future__ import annotations

import logging
import os
import sys
import threading
from pathlib import Path
from typing import Optional

logger = logging.getLogger(__name__)

# ── Module-level singleton state ─────────────────────────────────────────────
_engine = None
_engine_lock = threading.Lock()


def _add_ml_to_path() -> None:
    """Ensure /app/ml/scripts is on sys.path so GPUInferenceEngine is importable.

    The worker container mounts ../ml to /app/ml (see docker-compose.gpu.yml).
    We insert the path only once, guarded by a membership check.
    """
    ml_scripts = "/app/ml/scripts"
    if ml_scripts not in sys.path:
        sys.path.insert(0, ml_scripts)
        logger.debug("gpu_inference_service: added %s to sys.path", ml_scripts)


def _build_engine():
    """Construct and return a GPUInferenceEngine.

    Reads DEVICE, WEIGHTS_DIR, and WORKSPACE from the environment so the
    service works identically in Docker (via docker-compose.gpu.yml env vars)
    and on a bare EC2 instance (via shell environment).
    """
    device      = os.getenv("DEVICE",      "cuda").strip()
    weights_dir = os.getenv("WEIGHTS_DIR", "").strip()
    workspace   = os.getenv("WORKSPACE",   "/tmp/vton_workspace").strip()

    if not weights_dir:
        raise RuntimeError(
            "WEIGHTS_DIR is not set. "
            "Point it at the directory containing viton512.ckpt and warp_viton.pth."
        )
    if not Path(weights_dir).is_dir():
        raise RuntimeError(
            f"WEIGHTS_DIR={weights_dir!r} does not exist or is not a directory."
        )

    _add_ml_to_path()

    # Import here (not at module level) so the service file can be imported
    # in any context without immediately failing if torch/ML deps are absent.
    from gpu_inference import GPUInferenceEngine  # noqa: PLC0415

    logger.info(
        "gpu_inference_service: initialising GPUInferenceEngine "
        "(device=%s, weights_dir=%s, workspace=%s)",
        device, weights_dir, workspace,
    )
    engine = GPUInferenceEngine(
        weights_dir=weights_dir,
        device=device,
        workspace=workspace,
    )
    logger.info("gpu_inference_service: GPUInferenceEngine ready.")
    return engine


def get_gpu_engine():
    """Return the process-level GPUInferenceEngine singleton.

    Thread-safe lazy initialisation: the engine is created on the first call
    and reused on all subsequent calls.  Raises RuntimeError if WEIGHTS_DIR
    is missing or DEVICE is misconfigured.
    """
    global _engine
    if _engine is not None:
        return _engine
    with _engine_lock:
        # Double-checked locking: another thread may have initialised while
        # we were waiting for the lock.
        if _engine is None:
            _engine = _build_engine()
    return _engine


def run(person_path: str, garment_path: str, output_path: str,
        job_id: str = "") -> str:
    """Convenience top-level function — mirror of GPUInferenceEngine.run().

    Ensures the output directory exists before delegating to the engine so
    callers do not need to create it themselves.
    """
    Path(output_path).parent.mkdir(parents=True, exist_ok=True)
    engine = get_gpu_engine()
    return engine.run(person_path, garment_path, output_path, job_id=job_id)
