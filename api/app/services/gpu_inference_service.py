"""
gpu_inference_service.py — Singleton wrapper around GPUInferenceEngine.

This service owns the full GPU inference lifecycle:

  Bootstrap phase (once per worker process start):
    1. Verify CUDA is available — fail immediately if not.
    2. Call model_bootstrap.ensure_weights() — downloads checkpoints from S3
       if they are not already present in WEIGHTS_DIR.
    3. Construct GPUInferenceEngine — loads every model into GPU memory once.

  Inference phase (per request):
    4. Return the already-initialised engine; no reloading, no re-allocation.

Thread-safety
-------------
A threading.Lock guards the lazy-init path.  Once the engine is alive the
lock is released and never held again during inference.

Configuration (from environment / .env)
----------------------------------------
  DEVICE       — "cuda" (required on EC2); "cpu" for unit tests only
  WEIGHTS_DIR  — directory containing viton512.ckpt and warp_viton.pth
  WORKSPACE    — temp dir for third-party repo clones and working files
  S3_BUCKET    — bucket used for model bootstrap download
  S3_REGION    — AWS region of that bucket
  MODEL_S3_TAR — S3 key of the model archive (default: model/model.tar.gz)

Usage (inside a Celery task)
-----------------------------
    from app.services.gpu_inference_service import get_gpu_engine

    engine = get_gpu_engine()
    engine.run(person_path, garment_path, output_path, job_id=job_id)

    # or via the module-level convenience wrapper:
    import app.services.gpu_inference_service as _gpu_svc
    _gpu_svc.run(person_path, garment_path, output_path, job_id=job_id)
"""
from __future__ import annotations

import logging
import os
import sys
import threading
from pathlib import Path

logger = logging.getLogger(__name__)

# ── Module-level singleton state ─────────────────────────────────────────────
_engine = None
_engine_lock = threading.Lock()


# ── Internal helpers ─────────────────────────────────────────────────────────

def _add_ml_to_path() -> None:
    """Ensure /app/ml/scripts is on sys.path so GPUInferenceEngine is importable.

    The worker container mounts ../ml to /app/ml (see docker-compose.gpu.yml).
    We insert the path only once, guarded by a membership check.
    """
    ml_scripts = "/app/ml/scripts"
    if ml_scripts not in sys.path:
        sys.path.insert(0, ml_scripts)
        logger.debug("gpu_inference_service: added %s to sys.path", ml_scripts)


def _assert_cuda_available() -> None:
    """Raise RuntimeError immediately if CUDA is requested but unavailable."""
    device = os.getenv("DEVICE", "cuda").strip().lower()
    if device == "cuda":
        import torch
        if not torch.cuda.is_available():
            raise RuntimeError(
                "gpu_inference_service: DEVICE=cuda but torch.cuda.is_available() "
                "returned False.  Ensure the NVIDIA driver and nvidia-docker runtime "
                "are installed and the container was started with --gpus all."
            )
        gpu_name = torch.cuda.get_device_name(0)
        vram_gb  = torch.cuda.get_device_properties(0).total_memory / (1024 ** 3)
        logger.info(
            "gpu_inference_service: GPU detected — %s (%.1f GB VRAM).",
            gpu_name, vram_gb,
        )


def _run_bootstrap(weights_dir: str, s3_bucket: str, s3_region: str,
                   model_s3_tar: str) -> None:
    """Run the model weight bootstrap (download from S3 if needed)."""
    from app.services.model_bootstrap import ensure_weights
    ensure_weights(
        weights_dir  = weights_dir,
        model_s3_tar = model_s3_tar,
        s3_bucket    = s3_bucket,
        s3_region    = s3_region,
    )


def _build_engine():
    """Construct and return a GPUInferenceEngine.

    Steps (all fatal on failure):
      1. Assert CUDA is available.
      2. Bootstrap model weights from S3 if missing.
      3. Import and construct GPUInferenceEngine (loads models into GPU RAM).
    """
    device       = os.getenv("DEVICE",       "cuda").strip()
    weights_dir  = os.getenv("WEIGHTS_DIR",  "").strip()
    workspace    = os.getenv("WORKSPACE",    "/tmp/vton_workspace").strip()
    s3_bucket    = os.getenv("S3_BUCKET",    "").strip()
    s3_region    = os.getenv("S3_REGION",    "ap-south-1").strip()
    model_s3_tar = os.getenv("MODEL_S3_TAR", "model/model.tar.gz").strip()

    # ── 1. Validate required config ──
    if not weights_dir:
        raise RuntimeError(
            "gpu_inference_service: WEIGHTS_DIR is not set. "
            "Set it to the directory that will contain viton512.ckpt and warp_viton.pth."
        )

    # ── 2. GPU check ──
    logger.info("gpu_inference_service: Checking GPU availability...")
    _assert_cuda_available()

    # ── 3. Model bootstrap ──
    if s3_bucket:
        logger.info("gpu_inference_service: Running model bootstrap...")
        _run_bootstrap(weights_dir, s3_bucket, s3_region, model_s3_tar)
    else:
        logger.info(
            "gpu_inference_service: S3_BUCKET not set — skipping S3 bootstrap. "
            "Weights must already be present in %s.", weights_dir
        )
        if not Path(weights_dir).is_dir():
            raise RuntimeError(
                f"gpu_inference_service: WEIGHTS_DIR={weights_dir!r} does not exist "
                "and S3_BUCKET is not set, so weights cannot be downloaded."
            )

    # ── 4. Load ML models ──
    _add_ml_to_path()

    # Import lazily so this module can be imported in non-GPU environments
    # without immediately raising ImportError.
    from gpu_inference import GPUInferenceEngine  # noqa: PLC0415

    logger.info(
        "gpu_inference_service: Loading checkpoints (device=%s, weights_dir=%s, "
        "workspace=%s) — this may take 60-120 s on first boot...",
        device, weights_dir, workspace,
    )
    engine = GPUInferenceEngine(
        weights_dir = weights_dir,
        device      = device,
        workspace   = workspace,
    )
    logger.info("gpu_inference_service: GPU models loaded. Ready for inference.")
    return engine


# ── Public API ────────────────────────────────────────────────────────────────

def get_gpu_engine():
    """Return the process-level GPUInferenceEngine singleton.

    Thread-safe lazy initialisation: the engine is created on the first call
    and reused on every subsequent call.  Raises RuntimeError (never silently
    falls back) if the GPU, weights, or S3 access is misconfigured.
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


def run(person_path: str, garment_paths, output_path: str,
        job_id: str = "", garment_size: str = "M") -> str:
    """Convenience module-level function — mirror of GPUInferenceEngine.run().

    garment_paths may be a single path (str) or a list of paths — multiple
    photos of the same garment from different angles/zoom levels. The engine
    auto-picks the clearest one for inference.

    Ensures the output directory exists before delegating to the singleton
    engine so callers do not need to create it themselves.  Returns the
    auto-detected person body-size bucket (e.g. "M") used for fit scaling.
    """
    if isinstance(garment_paths, str):
        garment_paths = [garment_paths]
    Path(output_path).parent.mkdir(parents=True, exist_ok=True)
    engine = get_gpu_engine()
    engine.run(person_path, garment_paths, output_path, job_id=job_id, garment_size=garment_size)
    return engine.last_person_size_estimate
