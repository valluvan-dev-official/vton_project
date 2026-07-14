"""
model_bootstrap.py — Automatic model weight provisioning.

Called once at worker startup (before GPUInferenceEngine is initialised).
Ensures that viton512.ckpt and warp_viton.pth are present in WEIGHTS_DIR.

Bootstrap logic
---------------
1. If both files already exist → return immediately (fast path, no S3 call).
2. If either file is missing  → download MODEL_S3_TAR from S3, extract it
   into WEIGHTS_DIR, then delete the tar file.

The download uses the EC2 Instance Profile (IAM role) automatically via
boto3's default credential chain — no AWS_ACCESS_KEY_ID / SECRET required.

Failure policy
--------------
Any failure (missing bucket, permissions error, extraction error, GPU
unavailable) raises immediately so the worker process never enters an
inconsistent state.  Celery's --pool=solo means a single process serves
all tasks; an uninitialised worker would silently fail every job.
"""
from __future__ import annotations

import logging
import os
import tarfile
from pathlib import Path

logger = logging.getLogger(__name__)

# Files that must exist for the model to be considered fully bootstrapped.
REQUIRED_WEIGHTS = ["viton512.ckpt", "warp_viton.pth"]


def _weights_ready(weights_dir: Path) -> bool:
    """Return True if every required checkpoint file is present."""
    return all((weights_dir / f).exists() for f in REQUIRED_WEIGHTS)


def _download_from_s3(s3_key: str, dest_path: Path, bucket: str, region: str) -> None:
    """Download s3://<bucket>/<s3_key> to dest_path using the EC2 IAM role."""
    import boto3  # imported lazily; boto3 must be in requirements-gpu.txt

    logger.info(
        "model_bootstrap: Downloading s3://%s/%s → %s ...",
        bucket, s3_key, dest_path,
    )
    dest_path.parent.mkdir(parents=True, exist_ok=True)

    s3 = boto3.client("s3", region_name=region)
    # boto3 uses the EC2 instance profile automatically when no explicit
    # credentials are supplied — no AWS_ACCESS_KEY_ID needed.
    s3.download_file(bucket, s3_key, str(dest_path))
    logger.info("model_bootstrap: Download complete.")


def _extract_tar(tar_path: Path, dest_dir: Path) -> None:
    """Extract a .tar.gz archive and delete it afterwards."""
    logger.info("model_bootstrap: Extracting %s → %s ...", tar_path, dest_dir)
    dest_dir.mkdir(parents=True, exist_ok=True)
    with tarfile.open(str(tar_path), "r:gz") as tf:
        tf.extractall(str(dest_dir))
    logger.info("model_bootstrap: Extraction complete.")

    tar_path.unlink()
    logger.info("model_bootstrap: Deleted tar file %s.", tar_path)


def ensure_weights(
    weights_dir: str | Path,
    model_s3_tar: str,
    s3_bucket: str,
    s3_region: str,
) -> None:
    """Guarantee that model weights exist in weights_dir.

    Parameters
    ----------
    weights_dir:
        Directory where viton512.ckpt and warp_viton.pth must live.
        Created automatically if it does not exist.
    model_s3_tar:
        S3 key of the model archive, e.g. ``model/model.tar.gz``.
    s3_bucket:
        Bucket name — must be accessible via the EC2 instance profile.
    s3_region:
        AWS region of the bucket, e.g. ``ap-south-1``.

    Raises
    ------
    RuntimeError
        If download, extraction, or post-extraction verification fails.
    """
    weights_dir = Path(weights_dir)
    weights_dir.mkdir(parents=True, exist_ok=True)

    if _weights_ready(weights_dir):
        logger.info(
            "model_bootstrap: Checkpoints already present in %s — skipping download.",
            weights_dir,
        )
        return

    missing = [f for f in REQUIRED_WEIGHTS if not (weights_dir / f).exists()]
    logger.info(
        "model_bootstrap: Missing weights: %s. Bootstrapping from S3...", missing
    )

    tar_dest = weights_dir / "model.tar.gz"
    try:
        _download_from_s3(model_s3_tar, tar_dest, s3_bucket, s3_region)
    except Exception as exc:
        raise RuntimeError(
            f"model_bootstrap: Failed to download s3://{s3_bucket}/{model_s3_tar}: {exc}"
        ) from exc

    try:
        _extract_tar(tar_dest, weights_dir)
    except Exception as exc:
        raise RuntimeError(
            f"model_bootstrap: Failed to extract {tar_dest}: {exc}"
        ) from exc

    if not _weights_ready(weights_dir):
        still_missing = [f for f in REQUIRED_WEIGHTS if not (weights_dir / f).exists()]
        raise RuntimeError(
            f"model_bootstrap: Archive extracted but required files still missing: "
            f"{still_missing}. Check that model.tar.gz contains the expected files."
        )

    logger.info(
        "model_bootstrap: All checkpoints are ready in %s.", weights_dir
    )
