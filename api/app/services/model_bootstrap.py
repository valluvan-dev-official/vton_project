"""
model_bootstrap.py — Automatic model weight provisioning for IDM-VTON.

Bootstrap logic
---------------
1. If IDM-VTON weights already present → return immediately.
2. If missing → download idm_model.tar.gz from S3, extract, delete tar.

Uses EC2 Instance Profile (IAM role) via boto3 — no explicit AWS keys needed.
"""
from __future__ import annotations

import logging
import tarfile
from pathlib import Path

logger = logging.getLogger(__name__)

# IDM-VTON weight directories that must exist after extraction
REQUIRED_DIRS = ["idm_vton/unet", "idm_vton/image_encoder", "idm_vton/vae"]


def _weights_ready(weights_dir: Path) -> bool:
    return all((weights_dir / d).is_dir() for d in REQUIRED_DIRS)


def _download_from_s3(s3_key: str, dest_path: Path, bucket: str, region: str) -> None:
    import boto3
    logger.info("model_bootstrap: Downloading s3://%s/%s → %s ...", bucket, s3_key, dest_path)
    dest_path.parent.mkdir(parents=True, exist_ok=True)
    s3 = boto3.client("s3", region_name=region)
    s3.download_file(bucket, s3_key, str(dest_path))
    logger.info("model_bootstrap: Download complete.")


def _extract_tar(tar_path: Path, dest_dir: Path) -> None:
    logger.info("model_bootstrap: Extracting %s → %s ...", tar_path, dest_dir)
    dest_dir.mkdir(parents=True, exist_ok=True)
    with tarfile.open(str(tar_path), "r:gz") as tf:
        tf.extractall(str(dest_dir))
    tar_path.unlink()
    logger.info("model_bootstrap: Extraction complete, tar deleted.")


def ensure_weights(
    weights_dir: str | Path,
    model_s3_tar: str,
    s3_bucket: str,
    s3_region: str,
) -> None:
    weights_dir = Path(weights_dir)
    weights_dir.mkdir(parents=True, exist_ok=True)

    if _weights_ready(weights_dir):
        logger.info("model_bootstrap: IDM-VTON weights already present — skipping download.")
        return

    missing = [d for d in REQUIRED_DIRS if not (weights_dir / d).is_dir()]
    logger.info("model_bootstrap: Missing: %s. Downloading from S3...", missing)

    tar_dest = weights_dir / "idm_model.tar.gz"
    try:
        _download_from_s3(model_s3_tar, tar_dest, s3_bucket, s3_region)
    except Exception as exc:
        raise RuntimeError(
            f"model_bootstrap: Failed to download s3://{s3_bucket}/{model_s3_tar}: {exc}"
        ) from exc

    try:
        _extract_tar(tar_dest, weights_dir)
    except Exception as exc:
        raise RuntimeError(f"model_bootstrap: Extraction failed: {exc}") from exc

    if not _weights_ready(weights_dir):
        still_missing = [d for d in REQUIRED_DIRS if not (weights_dir / d).is_dir()]
        raise RuntimeError(
            f"model_bootstrap: Extracted but still missing: {still_missing}"
        )

    logger.info("model_bootstrap: IDM-VTON weights ready in %s.", weights_dir)
