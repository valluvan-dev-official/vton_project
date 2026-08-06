"""
Celery tasks for async try-on job processing.

Flow:
  1. Download person + garment(s) from S3 to temp local files
  2. Run inference via InferenceRouter — IDM-VTON, own trained model, or
     placeholder, whichever OWN_MODEL_CHECKPOINT / DEVICE / WEIGHTS_DIR
     selects (see app/services/inference.py). Swapping models is therefore
     a config change, not a code change.
  3. Upload result to S3
  4. Calculate SSIM quality score
  5. If score >= MIN_QUALITY_SCORE → auto-save as training pair + update pairs.json
  6. Update job status in DB (result_image_path stores the S3 key)

When STORAGE_BACKEND=local the download/upload steps are no-ops that just
copy from/to LOCAL_STORAGE_PATH, so the local dev workflow is unchanged.
"""
import json
import os
import sys
import tempfile
import uuid

if sys.platform != "win32":
    import fcntl
from datetime import datetime
from pathlib import Path

from celery import Celery
from skimage.metrics import structural_similarity as ssim
import numpy as np

from app.config import get_settings
from app.services.inference import get_inference_router
from app.services.storage import get_storage

settings = get_settings()

celery_app = Celery(
    "vton",
    broker=settings.CELERY_BROKER_URL,
    backend=settings.CELERY_RESULT_BACKEND,
)
celery_app.conf.update(
    task_serializer="json",
    result_serializer="json",
    accept_content=["json"],
    timezone="UTC",
    enable_utc=True,
)


# ── DB helpers ────────────────────────────────────────────────────────────────

def _update_job(job_id: str, updates: dict):
    """Synchronous DB update run inside a Celery task via asyncio."""
    import asyncio
    from sqlalchemy import update
    from sqlalchemy.ext.asyncio import create_async_engine, async_sessionmaker
    from app.models.job import Job

    engine = create_async_engine(settings.DATABASE_URL)
    Session = async_sessionmaker(engine, expire_on_commit=False)

    async def _run():
        async with Session() as session:
            await session.execute(update(Job).where(Job.id == job_id).values(**updates))
            await session.commit()
        await engine.dispose()

    asyncio.run(_run())


# ── Quality scoring ───────────────────────────────────────────────────────────

def _compute_ssim(path_a: str, path_b: str) -> float:
    from PIL import Image
    def load(p):
        img = Image.open(p).convert("RGB").resize((256, 256))
        return np.array(img)
    a, b = load(path_a), load(path_b)
    score, _ = ssim(a, b, full=True, channel_axis=2, data_range=255)
    return float(score)


# ── Training pair persistence ─────────────────────────────────────────────────

def _save_training_pair(job_id: str, person_path: str, garment_path: str,
                        result_path: str, score: float):
    """
    Copy the trio to storage/training_pairs/<job_id>/ and append to pairs.json.
    ml/src/data/dataset.py reads from this directory.
    With S3 backend the files are uploaded to S3_PREFIX_TRAINING/<job_id>/.
    """
    storage = get_storage()
    pair_dir = f"{settings.S3_PREFIX_TRAINING}/{job_id}"

    person_key  = f"{pair_dir}/person{Path(person_path).suffix}"
    garment_key = f"{pair_dir}/garment{Path(garment_path).suffix}"
    result_key  = f"{pair_dir}/result{Path(result_path).suffix}"

    storage.save(person_path,  person_key)
    storage.save(garment_path, garment_key)
    storage.save(result_path,  result_key)

    # Per-pair meta file — written locally and also uploaded
    meta = {
        "job_id":        job_id,
        "quality_score": score,
        "person":        person_key,
        "garment":       garment_key,
        "result":        result_key,
        "saved_at":      datetime.utcnow().isoformat(),
    }
    meta_path = Path(settings.LOCAL_STORAGE_PATH) / "training_pairs" / job_id / "meta.json"
    meta_path.parent.mkdir(parents=True, exist_ok=True)
    meta_path.write_text(json.dumps(meta, indent=2))

    # Central index — pairs.json (file-lock for concurrent workers)
    pairs_json = Path(settings.LOCAL_STORAGE_PATH) / "training_pairs" / "pairs.json"
    pairs_json.parent.mkdir(parents=True, exist_ok=True)

    # Read → append → write with an exclusive lock
    if sys.platform != "win32":
        try:
            with open(str(pairs_json), "a+") as fh:
                try:
                    fcntl.flock(fh, fcntl.LOCK_EX)
                    fh.seek(0)
                    raw = fh.read().strip()
                    entries = json.loads(raw) if raw else []
                    entries.append(meta)
                    fh.seek(0)
                    fh.truncate()
                    json.dump(entries, fh, indent=2)
                finally:
                    fcntl.flock(fh, fcntl.LOCK_UN)
        except Exception:
            existing = json.loads(pairs_json.read_text()) if pairs_json.exists() else []
            existing.append(meta)
            pairs_json.write_text(json.dumps(existing, indent=2))
    else:
        existing = json.loads(pairs_json.read_text()) if pairs_json.exists() else []
        existing.append(meta)
        pairs_json.write_text(json.dumps(existing, indent=2))


# ── S3 input helpers ──────────────────────────────────────────────────────────

def _resolve_local_path(s3_key_or_path: str, job_id: str, role: str) -> str:
    """Ensure the image is available as a local file and return its path.

    If STORAGE_BACKEND=s3 and the value looks like an S3 key (no leading /),
    download it to a temp file and return that path.

    If STORAGE_BACKEND=local or the path already exists, return it as-is
    (after the /app/storage path-fix for Docker vs host differences).
    """
    p = s3_key_or_path

    # Docker path fix (kept from original implementation)
    if p.startswith("/app/storage/") and not Path(p).exists():
        p = p.replace("/app/storage", str(settings.LOCAL_STORAGE_PATH), 1)

    if Path(p).exists():
        return p

    # Treat as S3 key — download to a temp file
    import logging
    logger = logging.getLogger(__name__)
    storage = get_storage()
    suffix = Path(p).suffix or ".jpg"
    tmp = tempfile.NamedTemporaryFile(
        suffix=suffix, delete=False,
        dir=Path(settings.LOCAL_STORAGE_PATH) / "tmp",
        prefix=f"{job_id}_{role}_",
    )
    tmp.close()
    logger.info("tasks: Downloading input (%s) from storage key %s...", role, p)
    storage.load(p, tmp.name)
    return tmp.name


# ── Main Celery task ──────────────────────────────────────────────────────────

@celery_app.task(bind=True, name="tasks.process_tryon_job", max_retries=2)
def process_tryon_job(self, job_id: str, person_image_path: str,
                       garment_image_paths, garment_size: str = "M"):
    """
    Main try-on pipeline:
      1. Resolve input images (download from S3 if needed)
      2. Run GPU inference — singleton engine, models loaded once
      3. Upload result to S3
      4. SSIM quality score
      5. Auto-save training pair when score >= MIN_QUALITY_SCORE
      6. Update DB — completed / failed
    """
    import logging
    logger = logging.getLogger(__name__)
    tmp_files: list[str] = []   # track temp files to clean up

    # Backward-compat: accept a single string path (old callers/tests) as
    # well as the new list-of-paths (multiple garment angles).
    if isinstance(garment_image_paths, str):
        garment_image_paths = [garment_image_paths]

    try:
        _update_job(job_id, {"status": "processing"})

        # ── 1. Resolve local paths for inference ──────────────────────────
        # Create tmp dir inside LOCAL_STORAGE_PATH (always accessible)
        tmp_dir = Path(settings.LOCAL_STORAGE_PATH) / "tmp"
        tmp_dir.mkdir(parents=True, exist_ok=True)

        local_person = _resolve_local_path(person_image_path, job_id, "person")
        local_garments = [
            _resolve_local_path(p, job_id, f"garment{i}")
            for i, p in enumerate(garment_image_paths)
        ]

        # Track any temp files created by _resolve_local_path
        for p in [local_person, *local_garments]:
            if str(tmp_dir) in p:
                tmp_files.append(p)

        # ── 2. Inference ──────────────────────────────────────────────────
        output_path = str(tmp_dir / f"{job_id}_output.jpg")
        logger.info(
            "tasks: Starting inference for job %s (%d garment image(s), garment_size=%s)...",
            job_id, len(local_garments), garment_size,
        )
        router = get_inference_router()
        router.run(
            local_person, local_garments, output_path, job_id=job_id, garment_size=garment_size,
        )
        person_size_estimate = router.last_person_size_estimate
        logger.info(
            "tasks: Inference complete for job %s (person_size_estimate=%s).",
            job_id, person_size_estimate,
        )

        # ── 3. Upload result to S3 ────────────────────────────────────────
        result_s3_key = f"{settings.S3_PREFIX_OUTPUT}/{job_id}.jpg"
        storage = get_storage()
        logger.info("tasks: Uploading output to S3 key %s...", result_s3_key)
        storage.save(output_path, result_s3_key)
        logger.info("tasks: Output uploaded — s3://%s/%s", settings.S3_BUCKET, result_s3_key)

        # result_image_path stores the S3 key (or local path in local mode)
        result_image_path = storage.url(result_s3_key)

        # ── 4. SSIM score ─────────────────────────────────────────────────
        score = _compute_ssim(local_person, output_path)
        logger.info("tasks: SSIM score for job %s: %.4f", job_id, score)

        # ── 5. Training pair auto-save ────────────────────────────────────
        saved = False
        if score >= settings.MIN_QUALITY_SCORE:
            _save_training_pair(job_id, local_person, local_garments[0], output_path, score)
            saved = True

        # ── 6. Mark completed ─────────────────────────────────────────────
        _update_job(job_id, {
            "status":               "completed",
            "result_image_path":    result_image_path,
            "quality_score":        score,
            "saved_as_training":    saved,
            "person_size_estimate": person_size_estimate,
        })

    except Exception as exc:
        _update_job(job_id, {"status": "failed", "error_message": str(exc)})
        raise self.retry(exc=exc, countdown=30)

    finally:
        # Clean up temp files created for S3-backend input downloads
        for p in tmp_files:
            try:
                Path(p).unlink(missing_ok=True)
            except Exception:
                pass
