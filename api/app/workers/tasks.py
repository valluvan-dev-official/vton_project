"""
Celery tasks for async try-on job processing.

Flow:
  1. Run inference (placeholder or real model)
  2. Calculate SSIM quality score vs the person image
  3. If score >= MIN_QUALITY_SCORE → auto-save as training pair + update pairs.json
  4. Update job status in DB
"""
import json
import sys
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
    """
    storage = get_storage()
    pair_dir = f"training_pairs/{job_id}"

    person_key  = f"{pair_dir}/person{Path(person_path).suffix}"
    garment_key = f"{pair_dir}/garment{Path(garment_path).suffix}"
    result_key  = f"{pair_dir}/result{Path(result_path).suffix}"

    storage.save(person_path,  person_key)
    storage.save(garment_path, garment_key)
    storage.save(result_path,  result_key)

    # Per-pair meta file
    meta = {
        "job_id":      job_id,
        "quality_score": score,
        "person":      person_key,
        "garment":     garment_key,
        "result":      result_key,
        "saved_at":    datetime.utcnow().isoformat(),
    }
    meta_path = Path(settings.LOCAL_STORAGE_PATH) / pair_dir / "meta.json"
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


# ── Main Celery task ──────────────────────────────────────────────────────────

@celery_app.task(bind=True, name="tasks.process_tryon_job", max_retries=2)
def process_tryon_job(self, job_id: str, person_image_path: str, garment_image_path: str):
    """
    Main try-on pipeline:
      1. Run inference (placeholder draws text + pastes garment, sleeps 3 s)
      2. SSIM quality score
      3. Auto-save training pair when score >= MIN_QUALITY_SCORE
      4. Update DB — completed / failed
    """
    try:
        _update_job(job_id, {"status": "processing"})

        # Translate Docker paths (/app/storage/...) to host paths when running outside Docker
        def _fix_path(p: str) -> str:
            if p.startswith("/app/storage/") and not Path(p).exists():
                return p.replace("/app/storage", str(settings.LOCAL_STORAGE_PATH), 1)
            return p

        person_image_path  = _fix_path(person_image_path)
        garment_image_path = _fix_path(garment_image_path)

        # 1. Inference
        output_path = str(
            Path(settings.LOCAL_STORAGE_PATH) / "outputs" / f"{job_id}.jpg"
        )
        router = get_inference_router()
        router.run(person_image_path, garment_image_path, output_path)

        # 2. SSIM score
        score = _compute_ssim(person_image_path, output_path)

        # 3. Training pair auto-save
        saved = False
        if score >= settings.MIN_QUALITY_SCORE:
            _save_training_pair(job_id, person_image_path, garment_image_path, output_path, score)
            saved = True

        # 4. Mark completed
        _update_job(job_id, {
            "status":           "completed",
            "result_image_path": output_path,
            "quality_score":    score,
            "saved_as_training": saved,
        })

    except Exception as exc:
        _update_job(job_id, {"status": "failed", "error_message": str(exc)})
        raise self.retry(exc=exc, countdown=30)
