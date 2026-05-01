"""
Celery tasks for async try-on job processing.

Flow:
  1. Run inference (placeholder or real model)
  2. Calculate SSIM quality score vs. the person image
  3. If score >= threshold → auto-save as training pair
  4. Update job record in DB
"""
import os
import uuid
import json
from pathlib import Path
from datetime import datetime

from celery import Celery
from skimage.metrics import structural_similarity as ssim
from skimage import io as skio
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


def _update_job_in_db(job_id: str, updates: dict):
    """Synchronous DB update called from within a Celery task."""
    import asyncio
    from sqlalchemy import update
    from sqlalchemy.ext.asyncio import create_async_engine, async_sessionmaker
    from app.models.job import Job

    engine = create_async_engine(settings.DATABASE_URL)
    SessionLocal = async_sessionmaker(engine, expire_on_commit=False)

    async def _run():
        async with SessionLocal() as session:
            await session.execute(update(Job).where(Job.id == job_id).values(**updates))
            await session.commit()
        await engine.dispose()

    asyncio.run(_run())


def _compute_ssim(img_path_a: str, img_path_b: str) -> float:
    """Return SSIM score [0, 1] between two images (resized to 256x256)."""
    from PIL import Image
    import cv2

    def load(p):
        img = Image.open(p).convert("RGB").resize((256, 256))
        return np.array(img)

    a, b = load(img_path_a), load(img_path_b)
    score, _ = ssim(a, b, full=True, channel_axis=2, data_range=255)
    return float(score)


def _save_training_pair(job_id: str, person_path: str, garment_path: str, result_path: str, score: float):
    """
    Copy the trio into storage/training_pairs/<job_id>/ and write a metadata JSON.
    ml/src/training/train.py reads from this directory.
    """
    storage = get_storage()
    pair_dir = f"training_pairs/{job_id}"

    person_key = f"{pair_dir}/person{Path(person_path).suffix}"
    garment_key = f"{pair_dir}/garment{Path(garment_path).suffix}"
    result_key = f"{pair_dir}/result{Path(result_path).suffix}"

    storage.save(person_path, person_key)
    storage.save(garment_path, garment_key)
    storage.save(result_path, result_key)

    meta = {
        "job_id": job_id,
        "ssim_score": score,
        "person": person_key,
        "garment": garment_key,
        "result": result_key,
        "saved_at": datetime.utcnow().isoformat(),
    }

    meta_path = Path(settings.LOCAL_STORAGE_PATH) / pair_dir / "meta.json"
    meta_path.parent.mkdir(parents=True, exist_ok=True)
    meta_path.write_text(json.dumps(meta, indent=2))


@celery_app.task(bind=True, name="tasks.process_tryon_job", max_retries=2)
def process_tryon_job(self, job_id: str, person_image_path: str, garment_image_path: str):
    """
    Main try-on task:
      - Runs inference
      - Scores result via SSIM
      - Saves as training pair when score >= TRAINING_PAIR_SSIM_THRESHOLD
      - Updates job status in PostgreSQL
    """
    try:
        _update_job_in_db(job_id, {"status": "processing"})

        # 1. Inference
        output_path = str(
            Path(settings.LOCAL_STORAGE_PATH) / "outputs" / f"{job_id}.jpg"
        )
        router = get_inference_router()
        router.run(person_image_path, garment_image_path, output_path)

        # 2. Quality score
        score = _compute_ssim(person_image_path, output_path)

        # 3. Auto-save training pair
        saved_as_pair = False
        if score >= settings.TRAINING_PAIR_SSIM_THRESHOLD:
            _save_training_pair(job_id, person_image_path, garment_image_path, output_path, score)
            saved_as_pair = True

        # 4. Mark completed
        _update_job_in_db(
            job_id,
            {
                "status": "completed",
                "result_image_path": output_path,
                "ssim_score": score,
                "saved_as_training_pair": saved_as_pair,
            },
        )

    except Exception as exc:
        _update_job_in_db(job_id, {"status": "failed", "error_message": str(exc)})
        raise self.retry(exc=exc, countdown=30)
