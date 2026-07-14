import uuid
import tempfile
import shutil
from pathlib import Path

from fastapi import APIRouter, UploadFile, File, Depends, HTTPException, Request
from sqlalchemy.ext.asyncio import AsyncSession

from app.database import get_db
from app.models.job import Job
from app.workers.tasks import process_tryon_job
from app.config import get_settings
from app.services.storage import get_storage

router = APIRouter()
settings = get_settings()

ALLOWED_TYPES = {"image/jpeg", "image/jpg", "image/png"}
ALLOWED_EXTENSIONS = {".jpg", ".jpeg", ".png"}
MAX_BYTES = settings.MAX_UPLOAD_SIZE_MB * 1024 * 1024


def _validate_image(file: UploadFile, field_name: str) -> None:
    """Raise HTTP 422 for wrong type or oversized uploads."""
    ext = Path(file.filename or "").suffix.lower()
    content_type = (file.content_type or "").lower()

    if ext not in ALLOWED_EXTENSIONS and content_type not in ALLOWED_TYPES:
        raise HTTPException(
            status_code=422,
            detail=f"{field_name}: Only jpg/png files allowed."
        )


async def _read_and_check_size(file: UploadFile, field_name: str) -> bytes:
    """Read the full upload and raise 422 if it exceeds MAX_UPLOAD_SIZE_MB."""
    data = await file.read()
    if len(data) > MAX_BYTES:
        raise HTTPException(
            status_code=422,
            detail=f"{field_name}: File too large. Max {settings.MAX_UPLOAD_SIZE_MB}MB."
        )
    return data


@router.post("/tryon", summary="Submit a virtual try-on job")
async def submit_tryon(
    request: Request,
    person_image: UploadFile = File(..., description="Front-facing person photo (jpg/png, max 10MB)"),
    garment_image: UploadFile = File(..., description="Garment photo (jpg/png, max 10MB)"),
    db: AsyncSession = Depends(get_db),
):
    # Validate file types
    _validate_image(person_image, "person_image")
    _validate_image(garment_image, "garment_image")

    # Read and check sizes
    person_data  = await _read_and_check_size(person_image, "person_image")
    garment_data = await _read_and_check_size(garment_image, "garment_image")

    job_id = str(uuid.uuid4())

    person_suffix  = Path(person_image.filename  or "person.jpg").suffix.lower()  or ".jpg"
    garment_suffix = Path(garment_image.filename or "garment.jpg").suffix.lower() or ".jpg"

    storage = get_storage()

    if settings.STORAGE_BACKEND.lower() == "s3":
        # ── S3 path: write to a temp file, upload to S3, then delete temp ──
        # The task receives S3 keys, downloads them back to local temp files,
        # and cleans up after inference.
        import logging
        logger = logging.getLogger(__name__)

        tmp_dir = Path(settings.LOCAL_STORAGE_PATH) / "tmp"
        tmp_dir.mkdir(parents=True, exist_ok=True)

        person_tmp  = tmp_dir / f"{job_id}_person{person_suffix}"
        garment_tmp = tmp_dir / f"{job_id}_garment{garment_suffix}"

        try:
            person_tmp.write_bytes(person_data)
            garment_tmp.write_bytes(garment_data)

            person_s3_key  = f"{settings.S3_PREFIX_INPUT_PERSON}/{job_id}{person_suffix}"
            garment_s3_key = f"{settings.S3_PREFIX_INPUT_GARMENT}/{job_id}{garment_suffix}"

            logger.info("tryon: Uploading person image to S3 key %s...", person_s3_key)
            storage.save(str(person_tmp), person_s3_key)

            logger.info("tryon: Uploading garment image to S3 key %s...", garment_s3_key)
            storage.save(str(garment_tmp), garment_s3_key)

        finally:
            person_tmp.unlink(missing_ok=True)
            garment_tmp.unlink(missing_ok=True)

        # Pass S3 keys to the Celery task
        person_ref  = person_s3_key
        garment_ref = garment_s3_key

    else:
        # ── Local path: save directly to LOCAL_STORAGE_PATH (original behaviour) ──
        input_dir = Path(settings.LOCAL_STORAGE_PATH) / "inputs" / job_id
        input_dir.mkdir(parents=True, exist_ok=True)

        person_path  = str(input_dir / f"person{person_suffix}")
        garment_path = str(input_dir / f"garment{garment_suffix}")

        with open(person_path, "wb") as f:
            f.write(person_data)
        with open(garment_path, "wb") as f:
            f.write(garment_data)

        person_ref  = person_path
        garment_ref = garment_path

    # Create DB record — person_image_path / garment_image_path hold either
    # the S3 key or the local path depending on the backend.
    job = Job(
        id=job_id,
        person_image_path=person_ref,
        garment_image_path=garment_ref,
    )
    db.add(job)
    await db.commit()

    # Dispatch async task
    process_tryon_job.delay(job_id, person_ref, garment_ref)

    return {
        "job_id": job_id,
        "status": "pending",
        "eta_seconds": 35,
    }
