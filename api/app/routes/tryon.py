import uuid
import shutil
from pathlib import Path

from fastapi import APIRouter, UploadFile, File, Depends, HTTPException, Request
from sqlalchemy.ext.asyncio import AsyncSession

from app.database import get_db
from app.models.job import Job
from app.workers.tasks import process_tryon_job
from app.config import get_settings

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
    person_data = await _read_and_check_size(person_image, "person_image")
    garment_data = await _read_and_check_size(garment_image, "garment_image")

    # Save to disk
    job_id = str(uuid.uuid4())
    input_dir = Path(settings.LOCAL_STORAGE_PATH) / "inputs" / job_id
    input_dir.mkdir(parents=True, exist_ok=True)

    person_suffix = Path(person_image.filename or "person.jpg").suffix.lower() or ".jpg"
    garment_suffix = Path(garment_image.filename or "garment.jpg").suffix.lower() or ".jpg"
    person_path = str(input_dir / f"person{person_suffix}")
    garment_path = str(input_dir / f"garment{garment_suffix}")

    with open(person_path, "wb") as f:
        f.write(person_data)
    with open(garment_path, "wb") as f:
        f.write(garment_data)

    # Create DB record
    job = Job(
        id=job_id,
        person_image_path=person_path,
        garment_image_path=garment_path,
    )
    db.add(job)
    await db.commit()

    # Dispatch async task
    process_tryon_job.delay(job_id, person_path, garment_path)

    return {
        "job_id": job_id,
        "status": "pending",
        "eta_seconds": 35,
    }
