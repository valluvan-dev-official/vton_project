import uuid
import shutil
from pathlib import Path

from fastapi import APIRouter, UploadFile, File, Depends, HTTPException
from sqlalchemy.ext.asyncio import AsyncSession

from app.database import get_db
from app.models.job import Job
from app.workers.tasks import process_tryon_job
from app.config import get_settings

router = APIRouter()
settings = get_settings()


@router.post("/tryon", summary="Submit a virtual try-on job")
async def submit_tryon(
    person_image: UploadFile = File(..., description="Person photo"),
    garment_image: UploadFile = File(..., description="Garment photo"),
    db: AsyncSession = Depends(get_db),
):
    job_id = str(uuid.uuid4())
    input_dir = Path(settings.LOCAL_STORAGE_PATH) / "inputs" / job_id
    input_dir.mkdir(parents=True, exist_ok=True)

    person_suffix = Path(person_image.filename or "person.jpg").suffix or ".jpg"
    garment_suffix = Path(garment_image.filename or "garment.jpg").suffix or ".jpg"
    person_path = str(input_dir / f"person{person_suffix}")
    garment_path = str(input_dir / f"garment{garment_suffix}")

    with open(person_path, "wb") as f:
        shutil.copyfileobj(person_image.file, f)
    with open(garment_path, "wb") as f:
        shutil.copyfileobj(garment_image.file, f)

    job = Job(
        id=job_id,
        person_image_path=person_path,
        garment_image_path=garment_path,
    )
    db.add(job)
    await db.commit()

    process_tryon_job.delay(job_id, person_path, garment_path)

    return {"job_id": job_id, "status": "pending"}
