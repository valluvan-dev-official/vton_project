import logging
import uuid
from pathlib import Path

from fastapi import APIRouter, UploadFile, File, Form, Depends, HTTPException, Request
from sqlalchemy.ext.asyncio import AsyncSession

from app.database import get_db
from app.models.job import Job
from app.workers.tasks import process_accessory_job
from app.config import get_settings
from app.services.accessory_engine import ACCESSORY_ENGINES
from app.routes.tryon import _ingest_job_inputs

router = APIRouter()
settings = get_settings()
logger = logging.getLogger(__name__)

ALLOWED_ACCESSORY_TYPES = set(ACCESSORY_ENGINES.keys())


def _validate_accessory_type(accessory_type: str) -> str:
    accessory_type = (accessory_type or "").strip().lower()
    if accessory_type not in ALLOWED_ACCESSORY_TYPES:
        raise HTTPException(
            status_code=422,
            detail=f"accessory_type: must be one of {sorted(ALLOWED_ACCESSORY_TYPES)} "
                   f"(others are planned but not implemented yet).",
        )
    return accessory_type


@router.post("/accessory-tryon", summary="Submit an accessory try-on job (async) — wrist, glasses, ...")
async def submit_accessory_tryon(
    request: Request,
    person_image: UploadFile = File(..., description="Front-facing person photo (jpg/png, max 10MB)"),
    accessory_image: UploadFile = File(..., description="The product's try-on cutout image (background-removed PNG)"),
    accessory_type: str = Form(..., description=f"One of {sorted(ALLOWED_ACCESSORY_TYPES)}"),
    db: AsyncSession = Depends(get_db),
):
    """
    Async sibling of /tryon, for non-garment engines (see the PRD's
    multi-engine plan) — same Job table, same /status/{job_id} and
    /result/{job_id} endpoints as garment jobs, just a different
    submission shape (single accessory image, no garment_size/category)
    and a much cheaper CPU compositing task instead of a GPU diffusion job.
    """
    accessory_type = _validate_accessory_type(accessory_type)
    job_id = str(uuid.uuid4())

    # Reuses the exact same ingestion/storage logic as garment jobs — the
    # accessory image is passed as a 1-element "garment_images" list purely
    # to reuse _ingest_job_inputs' validation/S3-upload code, not because
    # it's semantically a garment.
    person_ref, accessory_refs = await _ingest_job_inputs(job_id, person_image, [accessory_image])

    job = Job(
        id=job_id,
        person_image_path=person_ref,
        garment_image_path=accessory_refs[0],
        garment_image_paths=None,
        category=accessory_type,
    )
    db.add(job)
    await db.commit()

    process_accessory_job.delay(job_id, person_ref, accessory_refs[0], accessory_type)

    return {
        "job_id": job_id,
        "status": "pending",
        "eta_seconds": 5,
    }


@router.post(
    "/accessory-tryon/validate",
    summary="Pre-flight check — can this photo be used for the given accessory type?",
)
async def validate_accessory_photo(
    person_image: UploadFile = File(..., description="Front-facing person photo (jpg/png, max 10MB)"),
    accessory_type: str = Form(..., description=f"One of {sorted(ALLOWED_ACCESSORY_TYPES)}"),
):
    """
    Synchronous, no Celery job created — runs the landmark detector
    immediately and returns ok/reason. Lets the backend reject a photo with
    a specific, user-facing message ("No wrist visible in this photo")
    before ever submitting a job, instead of the caller waiting on a poll
    that's guaranteed to fail.
    """
    accessory_type = _validate_accessory_type(accessory_type)

    data = await person_image.read()
    tmp_dir = Path(settings.LOCAL_STORAGE_PATH) / "tmp"
    tmp_dir.mkdir(parents=True, exist_ok=True)
    suffix = Path(person_image.filename or "person.jpg").suffix.lower() or ".jpg"
    tmp_path = tmp_dir / f"validate_{uuid.uuid4()}{suffix}"
    try:
        tmp_path.write_bytes(data)
        engine = ACCESSORY_ENGINES[accessory_type]()
        ok, reason = engine.validate(str(tmp_path))
        return {"ok": ok, "reason": reason}
    finally:
        tmp_path.unlink(missing_ok=True)
