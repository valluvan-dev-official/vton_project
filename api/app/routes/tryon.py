import asyncio
import json
import logging
import uuid
from pathlib import Path

from fastapi import APIRouter, UploadFile, File, Form, Depends, HTTPException, Request
from fastapi.responses import Response, RedirectResponse
from sqlalchemy import select
from sqlalchemy.ext.asyncio import AsyncSession

from app.database import get_db, AsyncSessionLocal
from app.models.job import Job
from app.workers.tasks import process_tryon_job
from app.config import get_settings
from app.services.storage import get_storage

router = APIRouter()
settings = get_settings()
logger = logging.getLogger(__name__)

ALLOWED_TYPES = {"image/jpeg", "image/jpg", "image/png"}
ALLOWED_EXTENSIONS = {".jpg", ".jpeg", ".png"}
MAX_BYTES = settings.MAX_UPLOAD_SIZE_MB * 1024 * 1024
ALLOWED_GARMENT_SIZES = {"XS", "S", "M", "L", "XL", "XXL"}
MAX_GARMENT_IMAGES = 6

# Synchronous endpoint: how long to wait for the GPU worker before giving up.
SYNC_POLL_INTERVAL_SECONDS = 2
SYNC_POLL_TIMEOUT_SECONDS = 180


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


def _validate_garment_size(garment_size: str) -> str:
    garment_size = (garment_size or "M").strip().upper()
    if garment_size not in ALLOWED_GARMENT_SIZES:
        raise HTTPException(
            status_code=422,
            detail=f"garment_size: must be one of {sorted(ALLOWED_GARMENT_SIZES)}."
        )
    return garment_size


async def _ingest_job_inputs(
    job_id: str,
    person_image: UploadFile,
    garment_images: list[UploadFile],
) -> tuple[str, list[str]]:
    """Validate + persist the uploads, return (person_ref, garment_refs).

    Each ref is either an S3 key or a local path, depending on
    STORAGE_BACKEND — the same convention already used by the rest of the
    pipeline (Celery tasks resolve either kind transparently).
    """
    if not garment_images:
        raise HTTPException(status_code=422, detail="garment_images: at least one image required.")
    if len(garment_images) > MAX_GARMENT_IMAGES:
        raise HTTPException(
            status_code=422,
            detail=f"garment_images: max {MAX_GARMENT_IMAGES} images per job."
        )

    _validate_image(person_image, "person_image")
    for i, f in enumerate(garment_images):
        _validate_image(f, f"garment_images[{i}]")

    person_data = await _read_and_check_size(person_image, "person_image")
    garment_data_list = [
        await _read_and_check_size(f, f"garment_images[{i}]")
        for i, f in enumerate(garment_images)
    ]

    person_suffix = Path(person_image.filename or "person.jpg").suffix.lower() or ".jpg"
    garment_suffixes = [
        Path(f.filename or f"garment_{i}.jpg").suffix.lower() or ".jpg"
        for i, f in enumerate(garment_images)
    ]

    storage = get_storage()

    if settings.STORAGE_BACKEND.lower() == "s3":
        tmp_dir = Path(settings.LOCAL_STORAGE_PATH) / "tmp"
        tmp_dir.mkdir(parents=True, exist_ok=True)

        person_tmp = tmp_dir / f"{job_id}_person{person_suffix}"
        garment_tmps = [
            tmp_dir / f"{job_id}_garment{i}{suffix}"
            for i, suffix in enumerate(garment_suffixes)
        ]

        try:
            person_tmp.write_bytes(person_data)
            for tmp, data in zip(garment_tmps, garment_data_list):
                tmp.write_bytes(data)

            person_s3_key = f"{settings.S3_PREFIX_INPUT_PERSON}/{job_id}{person_suffix}"
            logger.info("tryon: Uploading person image to S3 key %s...", person_s3_key)
            storage.save(str(person_tmp), person_s3_key)

            garment_s3_keys = []
            for i, (tmp, suffix) in enumerate(zip(garment_tmps, garment_suffixes)):
                key = f"{settings.S3_PREFIX_INPUT_GARMENT}/{job_id}_{i}{suffix}"
                logger.info("tryon: Uploading garment image %d to S3 key %s...", i, key)
                storage.save(str(tmp), key)
                garment_s3_keys.append(key)
        finally:
            person_tmp.unlink(missing_ok=True)
            for tmp in garment_tmps:
                tmp.unlink(missing_ok=True)

        return person_s3_key, garment_s3_keys

    # ── Local backend ──
    input_dir = Path(settings.LOCAL_STORAGE_PATH) / "inputs" / job_id
    input_dir.mkdir(parents=True, exist_ok=True)

    person_path = str(input_dir / f"person{person_suffix}")
    with open(person_path, "wb") as fh:
        fh.write(person_data)

    garment_paths = []
    for i, (data, suffix) in enumerate(zip(garment_data_list, garment_suffixes)):
        p = str(input_dir / f"garment_{i}{suffix}")
        with open(p, "wb") as fh:
            fh.write(data)
        garment_paths.append(p)

    return person_path, garment_paths


@router.post("/tryon", summary="Submit a virtual try-on job (async)")
async def submit_tryon(
    request: Request,
    person_image: UploadFile = File(..., description="Front-facing person photo (jpg/png, max 10MB)"),
    garment_images: list[UploadFile] = File(
        ..., description=f"1-{MAX_GARMENT_IMAGES} photos of the SAME garment from different "
                          "angles/zoom levels (front, back, close-up of fabric/print, etc.). "
                          "The clearest one is auto-selected for inference."
    ),
    garment_size: str = Form("M", description="Garment size label: XS/S/M/L/XL/XXL"),
    db: AsyncSession = Depends(get_db),
):
    garment_size = _validate_garment_size(garment_size)
    job_id = str(uuid.uuid4())

    person_ref, garment_refs = await _ingest_job_inputs(job_id, person_image, garment_images)

    job = Job(
        id=job_id,
        person_image_path=person_ref,
        garment_image_path=garment_refs[0],
        garment_image_paths=json.dumps(garment_refs),
        garment_size=garment_size,
    )
    db.add(job)
    await db.commit()

    process_tryon_job.delay(job_id, person_ref, garment_refs, garment_size)

    return {
        "job_id": job_id,
        "status": "pending",
        "eta_seconds": 35,
        "garment_images_received": len(garment_refs),
    }


@router.post(
    "/tryon/sync",
    summary="Submit a try-on job and block until the result image is ready",
    response_class=Response,
)
async def submit_tryon_sync(
    request: Request,
    person_image: UploadFile = File(..., description="Front-facing person photo (jpg/png, max 10MB)"),
    garment_images: list[UploadFile] = File(
        ..., description=f"1-{MAX_GARMENT_IMAGES} photos of the SAME garment from different angles."
    ),
    garment_size: str = Form("M", description="Garment size label: XS/S/M/L/XL/XXL"),
    db: AsyncSession = Depends(get_db),
):
    """For programmatic integrations that want a single request/response
    instead of submit+poll: this dispatches the same Celery job, then blocks
    the HTTP request (polling the DB every 2s, up to 180s) and returns the
    raw result image bytes directly — Content-Type: image/jpeg.

    Use /tryon + /status + /result instead if your client can't hold a
    long-lived connection, or if you're firing many jobs concurrently.
    """
    garment_size = _validate_garment_size(garment_size)
    job_id = str(uuid.uuid4())

    person_ref, garment_refs = await _ingest_job_inputs(job_id, person_image, garment_images)

    job = Job(
        id=job_id,
        person_image_path=person_ref,
        garment_image_path=garment_refs[0],
        garment_image_paths=json.dumps(garment_refs),
        garment_size=garment_size,
    )
    db.add(job)
    await db.commit()

    process_tryon_job.delay(job_id, person_ref, garment_refs, garment_size)

    # Poll a fresh session each time so we see committed updates from the worker.
    elapsed = 0
    while elapsed < SYNC_POLL_TIMEOUT_SECONDS:
        await asyncio.sleep(SYNC_POLL_INTERVAL_SECONDS)
        elapsed += SYNC_POLL_INTERVAL_SECONDS
        async with AsyncSessionLocal() as session:
            result = await session.execute(select(Job).where(Job.id == job_id))
            job = result.scalar_one_or_none()
        if job is None:
            raise HTTPException(status_code=500, detail="Job disappeared during processing.")
        if job.status.value == "completed":
            break
        if job.status.value == "failed":
            raise HTTPException(status_code=502, detail=f"Inference failed: {job.error_message}")
    else:
        raise HTTPException(
            status_code=504,
            detail=f"Timed out after {SYNC_POLL_TIMEOUT_SECONDS}s waiting for job {job_id}. "
                   f"Poll /api/v1/status/{job_id} instead.",
        )

    # ── Fetch result bytes and return them directly ──
    if settings.STORAGE_BACKEND.lower() == "s3":
        from app.services.storage import S3StorageBackend
        storage = get_storage()
        if isinstance(storage, S3StorageBackend):
            prefix = f"https://{settings.S3_BUCKET}.s3.{settings.S3_REGION}.amazonaws.com/"
            key = job.result_image_path
            key = key[len(prefix):] if key.startswith(prefix) else key
            tmp_path = Path(settings.LOCAL_STORAGE_PATH) / "tmp" / f"{job_id}_result.jpg"
            tmp_path.parent.mkdir(parents=True, exist_ok=True)
            storage.load(key, str(tmp_path))
            data = tmp_path.read_bytes()
            tmp_path.unlink(missing_ok=True)
            return Response(content=data, media_type="image/jpeg",
                             headers={"X-Job-Id": job_id, "X-Person-Size-Estimate": job.person_size_estimate or ""})

    result_path = Path(job.result_image_path)
    if not result_path.exists():
        raise HTTPException(status_code=404, detail="Result file not found on disk.")
    return Response(content=result_path.read_bytes(), media_type="image/jpeg",
                     headers={"X-Job-Id": job_id, "X-Person-Size-Estimate": job.person_size_estimate or ""})
