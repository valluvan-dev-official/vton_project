import asyncio
import json
import logging
import uuid
from pathlib import Path

from typing import Optional

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


ALLOWED_CATEGORIES = {"upper_body", "lower_body", "dresses"}


def _validate_category(category: Optional[str]) -> str:
    """
    Which body region get_mask_location() should target. Defaults to
    upper_body — the same value every job was hardcoded to before this
    param existed — so callers that don't send it yet see no behavior change.
    """
    category = (category or "upper_body").strip().lower()
    if category not in ALLOWED_CATEGORIES:
        raise HTTPException(
            status_code=422,
            detail=f"category: must be one of {sorted(ALLOWED_CATEGORIES)}."
        )
    return category


ALLOWED_DRESS_SUBTYPES = {"saree", "salwar_suit"}


def _validate_dress_subtype(dress_subtype: Optional[str], category: str) -> Optional[str]:
    """
    Only meaningful when category == "dresses" — selects a dedicated
    sleeve-detection module (saree_handling.py / salwar_suit_handling.py)
    instead of the generic gown/frock dress_sleeve_detection.py default.
    Ignored (returned as None) for any other category, since e.g. a
    "salwar_suit" value on an upper_body job has nothing to act on.
    None/omitted preserves the pre-existing single "dresses" bucket
    behavior — the generic gown/frock handler — so existing callers that
    don't send it yet see no behavior change.
    """
    dress_subtype = (dress_subtype or "").strip().lower() or None
    if dress_subtype is None or category != "dresses":
        return None
    if dress_subtype not in ALLOWED_DRESS_SUBTYPES:
        raise HTTPException(
            status_code=422,
            detail=f"dress_subtype: must be one of {sorted(ALLOWED_DRESS_SUBTYPES)} (or omitted)."
        )
    return dress_subtype


# Sanity bounds only — not a measurement-accuracy claim. Used solely as an
# optional scale reference for Phase-1 shadow-mode fit analysis (see
# app/services/fit_analysis); never affects try-on rendering.
MIN_HEIGHT_CM, MAX_HEIGHT_CM = 100.0, 250.0


def _validate_height_cm(height_cm: Optional[float]) -> Optional[float]:
    if height_cm is None:
        return None
    if not (MIN_HEIGHT_CM <= height_cm <= MAX_HEIGHT_CM):
        raise HTTPException(
            status_code=422,
            detail=f"height_cm: must be between {MIN_HEIGHT_CM:.0f} and {MAX_HEIGHT_CM:.0f} if provided."
        )
    return height_cm


# Optional — identifies which merchant catalog entry (see
# app/services/garment_catalog.py) fit_analysis should look up for this
# job's garment_size. Used only for the Phase-1 shadow-mode fit_analysis
# metadata; never affects try-on rendering. Both must be provided together
# (a SKU with no merchant, or vice versa, can't resolve to a catalog row).
def _validate_merchant_sku(merchant: Optional[str], garment_sku: Optional[str]) -> tuple[Optional[str], Optional[str]]:
    merchant = (merchant or "").strip() or None
    garment_sku = (garment_sku or "").strip() or None
    if (merchant is None) != (garment_sku is None):
        raise HTTPException(
            status_code=422,
            detail="merchant and garment_sku must be provided together, or not at all.",
        )
    return merchant, garment_sku


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
    category: Optional[str] = Form(
        None, description="Optional — upper_body/lower_body/dresses. Tells the "
                           "model which body region this garment belongs to "
                           "(top, bottom, or a full one-piece). Defaults to "
                           "upper_body if omitted."
    ),
    dress_subtype: Optional[str] = Form(
        None, description="Optional, only meaningful when category=dresses — "
                           "saree/salwar_suit. Selects a dedicated sleeve-"
                           "detection module instead of the generic gown/frock "
                           "default. Ignored for any other category."
    ),
    height_cm: Optional[float] = Form(
        None, description="Optional — used only as a scale reference for the "
                           "Phase-1 shadow-mode fit_analysis metadata; never "
                           "affects try-on rendering."
    ),
    merchant: Optional[str] = Form(
        None, description="Optional, with garment_sku — identifies the merchant "
                           "catalog entry fit_analysis should use for this "
                           "garment_size. Never affects try-on rendering."
    ),
    garment_sku: Optional[str] = Form(
        None, description="Optional, with merchant — see merchant."
    ),
    db: AsyncSession = Depends(get_db),
):
    garment_size = _validate_garment_size(garment_size)
    category = _validate_category(category)
    dress_subtype = _validate_dress_subtype(dress_subtype, category)
    height_cm = _validate_height_cm(height_cm)
    merchant, garment_sku = _validate_merchant_sku(merchant, garment_sku)
    job_id = str(uuid.uuid4())

    person_ref, garment_refs = await _ingest_job_inputs(job_id, person_image, garment_images)

    job = Job(
        id=job_id,
        person_image_path=person_ref,
        garment_image_path=garment_refs[0],
        garment_image_paths=json.dumps(garment_refs),
        garment_size=garment_size,
        category=category,
    )
    db.add(job)
    await db.commit()

    process_tryon_job.delay(job_id, person_ref, garment_refs, garment_size, height_cm, merchant, garment_sku, category, dress_subtype)

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
    category: Optional[str] = Form(
        None, description="Optional — upper_body/lower_body/dresses. Defaults to upper_body."
    ),
    dress_subtype: Optional[str] = Form(
        None, description="Optional, only meaningful when category=dresses — "
                           "saree/salwar_suit. Selects a dedicated sleeve-"
                           "detection module instead of the generic gown/frock "
                           "default. Ignored for any other category."
    ),
    height_cm: Optional[float] = Form(
        None, description="Optional — used only as a scale reference for the "
                           "Phase-1 shadow-mode fit_analysis metadata; never "
                           "affects try-on rendering."
    ),
    merchant: Optional[str] = Form(
        None, description="Optional, with garment_sku — identifies the merchant "
                           "catalog entry fit_analysis should use for this "
                           "garment_size. Never affects try-on rendering."
    ),
    garment_sku: Optional[str] = Form(
        None, description="Optional, with merchant — see merchant."
    ),
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
    category = _validate_category(category)
    dress_subtype = _validate_dress_subtype(dress_subtype, category)
    height_cm = _validate_height_cm(height_cm)
    merchant, garment_sku = _validate_merchant_sku(merchant, garment_sku)
    job_id = str(uuid.uuid4())

    person_ref, garment_refs = await _ingest_job_inputs(job_id, person_image, garment_images)

    job = Job(
        id=job_id,
        person_image_path=person_ref,
        garment_image_path=garment_refs[0],
        garment_image_paths=json.dumps(garment_refs),
        garment_size=garment_size,
        category=category,
    )
    db.add(job)
    await db.commit()

    process_tryon_job.delay(job_id, person_ref, garment_refs, garment_size, height_cm, merchant, garment_sku, category, dress_subtype)

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
