import logging
from fastapi import APIRouter, Depends, HTTPException, Request
from fastapi.responses import FileResponse, RedirectResponse
from sqlalchemy import select
from sqlalchemy.ext.asyncio import AsyncSession
from pathlib import Path

from app.database import get_db
from app.models.job import Job
from app.config import get_settings
from app.services.storage import get_storage

router = APIRouter()
settings = get_settings()
logger = logging.getLogger(__name__)


def _make_result_url(request: Request, job: Job) -> str | None:
    """Return a browser-accessible URL for the result image.

    S3 backend
    ----------
    result_image_path contains either:
      - An S3 HTTPS URL  (https://<bucket>.s3.<region>.amazonaws.com/<key>)
      - An S3 key        (output/<job_id>.jpg)
    In both cases we return the public HTTPS URL.  If the bucket is private,
    generate a pre-signed URL instead (see generate_presigned_url in storage.py).

    Local backend
    -------------
    Falls back to the original /files/<relative-path> behaviour via the
    static file mount in main.py.
    """
    if not job.result_image_path:
        return None

    if settings.STORAGE_BACKEND.lower() == "s3":
        path = job.result_image_path
        # Already a full HTTPS URL
        if path.startswith("https://"):
            return path
        # S3 key — build the canonical URL
        return f"https://{settings.S3_BUCKET}.s3.{settings.S3_REGION}.amazonaws.com/{path}"

    # Local backend — original logic
    path_str = job.result_image_path.replace("\\", "/")
    marker = "storage/"
    idx = path_str.find(marker)
    if idx != -1:
        rel = path_str[idx + len(marker):]
        return str(request.base_url) + "files/" + rel
    return None


@router.get("/status/{job_id}", summary="Get job status")
async def get_status(job_id: str, request: Request, db: AsyncSession = Depends(get_db)):
    result = await db.execute(select(Job).where(Job.id == job_id))
    job = result.scalar_one_or_none()
    if not job:
        raise HTTPException(status_code=404, detail="Job not found.")
    return job.to_dict(result_url=_make_result_url(request, job))


@router.get("/result/{job_id}", summary="Download result image (binary)")
async def get_result(job_id: str, db: AsyncSession = Depends(get_db)):
    result = await db.execute(select(Job).where(Job.id == job_id))
    job = result.scalar_one_or_none()
    if not job:
        raise HTTPException(status_code=404, detail="Job not found.")
    if job.status.value != "completed" or not job.result_image_path:
        raise HTTPException(status_code=400, detail=f"Job not completed yet (status: {job.status.value}).")

    if settings.STORAGE_BACKEND.lower() == "s3":
        # For S3 backend: redirect to a pre-signed URL so the browser can
        # download the image directly from S3 without proxying through the API.
        from app.services.storage import S3StorageBackend
        storage = get_storage()
        if isinstance(storage, S3StorageBackend):
            path = job.result_image_path
            # Extract key from full URL if needed
            prefix = f"https://{settings.S3_BUCKET}.s3.{settings.S3_REGION}.amazonaws.com/"
            key = path[len(prefix):] if path.startswith(prefix) else path
            presigned = storage.generate_presigned_url(key, expires_in=3600)
            return RedirectResponse(url=presigned)

    # Local backend — original behaviour
    if not Path(job.result_image_path).exists():
        raise HTTPException(status_code=404, detail="Result file not found on disk.")
    return FileResponse(job.result_image_path, media_type="image/jpeg")


@router.get("/jobs", summary="List recent jobs")
async def list_jobs(limit: int = 20, request: Request = None, db: AsyncSession = Depends(get_db)):
    result = await db.execute(
        select(Job).order_by(Job.created_at.desc()).limit(limit)
    )
    jobs = result.scalars().all()
    return [j.to_dict(result_url=_make_result_url(request, j)) for j in jobs]
