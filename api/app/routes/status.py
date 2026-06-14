from fastapi import APIRouter, Depends, HTTPException, Request
from fastapi.responses import FileResponse
from sqlalchemy import select
from sqlalchemy.ext.asyncio import AsyncSession
from pathlib import Path

from app.database import get_db
from app.models.job import Job
from app.config import get_settings

router = APIRouter()
settings = get_settings()


def _make_result_url(request: Request, job: Job) -> str | None:
    """Return a browser-accessible URL for the result image via /files static mount."""
    if not job.result_image_path:
        return None
    # Normalize to forward slashes so both Docker (/app/storage/...) and
    # host (c:/vton_project/storage/...) paths work.
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
