from fastapi import APIRouter, Depends, HTTPException
from fastapi.responses import FileResponse
from sqlalchemy import select
from sqlalchemy.ext.asyncio import AsyncSession
from pathlib import Path

from app.database import get_db
from app.models.job import Job

router = APIRouter()


@router.get("/status/{job_id}", summary="Get job status")
async def get_status(job_id: str, db: AsyncSession = Depends(get_db)):
    result = await db.execute(select(Job).where(Job.id == job_id))
    job = result.scalar_one_or_none()
    if not job:
        raise HTTPException(status_code=404, detail="Job not found")
    return job.to_dict()


@router.get("/result/{job_id}", summary="Download result image")
async def get_result(job_id: str, db: AsyncSession = Depends(get_db)):
    result = await db.execute(select(Job).where(Job.id == job_id))
    job = result.scalar_one_or_none()
    if not job:
        raise HTTPException(status_code=404, detail="Job not found")
    if job.status != "completed" or not job.result_image_path:
        raise HTTPException(status_code=400, detail=f"Job is not completed yet (status: {job.status})")
    if not Path(job.result_image_path).exists():
        raise HTTPException(status_code=404, detail="Result file not found on disk")
    return FileResponse(job.result_image_path, media_type="image/jpeg")


@router.get("/jobs", summary="List recent jobs")
async def list_jobs(limit: int = 20, db: AsyncSession = Depends(get_db)):
    result = await db.execute(
        select(Job).order_by(Job.created_at.desc()).limit(limit)
    )
    jobs = result.scalars().all()
    return [j.to_dict() for j in jobs]
