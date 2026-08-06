import logging

from fastapi import APIRouter, Depends, HTTPException
from sqlalchemy import select
from sqlalchemy.ext.asyncio import AsyncSession

from app.database import get_db
from app.models.job import Job
from app.models.fit import BodyProfile, FitEstimate
from app.schemas.fit import FitEstimateRequest, FitEstimateResponse
from app.services import fit_calculator as fc

router = APIRouter()
logger = logging.getLogger(__name__)


@router.post(
    "/fit/estimate",
    summary="Estimate body shape and regional garment fit (Phase 1)",
    response_model=FitEstimateResponse,
)
async def estimate_fit(payload: FitEstimateRequest, db: AsyncSession = Depends(get_db)):
    if payload.job_id is not None:
        result = await db.execute(select(Job.id).where(Job.id == payload.job_id))
        if result.scalar_one_or_none() is None:
            raise HTTPException(status_code=404, detail=f"job_id {payload.job_id!r} not found.")

    body = fc.BodyMeasurements(
        height_cm=payload.body.height_cm,
        bust_cm=payload.body.bust_cm,
        waist_cm=payload.body.waist_cm,
        hips_cm=payload.body.hips_cm,
        shoulder_cm=payload.body.shoulder_cm,
    )
    garment_chart = [
        fc.GarmentMeasurements(
            size_label=g.size_label,
            bust_cm=g.bust_cm,
            waist_cm=g.waist_cm,
            hips_cm=g.hips_cm,
            shoulder_cm=g.shoulder_cm,
            length_cm=g.length_cm,
            sleeve_cm=g.sleeve_cm,
            stretch_category=g.stretch_category,
            fit_style=g.fit_style,
        )
        for g in payload.garment_measurements
    ]

    estimate = fc.estimate_fit(body, payload.selected_size, garment_chart)

    body_profile = BodyProfile(
        job_id=payload.job_id,
        height_cm=payload.body.height_cm,
        bust_cm=payload.body.bust_cm,
        waist_cm=payload.body.waist_cm,
        hips_cm=payload.body.hips_cm,
        shoulder_cm=payload.body.shoulder_cm,
    )
    db.add(body_profile)
    await db.flush()  # assigns body_profile.id within the same transaction

    fit_row = FitEstimate(
        job_id=payload.job_id,
        body_profile_id=body_profile.id,
        selected_size=estimate.selected_size,
        recommended_size=estimate.recommended_size,
        body_shape=estimate.body_shape,
        confidence=estimate.confidence,
        regional_fit=estimate.regional_fit,
        summary=estimate.summary,
        is_approximate=estimate.is_approximate,
    )
    db.add(fit_row)
    # Single commit — BodyProfile and FitEstimate are persisted atomically;
    # a failure here rolls back both via the AsyncSession context manager.
    await db.commit()
    await db.refresh(fit_row)

    return FitEstimateResponse(
        id=fit_row.id,
        job_id=fit_row.job_id,
        selected_size=fit_row.selected_size,
        recommended_size=fit_row.recommended_size,
        body_shape=fit_row.body_shape,
        confidence=fit_row.confidence,
        regional_fit=fit_row.regional_fit,
        summary=fit_row.summary,
        is_approximate=fit_row.is_approximate,
        disclaimer=fc.DISCLAIMER,
    )
