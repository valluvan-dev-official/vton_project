import logging

from fastapi import APIRouter, Depends, HTTPException, Query
from sqlalchemy.ext.asyncio import AsyncSession

from app.database import get_db
from app.schemas.garment_catalog import GarmentSizeChartIn, GarmentSizeChartOut
from app.services import garment_catalog as svc

router = APIRouter()
logger = logging.getLogger(__name__)


@router.post(
    "/garment-catalog/sizes",
    summary="Create or update one merchant SKU's size-chart entry",
    response_model=GarmentSizeChartOut,
)
async def upsert_size(payload: GarmentSizeChartIn, db: AsyncSession = Depends(get_db)):
    row = await svc.upsert_size_chart_entry(
        db,
        merchant=payload.merchant,
        sku=payload.sku,
        size_label=payload.size_label,
        garment_category=payload.garment_category,
        chest_cm=payload.chest_cm,
        shoulder_cm=payload.shoulder_cm,
        waist_cm=payload.waist_cm,
        hip_cm=payload.hip_cm,
        length_cm=payload.length_cm,
        sleeve_length_cm=payload.sleeve_length_cm,
        stretch_category=payload.stretch_category,
        fit_style=payload.fit_style,
    )
    await db.commit()
    await db.refresh(row)
    return GarmentSizeChartOut(**row.to_dict())


@router.get(
    "/garment-catalog/sizes",
    summary="List all size-chart entries for a merchant's SKU",
    response_model=list[GarmentSizeChartOut],
)
async def list_sizes(
    merchant: str = Query(..., min_length=1),
    sku: str = Query(..., min_length=1),
    db: AsyncSession = Depends(get_db),
):
    rows = await svc.list_size_chart_entries(db, merchant, sku)
    return [GarmentSizeChartOut(**r.to_dict()) for r in rows]


@router.get(
    "/garment-catalog/sizes/{merchant}/{sku}/{size_label}",
    summary="Get one merchant SKU's measurements for one size label",
    response_model=GarmentSizeChartOut,
)
async def get_size(merchant: str, sku: str, size_label: str, db: AsyncSession = Depends(get_db)):
    row = await svc.get_size_chart_entry(db, merchant, sku, size_label)
    if row is None:
        raise HTTPException(
            status_code=404,
            detail=f"No size-chart entry for merchant={merchant!r} sku={sku!r} size_label={size_label!r}.",
        )
    return GarmentSizeChartOut(**row.to_dict())
