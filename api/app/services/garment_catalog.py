"""DB-backed merchant garment size-chart access (Phase 2).

This is the ONLY place that bridges the SQLAlchemy-backed
MerchantGarmentSizeChart table to fit_analysis's framework-free
GarmentMeasurements dataclass — api/app/services/fit_analysis/ itself stays
free of any DB import, per its existing design (see models.py docstring).

Merchant-provided rows here are the sole authoritative source FitEngine is
allowed to treat as real sizing in production; the illustrative
DEFAULT_TSHIRT_SIZE_CHART in fit_analysis/garment_measurements.py is a
dev/test fallback only, gated separately by
settings.FIT_ANALYSIS_ALLOW_DEFAULT_CATALOG (see api/app/workers/tasks.py).
"""
from typing import Optional

from sqlalchemy import select
from sqlalchemy.ext.asyncio import AsyncSession

from app.models.garment_catalog import MerchantGarmentSizeChart, normalize_key
from app.services.fit_analysis.models import MEASUREMENT_SOURCE_MERCHANT_PROVIDED, GarmentMeasurements


def _normalize(value: str) -> str:
    return (value or "").strip()


async def get_size_chart_entry(
    db: AsyncSession, merchant: str, sku: str, size_label: str,
) -> Optional[MerchantGarmentSizeChart]:
    """Exact (merchant, sku, size_label) lookup, matched via the
    case/whitespace-insensitive *_key columns (see
    app/models/garment_catalog.py's normalization policy) — "AcmeApparel",
    "acmeapparel", and "  Acme Apparel " all resolve to the same row,
    matching the DB-level UNIQUE constraint exactly so a lookup can never
    miss a row the unique constraint would have refused to duplicate."""
    stmt = select(MerchantGarmentSizeChart).where(
        MerchantGarmentSizeChart.merchant_key == normalize_key(merchant),
        MerchantGarmentSizeChart.sku_key == normalize_key(sku),
        MerchantGarmentSizeChart.size_label_key == normalize_key(size_label),
    )
    result = await db.execute(stmt)
    return result.scalar_one_or_none()


async def list_size_chart_entries(
    db: AsyncSession, merchant: str, sku: str,
) -> list[MerchantGarmentSizeChart]:
    stmt = select(MerchantGarmentSizeChart).where(
        MerchantGarmentSizeChart.merchant_key == normalize_key(merchant),
        MerchantGarmentSizeChart.sku_key == normalize_key(sku),
    )
    result = await db.execute(stmt)
    return list(result.scalars().all())


async def upsert_size_chart_entry(
    db: AsyncSession,
    *,
    merchant: str,
    sku: str,
    size_label: str,
    garment_category: Optional[str] = None,
    chest_cm: Optional[float] = None,
    shoulder_cm: Optional[float] = None,
    waist_cm: Optional[float] = None,
    hip_cm: Optional[float] = None,
    length_cm: Optional[float] = None,
    sleeve_length_cm: Optional[float] = None,
    stretch_category: str = "none",
    fit_style: str = "regular",
) -> MerchantGarmentSizeChart:
    """Create or update the row for (merchant, sku, size_label). Does not
    commit — caller controls the transaction boundary (matches this
    codebase's existing route-layer convention, e.g. app/routes/fit.py)."""
    existing = await get_size_chart_entry(db, merchant, sku, size_label)

    fields = dict(
        garment_category=garment_category,
        chest_cm=chest_cm, shoulder_cm=shoulder_cm, waist_cm=waist_cm, hip_cm=hip_cm,
        length_cm=length_cm, sleeve_length_cm=sleeve_length_cm,
        stretch_category=stretch_category, fit_style=fit_style,
    )

    if existing is not None:
        for key, value in fields.items():
            setattr(existing, key, value)
        await db.flush()
        return existing

    row = MerchantGarmentSizeChart(
        merchant=_normalize(merchant), sku=_normalize(sku), size_label=_normalize(size_label),
        **fields,
    )
    db.add(row)
    await db.flush()
    return row


def to_garment_measurements(row: MerchantGarmentSizeChart) -> GarmentMeasurements:
    """Map a DB row to the dataclass FitEngine actually consumes."""
    return GarmentMeasurements(
        size_label=row.size_label,
        garment_type=row.garment_category,
        chest_cm=row.chest_cm,
        shoulder_cm=row.shoulder_cm,
        waist_cm=row.waist_cm,
        hip_cm=row.hip_cm,
        length_cm=row.length_cm,
        sleeve_length_cm=row.sleeve_length_cm,
        stretch_category=row.stretch_category,
        fit_style=row.fit_style,
        measurement_source=MEASUREMENT_SOURCE_MERCHANT_PROVIDED,
    )
