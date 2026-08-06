from typing import Literal, Optional

from pydantic import BaseModel, Field

StretchCategory = Literal["none", "low", "medium", "high"]
FitStyle = Literal["slim", "regular", "relaxed", "oversized"]


class BodyMeasurementsIn(BaseModel):
    height_cm: float = Field(..., ge=100, le=250)
    # Bust/waist/hips stay optional per Phase 1 requirements — a missing
    # value produces "insufficient_data" downstream, never a rejected request.
    bust_cm: Optional[float] = Field(default=None, ge=40, le=200)
    waist_cm: Optional[float] = Field(default=None, ge=40, le=200)
    hips_cm: Optional[float] = Field(default=None, ge=40, le=200)
    shoulder_cm: Optional[float] = Field(default=None, ge=20, le=70)


class GarmentMeasurementsIn(BaseModel):
    size_label: str = Field(..., min_length=1, max_length=16)
    bust_cm: Optional[float] = Field(default=None, ge=40, le=200)
    waist_cm: Optional[float] = Field(default=None, ge=40, le=200)
    hips_cm: Optional[float] = Field(default=None, ge=40, le=200)
    shoulder_cm: Optional[float] = Field(default=None, ge=20, le=70)
    length_cm: Optional[float] = Field(default=None, ge=10, le=150)
    sleeve_cm: Optional[float] = Field(default=None, ge=0, le=100)
    stretch_category: StretchCategory = "none"
    fit_style: FitStyle = "regular"


class FitEstimateRequest(BaseModel):
    job_id: Optional[str] = None
    body: BodyMeasurementsIn
    selected_size: str = Field(..., min_length=1, max_length=16)
    garment_category: Optional[str] = Field(default=None, max_length=64)
    # Inline garment size chart — no product catalog required for Phase 1.
    # A future revision can additionally resolve garment_category against
    # GarmentSizeChart rows in the DB; this endpoint already persists chart
    # rows it's given so that catalog can be built up incrementally.
    garment_measurements: list[GarmentMeasurementsIn] = Field(default_factory=list)


class RegionalFitOut(BaseModel):
    ease_cm: Optional[float]
    fit: str


class FitEstimateResponse(BaseModel):
    id: str
    job_id: Optional[str]
    selected_size: str
    recommended_size: Optional[str]
    body_shape: str
    confidence: float
    regional_fit: dict[str, RegionalFitOut]
    summary: str
    is_approximate: bool
    disclaimer: str
