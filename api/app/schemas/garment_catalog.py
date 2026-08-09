from typing import Literal, Optional

from pydantic import BaseModel, Field

StretchCategory = Literal["none", "low", "medium", "high"]
FitStyle = Literal["slim", "regular", "relaxed", "oversized"]


class GarmentSizeChartIn(BaseModel):
    merchant: str = Field(..., min_length=1, max_length=128)
    sku: str = Field(..., min_length=1, max_length=128)
    # Free-text — arbitrary merchant size labels are explicitly supported,
    # not restricted to XS/S/M/L/XL/XXL.
    size_label: str = Field(..., min_length=1, max_length=32)
    garment_category: Optional[str] = Field(default=None, max_length=64)

    chest_cm: Optional[float] = Field(default=None, gt=0, le=250)
    shoulder_cm: Optional[float] = Field(default=None, gt=0, le=100)
    waist_cm: Optional[float] = Field(default=None, gt=0, le=250)
    hip_cm: Optional[float] = Field(default=None, gt=0, le=250)
    length_cm: Optional[float] = Field(default=None, gt=0, le=200)
    sleeve_length_cm: Optional[float] = Field(default=None, gt=0, le=120)

    stretch_category: StretchCategory = "none"
    fit_style: FitStyle = "regular"


class GarmentSizeChartOut(GarmentSizeChartIn):
    id: str
    created_at: str
    updated_at: str
