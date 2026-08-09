"""Garment-side measurement helpers: lookups + a small illustrative default
size chart. The merchant/product catalog is always the authoritative
source (Phase-1 spec item 5) — DEFAULT_TSHIRT_SIZE_CHART below exists only
so Phase-1 shadow-mode analysis produces a real (clearly-labeled) result
end-to-end before a real merchant-catalog integration exists. Once real
per-product catalog data is wired in (Phase 2+), it should be passed to
FitEngine directly instead of this chart, which then serves only as a test
fixture / fallback.

*** DEVELOPMENT / TEST FALLBACK ONLY — NOT FOR PRODUCTION USE. ***
DEFAULT_TSHIRT_SIZE_CHART is a made-up, illustrative chart, not tied to any
real merchant or product. This module intentionally does NOT gate its own
use — callers (currently only api/app/workers/tasks.py._compute_fit_analysis)
are responsible for only calling get_default_garment_measurements() when
explicitly permitted by settings.FIT_ANALYSIS_ALLOW_DEFAULT_CATALOG
(defaults to False). When that flag is off, or no real merchant data is
available, the caller must produce
FitEngine.insufficient_garment_data() instead of silently falling back to
this chart as if it were authoritative.
"""
from typing import Optional

from .models import MEASUREMENT_SOURCE_MERCHANT_PROVIDED, GarmentMeasurements

# Illustrative generic T-shirt chart (cm), broadly in line with common
# unisex retail size charts. Not tied to any specific merchant/product —
# do not treat as authoritative for a real garment. See module docstring:
# gated behind settings.FIT_ANALYSIS_ALLOW_DEFAULT_CATALOG by the caller.
DEFAULT_TSHIRT_SIZE_CHART: dict[str, GarmentMeasurements] = {
    "XS": GarmentMeasurements(size_label="XS", garment_type="t-shirt", chest_cm=86.0, shoulder_cm=40.0, length_cm=64.0, sleeve_length_cm=18.0),
    "S":  GarmentMeasurements(size_label="S",  garment_type="t-shirt", chest_cm=92.0, shoulder_cm=42.0, length_cm=66.0, sleeve_length_cm=19.0),
    "M":  GarmentMeasurements(size_label="M",  garment_type="t-shirt", chest_cm=100.0, shoulder_cm=44.0, length_cm=69.0, sleeve_length_cm=21.0),
    "L":  GarmentMeasurements(size_label="L",  garment_type="t-shirt", chest_cm=108.0, shoulder_cm=46.5, length_cm=71.0, sleeve_length_cm=22.0),
    "XL": GarmentMeasurements(size_label="XL", garment_type="t-shirt", chest_cm=116.0, shoulder_cm=49.0, length_cm=73.0, sleeve_length_cm=23.0),
    "XXL": GarmentMeasurements(size_label="XXL", garment_type="t-shirt", chest_cm=124.0, shoulder_cm=51.5, length_cm=75.0, sleeve_length_cm=24.0),
}


def get_default_garment_measurements(size_label: str) -> Optional[GarmentMeasurements]:
    """Look up the illustrative default chart by size label (case-insensitive).
    Returns None for an unrecognized label rather than guessing."""
    return DEFAULT_TSHIRT_SIZE_CHART.get((size_label or "").strip().upper())


def validate_garment_measurements(garment: GarmentMeasurements) -> list[str]:
    """Returns a list of human-readable warnings (empty = clean). Does not
    raise — GarmentMeasurements.__post_init__ already rejects structurally
    invalid values (non-positive numbers, empty size_label); this is for
    softer, advisory checks a caller may want to surface."""
    warnings: list[str] = []
    if garment.measurement_source == MEASUREMENT_SOURCE_MERCHANT_PROVIDED:
        if garment.completeness_ratio() == 0.0:
            warnings.append(
                f"Garment {garment.size_label!r} has no chest/shoulder/length "
                f"measurements — fit regions will report insufficient_data."
            )
    return warnings
