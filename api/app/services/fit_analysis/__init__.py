"""fit_analysis — Phase 1, SHADOW MODE ONLY.

Computes an estimated body-measurement + garment-fit analysis from a photo
(and optional height), independently of the IDM-VTON render path. Nothing
in this package is imported by, or feeds into, ml/scripts/gpu_inference.py:
no mask, no garment scaling, no diffusion input is affected by anything
computed here. See FitResult.is_shadow_mode in models.py.

Public API:
    BodyMeasurementEstimate, GarmentMeasurements, FitResult, RegionFit
    BodyAnalyzer      — photo (+landmarks, +height) -> BodyMeasurementEstimate
    FitEngine         — (BodyMeasurementEstimate, GarmentMeasurements) -> FitResult
    PoseLandmarks, EngineReadOnlyPoseAdapter, try_get_pose_engine
    get_default_garment_measurements
"""
from .body_analyzer import BodyAnalyzer
from .fit_engine import FitEngine
from .garment_measurements import DEFAULT_TSHIRT_SIZE_CHART, get_default_garment_measurements
from .models import (
    MEASUREMENT_SOURCE_ESTIMATED,
    MEASUREMENT_SOURCE_ILLUSTRATIVE_DEFAULT,
    MEASUREMENT_SOURCE_MERCHANT_PROVIDED,
    MEASUREMENT_SOURCE_USER_PROVIDED,
    BodyMeasurementEstimate,
    FitResult,
    GarmentMeasurements,
    RegionFit,
)
from .pose_adapter import EngineReadOnlyPoseAdapter, PoseLandmarks, try_get_pose_engine

__all__ = [
    "BodyMeasurementEstimate",
    "GarmentMeasurements",
    "FitResult",
    "RegionFit",
    "BodyAnalyzer",
    "FitEngine",
    "PoseLandmarks",
    "EngineReadOnlyPoseAdapter",
    "try_get_pose_engine",
    "get_default_garment_measurements",
    "DEFAULT_TSHIRT_SIZE_CHART",
    "MEASUREMENT_SOURCE_ESTIMATED",
    "MEASUREMENT_SOURCE_USER_PROVIDED",
    "MEASUREMENT_SOURCE_MERCHANT_PROVIDED",
    "MEASUREMENT_SOURCE_ILLUSTRATIVE_DEFAULT",
]
