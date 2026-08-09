"""Typed data models for the (Phase 1, shadow-mode) fit-analysis engine.

No FastAPI/SQLAlchemy/torch imports here on purpose — mirrors the existing
app/services/fit_calculator.py convention: every model/function in this
package is a plain, deterministic Python value/transform so it can be unit
tested without the web stack, GPU, or any model weights, and reused from a
route, a Celery task, or a future CLI/batch job alike.

Hard rule enforced throughout fit_analysis: a single 2D photo can never
produce a *verified* real-world measurement. Every numeric body measurement
in this module is an estimate, and `measurement_source` must always say so
plainly — never presented as an exact physical measurement.
"""
from dataclasses import dataclass, field
from typing import Optional

# ── Measurement provenance ──────────────────────────────────────────────────
# Distinguishes "we guessed this from pixels" from "a human typed this in"
# from "the merchant's product catalog said this" — callers (API responses,
# UI) must be able to tell these apart and phrase confidence accordingly.

MEASUREMENT_SOURCE_ESTIMATED = "estimated"
MEASUREMENT_SOURCE_USER_PROVIDED = "user_provided"
MEASUREMENT_SOURCE_MERCHANT_PROVIDED = "merchant_provided"
# Made-up, illustrative numbers (see fit_analysis/garment_measurements.py's
# DEFAULT_TSHIRT_SIZE_CHART) — distinct from MEASUREMENT_SOURCE_MERCHANT_PROVIDED
# on purpose, so a consumer of fit_analysis output can never mistake
# dev/test placeholder data for a real merchant's authoritative sizing.
MEASUREMENT_SOURCE_ILLUSTRATIVE_DEFAULT = "illustrative_default"

VALID_MEASUREMENT_SOURCES = frozenset({
    MEASUREMENT_SOURCE_ESTIMATED,
    MEASUREMENT_SOURCE_USER_PROVIDED,
    MEASUREMENT_SOURCE_MERCHANT_PROVIDED,
    MEASUREMENT_SOURCE_ILLUSTRATIVE_DEFAULT,
})


def _validate_source(value: str, field_name: str = "measurement_source") -> str:
    if value not in VALID_MEASUREMENT_SOURCES:
        raise ValueError(
            f"{field_name}={value!r} is not one of {sorted(VALID_MEASUREMENT_SOURCES)}."
        )
    return value


# Same vocabulary as app/models/fit.py's STRETCH_CATEGORIES/FIT_STYLES,
# duplicated as plain constants rather than imported — keeps this package
# free of any SQLAlchemy/app.models import, consistent with the
# no-framework-imports rule at the top of this file.
VALID_STRETCH_CATEGORIES = frozenset({"none", "low", "medium", "high"})
VALID_FIT_STYLES = frozenset({"slim", "regular", "relaxed", "oversized"})


@dataclass
class BodyMeasurementEstimate:
    """Body measurements for one person, from one photo (+ optional height).

    Every *_cm field is an estimate unless measurement_source is
    "user_provided". None means "we don't have a value" — never fabricated,
    never silently defaulted to some population average.
    """
    height_cm: Optional[float] = None

    shoulder_width_cm: Optional[float] = None
    chest_cm: Optional[float] = None
    waist_cm: Optional[float] = None
    hip_cm: Optional[float] = None
    torso_length_cm: Optional[float] = None
    arm_length_cm: Optional[float] = None

    body_shape: Optional[str] = None  # descriptive only, e.g. "rectangle" — never fed into sizing math

    confidence: float = 0.0
    measurement_source: str = MEASUREMENT_SOURCE_ESTIMATED

    # Fractional half-width used to derive an uncertainty band around every
    # *_cm point estimate below (see confidence.estimate_uncertainty_pct).
    # None only when there's nothing to put a band around (no measurements
    # at all) — never used to imply a *_cm value is exact.
    uncertainty_pct: Optional[float] = None

    # Free-form breadcrumbs for why confidence landed where it did (e.g.
    # "no_pose_landmarks", "no_height_reference") — not part of the public
    # confidence *number*, just diagnostic context for logs/debugging.
    notes: list[str] = field(default_factory=list)

    def __post_init__(self):
        _validate_source(self.measurement_source)
        if not (0.0 <= self.confidence <= 1.0):
            raise ValueError(f"confidence={self.confidence!r} must be within [0.0, 1.0].")
        if self.uncertainty_pct is not None and not (0.0 <= self.uncertainty_pct <= 1.0):
            raise ValueError(f"uncertainty_pct={self.uncertainty_pct!r} must be within [0.0, 1.0].")

    def range_cm(self, value: Optional[float]) -> Optional[tuple[float, float]]:
        """(low, high) estimate around `value`, using this estimate's own
        uncertainty_pct. None if there's no value or no uncertainty to apply
        — never returns a band that implies more precision than we have."""
        if value is None or self.uncertainty_pct is None:
            return None
        delta = value * self.uncertainty_pct
        return (round(value - delta, 1), round(value + delta, 1))

    @property
    def chest_cm_range(self) -> Optional[tuple[float, float]]:
        return self.range_cm(self.chest_cm)

    @property
    def waist_cm_range(self) -> Optional[tuple[float, float]]:
        return self.range_cm(self.waist_cm)

    @property
    def hip_cm_range(self) -> Optional[tuple[float, float]]:
        return self.range_cm(self.hip_cm)

    @property
    def shoulder_width_cm_range(self) -> Optional[tuple[float, float]]:
        return self.range_cm(self.shoulder_width_cm)

    @property
    def torso_length_cm_range(self) -> Optional[tuple[float, float]]:
        return self.range_cm(self.torso_length_cm)

    @property
    def arm_length_cm_range(self) -> Optional[tuple[float, float]]:
        return self.range_cm(self.arm_length_cm)

    def as_dict(self) -> dict:
        return {
            "height_cm": self.height_cm,
            "shoulder_width_cm": self.shoulder_width_cm,
            "shoulder_width_cm_range": self.shoulder_width_cm_range,
            "chest_cm": self.chest_cm,
            "chest_cm_range": self.chest_cm_range,
            "waist_cm": self.waist_cm,
            "waist_cm_range": self.waist_cm_range,
            "hip_cm": self.hip_cm,
            "hip_cm_range": self.hip_cm_range,
            "torso_length_cm": self.torso_length_cm,
            "torso_length_cm_range": self.torso_length_cm_range,
            "arm_length_cm": self.arm_length_cm,
            "arm_length_cm_range": self.arm_length_cm_range,
            "body_shape": self.body_shape,
            "confidence": self.confidence,
            "measurement_source": self.measurement_source,
        }


@dataclass
class GarmentMeasurements:
    """Optional merchant/catalog measurements for one garment size.

    The merchant/catalog is the authoritative source for these — this
    module never derives chest/shoulder/length/sleeve/hip *garment*
    dimensions from the S/M/L/XL/XXL label alone. A label with no
    measurements attached is legitimate (measurement_source stays
    "merchant_provided" by convention even if every *_cm field is None —
    it simply means the catalog didn't supply that dimension); callers
    should treat missing fields as "insufficient_data" per region, not
    guess.
    """
    size_label: str
    garment_type: Optional[str] = None
    chest_cm: Optional[float] = None
    shoulder_cm: Optional[float] = None
    waist_cm: Optional[float] = None
    length_cm: Optional[float] = None
    sleeve_length_cm: Optional[float] = None
    hip_cm: Optional[float] = None
    measurement_source: str = MEASUREMENT_SOURCE_MERCHANT_PROVIDED

    # Stored/passed through for data fidelity (Phase 2 merchant catalog
    # schema requires them) but NOT yet consumed by fit_engine.py's ease
    # bands — FitEngine.evaluate()'s math is deliberately left unchanged so
    # existing, already-tested fit classification behavior doesn't shift as
    # a side effect of catalog integration. A future phase can fold these
    # into the ease calculation explicitly.
    stretch_category: str = "none"
    fit_style: str = "regular"

    def __post_init__(self):
        _validate_source(self.measurement_source)
        if not (self.size_label or "").strip():
            raise ValueError("size_label must be a non-empty string.")
        for name in ("chest_cm", "shoulder_cm", "waist_cm", "length_cm", "sleeve_length_cm", "hip_cm"):
            v = getattr(self, name)
            if v is not None and v <= 0:
                raise ValueError(f"{name}={v!r} must be positive when provided.")
        if self.stretch_category not in VALID_STRETCH_CATEGORIES:
            raise ValueError(
                f"stretch_category={self.stretch_category!r} is not one of "
                f"{sorted(VALID_STRETCH_CATEGORIES)}."
            )
        if self.fit_style not in VALID_FIT_STYLES:
            raise ValueError(f"fit_style={self.fit_style!r} is not one of {sorted(VALID_FIT_STYLES)}.")

    def completeness_ratio(self) -> float:
        """Fraction of the 3 fit-critical dimensions (chest/shoulder/length)
        that are present — used by confidence.py to discount FitResult
        confidence when the catalog entry is sparse."""
        critical = (self.chest_cm, self.shoulder_cm, self.length_cm)
        present = sum(1 for v in critical if v is not None)
        return present / len(critical)


@dataclass
class RegionFit:
    """Ease (garment - body, cm) and fit label for one body region."""
    ease_cm: Optional[float]
    fit: str  # one of FIT_LABELS in fit_engine.py, or "insufficient_data"


@dataclass
class FitResult:
    """Output of FitEngine.evaluate() — advisory only.

    is_shadow_mode is always True in Phase 1: this result must never be
    read by ml/scripts/gpu_inference.py or any renderer/mask/scaling code.
    It exists purely to be calculated, stored, and displayed for
    independent validation before any Phase 2 wiring is even considered.
    """
    selected_size: str
    overall_fit: str
    chest_fit: str
    shoulder_fit: str
    length_fit: str
    confidence: float
    explanation: str
    is_shadow_mode: bool = True

    def __post_init__(self):
        if not (0.0 <= self.confidence <= 1.0):
            raise ValueError(f"confidence={self.confidence!r} must be within [0.0, 1.0].")

    def as_dict(self) -> dict:
        return {
            "selected_size": self.selected_size,
            "overall_fit": self.overall_fit,
            "chest_fit": self.chest_fit,
            "shoulder_fit": self.shoulder_fit,
            "length_fit": self.length_fit,
            "confidence": self.confidence,
            "explanation": self.explanation,
            "is_shadow_mode": self.is_shadow_mode,
        }
