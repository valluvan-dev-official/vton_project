"""Confidence scoring for fit_analysis — the single source of truth for
every numeric confidence threshold used by body_analyzer.py and
fit_engine.py. Nothing else in this package should hardcode a confidence
number; add a constant here instead.

Rule of thumb (per the Phase-1 spec):
    front photo only                          -> medium/low
    photo + known height                       -> medium/high
    photo + height + reliable pose landmarks   -> higher
Never claim exact-sizing certainty from one photo — CONFIDENCE_CEILING caps
every path well short of 1.0.
"""

# ── Body-measurement confidence bands ───────────────────────────────────────
# Keyed by (has_pose_landmarks, has_height). pose_quality (0..1, fraction of
# the landmarks a pose adapter needed that were actually detected) can push
# the "pose + height" band up to CONFIDENCE_POSE_HEIGHT_HIGH_QUALITY.

CONFIDENCE_NO_POSE_NO_HEIGHT = 0.20     # image dimensions only — weakest signal
CONFIDENCE_NO_POSE_WITH_HEIGHT = 0.35   # a scale reference, but no body landmarks at all
CONFIDENCE_POSE_NO_HEIGHT = 0.50        # proportions known, but no real-world scale
CONFIDENCE_POSE_WITH_HEIGHT = 0.70      # proportions + scale reference
CONFIDENCE_POSE_WITH_HEIGHT_HIGH_QUALITY = 0.85  # + most/all landmarks confidently detected

# Fraction of required landmarks that must be present for "high quality".
POSE_QUALITY_HIGH_THRESHOLD = 0.85

# Never claim more than this from a single photo, regardless of inputs.
CONFIDENCE_CEILING = 0.90


def estimate_body_confidence(
    has_pose_landmarks: bool,
    has_height: bool,
    pose_quality: float | None = None,
) -> float:
    """Confidence for a BodyMeasurementEstimate, in [0, CONFIDENCE_CEILING]."""
    if not has_pose_landmarks:
        base = CONFIDENCE_NO_POSE_WITH_HEIGHT if has_height else CONFIDENCE_NO_POSE_NO_HEIGHT
        return min(base, CONFIDENCE_CEILING)

    if not has_height:
        return min(CONFIDENCE_POSE_NO_HEIGHT, CONFIDENCE_CEILING)

    if pose_quality is not None and pose_quality >= POSE_QUALITY_HIGH_THRESHOLD:
        return min(CONFIDENCE_POSE_WITH_HEIGHT_HIGH_QUALITY, CONFIDENCE_CEILING)

    return min(CONFIDENCE_POSE_WITH_HEIGHT, CONFIDENCE_CEILING)


# ── Fit-result confidence ───────────────────────────────────────────────────
# FitResult confidence blends body-measurement confidence with how complete
# the garment's own catalog data is (a "M" with only a chest measurement is
# less trustworthy to fit against than one with chest+shoulder+length).

# garment_completeness_ratio in [0, 1] contributes this much weight; the
# remainder is carried by body confidence. E.g. at 0.5, a fully-complete
# garment entry can add up to 50% on top of a body-confidence-only floor.
GARMENT_COMPLETENESS_WEIGHT = 0.5


def combine_fit_confidence(body_confidence: float, garment_completeness_ratio: float) -> float:
    """Blend body-estimate confidence with garment-data completeness.

    formula: body_confidence * (1 - W + W * completeness), W = GARMENT_COMPLETENESS_WEIGHT
    — a fully complete garment entry (ratio=1) leaves body_confidence
    untouched; a totally empty one (ratio=0) discounts it by W.
    """
    if not (0.0 <= body_confidence <= 1.0):
        raise ValueError(f"body_confidence={body_confidence!r} must be within [0.0, 1.0].")
    if not (0.0 <= garment_completeness_ratio <= 1.0):
        raise ValueError(f"garment_completeness_ratio={garment_completeness_ratio!r} must be within [0.0, 1.0].")

    factor = (1 - GARMENT_COMPLETENESS_WEIGHT) + GARMENT_COMPLETENESS_WEIGHT * garment_completeness_ratio
    combined = body_confidence * factor
    return round(min(max(combined, 0.0), CONFIDENCE_CEILING), 2)


# ── Measurement uncertainty (avoid false precision) ─────────────────────────
# A single 2D photo never yields an exact centimetre figure. Every
# image-derived *_cm value must be presented with an uncertainty band, not
# a bare point estimate — see BodyMeasurementEstimate.range_cm() in
# models.py, which uses this.

# Uncertainty never drops below this fraction of the value, no matter how
# confident the pose read was — it's still one photo, not a tape measure.
UNCERTAINTY_PCT_FLOOR = 0.06
# At confidence 0.0, the band widens to this fraction.
UNCERTAINTY_PCT_AT_ZERO_CONFIDENCE = 0.30


def estimate_uncertainty_pct(confidence: float) -> float:
    """Higher confidence -> narrower band, but never below the floor."""
    if not (0.0 <= confidence <= 1.0):
        raise ValueError(f"confidence={confidence!r} must be within [0.0, 1.0].")
    scaled = UNCERTAINTY_PCT_AT_ZERO_CONFIDENCE * (1.0 - confidence)
    return round(max(UNCERTAINTY_PCT_FLOOR, scaled), 3)
