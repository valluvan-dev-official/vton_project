"""Pure body-shape / garment-fit calculation logic (Phase 1).

No FastAPI, SQLAlchemy, or I/O imports here on purpose — every function is a
plain, deterministic transform over plain Python values so it can be unit
tested in isolation and reused from the API route (or a future CLI/batch job)
without dragging in the web/DB stack.

Hard rules enforced throughout this module:
  * Body shape is descriptive only — it never feeds into size/fit math.
  * Regional fit is computed independently per region (bust/waist/hips).
  * Missing measurements produce "insufficient_data", never a guess.
  * selected_size is always echoed back unchanged; only recommended_size is computed.
"""
from dataclasses import dataclass, field
from decimal import Decimal, ROUND_HALF_UP
from typing import Optional, Union

Number = Union[int, float, Decimal]

# ── Configurable thresholds (single source of truth — nothing else in this
# module hardcodes a numeric fit boundary) ──────────────────────────────────

# Body-shape classification ratio boundaries.
BODY_SHAPE_THRESHOLDS = {
    "bust_hip_ratio_low": 0.95,
    "bust_hip_ratio_high": 1.05,
    "waist_hip_ratio_hourglass_max": 0.75,
    "waist_hip_ratio_pear_max": 0.80,
    # Apple's waist must be close to (or exceed) bust/hips — deliberately
    # above pear/rectangle's 0.80 boundary so the two bands cannot overlap
    # (rectangle occupies the gap between hourglass_max and apple_min).
    "waist_bust_ratio_apple_min": 0.92,
    "waist_hip_ratio_apple_min": 0.92,
}

# Confidence formula: 0.50 floor, 0.95 ceiling, scaled by how far the
# measurements sit *inside* the winning bucket's own boundary conditions
# (headroom), not distance to an arbitrary threshold value.
CONFIDENCE_FLOOR = 0.50
CONFIDENCE_CEILING = 0.95
CONFIDENCE_HEADROOM_SCALE = 0.05  # ratio headroom at which confidence saturates

# Regional ease -> fit-label bands, in cm, evaluated on "effective_ease".
EASE_BANDS = (
    ("very_tight", None, -2),
    ("tight", -2, 0),
    ("fitted", 0, 4),
    ("regular", 4, 8),
    ("loose", 8, 14),
    ("very_loose", 14, None),
)

STRETCH_BONUS_CM = {
    "none": 0.0,
    "low": 1.5,
    "medium": 3.0,
    "high": 5.0,
}

FIT_STYLE_OFFSET_CM = {
    "slim": -2.0,
    "regular": 0.0,
    "relaxed": 3.0,
    "oversized": 6.0,
}

REGION_PENALTY = {
    "very_tight": 3,
    "tight": 1,
    "fitted": 0,
    "regular": 0,
    "loose": 1,
    "very_loose": 3,
    "insufficient_data": 2,
}

# Canonical size ordering used to break recommendation ties. Any label not in
# this list keeps its position in the chart (stable sort) after the known ones.
SIZE_ORDER = ["XXS", "XS", "S", "M", "L", "XL", "XXL", "3XL"]

DISCLAIMER = "AI fit estimate — actual fit may vary by brand, fabric and garment construction."


def _round1(value: Number) -> float:
    d = Decimal(str(value)).quantize(Decimal("0.1"), rounding=ROUND_HALF_UP)
    return float(d)


@dataclass
class BodyMeasurements:
    height_cm: Optional[Number] = None
    bust_cm: Optional[Number] = None
    waist_cm: Optional[Number] = None
    hips_cm: Optional[Number] = None
    shoulder_cm: Optional[Number] = None


@dataclass
class GarmentMeasurements:
    size_label: str
    bust_cm: Optional[Number] = None
    waist_cm: Optional[Number] = None
    hips_cm: Optional[Number] = None
    shoulder_cm: Optional[Number] = None
    length_cm: Optional[Number] = None
    sleeve_cm: Optional[Number] = None
    stretch_category: str = "none"
    fit_style: str = "regular"


@dataclass
class RegionalFitResult:
    ease_cm: Optional[float]
    fit: str


@dataclass
class FitEstimateResult:
    selected_size: str
    recommended_size: Optional[str]
    body_shape: str
    confidence: float
    regional_fit: dict
    summary: str
    is_approximate: bool = True
    disclaimer: str = DISCLAIMER


# ── Body-shape classification ───────────────────────────────────────────────

def _confidence_from_headroom(headroom: float) -> float:
    """Deterministic confidence in [CONFIDENCE_FLOOR, CONFIDENCE_CEILING].

    `headroom` is how far the measurements sit inside the winning bucket's
    defining inequalities (0 = exactly on the boundary, ambiguous;
    >= CONFIDENCE_HEADROOM_SCALE = deep inside the bucket, unambiguous).
    """
    normalized = min(max(headroom, 0.0) / CONFIDENCE_HEADROOM_SCALE, 1.0)
    conf = CONFIDENCE_FLOOR + (CONFIDENCE_CEILING - CONFIDENCE_FLOOR) * normalized
    return round(max(CONFIDENCE_FLOOR, min(CONFIDENCE_CEILING, conf)), 2)


def classify_body_shape(body: BodyMeasurements) -> tuple[str, float]:
    """Returns (body_shape, confidence). Descriptive only — never used for sizing."""
    if not (body.bust_cm and body.waist_cm and body.hips_cm):
        return "insufficient_data", 0.0

    bust, waist, hips = float(body.bust_cm), float(body.waist_cm), float(body.hips_cm)
    if bust <= 0 or waist <= 0 or hips <= 0:
        return "insufficient_data", 0.0

    t = BODY_SHAPE_THRESHOLDS
    bust_hip_low, bust_hip_high = t["bust_hip_ratio_low"], t["bust_hip_ratio_high"]
    hourglass_max, pear_max = t["waist_hip_ratio_hourglass_max"], t["waist_hip_ratio_pear_max"]
    apple_min_whr, apple_min_wbr = t["waist_hip_ratio_apple_min"], t["waist_bust_ratio_apple_min"]

    bhr = bust / hips
    whr = waist / hips
    wbr = waist / bust
    balanced_bust_hip = bust_hip_low <= bhr <= bust_hip_high

    if balanced_bust_hip and whr <= hourglass_max:
        headroom = min(bust_hip_high - bhr, bhr - bust_hip_low, hourglass_max - whr)
        return "hourglass", _confidence_from_headroom(headroom)
    if bhr <= bust_hip_low and whr <= pear_max:
        headroom = min(bust_hip_low - bhr, pear_max - whr)
        return "pear", _confidence_from_headroom(headroom)
    if bhr >= bust_hip_high:
        headroom = bhr - bust_hip_high
        return "inverted_triangle", _confidence_from_headroom(headroom)
    if whr >= apple_min_whr and wbr >= apple_min_wbr:
        headroom = min(whr - apple_min_whr, wbr - apple_min_wbr)
        return "apple", _confidence_from_headroom(headroom)
    # Rectangle: balanced bust/hip, waist neither deeply defined (hourglass)
    # nor absent (apple) — the gap between the two waist-ratio bands.
    headroom = min(bust_hip_high - bhr, bhr - bust_hip_low, whr - hourglass_max, apple_min_whr - whr)
    return "rectangle", _confidence_from_headroom(headroom)


# ── Regional ease / fit ─────────────────────────────────────────────────────

def _fit_label_for_ease(effective_ease: float) -> str:
    for label, lo, hi in EASE_BANDS:
        if lo is not None and effective_ease < lo:
            continue
        if hi is not None and effective_ease >= hi:
            continue
        return label
    return "very_loose"


def compute_regional_fit(
    body_value_cm: Optional[Number],
    garment_value_cm: Optional[Number],
    stretch_category: str,
    fit_style: str,
) -> RegionalFitResult:
    """ease_cm = garment - body. effective_ease adds stretch forgiveness and
    the fit-style offset, per the approved formula:
        effective_ease = ease_cm + stretch_bonus_cm + fit_style_offset_cm
    """
    if body_value_cm in (None, "") or garment_value_cm in (None, ""):
        return RegionalFitResult(ease_cm=None, fit="insufficient_data")

    body_v = float(body_value_cm)
    garment_v = float(garment_value_cm)
    if body_v <= 0 or garment_v <= 0:
        return RegionalFitResult(ease_cm=None, fit="insufficient_data")

    ease_cm = garment_v - body_v
    stretch_bonus = STRETCH_BONUS_CM.get(stretch_category, 0.0)
    fit_offset = FIT_STYLE_OFFSET_CM.get(fit_style, 0.0)
    effective_ease = ease_cm + stretch_bonus + fit_offset

    return RegionalFitResult(ease_cm=_round1(ease_cm), fit=_fit_label_for_ease(effective_ease))


def compute_all_regional_fit(body: BodyMeasurements, garment: Optional[GarmentMeasurements]) -> dict:
    if garment is None:
        return {
            "bust": RegionalFitResult(ease_cm=None, fit="insufficient_data").__dict__,
            "waist": RegionalFitResult(ease_cm=None, fit="insufficient_data").__dict__,
            "hips": RegionalFitResult(ease_cm=None, fit="insufficient_data").__dict__,
        }
    return {
        "bust": compute_regional_fit(body.bust_cm, garment.bust_cm, garment.stretch_category, garment.fit_style).__dict__,
        "waist": compute_regional_fit(body.waist_cm, garment.waist_cm, garment.stretch_category, garment.fit_style).__dict__,
        "hips": compute_regional_fit(body.hips_cm, garment.hips_cm, garment.stretch_category, garment.fit_style).__dict__,
    }


# ── Size recommendation ─────────────────────────────────────────────────────

def _size_sort_key(size_label: str, selected_size: str):
    label = (size_label or "").strip().upper()
    known = label in SIZE_ORDER
    # Known sizes sort by their canonical index; unknown sizes sort after all
    # known ones, in stable (chart-order) fashion via the caller's enumerate index.
    idx = SIZE_ORDER.index(label) if known else len(SIZE_ORDER)
    is_selected = 0 if label == (selected_size or "").strip().upper() else 1
    return (is_selected, idx if known else 0, not known)


def _penalty_for_chart_entry(body: BodyMeasurements, garment: GarmentMeasurements) -> int:
    regional = compute_all_regional_fit(body, garment)
    return sum(REGION_PENALTY[r["fit"]] for r in regional.values())


def recommend_size(
    body: BodyMeasurements,
    selected_size: str,
    garment_chart: list[GarmentMeasurements],
) -> Optional[str]:
    """Best-scoring size from garment_chart, ties broken toward selected_size,
    then by canonical size order, then by chart order. None if the chart is empty.
    """
    if not garment_chart:
        return None

    scored = []
    for original_idx, entry in enumerate(garment_chart):
        penalty = _penalty_for_chart_entry(body, entry)
        label = (entry.size_label or "").strip().upper()
        is_selected = 0 if label == (selected_size or "").strip().upper() else 1
        known = label in SIZE_ORDER
        order_idx = SIZE_ORDER.index(label) if known else (len(SIZE_ORDER) + original_idx)
        scored.append((penalty, is_selected, order_idx, entry.size_label))

    scored.sort(key=lambda t: (t[0], t[1], t[2]))
    return scored[0][3]


# ── Top-level orchestration ─────────────────────────────────────────────────

def _summary_for(body_shape: str, selected_size: str, recommended_size: Optional[str], regional: dict) -> str:
    if recommended_size is None:
        return "Not enough garment data to recommend a size."
    if all(r["fit"] == "insufficient_data" for r in regional.values()):
        return "Not enough body measurements to estimate fit."
    if recommended_size == selected_size:
        return f"Size {selected_size} looks like a good match based on your measurements."
    tight_regions = [name for name, r in regional.items() if r["fit"] in ("tight", "very_tight")]
    loose_regions = [name for name, r in regional.items() if r["fit"] in ("loose", "very_loose")]
    if tight_regions:
        return f"Size {selected_size} may be tight around the {', '.join(tight_regions)}; consider {recommended_size}."
    if loose_regions:
        return f"Size {selected_size} may be loose around the {', '.join(loose_regions)}; consider {recommended_size}."
    return f"Consider size {recommended_size} for a closer match."


def estimate_fit(
    body: BodyMeasurements,
    selected_size: str,
    garment_chart: list[GarmentMeasurements],
) -> FitEstimateResult:
    body_shape, confidence = classify_body_shape(body)

    selected_entry = next(
        (g for g in garment_chart if (g.size_label or "").strip().upper() == (selected_size or "").strip().upper()),
        None,
    )
    regional = compute_all_regional_fit(body, selected_entry)
    recommended = recommend_size(body, selected_size, garment_chart)
    summary = _summary_for(body_shape, selected_size, recommended, regional)

    return FitEstimateResult(
        selected_size=selected_size,
        recommended_size=recommended,
        body_shape=body_shape,
        confidence=confidence,
        regional_fit=regional,
        summary=summary,
    )
