"""FitEngine — the only place in fit_analysis that decides "tight" vs
"loose" vs "oversized". Deliberately ease-driven (garment measurement minus
estimated body measurement), never a lookup from the size label itself:
an "XL" tag means nothing on its own without knowing what XL actually
measures for *this* garment (see Phase-1 spec item 6).

All thresholds live in EASE_BANDS / LENGTH_EASE_BANDS below — nothing else
in this file hardcodes a fit boundary.
"""
from dataclasses import dataclass
from typing import Optional

from .confidence import combine_fit_confidence
from .models import BodyMeasurementEstimate, FitResult, GarmentMeasurements, RegionFit

# Returned instead of an apparently-authoritative fit when no real
# merchant/catalog garment measurements are available — see
# FitEngine.insufficient_garment_data() and garment_measurements.py's
# DEFAULT_TSHIRT_SIZE_CHART docstring (dev/test fallback only, never to be
# treated as authoritative merchant sizing in production).
INSUFFICIENT_GARMENT_DATA = "insufficient_garment_data"

# Regional ease bands (garment_cm - body_cm), evaluated for chest/shoulder.
# Same "ease -> label" spirit as app/services/fit_calculator.py's
# EASE_BANDS, reimplemented independently for this module's own (differently
# shaped) GarmentMeasurements/BodyMeasurementEstimate types.
FIT_LABELS = ("very_tight", "tight", "fitted", "regular", "loose", "very_loose")

EASE_BANDS: tuple[tuple[str, Optional[float], Optional[float]], ...] = (
    ("very_tight", None, -2.0),
    ("tight", -2.0, 0.0),
    ("fitted", 0.0, 4.0),
    ("regular", 4.0, 8.0),
    ("loose", 8.0, 14.0),
    ("very_loose", 14.0, None),
)

# Garment length vs. torso length uses its own bands — a few cm of extra
# length reads very differently than a few cm of extra chest ease.
LENGTH_EASE_BANDS: tuple[tuple[str, Optional[float], Optional[float]], ...] = (
    ("very_short", None, -4.0),
    ("short", -4.0, -1.0),
    ("regular", -1.0, 6.0),
    ("long", 6.0, 12.0),
    ("very_long", 12.0, None),
)

# overall_fit combination: worst-case label across chest/shoulder/length,
# ranked by how far from "regular" each label sits. Ties broken toward the
# chest reading (most load-bearing region for a top).
_OVERALL_RANK = {
    "insufficient_data": 0,
    "fitted": 1, "regular": 1, "short": 1, "long": 1,
    "tight": 2, "loose": 2,
    "very_tight": 3, "very_short": 3,
    "very_loose": 3, "very_long": 3,
}

# Final overall_fit vocabulary shown to callers — maps the raw regional
# label that "won" the ranking above onto the coarser public vocabulary.
_OVERALL_LABEL_MAP = {
    "very_tight": "very_tight", "tight": "tight",
    "fitted": "fitted", "regular": "regular", "short": "regular", "long": "regular",
    "loose": "loose", "very_loose": "oversized", "very_long": "oversized",
    "very_short": "very_tight",
    "insufficient_data": "insufficient_data",
}


def _label_for_ease(ease: float, bands) -> str:
    for label, lo, hi in bands:
        if lo is not None and ease < lo:
            continue
        if hi is not None and ease >= hi:
            continue
        return label
    return bands[-1][0]


def _region_fit(body_cm: Optional[float], garment_cm: Optional[float], bands) -> RegionFit:
    if body_cm is None or garment_cm is None:
        return RegionFit(ease_cm=None, fit="insufficient_data")
    if body_cm <= 0 or garment_cm <= 0:
        return RegionFit(ease_cm=None, fit="insufficient_data")
    ease = round(garment_cm - body_cm, 1)
    return RegionFit(ease_cm=ease, fit=_label_for_ease(ease, bands))


def _combine_overall(*fits: str) -> str:
    known = [f for f in fits if f != "insufficient_data"]
    if not known:
        return "insufficient_data"
    worst = max(known, key=lambda f: _OVERALL_RANK.get(f, 0))
    return _OVERALL_LABEL_MAP.get(worst, "regular")


def _explanation_for(chest: RegionFit, shoulder: RegionFit, length: RegionFit, overall: str) -> str:
    parts = []
    if chest.fit != "insufficient_data":
        parts.append(f"chest ease {chest.ease_cm:+.1f}cm ({chest.fit})")
    if shoulder.fit != "insufficient_data":
        parts.append(f"shoulder ease {shoulder.ease_cm:+.1f}cm ({shoulder.fit})")
    if length.fit != "insufficient_data":
        parts.append(f"length ease {length.ease_cm:+.1f}cm ({length.fit})")
    if not parts:
        return "Not enough body and/or garment measurements to estimate fit."
    return f"Estimated {overall} fit - " + ", ".join(parts) + "."


@dataclass
class FitEngine:
    """Stateless — safe to share a single instance across requests."""

    def evaluate(self, body: BodyMeasurementEstimate, garment: GarmentMeasurements) -> FitResult:
        chest = _region_fit(body.chest_cm, garment.chest_cm, EASE_BANDS)
        shoulder = _region_fit(body.shoulder_width_cm, garment.shoulder_cm, EASE_BANDS)
        length = _region_fit(body.torso_length_cm, garment.length_cm, LENGTH_EASE_BANDS)

        overall = _combine_overall(chest.fit, shoulder.fit, length.fit)
        explanation = _explanation_for(chest, shoulder, length, overall)

        confidence = combine_fit_confidence(body.confidence, garment.completeness_ratio())

        return FitResult(
            selected_size=garment.size_label,
            overall_fit=overall,
            chest_fit=chest.fit,
            shoulder_fit=shoulder.fit,
            length_fit=length.fit,
            confidence=confidence,
            explanation=explanation,
        )

    def insufficient_garment_data(self, selected_size: str) -> FitResult:
        """Use when no authoritative (merchant/catalog) garment measurement
        is available for `selected_size` — e.g. in production before a real
        catalog is wired up, where the illustrative
        DEFAULT_TSHIRT_SIZE_CHART must never be used as if it were real
        sizing data. Every region reports INSUFFICIENT_GARMENT_DATA and
        confidence is 0.0 — this must never be mistaken for a computed fit."""
        return FitResult(
            selected_size=selected_size,
            overall_fit=INSUFFICIENT_GARMENT_DATA,
            chest_fit=INSUFFICIENT_GARMENT_DATA,
            shoulder_fit=INSUFFICIENT_GARMENT_DATA,
            length_fit=INSUFFICIENT_GARMENT_DATA,
            confidence=0.0,
            explanation=(
                "No authoritative merchant garment measurements are available "
                f"for size {selected_size!r} — a fit estimate cannot be "
                "computed responsibly from the size label alone."
            ),
        )

    def recommend_size(
        self,
        body: BodyMeasurementEstimate,
        garment_chart: list[GarmentMeasurements],
    ) -> Optional[FitResult]:
        """Evaluate every entry in a size chart and return the one whose
        overall_fit is closest to "regular" (ties broken toward the
        chart's own order). None if the chart is empty."""
        if not garment_chart:
            return None

        # "regular" is the target; rank every other label by its distance
        # from that target using the same _OVERALL_RANK scale.
        def score(entry: GarmentMeasurements) -> tuple:
            result = self.evaluate(body, entry)
            return (_OVERALL_RANK.get(result.overall_fit, 99), result.overall_fit != "regular")

        scored = [(score(entry), idx, entry) for idx, entry in enumerate(garment_chart)]
        scored.sort(key=lambda t: (t[0], t[1]))
        best_entry = scored[0][2]
        return self.evaluate(body, best_entry)
