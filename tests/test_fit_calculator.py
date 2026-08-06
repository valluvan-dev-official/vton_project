"""Unit tests for the pure fit-calculation module (Phase 1).

Runs with plain pytest — no FastAPI/SQLAlchemy/DB dependencies.
"""
import pytest

from app.services.fit_calculator import (
    BodyMeasurements,
    GarmentMeasurements,
    classify_body_shape,
    compute_regional_fit,
    recommend_size,
    estimate_fit,
    CONFIDENCE_FLOOR,
    CONFIDENCE_CEILING,
    CONFIDENCE_HEADROOM_SCALE,
    DISCLAIMER,
)


# ── Body-shape classification ───────────────────────────────────────────────

def test_hourglass_shape():
    body = BodyMeasurements(height_cm=165, bust_cm=95, waist_cm=68, hips_cm=96)
    shape, conf = classify_body_shape(body)
    assert shape == "hourglass"
    assert CONFIDENCE_FLOOR <= conf <= CONFIDENCE_CEILING


def test_pear_shape():
    body = BodyMeasurements(height_cm=165, bust_cm=88, waist_cm=72, hips_cm=104)
    shape, conf = classify_body_shape(body)
    assert shape == "pear"
    assert CONFIDENCE_FLOOR <= conf <= CONFIDENCE_CEILING


def test_inverted_triangle_shape():
    body = BodyMeasurements(height_cm=175, bust_cm=104, waist_cm=84, hips_cm=90)
    shape, conf = classify_body_shape(body)
    assert shape == "inverted_triangle"


def test_apple_shape():
    body = BodyMeasurements(height_cm=165, bust_cm=98, waist_cm=92, hips_cm=100)
    shape, conf = classify_body_shape(body)
    assert shape == "apple"


def test_rectangle_shape():
    body = BodyMeasurements(height_cm=170, bust_cm=90, waist_cm=82, hips_cm=91)
    shape, conf = classify_body_shape(body)
    assert shape == "rectangle"


@pytest.mark.parametrize("bust,waist,hips", [
    (None, 70, 95),
    (90, None, 95),
    (90, 70, None),
    (0, 70, 95),
    (90, -1, 95),
])
def test_body_shape_insufficient_data(bust, waist, hips):
    body = BodyMeasurements(height_cm=165, bust_cm=bust, waist_cm=waist, hips_cm=hips)
    shape, conf = classify_body_shape(body)
    assert shape == "insufficient_data"
    assert conf == 0.0


def test_confidence_bounded_and_deterministic():
    body = BodyMeasurements(height_cm=165, bust_cm=95, waist_cm=68, hips_cm=96)
    _, conf1 = classify_body_shape(body)
    _, conf2 = classify_body_shape(body)
    assert conf1 == conf2
    assert CONFIDENCE_FLOOR <= conf1 <= CONFIDENCE_CEILING


def test_confidence_near_boundary_is_low():
    # bust/hip ratio essentially on the 0.95 threshold -> low confidence
    body = BodyMeasurements(height_cm=165, bust_cm=95.05, waist_cm=71, hips_cm=100)
    _, conf = classify_body_shape(body)
    assert conf == pytest.approx(CONFIDENCE_FLOOR, abs=0.05)


def test_confidence_far_from_boundary_is_high():
    body = BodyMeasurements(height_cm=165, bust_cm=100, waist_cm=60, hips_cm=100)
    _, conf = classify_body_shape(body)
    assert conf == CONFIDENCE_CEILING


# ── Regional ease / effective_ease formula ──────────────────────────────────

def test_effective_ease_formula_adds_stretch_and_fit_offset():
    # ease_cm = 96 - 92 = 4 ; stretch(low)=1.5 ; fit_style(relaxed)=+3
    # effective_ease = 4 + 1.5 + 3 = 8.5 -> "loose" band [8,14)
    result = compute_regional_fit(92, 96, "low", "relaxed")
    assert result.ease_cm == 4.0
    assert result.fit == "loose"


def test_effective_ease_slim_offset_is_negative():
    # ease_cm = 0 ; stretch(none)=0 ; fit_style(slim)=-2 -> effective_ease=-2 -> "very_tight" boundary
    result = compute_regional_fit(90, 90, "none", "slim")
    assert result.ease_cm == 0.0
    assert result.fit == "tight"  # -2 falls in [-2, 0) band


@pytest.mark.parametrize("ease,expected", [
    (-3, "very_tight"),
    (-2, "tight"),
    (-0.1, "tight"),
    (0, "fitted"),
    (3.9, "fitted"),
    (4, "regular"),
    (7.9, "regular"),
    (8, "loose"),
    (13.9, "loose"),
    (14, "very_loose"),
    (20, "very_loose"),
])
def test_ease_band_boundaries(ease, expected):
    body_v = 90
    garment_v = 90 + ease
    result = compute_regional_fit(body_v, garment_v, "none", "regular")
    assert result.fit == expected


def test_regional_fit_missing_body_measurement_is_insufficient():
    result = compute_regional_fit(None, 96, "none", "regular")
    assert result.fit == "insufficient_data"
    assert result.ease_cm is None


def test_regional_fit_missing_garment_measurement_is_insufficient():
    result = compute_regional_fit(90, None, "none", "regular")
    assert result.fit == "insufficient_data"
    assert result.ease_cm is None


# ── Size recommendation ─────────────────────────────────────────────────────

def _chart():
    return [
        GarmentMeasurements(size_label="S", bust_cm=88, waist_cm=72, hips_cm=94, stretch_category="none", fit_style="regular"),
        GarmentMeasurements(size_label="M", bust_cm=94, waist_cm=78, hips_cm=100, stretch_category="none", fit_style="regular"),
        GarmentMeasurements(size_label="L", bust_cm=100, waist_cm=84, hips_cm=106, stretch_category="none", fit_style="regular"),
        GarmentMeasurements(size_label="XL", bust_cm=106, waist_cm=90, hips_cm=112, stretch_category="none", fit_style="regular"),
    ]


def test_recommend_size_picks_best_fit():
    # Body matches M chart almost exactly (ease ~2-4cm across the board -> fitted/regular)
    body = BodyMeasurements(height_cm=165, bust_cm=91, waist_cm=75, hips_cm=97)
    rec = recommend_size(body, "S", _chart())
    assert rec == "M"


def test_recommend_size_prefers_selected_size_on_tie():
    # Body exactly matches both S and M's ease profile relative to their own bands
    # (contrived: same penalty by construction) -> selected_size should win ties.
    chart = [
        GarmentMeasurements(size_label="S", bust_cm=95, waist_cm=95, hips_cm=95, stretch_category="none", fit_style="regular"),
        GarmentMeasurements(size_label="M", bust_cm=95, waist_cm=95, hips_cm=95, stretch_category="none", fit_style="regular"),
    ]
    body = BodyMeasurements(height_cm=165, bust_cm=91, waist_cm=91, hips_cm=91)
    rec = recommend_size(body, "M", chart)
    assert rec == "M"


def test_recommend_size_unknown_label_preserves_chart_order():
    chart = [
        GarmentMeasurements(size_label="ONE-SIZE", bust_cm=95, waist_cm=95, hips_cm=95),
    ]
    body = BodyMeasurements(height_cm=165, bust_cm=91, waist_cm=91, hips_cm=91)
    rec = recommend_size(body, "M", chart)
    assert rec == "ONE-SIZE"


def test_recommend_size_empty_chart_returns_none():
    body = BodyMeasurements(height_cm=165, bust_cm=91, waist_cm=91, hips_cm=91)
    assert recommend_size(body, "M", []) is None


# ── End-to-end estimate_fit orchestration ───────────────────────────────────

def test_estimate_fit_full_flow():
    body = BodyMeasurements(height_cm=165, bust_cm=91, waist_cm=75, hips_cm=97)
    result = estimate_fit(body, "S", _chart())
    assert result.selected_size == "S"
    assert result.recommended_size in ("S", "M", "L", "XL")
    assert result.body_shape != "insufficient_data"
    assert result.is_approximate is True
    assert result.disclaimer == DISCLAIMER
    assert set(result.regional_fit.keys()) == {"bust", "waist", "hips"}


def test_estimate_fit_no_chart_never_guesses():
    body = BodyMeasurements(height_cm=165, bust_cm=91, waist_cm=75, hips_cm=97)
    result = estimate_fit(body, "M", [])
    assert result.recommended_size is None
    assert all(r["fit"] == "insufficient_data" for r in result.regional_fit.values())
    # body_shape is independent of garment availability
    assert result.body_shape != "insufficient_data"


def test_estimate_fit_selected_size_always_echoed():
    body = BodyMeasurements(height_cm=165, bust_cm=91, waist_cm=75, hips_cm=97)
    result = estimate_fit(body, "XL", _chart())
    assert result.selected_size == "XL"


def test_estimate_fit_insufficient_body_measurements():
    body = BodyMeasurements(height_cm=165, bust_cm=None, waist_cm=None, hips_cm=None)
    result = estimate_fit(body, "M", _chart())
    assert result.body_shape == "insufficient_data"
    assert result.confidence == 0.0
    assert all(r["fit"] == "insufficient_data" for r in result.regional_fit.values())
