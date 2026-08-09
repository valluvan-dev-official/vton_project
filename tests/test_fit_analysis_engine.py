"""Unit tests for FitEngine — Phase-1 spec item 6 (ease-driven classification,
no hard-coded "S=small person" mapping) and item 10 (tight/regular/loose/
oversized coverage, confidence calculation).
"""
import pytest

from api.app.services.fit_analysis.confidence import combine_fit_confidence
from api.app.services.fit_analysis.fit_engine import FitEngine
from api.app.services.fit_analysis.garment_measurements import DEFAULT_TSHIRT_SIZE_CHART
from api.app.services.fit_analysis.models import BodyMeasurementEstimate, GarmentMeasurements


def _body(chest_cm=90.0, shoulder_cm=44.0, torso_length_cm=55.0, confidence=0.7):
    return BodyMeasurementEstimate(
        chest_cm=chest_cm, shoulder_width_cm=shoulder_cm, torso_length_cm=torso_length_cm,
        confidence=confidence,
    )


def _garment_isolating_chest(chest_ease: float):
    """A garment with only chest_cm set (shoulder/length None) so overall_fit
    is driven purely by the chest region — isolates each ease band cleanly."""
    body = _body()
    return GarmentMeasurements(size_label="TEST", chest_cm=body.chest_cm + chest_ease)


class TestEaseBandClassification:
    def test_tight_fit(self):
        result = FitEngine().evaluate(_body(), _garment_isolating_chest(-1.0))
        assert result.chest_fit == "tight"
        assert result.overall_fit == "tight"

    def test_regular_fit(self):
        result = FitEngine().evaluate(_body(), _garment_isolating_chest(5.0))
        assert result.chest_fit == "regular"
        assert result.overall_fit == "regular"

    def test_loose_fit(self):
        result = FitEngine().evaluate(_body(), _garment_isolating_chest(10.0))
        assert result.chest_fit == "loose"
        assert result.overall_fit == "loose"

    def test_oversized_fit(self):
        result = FitEngine().evaluate(_body(), _garment_isolating_chest(16.0))
        assert result.chest_fit == "very_loose"
        assert result.overall_fit == "oversized"

    def test_very_tight_fit(self):
        result = FitEngine().evaluate(_body(), _garment_isolating_chest(-5.0))
        assert result.chest_fit == "very_tight"
        assert result.overall_fit == "very_tight"

    def test_fitted_is_the_regular_boundary_case(self):
        result = FitEngine().evaluate(_body(), _garment_isolating_chest(0.0))
        assert result.chest_fit == "fitted"


class TestNoHardcodedSizeLabelMapping:
    def test_same_label_can_be_any_fit_depending_on_measurements(self):
        """An 'M' garment must classify purely from its own measurements —
        not from the letter M itself."""
        body = _body(chest_cm=110.0)  # a large person
        small_m = GarmentMeasurements(size_label="M", chest_cm=95.0)   # too small for them
        large_m = GarmentMeasurements(size_label="M", chest_cm=125.0)  # loose on them

        tight_result = FitEngine().evaluate(body, small_m)
        loose_result = FitEngine().evaluate(body, large_m)

        assert tight_result.selected_size == "M"
        assert loose_result.selected_size == "M"
        assert tight_result.overall_fit != loose_result.overall_fit


class TestInsufficientData:
    def test_missing_body_chest_is_insufficient_data(self):
        body = _body(chest_cm=None)
        garment = GarmentMeasurements(size_label="M", chest_cm=100.0)
        result = FitEngine().evaluate(body, garment)
        assert result.chest_fit == "insufficient_data"

    def test_missing_garment_measurements_entirely(self):
        body = _body()
        garment = GarmentMeasurements(size_label="M")  # label only
        result = FitEngine().evaluate(body, garment)
        assert result.overall_fit == "insufficient_data"
        assert "Not enough" in result.explanation


class TestConfidenceCalculation:
    def test_full_garment_completeness_preserves_body_confidence(self):
        assert combine_fit_confidence(0.7, 1.0) == 0.7

    def test_zero_garment_completeness_discounts_confidence(self):
        result = combine_fit_confidence(0.7, 0.0)
        assert result < 0.7

    def test_confidence_never_exceeds_ceiling(self):
        from api.app.services.fit_analysis.confidence import CONFIDENCE_CEILING
        assert combine_fit_confidence(1.0, 1.0) <= CONFIDENCE_CEILING

    def test_confidence_rejects_out_of_range_inputs(self):
        with pytest.raises(ValueError):
            combine_fit_confidence(1.5, 0.5)
        with pytest.raises(ValueError):
            combine_fit_confidence(0.5, -0.1)

    def test_fit_result_confidence_reflects_garment_completeness(self):
        body = _body()
        complete = GarmentMeasurements(size_label="M", chest_cm=95.0, shoulder_cm=44.0, length_cm=69.0)
        sparse = GarmentMeasurements(size_label="M", chest_cm=95.0)

        r_complete = FitEngine().evaluate(body, complete)
        r_sparse = FitEngine().evaluate(body, sparse)
        assert r_complete.confidence > r_sparse.confidence


class TestSizeRecommendation:
    def test_recommend_size_picks_closest_to_regular(self):
        body = _body(chest_cm=100.0, shoulder_cm=44.0, torso_length_cm=55.0, confidence=0.8)
        chart = list(DEFAULT_TSHIRT_SIZE_CHART.values())
        result = FitEngine().recommend_size(body, chart)
        assert result is not None
        assert result.selected_size in DEFAULT_TSHIRT_SIZE_CHART

    def test_recommend_size_empty_chart_returns_none(self):
        assert FitEngine().recommend_size(_body(), []) is None
