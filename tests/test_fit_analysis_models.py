"""Unit tests for api/app/services/fit_analysis/models.py — pure dataclasses,
no FastAPI/SQLAlchemy/torch. Covers validation rules from Phase-1 spec item 2
(never represent image-derived values as exact measurements — enforced via
measurement_source validation) and item 5 (invalid garment measurements).
"""
import pytest

from api.app.services.fit_analysis.models import (
    MEASUREMENT_SOURCE_ESTIMATED,
    MEASUREMENT_SOURCE_ILLUSTRATIVE_DEFAULT,
    MEASUREMENT_SOURCE_MERCHANT_PROVIDED,
    MEASUREMENT_SOURCE_USER_PROVIDED,
    BodyMeasurementEstimate,
    FitResult,
    GarmentMeasurements,
)


class TestMeasurementSource:
    def test_all_four_sources_are_valid(self):
        for source in (
            MEASUREMENT_SOURCE_ESTIMATED,
            MEASUREMENT_SOURCE_USER_PROVIDED,
            MEASUREMENT_SOURCE_MERCHANT_PROVIDED,
            MEASUREMENT_SOURCE_ILLUSTRATIVE_DEFAULT,
        ):
            body = BodyMeasurementEstimate(measurement_source=source, confidence=0.5)
            assert body.measurement_source == source

    def test_illustrative_default_is_distinct_from_merchant_provided(self):
        assert MEASUREMENT_SOURCE_ILLUSTRATIVE_DEFAULT != MEASUREMENT_SOURCE_MERCHANT_PROVIDED

    def test_invalid_source_rejected(self):
        with pytest.raises(ValueError):
            BodyMeasurementEstimate(measurement_source="guessed", confidence=0.5)


class TestBodyMeasurementEstimate:
    def test_default_is_all_none_with_estimated_source(self):
        body = BodyMeasurementEstimate()
        assert body.height_cm is None
        assert body.chest_cm is None
        assert body.measurement_source == MEASUREMENT_SOURCE_ESTIMATED
        assert body.confidence == 0.0

    def test_confidence_out_of_range_rejected(self):
        with pytest.raises(ValueError):
            BodyMeasurementEstimate(confidence=1.5)
        with pytest.raises(ValueError):
            BodyMeasurementEstimate(confidence=-0.1)

    def test_as_dict_round_trips_fields(self):
        body = BodyMeasurementEstimate(height_cm=170.0, chest_cm=95.0, confidence=0.6)
        d = body.as_dict()
        assert d["height_cm"] == 170.0
        assert d["chest_cm"] == 95.0
        assert d["confidence"] == 0.6


class TestGarmentMeasurements:
    def test_valid_merchant_entry(self):
        g = GarmentMeasurements(size_label="M", chest_cm=100.0, shoulder_cm=44.0, length_cm=69.0)
        assert g.measurement_source == MEASUREMENT_SOURCE_MERCHANT_PROVIDED
        assert g.completeness_ratio() == 1.0

    def test_empty_size_label_rejected(self):
        with pytest.raises(ValueError):
            GarmentMeasurements(size_label="   ")

    @pytest.mark.parametrize("field_name", ["chest_cm", "shoulder_cm", "waist_cm", "length_cm", "sleeve_length_cm", "hip_cm"])
    def test_non_positive_measurement_rejected(self, field_name):
        with pytest.raises(ValueError):
            GarmentMeasurements(size_label="M", **{field_name: 0})
        with pytest.raises(ValueError):
            GarmentMeasurements(size_label="M", **{field_name: -5})

    def test_completeness_ratio_partial(self):
        g = GarmentMeasurements(size_label="M", chest_cm=100.0)  # only 1 of 3 critical fields
        assert g.completeness_ratio() == pytest.approx(1 / 3)

    def test_completeness_ratio_empty(self):
        g = GarmentMeasurements(size_label="M")
        assert g.completeness_ratio() == 0.0


class TestFitResult:
    def test_confidence_out_of_range_rejected(self):
        with pytest.raises(ValueError):
            FitResult(
                selected_size="M", overall_fit="regular", chest_fit="fitted",
                shoulder_fit="fitted", length_fit="regular", confidence=1.2,
                explanation="x",
            )

    def test_is_shadow_mode_defaults_true(self):
        r = FitResult(
            selected_size="M", overall_fit="regular", chest_fit="fitted",
            shoulder_fit="fitted", length_fit="regular", confidence=0.7,
            explanation="x",
        )
        assert r.is_shadow_mode is True
