"""Unit tests for garment_measurements.py — Phase-1 spec item 5 (merchant
data is authoritative; do not infer from S/M/L/XL/XXL labels alone) and the
"S/M/L/XL/XXL catalog measurements" test requirement.
"""
import pytest

from api.app.services.fit_analysis.garment_measurements import (
    DEFAULT_TSHIRT_SIZE_CHART,
    get_default_garment_measurements,
    validate_garment_measurements,
)
from api.app.services.fit_analysis.models import GarmentMeasurements


class TestDefaultSizeChart:
    @pytest.mark.parametrize("size", ["XS", "S", "M", "L", "XL", "XXL"])
    def test_every_standard_size_present(self, size):
        g = get_default_garment_measurements(size)
        assert g is not None
        assert g.size_label == size
        assert g.chest_cm is not None
        assert g.shoulder_cm is not None
        assert g.length_cm is not None

    def test_lookup_is_case_insensitive(self):
        assert get_default_garment_measurements("m") is not None
        assert get_default_garment_measurements("m").size_label == "M"

    def test_unknown_size_returns_none(self):
        assert get_default_garment_measurements("3XL") is None
        assert get_default_garment_measurements("") is None

    def test_measurements_increase_monotonically_with_size(self):
        """Sanity check on the illustrative chart itself — a bigger label
        must never measure smaller than the one before it."""
        order = ["XS", "S", "M", "L", "XL", "XXL"]
        chests = [DEFAULT_TSHIRT_SIZE_CHART[s].chest_cm for s in order]
        shoulders = [DEFAULT_TSHIRT_SIZE_CHART[s].shoulder_cm for s in order]
        assert chests == sorted(chests)
        assert shoulders == sorted(shoulders)

    def test_default_chart_is_illustrative_not_merchant_provided_source(self):
        """Phase 2 review fix: this chart is made-up dev/test data, not a
        real merchant's catalog — it must never claim measurement_source
        "merchant_provided" (that would be indistinguishable from a real
        catalog hit in fit_analysis output)."""
        for g in DEFAULT_TSHIRT_SIZE_CHART.values():
            assert g.measurement_source == "illustrative_default"
            assert g.measurement_source != "merchant_provided"


class TestValidateGarmentMeasurements:
    def test_complete_entry_has_no_warnings(self):
        g = GarmentMeasurements(size_label="M", chest_cm=100, shoulder_cm=44, length_cm=69)
        assert validate_garment_measurements(g) == []

    def test_empty_merchant_entry_warns(self):
        g = GarmentMeasurements(size_label="M")  # label only, no catalog data
        warnings = validate_garment_measurements(g)
        assert len(warnings) == 1
        assert "M" in warnings[0]

    def test_missing_garment_measurements_is_a_valid_state_not_an_error(self):
        """Phase-1 spec item 5: label-only is legitimate, just low-completeness —
        constructing it must never raise."""
        g = GarmentMeasurements(size_label="XL")
        assert g.chest_cm is None
        assert g.completeness_ratio() == 0.0
