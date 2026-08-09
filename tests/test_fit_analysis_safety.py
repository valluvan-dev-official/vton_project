"""Pre-commit safety checks for Phase 1 fit_analysis:

  - DEFAULT_TSHIRT_SIZE_CHART must never be silently treated as authoritative
    merchant sizing; FitEngine.insufficient_garment_data() is what a caller
    must use instead when no real catalog data exists.
  - Image-derived body cm measurements must never expose false precision —
    whole-cm rounding + an explicit (low, high) uncertainty range.
  - fit_analysis must never be reachable from ml/scripts/gpu_inference.py —
    a structural guard, not just a manual diff check.
"""
import re
from pathlib import Path

import pytest

from api.app.services.fit_analysis.body_analyzer import BodyAnalyzer
from api.app.services.fit_analysis.confidence import (
    UNCERTAINTY_PCT_FLOOR,
    estimate_uncertainty_pct,
)
from api.app.services.fit_analysis.fit_engine import INSUFFICIENT_GARMENT_DATA, FitEngine
from api.app.services.fit_analysis.garment_measurements import DEFAULT_TSHIRT_SIZE_CHART
from api.app.services.fit_analysis.models import BodyMeasurementEstimate
from api.app.services.fit_analysis.pose_adapter import PoseLandmarks


class TestInsufficientGarmentData:
    def test_shape_and_labels(self):
        result = FitEngine().insufficient_garment_data("M")
        assert result.selected_size == "M"
        assert result.overall_fit == INSUFFICIENT_GARMENT_DATA
        assert result.chest_fit == INSUFFICIENT_GARMENT_DATA
        assert result.shoulder_fit == INSUFFICIENT_GARMENT_DATA
        assert result.length_fit == INSUFFICIENT_GARMENT_DATA

    def test_confidence_is_zero_not_a_computed_value(self):
        """Must not look like a real (if low) computed confidence — 0.0 is
        an unambiguous 'no computation happened' signal."""
        result = FitEngine().insufficient_garment_data("XL")
        assert result.confidence == 0.0

    def test_never_equals_a_real_evaluate_result_label(self):
        """insufficient_garment_data must be visually/programmatically
        distinct from every real ease-band label FitEngine.evaluate() can
        produce, so a consumer can't mistake one for the other."""
        from api.app.services.fit_analysis.fit_engine import EASE_BANDS, LENGTH_EASE_BANDS
        real_labels = {b[0] for b in EASE_BANDS} | {b[0] for b in LENGTH_EASE_BANDS} | {"insufficient_data"}
        assert INSUFFICIENT_GARMENT_DATA not in real_labels


class TestDefaultCatalogIsNeverImplicitlyAuthoritative:
    def test_default_chart_entries_are_illustrative_not_wired_to_a_real_merchant(self):
        """Sanity: nothing in the chart claims a real product/merchant id —
        it's generic by construction."""
        for g in DEFAULT_TSHIRT_SIZE_CHART.values():
            assert g.garment_type == "t-shirt"  # generic category, not a SKU/product reference

    def test_module_docstring_documents_the_production_gate(self):
        """The gating contract (settings.FIT_ANALYSIS_ALLOW_DEFAULT_CATALOG)
        must be documented at the source, not just tribal knowledge."""
        import api.app.services.fit_analysis.garment_measurements as gm
        assert "FIT_ANALYSIS_ALLOW_DEFAULT_CATALOG" in gm.__doc__
        assert "NOT FOR PRODUCTION" in gm.__doc__

    def test_settings_default_is_off(self):
        """The flag itself must default to False — safe-by-default even if
        a deployment forgets to set it explicitly."""
        from api.app.config import Settings
        assert Settings().FIT_ANALYSIS_ALLOW_DEFAULT_CATALOG is False


class TestNoFalsePrecision:
    def _landmarks(self, quality=0.6):
        return PoseLandmarks(
            shoulder_width_px=140.0, torso_length_px=180.0,
            hip_width_px=95.0, arm_length_px=190.0,
            image_width=384, image_height=512, quality=quality,
        )

    def test_cm_values_are_whole_numbers_not_decimals(self):
        """A rough frontal-photo estimate should not look more precise than
        it is — no fractional centimetres."""
        body = BodyAnalyzer().analyze(height_cm=175.0, landmarks=self._landmarks())
        for value in (body.shoulder_width_cm, body.chest_cm, body.waist_cm,
                      body.hip_cm, body.torso_length_cm, body.arm_length_cm):
            if value is not None:
                assert value == int(value), f"{value} is not a whole number"

    def test_uncertainty_pct_is_always_set_when_measurements_exist(self):
        body = BodyAnalyzer().analyze(height_cm=175.0, landmarks=self._landmarks())
        assert body.uncertainty_pct is not None
        assert body.uncertainty_pct >= UNCERTAINTY_PCT_FLOOR

    def test_range_is_wider_than_a_single_point_and_brackets_it(self):
        body = BodyAnalyzer().analyze(height_cm=175.0, landmarks=self._landmarks())
        assert body.chest_cm is not None
        lo, hi = body.chest_cm_range
        assert lo < body.chest_cm < hi

    def test_range_widens_as_confidence_drops(self):
        low_q = BodyAnalyzer().analyze(height_cm=175.0, landmarks=self._landmarks(quality=0.1))
        high_q = BodyAnalyzer().analyze(height_cm=175.0, landmarks=self._landmarks(quality=0.95))
        low_lo, low_hi = low_q.chest_cm_range
        high_lo, high_hi = high_q.chest_cm_range
        assert (low_hi - low_lo) >= (high_hi - high_lo)

    def test_range_is_none_when_value_is_none(self):
        body = BodyAnalyzer().analyze(height_cm=None, landmarks=None)
        assert body.chest_cm is None
        assert body.chest_cm_range is None

    def test_as_dict_exposes_ranges_alongside_point_values(self):
        body = BodyAnalyzer().analyze(height_cm=175.0, landmarks=self._landmarks())
        d = body.as_dict()
        assert "chest_cm_range" in d
        assert "shoulder_width_cm_range" in d
        assert d["chest_cm_range"] is not None

    def test_estimate_uncertainty_pct_never_zero(self):
        """Even a maximally confident estimate keeps a nonzero band — a
        photo-derived figure is never claimed exact."""
        assert estimate_uncertainty_pct(1.0) >= UNCERTAINTY_PCT_FLOOR
        assert estimate_uncertainty_pct(1.0) > 0.0

    def test_measurement_source_still_estimated_even_with_range(self):
        body = BodyAnalyzer().analyze(height_cm=175.0, landmarks=self._landmarks())
        assert body.measurement_source == "estimated"


class TestNeverConnectedToRendering:
    """Structural guard, independent of manual diff review: fit_analysis
    must not appear anywhere in ml/scripts/gpu_inference.py's source, and
    gpu_inference.py must not be importable from within fit_analysis."""

    def test_gpu_inference_source_does_not_mention_fit_analysis(self):
        gpu_inference_path = (
            Path(__file__).resolve().parents[1] / "ml" / "scripts" / "gpu_inference.py"
        )
        source = gpu_inference_path.read_text(encoding="utf-8")
        assert "fit_analysis" not in source
        assert "FitEngine" not in source
        assert "BodyAnalyzer" not in source

    def test_fit_analysis_package_does_not_import_gpu_inference(self):
        fit_analysis_dir = (
            Path(__file__).resolve().parents[1] / "api" / "app" / "services" / "fit_analysis"
        )
        for py_file in fit_analysis_dir.glob("*.py"):
            source = py_file.read_text(encoding="utf-8")
            assert not re.search(r"^\s*(from|import)\s+gpu_inference\b", source, re.MULTILINE), (
                f"{py_file} imports gpu_inference directly"
            )
