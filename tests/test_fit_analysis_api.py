"""Backward-compatibility tests for the Phase-1 API/DB integration
(spec items 8, 10, 11):

  - Job.to_dict() gains an additive "fit_analysis" key (None when absent) —
    existing consumers reading the other keys are unaffected.
  - process_tryon_job keeps working when called with its pre-Phase-1
    positional signature (no height_cm) — old callers/tests don't break.
  - /tryon and /tryon/sync accept requests with no height_cm field at all
    (old client behavior) and validate height_cm bounds when provided.

Skips cleanly (via importorskip) wherever a dependency isn't installed in
the current environment, matching this test suite's existing convention
(see tests/test_fit_route.py, tests/test_letterbox_image.py).
"""
import inspect
import os
from datetime import datetime

import pytest

pytest.importorskip("sqlalchemy")

# Avoid requiring a postgres driver (asyncpg) in a lightweight test
# environment — matches tests/test_fit_route.py's own pattern. Must be set
# before app.database (and anything importing it) is first imported.
os.environ.setdefault("DATABASE_URL", "sqlite+aiosqlite:///./_test_fit_analysis_api.db")

from app.models.job import Job, JobStatus  # noqa: E402


class TestJobToDictBackwardCompatibility:
    def _make_job(self, **overrides):
        defaults = dict(
            id="job-1", status=JobStatus.completed,
            person_image_path="p.jpg", garment_image_path="g.jpg",
            garment_size="M", person_size_estimate="M",
            result_image_path="r.jpg", quality_score=0.9,
            saved_as_training=True, user_consent=True,
            error_message=None,
            created_at=datetime(2026, 1, 1), updated_at=datetime(2026, 1, 1),
        )
        defaults.update(overrides)
        return Job(**defaults)

    def test_existing_keys_all_still_present(self):
        job = self._make_job()
        d = job.to_dict()
        for key in (
            "id", "status", "result_url", "quality_score", "garment_size",
            "person_size_estimate", "saved_as_training", "user_consent",
            "error_message", "created_at", "updated_at",
        ):
            assert key in d

    def test_fit_analysis_defaults_to_none(self):
        """A job that predates Phase 1 (fit_analysis_json never set) must
        report fit_analysis: None, not raise or omit the key."""
        job = self._make_job()
        d = job.to_dict()
        assert d["fit_analysis"] is None

    def test_fit_analysis_parses_stored_json(self):
        job = self._make_job(fit_analysis_json='{"selected_size": "M", "overall_fit": "regular"}')
        d = job.to_dict()
        assert d["fit_analysis"] == {"selected_size": "M", "overall_fit": "regular"}

    def test_malformed_json_degrades_to_none_not_raise(self):
        job = self._make_job(fit_analysis_json="{not valid json")
        d = job.to_dict()
        assert d["fit_analysis"] is None

    def test_other_fields_unaffected_by_fit_analysis_presence(self):
        without = self._make_job().to_dict()
        with_fit = self._make_job(fit_analysis_json='{"x": 1}').to_dict()
        for key in ("id", "status", "quality_score", "garment_size", "person_size_estimate"):
            assert without[key] == with_fit[key]


class TestProcessTryonJobSignatureBackwardCompatibility:
    def setup_method(self):
        pytest.importorskip("celery")
        pytest.importorskip("skimage")

    def test_height_cm_is_optional_and_defaults_to_none(self):
        from app.workers.tasks import process_tryon_job

        sig = inspect.signature(process_tryon_job.run if hasattr(process_tryon_job, "run") else process_tryon_job)
        params = sig.parameters
        assert "height_cm" in params
        assert params["height_cm"].default is None

    def test_pre_phase1_positional_call_shape_still_matches(self):
        """The exact 4-positional-arg call already used by tryon.py's old
        code path (job_id, person_ref, garment_refs, garment_size) must
        still bind without a height_cm — i.e. it stays optional/trailing."""
        from app.workers.tasks import process_tryon_job

        fn = process_tryon_job.run if hasattr(process_tryon_job, "run") else process_tryon_job
        sig = inspect.signature(fn)
        # bind_partial with only the historical 4 args (skipping bound `self`
        # for a Celery task, which isn't part of the public call signature)
        names = [p for p in sig.parameters if p != "self"]
        assert names[:4] == ["job_id", "person_image_path", "garment_image_paths", "garment_size"]
        assert names[4] == "height_cm"


class TestTryonRouteHeightValidation:
    def setup_method(self):
        pytest.importorskip("celery")
        pytest.importorskip("sqlalchemy")

    def test_none_height_is_valid(self):
        from app.routes.tryon import _validate_height_cm
        assert _validate_height_cm(None) is None

    def test_in_range_height_passes_through(self):
        from app.routes.tryon import _validate_height_cm
        assert _validate_height_cm(175.0) == 175.0

    def test_out_of_range_height_rejected(self):
        from fastapi import HTTPException

        from app.routes.tryon import _validate_height_cm
        with pytest.raises(HTTPException):
            _validate_height_cm(30.0)
        with pytest.raises(HTTPException):
            _validate_height_cm(999.0)

    def test_garment_size_validation_unchanged(self):
        """Existing behavior (untouched by this feature) — still enforced."""
        from fastapi import HTTPException

        from app.routes.tryon import _validate_garment_size
        assert _validate_garment_size("m") == "M"
        with pytest.raises(HTTPException):
            _validate_garment_size("XXXL")
