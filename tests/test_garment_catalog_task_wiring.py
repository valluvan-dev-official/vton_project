"""Tests for _compute_fit_analysis()'s Phase 2 garment-resolution priority
in api/app/workers/tasks.py:

  1. real merchant catalog row (merchant, garment_sku, garment_size) if it exists
  2. illustrative DEFAULT_TSHIRT_SIZE_CHART, but ONLY when merchant/sku were
     not supplied at all, and only if FIT_ANALYSIS_ALLOW_DEFAULT_CATALOG=True
  3. otherwise insufficient_garment_data

The critical safety property under test: once a real (merchant, sku) was
given, a miss must NEVER silently fall back to the illustrative chart, even
if FIT_ANALYSIS_ALLOW_DEFAULT_CATALOG=True — that would mislabel one
specific (missing) product's sizing as generic/default data.
"""
import inspect

import pytest

pytest.importorskip("celery")
pytest.importorskip("skimage")
pytest.importorskip("PIL")

from PIL import Image  # noqa: E402

import app.workers.tasks as tasks  # noqa: E402
from app.services.fit_analysis.fit_engine import INSUFFICIENT_GARMENT_DATA  # noqa: E402
from app.services.fit_analysis.models import GarmentMeasurements  # noqa: E402


@pytest.fixture
def person_image_path(tmp_path):
    p = tmp_path / "person.jpg"
    Image.new("RGB", (200, 300), (255, 255, 255)).save(p)
    return str(p)


class TestMerchantCatalogTakesPriority:
    def test_real_merchant_hit_is_used_even_with_default_catalog_disabled(
        self, person_image_path, monkeypatch
    ):
        monkeypatch.setattr(tasks.settings, "FIT_ANALYSIS_ALLOW_DEFAULT_CATALOG", False, raising=False)
        fake_garment = GarmentMeasurements(size_label="XL", chest_cm=120.0, shoulder_cm=50.0, length_cm=75.0)
        monkeypatch.setattr(tasks, "_fetch_merchant_garment_measurements", lambda *a, **kw: fake_garment)

        import json
        result = tasks._compute_fit_analysis(
            person_image_path, "XL", 175.0, "M", merchant="Acme", garment_sku="SKU-1",
        )
        payload = json.loads(result)
        assert payload["selected_size"] == "XL"
        assert payload["overall_fit"] != INSUFFICIENT_GARMENT_DATA
        assert payload["measurement_source"] == "estimated"  # body's source, not garment's
        # Phase 2 review: garment_measurement_source is the field that must
        # clearly read "merchant_provided" when a real catalog SKU was used.
        assert payload["garment_measurement_source"] == "merchant_provided"

    def test_merchant_miss_never_falls_back_to_default_chart(
        self, person_image_path, monkeypatch
    ):
        """The critical safety property: even with the dev/test default
        chart explicitly enabled, a real-but-missing (merchant, sku)
        lookup must report insufficient_garment_data, not silently use
        generic illustrative numbers for that specific product."""
        monkeypatch.setattr(tasks.settings, "FIT_ANALYSIS_ALLOW_DEFAULT_CATALOG", True, raising=False)
        monkeypatch.setattr(tasks, "_fetch_merchant_garment_measurements", lambda *a, **kw: None)

        import json
        result = tasks._compute_fit_analysis(
            person_image_path, "M", 175.0, "M", merchant="UnknownBrand", garment_sku="NO-SUCH-SKU",
        )
        payload = json.loads(result)
        assert payload["overall_fit"] == INSUFFICIENT_GARMENT_DATA
        assert payload["confidence"] == 0.0
        assert payload["garment_measurement_source"] is None

    def test_no_merchant_sku_and_default_catalog_disabled_is_insufficient(
        self, person_image_path, monkeypatch
    ):
        monkeypatch.setattr(tasks.settings, "FIT_ANALYSIS_ALLOW_DEFAULT_CATALOG", False, raising=False)

        import json
        result = tasks._compute_fit_analysis(person_image_path, "M", 175.0, "M")
        payload = json.loads(result)
        assert payload["overall_fit"] == INSUFFICIENT_GARMENT_DATA

    def test_no_merchant_sku_falls_back_to_default_chart_when_explicitly_enabled(
        self, person_image_path, monkeypatch
    ):
        """Dev/test convenience path must still work when nothing about a
        real SKU was supplied at all."""
        monkeypatch.setattr(tasks.settings, "FIT_ANALYSIS_ALLOW_DEFAULT_CATALOG", True, raising=False)

        import json
        result = tasks._compute_fit_analysis(person_image_path, "M", 175.0, "M")
        payload = json.loads(result)
        assert payload["overall_fit"] != INSUFFICIENT_GARMENT_DATA
        # Phase 2 review: the illustrative dev/test chart must never be
        # reported as "merchant_provided" — that would be indistinguishable
        # from a real catalog hit.
        assert payload["garment_measurement_source"] == "illustrative_default"
        assert payload["garment_measurement_source"] != "merchant_provided"

    def test_merchant_without_sku_does_not_attempt_catalog_lookup(
        self, person_image_path, monkeypatch
    ):
        """Partial identifiers (should already be rejected at the API layer
        by _validate_merchant_sku, but defend in depth here too) must not
        crash or attempt a lookup with a None sku."""
        calls = []
        monkeypatch.setattr(
            tasks, "_fetch_merchant_garment_measurements",
            lambda *a, **kw: calls.append(a) or None,
        )
        import json
        result = tasks._compute_fit_analysis(
            person_image_path, "M", 175.0, "M", merchant="Acme", garment_sku=None,
        )
        assert calls == []  # never attempted — merchant alone is not enough
        payload = json.loads(result)
        assert payload["overall_fit"] == INSUFFICIENT_GARMENT_DATA


class TestBackwardCompatibleSignature:
    def test_process_tryon_job_signature_extended_not_broken(self):
        fn = tasks.process_tryon_job.run if hasattr(tasks.process_tryon_job, "run") else tasks.process_tryon_job
        sig = inspect.signature(fn)
        names = [p for p in sig.parameters if p != "self"]
        assert names == [
            "job_id", "person_image_path", "garment_image_paths", "garment_size",
            "height_cm", "merchant", "garment_sku",
        ]
        assert sig.parameters["height_cm"].default is None
        assert sig.parameters["merchant"].default is None
        assert sig.parameters["garment_sku"].default is None

    def test_compute_fit_analysis_still_works_with_old_4_positional_args(
        self, person_image_path
    ):
        """Old call shape (no merchant/sku) must still work unchanged."""
        import json
        result = tasks._compute_fit_analysis(person_image_path, "M", 175.0, "M")
        assert result is not None
        json.loads(result)  # must be valid JSON


class TestValidateMerchantSku:
    def test_both_none_is_valid(self):
        from app.routes.tryon import _validate_merchant_sku
        assert _validate_merchant_sku(None, None) == (None, None)

    def test_both_present_is_valid_and_trimmed(self):
        from app.routes.tryon import _validate_merchant_sku
        assert _validate_merchant_sku("  Acme  ", " SKU-1 ") == ("Acme", "SKU-1")

    def test_merchant_without_sku_rejected(self):
        from fastapi import HTTPException

        from app.routes.tryon import _validate_merchant_sku
        with pytest.raises(HTTPException):
            _validate_merchant_sku("Acme", None)

    def test_sku_without_merchant_rejected(self):
        from fastapi import HTTPException

        from app.routes.tryon import _validate_merchant_sku
        with pytest.raises(HTTPException):
            _validate_merchant_sku(None, "SKU-1")

    def test_empty_strings_treated_as_none(self):
        from app.routes.tryon import _validate_merchant_sku
        assert _validate_merchant_sku("   ", "   ") == (None, None)
