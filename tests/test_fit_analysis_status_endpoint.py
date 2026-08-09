"""Real FastAPI endpoint test for GET /api/v1/status/{job_id} — proves
(not just unit-tests in isolation) that:

  - the optional fit_analysis object round-trips through the actual HTTP
    response when present,
  - it's null (not a missing key, not an error) when absent, and
  - every pre-Phase-1 response field is still there unchanged — i.e. an
    existing client that only reads e.g. "status"/"result_url" keeps working.

Mirrors tests/test_fit_route.py's throwaway-sqlite-file pattern so it needs
no running Postgres/Celery/Redis.
"""
import asyncio
import os

import pytest

pytest.importorskip("sqlalchemy")
pytest.importorskip("aiosqlite")

TEST_DB_PATH = os.path.join(os.path.dirname(__file__), "_test_fit_analysis_status.db")
os.environ["DATABASE_URL"] = f"sqlite+aiosqlite:///{TEST_DB_PATH}"

from fastapi.testclient import TestClient  # noqa: E402

from app.main import app  # noqa: E402
from app.database import AsyncSessionLocal, Base, engine  # noqa: E402
from app.models.job import Job, JobStatus  # noqa: E402


@pytest.fixture(scope="module", autouse=True)
def _setup_db():
    if os.path.exists(TEST_DB_PATH):
        os.remove(TEST_DB_PATH)

    async def create():
        async with engine.begin() as conn:
            await conn.run_sync(Base.metadata.create_all)

    asyncio.run(create())
    yield
    asyncio.run(engine.dispose())
    if os.path.exists(TEST_DB_PATH):
        os.remove(TEST_DB_PATH)


@pytest.fixture
def client():
    return TestClient(app)


async def _insert_job(job_id: str, **overrides):
    defaults = dict(
        id=job_id, status=JobStatus.completed,
        person_image_path="p.jpg", garment_image_path="g.jpg",
        garment_size="M", person_size_estimate="M",
        result_image_path=None, quality_score=0.9,
    )
    defaults.update(overrides)
    async with AsyncSessionLocal() as session:
        session.add(Job(**defaults))
        await session.commit()


class TestStatusEndpointFitAnalysis:
    def test_status_includes_null_fit_analysis_when_absent(self, client):
        """A job that predates Phase 1 / had analysis fail — response must
        still include the key, valued null, not omit it or error."""
        job_id = "job-no-fit-analysis"
        asyncio.run(_insert_job(job_id))

        resp = client.get(f"/api/v1/status/{job_id}")
        assert resp.status_code == 200
        body = resp.json()
        assert "fit_analysis" in body
        assert body["fit_analysis"] is None

    def test_status_includes_populated_fit_analysis_object(self, client):
        job_id = "job-with-fit-analysis"
        fit_json = (
            '{"selected_size": "M", "overall_fit": "regular", "chest_fit": "fitted", '
            '"shoulder_fit": "regular", "length_fit": "regular", "confidence": 0.62, '
            '"explanation": "x", "is_shadow_mode": true, "estimated_person_size": "M", '
            '"measurement_source": "estimated"}'
        )
        asyncio.run(_insert_job(job_id, fit_analysis_json=fit_json))

        resp = client.get(f"/api/v1/status/{job_id}")
        assert resp.status_code == 200
        body = resp.json()
        assert body["fit_analysis"] is not None
        assert body["fit_analysis"]["selected_size"] == "M"
        assert body["fit_analysis"]["overall_fit"] == "regular"
        assert body["fit_analysis"]["is_shadow_mode"] is True

    def test_status_includes_insufficient_garment_data_shape(self, client):
        """The production-safe default path (no real catalog) — status
        must surface this clearly rather than a bare fit-looking object."""
        job_id = "job-insufficient-garment-data"
        fit_json = (
            '{"selected_size": "L", "overall_fit": "insufficient_garment_data", '
            '"chest_fit": "insufficient_garment_data", "shoulder_fit": "insufficient_garment_data", '
            '"length_fit": "insufficient_garment_data", "confidence": 0.0, "explanation": "x", '
            '"is_shadow_mode": true, "estimated_person_size": "M", "measurement_source": null}'
        )
        asyncio.run(_insert_job(job_id, fit_analysis_json=fit_json))

        resp = client.get(f"/api/v1/status/{job_id}")
        body = resp.json()
        assert body["fit_analysis"]["overall_fit"] == "insufficient_garment_data"
        assert body["fit_analysis"]["confidence"] == 0.0

    def test_all_pre_phase1_fields_still_present_and_unaffected(self, client):
        """Backward compatibility: an existing client reading only the old
        field set must see identical values whether or not fit_analysis
        is populated."""
        job_a, job_b = "job-compat-a", "job-compat-b"
        asyncio.run(_insert_job(job_a, quality_score=0.77, person_size_estimate="L"))
        asyncio.run(_insert_job(job_b, quality_score=0.77, person_size_estimate="L",
                                 fit_analysis_json='{"x": 1}'))

        body_a = client.get(f"/api/v1/status/{job_a}").json()
        body_b = client.get(f"/api/v1/status/{job_b}").json()

        # "id" deliberately excluded — it's expected to differ between two
        # different job rows; everything else must be identical regardless
        # of whether fit_analysis is populated.
        pre_existing_keys = (
            "status", "result_url", "quality_score", "garment_size",
            "person_size_estimate", "saved_as_training", "user_consent",
            "error_message",
        )
        for key in pre_existing_keys:
            assert key in body_a
            assert key in body_b
            assert body_a[key] == body_b[key], f"{key} differs based on fit_analysis presence"

    def test_status_404_for_unknown_job_unaffected(self, client):
        """Existing error behavior for a missing job must be untouched."""
        resp = client.get("/api/v1/status/does-not-exist")
        assert resp.status_code == 404
