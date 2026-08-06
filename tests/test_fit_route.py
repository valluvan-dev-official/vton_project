"""API/DB tests for POST /api/v1/fit/estimate (Phase 1).

Needs the full app stack (FastAPI, SQLAlchemy async, aiosqlite) — unlike
test_fit_calculator.py, which only needs pytest. Uses a throwaway SQLite
file DB via aiosqlite so it doesn't require a running Postgres instance.
"""
import asyncio
import os
import uuid

import pytest

TEST_DB_PATH = os.path.join(os.path.dirname(__file__), "_test_fit_route.db")
os.environ["DATABASE_URL"] = f"sqlite+aiosqlite:///{TEST_DB_PATH}"

from fastapi.testclient import TestClient  # noqa: E402

from app.main import app  # noqa: E402
from app.database import Base, engine, AsyncSessionLocal  # noqa: E402
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


def _valid_body(**overrides):
    body = {"height_cm": 165, "bust_cm": 91, "waist_cm": 75, "hips_cm": 97, "shoulder_cm": None}
    body.update(overrides)
    return body


def _chart():
    return [
        {"size_label": "S", "bust_cm": 88, "waist_cm": 72, "hips_cm": 94},
        {"size_label": "M", "bust_cm": 94, "waist_cm": 78, "hips_cm": 100},
        {"size_label": "L", "bust_cm": 100, "waist_cm": 84, "hips_cm": 106},
    ]


async def _insert_job(job_id: str):
    async with AsyncSessionLocal() as session:
        job = Job(
            id=job_id,
            status=JobStatus.pending,
            person_image_path="x",
            garment_image_path="y",
            garment_size="M",
        )
        session.add(job)
        await session.commit()


async def _fetch_row_counts():
    from sqlalchemy import select, func
    from app.models.fit import BodyProfile, FitEstimate

    async with AsyncSessionLocal() as session:
        bp_count = (await session.execute(select(func.count()).select_from(BodyProfile))).scalar_one()
        fe_count = (await session.execute(select(func.count()).select_from(FitEstimate))).scalar_one()
        return bp_count, fe_count


# ── Contract tests ───────────────────────────────────────────────────────────

def test_no_garment_chart_never_guesses(client):
    res = client.post("/api/v1/fit/estimate", json={
        "body": _valid_body(),
        "selected_size": "M",
        "garment_measurements": [],
    })
    assert res.status_code == 200
    data = res.json()
    assert data["recommended_size"] is None
    assert all(r["fit"] == "insufficient_data" for r in data["regional_fit"].values())
    assert data["body_shape"] != "insufficient_data"  # body shape is independent of garment availability
    assert data["disclaimer"].startswith("AI fit estimate")
    assert data["is_approximate"] is True


def test_with_garment_chart_recommends_size(client):
    res = client.post("/api/v1/fit/estimate", json={
        "body": _valid_body(),
        "selected_size": "S",
        "garment_measurements": _chart(),
    })
    assert res.status_code == 200
    data = res.json()
    assert data["selected_size"] == "S"
    assert data["recommended_size"] in ("S", "M", "L")
    assert set(data["regional_fit"].keys()) == {"bust", "waist", "hips"}


def test_missing_bust_waist_hips_is_insufficient_not_rejected(client):
    res = client.post("/api/v1/fit/estimate", json={
        "body": {"height_cm": 165, "bust_cm": None, "waist_cm": None, "hips_cm": None, "shoulder_cm": None},
        "selected_size": "M",
        "garment_measurements": _chart(),
    })
    assert res.status_code == 200
    data = res.json()
    assert data["body_shape"] == "insufficient_data"
    assert data["confidence"] == 0.0
    assert all(r["fit"] == "insufficient_data" for r in data["regional_fit"].values())


def test_missing_height_is_rejected_422(client):
    # height_cm is the one required body field per the approved contract.
    body = _valid_body()
    del body["height_cm"]
    res = client.post("/api/v1/fit/estimate", json={
        "body": body,
        "selected_size": "M",
        "garment_measurements": [],
    })
    assert res.status_code == 422


def test_out_of_range_measurement_is_422(client):
    res = client.post("/api/v1/fit/estimate", json={
        "body": _valid_body(height_cm=400),  # way outside 100-250
        "selected_size": "M",
        "garment_measurements": [],
    })
    assert res.status_code == 422


def test_selected_size_always_echoed(client):
    res = client.post("/api/v1/fit/estimate", json={
        "body": _valid_body(),
        "selected_size": "XL",
        "garment_measurements": _chart(),
    })
    assert res.status_code == 200
    assert res.json()["selected_size"] == "XL"


def test_unknown_job_id_returns_404(client):
    res = client.post("/api/v1/fit/estimate", json={
        "job_id": str(uuid.uuid4()),
        "body": _valid_body(),
        "selected_size": "M",
        "garment_measurements": [],
    })
    assert res.status_code == 404


def test_null_job_id_is_allowed(client):
    res = client.post("/api/v1/fit/estimate", json={
        "job_id": None,
        "body": _valid_body(),
        "selected_size": "M",
        "garment_measurements": [],
    })
    assert res.status_code == 200
    assert res.json()["job_id"] is None


def test_valid_job_id_is_linked(client):
    job_id = str(uuid.uuid4())
    asyncio.run(_insert_job(job_id))

    res = client.post("/api/v1/fit/estimate", json={
        "job_id": job_id,
        "body": _valid_body(),
        "selected_size": "M",
        "garment_measurements": [],
    })
    assert res.status_code == 200
    assert res.json()["job_id"] == job_id


def test_body_profile_and_fit_estimate_persisted_atomically(client):
    before_bp, before_fe = asyncio.run(_fetch_row_counts())

    res = client.post("/api/v1/fit/estimate", json={
        "body": _valid_body(),
        "selected_size": "M",
        "garment_measurements": _chart(),
    })
    assert res.status_code == 200

    after_bp, after_fe = asyncio.run(_fetch_row_counts())
    assert after_bp == before_bp + 1
    assert after_fe == before_fe + 1
