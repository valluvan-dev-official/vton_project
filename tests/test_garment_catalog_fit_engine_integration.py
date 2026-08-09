"""Proves the DB -> GarmentMeasurements -> FitEngine wiring actually works
end-to-end: two brands' "XL" rows, both fed through the exact same
production code path (garment_catalog.get_size_chart_entry +
to_garment_measurements), must produce DIFFERENT FitResults for the same
body — this is the whole point of keying the catalog by (merchant, sku,
size_label) instead of size_label alone.
"""
import os

import pytest

pytest.importorskip("sqlalchemy")
pytest.importorskip("aiosqlite")

TEST_DB_PATH = os.path.join(os.path.dirname(__file__), "_test_garment_catalog_fit_engine.db")
os.environ["DATABASE_URL"] = f"sqlite+aiosqlite:///{TEST_DB_PATH}"

import asyncio  # noqa: E402

import app.main  # noqa: E402, F401 — ensures every model (Job, BodyProfile, ...) is
                  # imported and registered on Base.metadata before create_all,
                  # since MerchantGarmentSizeChart's FK-adjacent siblings
                  # (via app.models.fit's STRETCH_CATEGORIES import) need
                  # Job's table to already be known.
from app.database import AsyncSessionLocal, Base, engine  # noqa: E402
from app.services import garment_catalog as svc  # noqa: E402
from app.services.fit_analysis.fit_engine import FitEngine  # noqa: E402
from app.services.fit_analysis.models import BodyMeasurementEstimate  # noqa: E402


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


def _same_body():
    return BodyMeasurementEstimate(
        chest_cm=100.0, shoulder_width_cm=44.0, torso_length_cm=55.0, confidence=0.7,
    )


async def _seed_two_brands():
    async with AsyncSessionLocal() as session:
        await svc.upsert_size_chart_entry(
            session, merchant="SnugBrand", sku="SKU-SNUG", size_label="XL",
            garment_category="t-shirt", chest_cm=101.0, shoulder_cm=44.5, length_cm=58.0,
        )
        await svc.upsert_size_chart_entry(
            session, merchant="RoomyBrand", sku="SKU-ROOMY", size_label="XL",
            garment_category="t-shirt", chest_cm=118.0, shoulder_cm=50.0, length_cm=70.0,
        )
        await session.commit()


class TestSameLabelDifferentBrandsProduceDifferentFits:
    def test_end_to_end_db_to_fit_result(self):
        asyncio.run(_seed_two_brands())

        async def _evaluate(merchant, sku):
            async with AsyncSessionLocal() as session:
                row = await svc.get_size_chart_entry(session, merchant, sku, "XL")
            assert row is not None
            garment = svc.to_garment_measurements(row)
            return FitEngine().evaluate(_same_body(), garment)

        snug_result = asyncio.run(_evaluate("SnugBrand", "SKU-SNUG"))
        roomy_result = asyncio.run(_evaluate("RoomyBrand", "SKU-ROOMY"))

        assert snug_result.selected_size == "XL"
        assert roomy_result.selected_size == "XL"
        # Same body, same size LABEL, genuinely different measurements ->
        # genuinely different fit classification.
        assert snug_result.overall_fit != roomy_result.overall_fit
        assert snug_result.chest_fit != roomy_result.chest_fit

    def test_missing_merchant_sku_size_combo_returns_none(self):
        async def _lookup():
            async with AsyncSessionLocal() as session:
                return await svc.get_size_chart_entry(session, "SnugBrand", "SKU-SNUG", "XXXL")

        assert asyncio.run(_lookup()) is None

    def test_measurement_source_is_merchant_provided(self):
        asyncio.run(_seed_two_brands())

        async def _fetch():
            async with AsyncSessionLocal() as session:
                return await svc.get_size_chart_entry(session, "SnugBrand", "SKU-SNUG", "XL")

        row = asyncio.run(_fetch())
        garment = svc.to_garment_measurements(row)
        assert garment.measurement_source == "merchant_provided"

    def test_stretch_and_fit_style_pass_through(self):
        async def _seed_and_fetch():
            async with AsyncSessionLocal() as session:
                await svc.upsert_size_chart_entry(
                    session, merchant="StretchBrand", sku="SKU-STRETCH", size_label="M",
                    chest_cm=95.0, stretch_category="high", fit_style="slim",
                )
                await session.commit()
                return await svc.get_size_chart_entry(session, "StretchBrand", "SKU-STRETCH", "M")

        row = asyncio.run(_seed_and_fetch())
        garment = svc.to_garment_measurements(row)
        assert garment.stretch_category == "high"
        assert garment.fit_style == "slim"

    def test_size_label_lookup_is_case_insensitive(self):
        async def _seed_and_fetch_lower():
            async with AsyncSessionLocal() as session:
                await svc.upsert_size_chart_entry(
                    session, merchant="CaseBrand", sku="SKU-CASE", size_label="XL", chest_cm=110.0,
                )
                await session.commit()
                return await svc.get_size_chart_entry(session, "CaseBrand", "SKU-CASE", "xl")

        row = asyncio.run(_seed_and_fetch_lower())
        assert row is not None
        assert row.size_label == "XL"
