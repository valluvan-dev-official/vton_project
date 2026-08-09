"""Phase 2 final review — merchant/sku/size_label normalization safety.

Policy under test (see app/models/garment_catalog.py's module docstring):
matching is CASE-INSENSITIVE and whitespace-insensitive. "AcmeApparel",
"acmeapparel", and "  Acme Apparel " must all resolve to the same logical
merchant — never silently create a duplicate "phantom" product. The
UNIQUE constraint is enforced on the normalized *_key columns at the
database level, not just in application code.
"""
import os

import pytest

pytest.importorskip("sqlalchemy")
pytest.importorskip("aiosqlite")

TEST_DB_PATH = os.path.join(os.path.dirname(__file__), "_test_garment_catalog_normalization.db")
os.environ["DATABASE_URL"] = f"sqlite+aiosqlite:///{TEST_DB_PATH}"

import asyncio  # noqa: E402

import app.main  # noqa: E402, F401 — full model registration, see test_garment_catalog_fit_engine_integration.py
from app.database import AsyncSessionLocal, Base, engine  # noqa: E402
from app.models.garment_catalog import normalize_key  # noqa: E402
from app.services import garment_catalog as svc  # noqa: E402
from fastapi.testclient import TestClient  # noqa: E402
from app.main import app  # noqa: E402


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


class TestNormalizeKeyFunction:
    def test_case_folded(self):
        assert normalize_key("AcmeApparel") == normalize_key("acmeapparel") == normalize_key("ACMEAPPAREL")

    def test_whitespace_trimmed_and_collapsed(self):
        assert normalize_key("  Acme   Apparel  ") == normalize_key("Acme Apparel")

    def test_none_and_empty_are_equal(self):
        assert normalize_key(None) == normalize_key("") == normalize_key("   ")

    def test_distinct_values_stay_distinct(self):
        assert normalize_key("AcmeApparel") != normalize_key("BoutiqueCo")


class TestModelValidatorsDeriveKeysAutomatically:
    def test_key_columns_set_from_display_values(self):
        from app.models.garment_catalog import MerchantGarmentSizeChart
        row = MerchantGarmentSizeChart(merchant="  Acme Apparel  ", sku=" SKU-1 ", size_label=" xl ")
        assert row.merchant == "Acme Apparel"  # trimmed, casing preserved for display
        assert row.merchant_key == "acme apparel"
        assert row.sku == "SKU-1"
        assert row.sku_key == "sku-1"
        assert row.size_label == "xl"
        assert row.size_label_key == "xl"


class TestUpsertNormalizationViaService:
    def test_upsert_then_lookup_with_different_case_finds_same_row(self):
        async def _run():
            async with AsyncSessionLocal() as session:
                created = await svc.upsert_size_chart_entry(
                    session, merchant="AcmeApparel", sku="TSHIRT-100", size_label="XL", chest_cm=115.0,
                )
                await session.commit()
                created_id = created.id

                found = await svc.get_size_chart_entry(session, "acmeapparel", "tshirt-100", "xl")
                assert found is not None
                assert found.id == created_id
                assert found.chest_cm == 115.0

        asyncio.run(_run())

    def test_upsert_with_different_case_updates_existing_row_not_duplicate(self):
        async def _run():
            async with AsyncSessionLocal() as session:
                await svc.upsert_size_chart_entry(
                    session, merchant="CaseBrand", sku="SKU-CASE-2", size_label="M", chest_cm=90.0,
                )
                await session.commit()

                # Re-upsert with different case/whitespace — must UPDATE the
                # same logical row, not create a second "phantom" one.
                await svc.upsert_size_chart_entry(
                    session, merchant="  casebrand  ", sku="sku-case-2", size_label="m", chest_cm=99.0,
                )
                await session.commit()

                rows = await svc.list_size_chart_entries(session, "CaseBrand", "SKU-CASE-2")
                assert len(rows) == 1
                assert rows[0].chest_cm == 99.0

        asyncio.run(_run())

    def test_db_level_unique_constraint_rejects_case_variant_duplicate_insert(self):
        """Even bypassing the service layer's existing-row check (raw ORM
        insert), the database's own UNIQUE constraint on the *_key columns
        must refuse a case/whitespace-variant duplicate."""
        from app.models.garment_catalog import MerchantGarmentSizeChart
        from sqlalchemy.exc import IntegrityError

        async def _run():
            async with AsyncSessionLocal() as session:
                session.add(MerchantGarmentSizeChart(merchant="DupBrand", sku="SKU-DUP", size_label="L", chest_cm=100.0))
                await session.commit()

            async with AsyncSessionLocal() as session:
                session.add(MerchantGarmentSizeChart(merchant="dupbrand", sku="sku-dup", size_label="l", chest_cm=999.0))
                with pytest.raises(IntegrityError):
                    await session.commit()

        asyncio.run(_run())


class TestApiLevelCaseInsensitiveDuplicatePrevention:
    def test_post_twice_with_different_case_updates_not_duplicates(self, client):
        client.post("/api/v1/garment-catalog/sizes", json=dict(
            merchant="WhitespaceBrand", sku="SKU-WS", size_label="M", chest_cm=90.0,
        ))
        client.post("/api/v1/garment-catalog/sizes", json=dict(
            merchant="  whitespacebrand ", sku=" sku-ws ", size_label=" m ", chest_cm=95.0,
        ))

        rows = client.get(
            "/api/v1/garment-catalog/sizes", params={"merchant": "WhitespaceBrand", "sku": "SKU-WS"}
        ).json()
        assert len(rows) == 1
        assert rows[0]["chest_cm"] == 95.0

    def test_get_with_different_case_finds_same_entry(self, client):
        client.post("/api/v1/garment-catalog/sizes", json=dict(
            merchant="LookupBrand", sku="SKU-LOOKUP", size_label="XL", chest_cm=110.0,
        ))
        resp = client.get("/api/v1/garment-catalog/sizes/lookupbrand/sku-lookup/xl")
        assert resp.status_code == 200
        assert resp.json()["chest_cm"] == 110.0


class TestUpsertCannotOverwriteAnotherMerchantsSku:
    def test_same_sku_different_merchant_stays_independent(self, client):
        """The critical isolation property: two merchants can legitimately
        reuse the same SKU string (they're different sellers) — updating
        one must never touch the other's row."""
        client.post("/api/v1/garment-catalog/sizes", json=dict(
            merchant="MerchantOne", sku="SHARED-SKU", size_label="M", chest_cm=90.0,
        ))
        client.post("/api/v1/garment-catalog/sizes", json=dict(
            merchant="MerchantTwo", sku="SHARED-SKU", size_label="M", chest_cm=200.0,
        ))

        # Update MerchantOne's entry only.
        client.post("/api/v1/garment-catalog/sizes", json=dict(
            merchant="MerchantOne", sku="SHARED-SKU", size_label="M", chest_cm=91.0,
        ))

        one = client.get("/api/v1/garment-catalog/sizes/MerchantOne/SHARED-SKU/M").json()
        two = client.get("/api/v1/garment-catalog/sizes/MerchantTwo/SHARED-SKU/M").json()
        assert one["chest_cm"] == 91.0
        assert two["chest_cm"] == 200.0  # untouched by MerchantOne's update

    def test_same_merchant_different_sku_same_size_stays_independent(self, client):
        client.post("/api/v1/garment-catalog/sizes", json=dict(
            merchant="MultiSkuBrand", sku="SKU-A", size_label="L", chest_cm=100.0,
        ))
        client.post("/api/v1/garment-catalog/sizes", json=dict(
            merchant="MultiSkuBrand", sku="SKU-B", size_label="L", chest_cm=150.0,
        ))
        client.post("/api/v1/garment-catalog/sizes", json=dict(
            merchant="MultiSkuBrand", sku="SKU-A", size_label="L", chest_cm=101.0,
        ))

        a = client.get("/api/v1/garment-catalog/sizes/MultiSkuBrand/SKU-A/L").json()
        b = client.get("/api/v1/garment-catalog/sizes/MultiSkuBrand/SKU-B/L").json()
        assert a["chest_cm"] == 101.0
        assert b["chest_cm"] == 150.0
