"""API/DB tests for the Phase 2 merchant garment size-chart endpoints
(POST/GET /api/v1/garment-catalog/sizes...).

Core requirement under test: two different merchants selling a product in
the same size_label ("XL") must be stored and retrieved independently, with
different measurements — this is exactly what app/models/fit.py's older
GarmentSizeChart (keyed by garment_category+size_label only) cannot do.

Uses a throwaway sqlite file DB, same pattern as tests/test_fit_route.py.
"""
import os

import pytest

pytest.importorskip("sqlalchemy")
pytest.importorskip("aiosqlite")

TEST_DB_PATH = os.path.join(os.path.dirname(__file__), "_test_garment_catalog_api.db")
os.environ["DATABASE_URL"] = f"sqlite+aiosqlite:///{TEST_DB_PATH}"

import asyncio  # noqa: E402

from fastapi.testclient import TestClient  # noqa: E402

from app.main import app  # noqa: E402
from app.database import Base, engine  # noqa: E402


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


def _entry(**overrides):
    base = dict(
        merchant="AcmeApparel", sku="TSHIRT-001", size_label="XL",
        garment_category="t-shirt", chest_cm=116.0, shoulder_cm=49.0,
        waist_cm=None, hip_cm=None, length_cm=73.0, sleeve_length_cm=23.0,
        stretch_category="none", fit_style="regular",
    )
    base.update(overrides)
    return base


class TestMultiBrandSameSizeLabel:
    def test_two_merchants_same_xl_label_store_independently(self, client):
        acme = _entry(merchant="AcmeApparel", sku="TSHIRT-001", size_label="XL", chest_cm=116.0)
        boutique = _entry(merchant="BoutiqueCo", sku="TEE-042", size_label="XL", chest_cm=104.0)

        r1 = client.post("/api/v1/garment-catalog/sizes", json=acme)
        r2 = client.post("/api/v1/garment-catalog/sizes", json=boutique)
        assert r1.status_code == 200
        assert r2.status_code == 200

        got_acme = client.get(
            "/api/v1/garment-catalog/sizes/AcmeApparel/TSHIRT-001/XL"
        ).json()
        got_boutique = client.get(
            "/api/v1/garment-catalog/sizes/BoutiqueCo/TEE-042/XL"
        ).json()

        assert got_acme["chest_cm"] == 116.0
        assert got_boutique["chest_cm"] == 104.0
        assert got_acme["chest_cm"] != got_boutique["chest_cm"]
        assert got_acme["id"] != got_boutique["id"]

    def test_three_brands_same_xl_all_different_measurements(self, client):
        brands = {
            "BrandA": 110.0,
            "BrandB": 118.0,
            "BrandC": 122.0,
        }
        for brand, chest in brands.items():
            client.post("/api/v1/garment-catalog/sizes", json=_entry(
                merchant=brand, sku="SKU-XL-TEST", size_label="XL", chest_cm=chest,
            ))

        results = {
            brand: client.get(f"/api/v1/garment-catalog/sizes/{brand}/SKU-XL-TEST/XL").json()["chest_cm"]
            for brand in brands
        }
        assert results == brands
        assert len(set(results.values())) == 3  # all genuinely distinct


class TestUpsertSemantics:
    def test_posting_same_key_twice_updates_not_duplicates(self, client):
        key = dict(merchant="UpsertBrand", sku="SKU-UP", size_label="M")
        client.post("/api/v1/garment-catalog/sizes", json=_entry(**key, chest_cm=90.0))
        client.post("/api/v1/garment-catalog/sizes", json=_entry(**key, chest_cm=95.0))

        rows = client.get(
            "/api/v1/garment-catalog/sizes", params={"merchant": "UpsertBrand", "sku": "SKU-UP"}
        ).json()
        assert len(rows) == 1
        assert rows[0]["chest_cm"] == 95.0


class TestArbitrarySizeLabels:
    @pytest.mark.parametrize("label", ["EU 42", "Size 10", "One Size", "2XL", "38R"])
    def test_non_standard_size_labels_accepted(self, client, label):
        client.post("/api/v1/garment-catalog/sizes", json=_entry(
            merchant="EuroBrand", sku="SKU-EURO", size_label=label,
        ))
        got = client.get(f"/api/v1/garment-catalog/sizes/EuroBrand/SKU-EURO/{label}")
        assert got.status_code == 200
        assert got.json()["size_label"] == label


class TestListAndNotFound:
    def test_list_returns_only_matching_merchant_and_sku(self, client):
        client.post("/api/v1/garment-catalog/sizes", json=_entry(
            merchant="ListBrand", sku="SKU-L", size_label="S", chest_cm=90.0,
        ))
        client.post("/api/v1/garment-catalog/sizes", json=_entry(
            merchant="ListBrand", sku="SKU-L", size_label="L", chest_cm=105.0,
        ))
        client.post("/api/v1/garment-catalog/sizes", json=_entry(
            merchant="OtherBrand", sku="SKU-L", size_label="S", chest_cm=200.0,
        ))

        rows = client.get(
            "/api/v1/garment-catalog/sizes", params={"merchant": "ListBrand", "sku": "SKU-L"}
        ).json()
        assert {r["size_label"] for r in rows} == {"S", "L"}
        assert all(r["merchant"] == "ListBrand" for r in rows)

    def test_get_missing_entry_is_404(self, client):
        resp = client.get("/api/v1/garment-catalog/sizes/NoSuchBrand/NoSuchSku/XL")
        assert resp.status_code == 404


class TestValidation:
    def test_invalid_stretch_category_rejected(self, client):
        resp = client.post("/api/v1/garment-catalog/sizes", json=_entry(stretch_category="extreme"))
        assert resp.status_code == 422

    def test_invalid_fit_style_rejected(self, client):
        resp = client.post("/api/v1/garment-catalog/sizes", json=_entry(fit_style="baggy"))
        assert resp.status_code == 422

    def test_negative_measurement_rejected(self, client):
        resp = client.post("/api/v1/garment-catalog/sizes", json=_entry(chest_cm=-10))
        assert resp.status_code == 422

    def test_missing_required_fields_rejected(self, client):
        resp = client.post("/api/v1/garment-catalog/sizes", json={"size_label": "M"})
        assert resp.status_code == 422

    def test_all_measurement_fields_optional(self, client):
        """A merchant might not have every dimension yet — must not force
        all fields, only merchant/sku/size_label are required."""
        resp = client.post("/api/v1/garment-catalog/sizes", json=dict(
            merchant="SparseBrand", sku="SKU-SPARSE", size_label="M",
        ))
        assert resp.status_code == 200
        assert resp.json()["chest_cm"] is None
