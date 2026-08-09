"""Explicit migration review/test for
api/alembic/versions/0004_add_merchant_garment_size_charts.py.

Runs the ACTUAL `alembic upgrade`/`downgrade` commands (not just importing
the migration module and eyeballing it) against a throwaway sqlite file, in
an isolated subprocess — a fresh Python process avoids this test suite's
existing @lru_cache(get_settings) / shared Base.metadata cross-file
contamination (the same class of issue documented in
tests/test_garment_catalog_fit_engine_integration.py), and is also a more
faithful rehearsal of the real `alembic upgrade head` operators would run.

Migration 0002's own upgrade() already documents that `jobs` is created by
app's `Base.metadata.create_all()` at startup, not by a migration — so this
test recreates that exact precondition (everything up through revision
0003 already applied) via create_all() + `alembic stamp`, then runs ONLY
migration 0004's upgrade/downgrade in isolation. This deliberately does not
re-test 0001-0003, which are unrelated to this change.
"""
import os
import subprocess
import sys
from pathlib import Path

import pytest

pytest.importorskip("sqlalchemy")
pytest.importorskip("aiosqlite")
pytest.importorskip("alembic")

API_DIR = Path(__file__).resolve().parents[1] / "api"
DB_FILENAME = "_test_migration_0004.db"
DB_PATH = API_DIR / DB_FILENAME


def _run(script: str) -> subprocess.CompletedProcess:
    env = dict(os.environ)
    env["DATABASE_URL"] = f"sqlite+aiosqlite:///{DB_FILENAME}"
    return subprocess.run(
        [sys.executable, "-c", script],
        cwd=str(API_DIR), env=env,
        capture_output=True, text=True, timeout=60,
    )


@pytest.fixture(autouse=True)
def _clean_db_file():
    DB_PATH.unlink(missing_ok=True)
    yield
    DB_PATH.unlink(missing_ok=True)


_CREATE_BASELINE_AND_STAMP = """
import asyncio
import app.main
from app.database import Base, engine
from alembic.config import Config
from alembic import command

async def create_baseline():
    async with engine.begin() as conn:
        tables = [t for name, t in Base.metadata.tables.items() if name != "merchant_garment_size_charts"]
        await conn.run_sync(lambda c: Base.metadata.create_all(c, tables=tables))
    await engine.dispose()

asyncio.run(create_baseline())
command.stamp(Config("alembic.ini"), "0003_add_fit_analysis_to_jobs")
"""

_UPGRADE = """
from alembic.config import Config
from alembic import command
command.upgrade(Config("alembic.ini"), "head")
"""

_DOWNGRADE = """
from alembic.config import Config
from alembic import command
command.downgrade(Config("alembic.ini"), "0003_add_fit_analysis_to_jobs")
"""

_INSPECT_SCHEMA = """
import asyncio, json, sqlalchemy as sa
from app.database import engine

async def inspect():
    async with engine.connect() as conn:
        def _do(c):
            insp = sa.inspect(c)
            tables = insp.get_table_names()
            out = {"tables": tables, "target_present": "merchant_garment_size_charts" in tables}
            if out["target_present"]:
                out["columns"] = sorted(col["name"] for col in insp.get_columns("merchant_garment_size_charts"))
                out["unique_constraints"] = insp.get_unique_constraints("merchant_garment_size_charts")
                out["indexes"] = insp.get_indexes("merchant_garment_size_charts")
                out["check_constraints"] = sorted(
                    cc["name"] for cc in insp.get_check_constraints("merchant_garment_size_charts")
                )
            return out
        result = await conn.run_sync(_do)
    await engine.dispose()
    print(json.dumps(result))

asyncio.run(inspect())
"""

_TEST_CONSTRAINTS = """
import asyncio
from sqlalchemy import text
from sqlalchemy.exc import IntegrityError
from app.database import engine

async def run():
    results = {}
    async with engine.begin() as conn:
        await conn.execute(text(
            "INSERT INTO merchant_garment_size_charts "
            "(id, merchant, sku, size_label, merchant_key, sku_key, size_label_key, "
            "stretch_category, fit_style, created_at, updated_at) VALUES "
            "('1','A','B','C','a','b','c','none','regular','2026-01-01','2026-01-01')"
        ))
    results["first_insert_ok"] = True

    try:
        async with engine.begin() as conn:
            await conn.execute(text(
                "INSERT INTO merchant_garment_size_charts "
                "(id, merchant, sku, size_label, merchant_key, sku_key, size_label_key, "
                "stretch_category, fit_style, created_at, updated_at) VALUES "
                "('2','a','b','c','a','b','c','none','regular','2026-01-01','2026-01-01')"
            ))
        results["duplicate_key_rejected"] = False
    except IntegrityError:
        results["duplicate_key_rejected"] = True

    try:
        async with engine.begin() as conn:
            await conn.execute(text(
                "INSERT INTO merchant_garment_size_charts "
                "(id, merchant, sku, size_label, merchant_key, sku_key, size_label_key, "
                "stretch_category, fit_style, created_at, updated_at) VALUES "
                "('3','X','Y','Z','x','y','z','WRONG','regular','2026-01-01','2026-01-01')"
            ))
        results["invalid_stretch_category_rejected"] = False
    except IntegrityError:
        results["invalid_stretch_category_rejected"] = True

    try:
        async with engine.begin() as conn:
            await conn.execute(text(
                "INSERT INTO merchant_garment_size_charts "
                "(id, merchant, sku, size_label, merchant_key, sku_key, size_label_key, "
                "stretch_category, fit_style, created_at, updated_at) VALUES "
                "('4','X','Y','Z2','x','y','z2','none','WRONG','2026-01-01','2026-01-01')"
            ))
        results["invalid_fit_style_rejected"] = False
    except IntegrityError:
        results["invalid_fit_style_rejected"] = True

    await engine.dispose()
    import json
    print(json.dumps(results))

asyncio.run(run())
"""


def _assert_ok(proc: subprocess.CompletedProcess, step: str):
    assert proc.returncode == 0, f"{step} failed:\\nSTDOUT: {proc.stdout}\\nSTDERR: {proc.stderr}"


class TestMigration0004UpgradeDowngrade:
    def test_upgrade_creates_expected_schema(self):
        _assert_ok(_run(_CREATE_BASELINE_AND_STAMP), "baseline+stamp")
        _assert_ok(_run(_UPGRADE), "upgrade to head")

        import json
        proc = _run(_INSPECT_SCHEMA)
        _assert_ok(proc, "inspect schema")
        schema = json.loads(proc.stdout.strip().splitlines()[-1])

        assert schema["target_present"] is True

        expected_columns = {
            "id", "merchant", "sku", "size_label",
            "merchant_key", "sku_key", "size_label_key",
            "garment_category", "chest_cm", "shoulder_cm", "waist_cm", "hip_cm",
            "length_cm", "sleeve_length_cm", "stretch_category", "fit_style",
            "created_at", "updated_at",
        }
        assert set(schema["columns"]) == expected_columns

        unique = schema["unique_constraints"]
        assert len(unique) == 1
        assert unique[0]["name"] == "uq_merchant_sku_size_key"
        assert set(unique[0]["column_names"]) == {"merchant_key", "sku_key", "size_label_key"}

        indexes = schema["indexes"]
        assert any(
            idx["name"] == "ix_merchant_garment_size_charts_merchant_sku_key"
            and set(idx["column_names"]) == {"merchant_key", "sku_key"}
            for idx in indexes
        )

        assert set(schema["check_constraints"]) == {
            "ck_merchant_garment_stretch_category", "ck_merchant_garment_fit_style",
        }

    def test_constraints_are_actually_enforced(self):
        _assert_ok(_run(_CREATE_BASELINE_AND_STAMP), "baseline+stamp")
        _assert_ok(_run(_UPGRADE), "upgrade to head")

        import json
        proc = _run(_TEST_CONSTRAINTS)
        _assert_ok(proc, "test constraints")
        results = json.loads(proc.stdout.strip().splitlines()[-1])

        assert results["first_insert_ok"] is True
        assert results["duplicate_key_rejected"] is True
        assert results["invalid_stretch_category_rejected"] is True
        assert results["invalid_fit_style_rejected"] is True

    def test_downgrade_removes_only_this_table(self):
        _assert_ok(_run(_CREATE_BASELINE_AND_STAMP), "baseline+stamp")
        _assert_ok(_run(_UPGRADE), "upgrade to head")
        _assert_ok(_run(_DOWNGRADE), "downgrade to 0003")

        import json
        proc = _run(_INSPECT_SCHEMA)
        _assert_ok(proc, "inspect schema after downgrade")
        schema = json.loads(proc.stdout.strip().splitlines()[-1])

        assert schema["target_present"] is False
        # Everything from the baseline (jobs, garment_size_charts, etc.)
        # must still be there — downgrade must not touch unrelated tables.
        assert "jobs" in schema["tables"]
        assert "garment_size_charts" in schema["tables"]
        assert "body_profiles" in schema["tables"]
        assert "fit_estimates" in schema["tables"]

    def test_upgrade_is_idempotent_reentry_safe_via_reported_revision(self):
        """A second 'upgrade head' after already being at head must be a
        no-op, not an error (standard Alembic guarantee) — sanity check
        that 0004 didn't break that."""
        _assert_ok(_run(_CREATE_BASELINE_AND_STAMP), "baseline+stamp")
        _assert_ok(_run(_UPGRADE), "first upgrade to head")
        _assert_ok(_run(_UPGRADE), "second upgrade to head (no-op)")
