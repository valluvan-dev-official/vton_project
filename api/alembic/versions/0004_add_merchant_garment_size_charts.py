"""add merchant_garment_size_charts (Phase 2 — production merchant catalog)

New table only, no changes to any existing table. Keyed by
(merchant, sku, size_label) so the same size label ("XL") can carry
different measurements per merchant/product — unlike the pre-existing
garment_size_charts table (0002), which is keyed by (garment_category,
size_label) only and cannot represent that.

Normalization (Phase 2 review): merchant/sku/size_label matching is
case- and whitespace-insensitive — see
app/models/garment_catalog.py's module docstring. merchant_key/sku_key/
size_label_key hold the lowercased, whitespace-collapsed comparison form;
the UNIQUE constraint and lookup index are on those columns, not the
display columns, so the database itself refuses a case/whitespace-variant
duplicate of the same logical product+size (not just app-level checks).

Revision ID: 0004_add_merchant_garment_size_charts
Revises: 0003_add_fit_analysis_to_jobs
Create Date: 2026-08-10
"""
from typing import Sequence, Union

import sqlalchemy as sa
from alembic import op

revision: str = "0004_add_merchant_garment_size_charts"
down_revision: Union[str, None] = "0003_add_fit_analysis_to_jobs"
branch_labels: Union[str, Sequence[str], None] = None
depends_on: Union[str, Sequence[str], None] = None

STRETCH_CATEGORIES = ("none", "low", "medium", "high")
FIT_STYLES = ("slim", "regular", "relaxed", "oversized")


def upgrade() -> None:
    op.create_table(
        "merchant_garment_size_charts",
        sa.Column("id", sa.String(36), primary_key=True),
        sa.Column("merchant", sa.String(128), nullable=False),
        sa.Column("sku", sa.String(128), nullable=False),
        sa.Column("size_label", sa.String(32), nullable=False),
        sa.Column("merchant_key", sa.String(128), nullable=False),
        sa.Column("sku_key", sa.String(128), nullable=False),
        sa.Column("size_label_key", sa.String(32), nullable=False),
        sa.Column("garment_category", sa.String(64), nullable=True),
        sa.Column("chest_cm", sa.Float, nullable=True),
        sa.Column("shoulder_cm", sa.Float, nullable=True),
        sa.Column("waist_cm", sa.Float, nullable=True),
        sa.Column("hip_cm", sa.Float, nullable=True),
        sa.Column("length_cm", sa.Float, nullable=True),
        sa.Column("sleeve_length_cm", sa.Float, nullable=True),
        sa.Column("stretch_category", sa.String(16), nullable=False, server_default="none"),
        sa.Column("fit_style", sa.String(16), nullable=False, server_default="regular"),
        sa.Column("created_at", sa.DateTime, nullable=False),
        sa.Column("updated_at", sa.DateTime, nullable=False),
        sa.UniqueConstraint("merchant_key", "sku_key", "size_label_key", name="uq_merchant_sku_size_key"),
        sa.CheckConstraint(f"stretch_category IN {STRETCH_CATEGORIES}", name="ck_merchant_garment_stretch_category"),
        sa.CheckConstraint(f"fit_style IN {FIT_STYLES}", name="ck_merchant_garment_fit_style"),
    )
    op.create_index(
        "ix_merchant_garment_size_charts_merchant_sku_key",
        "merchant_garment_size_charts", ["merchant_key", "sku_key"],
    )


def downgrade() -> None:
    op.drop_index("ix_merchant_garment_size_charts_merchant_sku_key", table_name="merchant_garment_size_charts")
    op.drop_table("merchant_garment_size_charts")
