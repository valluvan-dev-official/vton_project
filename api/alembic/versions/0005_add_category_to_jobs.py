"""add category to jobs (multi-engine try-on: upper_body/lower_body/dresses)

Single non-nullable column with a server_default of "upper_body" on the
existing `jobs` table — safe, additive, backward-compatible: existing rows
backfill to "upper_body" (matching the behavior every job had before this
column existed, since get_mask_location() was hardcoded to that literal),
and callers that don't send `category` yet keep getting the same result.

Revision ID: 0005_add_category_to_jobs
Revises: 0004_add_merchant_garment_size_charts
Create Date: 2026-08-13
"""
from typing import Sequence, Union

import sqlalchemy as sa
from alembic import op

revision: str = "0005_add_category_to_jobs"
down_revision: Union[str, None] = "0004_add_merchant_garment_size_charts"
branch_labels: Union[str, Sequence[str], None] = None
depends_on: Union[str, Sequence[str], None] = None


def upgrade() -> None:
    op.add_column(
        "jobs",
        sa.Column("category", sa.String(length=16), nullable=False, server_default="upper_body"),
    )


def downgrade() -> None:
    op.drop_column("jobs", "category")
