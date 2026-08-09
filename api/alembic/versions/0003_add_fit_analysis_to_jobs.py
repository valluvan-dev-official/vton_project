"""add fit_analysis_json to jobs (Phase 1 shadow-mode fit analysis)

Single nullable column on the existing `jobs` table — safe, additive,
backward-compatible (NULL for all existing rows; existing readers of
`jobs` that don't know this column exists are unaffected). See
api/app/services/fit_analysis/ for what populates it.

Revision ID: 0003_add_fit_analysis_to_jobs
Revises: 0002_add_fit_tables
Create Date: 2026-08-09
"""
from typing import Sequence, Union

import sqlalchemy as sa
from alembic import op

revision: str = "0003_add_fit_analysis_to_jobs"
down_revision: Union[str, None] = "0002_add_fit_tables"
branch_labels: Union[str, Sequence[str], None] = None
depends_on: Union[str, Sequence[str], None] = None


def upgrade() -> None:
    op.add_column("jobs", sa.Column("fit_analysis_json", sa.Text(), nullable=True))


def downgrade() -> None:
    op.drop_column("jobs", "fit_analysis_json")
