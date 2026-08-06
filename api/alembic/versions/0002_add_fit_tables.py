"""add Phase 1 fit tables — body_profiles, garment_size_charts, fit_estimates

Adds three new tables only. `jobs` is untouched — `fit_estimates.job_id` and
`body_profiles.job_id` are nullable foreign keys into it.

Because `jobs` is created by the app's own `Base.metadata.create_all()` at
startup rather than by a migration (see 0001_baseline), this revision
guards against running on a database where `jobs` doesn't exist yet: the FK
target must be present at DB level before the constraint can be created.
Start the API once (or otherwise ensure `jobs` exists) before applying this
migration on a brand new database.

Revision ID: 0002_add_fit_tables
Revises: 0001_baseline
Create Date: 2026-08-06
"""
from typing import Sequence, Union

import sqlalchemy as sa
from sqlalchemy.dialects.postgresql import JSONB
from alembic import op

revision: str = "0002_add_fit_tables"
down_revision: Union[str, None] = "0001_baseline"
branch_labels: Union[str, Sequence[str], None] = None
depends_on: Union[str, Sequence[str], None] = None

_JSONType = sa.JSON().with_variant(JSONB(), "postgresql")

STRETCH_CATEGORIES = ("none", "low", "medium", "high")
FIT_STYLES = ("slim", "regular", "relaxed", "oversized")


def upgrade() -> None:
    bind = op.get_bind()
    inspector = sa.inspect(bind)
    if "jobs" not in inspector.get_table_names():
        raise RuntimeError(
            "Cannot apply 0002_add_fit_tables: the 'jobs' table does not exist yet. "
            "fit_estimates/body_profiles reference jobs.id via a foreign key. "
            "Start the API once (app startup runs Base.metadata.create_all() and "
            "creates 'jobs'), or otherwise ensure 'jobs' exists, then re-run "
            "`alembic upgrade head`."
        )

    op.create_table(
        "garment_size_charts",
        sa.Column("id", sa.String(36), primary_key=True),
        sa.Column("garment_category", sa.String(64), nullable=False),
        sa.Column("size_label", sa.String(16), nullable=False),
        sa.Column("bust_cm", sa.Float, nullable=True),
        sa.Column("waist_cm", sa.Float, nullable=True),
        sa.Column("hips_cm", sa.Float, nullable=True),
        sa.Column("shoulder_cm", sa.Float, nullable=True),
        sa.Column("length_cm", sa.Float, nullable=True),
        sa.Column("sleeve_cm", sa.Float, nullable=True),
        sa.Column("stretch_category", sa.String(16), nullable=False, server_default="none"),
        sa.Column("fit_style", sa.String(16), nullable=False, server_default="regular"),
        sa.Column("created_at", sa.DateTime, nullable=False),
        sa.UniqueConstraint("garment_category", "size_label", name="uq_garment_category_size"),
        sa.CheckConstraint(f"stretch_category IN {STRETCH_CATEGORIES}", name="ck_garment_stretch_category"),
        sa.CheckConstraint(f"fit_style IN {FIT_STYLES}", name="ck_garment_fit_style"),
    )
    op.create_index("ix_garment_size_charts_category", "garment_size_charts", ["garment_category"])

    op.create_table(
        "body_profiles",
        sa.Column("id", sa.String(36), primary_key=True),
        sa.Column("job_id", sa.String(36), sa.ForeignKey("jobs.id"), nullable=True),
        sa.Column("height_cm", sa.Float, nullable=False),
        sa.Column("bust_cm", sa.Float, nullable=True),
        sa.Column("waist_cm", sa.Float, nullable=True),
        sa.Column("hips_cm", sa.Float, nullable=True),
        sa.Column("shoulder_cm", sa.Float, nullable=True),
        sa.Column("created_at", sa.DateTime, nullable=False),
    )
    op.create_index("ix_body_profiles_job_id", "body_profiles", ["job_id"])

    op.create_table(
        "fit_estimates",
        sa.Column("id", sa.String(36), primary_key=True),
        sa.Column("job_id", sa.String(36), sa.ForeignKey("jobs.id"), nullable=True),
        sa.Column("body_profile_id", sa.String(36), sa.ForeignKey("body_profiles.id"), nullable=False),
        sa.Column("selected_size", sa.String(16), nullable=False),
        sa.Column("recommended_size", sa.String(16), nullable=True),
        sa.Column("body_shape", sa.String(32), nullable=False),
        sa.Column("confidence", sa.Float, nullable=False),
        sa.Column("regional_fit", _JSONType, nullable=False),
        sa.Column("summary", sa.Text, nullable=False),
        sa.Column("is_approximate", sa.Boolean, nullable=False, server_default=sa.true()),
        sa.Column("created_at", sa.DateTime, nullable=False),
    )
    op.create_index("ix_fit_estimates_job_id", "fit_estimates", ["job_id"])
    op.create_index("ix_fit_estimates_body_profile_id", "fit_estimates", ["body_profile_id"])


def downgrade() -> None:
    op.drop_index("ix_fit_estimates_body_profile_id", table_name="fit_estimates")
    op.drop_index("ix_fit_estimates_job_id", table_name="fit_estimates")
    op.drop_table("fit_estimates")

    op.drop_index("ix_body_profiles_job_id", table_name="body_profiles")
    op.drop_table("body_profiles")

    op.drop_index("ix_garment_size_charts_category", table_name="garment_size_charts")
    op.drop_table("garment_size_charts")
