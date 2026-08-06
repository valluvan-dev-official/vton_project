"""baseline — represents the pre-Alembic schema

The `jobs` table (and everything before Phase 1) was created by
`Base.metadata.create_all()` at app startup (see app/database.py:init_db),
not by a migration — this repo had no migration history until now.

This revision is deliberately a no-op. Its only purpose is a stable
baseline id so that:
  * A FRESH database can run `alembic upgrade head` from empty and land on
    the same history as a deployed one.
  * An EXISTING deployed database (jobs table already present, created by
    create_all, no alembic_version table yet) can be marked as "already at
    this point" via `alembic stamp 0001_baseline` without alembic trying to
    (re)create jobs itself.

See api/alembic/README.deploy.md for the exact commands for each case.

Revision ID: 0001_baseline
Revises:
Create Date: 2026-08-06
"""
from typing import Sequence, Union

revision: str = "0001_baseline"
down_revision: Union[str, None] = None
branch_labels: Union[str, Sequence[str], None] = None
depends_on: Union[str, Sequence[str], None] = None


def upgrade() -> None:
    # Intentionally empty — jobs is owned by app startup's create_all(),
    # not by Alembic. See module docstring.
    pass


def downgrade() -> None:
    pass
