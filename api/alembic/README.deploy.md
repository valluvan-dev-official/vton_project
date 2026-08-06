# Deploying the Phase 1 fit-tables migration

`jobs` is created by the app itself (`Base.metadata.create_all()` in
`app/database.py`, run on startup), not by Alembic — this repo had no
migration history before Phase 1. `0001_baseline` is a no-op revision that
exists only to give Alembic a stable starting point on top of that
pre-existing schema.

Run all commands from `api/`.

## Existing deployed database (jobs table already exists, no alembic_version table)

```
alembic stamp 0001_baseline   # mark the pre-existing schema as the baseline, without running anything
alembic upgrade head          # applies 0002_add_fit_tables only
```

## Brand new database

Start the API once so `init_db()` creates `jobs` (or otherwise ensure `jobs`
exists), then:

```
alembic upgrade head          # 0001_baseline (no-op), then 0002_add_fit_tables
```

If `jobs` does not exist yet, `0002_add_fit_tables` raises a `RuntimeError`
with this same instruction instead of failing with an opaque FK error.

## Rolling back Phase 1 only

```
alembic downgrade 0001_baseline
```

Drops `fit_estimates`, `body_profiles`, `garment_size_charts`. Does not touch
`jobs` or any existing data.
