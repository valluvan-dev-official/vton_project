"""
Celery tasks for async try-on job processing.

Flow:
  1. Download person + garment(s) from S3 to temp local files
  2. Run inference via InferenceRouter — IDM-VTON, own trained model, or
     placeholder, whichever OWN_MODEL_CHECKPOINT / DEVICE / WEIGHTS_DIR
     selects (see app/services/inference.py). Swapping models is therefore
     a config change, not a code change.
  3. Upload result to S3
  4. Calculate SSIM quality score
  5. If score >= MIN_QUALITY_SCORE → auto-save as training pair + update pairs.json
  6. Update job status in DB (result_image_path stores the S3 key)

When STORAGE_BACKEND=local the download/upload steps are no-ops that just
copy from/to LOCAL_STORAGE_PATH, so the local dev workflow is unchanged.
"""
import json
import os
import sys
import tempfile
import uuid

if sys.platform != "win32":
    import fcntl
from datetime import datetime
from pathlib import Path

from celery import Celery
from skimage.metrics import structural_similarity as ssim
import numpy as np

from app.config import get_settings
from app.services.inference import get_inference_router
from app.services.storage import get_storage

settings = get_settings()

celery_app = Celery(
    "vton",
    broker=settings.CELERY_BROKER_URL,
    backend=settings.CELERY_RESULT_BACKEND,
)
celery_app.conf.update(
    task_serializer="json",
    result_serializer="json",
    accept_content=["json"],
    timezone="UTC",
    enable_utc=True,
)


# ── DB helpers ────────────────────────────────────────────────────────────────

def _update_job(job_id: str, updates: dict):
    """Synchronous DB update run inside a Celery task via asyncio."""
    import asyncio
    from sqlalchemy import update
    from sqlalchemy.ext.asyncio import create_async_engine, async_sessionmaker
    from app.models.job import Job

    engine = create_async_engine(settings.DATABASE_URL)
    Session = async_sessionmaker(engine, expire_on_commit=False)

    async def _run():
        async with Session() as session:
            await session.execute(update(Job).where(Job.id == job_id).values(**updates))
            await session.commit()
        await engine.dispose()

    asyncio.run(_run())


def _fetch_merchant_garment_measurements(merchant: str, sku: str, size_label: str):
    """Synchronous DB lookup of one merchant SKU's size-chart entry, mapped
    to fit_analysis's GarmentMeasurements — same sync-wrapped-async pattern
    as _update_job above (fresh engine/session per call; Celery tasks are
    sync, so each call gets its own short-lived asyncio loop rather than
    sharing one across the worker process).

    Returns None if no such (merchant, sku, size_label) row exists — the
    caller (_compute_fit_analysis) treats that as insufficient_garment_data,
    never as a reason to fail the job.
    """
    import asyncio
    from sqlalchemy.ext.asyncio import create_async_engine, async_sessionmaker
    from app.services.garment_catalog import get_size_chart_entry, to_garment_measurements

    engine = create_async_engine(settings.DATABASE_URL)
    Session = async_sessionmaker(engine, expire_on_commit=False)

    async def _run():
        async with Session() as session:
            row = await get_size_chart_entry(session, merchant, sku, size_label)
            result = to_garment_measurements(row) if row is not None else None
        await engine.dispose()
        return result

    return asyncio.run(_run())


# ── Quality scoring ───────────────────────────────────────────────────────────

def _compute_ssim(path_a: str, path_b: str) -> float:
    from PIL import Image
    def load(p):
        img = Image.open(p).convert("RGB").resize((256, 256))
        return np.array(img)
    a, b = load(path_a), load(path_b)
    score, _ = ssim(a, b, full=True, channel_axis=2, data_range=255)
    return float(score)


# ── Fit analysis (Phase 1, shadow mode) ─────────────────────────────────────
#
# Computed AFTER inference has already produced a result, purely as
# additional metadata. Never influences the try-on render — see
# app/services/fit_analysis/__init__.py. Any failure here is caught and
# logged; it must never fail the try-on job itself.

def _compute_fit_analysis(
    local_person: str, garment_size: str, height_cm, person_size_estimate: str,
    merchant=None, garment_sku=None,
) -> "str | None":
    """Returns a JSON string (FitAnalysisOut shape) or None if analysis
    could not be computed at all (an unexpected error). Missing/insufficient
    garment catalog data is NOT treated as an error — it still returns a
    JSON payload, with every fit field reporting "insufficient_garment_data"
    (see FitEngine.insufficient_garment_data()) rather than silently
    presenting the illustrative DEFAULT_TSHIRT_SIZE_CHART as authoritative
    merchant sizing.

    Garment measurement resolution priority (Phase 2):
      1. Real merchant catalog row for (merchant, garment_sku, garment_size)
         — see app/services/garment_catalog.py. Authoritative; used
         whenever both merchant and garment_sku are supplied, regardless
         of FIT_ANALYSIS_ALLOW_DEFAULT_CATALOG.
      2. The illustrative DEFAULT_TSHIRT_SIZE_CHART — dev/test only, and
         only when FIT_ANALYSIS_ALLOW_DEFAULT_CATALOG is explicitly True.
      3. Neither available -> insufficient_garment_data.

    Timed and logged on every path (success, insufficient data, or
    failure) so completion-time impact on the try-on job is always visible.
    """
    import logging
    import time
    logger = logging.getLogger(__name__)
    start = time.perf_counter()

    try:
        import json as _json
        from PIL import Image

        from app.services.fit_analysis import (
            BodyAnalyzer,
            EngineReadOnlyPoseAdapter,
            FitEngine,
            get_default_garment_measurements,
            try_get_pose_engine,
        )

        engine_instance = FitEngine()

        garment = None
        garment_source_note = "none"

        if merchant and garment_sku:
            garment = _fetch_merchant_garment_measurements(merchant, garment_sku, garment_size)
            garment_source_note = "merchant_catalog" if garment is not None else "merchant_catalog_miss"

        # DEFAULT_TSHIRT_SIZE_CHART is a dev/test fallback ONLY (see
        # garment_measurements.py) — never used as if it were real merchant
        # data unless explicitly opted into via settings, and never used at
        # all once a real (merchant, garment_sku) lookup has already run
        # (whether it hit or missed) — falling back to illustrative data
        # after an explicit real-catalog lookup failed would silently
        # mislabel a specific product's sizing as generic.
        if garment is None and garment_source_note == "none" and settings.FIT_ANALYSIS_ALLOW_DEFAULT_CATALOG:
            garment = get_default_garment_measurements(garment_size)
            garment_source_note = "default_chart" if garment is not None else "none"

        if garment is None:
            logger.info(
                "fit_analysis: no authoritative garment measurements for size "
                "%r (merchant=%r sku=%r resolution=%s FIT_ANALYSIS_ALLOW_DEFAULT_CATALOG=%s) "
                "— reporting insufficient_garment_data rather than an apparent fit.",
                garment_size, merchant, garment_sku, garment_source_note,
                settings.FIT_ANALYSIS_ALLOW_DEFAULT_CATALOG,
            )
            fit_result = engine_instance.insufficient_garment_data(garment_size)
            payload = fit_result.as_dict()
            payload["estimated_person_size"] = person_size_estimate
            # "measurement_source" here is the BODY estimate's source (see
            # below on the success path) — kept None on this path since no
            # body analysis was attempted either. "garment_measurement_source"
            # is the garment side specifically — see its comment below.
            payload["measurement_source"] = None
            payload["garment_measurement_source"] = None
            return _json.dumps(payload)

        # Read-only reuse of the already-loaded GPU pose engine (see
        # pose_adapter.py) — never constructs/loads a fresh model; returns
        # None immediately for non-GPU deployments (try_get_pose_engine's
        # own device/weights_dir short-circuit).
        landmarks = None
        engine = try_get_pose_engine(settings.DEVICE, settings.WEIGHTS_DIR)
        if engine is not None:
            person_img = Image.open(local_person).convert("RGB")
            landmarks = EngineReadOnlyPoseAdapter(engine).get_landmarks(person_img)

        body = BodyAnalyzer().analyze(height_cm=height_cm, landmarks=landmarks)
        fit_result = engine_instance.evaluate(body, garment)

        payload = fit_result.as_dict()
        payload["estimated_person_size"] = person_size_estimate
        # "measurement_source" = how the BODY measurements were obtained
        # (always "estimated" today — image-derived). "garment_measurement_
        # source" = how the GARMENT measurements were obtained: this is the
        # field that must read "merchant_provided" when a real catalog SKU
        # was matched, vs. "illustrative_default" for the dev/test chart —
        # never conflate the two (Phase 2 review fix: previously only the
        # body's source was surfaced at all).
        payload["measurement_source"] = body.measurement_source
        payload["garment_measurement_source"] = garment.measurement_source
        return _json.dumps(payload)

    except Exception:
        logger.warning(
            "fit_analysis: shadow-mode computation failed; job result is "
            "unaffected, fit_analysis will be omitted.", exc_info=True,
        )
        return None

    finally:
        elapsed_ms = (time.perf_counter() - start) * 1000
        logger.info(
            "fit_analysis: _compute_fit_analysis completed in %.1fms "
            "(shadow mode — no effect on try-on job outcome).", elapsed_ms,
        )


# ── Training pair persistence ─────────────────────────────────────────────────

def _save_training_pair(job_id: str, person_path: str, garment_path: str,
                        result_path: str, score: float):
    """
    Copy the trio to storage/training_pairs/<job_id>/ and append to pairs.json.
    ml/src/data/dataset.py reads from this directory.
    With S3 backend the files are uploaded to S3_PREFIX_TRAINING/<job_id>/.
    """
    storage = get_storage()
    pair_dir = f"{settings.S3_PREFIX_TRAINING}/{job_id}"

    person_key  = f"{pair_dir}/person{Path(person_path).suffix}"
    garment_key = f"{pair_dir}/garment{Path(garment_path).suffix}"
    result_key  = f"{pair_dir}/result{Path(result_path).suffix}"

    storage.save(person_path,  person_key)
    storage.save(garment_path, garment_key)
    storage.save(result_path,  result_key)

    # Per-pair meta file — written locally and also uploaded
    meta = {
        "job_id":        job_id,
        "quality_score": score,
        "person":        person_key,
        "garment":       garment_key,
        "result":        result_key,
        "saved_at":      datetime.utcnow().isoformat(),
    }
    meta_path = Path(settings.LOCAL_STORAGE_PATH) / "training_pairs" / job_id / "meta.json"
    meta_path.parent.mkdir(parents=True, exist_ok=True)
    meta_path.write_text(json.dumps(meta, indent=2))

    # Central index — pairs.json (file-lock for concurrent workers)
    pairs_json = Path(settings.LOCAL_STORAGE_PATH) / "training_pairs" / "pairs.json"
    pairs_json.parent.mkdir(parents=True, exist_ok=True)

    # Read → append → write with an exclusive lock
    if sys.platform != "win32":
        try:
            with open(str(pairs_json), "a+") as fh:
                try:
                    fcntl.flock(fh, fcntl.LOCK_EX)
                    fh.seek(0)
                    raw = fh.read().strip()
                    entries = json.loads(raw) if raw else []
                    entries.append(meta)
                    fh.seek(0)
                    fh.truncate()
                    json.dump(entries, fh, indent=2)
                finally:
                    fcntl.flock(fh, fcntl.LOCK_UN)
        except Exception:
            existing = json.loads(pairs_json.read_text()) if pairs_json.exists() else []
            existing.append(meta)
            pairs_json.write_text(json.dumps(existing, indent=2))
    else:
        existing = json.loads(pairs_json.read_text()) if pairs_json.exists() else []
        existing.append(meta)
        pairs_json.write_text(json.dumps(existing, indent=2))


# ── S3 input helpers ──────────────────────────────────────────────────────────

def _resolve_local_path(s3_key_or_path: str, job_id: str, role: str) -> str:
    """Ensure the image is available as a local file and return its path.

    If STORAGE_BACKEND=s3 and the value looks like an S3 key (no leading /),
    download it to a temp file and return that path.

    If STORAGE_BACKEND=local or the path already exists, return it as-is
    (after the /app/storage path-fix for Docker vs host differences).
    """
    p = s3_key_or_path

    # Docker path fix (kept from original implementation)
    if p.startswith("/app/storage/") and not Path(p).exists():
        p = p.replace("/app/storage", str(settings.LOCAL_STORAGE_PATH), 1)

    if Path(p).exists():
        return p

    # Treat as S3 key — download to a temp file
    import logging
    logger = logging.getLogger(__name__)
    storage = get_storage()
    suffix = Path(p).suffix or ".jpg"
    tmp = tempfile.NamedTemporaryFile(
        suffix=suffix, delete=False,
        dir=Path(settings.LOCAL_STORAGE_PATH) / "tmp",
        prefix=f"{job_id}_{role}_",
    )
    tmp.close()
    logger.info("tasks: Downloading input (%s) from storage key %s...", role, p)
    storage.load(p, tmp.name)
    return tmp.name


# ── Main Celery task ──────────────────────────────────────────────────────────

@celery_app.task(bind=True, name="tasks.process_tryon_job", max_retries=2)
def process_tryon_job(self, job_id: str, person_image_path: str,
                       garment_image_paths, garment_size: str = "M",
                       height_cm=None, merchant=None, garment_sku=None,
                       category: str = "upper_body", dress_subtype: str = None):
    """
    Main try-on pipeline:
      1. Resolve input images (download from S3 if needed)
      2. Run GPU inference — singleton engine, models loaded once
      3. Upload result to S3
      4. SSIM quality score
      5. Auto-save training pair when score >= MIN_QUALITY_SCORE
      5b. (Phase 1/2, shadow mode) Compute fit_analysis metadata —
          best-effort, never affects render output or job success/failure.
          See _compute_fit_analysis() above.
      6. Update DB — completed / failed

    height_cm, merchant, garment_sku are optional and default to None so
    existing callers that invoke this task with an older, shorter
    positional signature keep working unchanged. category defaults to
    "upper_body" for the same reason — that's the literal every job was
    hardcoded to before this param existed.

    dress_subtype: "saree" | "salwar_suit" | None — only meaningful when
    category == "dresses"; selects a dedicated sleeve-detection module in
    the GPU inference engine (see GPUInferenceEngine.run()'s docstring).
    Defaults to None (falls back to the generic gown/frock dress handler)
    so existing callers that don't send it yet see no behavior change.
    """
    import logging
    logger = logging.getLogger(__name__)
    tmp_files: list[str] = []   # track temp files to clean up

    # Backward-compat: accept a single string path (old callers/tests) as
    # well as the new list-of-paths (multiple garment angles).
    if isinstance(garment_image_paths, str):
        garment_image_paths = [garment_image_paths]

    try:
        _update_job(job_id, {"status": "processing"})

        # ── 1. Resolve local paths for inference ──────────────────────────
        # Create tmp dir inside LOCAL_STORAGE_PATH (always accessible)
        tmp_dir = Path(settings.LOCAL_STORAGE_PATH) / "tmp"
        tmp_dir.mkdir(parents=True, exist_ok=True)

        local_person = _resolve_local_path(person_image_path, job_id, "person")
        local_garments = [
            _resolve_local_path(p, job_id, f"garment{i}")
            for i, p in enumerate(garment_image_paths)
        ]

        # Track any temp files created by _resolve_local_path
        for p in [local_person, *local_garments]:
            if str(tmp_dir) in p:
                tmp_files.append(p)

        # ── 2. Inference ──────────────────────────────────────────────────
        output_path = str(tmp_dir / f"{job_id}_output.jpg")
        logger.info(
            "tasks: Starting inference for job %s (%d garment image(s), garment_size=%s, category=%s)...",
            job_id, len(local_garments), garment_size, category,
        )
        router = get_inference_router()
        router.run(
            local_person, local_garments, output_path, job_id=job_id,
            garment_size=garment_size, category=category, dress_subtype=dress_subtype,
        )
        person_size_estimate = router.last_person_size_estimate
        logger.info(
            "tasks: Inference complete for job %s (person_size_estimate=%s).",
            job_id, person_size_estimate,
        )

        # ── 3. Upload result to S3 ────────────────────────────────────────
        result_s3_key = f"{settings.S3_PREFIX_OUTPUT}/{job_id}.jpg"
        storage = get_storage()
        logger.info("tasks: Uploading output to S3 key %s...", result_s3_key)
        storage.save(output_path, result_s3_key)
        logger.info("tasks: Output uploaded — s3://%s/%s", settings.S3_BUCKET, result_s3_key)

        # result_image_path stores the S3 key (or local path in local mode)
        result_image_path = storage.url(result_s3_key)

        # ── 4. SSIM score ─────────────────────────────────────────────────
        score = _compute_ssim(local_person, output_path)
        logger.info("tasks: SSIM score for job %s: %.4f", job_id, score)

        # ── 5. Training pair auto-save ────────────────────────────────────
        saved = False
        if score >= settings.MIN_QUALITY_SCORE:
            _save_training_pair(job_id, local_person, local_garments[0], output_path, score)
            saved = True

        # ── 5b. Fit analysis (Phase 1/2, shadow mode) ─────────────────────
        # Never allowed to affect the result above — computed and stored
        # only. See _compute_fit_analysis().
        fit_analysis_json = _compute_fit_analysis(
            local_person, garment_size, height_cm, person_size_estimate,
            merchant, garment_sku,
        )

        # ── 6. Mark completed ─────────────────────────────────────────────
        _update_job(job_id, {
            "status":               "completed",
            "result_image_path":    result_image_path,
            "quality_score":        score,
            "saved_as_training":    saved,
            "person_size_estimate": person_size_estimate,
            "fit_analysis_json":    fit_analysis_json,
        })

    except Exception as exc:
        _update_job(job_id, {"status": "failed", "error_message": str(exc)})
        raise self.retry(exc=exc, countdown=30)

    finally:
        # Clean up temp files created for S3-backend input downloads
        for p in tmp_files:
            try:
                Path(p).unlink(missing_ok=True)
            except Exception:
                pass


# ── Accessory (wrist/glasses) Celery task ───────────────────────────────────
#
# Deliberately separate from process_tryon_job above rather than a shared
# task with branching — accessory jobs skip SSIM quality scoring,
# training-pair auto-save, and fit_analysis entirely: those are all
# diffusion-specific concerns (comparing a generative re-render against the
# original, or garment-size-chart fit estimation) that don't apply to a
# deterministic landmark-overlay composite. See app/services/accessory_engine.py.

@celery_app.task(bind=True, name="tasks.process_accessory_job", max_retries=2)
def process_accessory_job(self, job_id: str, person_image_path: str,
                           accessory_image_path: str, accessory_type: str):
    import logging
    from app.services.accessory_engine import ACCESSORY_ENGINES, LandmarkNotDetectedError

    logger = logging.getLogger(__name__)
    tmp_files: list[str] = []

    try:
        _update_job(job_id, {"status": "processing"})

        tmp_dir = Path(settings.LOCAL_STORAGE_PATH) / "tmp"
        tmp_dir.mkdir(parents=True, exist_ok=True)

        local_person = _resolve_local_path(person_image_path, job_id, "person")
        local_accessory = _resolve_local_path(accessory_image_path, job_id, "accessory")
        for p in (local_person, local_accessory):
            if str(tmp_dir) in p:
                tmp_files.append(p)

        output_path = str(tmp_dir / f"{job_id}_output.jpg")
        logger.info("tasks: Starting accessory inference for job %s (accessory_type=%s)...",
                    job_id, accessory_type)

        engine_factory = ACCESSORY_ENGINES.get(accessory_type)
        if engine_factory is None:
            raise ValueError(f"No engine registered for accessory_type={accessory_type!r}.")
        engine = engine_factory()
        engine.run(local_person, local_accessory, output_path)
        logger.info("tasks: Accessory inference complete for job %s.", job_id)

        result_s3_key = f"{settings.S3_PREFIX_OUTPUT}/{job_id}.jpg"
        storage = get_storage()
        storage.save(output_path, result_s3_key)
        result_image_path = storage.url(result_s3_key)

        _update_job(job_id, {
            "status":            "completed",
            "result_image_path": result_image_path,
        })

    except LandmarkNotDetectedError as exc:
        # Not transient — retrying won't find a wrist that isn't in the
        # photo. Fail immediately instead of burning 2 retries/60s.
        _update_job(job_id, {"status": "failed", "error_message": str(exc)})

    except Exception as exc:
        _update_job(job_id, {"status": "failed", "error_message": str(exc)})
        raise self.retry(exc=exc, countdown=15)

    finally:
        for p in tmp_files:
            try:
                Path(p).unlink(missing_ok=True)
            except Exception:
                pass
