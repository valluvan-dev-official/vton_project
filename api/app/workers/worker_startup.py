#!/usr/bin/env python3
"""
worker_startup.py — Celery worker initialisation hook.

Registers a ``worker_ready`` signal handler that:
  1. Runs the model bootstrap (downloads weights from S3 if missing).
  2. Initialises the GPUInferenceEngine singleton (loads all models into GPU RAM).

This means models are loaded ONCE at worker startup rather than on the first
request, so the first inference call has no extra latency and the worker log
clearly shows when it is ready to serve traffic.

Usage (in docker-compose.gpu.yml):
  command: celery -A app.workers.tasks.celery_app worker
           --loglevel=info --pool=solo
           -I app.workers.worker_startup
"""
import logging

from celery.signals import worker_ready

logger = logging.getLogger(__name__)


@worker_ready.connect
def on_worker_ready(sender, **kwargs):
    """Called by Celery immediately after the worker process has started."""
    logger.info("=" * 60)
    logger.info("worker_startup: Worker process ready — initialising GPU engine...")
    logger.info("=" * 60)

    try:
        import app.services.gpu_inference_service as _gpu_svc
        # Calling get_gpu_engine() triggers:
        #   1. CUDA availability check
        #   2. Model bootstrap (S3 download if weights missing)
        #   3. GPUInferenceEngine construction (loads all checkpoints)
        _gpu_svc.get_gpu_engine()
        logger.info("=" * 60)
        logger.info("worker_startup: GPU models loaded. Ready for inference.")
        logger.info("=" * 60)
    except Exception as exc:
        logger.error("=" * 60)
        logger.error("worker_startup: FATAL — GPU engine initialisation failed: %s", exc)
        logger.error("worker_startup: Worker will exit to prevent silent failures.")
        logger.error("=" * 60)
        # Re-raise so Celery/Docker marks the container as failed and restarts it.
        raise SystemExit(1) from exc
