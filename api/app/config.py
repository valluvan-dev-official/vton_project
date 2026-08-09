from pydantic_settings import BaseSettings
from functools import lru_cache
from pathlib import Path

# Resolve .env relative to this file so it works regardless of CWD
_ENV_FILE = Path(__file__).parent.parent / ".env"


class Settings(BaseSettings):
    # Database
    DATABASE_URL: str = "postgresql+asyncpg://vton:vton@postgres:5432/vton"

    # Celery / Redis
    CELERY_BROKER_URL: str = "redis://redis:6379/0"
    CELERY_RESULT_BACKEND: str = "redis://redis:6379/0"

    # Storage
    STORAGE_BACKEND: str = "s3"             # "local" | "s3"
    LOCAL_STORAGE_PATH: str = "/app/storage"

    # S3
    # On EC2 the IAM instance profile is used automatically by boto3.
    # AWS_ACCESS_KEY_ID / AWS_SECRET_ACCESS_KEY are NOT required and should
    # NOT be set in production.  They are accepted here so local development
    # against a real bucket still works when credentials are in the environment.
    S3_BUCKET: str = "amazon-sagemaker-960583974175-ap-south-1-ban3nm5kd4wvi9"
    S3_REGION: str = "ap-south-1"
    AWS_ACCESS_KEY_ID: str = ""      # leave blank on EC2 (uses instance profile)
    AWS_SECRET_ACCESS_KEY: str = ""  # leave blank on EC2 (uses instance profile)

    # S3 folder prefixes (no trailing slash)
    S3_PREFIX_INPUT_PERSON:  str = "input/person"
    S3_PREFIX_INPUT_GARMENT: str = "input/garment"
    S3_PREFIX_OUTPUT:        str = "output"
    S3_PREFIX_TRAINING:      str = "shared/training_pairs"

    # ── GPU EC2 local inference ──────────────────────────────────────────────
    DEVICE:      str = "cuda"                   # "cuda" on GPU instance; "cpu" for testing
    WEIGHTS_DIR: str = "/app/ml/weights"        # viton512.ckpt + warp_viton.pth live here
    WORKSPACE:   str = "/tmp/vton_workspace"    # temp dir for repo clones / artefacts

    # ── Model bootstrap ──────────────────────────────────────────────────────
    # S3 key of the model archive downloaded when WEIGHTS_DIR is empty.
    # Set to "" to disable automatic download (weights must be present already).
    MODEL_S3_TAR: str = "model/model.tar.gz"

    # Retained for backward compatibility (not used in GPU-EC2 mode)
    USE_OWN_MODEL: bool = False
    MODEL_PATH:    str  = ""

    # ── Own trained model (Phase 4) ───────────────────────────────────────────
    # Path to a checkpoint saved by ml/src/training/train.py (VTONPipeline).
    # When set and the file exists, InferenceRouter switches to it instead of
    # IDM-VTON automatically on worker startup. Leave blank to keep IDM-VTON.
    OWN_MODEL_CHECKPOINT: str = ""

    # Quality threshold — results with SSIM >= this auto-save as training pairs
    MIN_QUALITY_SCORE:          float = 0.65
    TRAINING_PAIR_SSIM_THRESHOLD: float = 0.65

    # API limits
    MAX_UPLOAD_SIZE_MB: int = 10

    # ── Fit analysis (Phase 1, shadow mode) ──────────────────────────────────
    # Gates api/app/services/fit_analysis/garment_measurements.py's
    # illustrative DEFAULT_TSHIRT_SIZE_CHART. Defaults to False (safe) so a
    # production deployment never silently presents made-up garment
    # measurements as if they were real merchant/catalog sizing — with this
    # off (or no real catalog data available for a given size), fit_analysis
    # reports "insufficient_garment_data" instead of an apparent fit.
    # Set True only for local dev/test where no real catalog is wired up.
    FIT_ANALYSIS_ALLOW_DEFAULT_CATALOG: bool = False

    # ── SageMaker Async Inference (disabled for GPU-EC2 deployment) ──────────
    # To switch back to SageMaker, uncomment the fields below, populate .env,
    # and restore the [SAGEMAKER] lines in tasks.py and inference.py.
    #
    # SAGEMAKER_ENDPOINT_NAME: str = ""
    # SAGEMAKER_REGION: str = "us-east-1"
    # SAGEMAKER_S3_BUCKET: str = ""
    # SAGEMAKER_ASYNC_INPUT_PREFIX: str = "dci-vton/async-input"
    # SAGEMAKER_POLL_INTERVAL_SECONDS: int = 5
    # SAGEMAKER_POLL_TIMEOUT_SECONDS: int = 900

    class Config:
        env_file = str(_ENV_FILE)
        extra = "ignore"


@lru_cache
def get_settings() -> Settings:
    return Settings()
