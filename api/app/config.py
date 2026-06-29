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
    STORAGE_BACKEND: str = "local"          # "local" | "s3"
    LOCAL_STORAGE_PATH: str = "/app/storage"

    # S3 (only needed when STORAGE_BACKEND=s3)
    S3_BUCKET: str = ""
    S3_REGION: str = "us-east-1"
    AWS_ACCESS_KEY_ID: str = ""
    AWS_SECRET_ACCESS_KEY: str = ""

    # ML inference
    USE_OWN_MODEL: bool = False
    MODEL_PATH: str = ""
    DEVICE: str = "cpu"
    WEIGHTS_DIR: str = ""

    # Quality threshold — results with SSIM >= this auto-save as training pairs
    MIN_QUALITY_SCORE: float = 0.65
    TRAINING_PAIR_SSIM_THRESHOLD: float = 0.65

    # API limits
    MAX_UPLOAD_SIZE_MB: int = 10

    # SageMaker Async Inference (DCI-VTON)
    SAGEMAKER_ENDPOINT_NAME: str = ""
    SAGEMAKER_REGION: str = "us-east-1"
    SAGEMAKER_S3_BUCKET: str = ""
    SAGEMAKER_ASYNC_INPUT_PREFIX: str = "dci-vton/async-input"
    SAGEMAKER_POLL_INTERVAL_SECONDS: int = 5
    SAGEMAKER_POLL_TIMEOUT_SECONDS: int = 900

    class Config:
        env_file = str(_ENV_FILE)
        extra = "ignore"


@lru_cache
def get_settings() -> Settings:
    return Settings()
