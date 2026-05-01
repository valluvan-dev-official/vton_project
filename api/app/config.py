from pydantic_settings import BaseSettings
from functools import lru_cache


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

    # Quality threshold — results with SSIM >= this auto-save as training pairs
    MIN_QUALITY_SCORE: float = 0.65
    TRAINING_PAIR_SSIM_THRESHOLD: float = 0.65

    # API limits
    MAX_UPLOAD_SIZE_MB: int = 10

    class Config:
        env_file = ".env"
        extra = "ignore"


@lru_cache
def get_settings() -> Settings:
    return Settings()
