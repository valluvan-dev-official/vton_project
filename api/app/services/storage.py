"""
Storage abstraction layer.

Switch between local and S3 by setting STORAGE_BACKEND in .env — no code changes needed.

S3 backend uses the EC2 IAM instance profile automatically (boto3 default
credential chain).  No AWS_ACCESS_KEY_ID or AWS_SECRET_ACCESS_KEY required
or used when running on EC2.
"""
import logging
import os
import shutil
from abc import ABC, abstractmethod
from pathlib import Path

from app.config import get_settings

settings = get_settings()
logger = logging.getLogger(__name__)


class StorageBackend(ABC):
    @abstractmethod
    def save(self, source_path: str, dest_key: str) -> str:
        """Save file from source_path to storage. Returns the stored path/URL."""

    @abstractmethod
    def load(self, key: str, dest_path: str) -> str:
        """Copy file from storage key to dest_path. Returns dest_path."""

    @abstractmethod
    def exists(self, key: str) -> bool:
        """Return True if the key exists in storage."""

    @abstractmethod
    def delete(self, key: str) -> None:
        """Delete a file from storage."""

    @abstractmethod
    def url(self, key: str) -> str:
        """Return a URL or absolute path that can be used to serve the file."""


class LocalStorageBackend(StorageBackend):
    """Stores files on the local filesystem under LOCAL_STORAGE_PATH."""

    def __init__(self, base_path: str | None = None):
        self.base = Path(base_path or settings.LOCAL_STORAGE_PATH)
        self.base.mkdir(parents=True, exist_ok=True)

    def _full(self, key: str) -> Path:
        return self.base / key

    def save(self, source_path: str, dest_key: str) -> str:
        dest = self._full(dest_key)
        dest.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(source_path, dest)
        return str(dest)

    def load(self, key: str, dest_path: str) -> str:
        shutil.copy2(self._full(key), dest_path)
        return dest_path

    def exists(self, key: str) -> bool:
        return self._full(key).exists()

    def delete(self, key: str) -> None:
        p = self._full(key)
        if p.exists():
            p.unlink()

    def url(self, key: str) -> str:
        return self._full(key).as_posix()


class S3StorageBackend(StorageBackend):
    """Stores files in AWS S3.

    Authentication uses the EC2 IAM instance profile automatically via
    boto3's default credential chain.  On EC2, no explicit credentials are
    needed or used.  For local development you can set AWS_PROFILE or
    AWS_ACCESS_KEY_ID / AWS_SECRET_ACCESS_KEY in the environment, but those
    are never required in production.
    """

    def __init__(self):
        self.bucket = settings.S3_BUCKET
        self.region = settings.S3_REGION
        self._s3 = None  # lazy-initialised — see _client property

    @property
    def _client(self):
        """Return a cached boto3 S3 client, creating it on first use."""
        if self._s3 is None:
            import boto3  # imported lazily so local backend doesn't need boto3
            # Do NOT pass aws_access_key_id / aws_secret_access_key.
            # boto3 automatically uses the EC2 instance profile when running on EC2.
            self._s3 = boto3.client("s3", region_name=self.region)
        return self._s3

    def save(self, source_path: str, dest_key: str) -> str:
        logger.info("storage: Uploading %s → s3://%s/%s", source_path, self.bucket, dest_key)
        self._client.upload_file(source_path, self.bucket, dest_key)
        logger.info("storage: Upload complete — s3://%s/%s", self.bucket, dest_key)
        return dest_key

    def load(self, key: str, dest_path: str) -> str:
        logger.info("storage: Downloading s3://%s/%s → %s", self.bucket, key, dest_path)
        Path(dest_path).parent.mkdir(parents=True, exist_ok=True)
        self._client.download_file(self.bucket, key, dest_path)
        return dest_path

    def exists(self, key: str) -> bool:
        from botocore.exceptions import ClientError
        try:
            self._client.head_object(Bucket=self.bucket, Key=key)
            return True
        except ClientError:
            return False

    def delete(self, key: str) -> None:
        self._client.delete_object(Bucket=self.bucket, Key=key)

    def url(self, key: str) -> str:
        """Return the public HTTPS URL for the object.

        For private buckets call generate_presigned_url() instead.
        """
        return f"https://{self.bucket}.s3.{self.region}.amazonaws.com/{key}"

    def generate_presigned_url(self, key: str, expires_in: int = 3600) -> str:
        """Return a pre-signed URL valid for `expires_in` seconds."""
        return self._client.generate_presigned_url(
            "get_object",
            Params={"Bucket": self.bucket, "Key": key},
            ExpiresIn=expires_in,
        )

    def upload_bytes(self, data: bytes, dest_key: str,
                     content_type: str = "image/jpeg") -> str:
        """Upload raw bytes directly without writing to disk first."""
        import io
        logger.info(
            "storage: Uploading %d bytes → s3://%s/%s",
            len(data), self.bucket, dest_key,
        )
        self._client.upload_fileobj(
            io.BytesIO(data),
            self.bucket,
            dest_key,
            ExtraArgs={"ContentType": content_type},
        )
        return dest_key


# ── Singleton factory ────────────────────────────────────────────────────────

_storage_instance: StorageBackend | None = None


def get_storage() -> StorageBackend:
    """
    Factory — returns the configured backend singleton.
    Change STORAGE_BACKEND in .env to switch; no code change required.
    """
    global _storage_instance
    if _storage_instance is None:
        backend = settings.STORAGE_BACKEND.lower()
        if backend == "s3":
            _storage_instance = S3StorageBackend()
        else:
            _storage_instance = LocalStorageBackend()
    return _storage_instance


def get_s3_client():
    """Return a raw boto3 S3 client using the EC2 IAM role.

    Convenience helper for code that needs direct S3 access outside the
    StorageBackend abstraction (e.g. model_bootstrap).
    """
    import boto3
    return boto3.client("s3", region_name=settings.S3_REGION)
