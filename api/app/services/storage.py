"""
Storage abstraction layer.
Switch between local and S3 by setting STORAGE_BACKEND in .env — no code changes needed.
"""
import os
import shutil
from abc import ABC, abstractmethod
from pathlib import Path

from app.config import get_settings

settings = get_settings()


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
    """Stores files in AWS S3. Requires boto3 and S3_* env vars."""

    def __init__(self):
        import boto3  # imported lazily so local backend doesn't need boto3
        self.bucket = settings.S3_BUCKET
        self.s3 = boto3.client(
            "s3",
            region_name=settings.S3_REGION,
            aws_access_key_id=settings.AWS_ACCESS_KEY_ID,
            aws_secret_access_key=settings.AWS_SECRET_ACCESS_KEY,
        )

    def save(self, source_path: str, dest_key: str) -> str:
        self.s3.upload_file(source_path, self.bucket, dest_key)
        return dest_key

    def load(self, key: str, dest_path: str) -> str:
        self.s3.download_file(self.bucket, key, dest_path)
        return dest_path

    def exists(self, key: str) -> bool:
        from botocore.exceptions import ClientError
        try:
            self.s3.head_object(Bucket=self.bucket, Key=key)
            return True
        except ClientError:
            return False

    def delete(self, key: str) -> None:
        self.s3.delete_object(Bucket=self.bucket, Key=key)

    def url(self, key: str) -> str:
        return f"https://{self.bucket}.s3.{settings.S3_REGION}.amazonaws.com/{key}"


_storage_instance: StorageBackend | None = None


def get_storage() -> StorageBackend:
    """
    Factory — returns the configured backend.
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
