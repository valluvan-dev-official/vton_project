"""
SageMaker Async Inference client — replaces the Kaggle notebook backend.

Flow:
  1. Encode person + garment images as base64 JSON, upload to S3 (input).
  2. Call sagemaker-runtime.invoke_endpoint_async — returns an OutputLocation
     (and FailureLocation) in S3 before the job has even started running.
  3. Poll S3 for either object to appear (the endpoint auto-scales from
     zero if idle, so the first poll may need to wait for a cold start).
  4. Download the result JPEG (or read the failure JSON and raise).

This mirrors the shape of the old `_run_kaggle` / `_trigger_kaggle_notebook`
/ `_poll_kaggle_notebook` / `_download_kaggle_result` flow in
inference.py, but talks to a real managed AWS endpoint instead of pushing
notebooks to Kaggle.
"""
import base64
import json
import logging
import time
import uuid
from pathlib import Path
from urllib.parse import urlparse

import boto3
from botocore.exceptions import ClientError

from app.config import get_settings

logger = logging.getLogger(__name__)


def _split_s3_uri(uri: str) -> tuple[str, str]:
    parsed = urlparse(uri)
    return parsed.netloc, parsed.path.lstrip("/")


class SageMakerAsyncClient:
    """Thin wrapper around invoke_endpoint_async + S3 polling."""

    def __init__(self):
        settings = get_settings()
        self.endpoint_name = settings.SAGEMAKER_ENDPOINT_NAME
        self.region = settings.SAGEMAKER_REGION
        self.input_bucket = settings.SAGEMAKER_S3_BUCKET
        self.input_prefix = settings.SAGEMAKER_ASYNC_INPUT_PREFIX.rstrip("/")
        self.poll_interval_seconds = settings.SAGEMAKER_POLL_INTERVAL_SECONDS
        self.poll_timeout_seconds = settings.SAGEMAKER_POLL_TIMEOUT_SECONDS

        self._runtime = boto3.client("sagemaker-runtime", region_name=self.region)
        self._s3 = boto3.client("s3", region_name=self.region)

    # ── Submit ───────────────────────────────────────────────────────────────

    def _upload_input(self, job_id: str, person_path: str, garment_path: str) -> str:
        person_b64 = base64.b64encode(Path(person_path).read_bytes()).decode("utf-8")
        garment_b64 = base64.b64encode(Path(garment_path).read_bytes()).decode("utf-8")
        payload = json.dumps({
            "person": person_b64,
            "garment": garment_b64,
            "job_id": job_id,
        })

        key = f"{self.input_prefix}/{job_id}.json"
        self._s3.put_object(
            Bucket=self.input_bucket,
            Key=key,
            Body=payload.encode("utf-8"),
            ContentType="application/json",
        )
        input_uri = f"s3://{self.input_bucket}/{key}"
        logger.info(f"[SageMaker] Uploaded input to {input_uri}")
        return input_uri

    def _invoke(self, input_s3_uri: str, job_id: str) -> dict:
        resp = self._runtime.invoke_endpoint_async(
            EndpointName=self.endpoint_name,
            InputLocation=input_s3_uri,
            ContentType="application/json",
            Accept="image/jpeg",
            InvocationTimeoutSeconds=900,
            InferenceId=job_id,
        )
        logger.info(
            f"[SageMaker] Invoked async endpoint job_id={job_id} "
            f"output={resp.get('OutputLocation')} failure={resp.get('FailureLocation')}"
        )
        return resp

    # ── Poll + download ──────────────────────────────────────────────────────

    def _wait_for_result(self, output_uri: str, failure_uri: str | None) -> bytes:
        out_bucket, out_key = _split_s3_uri(output_uri)
        fail_bucket, fail_key = (_split_s3_uri(failure_uri) if failure_uri else (None, None))

        deadline = time.time() + self.poll_timeout_seconds
        logger.info(f"[SageMaker] Polling for result at {output_uri} (timeout={self.poll_timeout_seconds}s)")

        while time.time() < deadline:
            try:
                obj = self._s3.get_object(Bucket=out_bucket, Key=out_key)
                return obj["Body"].read()
            except ClientError as exc:
                if exc.response.get("Error", {}).get("Code") not in ("NoSuchKey", "404"):
                    raise

            if fail_bucket:
                try:
                    obj = self._s3.get_object(Bucket=fail_bucket, Key=fail_key)
                    error_body = obj["Body"].read().decode("utf-8", errors="replace")
                    raise RuntimeError(f"SageMaker async inference failed: {error_body[:500]}")
                except ClientError as exc:
                    if exc.response.get("Error", {}).get("Code") not in ("NoSuchKey", "404"):
                        raise

            time.sleep(self.poll_interval_seconds)

        raise TimeoutError(f"SageMaker async result not available after {self.poll_timeout_seconds}s")

    # ── Public API ───────────────────────────────────────────────────────────

    def run(self, person_path: str, garment_path: str, output_path: str, job_id: str = "") -> str:
        """Submit a try-on job and block until the result image is downloaded."""
        if not job_id:
            job_id = str(uuid.uuid4())

        input_uri = self._upload_input(job_id, person_path, garment_path)
        resp = self._invoke(input_uri, job_id)

        result_bytes = self._wait_for_result(resp["OutputLocation"], resp.get("FailureLocation"))

        Path(output_path).parent.mkdir(parents=True, exist_ok=True)
        Path(output_path).write_bytes(result_bytes)
        logger.info(f"[SageMaker] Result saved to {output_path}")
        return output_path


_client: SageMakerAsyncClient | None = None


def get_sagemaker_client() -> SageMakerAsyncClient:
    global _client
    if _client is None:
        _client = SageMakerAsyncClient()
    return _client
