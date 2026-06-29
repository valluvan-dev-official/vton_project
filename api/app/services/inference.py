"""
Inference router — Phase 1: placeholder | Phase 3: SageMaker DCI-VTON | Phase 4: own model

Flow (Phase 3, SageMaker):
  1. Upload person + garment images as a JSON payload to S3
  2. Invoke the SageMaker Async Inference endpoint (sagemaker-runtime.invoke_endpoint_async)
  3. Poll the S3 output location until the result (or failure) object appears
  4. Download result image
  5. Return output path

This replaces the previous Kaggle-notebook backend. No Kaggle dependency remains.
"""
import logging
import time
from pathlib import Path
from PIL import Image, ImageDraw, ImageFont

logger = logging.getLogger(__name__)

# Read from settings (pydantic reads .env file) — not os.getenv which misses .env on host
from app.config import get_settings as _get_settings
_s = _get_settings()


class InferenceRouter:
    def __init__(self):
        self.use_own_model: bool = False
        self._model = None
        self._gpu_engine = None
        self._sagemaker_client = None

        import os

        # Decide mode — priority: own model (Phase 4) > SageMaker > local GPU > placeholder
        self._mode = "placeholder"
        device = os.getenv("DEVICE", "cpu").strip().lower()
        weights_dir = os.getenv("WEIGHTS_DIR", "").strip()

        if _s.SAGEMAKER_ENDPOINT_NAME and _s.SAGEMAKER_S3_BUCKET:
            self._mode = "sagemaker"
            logger.info("InferenceRouter: SageMaker Async Inference mode active "
                        f"(endpoint={_s.SAGEMAKER_ENDPOINT_NAME}).")
        elif device == "cuda" and weights_dir and Path(weights_dir).exists():
            self._mode = "local_gpu"
            logger.info("InferenceRouter: Local GPU mode — loading models...")
            try:
                from ml.scripts.gpu_inference import GPUInferenceEngine
                self._gpu_engine = GPUInferenceEngine(weights_dir=weights_dir, device="cuda")
                logger.info("InferenceRouter: Local GPU mode active.")
            except Exception as exc:
                logger.warning(f"Local GPU init failed: {exc}. Falling back.")
                self._mode = "placeholder"
        else:
            logger.info("InferenceRouter: Placeholder mode "
                        "(set SAGEMAKER_ENDPOINT_NAME + SAGEMAKER_S3_BUCKET to enable DCI-VTON).")

        # Phase 4 — own model auto-load
        ckpt = os.getenv("OWN_MODEL_CHECKPOINT", "").strip()
        if ckpt and Path(ckpt).exists():
            try:
                self.switch_to_own_model(ckpt)
            except Exception as exc:
                logger.warning(f"Own model load failed: {exc}. Falling back.")

    # ── Public API ─────────────────────────────────────────────────────────────

    def run(self, person_image_path: str, garment_image_path: str, output_path: str) -> str:
        if self.use_own_model and self._model:
            return self._run_own_model(person_image_path, garment_image_path, output_path)
        if self._mode == "sagemaker":
            try:
                return self._run_sagemaker(person_image_path, garment_image_path, output_path)
            except Exception as exc:
                logger.warning(f"SageMaker inference failed: {exc}. Falling back to placeholder.")
        if self._mode == "local_gpu" and self._gpu_engine:
            try:
                job_id = Path(output_path).stem
                return self._gpu_engine.run(person_image_path, garment_image_path, output_path, job_id)
            except Exception as exc:
                logger.warning(f"Local GPU inference failed: {exc}. Falling back to placeholder.")
        return self._run_placeholder(person_image_path, garment_image_path, output_path)

    def switch_to_own_model(self, model_path: str) -> None:
        self._model = self._load_model(model_path)
        self.use_own_model = True
        logger.info(f"Switched to own model: {model_path}")

    # ── Phase 1: Placeholder ───────────────────────────────────────────────────

    def _run_placeholder(self, person_path: str, garment_path: str, output_path: str) -> str:
        time.sleep(3)
        person_img  = Image.open(person_path).convert("RGB").resize((512, 512))
        garment_img = Image.open(garment_path).convert("RGB").resize((256, 256))
        canvas = person_img.copy()
        canvas.paste(garment_img, (128, 100))
        draw = ImageDraw.Draw(canvas)
        draw.rectangle([(0, 0), (512, 40)], fill=(20, 20, 20))
        try:
            font = ImageFont.truetype("/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf", 22)
        except (IOError, OSError):
            font = ImageFont.load_default()
        draw.text((10, 10), "VTON PLACEHOLDER", fill="white", font=font)
        Path(output_path).parent.mkdir(parents=True, exist_ok=True)
        canvas.save(output_path, "JPEG", quality=90)
        return output_path

    # ── Phase 3: SageMaker Async Inference ──────────────────────────────────────

    def _run_sagemaker(self, person_path: str, garment_path: str, output_path: str) -> str:
        from app.services.sagemaker_client import get_sagemaker_client
        job_id = Path(output_path).stem
        logger.info(f"[SageMaker] Starting inference for job {job_id}")
        client = get_sagemaker_client()
        result_path = client.run(person_path, garment_path, output_path, job_id=job_id)
        logger.info(f"[SageMaker] Inference complete for job {job_id}")
        return result_path

    # ── Phase 4: Own Model ─────────────────────────────────────────────────────

    def _load_model(self, model_path: str):
        raise NotImplementedError(f"Wire up your trained model loader. checkpoint={model_path}")

    def _run_own_model(self, person_path: str, garment_path: str, output_path: str) -> str:
        raise NotImplementedError("Wire up your trained model inference here.")


_router: InferenceRouter | None = None


def get_inference_router() -> InferenceRouter:
    global _router
    if _router is None:
        _router = InferenceRouter()
    return _router
