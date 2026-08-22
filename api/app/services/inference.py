"""
Inference router — placeholder | local GPU (IDM-VTON) | own trained model | SageMaker

Selection priority (decided once at worker startup, in InferenceRouter.__init__):
  1. own model   — if OWN_MODEL_CHECKPOINT is set and the file exists
  2. sagemaker   — disabled for GPU-EC2 deployment (commented out below)
  3. local_gpu   — if DEVICE=cuda and WEIGHTS_DIR exists (IDM-VTON)
  4. placeholder — fallback when nothing above is configured

This is the intended single entry point for try-on inference — tasks.py
calls get_inference_router().run(...) rather than talking to any specific
backend directly, so switching models is a config change (OWN_MODEL_CHECKPOINT
in .env), not a code change.
"""
import logging
import time
from pathlib import Path
from PIL import Image, ImageDraw, ImageFont

logger = logging.getLogger(__name__)

# Read from settings (pydantic reads .env file) — not os.getenv, which misses
# values set only in .env when the process env doesn't also have them.
from app.config import get_settings as _get_settings
_s = _get_settings()

# Where ml/src lives relative to this file (api/app/services/inference.py ->
# vton_project/ml). In the GPU worker container ml/ is bind-mounted to
# /app/ml (see docker-compose.gpu.yml), so that path is tried first.
_ML_ROOT_CANDIDATES = [
    Path("/app/ml"),
    Path(__file__).resolve().parents[3] / "ml",
]


def _ensure_ml_src_on_path() -> None:
    """Make `import src.<...>` (ml/src/...) resolve, regardless of whether
    we're running inside the Docker worker or a local dev checkout."""
    import sys
    for candidate in _ML_ROOT_CANDIDATES:
        if candidate.is_dir():
            p = str(candidate)
            if p not in sys.path:
                sys.path.insert(0, p)
            return
    raise RuntimeError(
        f"inference: could not locate the ml/ directory (tried {_ML_ROOT_CANDIDATES})."
    )


class InferenceRouter:
    def __init__(self):
        self.use_own_model: bool = False
        self._model = None
        self._sagemaker_client = None
        self.last_person_size_estimate = "M"

        device = (_s.DEVICE or "cpu").strip().lower()
        weights_dir = (_s.WEIGHTS_DIR or "").strip()

        # [SAGEMAKER] Disabled for GPU-EC2 deployment.
        # To re-enable SageMaker, restore SAGEMAKER_ENDPOINT_NAME / SAGEMAKER_S3_BUCKET
        # in config.py and .env, then remove the leading # from the block below.
        # if _s.SAGEMAKER_ENDPOINT_NAME and _s.SAGEMAKER_S3_BUCKET:
        #     self._mode = "sagemaker"
        #     logger.info("InferenceRouter: SageMaker Async Inference mode active "
        #                 f"(endpoint={_s.SAGEMAKER_ENDPOINT_NAME}).")
        # elif device == "cuda" and weights_dir and Path(weights_dir).exists():
        if device == "cuda" and weights_dir and Path(weights_dir).exists():
            self._mode = "local_gpu"
            logger.info(
                "InferenceRouter: Local GPU (IDM-VTON) mode selected — "
                "engine loads lazily via gpu_inference_service on first job."
            )
        else:
            self._mode = "placeholder"
            logger.info(
                "InferenceRouter: Placeholder mode "
                "(set DEVICE=cuda + WEIGHTS_DIR to enable local GPU inference)."
            )

        # Own model (Phase 4) — auto-switch if a checkpoint is configured.
        # Takes priority over everything above once loaded (see run()).
        own_ckpt = (_s.OWN_MODEL_CHECKPOINT or "").strip()
        if own_ckpt and Path(own_ckpt).exists():
            try:
                self.switch_to_own_model(own_ckpt)
            except Exception as exc:
                logger.warning(f"InferenceRouter: own model load failed: {exc}. Falling back.")

    # ── Preprocessing ──────────────────────────────────────────────────────────

    def _preprocess_garment(self, garment_path: str) -> str:
        """Remove garment background → white bg. Falls back to original on failure.

        This matters beyond cosmetics: GPUInferenceEngine's sleeve-type
        detection and best-image scoring both assume a near-white background
        to separate garment pixels from background, so every garment image
        handed to any backend goes through this first.
        """
        try:
            from rembg import remove as rembg_remove
            img = Image.open(garment_path).convert("RGBA")
            result = rembg_remove(img)
            white_bg = Image.new("RGBA", result.size, (255, 255, 255, 255))
            white_bg.paste(result, mask=result.split()[3])
            out = white_bg.convert("RGB")
            clean_path = str(Path(garment_path).with_suffix("")) + "_clean.jpg"
            out.save(clean_path, "JPEG", quality=95)
            logger.info(f"[Preprocess] Garment bg removed → {clean_path}")
            return clean_path
        except Exception as exc:
            logger.warning(f"[Preprocess] rembg failed ({exc}), using original garment.")
            return garment_path

    # ── Public API ─────────────────────────────────────────────────────────────

    def run(self, person_image_path: str, garment_image_paths,
            output_path: str, job_id: str = "", garment_size: str = "M",
            category: str = "upper_body", dress_subtype: str | None = None) -> str:
        """garment_image_paths: a single path, or a list of paths (multiple
        angles of the same garment — only the local_gpu/IDM-VTON backend
        actually makes use of more than one; other backends use the first).

        category: "upper_body" | "lower_body" | "dresses" — only the
        local_gpu/IDM-VTON backend acts on this (threaded into
        get_mask_location()); other backends (own model, placeholder,
        SageMaker) ignore it, same as before this param existed.

        dress_subtype: "saree" | "salwar_suit" | None — only meaningful
        when category == "dresses"; only the local_gpu/IDM-VTON backend
        acts on it (selects a dedicated sleeve-detection module — see
        GPUInferenceEngine.run()'s docstring). Other backends ignore it.

        Returns output_path. The auto-detected person body-size bucket (when
        available) is left on self.last_person_size_estimate for the caller
        to read — kept off the return value so this stays a drop-in
        replacement for any backend that doesn't estimate it (own model,
        placeholder, SageMaker all report "M" — unknown/neutral).
        """
        if isinstance(garment_image_paths, str):
            garment_image_paths = [garment_image_paths]
        garment_size = (garment_size or "M").strip().upper()
        category = (category or "upper_body").strip().lower()
        dress_subtype = (dress_subtype or "").strip().lower() or None
        job_id = job_id or Path(output_path).stem

        clean_garment_paths = [self._preprocess_garment(p) for p in garment_image_paths]
        self.last_person_size_estimate = "M"

        if self.use_own_model and self._model:
            try:
                return self._run_own_model(person_image_path, clean_garment_paths[0], output_path)
            except Exception as exc:
                logger.warning(f"Own-model inference failed: {exc}. Falling back.")

        if self._mode == "sagemaker":
            try:
                return self._run_sagemaker(person_image_path, clean_garment_paths[0], output_path)
            except Exception as exc:
                logger.warning(f"SageMaker inference failed: {exc}. Falling back to placeholder.")

        if self._mode == "local_gpu":
            try:
                from app.services.gpu_inference_service import run as _gpu_run
                self.last_person_size_estimate = _gpu_run(
                    person_image_path, clean_garment_paths, output_path,
                    job_id=job_id, garment_size=garment_size, category=category,
                    dress_subtype=dress_subtype,
                )
                return output_path
            except Exception as exc:
                logger.exception(
                    f"Local GPU inference failed: {exc}. Falling back to placeholder."
                )

        return self._run_placeholder(person_image_path, clean_garment_paths[0], output_path)

    def switch_to_own_model(self, model_path: str) -> None:
        self._model = self._load_model(model_path)
        self.use_own_model = True
        logger.info(f"InferenceRouter: switched to own model checkpoint: {model_path}")

    def switch_to_idm_vton(self) -> None:
        """Undo switch_to_own_model() — go back to IDM-VTON / placeholder
        (whichever local_gpu detection picked at startup) without a restart."""
        self.use_own_model = False
        self._model = None
        logger.info("InferenceRouter: switched back to IDM-VTON / placeholder.")

    # ── Placeholder ─────────────────────────────────────────────────────────────

    def _run_placeholder(self, person_path: str, garment_path: str, output_path: str) -> str:
        time.sleep(3)
        person_img = Image.open(person_path).convert("RGB").resize((512, 512))
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

    # ── SageMaker Async Inference ────────────────────────────────────────────────

    def _run_sagemaker(self, person_path: str, garment_path: str, output_path: str) -> str:
        from app.services.sagemaker_client import get_sagemaker_client
        job_id = Path(output_path).stem
        logger.info(f"[SageMaker] Starting inference for job {job_id}")
        client = get_sagemaker_client()
        result_path = client.run(person_path, garment_path, output_path, job_id=job_id)
        logger.info(f"[SageMaker] Inference complete for job {job_id}")
        return result_path

    # ── Own model (VTONPipeline, trained by ml/src/training/train.py) ───────────

    def _load_model(self, model_path: str):
        _ensure_ml_src_on_path()
        from src.inference.infer import VTONInference
        device = (_s.DEVICE or "cpu").strip()
        logger.info(f"InferenceRouter: loading own-model checkpoint {model_path} (device={device})...")
        return VTONInference(checkpoint_path=model_path, device=device)

    def _run_own_model(self, person_path: str, garment_path: str, output_path: str) -> str:
        return self._model.run(person_path, garment_path, output_path)


_router: InferenceRouter | None = None


def get_inference_router() -> InferenceRouter:
    global _router
    if _router is None:
        _router = InferenceRouter()
    return _router
