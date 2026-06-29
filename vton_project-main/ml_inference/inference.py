"""
inference.py — SageMaker inference entry point (BYOC / PyTorch toolkit).

Implements the four functions the SageMaker PyTorch inference toolkit
(TorchServe) calls:

    model_fn(model_dir)                  -> load all models ONCE at startup
    input_fn(request_body, content_type) -> parse person + garment images
    predict_fn(input_data, model)        -> run the try-on pipeline
    output_fn(prediction, accept)        -> serialize the result image

Key guarantees required by the task:
  * All AI models are loaded a single time in model_fn (process lifetime).
  * Checkpoints are NEVER reloaded per request.
  * Models stay resident in GPU memory between requests.

Request formats accepted by input_fn:
  * multipart/form-data  with fields `person_image` and `garment_image`
  * application/json      {"person": "<b64>", "garment": "<b64>", "job_id": "..."}
  * application/x-npy / octet-stream are NOT supported (images only)

Response (output_fn):
  * accept image/jpeg          -> raw JPEG bytes
  * accept application/json    -> {"image": "<b64-jpeg>", "job_id": "..."}
"""
from __future__ import annotations

import base64
import io
import json
import logging
import os

from PIL import Image

# Support both `python -m ml_inference.inference` (package import) and the
# SageMaker toolkit loading this file as a top-level module.
try:
    from .predictor import VTONPredictor
except ImportError:  # pragma: no cover - toolkit loads as top-level module
    from predictor import VTONPredictor

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

JPEG_CONTENT_TYPE = "image/jpeg"
JSON_CONTENT_TYPE = "application/json"

# Where the third-party repos are baked into the image (see Dockerfile).
# Falls back to a runtime clone inside the workspace if unset.
_REPOS_DIR = os.getenv("VTON_REPOS_DIR") or None
_WORKSPACE = os.getenv("VTON_WORKSPACE", "/tmp/vton_workspace")


# ── 1. model_fn — load once ──────────────────────────────────────────────────

def model_fn(model_dir: str) -> VTONPredictor:
    """Called once by SageMaker at container startup.

    `model_dir` is where the model.tar.gz was extracted (/opt/ml/model),
    containing viton512.ckpt, warp_viton.pth and (optionally)
    densepose_rcnn_R_50_FPN_s1x.pkl.
    """
    import torch
    device = "cuda" if torch.cuda.is_available() else "cpu"
    logger.info("model_fn: loading DCI-VTON models from %s on %s", model_dir, device)
    predictor = VTONPredictor.from_pretrained(
        weights_dir=model_dir,
        device=device,
        workspace=_WORKSPACE,
        repos_dir=_REPOS_DIR,
    )
    logger.info("model_fn: model ready.")
    return predictor


# ── 2. input_fn — parse request ──────────────────────────────────────────────

def _decode_image(data: bytes) -> Image.Image:
    return Image.open(io.BytesIO(data)).convert("RGB")


def input_fn(request_body, content_type: str = JSON_CONTENT_TYPE) -> dict:
    """Parse the incoming request into {person, garment, job_id}."""
    content_type = (content_type or "").lower()
    logger.info("input_fn: content_type=%s", content_type)

    if content_type.startswith("application/json"):
        if isinstance(request_body, (bytes, bytearray)):
            request_body = request_body.decode("utf-8")
        payload = json.loads(request_body)
        person_b64 = payload.get("person") or payload.get("person_image")
        garment_b64 = payload.get("garment") or payload.get("garment_image")
        if not person_b64 or not garment_b64:
            raise ValueError("JSON request must include 'person' and 'garment' base64 fields.")
        return {
            "person": _decode_image(base64.b64decode(person_b64)),
            "garment": _decode_image(base64.b64decode(garment_b64)),
            "job_id": str(payload.get("job_id", "")),
        }

    if content_type.startswith("multipart/form-data"):
        # Parse multipart without extra deps using the email parser.
        from requests_toolbelt.multipart import decoder as _decoder  # type: ignore
        multipart = _decoder.MultipartDecoder(
            request_body if isinstance(request_body, (bytes, bytearray)) else request_body.encode(),
            content_type,
        )
        parts: dict[str, bytes] = {}
        for part in multipart.parts:
            disp = part.headers.get(b"Content-Disposition", b"").decode()
            name = None
            for token in disp.split(";"):
                token = token.strip()
                if token.startswith("name="):
                    name = token.split("=", 1)[1].strip('"')
            if name:
                parts[name] = part.content
        person = parts.get("person_image") or parts.get("person")
        garment = parts.get("garment_image") or parts.get("garment")
        if not person or not garment:
            raise ValueError("multipart request must include 'person_image' and 'garment_image'.")
        return {
            "person": _decode_image(person),
            "garment": _decode_image(garment),
            "job_id": "",
        }

    raise ValueError(f"Unsupported content type: {content_type}")


# ── 3. predict_fn — run inference ────────────────────────────────────────────

def predict_fn(input_data: dict, model: VTONPredictor) -> Image.Image:
    """Run the full try-on pipeline. `model` is the VTONPredictor from model_fn."""
    logger.info("predict_fn: running inference job_id=%s", input_data.get("job_id"))
    return model.predict(
        input_data["person"],
        input_data["garment"],
        job_id=input_data.get("job_id", ""),
    )


# ── 4. output_fn — serialize result ──────────────────────────────────────────

def output_fn(prediction: Image.Image, accept: str = JPEG_CONTENT_TYPE):
    """Serialize the result PIL image to the requested accept type."""
    accept = (accept or JPEG_CONTENT_TYPE).lower()
    buf = io.BytesIO()
    prediction.save(buf, format="JPEG", quality=95)
    jpeg_bytes = buf.getvalue()

    if accept.startswith("application/json"):
        body = json.dumps({"image": base64.b64encode(jpeg_bytes).decode("utf-8")})
        return body, JSON_CONTENT_TYPE

    # Default: raw JPEG bytes
    return jpeg_bytes, JPEG_CONTENT_TYPE
