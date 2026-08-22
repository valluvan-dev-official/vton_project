"""
Handbag "held in hand" GPU inference engine — SDXL inpainting + IP-Adapter.

Why this exists (contrast with api/app/services/accessory_engine.py's
HandbagOverlayEngine): the CPU overlay engine composites a flat product
cutout behind the hand — it can never show fingers wrapping in front of a
handle/strap, because it never touches the hand pixels themselves. Getting
that requires regenerating the grip region, which is a generative inpainting
problem (same category as garment try-on), not a compositing one — so this
engine lives here, next to gpu_inference.py, and runs as a GPU job.

Pipeline:
  1. MediaPipe Hands -> wrist/knuckle landmarks -> grip point + inpaint mask
     region (same landmark math as HandbagOverlayEngine, duplicated rather
     than imported since ml/ and api/app/services/ are separate deployable
     units — see docker-compose.gpu.yml, ml/ is bind-mounted into the GPU
     worker, api/app/services is baked into the api image).
  2. SDXL inpainting pipeline, conditioned on the bag's own product photo via
     IP-Adapter (same "condition generation on a reference image" trick
     IDM-VTON itself is built on) so the regenerated region actually looks
     like the specific product, not a generic bag.
  3. Only the masked region (grip + hanging-bag area below it) is
     regenerated; everything outside the mask is preserved from the
     original photo by the inpaint pipeline itself.

NOT yet empirically tuned against real photos/hardware — prompt wording,
mask sizing, and inference steps are a reasonable first pass, same caveat
every other first-phase module in this codebase carries (see
WristOverlayEngine's docstring for the same disclaimer pattern). Expect to
revisit once this runs against a real GPU with real test photos.
"""
import logging
import os

import numpy as np
import cv2
from PIL import Image, ImageDraw, ImageFilter

logger = logging.getLogger(__name__)


class HandDetectionError(Exception):
    """Raised when no hand/wrist landmarks can be found in the person photo."""


# SDXL inpainting checkpoint — a dedicated inpainting fine-tune of SDXL
# (not the base txt2img checkpoint, which isn't trained for the
# mask-conditioned denoising this needs). Downloaded once, cached under
# WEIGHTS_DIR/handbag_grip like the IDM-VTON weights are cached under
# WEIGHTS_DIR — see gpu_inference.py's weight layout.
SDXL_INPAINT_MODEL_ID = "diffusers/stable-diffusion-xl-1.0-inpainting-0.1"
IP_ADAPTER_REPO = "h94/IP-Adapter"
IP_ADAPTER_SUBFOLDER = "sdxl_models"
IP_ADAPTER_WEIGHT = "ip-adapter_sdxl.bin"

GRIP_PROMPT = (
    "a person's hand naturally holding a handbag by its handle, fingers "
    "gently curled around the handle, bag hanging down, photorealistic, "
    "natural lighting, sharp focus, high detail skin and fabric texture"
)
GRIP_NEGATIVE_PROMPT = (
    "blurry, deformed hand, extra fingers, missing fingers, fused fingers, "
    "extra limbs, cartoon, illustration, distorted, low quality, watermark"
)


class HandbagGripEngine:
    """
    MediaPipe Hands landmark indices — identical to HandbagOverlayEngine /
    WristOverlayEngine, kept in sync deliberately (same model, same points).
    """
    WRIST = 0
    INDEX_MCP = 5
    PINKY_MCP = 17

    MIN_HAND_CONFIDENCE = 0.75
    MIN_WRIST_WIDTH_FRACTION = 0.03
    GRIP_OFFSET_FRACTION = 0.35

    # How large the inpaint mask is, in units of knuckle_span — wide enough
    # to cover the grip point and the bag hanging below it, tall enough for
    # a typical handbag silhouette. First-pass estimate.
    MASK_WIDTH_SPAN_MULTIPLIER = 3.0
    MASK_HEIGHT_SPAN_MULTIPLIER = 5.0
    MASK_FEATHER_PX = 12  # soft edge so the inpainted region blends, not a hard seam

    def __init__(self, weights_dir: str | None = None, device: str = "cuda"):
        self._weights_dir = weights_dir or os.getenv("WEIGHTS_DIR", "")
        self._device = device
        self._hands = None
        self._pipe = None

    # ── Lazy model loading ──────────────────────────────────────────────

    def _get_hands(self):
        if self._hands is None:
            import mediapipe as mp
            self._hands = mp.solutions.hands.Hands(
                static_image_mode=True,
                max_num_hands=1,
                min_detection_confidence=self.MIN_HAND_CONFIDENCE,
            )
        return self._hands

    def _get_pipe(self):
        if self._pipe is None:
            import torch
            from diffusers import AutoPipelineForInpainting

            logger.info("HandbagGripEngine: loading SDXL inpainting pipeline (%s)...",
                        SDXL_INPAINT_MODEL_ID)
            cache_dir = (
                os.path.join(self._weights_dir, "handbag_grip")
                if self._weights_dir else None
            )
            pipe = AutoPipelineForInpainting.from_pretrained(
                SDXL_INPAINT_MODEL_ID,
                torch_dtype=torch.float16 if self._device == "cuda" else torch.float32,
                cache_dir=cache_dir,
            )
            pipe.load_ip_adapter(
                IP_ADAPTER_REPO, subfolder=IP_ADAPTER_SUBFOLDER, weight_name=IP_ADAPTER_WEIGHT,
            )
            # Moderate strength: keeps IP-Adapter's influence on product
            # fidelity strong without letting it override the text prompt's
            # control over hand/grip anatomy entirely. Not tuned yet.
            pipe.set_ip_adapter_scale(0.7)
            pipe = pipe.to(self._device)
            self._pipe = pipe
            logger.info("HandbagGripEngine: pipeline loaded.")
        return self._pipe

    # ── Hand landmark detection (same math as HandbagOverlayEngine) ────

    def detect_hand_points(self, person_bgr: np.ndarray):
        """Returns (index_mcp_xy, pinky_mcp_xy, grip_xy, knuckle_span_px) or
        None if no genuinely visible hand is found."""
        h, w = person_bgr.shape[:2]
        rgb = cv2.cvtColor(person_bgr, cv2.COLOR_BGR2RGB)
        result = self._get_hands().process(rgb)
        if not result.multi_hand_landmarks:
            return None

        if result.multi_handedness:
            score = result.multi_handedness[0].classification[0].score
            if score < self.MIN_HAND_CONFIDENCE:
                logger.info("HandbagGripEngine: rejected, handedness score %.2f < %.2f",
                            score, self.MIN_HAND_CONFIDENCE)
                return None

        lm = result.multi_hand_landmarks[0].landmark

        def pt(i):
            return np.array([lm[i].x * w, lm[i].y * h])

        wrist = pt(self.WRIST)
        index_mcp = pt(self.INDEX_MCP)
        pinky_mcp = pt(self.PINKY_MCP)

        knuckle_span = float(np.linalg.norm(index_mcp - pinky_mcp))
        if knuckle_span < w * self.MIN_WRIST_WIDTH_FRACTION:
            logger.info("HandbagGripEngine: rejected, knuckle span %.0fpx too narrow", knuckle_span)
            return None

        knuckle_mid = (index_mcp + pinky_mcp) / 2
        away_from_wrist = knuckle_mid - wrist
        norm = np.linalg.norm(away_from_wrist)
        away_from_wrist = away_from_wrist / norm if norm > 1e-6 else np.array([0.0, -1.0])
        grip = knuckle_mid + away_from_wrist * knuckle_span * self.GRIP_OFFSET_FRACTION

        return index_mcp, pinky_mcp, grip, knuckle_span

    def _build_mask(self, person_size: tuple[int, int], grip_xy: np.ndarray,
                     knuckle_span: float, away_dir: np.ndarray) -> Image.Image:
        """
        White = regenerate, black = keep original. A soft-edged rounded
        rectangle centered on the grip point, extended further in the
        away-from-wrist direction (downward, where the bag hangs) than
        toward the wrist, since the bag body occupies that side.
        """
        w, h = person_size
        mask = Image.new("L", (w, h), 0)
        draw = ImageDraw.Draw(mask)

        half_w = knuckle_span * self.MASK_WIDTH_SPAN_MULTIPLIER / 2
        extent_down = knuckle_span * self.MASK_HEIGHT_SPAN_MULTIPLIER * 0.8
        extent_up = knuckle_span * self.MASK_HEIGHT_SPAN_MULTIPLIER * 0.2

        center = grip_xy + away_dir * (extent_down - extent_up) / 2
        box = [
            center[0] - half_w, center[1] - extent_up - half_w,
            center[0] + half_w, center[1] + extent_down + half_w,
        ]
        draw.rounded_rectangle(box, radius=half_w, fill=255)
        mask = mask.filter(ImageFilter.GaussianBlur(self.MASK_FEATHER_PX))
        return mask

    # ── Public interface — matches HandbagOverlayEngine's shape so it's a
    # drop-in swap in ACCESSORY_ENGINES ──────────────────────────────────

    def validate(self, person_path: str) -> tuple[bool, str | None]:
        person_bgr = cv2.imread(person_path)
        if person_bgr is None:
            return False, "This photo couldn't be read — please choose a different one."
        if self.detect_hand_points(person_bgr) is None:
            return False, ("No hand clearly visible in this photo — try a photo with your hand "
                            "fully visible, not tucked away or out of frame.")
        return True, None

    def run(self, person_path: str, accessory_path: str, output_path: str,
            num_inference_steps: int = 30, guidance_scale: float = 5.0) -> None:
        person_bgr = cv2.imread(person_path)
        if person_bgr is None:
            raise ValueError(f"Could not read person image: {person_path}")

        points = self.detect_hand_points(person_bgr)
        if points is None:
            raise HandDetectionError(
                "No hand clearly visible in this photo — try a photo with your hand fully "
                "visible, not tucked away or out of frame."
            )
        index_mcp, pinky_mcp, grip, knuckle_span = points

        h, w = person_bgr.shape[:2]
        wrist_to_grip_dir = grip - (index_mcp + pinky_mcp) / 2
        norm = np.linalg.norm(wrist_to_grip_dir)
        away_dir = wrist_to_grip_dir / norm if norm > 1e-6 else np.array([0.0, 1.0])

        person_img = Image.open(person_path).convert("RGB")
        accessory_img = Image.open(accessory_path).convert("RGB")
        mask = self._build_mask((w, h), grip, knuckle_span, away_dir)

        pipe = self._get_pipe()
        result = pipe(
            prompt=GRIP_PROMPT,
            negative_prompt=GRIP_NEGATIVE_PROMPT,
            image=person_img,
            mask_image=mask,
            ip_adapter_image=accessory_img,
            num_inference_steps=num_inference_steps,
            guidance_scale=guidance_scale,
        ).images[0]

        result = result.resize(person_img.size, Image.LANCZOS)
        result.convert("RGB").save(output_path, "JPEG", quality=95)
        logger.info(
            "HandbagGripEngine: inpainted grip region at (%.0f, %.0f), knuckle_span=%.0fpx -> %s",
            grip[0], grip[1], knuckle_span, output_path,
        )
