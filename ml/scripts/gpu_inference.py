"""
IDM-VTON GPU Inference Engine

Pipeline:
  1. Human Parser (SCHP)  → agnostic person image + mask
  2. OpenPose             → body keypoints
  3. IDM-VTON             → SDXL + IP-Adapter garment try-on

Usage:
    engine = GPUInferenceEngine(weights_dir="/app/ml/weights", device="cuda")
    engine.run(person_path, garment_path, output_path, job_id)
"""
import sys
import os
import hashlib
import subprocess
import logging
import argparse
from pathlib import Path

import numpy as np
import torch
import cv2
from PIL import Image

logger = logging.getLogger(__name__)

SIZE_W, SIZE_H = 768, 1024
PARSE_W, PARSE_H = 384, 512


def _debug_visualization_enabled() -> bool:
    return os.getenv("DEBUG_VISUALIZATION", "false").strip().lower() == "true"


def _save_debug_image(img, path: Path, label: str) -> None:
    """Best-effort save of an intermediate pipeline image for debugging.

    Never raises — a failure here must not affect the real inference run.
    """
    try:
        path.parent.mkdir(parents=True, exist_ok=True)
        if isinstance(img, Image.Image):
            img.save(str(path))
        else:
            Image.fromarray(np.asarray(img).astype(np.uint8)).save(str(path))
        logger.info("[DEBUG_VISUALIZATION] saved %s: %s", label, path)
    except Exception:
        logger.warning("[DEBUG_VISUALIZATION] failed to save %s: %s", label, path, exc_info=True)


class GPUInferenceEngine:
    """Preloads all IDM-VTON models once, then runs inference per-job."""

    def __init__(self, weights_dir: str, device: str = "cuda",
                 workspace: str = "/tmp/vton_workspace"):
        self.device      = torch.device(device)
        self.weights_dir = Path(weights_dir)
        self.idm_path    = self.weights_dir / "idm_vton"
        self.workspace   = Path(workspace)
        self.workspace.mkdir(parents=True, exist_ok=True)

        self._ensure_repos()
        self._load_human_parser()
        self._load_openpose()
        self._load_idm_pipeline()
        logger.info("GPUInferenceEngine (IDM-VTON): all models loaded.")

    # ── Repo bootstrap ────────────────────────────────────────────────────────

    def _ensure_repos(self):
        repo_dir = self.workspace / "repos"
        repo_dir.mkdir(exist_ok=True)

        self.idm_repo = repo_dir / "IDM-VTON"
        if not self.idm_repo.exists():
            logger.info("Cloning IDM-VTON repo...")
            subprocess.run([
                "git", "clone", "--depth=1",
                "https://github.com/yisol/IDM-VTON.git",
                str(self.idm_repo),
            ], check=True)
            logger.info("IDM-VTON repo cloned.")

        # Link ckpt/ from extracted weights into the cloned repo so humanparsing/openpose find them
        ckpt_src = self.idm_path / "ckpt"
        ckpt_dst = self.idm_repo / "ckpt"
        if ckpt_src.exists() and not ckpt_dst.is_symlink():
            if ckpt_dst.exists():
                import shutil
                shutil.rmtree(str(ckpt_dst))
            ckpt_dst.symlink_to(ckpt_src.resolve())
            logger.info("Linked ckpt/ from weights into IDM-VTON repo.")

        # Prepend so IDM-VTON's src/, preprocess/, gradio_demo/ are importable
        for sub in ["", "src", "preprocess", "gradio_demo"]:
            p = str(self.idm_repo / sub) if sub else str(self.idm_repo)
            if p not in sys.path:
                sys.path.insert(0, p)

        # Pre-import utils_mask and openpose util by file path (no __init__.py in those dirs)
        import importlib.util as _ilu

        _spec = _ilu.spec_from_file_location("utils_mask", str(self.idm_repo / "gradio_demo" / "utils_mask.py"))
        _mod = _ilu.module_from_spec(_spec)
        _spec.loader.exec_module(_mod)
        self._get_mask_location = _mod.get_mask_location


    # ── Model loading ─────────────────────────────────────────────────────────

    def _load_human_parser(self):
        from humanparsing.run_parsing import Parsing
        gpu_id = 0 if self.device.type == "cuda" else -1
        self._parser = Parsing(gpu_id)
        logger.info("Human parser (SCHP) loaded.")

    def _load_openpose(self):
        from openpose.run_openpose import OpenPose
        gpu_id = 0 if self.device.type == "cuda" else -1
        self._openpose = OpenPose(gpu_id)
        logger.info("OpenPose loaded.")

    def _load_idm_pipeline(self):
        from src.tryon_pipeline import StableDiffusionXLInpaintPipeline as TryonPipeline
        from src.unet_hacked_garmnet import UNet2DConditionModel as GarmentUNet
        from src.unet_hacked_tryon import UNet2DConditionModel as TryonUNet
        from transformers import CLIPImageProcessor, CLIPVisionModelWithProjection
        from diffusers import AutoencoderKL

        base = str(self.idm_path)

        unet = TryonUNet.from_pretrained(
            base, subfolder="unet", torch_dtype=torch.float16
        )
        unet.requires_grad_(False)

        unet_encoder = GarmentUNet.from_pretrained(
            base, subfolder="unet_encoder", torch_dtype=torch.float16
        )
        unet_encoder.requires_grad_(False)

        image_encoder = CLIPVisionModelWithProjection.from_pretrained(
            base, subfolder="image_encoder", torch_dtype=torch.float16
        )
        image_encoder.requires_grad_(False)

        vae = AutoencoderKL.from_pretrained(
            base, subfolder="vae", torch_dtype=torch.float16
        )

        self._pipe = TryonPipeline.from_pretrained(
            base,
            unet=unet,
            vae=vae,
            feature_extractor=CLIPImageProcessor(),
            image_encoder=image_encoder,
            UNet_Encoder=unet_encoder,
            torch_dtype=torch.float16,
            add_watermarker=False,
            safety_checker=None,
        )
        self._pipe.unet_encoder = unet_encoder
        self._pipe.to(self.device)
        logger.info("IDM-VTON pipeline loaded.")

    # ── Preprocessing ─────────────────────────────────────────────────────────

    def _render_pose_image(self, keypoints_dict: dict, width: int, height: int) -> Image.Image:
        """Render OpenPose keypoints dict → PIL RGB image using cv2."""
        candidate = keypoints_dict.get("pose_keypoints_2d", [])
        canvas = np.zeros((height, width, 3), dtype=np.uint8)
        sx, sy = width / 384.0, height / 512.0
        limbs = [(0,1),(1,2),(2,3),(3,4),(1,5),(5,6),(6,7),(1,8),(8,9),(9,10),
                 (1,11),(11,12),(12,13),(0,14),(14,16),(0,15),(15,17)]
        colors = [(255,0,0),(255,85,0),(255,170,0),(255,255,0),(170,255,0),
                  (85,255,0),(0,255,0),(0,255,85),(0,255,170),(0,255,255),
                  (0,170,255),(0,85,255),(0,0,255),(85,0,255),(170,0,255),
                  (255,0,255),(255,0,170),(255,0,85)]
        for i, (a, b) in enumerate(limbs):
            if a < len(candidate) and b < len(candidate):
                x1, y1 = int(candidate[a][0] * sx), int(candidate[a][1] * sy)
                x2, y2 = int(candidate[b][0] * sx), int(candidate[b][1] * sy)
                if not (x1 == 0 and y1 == 0) and not (x2 == 0 and y2 == 0):
                    cv2.line(canvas, (x1, y1), (x2, y2), colors[i % len(colors)], 3)
        for pt in candidate:
            x, y = int(pt[0] * sx), int(pt[1] * sy)
            if not (x == 0 and y == 0):
                cv2.circle(canvas, (x, y), 5, (255, 255, 255), -1)
        return Image.fromarray(canvas)

    def _detect_sleeve_type(self, garment_pil: Image.Image) -> str:
        """Detect sleeve length from garment image. Returns 'half' or 'full'."""
        img = np.array(garment_pil.convert("RGB"))
        h, w = img.shape[:2]

        # Non-white pixels = garment pixels
        white = (img[:, :, 0] > 240) & (img[:, :, 1] > 240) & (img[:, :, 2] > 240)
        garment = ~white

        rows = np.any(garment, axis=1)
        if not rows.any():
            return "half"

        top    = int(np.argmax(rows))
        bottom = int(h - np.argmax(rows[::-1]) - 1)
        g_h    = bottom - top
        if g_h < 10:
            return "half"

        # Check if garment pixels exist on left/right sides at 45-65% of garment height
        # Full sleeve shirts have fabric on the sides at this zone; half sleeve shirts don't
        mid_top = top + int(g_h * 0.45)
        mid_bot = top + int(g_h * 0.65)
        mid_section = garment[mid_top:mid_bot, :]

        left_sleeve  = mid_section[:, :w // 4].any()
        right_sleeve = mid_section[:, 3 * w // 4:].any()

        sleeve_type = "full" if (left_sleeve and right_sleeve) else "half"
        logger.info("Sleeve detection: %s", sleeve_type)
        return sleeve_type

    def _get_agnostic_mask(self, person_pil: Image.Image, garment_pil: Image.Image,
                           debug_dir: Path | None = None):
        """Parse person → agnostic image + binary mask using SCHP + get_mask_location."""
        parse_result, _ = self._parser(person_pil.resize((PARSE_W, PARSE_H)))
        keypoints = self._openpose(person_pil.resize((PARSE_W, PARSE_H)))

        if debug_dir is not None:
            _save_debug_image(parse_result, debug_dir / "02_parsing_mask.png", "human parsing mask")

        mask, mask_gray = self._get_mask_location("hd", "upper_body", parse_result, keypoints)
        mask = mask.resize((SIZE_W, SIZE_H))

        # For half-sleeve garments, remove arm regions from mask so arms stay visible.
        # For full-sleeve garments, keep mask intact so sleeves cover the arms correctly.
        sleeve_type = self._detect_sleeve_type(garment_pil)
        if sleeve_type == "half":
            mask_np = np.array(mask)
            candidate = keypoints.get("pose_keypoints_2d", [])
            sx, sy = SIZE_W / 384.0, SIZE_H / 512.0
            # Elbow + wrist joints only (not shoulder) — keeps shoulder area masked
            arm_joints = [3, 4, 6, 7]
            for idx in arm_joints:
                if idx < len(candidate):
                    cx = int(candidate[idx][0] * sx)
                    cy = int(candidate[idx][1] * sy)
                    if cx > 0 or cy > 0:
                        cv2.circle(mask_np, (cx, cy), 30, 0, -1)
            mask = Image.fromarray(mask_np)

        import torchvision.transforms as T
        tensor_tf = T.Compose([
            T.ToTensor(),
            T.Normalize([0.5]*3, [0.5]*3),
        ])

        mask_gray_t = (1 - T.ToTensor()(mask)) * tensor_tf(person_pil.resize((SIZE_W, SIZE_H)))
        from torchvision.transforms.functional import to_pil_image
        mask_gray_img = to_pil_image((mask_gray_t + 1.0) / 2.0)

        agnostic = person_pil.resize((SIZE_W, SIZE_H)).copy()
        agnostic.paste(mask_gray_img, None, Image.fromarray(np.uint8(mask)))

        if debug_dir is not None:
            _save_debug_image(mask, debug_dir / "04_agnostic_mask.png", "agnostic mask")

        return agnostic, mask, keypoints

    # ── Inference ─────────────────────────────────────────────────────────────

    def run(self, person_path: str, garment_path: str,
            output_path: str, job_id: str = "") -> str:
        import torchvision.transforms as T

        if not job_id:
            job_id = Path(output_path).stem

        debug_dir = self.workspace / "debug" / job_id if _debug_visualization_enabled() else None

        tensor_tf = T.Compose([T.ToTensor(), T.Normalize([0.5]*3, [0.5]*3)])

        person_pil  = Image.open(person_path).convert("RGB")
        garment_pil = Image.open(garment_path).convert("RGB").resize((SIZE_W, SIZE_H))

        if debug_dir is not None:
            _save_debug_image(person_pil, debug_dir / "01_person_original.jpg", "original person image")

        # ── Step 1: Human parse + agnostic mask ──
        agnostic_pil, mask_pil, keypoints = self._get_agnostic_mask(person_pil, garment_pil, debug_dir=debug_dir)
        person_pil = person_pil.resize((SIZE_W, SIZE_H))

        # ── Step 1b: Scale garment to person shoulder width ──
        candidate = keypoints.get("pose_keypoints_2d", [])
        sx = SIZE_W / PARSE_W
        if len(candidate) > 5:
            r_shoulder = candidate[2]
            l_shoulder = candidate[5]
            if (r_shoulder[0] > 0 or r_shoulder[1] > 0) and (l_shoulder[0] > 0 or l_shoulder[1] > 0):
                person_shoulder_w = abs(l_shoulder[0] - r_shoulder[0]) * sx
                # Reference: assume garment occupies ~55% of SIZE_W at standard fit
                ref_shoulder_w = SIZE_W * 0.55
                scale = person_shoulder_w / ref_shoulder_w
                scale = max(0.7, min(scale, 1.2))  # clamp: avoid extreme scaling
                new_w = int(SIZE_W * scale)
                new_h = int(SIZE_H * scale)
                garment_scaled = garment_pil.resize((new_w, new_h), Image.LANCZOS)
                # Paste on white canvas of SIZE_W x SIZE_H (center it)
                canvas = Image.new("RGB", (SIZE_W, SIZE_H), (255, 255, 255))
                paste_x = (SIZE_W - new_w) // 2
                paste_y = (SIZE_H - new_h) // 2
                canvas.paste(garment_scaled, (paste_x, paste_y))
                garment_pil = canvas
                logger.info("Garment scaled by %.2f (shoulder_w=%.0fpx)", scale, person_shoulder_w)

        # ── Step 2: Prepare tensors ──
        pose_img       = self._render_pose_image(keypoints, SIZE_W, SIZE_H)
        pose_tensor    = tensor_tf(pose_img).unsqueeze(0).to(self.device, torch.float16)
        garment_tensor = tensor_tf(garment_pil).unsqueeze(0).to(self.device, torch.float16)

        if debug_dir is not None:
            _save_debug_image(pose_img, debug_dir / "03_openpose_keypoints.png", "OpenPose keypoint visualization")

        # ── Step 3: Encode prompts ──
        prompt          = "a photo of a person wearing a garment"
        negative_prompt = "monochrome, lowres, bad anatomy, worst quality, low quality"
        garment_desc    = "a garment"

        with torch.inference_mode():
            (
                prompt_embeds,
                negative_prompt_embeds,
                pooled_prompt_embeds,
                negative_pooled_prompt_embeds,
            ) = self._pipe.encode_prompt(
                prompt,
                num_images_per_prompt=1,
                do_classifier_free_guidance=True,
                negative_prompt=negative_prompt,
            )

            prompt_embeds_cloth, _, _, _ = self._pipe.encode_prompt(
                garment_desc,
                num_images_per_prompt=1,
                do_classifier_free_guidance=False,
                negative_prompt=negative_prompt,
            )

        # ── Step 4: Run IDM-VTON pipeline ──
        _seed     = int(hashlib.md5(job_id.encode()).hexdigest()[:8], 16) % (2**31)
        generator = torch.Generator(device="cpu").manual_seed(_seed)

        with torch.inference_mode():
            images = self._pipe(
                prompt_embeds=prompt_embeds.to(self.device, torch.float16),
                negative_prompt_embeds=negative_prompt_embeds.to(self.device, torch.float16),
                pooled_prompt_embeds=pooled_prompt_embeds.to(self.device, torch.float16),
                negative_pooled_prompt_embeds=negative_pooled_prompt_embeds.to(self.device, torch.float16),
                num_inference_steps=30,
                generator=generator,
                strength=1.0,
                pose_img=pose_tensor,
                text_embeds_cloth=prompt_embeds_cloth.to(self.device, torch.float16),
                cloth=garment_tensor,
                mask_image=mask_pil,
                image=person_pil,
                height=SIZE_H,
                width=SIZE_W,
                ip_adapter_image=garment_pil,
                guidance_scale=2.5,
            )[0]

        # ── Step 5: Save result ──
        result = images[0]
        Path(output_path).parent.mkdir(parents=True, exist_ok=True)
        result.save(output_path, "JPEG", quality=95)
        logger.info("IDM-VTON result saved: %s", output_path)

        if debug_dir is not None:
            # No explicit garment-warping step exists in the IDM-VTON pipeline (the
            # garment is conditioned into the UNet via IP-Adapter/unet_encoder, not
            # via a separate warp module), so there is no discrete "warped garment"
            # image to export here.
            _save_debug_image(result, debug_dir / "06_final_output.jpg", "final output image")
            logger.info("[DEBUG_VISUALIZATION] debug images for job %s saved to %s", job_id, debug_dir)
        return output_path


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    parser = argparse.ArgumentParser()
    parser.add_argument("--person",      required=True)
    parser.add_argument("--garment",     required=True)
    parser.add_argument("--output",      required=True)
    parser.add_argument("--job-id",      default="")
    parser.add_argument("--weights-dir", default="/app/ml/weights")
    parser.add_argument("--device",      default="cuda")
    args = parser.parse_args()

    engine = GPUInferenceEngine(args.weights_dir, args.device)
    engine.run(args.person, args.garment, args.output, args.job_id)
