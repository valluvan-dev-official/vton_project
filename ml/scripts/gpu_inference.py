"""
IDM-VTON GPU Inference Engine

Pipeline:
  1. Human Parser (SCHP)  → agnostic person image + mask
  2. OpenPose             → body keypoints
  3. IDM-VTON             → SDXL + IP-Adapter garment try-on

Usage:
    engine = GPUInferenceEngine(weights_dir="/app/ml/weights", device="cuda")
    engine.run(person_path, garment_paths, output_path, job_id)
"""
import os
import sys
import hashlib
import subprocess
import logging
import argparse
from pathlib import Path

import numpy as np
import torch
import cv2
from PIL import Image

from letterbox_geometry import LetterboxTransform, compute_letterbox_geometry
from mask_gap_correction import correct_agnostic_mask_gap, scale_keypoints
from collar_mask_correction import correct_collar_mask

logger = logging.getLogger(__name__)

SIZE_W, SIZE_H = 768, 1024
PARSE_W, PARSE_H = 384, 512

SIZE_ORDER = ["XS", "S", "M", "L", "XL", "XXL"]


# ── Letterbox (aspect-preserving resize + pad) ───────────────────────────────
#
# Replaces a naive Image.resize(), which stretches/squishes the subject
# whenever the source photo isn't already SIZE_W:SIZE_H (768:1024) — that
# stretching is what made bodies look shrunk/distorted in the output. The
# padding added here is undone on the final result before saving (see
# _unletterbox_image, called at the end of run()).
#
# The pure-arithmetic part (compute_letterbox_geometry) lives in
# letterbox_geometry.py so it's unit-testable without PIL/torch/cv2; these two
# functions are the PIL-dependent wrappers around it.

def _letterbox_image(img: Image.Image, target_w: int, target_h: int,
                      fill=(255, 255, 255), resample=Image.LANCZOS):
    """Resize preserving aspect ratio, pad to (target_w, target_h).

    Returns (canvas, transform) — pass `transform` to _unletterbox_image()
    later to invert the padding/resize exactly.
    """
    w, h = img.size
    t = compute_letterbox_geometry(w, h, target_w, target_h)
    resized = img.resize((t.new_w, t.new_h), resample)
    canvas = Image.new(img.mode if img.mode in ("RGB", "L") else "RGB", (target_w, target_h), fill)
    canvas.paste(resized, (t.pad_x, t.pad_y))
    return canvas, t


def _unletterbox_image(img: Image.Image, transform: LetterboxTransform,
                        resample=Image.LANCZOS) -> Image.Image:
    """Crop out the letterbox padding and resize back to the original size."""
    box = (transform.pad_x, transform.pad_y,
           transform.pad_x + transform.new_w, transform.pad_y + transform.new_h)
    return img.crop(box).resize((transform.orig_w, transform.orig_h), resample)


# ── Debug artifact visualization ─────────────────────────────────────────────
#
# Opt-in via DEBUG_VISUALIZATION=true (see docker-compose.gpu.yml). Writes
# every intermediate stage of one job to workspace/debug/<job_id>/ for visual
# QA — disabled by default so it costs nothing in normal operation.

def _debug_visualization_enabled() -> bool:
    return os.getenv("DEBUG_VISUALIZATION", "false").strip().lower() in ("1", "true", "yes")


def _save_debug_image(img, path: Path, description: str = "") -> None:
    """Save a debug artifact. Accepts a PIL Image or a numpy array."""
    try:
        path.parent.mkdir(parents=True, exist_ok=True)
        if isinstance(img, Image.Image):
            img.save(path)
        else:
            arr = np.asarray(img)
            if arr.dtype != np.uint8:
                arr = np.clip(arr, 0, 255).astype(np.uint8)
            Image.fromarray(arr).save(path)
        logger.debug("Debug artifact saved: %s (%s)", path, description)
    except Exception:
        logger.exception("Failed to save debug artifact %s (%s)", path, description)


class GPUInferenceEngine:
    """Preloads all IDM-VTON models once, then runs inference per-job."""

    def __init__(self, weights_dir: str, device: str = "cuda",
                 workspace: str = "/tmp/vton_workspace"):
        self.device      = torch.device(device)
        self.weights_dir = Path(weights_dir)
        self.idm_path    = self.weights_dir / "idm_vton"
        self.workspace   = Path(workspace)
        self.workspace.mkdir(parents=True, exist_ok=True)
        self.last_person_size_estimate = "M"

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

    # ── Multi-garment-image handling ─────────────────────────────────────────

    def _pick_best_garment_image(self, garment_paths: list[str]) -> str:
        """Given several photos of the SAME garment (different angles/zoom/
        lighting), pick the one best suited for the spatial `cloth` channel:
        sharp focus and a large, clearly-visible garment (not a tiny/cropped/
        blurry shot).

        The spatial channel needs one geometrically coherent image — the
        other angles still contribute via _build_ip_adapter_embeds().
        """
        if len(garment_paths) == 1:
            return garment_paths[0]

        best_path, best_score = garment_paths[0], -1.0
        for p in garment_paths:
            try:
                img = np.array(Image.open(p).convert("RGB"))
            except Exception:
                continue
            gray = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)
            sharpness = cv2.Laplacian(gray, cv2.CV_64F).var()

            # Coverage: fraction of non-near-white pixels (garment vs. blank background)
            white = (img[:, :, 0] > 240) & (img[:, :, 1] > 240) & (img[:, :, 2] > 240)
            coverage = 1.0 - white.mean()

            score = sharpness * max(coverage, 0.05)
            logger.info("Garment candidate %s: sharpness=%.1f coverage=%.2f score=%.1f",
                        p, sharpness, coverage, score)
            if score > best_score:
                best_path, best_score = p, score

        logger.info("Selected garment image: %s", best_path)
        return best_path

    def _encode_clip_image_embeds(self, pil_img: Image.Image):
        """Run one image through the pipeline's IP-Adapter CLIP vision encoder
        and return its [1, embed_dim] image embedding (fp16, on self.device)."""
        pixel_values = self._pipe.feature_extractor(images=pil_img, return_tensors="pt").pixel_values
        pixel_values = pixel_values.to(self.device, dtype=self._pipe.image_encoder.dtype)
        with torch.inference_mode():
            return self._pipe.image_encoder(pixel_values).image_embeds

    def _build_ip_adapter_embeds(self, garment_pils: list[Image.Image]):
        """Average CLIP image embeddings across ALL submitted garment angles
        into one appearance-conditioning embedding (front view, close-up
        print/texture shot, back view, etc. all contribute), rather than
        conditioning on just a single selected photo.

        Returns a CFG-ready embedding tensor ([uncond, cond] stacked, per
        diffusers IP-Adapter convention) suitable for `ip_adapter_image_embeds`,
        or None if anything about the pipeline's IP-Adapter internals doesn't
        match expectations — callers should fall back to `ip_adapter_image=`
        with a single image in that case rather than fail the whole job.
        """
        try:
            embeds = [self._encode_clip_image_embeds(img) for img in garment_pils]
            avg_embed = torch.stack(embeds, dim=0).mean(dim=0)  # [1, embed_dim]
            negative_embed = torch.zeros_like(avg_embed)
            cfg_embed = torch.cat([negative_embed, avg_embed], dim=0)  # [uncond, cond]
            return [cfg_embed.to(self.device, torch.float16)]
        except Exception:
            logger.exception(
                "Multi-image IP-Adapter embedding averaging failed — "
                "falling back to single-image ip_adapter_image."
            )
            return None

    # ── Person-size estimation + size-aware fit ──────────────────────────────

    def _estimate_person_size(self, keypoints: dict, fallback_height_px: float) -> str:
        """Auto-detect a body-size bucket (S/M/L/XL/...) from pose keypoints.

        Pixel measurements are scale-free once normalised by the person's own
        height in the *same* image, so no real-world reference object is
        needed. We use shoulder width (keypoints 2, 5) divided by an
        estimated standing height (neck→mid-hip torso length × ~3.05, the
        typical torso:height ratio). Thresholds are calibrated against
        average adult shoulder-width/height ratios and are necessarily
        approximate — good enough to pick a relative garment fit, not a
        substitute for real anthropometric measurement.
        """
        candidate = keypoints.get("pose_keypoints_2d", [])

        def pt(i):
            if i >= len(candidate):
                return None
            x, y = candidate[i][0], candidate[i][1]
            return (x, y) if (x > 0 or y > 0) else None

        r_sh, l_sh, neck = pt(2), pt(5), pt(1)
        r_hip, l_hip = pt(8), pt(11)

        if not (r_sh and l_sh):
            return "M"

        shoulder_w = abs(l_sh[0] - r_sh[0])

        hip_pts = [p for p in (r_hip, l_hip) if p]
        if neck and hip_pts:
            mid_hip_y = sum(p[1] for p in hip_pts) / len(hip_pts)
            torso_h = abs(mid_hip_y - neck[1])
            body_h = torso_h * 3.05 if torso_h > 0 else fallback_height_px
        else:
            body_h = fallback_height_px

        if body_h <= 0 or shoulder_w <= 0:
            return "M"

        ratio = shoulder_w / body_h
        thresholds = [(0.235, "S"), (0.255, "M"), (0.275, "L"), (0.295, "XL")]
        for limit, label in thresholds:
            if ratio <= limit:
                return label
        return "XXL"

    def _fit_scale_factor(self, person_size: str, garment_size: str) -> float:
        """Combine detected person size + requested garment size into a scale
        multiplier: an L garment on an M person should sit looser/larger than
        an M garment on an M person, and vice versa for a size-down."""
        p_idx = SIZE_ORDER.index(person_size) if person_size in SIZE_ORDER else SIZE_ORDER.index("M")
        g_idx = SIZE_ORDER.index(garment_size) if garment_size in SIZE_ORDER else p_idx

        diff = g_idx - p_idx
        factor = 1.0 + diff * 0.06  # ~6% garment growth per size step
        return max(0.75, min(factor, 1.35))

    # ── Agnostic mask ─────────────────────────────────────────────────────────

    def _get_agnostic_mask(self, person_pil: Image.Image, garment_pil: Image.Image,
                            garment_size: str = "M", debug_dir: Path | None = None):
        """Parse person → agnostic image + binary mask using SCHP + get_mask_location.

        person_pil is expected to already be letterboxed to SIZE_W:SIZE_H
        (same 3:4 aspect as PARSE_W:PARSE_H), so the resize below is a
        uniform downscale, not a distortion.
        """
        parse_result, _ = self._parser(person_pil.resize((PARSE_W, PARSE_H)))
        keypoints = self._openpose(person_pil.resize((PARSE_W, PARSE_H)))

        self.last_person_size_estimate = self._estimate_person_size(keypoints, PARSE_H)

        if debug_dir is not None:
            _save_debug_image(parse_result, debug_dir / "02_parsing_mask.png", "human parsing mask")

        mask, mask_gray = self._get_mask_location("hd", "upper_body", parse_result, keypoints)
        # NEAREST: `mask` is a binary/label image — smooth resampling would
        # blur 0/255 edges into intermediate gray values.
        mask = mask.resize((SIZE_W, SIZE_H), Image.NEAREST)

        # ── Remove parser-confirmed background wrongly bridged into the mask ──
        # get_mask_location()'s dilated shoulder-elbow-wrist arm line can cross
        # the empty background next to a bent elbow, marking real background as
        # "editable" — IDM-VTON then paints garment fabric into that gap. This
        # removes only that specific enclosed component per arm; clothes,
        # actual arm pixels, and valid outward sleeve space are untouched.
        parse_np_full = np.array(parse_result.resize((SIZE_W, SIZE_H), Image.NEAREST))
        sx, sy = SIZE_W / float(PARSE_W), SIZE_H / float(PARSE_H)
        scaled_keypoints = scale_keypoints(keypoints, sx, sy)
        corrected_np, gap_diag, gap_debug_np = correct_agnostic_mask_gap(
            parse_np_full, np.array(mask), scaled_keypoints
        )
        mask = Image.fromarray(corrected_np)

        if debug_dir is not None:
            _save_debug_image(gap_debug_np, debug_dir / "04a_detected_arm_torso_gaps.png",
                               "detected bent-arm/torso background gap components")
            _save_debug_image(mask, debug_dir / "04b_corrected_agnostic_mask.png",
                               "agnostic mask after arm/torso gap correction")

        # ── Add leftover original-garment pixels around the neckline/collar ──
        # The base mask sometimes leaves a thin band of the old T-shirt visible
        # directly below the neck / between the shoulders (especially with dark
        # garments), which then shows through the newly generated collar. This
        # adds only that specific leftover band; it never touches background,
        # arm, or neck-skin pixels, and is a separate, independent correction
        # from the arm/torso-gap fix above.
        collar_corrected_np, collar_candidate_np, collar_protection_np, collar_diag = correct_collar_mask(
            parse_np_full, np.array(mask), scaled_keypoints
        )
        mask = Image.fromarray(collar_corrected_np)

        if debug_dir is not None:
            _save_debug_image(collar_candidate_np, debug_dir / "04c_detected_collar_region.png",
                               "raw garment-labelled collar candidate (before protection)")
            _save_debug_image(collar_protection_np, debug_dir / "04d_collar_protection_mask.png",
                               "protected pixels excluded from collar correction")
            _save_debug_image(mask, debug_dir / "04e_final_corrected_agnostic_mask.png",
                               "agnostic mask after arm-gap and collar correction")

        # Grow/shrink the mask region for a size-up/size-down garment so the
        # repainted area matches how loose or tight the requested size should
        # actually sit versus the detected body size.
        fit_factor = self._fit_scale_factor(self.last_person_size_estimate, garment_size)
        if abs(fit_factor - 1.0) > 1e-3:
            kernel_px = int(round(abs(fit_factor - 1.0) * 40))  # ~40px per 100% delta
            if kernel_px > 0:
                kernel = np.ones((kernel_px, kernel_px), np.uint8)
                mask_np = np.array(mask)
                if fit_factor > 1.0:
                    mask_np = cv2.dilate(mask_np, kernel, iterations=1)
                else:
                    mask_np = cv2.erode(mask_np, kernel, iterations=1)
                mask = Image.fromarray(mask_np)

        # For half-sleeve garments, remove arm regions from mask so arms stay visible.
        # For full-sleeve garments, keep mask intact so sleeves cover the arms correctly.
        sleeve_type = self._detect_sleeve_type(garment_pil)
        if sleeve_type == "half":
            mask_np = np.array(mask)
            candidate = keypoints.get("pose_keypoints_2d", [])
            sx2, sy2 = SIZE_W / 384.0, SIZE_H / 512.0
            # Elbow + wrist joints only (not shoulder) — keeps shoulder area masked
            arm_joints = [3, 4, 6, 7]
            for idx in arm_joints:
                if idx < len(candidate):
                    cx = int(candidate[idx][0] * sx2)
                    cy = int(candidate[idx][1] * sy2)
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

    def run(self, person_path: str, garment_paths, output_path: str,
            job_id: str = "", garment_size: str = "M") -> str:
        """garment_paths: path to a single garment photo, or a list of paths —
        multiple angles/zoom levels of the SAME garment. The clearest one
        drives the spatial garment-warping channel (see
        _pick_best_garment_image); ALL of them contribute to the appearance/
        texture conditioning via averaged CLIP embeddings (see
        _build_ip_adapter_embeds)."""
        import torchvision.transforms as T

        if not job_id:
            job_id = Path(output_path).stem
        garment_size = (garment_size or "M").strip().upper()
        if isinstance(garment_paths, str):
            garment_paths = [garment_paths]

        debug_dir = self.workspace / "debug" / job_id if _debug_visualization_enabled() else None

        tensor_tf = T.Compose([T.ToTensor(), T.Normalize([0.5]*3, [0.5]*3)])

        garment_path = self._pick_best_garment_image(garment_paths)
        person_pil_raw = Image.open(person_path).convert("RGB")
        garment_orig = Image.open(garment_path).convert("RGB")

        # Raw (un-letterboxed) copies of every submitted angle, for the
        # appearance/texture (IP-Adapter) conditioning below — CLIP's own
        # feature extractor handles resizing, so no letterbox needed here.
        all_garment_pils_raw = [Image.open(p).convert("RGB") for p in garment_paths]

        if debug_dir is not None:
            _save_debug_image(person_pil_raw, debug_dir / "01_person_original.jpg", "original person image")

        # ── Step 0: Letterbox person + garment to SIZE_W x SIZE_H without
        # distorting geometry (replaces a direct stretch-resize that squashed
        # non-3:4 photos). Parsing, OpenPose and masks are all derived from
        # this same letterboxed canvas below, so they inherit its padding
        # automatically — no separate letterbox call is needed for them. The
        # padding is undone on the final result before saving (Step 5).
        person_pil, letterbox_transform = _letterbox_image(person_pil_raw, SIZE_W, SIZE_H)
        garment_pil, _ = _letterbox_image(garment_orig, SIZE_W, SIZE_H)

        if debug_dir is not None:
            _save_debug_image(person_pil, debug_dir / "01b_person_letterboxed.jpg",
                               "letterboxed person image (pipeline input)")

        # ── Step 1: Human parse + agnostic mask ──
        agnostic_pil, mask_pil, keypoints = self._get_agnostic_mask(
            person_pil, garment_pil, garment_size=garment_size, debug_dir=debug_dir,
        )
        person_size = self.last_person_size_estimate
        logger.info("Detected person size: %s | requested garment size: %s",
                    person_size, garment_size)

        # ── Step 1b: Scale garment to person shoulder width + requested fit ──
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
                # Blend in the requested garment size vs. detected person size —
                # an L garment on an M person should render bigger/looser than
                # an M garment on the same person, and vice versa.
                scale *= self._fit_scale_factor(person_size, garment_size)
                scale = max(0.65, min(scale, 1.45))  # clamp: avoid extreme scaling
                new_w = int(SIZE_W * scale)
                new_h = int(SIZE_H * scale)
                garment_scaled = garment_pil.resize((new_w, new_h), Image.LANCZOS)
                # Paste on white canvas of SIZE_W x SIZE_H (center it)
                canvas = Image.new("RGB", (SIZE_W, SIZE_H), (255, 255, 255))
                paste_x = (SIZE_W - new_w) // 2
                paste_y = (SIZE_H - new_h) // 2
                canvas.paste(garment_scaled, (paste_x, paste_y))
                garment_pil = canvas
                logger.info("Garment scaled by %.2f (shoulder_w=%.0fpx, fit=%s->%s)",
                            scale, person_shoulder_w, person_size, garment_size)

        # ── Step 2: Prepare tensors ──
        pose_img       = self._render_pose_image(keypoints, SIZE_W, SIZE_H)
        pose_tensor    = tensor_tf(pose_img).unsqueeze(0).to(self.device, torch.float16)
        garment_tensor = tensor_tf(garment_pil).unsqueeze(0).to(self.device, torch.float16)

        # Appearance/texture conditioning — average CLIP embeddings across
        # every submitted garment angle so print/texture details visible only
        # in a close-up shot (say) still inform the result, not just whichever
        # single image drives the spatial `cloth` channel above.
        ip_adapter_embeds = None
        if len(all_garment_pils_raw) > 1:
            ip_adapter_embeds = self._build_ip_adapter_embeds(all_garment_pils_raw)

        if debug_dir is not None:
            _save_debug_image(garment_pil, debug_dir / "03_warped_garment.jpg",
                               "garment scaled/positioned for the pipeline")
            _save_debug_image(pose_img, debug_dir / "03b_pose_keypoints.png", "rendered pose keypoints")

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

        # Pass either the pre-computed multi-image averaged embedding, or
        # fall back to the single best garment image — whichever is available.
        ip_adapter_kwargs = (
            {"ip_adapter_image_embeds": ip_adapter_embeds}
            if ip_adapter_embeds is not None
            else {"ip_adapter_image": garment_pil}
        )

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
                guidance_scale=2.5,
                **ip_adapter_kwargs,
            )[0]

        # ── Step 5: Undo letterbox padding, restore original aspect ratio, save ──
        result = images[0]

        if debug_dir is not None:
            _save_debug_image(result, debug_dir / "05_raw_pipeline_output.jpg",
                               "pipeline output before un-letterboxing")

        result = _unletterbox_image(result, letterbox_transform)
        Path(output_path).parent.mkdir(parents=True, exist_ok=True)
        result.save(output_path, "JPEG", quality=95)

        if debug_dir is not None:
            _save_debug_image(result, debug_dir / "06_final_output.jpg",
                               "final output, restored to original aspect ratio")

        logger.info("IDM-VTON result saved: %s (restored to original %dx%d)",
                    output_path, letterbox_transform.orig_w, letterbox_transform.orig_h)
        return output_path


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    parser = argparse.ArgumentParser()
    parser.add_argument("--person",      required=True)
    parser.add_argument("--garment",     required=True, nargs="+",
                         help="One or more paths — same garment, different angles.")
    parser.add_argument("--output",      required=True)
    parser.add_argument("--job-id",      default="")
    parser.add_argument("--weights-dir", default="/app/ml/weights")
    parser.add_argument("--device",      default="cuda")
    parser.add_argument("--garment-size", default="M")
    args = parser.parse_args()

    engine = GPUInferenceEngine(args.weights_dir, args.device)
    engine.run(args.person, args.garment, args.output, args.job_id, args.garment_size)
