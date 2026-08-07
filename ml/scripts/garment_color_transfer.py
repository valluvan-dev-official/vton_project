"""Post-process color correction for the diffusion-generated garment region.

Why this exists: IDM-VTON regenerates the garment region from noise via
diffusion (conditioned on the source garment through cross-attention /
IP-Adapter) rather than compositing the source pixels directly. Even with a
correctly-configured pipeline, diffusion-reproduced color is never
pixel-exact to the input — saturated colors (pure red in particular) can
drift toward a different hue (commonly reported as red -> brown/olive/green
with SDXL-family fp16 pipelines).

This module is a deterministic, GPU-independent safety net: after
generation, it pulls the *statistics* (LAB mean/std) of the repainted
region back toward the true source-garment color, while preserving the
diffusion model's shading/folds/texture (via `strength` < 1.0 blending,
not a hard overwrite).

Pure numpy/PIL/cv2 — no torch — so it's unit-testable without a GPU.
"""
from __future__ import annotations

import numpy as np
import cv2


def garment_reference_lab_stats(
    garment_rgb: np.ndarray, white_threshold: int = 240
) -> tuple[np.ndarray, np.ndarray]:
    """Mean/std of the garment photo's own color in LAB space.

    Excludes near-white pixels (threshold on all 3 RGB channels) so a plain
    product-photo backdrop doesn't dilute the garment's actual color signal.
    Falls back to the whole image if every pixel is near-white (e.g. a white
    garment) so this never divides by an empty selection.
    """
    garment_rgb = np.asarray(garment_rgb)
    non_backdrop = ~np.all(garment_rgb >= white_threshold, axis=-1)
    if not non_backdrop.any():
        non_backdrop = np.ones(garment_rgb.shape[:2], dtype=bool)

    lab = cv2.cvtColor(garment_rgb, cv2.COLOR_RGB2LAB).astype(np.float64)
    pixels = lab[non_backdrop]
    return pixels.mean(axis=0), pixels.std(axis=0) + 1e-6


def apply_garment_color_transfer(
    result_rgb: np.ndarray,
    mask: np.ndarray,
    reference_garment_rgb: np.ndarray,
    strength: float = 0.85,
) -> np.ndarray:
    """Nudge `result_rgb`'s color, only inside `mask`, toward
    `reference_garment_rgb`'s true color (Reinhard-style LAB mean/std match).

    Args:
        result_rgb: HxWx3 uint8 RGB — the diffusion pipeline's output.
        mask: HxW — the same agnostic/repaint mask used for inference
            (garment region == truthy/>127). Everything outside the mask is
            returned byte-for-byte unchanged.
        reference_garment_rgb: HxWx3 uint8 RGB — the actual source garment
            photo (the ground-truth color to match against).
        strength: 0.0 = leave the model's output untouched; 1.0 = fully
            match the reference's color statistics. Kept < 1.0 by default
            so generated shading/folds/highlights survive the correction —
            only the overall hue/saturation cast is pulled back into line.

    Returns:
        HxWx3 uint8 RGB, same shape as result_rgb.
    """
    result_rgb = np.asarray(result_rgb)
    mask_bool = np.asarray(mask) > 127 if np.asarray(mask).dtype != bool else np.asarray(mask)

    if not mask_bool.any():
        return result_rgb

    target_mean, target_std = garment_reference_lab_stats(reference_garment_rgb)

    result_lab = cv2.cvtColor(result_rgb, cv2.COLOR_RGB2LAB).astype(np.float64)
    region = result_lab[mask_bool]
    region_mean = region.mean(axis=0)
    region_std = region.std(axis=0) + 1e-6

    normalized = (region - region_mean) / region_std
    matched = normalized * target_std + target_mean
    blended = region * (1.0 - strength) + matched * strength

    result_lab[mask_bool] = blended
    result_lab = np.clip(result_lab, 0, 255).astype(np.uint8)
    return cv2.cvtColor(result_lab, cv2.COLOR_LAB2RGB)
