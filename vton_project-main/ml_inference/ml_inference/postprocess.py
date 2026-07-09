"""
postprocess.py — Everything that happens AFTER the diffusion model.

Faithful refactor of Cell 10 of `GPUInferenceEngine.run()`:
LAB-space color correction of the generated garment region followed by a
feathered composite back onto the original person image. Numeric
behaviour is unchanged.
"""
from __future__ import annotations

import logging

import cv2
import numpy as np
from PIL import Image
from skimage.exposure import match_histograms

logger = logging.getLogger(__name__)


def color_correct_and_composite(
    x_out: np.ndarray,          # diffusion output, float HxWx3 in [0,1]
    person_np: np.ndarray,      # original person image, uint8 HxWx3
    garment_np: np.ndarray,     # original garment image, uint8 HxWx3
    shirt_base_mask: np.ndarray,
    g_pred: np.ndarray,
) -> Image.Image:
    result_np = (x_out * 255).astype(np.uint8)
    shirt_mask = shirt_base_mask > 0
    garment_fg_mask = g_pred != 0

    if shirt_mask.sum() > 200 and garment_fg_mask.sum() > 200:
        result_lab = cv2.cvtColor(result_np, cv2.COLOR_RGB2LAB).astype(np.float32)
        garment_lab = cv2.cvtColor(garment_np, cv2.COLOR_RGB2LAB).astype(np.float32)
        corrected_lab = result_lab.copy()

        ref_L = garment_lab[:, :, 0][garment_fg_mask].mean()
        res_L = result_lab[:, :, 0][shirt_mask].mean()
        dL = (ref_L - res_L) * 0.35
        corrected_lab[:, :, 0][shirt_mask] = np.clip(result_lab[:, :, 0][shirt_mask] + dL, 0, 255)

        corrected_lab[:, :, 1][shirt_mask] = match_histograms(
            result_lab[:, :, 1][shirt_mask], garment_lab[:, :, 1][garment_fg_mask])
        corrected_lab[:, :, 2][shirt_mask] = match_histograms(
            result_lab[:, :, 2][shirt_mask], garment_lab[:, :, 2][garment_fg_mask])

        color_corrected_np = cv2.cvtColor(corrected_lab.astype(np.uint8), cv2.COLOR_LAB2RGB)
    else:
        color_corrected_np = result_np

    k_comp = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (7, 7))
    composite_mask = cv2.dilate(shirt_base_mask, k_comp, iterations=1).astype(np.float32)
    composite_mask = cv2.GaussianBlur(composite_mask, (11, 11), 3.0)
    composite_mask = composite_mask[:, :, np.newaxis]

    final_np = (color_corrected_np.astype(np.float32) * composite_mask +
                person_np.astype(np.float32) * (1.0 - composite_mask)).astype(np.uint8)

    return Image.fromarray(final_np)
