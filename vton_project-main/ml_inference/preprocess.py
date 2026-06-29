"""
preprocess.py — Everything that happens BEFORE the diffusion model.

Faithful refactor of the preprocessing portion of
`GPUInferenceEngine.run()` (Cells 5/6/7 in the original notebook):

  * SegFormer parsing of person + garment
  * agnostic mask + collar-strip construction
  * DensePose IUV (computed for parity with the original pipeline)
  * AFWM garment warping

No numeric behaviour has changed; the code has only been grouped into
named functions and a single `preprocess()` entry point that returns a
`PreprocessResult` consumed by `predictor.py`.
"""
from __future__ import annotations

import logging
from dataclasses import dataclass

import cv2
import numpy as np
import torch
import torch.nn.functional as F
from PIL import Image

from .model_loader import ModelBundle, SIZE

logger = logging.getLogger(__name__)


@dataclass
class PreprocessResult:
    person_np: np.ndarray
    garment_np: np.ndarray
    agnostic_pil: Image.Image
    garment_clean_pil: Image.Image
    warped_garment_pil: Image.Image
    agnostic_mask: np.ndarray      # uint8 HxW {0,1}
    shirt_base_mask: np.ndarray    # uint8 HxW {0,1} (== blend_mask)
    g_pred: np.ndarray             # garment segmentation labels


# ── SegFormer parsing ────────────────────────────────────────────────────────

def segment(bundle: ModelBundle, pil_img: Image.Image) -> np.ndarray:
    inputs = bundle.seg_processor(images=pil_img, return_tensors="pt")
    with torch.no_grad():
        logits = bundle.seg_model(**inputs).logits
    up = F.interpolate(logits, size=(SIZE, SIZE), mode="bilinear", align_corners=False)
    return up.argmax(dim=1).squeeze().numpy()


# ── DensePose IUV ────────────────────────────────────────────────────────────

def get_densepose_iuv(bundle: ModelBundle, person_np_img: np.ndarray) -> np.ndarray:
    outputs = bundle.dp_predictor(person_np_img[:, :, ::-1])
    instances = outputs["instances"]
    iuv_full = np.zeros((SIZE, SIZE, 3), dtype=np.float32)
    if len(instances) == 0:
        return iuv_full
    best = instances.scores.cpu().numpy().argmax()
    result = instances.pred_densepose[best]
    bbox = instances.pred_boxes.tensor.cpu().numpy()[best]
    fine_segm = result.fine_segm.cpu()
    u_map = result.u.cpu()
    v_map = result.v.cpu()
    part_idx_t = fine_segm.argmax(dim=0)
    idx_exp = part_idx_t.unsqueeze(0)
    u_vals = u_map.gather(0, idx_exp).squeeze(0).numpy()
    v_vals = v_map.gather(0, idx_exp).squeeze(0).numpy()
    part_idx = part_idx_t.numpy().astype(np.float32)
    x1, y1, x2, y2 = map(int, bbox)
    bh, bw = max(1, y2 - y1), max(1, x2 - x1)
    I_r = cv2.resize(part_idx, (bw, bh))
    U_r = cv2.resize(u_vals, (bw, bh))
    V_r = cv2.resize(v_vals, (bw, bh))
    I_f = np.zeros((SIZE, SIZE), dtype=np.float32)
    U_f = np.zeros((SIZE, SIZE), dtype=np.float32)
    V_f = np.zeros((SIZE, SIZE), dtype=np.float32)
    y1c, y2c = max(0, y1), min(SIZE, y2)
    x1c, x2c = max(0, x1), min(SIZE, x2)
    dh, dw = y2c - y1c, x2c - x1c
    I_f[y1c:y2c, x1c:x2c] = I_r[:dh, :dw] / 112.0
    U_f[y1c:y2c, x1c:x2c] = U_r[:dh, :dw]
    V_f[y1c:y2c, x1c:x2c] = V_r[:dh, :dw]
    return np.stack([I_f, U_f, V_f], axis=-1)


# ── Mask construction ────────────────────────────────────────────────────────

def build_masks(person_np: np.ndarray, garment_np: np.ndarray,
                pred: np.ndarray, g_pred: np.ndarray):
    """Build agnostic mask, collar strip and the cleaned garment.

    Returns (agnostic_pil, agnostic_mask, shirt_base_mask, garment_clean_pil).
    """
    garment_clean_np = garment_np.copy()
    garment_clean_np[g_pred == 0] = [128, 128, 128]
    garment_clean_pil = Image.fromarray(garment_clean_np)

    upper_labels = [4, 5, 7]
    shirt_base_mask = np.isin(pred, upper_labels).astype(np.uint8)

    garment_fg = (g_pred != 0)
    garment_fg_ys = np.where(garment_fg.any(axis=1))[0]
    if len(garment_fg_ys) > 0:
        g_neckline_y = int(garment_fg_ys.min())
        neckline_row_cols = np.where(garment_fg[g_neckline_y])[0]
        g_neck_width = int(neckline_row_cols.max() - neckline_row_cols.min()) if len(neckline_row_cols) > 0 else 120
    else:
        g_neckline_y, g_neck_width = 100, 120

    shirt_ys = np.where(shirt_base_mask.any(axis=1))[0]
    if len(shirt_ys) > 0:
        shirt_top_y = int(shirt_ys.min())
        shirt_cols = np.where(shirt_base_mask.any(axis=0))[0]
        person_center_x = int(shirt_cols.mean()) if len(shirt_cols) > 0 else SIZE // 2
        half_w = max(80, g_neck_width // 2 + 30)
        strip_x1 = max(0, person_center_x - half_w)
        strip_x2 = min(SIZE, person_center_x + half_w)
        collar_erase_up = max(50, int(g_neckline_y * 0.5))
        collar_strip = np.zeros((SIZE, SIZE), dtype=np.uint8)
        collar_strip[max(0, shirt_top_y - collar_erase_up):min(SIZE, shirt_top_y + 30), strip_x1:strip_x2] = 1
    else:
        collar_strip = np.zeros((SIZE, SIZE), dtype=np.uint8)

    k = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (15, 15))
    shirt_dilated = cv2.dilate(shirt_base_mask, k, iterations=1)
    agnostic_mask = np.clip(shirt_dilated.astype(np.int32) + collar_strip.astype(np.int32), 0, 1).astype(np.uint8)

    agnostic_np = person_np.copy()
    agnostic_np[agnostic_mask > 0] = [128, 128, 128]
    agnostic_pil = Image.fromarray(agnostic_np)

    return agnostic_pil, agnostic_mask, shirt_base_mask, garment_clean_pil


# ── AFWM warp ────────────────────────────────────────────────────────────────

def warp_garment(bundle: ModelBundle, pred: np.ndarray, agnostic_pil: Image.Image,
                 agnostic_mask: np.ndarray, garment_clean_pil: Image.Image) -> Image.Image:
    import torchvision.transforms as T

    device = bundle.device

    seg_to_parse = {0: 0, 2: 1, 11: 2, 3: 3, 4: 3, 7: 3, 5: 4, 6: 5,
                    8: 6, 14: 7, 15: 8, 12: 9, 13: 10, 9: 11, 10: 12}
    parse_map = np.zeros((13, SIZE, SIZE), dtype=np.float32)
    for seg_lbl, parse_ch in seg_to_parse.items():
        parse_map[parse_ch][pred == seg_lbl] = 1.0
    parse_map[3][agnostic_mask > 0] = 0
    parse_map[7][agnostic_mask > 0] = 0
    parse_map[8][agnostic_mask > 0] = 0
    parse_map[2][agnostic_mask > 0] = 1.0

    warp_tf = T.Compose([T.ToTensor(), T.Normalize([0.5] * 3, [0.5] * 3)])
    agnostic_t = warp_tf(agnostic_pil).unsqueeze(0).to(device)
    parse_t = torch.from_numpy(parse_map).unsqueeze(0).to(device)
    cond_input = torch.cat([agnostic_t, parse_t], dim=1)
    garment_t = warp_tf(garment_clean_pil).unsqueeze(0).to(device)

    with torch.no_grad():
        warped_cloth, _ = bundle.warp_model(cond_input, garment_t)
        warped_np = warped_cloth.squeeze().permute(1, 2, 0).cpu().numpy()
        warped_np = ((warped_np + 1) / 2 * 255).clip(0, 255).astype(np.uint8)
        return Image.fromarray(warped_np)


# ── Top-level preprocessing ──────────────────────────────────────────────────

def preprocess(bundle: ModelBundle, person_pil: Image.Image,
               garment_pil: Image.Image) -> PreprocessResult:
    """Run segmentation, mask building, DensePose and warp.

    `person_pil` / `garment_pil` must already be RGB and 512x512.
    """
    person_np = np.array(person_pil)
    garment_np = np.array(garment_pil)

    pred = segment(bundle, person_pil)
    g_pred = segment(bundle, garment_pil)

    agnostic_pil, agnostic_mask, shirt_base_mask, garment_clean_pil = build_masks(
        person_np, garment_np, pred, g_pred
    )

    # Computed for parity with the original pipeline.
    _ = get_densepose_iuv(bundle, person_np)

    warped_garment_pil = warp_garment(bundle, pred, agnostic_pil, agnostic_mask, garment_clean_pil)

    return PreprocessResult(
        person_np=person_np,
        garment_np=garment_np,
        agnostic_pil=agnostic_pil,
        garment_clean_pil=garment_clean_pil,
        warped_garment_pil=warped_garment_pil,
        agnostic_mask=agnostic_mask,
        shirt_base_mask=shirt_base_mask,
        g_pred=g_pred,
    )
