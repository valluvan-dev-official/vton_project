"""
Correction for leftover original-garment pixels around the neckline/collar.

The base agnostic mask (get_mask_location()'s "hd" upper-body mask, after the
existing bent-arm/torso-gap correction in mask_gap_correction.py) sometimes
does not fully cover the original T-shirt's collar: a thin band of the old
garment survives directly below the neck and between the shoulders,
especially with dark T-shirts. IDM-VTON then renders the new collar with a
dark/black patch of the old fabric still visible underneath or around it.

This module adds *only* that leftover-garment band back into the editable
mask. It never guesses at unconfirmed parsing labels (face/hair/chin) — the
geometric candidate region is derived purely from the neck and shoulder
OpenPose keypoints, and the real safety net is the parser-label filter: only
pixels the parser itself already calls "old garment" (upper_clothes=4,
dress=7) are ever added, and pixels the parser calls background(0),
left_arm(14), right_arm(15), or neck/skin(18) are always excluded,
regardless of geometry.

Dependencies: numpy + opencv only — importable and unit-testable without
PIL, torch, or a loaded GPU model. Does not import or modify
mask_gap_correction.py.
"""
from __future__ import annotations

import logging
from typing import Optional

import cv2
import numpy as np

logger = logging.getLogger(__name__)

# OpenPose COCO-18 keypoint indices — same convention already used throughout
# gpu_inference.py and mask_gap_correction.py.
NECK = 1
RIGHT_SHOULDER = 2
LEFT_SHOULDER = 5

# Parser labels confirmed from the live IDM-VTON utils_mask.py.
BACKGROUND_LABEL = 0
UPPER_CLOTHES_LABEL = 4
DRESS_LABEL = 7
LEFT_ARM_LABEL = 14
RIGHT_ARM_LABEL = 15
NECK_SKIN_LABEL = 18

GARMENT_LABELS = (UPPER_CLOTHES_LABEL, DRESS_LABEL)
PROTECTED_LABELS = (BACKGROUND_LABEL, LEFT_ARM_LABEL, RIGHT_ARM_LABEL, NECK_SKIN_LABEL)

# Tunables — all fractions of measured shoulder width, never fixed pixels.
DEFAULT_TOP_MARGIN_FRAC    = 0.15   # how far above neck.y the candidate reaches
DEFAULT_CHEST_EXTENT_FRAC  = 0.60   # how far below neck.y the candidate reaches (upper-chest only)
DEFAULT_HALF_WIDTH_FRAC    = 0.325  # candidate half-width vs shoulder_width (stays inside shoulders)
DEFAULT_MIN_SHOULDER_WIDTH = 1e-3
JOINT_CONFIDENCE_FLOOR     = 0.05   # same convention as mask_gap_correction._joint


def _joint(points: list, idx: int) -> Optional[tuple]:
    """Return (x, y) for keypoint `idx`, or None if missing/invalid/low-confidence.

    Identical convention to mask_gap_correction._joint: (0, 0) means "not
    detected"; an optional third [x, y, confidence] element is honoured
    against the project's confidence floor.
    """
    if points is None or idx >= len(points):
        return None
    p = points[idx]
    if len(p) < 2:
        return None
    x, y = float(p[0]), float(p[1])
    if x == 0.0 and y == 0.0:
        return None
    if len(p) >= 3 and p[2] is not None and float(p[2]) < JOINT_CONFIDENCE_FLOOR:
        return None
    return (x, y)


def _binary(mask: np.ndarray) -> np.ndarray:
    """Explicit {0, 255} uint8 normalization."""
    return np.where(np.asarray(mask) > 0, 255, 0).astype(np.uint8)


def correct_collar_mask(
    parse_array: np.ndarray,
    mask_array: np.ndarray,
    keypoints: dict,
    top_margin_frac: float = DEFAULT_TOP_MARGIN_FRAC,
    chest_extent_frac: float = DEFAULT_CHEST_EXTENT_FRAC,
    half_width_frac: float = DEFAULT_HALF_WIDTH_FRAC,
):
    """Add leftover old-garment pixels around the neckline/collar to the mask.

    parse_array : 2D int array of parser labels, shape (H, W).
    mask_array  : 2D array, same (H, W) — nonzero = editable/inpaint. This is
                  expected to already be the arm-gap-corrected mask.
    keypoints   : {"pose_keypoints_2d": [[x, y] or [x, y, conf], ...]}, in the
                  SAME pixel coordinate space as the two arrays above.

    Returns (corrected_mask, candidate_debug, protection_debug, diagnostics)
      corrected_mask   : np.uint8 {0, 255} — mask_array with the leftover
                          collar band added, or mask_array unchanged
                          (binary-normalized) if skipped/reverted.
      candidate_debug  : np.uint8 {0, 255} — raw garment-labelled candidate
                          region, BEFORE protection subtraction.
      protection_debug : np.uint8 {0, 255} — protected pixels (255) /
                          everything else (0).
      diagnostics      : dict with side="collar", skipped, added_px, reason.
    """
    if parse_array.shape[:2] != mask_array.shape[:2]:
        raise ValueError(
            f"parse_array shape {parse_array.shape[:2]} != mask_array shape {mask_array.shape[:2]}"
        )

    H, W = mask_array.shape[:2]
    base_mask = _binary(mask_array)
    empty_debug = np.zeros((H, W), dtype=np.uint8)

    def _skip(reason: str):
        diag = {"side": "collar", "skipped": True, "added_px": 0, "reason": reason}
        logger.info("[collar_mask_correction] skipped=True added_px=0 reason=%s", reason)
        return base_mask, empty_debug, empty_debug, diag

    pts = (keypoints or {}).get("pose_keypoints_2d", [])
    neck = _joint(pts, NECK)
    r_shoulder = _joint(pts, RIGHT_SHOULDER)
    l_shoulder = _joint(pts, LEFT_SHOULDER)

    if neck is None or r_shoulder is None or l_shoulder is None:
        return _skip("missing_or_invalid_joints")

    shoulder_width = float(np.linalg.norm(np.array(r_shoulder) - np.array(l_shoulder)))
    if shoulder_width < DEFAULT_MIN_SHOULDER_WIDTH:
        return _skip("invalid_shoulder_width")

    # ── Candidate geometry ──────────────────────────────────────────────────
    # Horizontal: stays strictly inside the shoulder line (req: "inside the
    # shoulder boundaries").
    center_x = (r_shoulder[0] + l_shoulder[0]) / 2.0
    half_width = half_width_frac * shoulder_width
    left_x = center_x - half_width
    right_x = center_x + half_width

    # Vertical: anchored on neck.y directly (NOT clamped to shoulder.y — that
    # would exclude the neck-to-shoulder band where leftover collar fabric
    # actually lives). A small dynamic margin above neck.y catches the collar
    # rim immediately around/below the neck point; the real protection
    # against reaching the face/chin/hair is the parser-label filter below,
    # not this geometric ceiling.
    top_y = max(0.0, neck[1] - top_margin_frac * shoulder_width)
    bottom_y = min(float(H - 1), neck[1] + chest_extent_frac * shoulder_width)

    if right_x - left_x < 1.0 or bottom_y - top_y < 1.0:
        return _skip("empty_candidate_region")

    center_y = (top_y + bottom_y) / 2.0
    radius_x = max(1.0, (right_x - left_x) / 2.0)
    radius_y = max(1.0, (bottom_y - top_y) / 2.0)

    candidate_region = np.zeros((H, W), dtype=np.uint8)
    cv2.ellipse(
        candidate_region,
        (int(round(center_x)), int(round(center_y))),
        (int(round(radius_x)), int(round(radius_y))),
        0, 0, 360, 1, -1,
    )
    if not candidate_region.any():
        return _skip("empty_candidate_region")

    # ── Raw candidate: garment-labelled pixels inside the geometric region ──
    garment_px = np.isin(parse_array, GARMENT_LABELS)
    raw_candidate = (candidate_region.astype(bool)) & garment_px
    candidate_debug = _binary(raw_candidate)

    if not raw_candidate.any():
        return _skip("no_original_garment_pixels")

    # ── Protection mask (standalone — everything that must never be added) ──
    protected_labels_px = np.isin(parse_array, PROTECTED_LABELS)
    outside_bounds = ~candidate_region.astype(bool)
    protection_mask = protected_labels_px | outside_bounds
    protection_debug = _binary(protection_mask)

    # ── Final pixels to add ──────────────────────────────────────────────────
    already_editable = base_mask > 0
    to_add = raw_candidate & (~protection_mask) & (~already_editable)

    if not to_add.any():
        # Garment pixels exist in-region but are already editable, or were
        # fully excluded by the protection mask.
        return _skip("no_original_garment_pixels")

    # ── Validation (before returning) ────────────────────────────────────────
    corrected = base_mask.copy()
    corrected[to_add] = 255
    corrected = _binary(corrected)

    added_mask = (corrected == 255) & (base_mask == 0)

    if corrected.shape != mask_array.shape[:2] or corrected.dtype != np.uint8:
        logger.info("[collar_mask_correction] skipped=True added_px=0 reason=validation_failed_shape_or_dtype")
        return base_mask, candidate_debug, protection_debug, {
            "side": "collar", "skipped": True, "added_px": 0,
            "reason": "validation_failed_shape_or_dtype",
        }

    if added_mask.any() and np.isin(parse_array[added_mask], PROTECTED_LABELS).any():
        logger.info("[collar_mask_correction] skipped=True added_px=0 reason=unsafe_protection_overlap")
        return base_mask, candidate_debug, protection_debug, {
            "side": "collar", "skipped": True, "added_px": 0,
            "reason": "unsafe_protection_overlap",
        }

    if added_mask.any():
        ys = np.where(added_mask)[0]
        if ys.max() > bottom_y + 1.0:
            logger.info("[collar_mask_correction] skipped=True added_px=0 reason=validation_failed_lower_body_reach")
            return base_mask, candidate_debug, protection_debug, {
                "side": "collar", "skipped": True, "added_px": 0,
                "reason": "validation_failed_lower_body_reach",
            }

    added_px = int(added_mask.sum())
    diag = {"side": "collar", "skipped": False, "added_px": added_px, "reason": "applied"}
    logger.info("[collar_mask_correction] skipped=False added_px=%d reason=applied", added_px)
    return corrected, candidate_debug, protection_debug, diag
