"""
Correction for the IDM-VTON "bridged background gap" mask artifact.

get_mask_location()'s upper-body "hd" mask draws a thick, dilated line along
the shoulder->elbow->wrist OpenPose path and unions it into the inpainting
mask. When an arm is bent, that dilated line can cross the real, empty
background between the forearm and the torso — flipping parser-confirmed
background pixels to "editable". IDM-VTON then paints garment fabric into
that gap, producing a stray rectangular/flap-shaped artifact.

This module finds and removes *only* that specific bridged-background
component per arm, using parser labels + OpenPose geometry. It never
intersects the whole mask with a clothing label and never blanket-removes
background — only an enclosed component that is anatomically consistent with
"the gap next to a bent elbow", scaled off the person's own shoulder/arm
measurements (so it works at any resolution or body size).

Dependencies: numpy + opencv only — importable and unit-testable without
PIL, torch, or a loaded GPU model.
"""
from __future__ import annotations

import logging
from typing import Optional

import cv2
import numpy as np

logger = logging.getLogger(__name__)

# OpenPose COCO-18 keypoint indices (same convention used elsewhere in
# gpu_inference.py, e.g. the shoulder-scaling and half-sleeve carve-out code).
_NECK = 1
_ARM_JOINTS = {
    "right": {"shoulder": 2, "elbow": 3, "wrist": 4},
    "left":  {"shoulder": 5, "elbow": 6, "wrist": 7},
}

# Parser label used for "background" by the live IDM-VTON utils_mask.py.
BACKGROUND_LABEL = 0

# Tunables — all expressed as fractions of measured body geometry
# (shoulder width / upper-arm length), never as fixed pixel counts, so
# behaviour is resolution- and body-size-independent.
DEFAULT_STRAIGHT_ANGLE_DEG = 155.0   # elbow interior angle >= this => arm treated as straight
DEFAULT_MIN_AREA_FRAC      = 0.01    # component area vs. (shoulder_width * upper_arm_len)
DEFAULT_MAX_AREA_FRAC      = 0.9
DEFAULT_MIN_ROI_OVERLAP    = 0.5     # fraction of component inside the arm/torso wedge ROI
DEFAULT_ROI_BUFFER_FRAC    = 0.12    # ROI dilation, as a fraction of upper-arm length


def _joint(points: list, idx: int) -> Optional[tuple]:
    """Return (x, y) for keypoint `idx`, or None if missing/invalid/low-confidence.

    Mirrors the existing convention in gpu_inference.py: (0, 0) means "not
    detected". A third [x, y, confidence] element, if present, is honoured.
    """
    if points is None or idx >= len(points):
        return None
    p = points[idx]
    if len(p) < 2:
        return None
    x, y = float(p[0]), float(p[1])
    if x == 0.0 and y == 0.0:
        return None
    if len(p) >= 3 and p[2] is not None and float(p[2]) < 0.05:
        return None
    return (x, y)


def scale_keypoints(keypoints: dict, sx: float, sy: float) -> dict:
    """Return a copy of an OpenPose-style keypoints dict with every (x, y)
    scaled by (sx, sy). Any trailing elements (e.g. confidence) pass through
    unscaled."""
    pts = (keypoints or {}).get("pose_keypoints_2d", [])
    scaled = []
    for p in pts:
        if len(p) >= 2:
            scaled.append([p[0] * sx, p[1] * sy, *p[2:]])
        else:
            scaled.append(list(p))
    return {"pose_keypoints_2d": scaled}


def _bend_angle_degrees(shoulder: tuple, elbow: tuple, wrist: tuple) -> Optional[float]:
    """Interior angle at the elbow, in degrees. ~180 = straight arm, smaller = bent."""
    v1 = np.array(shoulder, dtype=np.float64) - np.array(elbow, dtype=np.float64)
    v2 = np.array(wrist, dtype=np.float64) - np.array(elbow, dtype=np.float64)
    n1, n2 = np.linalg.norm(v1), np.linalg.norm(v2)
    if n1 < 1e-6 or n2 < 1e-6:
        return None
    cos_a = float(np.clip(np.dot(v1, v2) / (n1 * n2), -1.0, 1.0))
    return float(np.degrees(np.arccos(cos_a)))


def _find_arm_torso_gap_component(
    parse_array: np.ndarray,
    mask_array: np.ndarray,
    side: str,
    shoulder, elbow, wrist,
    torso_center_x: float,
    shoulder_width: float,
    straight_angle_thresh: float = DEFAULT_STRAIGHT_ANGLE_DEG,
    min_area_frac: float = DEFAULT_MIN_AREA_FRAC,
    max_area_frac: float = DEFAULT_MAX_AREA_FRAC,
    min_roi_overlap: float = DEFAULT_MIN_ROI_OVERLAP,
    roi_buffer_frac: float = DEFAULT_ROI_BUFFER_FRAC,
):
    """Find the (at most one) bridged-background connected component for one
    arm. Returns (component_bool_mask_or_None, diagnostics_dict)."""
    diag = {"side": side, "skipped": True, "reason": None, "removed_px": 0}

    if shoulder is None or elbow is None or wrist is None:
        diag["reason"] = "missing_or_invalid_joints"
        return None, diag

    upper_arm_len = float(np.linalg.norm(np.array(elbow) - np.array(shoulder)))
    forearm_len   = float(np.linalg.norm(np.array(wrist) - np.array(elbow)))
    if upper_arm_len < 1e-3 or forearm_len < 1e-3 or shoulder_width < 1e-3:
        diag["reason"] = "degenerate_geometry"
        return None, diag

    angle = _bend_angle_degrees(shoulder, elbow, wrist)
    if angle is None:
        diag["reason"] = "degenerate_geometry"
        return None, diag
    if angle >= straight_angle_thresh:
        diag["reason"] = f"arm_straight(angle={angle:.1f}deg)"
        return None, diag

    H, W = mask_array.shape[:2]

    # ROI = convex hull of the arm polyline plus its projection onto the
    # torso centerline — the wedge where a bent-elbow background gap lives.
    torso_at_elbow = (torso_center_x, elbow[1])
    torso_at_wrist = (torso_center_x, wrist[1])
    roi_pts = np.array([shoulder, elbow, wrist, torso_at_wrist, torso_at_elbow], dtype=np.float32)
    hull = cv2.convexHull(roi_pts)
    roi_mask = np.zeros((H, W), dtype=np.uint8)
    cv2.fillConvexPoly(roi_mask, hull.astype(np.int32), 1)

    buf = max(1, int(round(roi_buffer_frac * upper_arm_len)))
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (2 * buf + 1, 2 * buf + 1))
    roi_mask = cv2.dilate(roi_mask, kernel, iterations=1)

    bg = (parse_array == BACKGROUND_LABEL)
    editable = (mask_array > 0)
    candidate = (bg & editable).astype(np.uint8)

    if not candidate.any():
        diag["reason"] = "no_background_in_mask"
        return None, diag

    num_labels, labels, stats, _ = cv2.connectedComponentsWithStats(candidate, connectivity=8)

    ref_area = max(1.0, shoulder_width * upper_arm_len)
    min_area = min_area_frac * ref_area
    max_area = max_area_frac * ref_area

    best = None
    notes = []
    for lbl in range(1, num_labels):
        x, y, w, h, area = stats[lbl]
        if area < min_area:
            notes.append(f"#{lbl} rejected too_small(area={area:.0f}<{min_area:.0f})")
            continue
        if area > max_area:
            notes.append(f"#{lbl} rejected too_large(area={area:.0f}>{max_area:.0f})")
            continue
        # Reject components touching the image border — those are the open
        # background around the person, not an enclosed arm/torso pocket.
        if x <= 0 or y <= 0 or x + w >= W or y + h >= H:
            notes.append(f"#{lbl} rejected touches_border")
            continue
        comp_mask = (labels == lbl)
        overlap = float((comp_mask & (roi_mask > 0)).sum()) / float(comp_mask.sum())
        if overlap < min_roi_overlap:
            notes.append(f"#{lbl} rejected low_roi_overlap({overlap:.2f}<{min_roi_overlap})")
            continue
        notes.append(f"#{lbl} accepted area={area:.0f} overlap={overlap:.2f}")
        if best is None or area > best[1]:
            best = (comp_mask, area)

    if best is None:
        diag["reason"] = "no_eligible_component" + ("; " + "; ".join(notes) if notes else "")
        return None, diag

    diag["skipped"] = False
    diag["removed_px"] = int(best[1])
    diag["reason"] = "gap_component_removed; " + "; ".join(notes)
    return best[0], diag


def correct_agnostic_mask_gap(
    parse_array: np.ndarray,
    mask_array: np.ndarray,
    keypoints: dict,
    **kwargs,
):
    """Remove parser-confirmed background pixels wrongly bridged into the
    inpainting mask by a bent arm, independently per side.

    parse_array : 2D int array of parser labels, shape (H, W).
    mask_array  : 2D array, same (H, W) — nonzero = editable/inpaint.
    keypoints   : {"pose_keypoints_2d": [[x, y] or [x, y, conf], ...]},
                  in the SAME pixel coordinate space as the two arrays above
                  (scale with `scale_keypoints` first if needed).

    Returns (corrected_mask: np.uint8 {0,255}, diagnostics: list[dict],
             gap_debug_mask: np.uint8 {0,255}).
    """
    if parse_array.shape[:2] != mask_array.shape[:2]:
        raise ValueError(
            f"parse_array shape {parse_array.shape[:2]} != mask_array shape {mask_array.shape[:2]}"
        )

    pts = (keypoints or {}).get("pose_keypoints_2d", [])
    r_shoulder = _joint(pts, _ARM_JOINTS["right"]["shoulder"])
    l_shoulder = _joint(pts, _ARM_JOINTS["left"]["shoulder"])

    corrected = (mask_array > 0).astype(np.uint8) * 255
    gap_debug = np.zeros(mask_array.shape[:2], dtype=np.uint8)
    diagnostics = []

    torso_center_x = None
    shoulder_width = None
    if r_shoulder is not None and l_shoulder is not None:
        torso_center_x = (r_shoulder[0] + l_shoulder[0]) / 2.0
        shoulder_width = float(np.linalg.norm(np.array(r_shoulder) - np.array(l_shoulder)))

    sides = {
        "right": (r_shoulder, _joint(pts, _ARM_JOINTS["right"]["elbow"]), _joint(pts, _ARM_JOINTS["right"]["wrist"])),
        "left":  (l_shoulder, _joint(pts, _ARM_JOINTS["left"]["elbow"]),  _joint(pts, _ARM_JOINTS["left"]["wrist"])),
    }

    for side, (shoulder, elbow, wrist) in sides.items():
        if torso_center_x is None or shoulder_width is None:
            diag = {"side": side, "skipped": True,
                     "reason": "missing_shoulder_pair_for_torso_reference", "removed_px": 0}
            diagnostics.append(diag)
            logger.info("[mask_gap_correction] side=%s skipped=True removed_px=0 reason=%s",
                        side, diag["reason"])
            continue

        comp, diag = _find_arm_torso_gap_component(
            parse_array, corrected, side, shoulder, elbow, wrist,
            torso_center_x, shoulder_width, **kwargs,
        )
        diagnostics.append(diag)
        logger.info("[mask_gap_correction] side=%s skipped=%s removed_px=%d reason=%s",
                    diag["side"], diag["skipped"], diag["removed_px"], diag["reason"])
        if comp is not None:
            corrected[comp] = 0
            gap_debug[comp] = 255

    # Explicit binary threshold — guarantees {0, 255} output regardless of
    # any intermediate dtype/bool artifacts above.
    corrected = np.where(corrected > 127, 255, 0).astype(np.uint8)
    return corrected, diagnostics, gap_debug
