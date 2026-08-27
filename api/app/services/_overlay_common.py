"""
Shared building blocks for the CPU landmark-overlay accessory engines
(glasses, wrist, earring, ring, hat, necklace).

Every one of those engines does the same four things:
  1. find landmarks on the person photo (MediaPipe Face Mesh or Hands),
  2. find matching anchor points on the product cutout's own alpha
     silhouette (so we never assume a fixed product-photo framing),
  3. warp the cutout so its anchors land on the person's anchors, and
  4. alpha-composite it (optionally with a soft contact shadow) and save.

Steps 2-4 are identical maths regardless of accessory, so they live here
instead of being copy-pasted into every engine (which is how the first
wrist/glasses/handbag pass drifted out of sync). Only step 1 -- which
landmarks, and how they map to the product's anchors -- is per-engine.
"""
from __future__ import annotations

import logging

import cv2
import numpy as np
from PIL import Image, ImageFilter

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Product-cutout anchor derivation
# ---------------------------------------------------------------------------
def silhouette_anchors(accessory_rgba: np.ndarray) -> tuple[
    tuple[float, float], tuple[float, float], tuple[float, float]
]:
    """(left, right, top_center) points from the cutout's own alpha content.

    left/right = centroid of each half of the non-transparent silhouette
    (split at its horizontal midpoint); top = topmost opaque pixel in the
    central vertical band. For a glasses cutout that's left lens / right lens
    / bridge; for a bag it's the two sides / the handle; for a hat it's the
    two brim edges / the crown -- all regardless of how the product photo is
    cropped or padded.
    """
    alpha = accessory_rgba[:, :, 3]
    ys, xs = np.nonzero(alpha > 10)
    if len(xs) == 0:
        raise ValueError("Accessory image has no visible (non-transparent) content.")

    x_min, x_max = float(xs.min()), float(xs.max())
    y_min, y_max = float(ys.min()), float(ys.max())
    mid_x = (x_min + x_max) / 2

    left_mask = xs < mid_x
    right_mask = ~left_mask
    left_point = (
        (float(xs[left_mask].mean()), float(ys[left_mask].mean()))
        if left_mask.any() else (x_min, (y_min + y_max) / 2)
    )
    right_point = (
        (float(xs[right_mask].mean()), float(ys[right_mask].mean()))
        if right_mask.any() else (x_max, (y_min + y_max) / 2)
    )

    span = max(x_max - x_min, 1.0)
    central = (xs > mid_x - span * 0.15) & (xs < mid_x + span * 0.15)
    top_point = (mid_x, float(ys[central].min())) if central.any() else (mid_x, y_min)
    return left_point, right_point, top_point


def silhouette_bbox(accessory_rgba: np.ndarray) -> tuple[int, int, int, int]:
    """(x_min, y_min, x_max, y_max) of the cutout's opaque content."""
    alpha = accessory_rgba[:, :, 3]
    ys, xs = np.nonzero(alpha > 10)
    if len(xs) == 0:
        raise ValueError("Accessory image has no visible (non-transparent) content.")
    return int(xs.min()), int(ys.min()), int(xs.max()), int(ys.max())


# ---------------------------------------------------------------------------
# Warps
# ---------------------------------------------------------------------------
def affine_from_3pt(
    src_pts: np.ndarray, dst_pts: np.ndarray, accessory_np: np.ndarray,
    out_w: int, out_h: int,
) -> np.ndarray:
    """Full 6-DOF affine (rotation + independent x/y scale + shear) from a
    3-point correspondence. Use when the product isn't radially symmetric
    (glasses, hat, bag) and a turned/tilted head introduces shear a rigid
    rotate+scale can't express.
    """
    M = cv2.getAffineTransform(
        np.float32(src_pts[:3]), np.float32(dst_pts[:3])
    )
    return cv2.warpAffine(
        accessory_np, M, (out_w, out_h),
        flags=cv2.INTER_LANCZOS4,
        borderMode=cv2.BORDER_CONSTANT, borderValue=(0, 0, 0, 0),
    )


def similarity_from_2pt(
    src_a, src_b, dst_a, dst_b, accessory_np: np.ndarray,
    out_w: int, out_h: int, extra_rotation_deg: float = 0.0,
) -> np.ndarray:
    """4-DOF similarity (uniform scale + one rotation + translate) mapping the
    segment src_a->src_b onto dst_a->dst_b. Use for objects that ARE
    symmetric about one axis (watch band, ring) where shear would only
    introduce distortion.

    extra_rotation_deg is applied about dst_a's midpoint after the fit --
    e.g. +90 to turn a product shot "upright" onto a horizontal wrist axis.
    """
    src_a = np.asarray(src_a, np.float64)
    src_b = np.asarray(src_b, np.float64)
    dst_a = np.asarray(dst_a, np.float64)
    dst_b = np.asarray(dst_b, np.float64)

    src_vec = src_b - src_a
    dst_vec = dst_b - dst_a
    src_len = float(np.hypot(*src_vec)) or 1.0
    dst_len = float(np.hypot(*dst_vec)) or 1.0
    scale = dst_len / src_len

    ang = np.arctan2(dst_vec[1], dst_vec[0]) - np.arctan2(src_vec[1], src_vec[0])
    ang += np.radians(extra_rotation_deg)
    cos_a, sin_a = np.cos(ang) * scale, np.sin(ang) * scale

    # 2x3 affine: rotate+scale about src_a, then translate so src_a -> dst_a.
    R = np.array([[cos_a, -sin_a], [sin_a, cos_a]], np.float64)
    t = dst_a - R @ src_a
    M = np.array([[R[0, 0], R[0, 1], t[0]], [R[1, 0], R[1, 1], t[1]]], np.float64)
    return cv2.warpAffine(
        accessory_np, M.astype(np.float32), (out_w, out_h),
        flags=cv2.INTER_LANCZOS4,
        borderMode=cv2.BORDER_CONSTANT, borderValue=(0, 0, 0, 0),
    )


# ---------------------------------------------------------------------------
# Post-warp cleanup + compositing
# ---------------------------------------------------------------------------
def boost_alpha(warped: np.ndarray, gamma: float = 0.45) -> np.ndarray:
    """Push partially-transparent pixels toward opaque with a gamma curve.

    Thin accessories (wire-frame glasses, fine chains, straps) are mostly
    anti-aliased edge pixels with partial alpha; the warp's interpolation --
    especially scaling a small product photo up onto a large face -- fades
    them further toward invisible. gamma<1 restores them while leaving fully
    transparent (0) and fully opaque (255) pixels untouched.
    """
    a = warped[:, :, 3].astype(np.float32) / 255.0
    a = np.power(a, gamma)
    warped = warped.copy()
    warped[:, :, 3] = np.clip(a * 255.0, 0, 255).astype(np.uint8)
    return warped


def composite(
    person_path: str, warped_rgba: np.ndarray, output_path: str,
    shadow: bool = False, shadow_offset: tuple[int, int] = (0, 6),
    shadow_blur: int = 9, shadow_opacity: float = 0.28,
) -> None:
    """Alpha-composite the warped cutout onto the person photo and save JPEG.

    When shadow=True a blurred, offset copy of the cutout's alpha is laid
    down first as a soft contact shadow so the accessory reads as sitting
    ON the person rather than pasted above them.
    """
    person = Image.open(person_path).convert("RGBA")
    warped_pil = Image.fromarray(warped_rgba, mode="RGBA")

    if shadow:
        a = Image.fromarray(warped_rgba[:, :, 3], mode="L").filter(
            ImageFilter.GaussianBlur(shadow_blur)
        )
        shadow_layer = Image.new("RGBA", person.size, (0, 0, 0, 0))
        tinted = Image.new("RGBA", person.size, (0, 0, 0, 255))
        a = a.point(lambda v: int(v * shadow_opacity))
        shadow_layer.paste(tinted, shadow_offset, a)
        person.alpha_composite(shadow_layer)

    person.alpha_composite(warped_pil)
    person.convert("RGB").save(output_path, "JPEG", quality=95)


# ---------------------------------------------------------------------------
# Geometry helpers
# ---------------------------------------------------------------------------
def roll_deg(image_left_eye, image_right_eye) -> float:
    """In-plane head tilt from the two eye corners, degrees (0 = level)."""
    dx = image_right_eye[0] - image_left_eye[0]
    dy = image_right_eye[1] - image_left_eye[1]
    return float(np.degrees(np.arctan2(dy, dx)))


def rotate_about(point, origin, deg: float):
    """Rotate `point` about `origin` by `deg` (image coords, y-down)."""
    ang = np.radians(deg)
    px, py = point[0] - origin[0], point[1] - origin[1]
    c, s = np.cos(ang), np.sin(ang)
    return (origin[0] + px * c - py * s, origin[1] + px * s + py * c)
