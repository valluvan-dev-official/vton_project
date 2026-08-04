"""
Pure-arithmetic aspect-ratio-preserving letterbox geometry.

No PIL/numpy/torch/cv2 imports — intentionally dependency-free so
`compute_letterbox_geometry` can be unit tested (and reused) in any
environment, including ones without the ML stack installed.
"""
from typing import NamedTuple


class LetterboxTransform(NamedTuple):
    """Geometry needed to invert a letterbox resize back to the original size."""
    orig_w: int
    orig_h: int
    new_w: int
    new_h: int
    pad_x: int
    pad_y: int


def compute_letterbox_geometry(orig_w: int, orig_h: int,
                                target_w: int, target_h: int) -> LetterboxTransform:
    """Scale (orig_w, orig_h) to fit inside (target_w, target_h) preserving
    aspect ratio, then center it via padding.

    Works for any input size — square, portrait, or landscape — nothing here
    is hard-coded to a specific job's dimensions.
    """
    if orig_w <= 0 or orig_h <= 0 or target_w <= 0 or target_h <= 0:
        raise ValueError(
            f"Invalid dimensions: orig=({orig_w},{orig_h}) target=({target_w},{target_h})"
        )
    scale = min(target_w / orig_w, target_h / orig_h)
    # Safeguard: round() can push the fitted axis 1px past the canvas on some
    # fractional scales — clamp so the resized image never exceeds target.
    new_w = min(target_w, max(1, round(orig_w * scale)))
    new_h = min(target_h, max(1, round(orig_h * scale)))
    pad_x = (target_w - new_w) // 2
    pad_y = (target_h - new_h) // 2
    return LetterboxTransform(orig_w, orig_h, new_w, new_h, pad_x, pad_y)
