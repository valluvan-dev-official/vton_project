"""Sleeve-length detection for dresses/gowns/sarees.

Kept separate from GPUInferenceEngine._detect_sleeve_type (the upper_body
shirt logic in gpu_inference.py), which stays untouched. That single fixed
45-65%-height band check works for a plain shirt's uniform-width sleeve, but
fails for the voluminous/puffed sleeve shapes common in dresses, gowns, and
sarees — e.g. a bishop/balloon sleeve that's wide at the shoulder (~20-40%
band), narrows sharply mid-arm, then gathers at a fitted cuff (~70-85%
band). A single band landing in that narrow gap misreads a full-length
sleeve as "half", which then triggers an elbow/wrist mask circle-punch in
the caller, carving a hard hole that lets the ORIGINAL photo's bare skin
show through as a pale patch where gathered/puffed sleeve fabric should
have been (2026-08-22 gown/saree white-elbow-patch incident).
"""

import numpy as np
from PIL import Image

# Multiple height bands instead of one fixed zone, so a puffed/gathered
# sleeve shape is caught wherever its side fabric actually sits.
_BANDS = [(0.25, 0.40), (0.45, 0.65), (0.70, 0.85)]


def detect_dress_sleeve_type(garment_pil: Image.Image) -> str:
    """Detect sleeve length from a dress/gown/saree garment image.

    Returns 'half' or 'full'. Treated as 'full' if either side has garment
    fabric in ANY of the checked height bands, catching puffed/gathered/
    tapered sleeve shapes that a single band would miss, while a genuinely
    short/cap/sleeveless garment still has no side fabric in any of them.
    """
    img = np.array(garment_pil.convert("RGB"))
    h, w = img.shape[:2]

    white = (img[:, :, 0] > 240) & (img[:, :, 1] > 240) & (img[:, :, 2] > 240)
    garment = ~white

    rows = np.any(garment, axis=1)
    if not rows.any():
        return "half"

    top = int(np.argmax(rows))
    bottom = int(h - np.argmax(rows[::-1]) - 1)
    g_h = bottom - top
    if g_h < 10:
        return "half"

    left_sleeve = right_sleeve = False
    for band_start, band_end in _BANDS:
        mid_top = top + int(g_h * band_start)
        mid_bot = top + int(g_h * band_end)
        mid_section = garment[mid_top:mid_bot, :]
        if mid_section[:, :w // 4].any():
            left_sleeve = True
        if mid_section[:, 3 * w // 4:].any():
            right_sleeve = True

    return "full" if (left_sleeve and right_sleeve) else "half"
