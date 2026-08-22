"""Sleeve detection for salwar suits (kurta top + salwar/churidar bottom,
sold and tried on as one "dresses"-category garment_dress product).

Kept separate from dress_sleeve_detection.py (gowns/frocks) and
saree_handling.py (sarees, which skip sleeve detection entirely) even though
the underlying heuristic is currently the same multi-band approach as gowns
— salwar-suit kurtas share the same failure mode as a gown/Anarkali (a
kurta's sleeve can be loose/A-line and not uniformly wide top-to-bottom, so
a single fixed height-band check can miss it), but this stays a dedicated
module/function per garment type so each can be tuned independently as
real-world results come in (e.g. if kurta sleeves turn out to need a
different band range than gowns), rather than three subtypes silently
sharing one function that changes underneath all of them at once.
"""

import numpy as np
from PIL import Image

# Multiple height bands instead of one fixed zone — same rationale as
# dress_sleeve_detection.py's gown handling: a kurta sleeve is not
# guaranteed to be uniform-width top-to-bottom.
_BANDS = [(0.25, 0.40), (0.45, 0.65), (0.70, 0.85)]


def detect_salwar_suit_sleeve_type(garment_pil: Image.Image) -> str:
    """Detect sleeve length from a salwar-suit (kurta top) garment image.

    Returns 'half' or 'full'. Treated as 'full' if either side has garment
    fabric in ANY of the checked height bands.
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
