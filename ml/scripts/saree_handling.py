"""Sleeve/arm-mask handling for sarees.

Kept separate from dress_sleeve_detection.py (gowns/frocks/salwar-suits) and
gpu_inference.py's _detect_sleeve_type (shirts). A saree has no fitted sleeve
of its own — the drape itself (pallu, pleats) covers the torso and one
shoulder, and any sleeve on the body comes from the BLOUSE worn underneath,
which is a separate garment not usually part of the saree product photo at
all. Running a shirt/gown-style "is there fabric at the elbow" heuristic
against a saree photo is checking for something that isn't there by design,
and — per the same elbow/wrist circle-punch mechanism documented in
gpu_inference.py's "half"-sleeve branch — carving a hole there would let the
original photo's bare arm show through, which is simply wrong for a draped
garment where the arms are meant to stay as the base agnostic mask predicts.
"""

from PIL import Image


def detect_saree_sleeve_type(garment_pil: Image.Image) -> str:
    """Sarees always resolve to 'full' — i.e. never punch an elbow/wrist
    hole in the mask. There is no sleeve-length concept to detect from a
    saree photo; the caller's arm coverage should come entirely from
    get_mask_location()'s own base agnostic-mask prediction (plus whatever
    blouse sleeve is actually visible on the PERSON's own photo), not from
    guessing sleeve length off the drape image.
    """
    return "full"
