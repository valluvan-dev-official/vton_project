"""Preprocessing utilities: human parsing, DensePose, agnostic mask generation."""
from pathlib import Path
import numpy as np
from PIL import Image


def resize_and_pad(img: Image.Image, target_size: int = 512) -> Image.Image:
    """Resize keeping aspect ratio, then pad to square."""
    img.thumbnail((target_size, target_size), Image.LANCZOS)
    canvas = Image.new("RGB", (target_size, target_size), (255, 255, 255))
    offset = ((target_size - img.width) // 2, (target_size - img.height) // 2)
    canvas.paste(img, offset)
    return canvas


def generate_agnostic_mask(
    person_img: Image.Image,
    parsing_map: np.ndarray,
    torso_labels: tuple[int, ...] = (5, 6, 7),  # SCHP label IDs for upper-body garments
) -> Image.Image:
    """
    Zero out the torso region using a human-parsing segmentation map.
    Returns a binary PIL mask (L mode).
    """
    mask = np.zeros(parsing_map.shape, dtype=np.uint8)
    for label in torso_labels:
        mask[parsing_map == label] = 255
    return Image.fromarray(mask, mode="L")


def extract_densepose_features(image_path: str, output_path: str) -> str:
    """
    Placeholder for DensePose feature extraction.
    In production, call the DensePose detectron2 predictor here.
    """
    img = Image.open(image_path).convert("RGB")
    # Stub: save a grey image as a stand-in for UV coordinates
    dp = Image.fromarray(np.full((*np.array(img).shape[:2], 3), 128, dtype=np.uint8))
    Path(output_path).parent.mkdir(parents=True, exist_ok=True)
    dp.save(output_path)
    return output_path


def preprocess_pair(person_path: str, garment_path: str, out_dir: str, size: int = 512) -> dict:
    """Full preprocessing pipeline for one person/garment pair."""
    out = Path(out_dir)
    out.mkdir(parents=True, exist_ok=True)

    person = resize_and_pad(Image.open(person_path).convert("RGB"), size)
    garment = resize_and_pad(Image.open(garment_path).convert("RGB"), size)

    person_out = str(out / "person.jpg")
    garment_out = str(out / "garment.jpg")
    person.save(person_out)
    garment.save(garment_out)

    dp_out = str(out / "densepose.jpg")
    extract_densepose_features(person_out, dp_out)

    parsing_stub = np.zeros((size, size), dtype=np.uint8)
    mask = generate_agnostic_mask(person, parsing_stub)
    mask.save(str(out / "agnostic_mask.png"))

    return {"person": person_out, "garment": garment_out, "densepose": dp_out}
