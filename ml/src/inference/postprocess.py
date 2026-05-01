"""Post-processing utilities: blending, background restoration, sharpening."""
import numpy as np
from PIL import Image, ImageFilter


def blend_result(person_img: Image.Image, result_img: Image.Image, mask: Image.Image) -> Image.Image:
    """
    Blend result onto person using a soft mask so unchanged regions remain pixel-perfect.
    mask: 'L' mode PIL image — white = use result, black = use person.
    """
    person_np = np.array(person_img.convert("RGBA"))
    result_np = np.array(result_img.convert("RGBA"))
    mask_np = np.array(mask.convert("L")) / 255.0

    blended = (result_np * mask_np[..., None] + person_np * (1 - mask_np[..., None])).astype(np.uint8)
    return Image.fromarray(blended).convert("RGB")


def sharpen(img: Image.Image, radius: float = 1.0, percent: int = 120, threshold: int = 3) -> Image.Image:
    return img.filter(ImageFilter.UnsharpMask(radius=radius, percent=percent, threshold=threshold))


def resize_to_original(result_img: Image.Image, original_size: tuple[int, int]) -> Image.Image:
    return result_img.resize(original_size, Image.LANCZOS)
