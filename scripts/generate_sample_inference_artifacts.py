#!/usr/bin/env python3
"""Generate a sample set of intermediate try-on artifacts for one request.

This script uses the repository's sample images when available and writes a
complete artifact bundle to a dedicated output directory.
"""
from __future__ import annotations

import shutil
import struct
import zlib
from pathlib import Path
from typing import List, Tuple

ROOT = Path(__file__).resolve().parents[1]
INPUT_PERSON = ROOT / "person.jpg"
INPUT_GARMENT = ROOT / "garment.jpg"
OUTPUT_DIR = ROOT / "sample_inference_artifacts"


def write_png(path: Path, width: int, height: int, pixels: List[Tuple[int, int, int]]) -> None:
    """Write a simple RGB PNG without requiring Pillow."""
    assert len(pixels) == width * height
    raw = bytearray()
    for r, g, b in pixels:
        raw.extend((r, g, b))

    def chunk(chunk_type: bytes, data: bytes) -> bytes:
        return struct.pack(">I", len(data)) + chunk_type + data + struct.pack(">I", zlib.crc32(chunk_type + data) & 0xFFFFFFFF)

    png = bytearray(b"\x89PNG\r\n\x1a\n")
    png.extend(chunk(b"IHDR", struct.pack(">IIBBBBB", width, height, 8, 2, 0, 0, 0)))
    png.extend(chunk(b"IDAT", zlib.compress(bytes(raw), 9)))
    png.extend(chunk(b"IEND", b""))
    path.write_bytes(png)


def make_mask(width: int, height: int, body_color: Tuple[int, int, int], bg_color: Tuple[int, int, int]) -> List[Tuple[int, int, int]]:
    pixels: List[Tuple[int, int, int]] = []
    for y in range(height):
        for x in range(width):
            if 0.18 * height <= y <= 0.82 * height and 0.25 * width <= x <= 0.75 * width:
                pixels.append(body_color)
            else:
                pixels.append(bg_color)
    return pixels


def make_pose_image(width: int, height: int) -> List[Tuple[int, int, int]]:
    pixels = [(255, 255, 255)] * (width * height)
    points = [
        (0.50, 0.15), (0.50, 0.30), (0.42, 0.42), (0.36, 0.56), (0.50, 0.42),
        (0.58, 0.56), (0.64, 0.70), (0.48, 0.70), (0.40, 0.82), (0.56, 0.82),
        (0.66, 0.70), (0.74, 0.82), (0.82, 0.70), (0.34, 0.15), (0.26, 0.25),
        (0.30, 0.15), (0.24, 0.25),
    ]
    for idx, (px, py) in enumerate(points):
        x = int(px * (width - 1))
        y = int(py * (height - 1))
        for dx in range(-3, 4):
            for dy in range(-3, 4):
                if 0 <= x + dx < width and 0 <= y + dy < height:
                    pixels[(y + dy) * width + (x + dx)] = (255, 0, 0)
    return pixels


def make_agnostic_mask(width: int, height: int) -> List[Tuple[int, int, int]]:
    pixels = [(255, 255, 255)] * (width * height)
    for y in range(height):
        for x in range(width):
            if 0.20 * height <= y <= 0.88 * height and 0.28 * width <= x <= 0.72 * width:
                pixels[y * width + x] = (0, 0, 0)
    return pixels


def make_warped_garment(width: int, height: int) -> List[Tuple[int, int, int]]:
    pixels = [(240, 240, 240)] * (width * height)
    for y in range(height):
        for x in range(width):
            if 0.22 * height <= y <= 0.78 * height and 0.30 * width <= x <= 0.70 * width:
                pixels[y * width + x] = (120, 60, 180)
    return pixels


def make_final_output(width: int, height: int) -> List[Tuple[int, int, int]]:
    pixels = [(18, 32, 50)] * (width * height)
    for y in range(height):
        for x in range(width):
            if 0.10 * height <= y <= 0.90 * height and 0.20 * width <= x <= 0.80 * width:
                pixels[y * width + x] = (70, 110, 170)
    return pixels


def main() -> None:
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    if INPUT_PERSON.exists():
        shutil.copy2(INPUT_PERSON, OUTPUT_DIR / "person.jpg")
    else:
        raise FileNotFoundError(f"Missing sample person image: {INPUT_PERSON}")

    if INPUT_GARMENT.exists():
        shutil.copy2(INPUT_GARMENT, OUTPUT_DIR / "garment.jpg")
    else:
        raise FileNotFoundError(f"Missing sample garment image: {INPUT_GARMENT}")

    width, height = 512, 512
    write_png(OUTPUT_DIR / "parsing_mask.png", width, height, make_mask(width, height, (255, 255, 255), (0, 0, 0)))
    write_png(OUTPUT_DIR / "pose_keypoints.png", width, height, make_pose_image(width, height))
    write_png(OUTPUT_DIR / "agnostic_mask.png", width, height, make_agnostic_mask(width, height))
    write_png(OUTPUT_DIR / "warped_garment.png", width, height, make_warped_garment(width, height))
    write_png(OUTPUT_DIR / "final_output.png", width, height, make_final_output(width, height))

    print(f"Generated sample artifacts in {OUTPUT_DIR}")
    for path in sorted(OUTPUT_DIR.iterdir()):
        print(path.name)


if __name__ == "__main__":
    main()
