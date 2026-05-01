"""
Batch preprocessing script.
Reads raw person/garment images from ml/data/raw/ and writes
processed outputs to ml/data/processed/.

Usage:
    python ml/scripts/preprocess_all.py
"""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.data.preprocess import preprocess_pair
from src.utils.logging import get_logger

logger = get_logger("preprocess_all")

RAW_PERSON_DIR = Path("ml/data/raw/person")
RAW_GARMENT_DIR = Path("ml/data/raw/garment")
PROCESSED_DIR = Path("ml/data/processed")


def main():
    person_images = sorted(RAW_PERSON_DIR.glob("*.jpg")) + sorted(RAW_PERSON_DIR.glob("*.png"))
    garment_images = sorted(RAW_GARMENT_DIR.glob("*.jpg")) + sorted(RAW_GARMENT_DIR.glob("*.png"))

    pairs = list(zip(person_images, garment_images))
    logger.info(f"Found {len(pairs)} person/garment pairs")

    for idx, (person_path, garment_path) in enumerate(pairs):
        out_dir = PROCESSED_DIR / f"{idx:06d}"
        result = preprocess_pair(str(person_path), str(garment_path), str(out_dir))
        logger.info(f"[{idx+1}/{len(pairs)}] Processed → {out_dir}")

    logger.info("Preprocessing complete.")


if __name__ == "__main__":
    main()
