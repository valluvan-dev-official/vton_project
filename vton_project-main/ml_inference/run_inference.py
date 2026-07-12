"""
run_inference.py — Standalone CLI entry point for running the DCI-VTON
try-on pipeline outside of SageMaker (e.g. local GPU box, EC2, container
shell), without going through model_fn/input_fn/predict_fn/output_fn.

Usage:
    python run_inference.py <person_image_path> <garment_image_path> <output_image_path>
"""
from __future__ import annotations

import os
import sys

MODEL_DIR = "/workspace/model"


def _fail(message: str) -> None:
    print(f"ERROR: {message}", file=sys.stderr)
    sys.exit(1)


def main() -> None:
    if len(sys.argv) != 4:
        _fail(
            "Usage: python run_inference.py <person_image_path> "
            "<garment_image_path> <output_image_path>"
        )

    person_path, garment_path, output_path = sys.argv[1:4]

    if not os.path.isfile(person_path):
        _fail(f"person image not found: {person_path}")
    if not os.path.isfile(garment_path):
        _fail(f"garment image not found: {garment_path}")
    if not os.path.isdir(MODEL_DIR):
        _fail(f"model directory not found: {MODEL_DIR}")

    from ml_inference.predictor import VTONPredictor

    print("Loading model...")
    predictor = VTONPredictor.from_pretrained(weights_dir=MODEL_DIR, device="cuda")

    print("Running inference...")
    predictor.run(person_path, garment_path, output_path)

    print(f"Saved output to {output_path}")


if __name__ == "__main__":
    main()
