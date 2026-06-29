"""
ml_inference — Kaggle-free, SageMaker-ready DCI-VTON inference package.

This package is a structural refactor of `ml/scripts/gpu_inference.py`.
The model math is unchanged; it is split into:

    model_loader.py  — load every model once (ModelBundle)
    preprocess.py    — segmentation, mask, densepose, warp
    postprocess.py   — color correction + composite
    predictor.py     — VTONPredictor: orchestrates the full pipeline
    inference.py     — SageMaker handlers (model_fn / input_fn / predict_fn / output_fn)
"""
from .model_loader import ModelBundle, load_all_models, SIZE
from .predictor import VTONPredictor

__all__ = ["ModelBundle", "load_all_models", "VTONPredictor", "SIZE"]
