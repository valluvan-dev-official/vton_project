# ── DCI-VTON SageMaker Async Inference image ─────────────────────────────────
#
# Base: AWS Deep Learning Container for SageMaker PyTorch *inference*.
# Pulled from the private DLC registry (763104351884) — must log in first:
#   aws ecr get-login-password --region $REGION | \
#     docker login --username AWS --password-stdin \
#     763104351884.dkr.ecr.$REGION.amazonaws.com
#
# Tag verified to exist in ap-south-1 via:
#   aws ecr describe-images --registry-id 763104351884 \
#     --region ap-south-1 --repository-name pytorch-inference \
#     --image-ids imageTag=2.1.0-gpu-py310-cu118-ubuntu20.04-sagemaker
#
# Ships: CUDA 11.8 + cuDNN + PyTorch 2.1 + TorchServe +
#        sagemaker-pytorch-inference toolkit (model_fn/input_fn/predict_fn/output_fn).
#        The `serve` entrypoint is provided by the base image.
#
# Build (run build_and_push.ps1 which handles the login automatically):
#   .\build_and_push.ps1 -AccountId 133946906587 -Region ap-south-1
#
ARG REGION=ap-south-1
FROM 763104351884.dkr.ecr.${REGION}.amazonaws.com/pytorch-inference:2.1.0-gpu-py310-cu118-ubuntu20.04-sagemaker

ENV DEBIAN_FRONTEND=noninteractive \
    PYTHONUNBUFFERED=1 \
    # Pre-baked repo + HF cache locations (avoid network at cold start)
    VTON_REPOS_DIR=/opt/ml/repos \
    VTON_WORKSPACE=/tmp/vton_workspace \
    HF_HOME=/opt/ml/hf_cache \
    TRANSFORMERS_CACHE=/opt/ml/hf_cache \
    # Detectron2 builds CUDA ops — target common SageMaker GPUs
    # (T4=7.5, A10G=8.6, V100=7.0, L4=8.9)
    TORCH_CUDA_ARCH_LIST="7.0;7.5;8.0;8.6;8.9" \
    FORCE_CUDA=1

# ── System deps ──
RUN apt-get update && apt-get install -y --no-install-recommends \
        libgl1 libglib2.0-0 git build-essential ninja-build curl \
    && rm -rf /var/lib/apt/lists/*

# ── Bake third-party source repos at build time (cloned before pip install
#    so this layer is not invalidated by requirements.txt changes, and so
#    the detectron2 source tree is present for the wheel install step below) ──
RUN mkdir -p ${VTON_REPOS_DIR} \
    && git clone --depth=1 https://github.com/bcmi/DCI-VTON-Virtual-Try-On.git ${VTON_REPOS_DIR}/DCI-VTON-Virtual-Try-On \
    && git clone --depth=1 https://github.com/CompVis/taming-transformers.git   ${VTON_REPOS_DIR}/taming-transformers \
    && git clone --depth=1 https://github.com/geyuying/PF-AFN.git               ${VTON_REPOS_DIR}/PF-AFN \
    && git clone --depth=1 https://github.com/facebookresearch/detectron2.git   ${VTON_REPOS_DIR}/detectron2_repo

# ── Python deps (detectron2 excluded — see next step) ──
COPY requirements.txt /tmp/requirements.txt
RUN pip install --no-cache-dir -r /tmp/requirements.txt

# ── Detectron2: build and install from the already-cloned local source tree.
#    No official prebuilt wheel exists for PyTorch 2.x (last release: v0.6, 2021).
#    The editable install (-e) compiles the detectron2._C CUDA extension in-place,
#    which DefaultPredictor requires. The same source tree also provides:
#      - projects/DensePose/ (added to sys.path by model_loader.py::load_densepose)
#      - projects/DensePose/configs/*.yaml (read by cfg.merge_from_file at startup)
#    Both the compiled extension and the source tree layout are required. ──
RUN pip install --no-cache-dir -e ${VTON_REPOS_DIR}/detectron2_repo

# ── Warm the SegFormer weights into the image so cold start is offline ──
RUN python -c "from transformers import SegformerImageProcessor, SegformerForSemanticSegmentation; \
    SegformerImageProcessor.from_pretrained('mattmdjaga/segformer_b2_clothes'); \
    SegformerForSemanticSegmentation.from_pretrained('mattmdjaga/segformer_b2_clothes')"

# ── Install the ml_inference package ─────────────────────────────────────────
#
#  1. Copy the entire ml_inference/ build context (which contains setup.py
#     and the ml_inference/ sub-package) to a staging directory.
#  2. pip install -e installs it into site-packages so that anywhere inside
#     the container `import ml_inference` resolves correctly.
#  3. Copy inference.py (the SageMaker SAGEMAKER_PROGRAM entry point) to
#     /opt/ml/code/ where the toolkit expects it.
#
#  Layout after this block:
#    /tmp/ml_inference_src/          <- package source (editable install target)
#      setup.py
#      inference.py
#      ml_inference/
#        __init__.py
#        predictor.py
#        preprocess.py
#        postprocess.py
#        model_loader.py
#    /opt/ml/code/
#      inference.py                  <- toolkit entry point (SAGEMAKER_PROGRAM)
#
COPY . /tmp/ml_inference_src/
RUN pip install --no-cache-dir -e /tmp/ml_inference_src/

RUN mkdir -p /opt/ml/code && \
    cp /tmp/ml_inference_src/inference.py /opt/ml/code/inference.py

# Sanity-check: confirm the package and entry point are importable.
# This runs at image build time — a failure here aborts the build immediately
# and surfaces the error before any ECR push or SageMaker deploy.
RUN python - <<'EOF'
import sys

# 1. Package must be importable from anywhere (installed into site-packages)
import ml_inference
print("OK: import ml_inference ->", ml_inference.__file__)

# 2. Entry point must be importable as a top-level module (toolkit does this)
sys.path.insert(0, "/opt/ml/code")
import inference
print("OK: import inference ->", inference.__file__)

# 3. All four SageMaker handler functions must exist
for fn in ("model_fn", "input_fn", "predict_fn", "output_fn"):
    assert callable(getattr(inference, fn)), f"MISSING: inference.{fn}"
    print(f"OK: inference.{fn} is callable")

# 4. VTONPredictor must be importable from the package
from ml_inference.predictor import VTONPredictor
print("OK: VTONPredictor importable")

print("\nAll import checks passed.")
EOF

# ── SageMaker toolkit environment ────────────────────────────────────────────
ENV SAGEMAKER_PROGRAM=inference.py \
    SAGEMAKER_SUBMIT_DIRECTORY=/opt/ml/code

# ── Health check — TorchServe ping endpoint used by SageMaker ──
HEALTHCHECK --interval=30s --timeout=10s --start-period=600s --retries=3 \
    CMD curl -fsS http://localhost:8080/ping || exit 1

# `serve` is provided by the SageMaker PyTorch inference base image and
# starts TorchServe with the toolkit handler.
CMD ["serve"]
