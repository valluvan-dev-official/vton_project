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
COPY requirements.txt /opt/ml/requirements.txt
RUN pip install --no-cache-dir -r /opt/ml/requirements.txt

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

# ── Inference code — SageMaker toolkit looks in /opt/ml/code ──
ENV SAGEMAKER_PROGRAM=inference.py \
    SAGEMAKER_SUBMIT_DIRECTORY=/opt/ml/code
COPY . /opt/ml/code/
# Make the package importable both as `ml_inference.*` and as top-level modules.
RUN cp -r /opt/ml/code /opt/ml/code/ml_inference 2>/dev/null || true

# ── Health check — TorchServe ping endpoint used by SageMaker ──
HEALTHCHECK --interval=30s --timeout=10s --start-period=600s --retries=3 \
    CMD curl -fsS http://localhost:8080/ping || exit 1

# `serve` is provided by the SageMaker PyTorch inference base image and
# starts TorchServe with the toolkit handler.
CMD ["serve"]
