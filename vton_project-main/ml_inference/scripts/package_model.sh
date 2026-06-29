#!/usr/bin/env bash
# package_model.sh — Build the SageMaker model.tar.gz artifact and upload it to S3.
#
# SageMaker extracts this archive into /opt/ml/model inside the container
# (the `model_dir` argument passed to inference.model_fn). Code is NOT
# included here — it is baked into the Docker image (BYOC), so the
# archive only needs to contain weights + configs.
#
# Usage:
#   ./package_model.sh <s3-bucket> [s3-prefix]
#
# Example:
#   ./package_model.sh dci-vton-artifacts-960583974175 dci-vton/model
#
# Prerequisites (place these here first — see checkpoints/README.md):
#   ml_inference/checkpoints/viton512.ckpt
#   ml_inference/checkpoints/warp_viton.pth
#   ml_inference/checkpoints/densepose_rcnn_R_50_FPN_s1x.pkl   (optional —
#       downloaded at runtime if absent, but packaging it keeps the
#       endpoint container fully offline)
set -euo pipefail

S3_BUCKET="${1:?Usage: package_model.sh <s3-bucket> [s3-prefix]}"
S3_PREFIX="${2:-dci-vton/model}"

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
ML_INFERENCE_DIR="$(cd "${SCRIPT_DIR}/.." && pwd)"
CKPT_DIR="${ML_INFERENCE_DIR}/checkpoints"
OUT_TAR="${ML_INFERENCE_DIR}/model.tar.gz"

REQUIRED_FILES=("viton512.ckpt" "warp_viton.pth")
for f in "${REQUIRED_FILES[@]}"; do
  if [[ ! -f "${CKPT_DIR}/${f}" ]]; then
    echo "ERROR: missing required checkpoint: ${CKPT_DIR}/${f}" >&2
    echo "       See ml_inference/checkpoints/README.md for the download source." >&2
    exit 1
  fi
done

echo "==> Packaging checkpoints from ${CKPT_DIR}"
TAR_ARGS=(viton512.ckpt warp_viton.pth)
if [[ -f "${CKPT_DIR}/densepose_rcnn_R_50_FPN_s1x.pkl" ]]; then
  TAR_ARGS+=(densepose_rcnn_R_50_FPN_s1x.pkl)
else
  echo "    (densepose_rcnn_R_50_FPN_s1x.pkl not found — will be downloaded at container startup)"
fi

# Optional deployment config bundled alongside the weights.
if [[ -d "${ML_INFERENCE_DIR}/configs" ]]; then
  TAR_ARGS+=(-C "${ML_INFERENCE_DIR}" configs)
fi

tar -czvf "${OUT_TAR}" -C "${CKPT_DIR}" "${TAR_ARGS[@]}"

echo "==> Built ${OUT_TAR} ($(du -h "${OUT_TAR}" | cut -f1))"

S3_URI="s3://${S3_BUCKET}/${S3_PREFIX}/model.tar.gz"
echo "==> Uploading to ${S3_URI}"
aws s3 cp "${OUT_TAR}" "${S3_URI}"

echo "==> Done."
echo "Model artifact: ${S3_URI}"
echo "Pass this URI as --model-data-url to deploy_sagemaker_endpoint.py"
