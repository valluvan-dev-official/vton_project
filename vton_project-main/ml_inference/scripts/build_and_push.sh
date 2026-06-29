#!/usr/bin/env bash
# build_and_push.sh — Build the DCI-VTON SageMaker inference image and push it to ECR.
#
# Usage:
#   ./build_and_push.sh <aws-account-id> <region> [repo-name] [tag]
#
# Example:
#   ./build_and_push.sh 960583974175 us-east-1 dci-vton-sagemaker latest
set -euo pipefail

ACCOUNT_ID="${1:?Usage: build_and_push.sh <aws-account-id> <region> [repo-name] [tag]}"
REGION="${2:?Usage: build_and_push.sh <aws-account-id> <region> [repo-name] [tag]}"
REPO_NAME="${3:-dci-vton-sagemaker}"
TAG="${4:-latest}"

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
ML_INFERENCE_DIR="$(cd "${SCRIPT_DIR}/.." && pwd)"

ECR_URI="${ACCOUNT_ID}.dkr.ecr.${REGION}.amazonaws.com/${REPO_NAME}"
SAGEMAKER_DLC_REGISTRY="763104351884.dkr.ecr.${REGION}.amazonaws.com"

echo "==> 1/5 Logging in to the SageMaker DLC base-image registry (${SAGEMAKER_DLC_REGISTRY})"
aws ecr get-login-password --region "${REGION}" \
  | docker login --username AWS --password-stdin "${SAGEMAKER_DLC_REGISTRY}"

echo "==> 2/5 Ensuring destination ECR repo exists (${REPO_NAME})"
aws ecr describe-repositories --region "${REGION}" --repository-names "${REPO_NAME}" >/dev/null 2>&1 \
  || aws ecr create-repository --region "${REGION}" --repository-name "${REPO_NAME}"

echo "==> 3/5 Logging in to destination ECR registry (${ECR_URI})"
aws ecr get-login-password --region "${REGION}" \
  | docker login --username AWS --password-stdin "${ACCOUNT_ID}.dkr.ecr.${REGION}.amazonaws.com"

echo "==> 4/5 Building image ${ECR_URI}:${TAG}"
docker build \
  --build-arg REGION="${REGION}" \
  -t "${REPO_NAME}:${TAG}" \
  -t "${ECR_URI}:${TAG}" \
  -f "${ML_INFERENCE_DIR}/Dockerfile" \
  "${ML_INFERENCE_DIR}"

echo "==> 5/5 Pushing ${ECR_URI}:${TAG}"
docker push "${ECR_URI}:${TAG}"

echo "==> Done."
echo "Image URI: ${ECR_URI}:${TAG}"
echo "Pass this URI as --image-uri to deploy_sagemaker_endpoint.py"
