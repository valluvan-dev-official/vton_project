#!/usr/bin/env bash
# build_and_push.sh — Build the DCI-VTON SageMaker inference image and push it to ECR.
#
# SageMaker only accepts single-platform images with a Docker V2 (schema2)
# manifest (application/vnd.docker.distribution.manifest.v2+json). Docker 25+
# always builds through BuildKit/buildx, and buildx attaches provenance/SBOM
# attestations by default. Those attestations are stored as extra manifests,
# which forces the pushed ref to be an OCI *image index*
# (application/vnd.oci.image.index.v1+json) even for a single-platform build.
# SageMaker's manifest fetcher rejects that media type.
#
# Fix: build with buildx via a docker-container builder, explicitly disable
# provenance/SBOM attestations, force linux/amd64, and force Docker (not OCI)
# media types on the output so the final pushed manifest is schema2 v2.
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
BUILDER_NAME="sagemaker-docker-v2-builder"

echo "==> 1/6 Logging in to the SageMaker DLC base-image registry (${SAGEMAKER_DLC_REGISTRY})"
aws ecr get-login-password --region "${REGION}" \
  | docker login --username AWS --password-stdin "${SAGEMAKER_DLC_REGISTRY}"

echo "==> 2/6 Ensuring destination ECR repo exists (${REPO_NAME})"
aws ecr describe-repositories --region "${REGION}" --repository-names "${REPO_NAME}" >/dev/null 2>&1 \
  || aws ecr create-repository --region "${REGION}" --repository-name "${REPO_NAME}"

echo "==> 3/6 Logging in to destination ECR registry (${ECR_URI})"
aws ecr get-login-password --region "${REGION}" \
  | docker login --username AWS --password-stdin "${ACCOUNT_ID}.dkr.ecr.${REGION}.amazonaws.com"

echo "==> 4/6 Ensuring buildx builder '${BUILDER_NAME}' (docker-container driver) exists"
if ! docker buildx inspect "${BUILDER_NAME}" >/dev/null 2>&1; then
  docker buildx create --name "${BUILDER_NAME}" --driver docker-container --bootstrap
fi
docker buildx use "${BUILDER_NAME}"

echo "==> 5/6 Building + pushing ${ECR_URI}:${TAG} (linux/amd64, Docker V2 manifest, no attestations)"
docker buildx build \
  --builder "${BUILDER_NAME}" \
  --platform linux/amd64 \
  --provenance=false \
  --sbom=false \
  --build-arg REGION="${REGION}" \
  -t "${REPO_NAME}:${TAG}" \
  -f "${ML_INFERENCE_DIR}/Dockerfile" \
  --output "type=image,name=${ECR_URI}:${TAG},push=true,oci-mediatypes=false" \
  "${ML_INFERENCE_DIR}"

echo "==> 6/6 Verifying pushed manifest media type"
MEDIA_TYPE=$(docker buildx imagetools inspect "${ECR_URI}:${TAG}" --raw | python3 -c "import json,sys; print(json.load(sys.stdin).get('mediaType',''))" 2>/dev/null || true)
echo "    mediaType: ${MEDIA_TYPE:-<unable to inspect>}"
case "${MEDIA_TYPE}" in
  application/vnd.docker.distribution.manifest.v2+json)
    echo "    OK: Docker V2 manifest — SageMaker-compatible." ;;
  *)
    echo "    WARNING: expected application/vnd.docker.distribution.manifest.v2+json, got '${MEDIA_TYPE}'." >&2 ;;
esac

echo "==> Done."
echo "Image URI: ${ECR_URI}:${TAG}"
echo "Pass this URI as --image-uri to deploy_sagemaker_endpoint.py"
