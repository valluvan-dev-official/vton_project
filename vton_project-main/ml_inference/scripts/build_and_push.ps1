# build_and_push.ps1 — Build the DCI-VTON SageMaker inference image and push it to ECR.
#
# Base image: 763104351884.dkr.ecr.<REGION>.amazonaws.com/pytorch-inference
#             Tag: 2.1.0-gpu-py310-cu118-ubuntu20.04-sagemaker
#             Verified present in ap-south-1 via aws ecr describe-images.
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
# NOTE: --provenance=false --sbom=false alone is not always sufficient — on
# some buildx/BuildKit versions the image exporter still defaults to OCI
# media types because the attestation subsystem's default is set at the
# builder level, not just per-build. BUILDX_NO_DEFAULT_ATTESTATIONS=1 kills
# that default outright; combined with oci-mediatypes=false on the exporter,
# this reliably yields application/vnd.docker.distribution.manifest.v2+json.
#
# Prerequisites (run manually before this script):
#   aws ecr get-login-password --region ap-south-1 | docker login --username AWS --password-stdin 763104351884.dkr.ecr.ap-south-1.amazonaws.com
#   aws ecr get-login-password --region ap-south-1 | docker login --username AWS --password-stdin <AccountId>.dkr.ecr.ap-south-1.amazonaws.com
#
# Usage:
#   .\build_and_push.ps1 -AccountId 133946906587 -Region ap-south-1 -RepoName dci-vton-sagemaker -Tag latest
param(
    [Parameter(Mandatory = $true)][string]$AccountId,
    [Parameter(Mandatory = $true)][string]$Region,
    [string]$RepoName = "dci-vton-sagemaker",
    [string]$Tag = "latest"
)
$ErrorActionPreference = "Stop"

$ScriptDir = Split-Path -Parent $MyInvocation.MyCommand.Path
$MlInferenceDir = Resolve-Path (Join-Path $ScriptDir "..")

$EcrUri       = "$AccountId.dkr.ecr.$Region.amazonaws.com/$RepoName"
$DlcRegistry  = "763104351884.dkr.ecr.$Region.amazonaws.com"
$DestRegistry = "$AccountId.dkr.ecr.$Region.amazonaws.com"
$BuilderName  = "sagemaker-docker-v2-builder"

Write-Host "==> Using existing Docker login for AWS ECR"

Write-Host "==> 1/4 Ensuring destination ECR repo exists ($RepoName in $Region)"
aws ecr describe-repositories --region $Region --repository-names $RepoName *>$null
if ($LASTEXITCODE -ne 0) {
    aws ecr create-repository --region $Region --repository-name $RepoName
}

# Must be set before the builder is created AND before the build runs — this
# disables BuildKit's default attestation behavior at the source, rather than
# relying solely on the per-build --provenance/--sbom flags.
$env:BUILDX_NO_DEFAULT_ATTESTATIONS = "1"

Write-Host "==> 2/4 Ensuring buildx builder '$BuilderName' (docker-container driver) exists"
docker buildx inspect $BuilderName *>$null
if ($LASTEXITCODE -ne 0) {
    docker buildx create --name $BuilderName --driver docker-container --bootstrap
    if ($LASTEXITCODE -ne 0) { throw "docker buildx create failed" }
}
docker buildx use $BuilderName

Write-Host "==> 3/4 Building + pushing ${EcrUri}:${Tag} (linux/amd64, Docker V2 manifest, no attestations)"
docker buildx build `
    --builder $BuilderName `
    --platform linux/amd64 `
    --provenance=false `
    --sbom=false `
    --build-arg REGION=$Region `
    -t "${RepoName}:${Tag}" `
    -t "${EcrUri}:${Tag}" `
    -f (Join-Path $MlInferenceDir "Dockerfile") `
    --output "type=image,push=true,oci-mediatypes=false" `
    $MlInferenceDir
if ($LASTEXITCODE -ne 0) { throw "docker buildx build failed" }

Write-Host "==> 4/4 Verifying pushed manifest media type"
$raw = docker buildx imagetools inspect "${EcrUri}:${Tag}" --raw
$mediaType = ($raw | ConvertFrom-Json).mediaType
Write-Host "    mediaType: $mediaType"
if ($mediaType -eq "application/vnd.docker.distribution.manifest.v2+json") {
    Write-Host "    OK: Docker V2 manifest -- SageMaker-compatible."
} else {
    Write-Warning "Expected application/vnd.docker.distribution.manifest.v2+json, got '$mediaType'."
}

Write-Host "==> Done. Image URI: ${EcrUri}:${Tag}"
Write-Host "Pass this URI as --image-uri to deploy_sagemaker_endpoint.py"
