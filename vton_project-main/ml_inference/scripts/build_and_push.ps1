# build_and_push.ps1 — Build the DCI-VTON SageMaker inference image and push it to ECR.
#
# Base image: 763104351884.dkr.ecr.<REGION>.amazonaws.com/pytorch-inference
#             Tag: 2.1.0-gpu-py310-cu118-ubuntu20.04-sagemaker
#             Verified present in ap-south-1 via aws ecr describe-images.
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

Write-Host "==> Using existing Docker login for AWS ECR"

Write-Host "==> 1/3 Ensuring destination ECR repo exists ($RepoName in $Region)"
aws ecr describe-repositories --region $Region --repository-names $RepoName *>$null
if ($LASTEXITCODE -ne 0) {
    aws ecr create-repository --region $Region --repository-name $RepoName
}

Write-Host "==> 2/3 Building image ${EcrUri}:${Tag}"
docker build `
    --build-arg REGION=$Region `
    -t "${RepoName}:${Tag}" `
    -t "${EcrUri}:${Tag}" `
    -f (Join-Path $MlInferenceDir "Dockerfile") `
    $MlInferenceDir
if ($LASTEXITCODE -ne 0) { throw "docker build failed" }

Write-Host "==> 3/3 Pushing ${EcrUri}:${Tag}"
docker push "${EcrUri}:${Tag}"
if ($LASTEXITCODE -ne 0) { throw "docker push failed" }

Write-Host "==> Done. Image URI: ${EcrUri}:${Tag}"
Write-Host "Pass this URI as --image-uri to deploy_sagemaker_endpoint.py"
