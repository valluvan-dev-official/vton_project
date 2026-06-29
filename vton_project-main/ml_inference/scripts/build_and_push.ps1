# build_and_push.ps1 — Build the DCI-VTON SageMaker inference image and push it to ECR.
#
# Usage:
#   .\build_and_push.ps1 -AccountId 960583974175 -Region us-east-1 -RepoName dci-vton-sagemaker -Tag latest
param(
    [Parameter(Mandatory = $true)][string]$AccountId,
    [Parameter(Mandatory = $true)][string]$Region,
    [string]$RepoName = "dci-vton-sagemaker",
    [string]$Tag = "latest"
)
$ErrorActionPreference = "Stop"

$ScriptDir = Split-Path -Parent $MyInvocation.MyCommand.Path
$MlInferenceDir = Resolve-Path (Join-Path $ScriptDir "..")

$EcrUri = "$AccountId.dkr.ecr.$Region.amazonaws.com/$RepoName"
$SagemakerDlcRegistry = "763104351884.dkr.ecr.$Region.amazonaws.com"

Write-Host "==> 1/5 Logging in to the SageMaker DLC base-image registry ($SagemakerDlcRegistry)"
(aws ecr get-login-password --region $Region) | docker login --username AWS --password-stdin $SagemakerDlcRegistry
if ($LASTEXITCODE -ne 0) { throw "docker login (DLC registry) failed" }

Write-Host "==> 2/5 Ensuring destination ECR repo exists ($RepoName)"
aws ecr describe-repositories --region $Region --repository-names $RepoName *>$null
if ($LASTEXITCODE -ne 0) {
    aws ecr create-repository --region $Region --repository-name $RepoName
}

Write-Host "==> 3/5 Logging in to destination ECR registry ($EcrUri)"
(aws ecr get-login-password --region $Region) | docker login --username AWS --password-stdin "$AccountId.dkr.ecr.$Region.amazonaws.com"
if ($LASTEXITCODE -ne 0) { throw "docker login (destination registry) failed" }

Write-Host "==> 4/5 Building image ${EcrUri}:${Tag}"
docker build `
    --build-arg REGION=$Region `
    -t "${RepoName}:${Tag}" `
    -t "${EcrUri}:${Tag}" `
    -f (Join-Path $MlInferenceDir "Dockerfile") `
    $MlInferenceDir
if ($LASTEXITCODE -ne 0) { throw "docker build failed" }

Write-Host "==> 5/5 Pushing ${EcrUri}:${Tag}"
docker push "${EcrUri}:${Tag}"
if ($LASTEXITCODE -ne 0) { throw "docker push failed" }

Write-Host "==> Done. Image URI: ${EcrUri}:${Tag}"
Write-Host "Pass this URI as -ImageUri to deploy_sagemaker_endpoint.py"
