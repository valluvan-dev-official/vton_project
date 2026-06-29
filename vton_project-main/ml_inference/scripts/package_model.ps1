# package_model.ps1 — Build the SageMaker model.tar.gz artifact and upload it to S3.
#
# Usage:
#   .\package_model.ps1 -S3Bucket dci-vton-artifacts-960583974175 -S3Prefix dci-vton/model
#
# Prerequisites (place these here first — see checkpoints\README.md):
#   ml_inference\checkpoints\viton512.ckpt
#   ml_inference\checkpoints\warp_viton.pth
#   ml_inference\checkpoints\densepose_rcnn_R_50_FPN_s1x.pkl   (optional)
param(
    [Parameter(Mandatory = $true)][string]$S3Bucket,
    [string]$S3Prefix = "dci-vton/model"
)
$ErrorActionPreference = "Stop"

$ScriptDir = Split-Path -Parent $MyInvocation.MyCommand.Path
$MlInferenceDir = Resolve-Path (Join-Path $ScriptDir "..")
$CkptDir = Join-Path $MlInferenceDir "checkpoints"
$OutTar = Join-Path $MlInferenceDir "model.tar.gz"

foreach ($f in @("viton512.ckpt", "warp_viton.pth")) {
    $p = Join-Path $CkptDir $f
    if (-not (Test-Path $p)) {
        Write-Error "Missing required checkpoint: $p (see checkpoints\README.md)"
    }
}

Write-Host "==> Packaging checkpoints from $CkptDir"
$tarArgs = @("-czvf", $OutTar, "-C", $CkptDir, "viton512.ckpt", "warp_viton.pth")
if (Test-Path (Join-Path $CkptDir "densepose_rcnn_R_50_FPN_s1x.pkl")) {
    $tarArgs += "densepose_rcnn_R_50_FPN_s1x.pkl"
} else {
    Write-Host "    (densepose pkl not found — will be downloaded at container startup)"
}
$configsDir = Join-Path $MlInferenceDir "configs"
if (Test-Path $configsDir) {
    $tarArgs += @("-C", $MlInferenceDir, "configs")
}

& tar @tarArgs
if ($LASTEXITCODE -ne 0) { throw "tar failed" }

Write-Host "==> Built $OutTar"

$S3Uri = "s3://$S3Bucket/$S3Prefix/model.tar.gz"
Write-Host "==> Uploading to $S3Uri"
aws s3 cp $OutTar $S3Uri
if ($LASTEXITCODE -ne 0) { throw "aws s3 cp failed" }

Write-Host "==> Done. Model artifact: $S3Uri"
Write-Host "Pass this URI as -ModelDataUrl to deploy_sagemaker_endpoint.py"
