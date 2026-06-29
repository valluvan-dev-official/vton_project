# Checkpoints

This folder is a **staging area** for the model weights that get packaged
into the SageMaker `model.tar.gz` artifact. The files are large and are
**not** committed to git (see `.gitignore`).

## Required files

| File | Size | Source |
|------|------|--------|
| `viton512.ckpt` | ~3.5 GB | DCI-VTON Google Drive (https://drive.google.com/drive/folders/11BJo59iXVu2_NknKMbN0jKtFV06HTn5K) |
| `warp_viton.pth` | ~140 MB | DCI-VTON Google Drive (same folder) |
| `densepose_rcnn_R_50_FPN_s1x.pkl` | ~250 MB | auto-downloaded on first load from `dl.fbaipublicfiles.com` (place here to avoid a runtime download) |

SegFormer (`mattmdjaga/segformer_b2_clothes`) and the OpenCLIP/CLIP vision
weights are pulled from the Hugging Face Hub by `transformers` — bake them
into the image (see Dockerfile `HF_HOME` warm-up) so cold start does not
hit the network.

## Packaging for SageMaker

SageMaker expects a `model.tar.gz` whose contents are extracted into
`/opt/ml/model` (the `model_dir` passed to `model_fn`).

```bash
cd ml_inference/checkpoints
tar -czvf ../model.tar.gz viton512.ckpt warp_viton.pth densepose_rcnn_R_50_FPN_s1x.pkl
aws s3 cp ../model.tar.gz s3://<your-bucket>/dci-vton/model.tar.gz
```

Then point the SageMaker `Model` at that S3 URI and the ECR image built
from `ml_inference/Dockerfile`.
