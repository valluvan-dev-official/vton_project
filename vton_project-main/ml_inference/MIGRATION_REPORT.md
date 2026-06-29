# DCI-VTON → Amazon SageMaker Async Inference — Migration Report

Status: **prepared, not deployed.** No model code was rewritten. No business
logic was modified. The existing FastAPI + Celery application is unchanged
and continues to work exactly as before.

---

## Phase 1 — Project Analysis

### 1. Files responsible for loading each model

All real model loading lives in **one file**:
[`ml/scripts/gpu_inference.py`](../ml/scripts/gpu_inference.py) — class `GPUInferenceEngine`.

| Model / weight | Loader (original) | Refactored into |
|----------------|-------------------|-----------------|
| `viton512.ckpt` (Stable Diffusion / DCI-VTON LDM) | `_load_dci_model()` (`torch.load` + `instantiate_from_config`) | `ml_inference/model_loader.py::load_dci` |
| `warp_viton.pth` (AFWM warp, PF-AFN) | `_load_afwm()` (`load_checkpoint`) | `ml_inference/model_loader.py::load_afwm` |
| SegFormer (`mattmdjaga/segformer_b2_clothes`) | `_load_segformer()` | `ml_inference/model_loader.py::load_segformer` |
| DensePose (`densepose_rcnn_R_50_FPN_s1x.pkl`, detectron2) | `_load_densepose()` | `ml_inference/model_loader.py::load_densepose` |
| OpenCLIP / CLIP vision (via DCI's `ldm` encoders) | inside `_load_dci_model()` (the `CLIPVisionModel` patch + `get_learned_conditioning`) | `ml_inference/model_loader.py::load_dci` (patch) + `predictor.py::_diffuse` (usage) |
| Stable Diffusion pipeline (LDM + `DDIMSampler`) | `_load_dci_model()` | `ml_inference/model_loader.py::load_dci` |

Supporting files:
- [`api/app/services/inference.py`](../api/app/services/inference.py) — `InferenceRouter` *selects* a backend (placeholder / Kaggle / local GPU) but does not itself load the heavy models except by delegating to `GPUInferenceEngine`.
- [`ml/weights/README.md`](../ml/weights/README.md), [`ml/kaggle/README_CHECKPOINTS.md`](../ml/kaggle/README_CHECKPOINTS.md) — where the checkpoints come from.
- [`ml/configs/inference.yaml`](../ml/configs/inference.yaml) — references `ml/checkpoints/best.pt` (the *own-model* Phase-4 stub, unused by the live DCI pipeline).

### 2. Complete inference flow

```
Person Image ─┐
              ├─► (resize 512×512)
Garment Image ┘
        │
        ▼
  PREPROCESSING                              [preprocess.py]
   • SegFormer parse person  → pred
   • SegFormer parse garment → g_pred
   • build agnostic mask + collar strip + cleaned garment   (build_masks)
   • DensePose IUV (computed for parity)                     (get_densepose_iuv)
        │
        ▼
  WARP MODEL (AFWM / warp_viton.pth)         [preprocess.py::warp_garment]
   • cond = [agnostic ⊕ 13-ch parse], garment → warped_garment
        │
        ▼
  DIFFUSION MODEL (viton512.ckpt, LDM+DDIM)  [predictor.py::_diffuse]
   • CLIP conditioning from cleaned garment
   • VAE-encode agnostic (inpaint) + warped garment (init latent)
   • DDIM sampling (20 steps of 50, CFG=12.0) with inpaint mask
   • VAE-decode → raw RGB
        │
        ▼
  POSTPROCESS                                [postprocess.py]
   • LAB color-match generated region to source garment
   • feathered composite back onto original person
        │
        ▼
  Output Image (JPEG, quality 95)
```

Original monolith: `GPUInferenceEngine.run()` (Cells 5/6/7/9/10).
Refactored: `VTONPredictor.predict()` calling `preprocess → _diffuse → postprocess`.

### 3. Kaggle-specific components (every usage)

All Kaggle coupling is confined to **two files**; the model pipeline itself has **zero** Kaggle code.

| Component | Where | What it does |
|-----------|-------|--------------|
| Kaggle API auth / OAuth token refresh | `api/app/services/inference.py::_kaggle_env`, `_sync_kaggle_oauth_token` | Builds env for the `kaggle` CLI; refreshes `KGAT_` OAuth tokens. |
| `kaggle datasets version/create` (temporary uploads) | `inference.py::_upload_to_kaggle_dataset`, `_wait_for_dataset_ready` | Uploads person/garment as a Kaggle dataset. |
| `kaggle kernels push` (notebook execution) | `inference.py::_trigger_kaggle_notebook` | Bakes images as base64 into `ml/kaggle/dci_vton_inference.ipynb`, pushes to run on a T4. |
| Kernel status / output polling | `inference.py::_poll_kaggle_notebook` | Polls `ListKernelSessionOutput` every 30s up to 2400s; T4-vs-P100 retry logic. |
| Result download | `inference.py::_download_kaggle_result` | 4 fallback strategies (presigned URL, list-output, REST zip, CLI). |
| Orchestration / GPU retry loop | `inference.py::_run_kaggle` | Retries up to 8× to dodge incompatible P100 GPUs. |
| Notebook + kernel metadata | `ml/kaggle/dci_vton_inference.ipynb`, `ml/kaggle/kernel-metadata.json`, `ml/kaggle/upload_notebook.py`, `ml/kaggle/README_CHECKPOINTS.md` | The Kaggle notebook and its uploader. |
| Config | `api/app/config.py` (`KAGGLE_USERNAME/KEY/NOTEBOOK_SLUG/DATASET_SLUG`); `kaggle==2.1.2` in `api/requirements.txt` | Kaggle settings + CLI dependency. |
| Compose mount | `api/docker-compose.yml` (`$USERPROFILE/.kaggle:/root/.kaggle`) | Mounts Kaggle creds into the worker. |

---

## Phase 2 — Extracted Inference Engine

New self-contained package `ml_inference/`. It is a **structural** refactor of
`gpu_inference.py` — the numerics are copied verbatim (segmentation labels,
mask geometry, warp parse map, DDIM steps=20/50, CFG=12.0, seed derivation,
LAB color correction, composite feathering).

```
ml_inference/
├── __init__.py          # package exports
├── inference.py         # SageMaker handlers (Phase 3)
├── predictor.py         # VTONPredictor — orchestrates preprocess→diffuse→postprocess
├── preprocess.py        # SegFormer parse, masks, DensePose, AFWM warp
├── postprocess.py       # LAB color correction + composite
├── model_loader.py      # load all models ONCE → ModelBundle
├── requirements.txt     # GPU/inference deps
├── Dockerfile           # production SageMaker image (Phase 4)
├── configs/
│   └── inference.yaml   # deployment knobs
├── checkpoints/
│   ├── README.md        # where to place viton512.ckpt / warp_viton.pth / densepose pkl
│   └── .gitignore-d     # weights never committed
└── MIGRATION_REPORT.md  # this file
```

`VTONPredictor.run(person, garment, output, job_id)` keeps the **same
signature** as `GPUInferenceEngine.run()`, so it is a drop-in replacement for
the local-GPU path if desired.

---

## Phase 3 — SageMaker Inference Handlers

`ml_inference/inference.py` implements the SageMaker PyTorch toolkit contract:

| Function | Behaviour |
|----------|-----------|
| `model_fn(model_dir)` | Loads **all** models **once** at container start from `/opt/ml/model`; picks `cuda` if available; returns a `VTONPredictor`. Checkpoints are never reloaded; models stay resident in GPU memory for the process lifetime. |
| `input_fn(body, content_type)` | Accepts `application/json` (`{"person","garment","job_id"}` base64) **and** `multipart/form-data` (`person_image`/`garment_image`); decodes to PIL. |
| `predict_fn(input, model)` | Calls `model.predict(person, garment, job_id)` — the full pipeline. |
| `output_fn(prediction, accept)` | Returns raw JPEG bytes (`image/jpeg`) or base64 JSON (`application/json`). |

---

## Phase 4 — Production Docker Image

`ml_inference/Dockerfile` — based on the AWS SageMaker **PyTorch inference**
Deep Learning Container (`pytorch-inference:2.1.0-gpu-py310-cu121`), which
ships CUDA 12.1, cuDNN, PyTorch, TorchServe and the inference toolkit.

Includes: CUDA, PyTorch, detectron2 + DensePose, SegFormer (Transformers),
OpenCLIP/CLIP, the DCI-VTON / taming-transformers / PF-AFN repos (baked at
build time so cold start does not `git clone`), SegFormer weights warmed into
the image, `SAGEMAKER_PROGRAM=inference.py`, and a TorchServe `/ping`
HEALTHCHECK.

---

## Phase 5 — Migration Summary

### 1. Existing architecture
Client → FastAPI (`/api/v1/tryon`) → Postgres job row → Celery task →
`InferenceRouter` → **Kaggle** (`kaggle kernels push` a notebook onto a T4,
poll ≤40 min, download result) → SSIM scoring → result on disk/S3.

### 2. New SageMaker architecture
Client → FastAPI → Celery task → **`boto3 sagemaker-runtime
.invoke_endpoint_async`** (input JSON in S3) → SageMaker **Async** endpoint
running this container (models resident in GPU) → output JPEG written to the
S3 output path → success/error SNS notification → Celery finalizes the job.
Scale-to-zero when idle; internal queue absorbs bursts. **No Kaggle.**

### 3. Folder structure
See Phase 2 — new top-level `ml_inference/` package. Nothing else moved.

### 4. Files created
- `ml_inference/__init__.py`
- `ml_inference/model_loader.py`
- `ml_inference/preprocess.py`
- `ml_inference/postprocess.py`
- `ml_inference/predictor.py`
- `ml_inference/inference.py`
- `ml_inference/requirements.txt`
- `ml_inference/Dockerfile`
- `ml_inference/configs/inference.yaml`
- `ml_inference/checkpoints/README.md`
- `ml_inference/.gitignore`
- `ml_inference/MIGRATION_REPORT.md`

### 5. Files modified
**None.** The existing project was deliberately left byte-for-byte unchanged
to preserve all current functionality (verified: `git status` shows only
`ml_inference/` added).

### 6. Files removed
**None removed yet** (removal would change current behaviour). Once the
SageMaker endpoint is validated, the following become **safe to delete**:
- `ml/kaggle/dci_vton_inference.ipynb`
- `ml/kaggle/kernel-metadata.json`
- `ml/kaggle/upload_notebook.py`
- `ml/kaggle/README_CHECKPOINTS.md`
- The Kaggle methods in `api/app/services/inference.py`
  (`_kaggle_env`, `_sync_kaggle_oauth_token`, `_run_kaggle`,
  `_upload_to_kaggle_dataset`, `_wait_for_dataset_ready`,
  `_trigger_kaggle_notebook`, `_poll_kaggle_notebook`,
  `_download_kaggle_result`, and the `kaggle` mode branch).
- `KAGGLE_*` settings in `api/app/config.py`; `kaggle==2.1.2` in
  `api/requirements.txt`; the `.kaggle` mount in `api/docker-compose.yml`.

### 7. Code that still depends on Kaggle
- `api/app/services/inference.py` — the `kaggle` mode is still the **default
  active path** when `KAGGLE_*` env vars are set. It was not touched.
- `ml/kaggle/*` — unchanged.
- **Recommended integration (additive, non-breaking):** add a `sagemaker`
  branch to `InferenceRouter.run()` gated on a new `SAGEMAKER_ENDPOINT_NAME`
  env var, taking priority over `kaggle`. When the var is unset, behaviour is
  identical to today. Sketch:
  ```python
  # in InferenceRouter.run(), before the kaggle branch:
  if os.getenv("SAGEMAKER_ENDPOINT_NAME"):
      return self._run_sagemaker(person, garment, output)  # boto3 invoke_endpoint_async
  ```
  This keeps the Celery async job model intact — the worker submits to the
  async endpoint and polls the S3 output location instead of Kaggle.

### 8. Estimated GPU memory usage
- DCI-VTON LDM (UNet + VAE, fp16 conditioning) ........ ~3.5–4.5 GB
- AFWM warp .......................................... ~0.3 GB
- DensePose (R50-FPN) ................................ ~1.5 GB
- SegFormer-B2 ...................................... ~0.5 GB
- CLIP vision encoder ............................... ~0.6 GB
- Activations / DDIM @ 512² batch=1 ................. ~1.5–3 GB
- **Total working set ≈ 8–11 GB** → a **16 GB** GPU is comfortable.

### 9. Recommended SageMaker GPU instance type
- **Primary: `ml.g5.xlarge`** (1× NVIDIA A10G, 24 GB) — best price/perf,
  ample headroom, Ampere is fully supported (no P100-style incompatibility).
- Budget alternative: **`ml.g4dn.xlarge`** (1× T4, 16 GB) — matches the GPU
  the pipeline was tuned on; fits but with less headroom.
- For Async endpoints, set `MinCapacity=0` to **scale to zero** between bursts.

### 10. Estimated cold-start time
- Container pull (large CUDA image, first time) ...... 1–3 min (cached after)
- `model_fn` load (3.5 GB ckpt + 4 models to GPU) .... ~60–120 s
- Repos & SegFormer baked into image → **no** runtime clone/download.
- DensePose `.pkl`: ~0 s if packaged in `model.tar.gz`, else ~10–20 s download.
- **Cold start ≈ 2–4 min**; warm inference ≈ **20–30 s/request** (unchanged
  from the local-GPU path). Async endpoint queues requests during scale-up.

### 11. Blockers before deployment
1. **Checkpoints not in repo** — `viton512.ckpt` (~3.5 GB) and `warp_viton.pth`
   must be downloaded and packaged into `model.tar.gz` on S3 (see
   `checkpoints/README.md`).
2. **Base image entitlement** — ECR pull from `763104351884` requires
   `aws ecr get-login-password` for the target region.
3. **detectron2/DensePose build** — must compile against the base image's
   CUDA/PyTorch; `TORCH_CUDA_ARCH_LIST` is set, but verify the build in CI.
   (If the wheel fails, pin a detectron2 commit compatible with torch 2.1.)
4. **`viton512.yaml` provenance** — the LDM architecture config comes from the
   cloned DCI repo at build time; pin the repo commit for reproducibility.
5. **Payload limits** — SageMaker Async input is fetched from S3 (good for
   large images); ensure the FastAPI/Celery side uploads inputs to S3 and
   reads outputs from the async output path.
6. **Internet at runtime** — image bakes repos + SegFormer, but DensePose pkl
   and any HF cache miss need either packaging or VPC egress. Recommend
   packaging the DensePose pkl to keep the endpoint fully offline.
7. **`os.chdir` in `load_dci`** — the DCI repo resolves relative paths against
   CWD; harmless in the single-purpose container but noted (carried over
   verbatim from the original).
8. **First-request warm-up** — TorchServe `start-period` is set to 600 s in
   the HEALTHCHECK to cover model load; tune endpoint `ModelDataDownloadTimeout`
   / `ContainerStartupHealthCheckTimeout` accordingly (e.g. 900 s).
```
