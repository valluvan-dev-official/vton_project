# VTON Project

Virtual Try-On system: FastAPI async API + Celery workers + ML training pipeline.

## Architecture

```
vton-project/
├── api/        FastAPI server — job submission, Celery tasks, PostgreSQL, storage
├── ml/         ML pipeline — model, training loop, inference
├── storage/    Shared storage (inputs / outputs / training_pairs)
└── tests/      Integration tests
```

## Quick Start

```bash
# 1. Copy and fill in your .env
cp .env.example .env   # (edit .env as needed)

# 2. Start all services
cd api
docker compose up --build

# 3. Submit a try-on job
curl -X POST http://localhost:8000/api/v1/tryon \
  -F "person_image=@person.jpg" \
  -F "garment_image=@garment.jpg"

# 4. Poll job status
curl http://localhost:8000/api/v1/status/<job_id>

# 5. Download result
curl -o result.jpg http://localhost:8000/api/v1/result/<job_id>
```

## Storage Backend

Controlled by `STORAGE_BACKEND` in `.env`:

| Value   | Description                              |
|---------|------------------------------------------|
| `local` | Files saved to `LOCAL_STORAGE_PATH`      |
| `s3`    | Files saved to S3 (requires S3_* vars)   |

Switch by editing `.env` — no code change required.

## Training Pipeline

Results with SSIM ≥ `TRAINING_PAIR_SSIM_THRESHOLD` (default 0.65) are automatically saved to `storage/training_pairs/` by the Celery worker.

```bash
# Full training
cd ml && bash scripts/train_full.sh

# LoRA fine-tuning
cd ml && bash scripts/train_lora.sh
```

## Tests

```bash
pytest tests/ -v
```
