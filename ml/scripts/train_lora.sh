#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
ROOT="$(cd "$SCRIPT_DIR/.." && pwd)"

echo "=== VTON LoRA Fine-tuning ==="
echo "Root: $ROOT"

cd "$ROOT"
python -m src.training.train --config configs/train_lora.yaml
