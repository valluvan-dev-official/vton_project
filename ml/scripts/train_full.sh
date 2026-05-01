#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
ROOT="$(cd "$SCRIPT_DIR/.." && pwd)"

echo "=== VTON Full Training ==="
echo "Root: $ROOT"

cd "$ROOT"
python -m src.training.train --config configs/train_full.yaml
