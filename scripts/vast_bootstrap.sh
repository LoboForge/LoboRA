#!/usr/bin/env bash
# Bootstrap a Vast box for LoboRA (2x H100 80GB recommended).
set -euo pipefail
ROOT="${1:-/workspace/LoboRA}"
python3 -m pip install -U pip
python3 -m pip install -e "${ROOT}[gpu]"
echo "Next:"
echo "  python scripts/download_weights.py --dest /workspace/models/MiniMax-H3"
echo "  lobora configs/ref2va_bf16_80gb.yaml --dataset-path /workspace/dataset --output-dir /workspace/output/run0"
