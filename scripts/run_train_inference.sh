#!/usr/bin/env bash
set -euo pipefail

# Move to project root
cd "$(dirname "$0")/.."

CONFIG="${1:-configs/default.yaml}"
if [[ $# -ge 1 ]]; then shift || true; fi

python -m src.pipeline.train_inference --config "$CONFIG" "$@"