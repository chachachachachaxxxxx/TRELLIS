#!/usr/bin/env bash

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "${SCRIPT_DIR}/.." && pwd)"
cd "${REPO_ROOT}"

if [[ "${1:-}" == "-h" || "${1:-}" == "--help" ]]; then
  cat <<'EOF'
Usage:
  bash scripts/run_trace_trellis_text_xlarge.sh [prompt] [extra args...]

Examples:
  bash scripts/run_trace_trellis_text_xlarge.sh
  bash scripts/run_trace_trellis_text_xlarge.sh "A small wooden chair"
  bash scripts/run_trace_trellis_text_xlarge.sh "A small wooden chair" --ss-steps 20 --slat-steps 20

Environment variables:
  MODEL       Model repo or local path
  OUT_DIR     Output root directory
  PROMPT      Default prompt when no positional prompt is given
  SEED        Random seed
  SS_STEPS    Sparse-structure sampling steps
  SLAT_STEPS  SLat sampling steps
  FORMATS     Decode formats, comma-separated
EOF
  exit 0
fi

export SPCONV_ALGO="${SPCONV_ALGO:-native}"

MODEL="${MODEL:-microsoft/TRELLIS-text-xlarge}"
OUT_DIR="${OUT_DIR:-outputs/trellis_text_xlarge_trace}"
PROMPT="${PROMPT:-A ceramic teapot shaped like a rabbit}"
SEED="${SEED:-1}"
SS_STEPS="${SS_STEPS:-12}"
SLAT_STEPS="${SLAT_STEPS:-12}"
FORMATS="${FORMATS:-mesh,gaussian,radiance_field}"

if [[ $# -gt 0 ]]; then
  PROMPT="$1"
  shift
fi

python scripts/trace_trellis_text_xlarge.py \
  --prompt "${PROMPT}" \
  --model "${MODEL}" \
  --output-dir "${OUT_DIR}" \
  --seed "${SEED}" \
  --ss-steps "${SS_STEPS}" \
  --slat-steps "${SLAT_STEPS}" \
  --formats "${FORMATS}" \
  "$@"

python scripts/trace_trellis_text_xlarge.py \
  --prompt "A small wooden chair" \
  --output-dir "outputs/trellis_text_xlarge_trace" \
  --seed 1 \
  --ss-steps 12 \
  --slat-steps 12 \
  --skip-decode
