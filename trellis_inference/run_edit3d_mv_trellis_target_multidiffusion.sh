#!/bin/bash

cd "$(dirname "$0")/.." || exit 1

export ATTN_BACKEND="${ATTN_BACKEND:-flash_attn}"
export SPARSE_ATTN_BACKEND="${SPARSE_ATTN_BACKEND:-$ATTN_BACKEND}"
export SPCONV_ALGO="${SPCONV_ALGO:-native}"

DATASET_ROOT="${DATASET_ROOT:-/cache/wangxinxing/data/trellis_edit_benchmark/edit3d_mv_pseudosource_micro10}"
OUTPUT_ROOT="${OUTPUT_ROOT:-/cache/wangxinxing/data/trellis_edit_benchmark/pred_mv}"
RUN_NAME="${RUN_NAME:-trellis_mv_target_multidiffusion_seed1}"
GPUS="${GPUS:-0}"

python trellis_inference/batch_generate_from_multiview_targets.py \
  --dataset-root "$DATASET_ROOT" \
  --output-root "$OUTPUT_ROOT" \
  --run-name "$RUN_NAME" \
  --mode multidiffusion \
  --view-kind target \
  --seed 1 \
  --parallel \
  --gpus "$GPUS"
