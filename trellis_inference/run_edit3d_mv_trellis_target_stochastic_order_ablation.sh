#!/bin/bash

cd "$(dirname "$0")/.." || exit 1

export ATTN_BACKEND="${ATTN_BACKEND:-flash_attn}"
export SPARSE_ATTN_BACKEND="${SPARSE_ATTN_BACKEND:-$ATTN_BACKEND}"
export SPCONV_ALGO="${SPCONV_ALGO:-native}"

DATASET_ROOT="${DATASET_ROOT:-/cache/wangxinxing/data/trellis_edit_benchmark/edit3d_mv_pseudosource_micro10}"
OUTPUT_ROOT="${OUTPUT_ROOT:-/cache/wangxinxing/data/trellis_edit_benchmark/pred_mv}"
RUN_NAME_BASE="${RUN_NAME_BASE:-trellis_mv_target_stochastic_seed1}"
GPUS="${GPUS:-0}"
ORDERS="${ORDERS:-azimuth reverse_azimuth front_first_clockwise front_first_counterclockwise}"

for ORDER in $ORDERS; do
  echo "============================================================"
  echo "Running stochastic order ablation: $ORDER"
  echo "============================================================"

  python trellis_inference/batch_generate_from_multiview_targets.py \
    --dataset-root "$DATASET_ROOT" \
    --output-root "$OUTPUT_ROOT" \
    --run-name "${RUN_NAME_BASE}_${ORDER}" \
    --mode stochastic \
    --view-kind target \
    --view-order "$ORDER" \
    --seed 1 \
    --parallel \
    --gpus "$GPUS"
done
