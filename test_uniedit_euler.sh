#!/bin/bash
# Test script for UniEdit Euler method

set -e

# Environment setup
export CUDA_VISIBLE_DEVICES=3
export ATTN_BACKEND=xformers
export SPARSE_ATTN_BACKEND=xformers
export SPCONV_ALGO=native

# Test case configuration
SOURCE_IMAGE="assets/edit_example/images/2d_render.png"
EDIT_IMAGE="assets/edit_example/images/2d_edit.png"
MASK_GLB="assets/edit_example/mask.glb"
SOURCE_MODEL="assets/edit_example"
CASE_NAME="test_uniedit_euler"
SEED=1

# Method parameters
SS_OMEGA=1.0
SLAT_OMEGA=1.0
SS_ALPHA=0.5
SLAT_ALPHA=0.5
ZERO_INIT=false

echo "Testing UniEdit Euler method..."
echo "Source: $SOURCE_IMAGE"
echo "Edit: $EDIT_IMAGE"
echo "Mask: $MASK_GLB"
echo "Source model: $SOURCE_MODEL"
echo "Parameters: ss_omega=$SS_OMEGA, slat_omega=$SLAT_OMEGA, ss_alpha=$SS_ALPHA, slat_alpha=$SLAT_ALPHA"

python run_edit_experiment.py \
  --method image_uniedit_euler \
  --source-image "$SOURCE_IMAGE" \
  --edit-image "$EDIT_IMAGE" \
  --mask-glb "$MASK_GLB" \
  --source-model "$SOURCE_MODEL" \
  --case-name "$CASE_NAME" \
  --seed "$SEED" \
  --preprocess \
  --skip-render \
  --extra-params ss_omega="$SS_OMEGA" \
  --extra-params slat_omega="$SLAT_OMEGA" \
  --extra-params ss_alpha="$SS_ALPHA" \
  --extra-params slat_alpha="$SLAT_ALPHA" \
  --extra-params zero_init="$ZERO_INIT" \
  --extra-params cfg_interval="(0.5,1.0)" \
  --extra-params decode_modes="[\"gaussian\",\"mesh\"]"

echo "Test complete! Check outputs/image_uniedit_euler/$CASE_NAME/"
