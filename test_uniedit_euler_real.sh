#!/bin/bash
# Real test for UniEdit Euler method using existing source assets

set -e

# Environment setup
export CUDA_VISIBLE_DEVICES=3
export ATTN_BACKEND=xformers
export SPARSE_ATTN_BACKEND=xformers
export SPCONV_ALGO=native
export http_proxy=http://127.0.0.1:7890
export https_proxy=http://127.0.0.1:7890

# Test case configuration
SOURCE_IMAGE="assets/edit_example/images/2d_render.png"
EDIT_IMAGE="assets/edit_example/images/2d_edit.png"
MASK_GLB="assets/edit_example/mask.glb"
SOURCE_ASSETS="outputs/z_output/example_edit_trace_hammer_interactive/image_cross_attention_trace/2d_render"
CASE_NAME="real_test_euler"
SEED=1

# Method parameters
SS_OMEGA=1.0
SLAT_OMEGA=1.0
SS_ALPHA=0.5
SLAT_ALPHA=0.5
ZERO_INIT=false

echo "=========================================="
echo "Testing UniEdit Euler on Real Example"
echo "=========================================="
echo "Source: $SOURCE_IMAGE"
echo "Edit: $EDIT_IMAGE"
echo "Mask: $MASK_GLB"
echo "Source assets: $SOURCE_ASSETS"
echo "Parameters:"
echo "  ss_omega=$SS_OMEGA"
echo "  slat_omega=$SLAT_OMEGA"
echo "  ss_alpha=$SS_ALPHA"
echo "  slat_alpha=$SLAT_ALPHA"
echo "  zero_init=$ZERO_INIT"
echo "=========================================="

# Check if source assets exist
if [ ! -f "$SOURCE_ASSETS/slat_coords.npy" ]; then
    echo "Error: Source assets not found at $SOURCE_ASSETS"
    exit 1
fi

echo "✓ Source assets found"
echo ""

# Run the test
python run_edit_experiment.py \
  --method image_uniedit_euler \
  --source-image "$SOURCE_IMAGE" \
  --edit-image "$EDIT_IMAGE" \
  --mask-glb "$MASK_GLB" \
  --source-model "$SOURCE_ASSETS" \
  --case-name "$CASE_NAME" \
  --seed "$SEED" \
  --skip-render \
  --extra-params ss_omega="$SS_OMEGA" \
  --extra-params slat_omega="$SLAT_OMEGA" \
  --extra-params ss_alpha="$SS_ALPHA" \
  --extra-params slat_alpha="$SLAT_ALPHA" \
  --extra-params zero_init="$ZERO_INIT" \
  --extra-params cfg_interval="(0.5,1.0)" \
  --extra-params decode_modes='["gaussian","mesh"]'

echo ""
echo "=========================================="
echo "Test complete!"
echo "Output: outputs/image_uniedit_euler/$CASE_NAME/"
echo "=========================================="
