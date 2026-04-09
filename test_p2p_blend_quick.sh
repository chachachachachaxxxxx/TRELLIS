#!/bin/bash
# Quick test: P2P Latent Blend with SS blending only (faster)

set -e

source ~/miniforge3/etc/profile.d/conda.sh
conda activate hammer

CASE_NAME="p2p_blend_quick_5"
SEED=1
GPU=2
SOURCE_MODEL="outputs/source_assets_test0_multiview"

export CUDA_VISIBLE_DEVICES=$GPU
export SPCONV_ALGO=native
export HTTP_PROXY=http://127.0.0.1:7890
export HTTPS_PROXY=http://127.0.0.1:7890

echo "=========================================="
echo "Quick Test: image_p2p_latent_blend"
echo "SS blending: ON"
echo "SLAT blending: ON"
echo "Inversion: simple Euler"
echo "Steps: 25 (reduced for speed)"
echo "GPU: CUDA $GPU"
echo "=========================================="

python run_edit_experiment.py \
  --method image_p2p_latent_blend \
  --asset-dir "$SOURCE_MODEL" \
  --source-image assets/edit_example/images/2d_render.png \
  --edit-image assets/edit_example/images/2d_edit.png \
  --mask-image assets/edit_example/images/2d_mask.png \
  --case-name "$CASE_NAME" \
  --mask-glb assets/edit_example/mask.glb \
  --seed $SEED \
  --preprocess \
  --ss-steps 25 \
  --slat-steps 25 \
  --extra-param blend_ss_enabled=true \
  --extra-param blend_slat_enabled=true \
  --extra-param inversion_mode=simple 

if [ $? -eq 0 ]; then
    echo ""
    echo "✅ Test passed!"
    echo "Output: outputs/image_p2p_latent_blend/$CASE_NAME"
else
    echo ""
    echo "❌ Test failed"
    exit 1
fi
