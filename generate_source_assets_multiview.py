"""使用多视角渲染生成正确格式的 source assets"""
import os
import sys

# 设置环境变量 - 指定使用 CUDA 设备 3
os.environ['CUDA_VISIBLE_DEVICES'] = '3'
os.environ['SPCONV_ALGO'] = 'native'
os.environ['ATTN_BACKEND'] = 'flash_attn'

from pathlib import Path
from editing.preprocess.source_assets import generate_source_assets_from_model

# 配置
source_model = Path("assets/edit_example/model.glb")
output_dir = Path("outputs/source_assets_test0_multiview")

print("=" * 60)
print("使用多视角渲染生成 Source Assets")
print("=" * 60)
print(f"源模型: {source_model}")
print(f"输出目录: {output_dir}")
print()

if not source_model.exists():
    print(f"❌ 源模型不存在: {source_model}")
    sys.exit(1)

try:
    result = generate_source_assets_from_model(
        model_path=source_model,
        output_dir=output_dir,
        num_views=150,
        resolution=512,
        feature_model="dinov2_vitl14_reg",
        batch_size=10,
        engine="CYCLES",  # 使用 CYCLES 引擎
    )

    print("\n" + "=" * 60)
    print("✅ 生成完成！")
    print("=" * 60)
    print(f"Voxels: {result['voxels_path']}")
    print(f"Features: {result['features_path']}")
    print(f"Transforms: {result['transforms_path']}")
    print(f"Views: {result['num_views']}")
    print(f"Method: {result['method']}")

except Exception as e:
    print(f"\n❌ 生成失败: {e}")
    import traceback
    traceback.print_exc()
    sys.exit(1)
