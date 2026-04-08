"""根据 trellis_inference/example_annotated.py 生成完整资产"""
import os
os.environ['SPCONV_ALGO'] = 'native'
os.environ['ATTN_BACKEND'] = 'flash_attn'

import torch
import numpy as np
from PIL import Image
from trellis.pipelines import TrellisImageTo3DPipeline
from trellis.utils import postprocessing_utils

# 设置
seed = 1
output_dir = "outputs/source_assets_test0"
os.makedirs(output_dir, exist_ok=True)

print("=" * 60)
print("生成 Source Assets")
print("=" * 60)

# 加载 pipeline
print("\n[1/6] 加载 pipeline...")
pipeline = TrellisImageTo3DPipeline.from_pretrained("microsoft/TRELLIS-image-large")
pipeline.cuda()

# 加载并预处理图像
print("[2/6] 加载并预处理图像...")
image = Image.open("assets/edit_example/images/2d_render.png")
image = pipeline.preprocess_image(image)

# 编码图像条件
print("[3/6] 编码图像条件...")
cond = pipeline.get_cond([image])

# 采样稀疏结构
print("[4/6] 采样稀疏结构...")
torch.manual_seed(seed)
coords = pipeline.sample_sparse_structure(
    cond,
    num_samples=1,
    sampler_params={
        "steps": 12,
        "cfg_strength": 7.5,
    },
)
print(f"  → 稀疏结构坐标数量: {coords.shape[0]}")

# 采样 SLAT
print("[5/6] 采样 SLAT...")
slat = pipeline.sample_slat(
    cond,
    coords,
    sampler_params={
        "steps": 12,
        "cfg_strength": 3,
    },
)
print(f"  → SLAT 特征形状: {slat.feats.shape}")

# 解码（用于生成最终文件）
print("[6/6] 解码...")
outputs = pipeline.decode_slat(slat, formats=["mesh", "gaussian"])

print("\n" + "=" * 60)
print("保存资产文件")
print("=" * 60)

# 1. 保存 voxels.ply
coords_np = coords.cpu().numpy()
with open(f"{output_dir}/voxels.ply", 'w') as f:
    f.write("ply\n")
    f.write("format ascii 1.0\n")
    f.write(f"element vertex {len(coords_np)}\n")
    f.write("property float x\n")
    f.write("property float y\n")
    f.write("property float z\n")
    f.write("end_header\n")
    for coord in coords_np:
        f.write(f"{coord[0]} {coord[1]} {coord[2]}\n")
print(f"✓ voxels.ply ({len(coords_np)} voxels)")

# 2. 保存 features.npz
feats = slat.feats.detach().cpu().numpy()
coords_slat = slat.coords.detach().cpu().numpy()
np.savez(
    f"{output_dir}/features.npz",
    feats=feats,
    coords=coords_slat,
)
print(f"✓ features.npz (feats: {feats.shape}, coords: {coords_slat.shape})")

# 3. 保存 GLB
glb = postprocessing_utils.to_glb(
    outputs['gaussian'][0],
    outputs['mesh'][0],
    simplify=0.95,
    texture_size=1024,
)
glb.export(f"{output_dir}/sample.glb")
print(f"✓ sample.glb")

# 4. 保存 PLY
outputs['gaussian'][0].save_ply(f"{output_dir}/sample.ply")
print(f"✓ sample.ply")

print("\n" + "=" * 60)
print(f"完成！资产保存在: {output_dir}")
print("=" * 60)
print("\n包含文件:")
print(f"  • voxels.ply      - {len(coords_np)} 个稀疏结构坐标")
print(f"  • features.npz    - SLAT 特征 (shape: {feats.shape})")
print(f"  • sample.glb      - 最终 3D 模型 (GLB 格式)")
print(f"  • sample.ply      - 最终 3D 模型 (PLY 格式)")
