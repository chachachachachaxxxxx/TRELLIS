"""
TRELLIS 图像到3D模型 —— 全流程深度注解版

整体架构概览:
=============
TRELLIS 的核心思想是 Structured LATent (SLAT) —— 结构化潜在表示。
整个生成过程分为以下阶段:

  [输入图片] 
      │
      ▼
  ┌─────────────────────────────────┐
  │ 阶段1: 图像预处理                │  去背景 → 裁剪 → 缩放到 518×518
  └─────────────────────────────────┘
      │
      ▼
  ┌─────────────────────────────────┐
  │ 阶段2: 图像特征编码 (DINOv2)     │  提取 1370 个 patch token，每个 1024 维
  └─────────────────────────────────┘
      │
      ▼
  ┌─────────────────────────────────┐
  │ 阶段3: 稀疏结构采样              │  Flow Matching + DiT → 生成 3D 占用网格
  │   (Sparse Structure Sampling)   │  确定哪些体素位置"有东西"
  └─────────────────────────────────┘
      │
      ▼
  ┌─────────────────────────────────┐
  │ 阶段4: 结构化潜在采样            │  在稀疏坐标上，用 Flow Matching + Sparse DiT
  │   (SLat Sampling)              │  生成每个体素的潜在特征向量
  └─────────────────────────────────┘
      │
      ▼
  ┌─────────────────────────────────┐
  │ 阶段5: 多格式解码               │  将 SLat 分别解码为:
  │   (Multi-format Decoding)       │  • 3D Gaussian Splatting
  │                                 │  • 辐射场 (Radiance Field / Strivec)
  │                                 │  • 网格 (Mesh，通过 FlexiCubes)
  └─────────────────────────────────┘
      │
      ▼
  ┌─────────────────────────────────┐
  │ 阶段6: 后处理与导出              │  渲染视频 / 导出 GLB / 导出 PLY
  └─────────────────────────────────┘
"""

import os
os.environ['SPCONV_ALGO'] = 'native'

import imageio
import torch
import torch.nn.functional as F
import numpy as np
from PIL import Image
from trellis.pipelines import TrellisImageTo3DPipeline
from trellis.utils import render_utils, postprocessing_utils


# ============================================================================
# 阶段 0: 加载 Pipeline（加载所有子模型）
# ============================================================================
#
# from_pretrained 内部流程:
#   1. 下载/读取 pipeline.json 配置文件
#   2. 根据配置加载以下 6 个子模型:
#      ┌──────────────────────────────────┬─────────────────────────────────────┐
#      │ 模型名称                         │ 作用                               │
#      ├──────────────────────────────────┼─────────────────────────────────────┤
#      │ image_cond_model (DINOv2)        │ 图像特征提取器                      │
#      │ sparse_structure_flow_model      │ 稀疏结构 Flow Model (3D DiT)       │
#      │ sparse_structure_decoder         │ 稀疏结构 VAE 解码器                 │
#      │ slat_flow_model                  │ SLat Flow Model (Sparse DiT)       │
#      │ slat_decoder_gs                  │ SLat → 3D Gaussian 解码器          │
#      │ slat_decoder_rf                  │ SLat → 辐射场解码器                 │
#      │ slat_decoder_mesh                │ SLat → 网格解码器                   │
#      └──────────────────────────────────┴─────────────────────────────────────┘
#   3. 初始化两个采样器:
#      - sparse_structure_sampler: FlowEulerGuidanceIntervalSampler
#      - slat_sampler:             FlowEulerGuidanceIntervalSampler
#   4. 加载 slat_normalization (均值/标准差) 用于反归一化

pipeline = TrellisImageTo3DPipeline.from_pretrained("microsoft/TRELLIS-image-large")
pipeline.cuda()

image = Image.open("assets/example_image/T.png")


# ============================================================================
# 阶段 1: 图像预处理 (pipeline.preprocess_image)
# ============================================================================
#
# 内部流程 (trellis/pipelines/trellis_image_to_3d.py 第82-116行):
#
#   1. 检查是否有 alpha 通道:
#      - 如果有有效 alpha → 直接使用（说明已经去过背景）
#      - 如果没有 → 用 rembg (U2Net 模型) 自动去除背景
#
#   2. 基于 alpha 通道计算物体的包围盒 (bounding box):
#      alpha = output_np[:, :, 3]
#      bbox = np.argwhere(alpha > 0.8 * 255)  # 找到所有不透明像素
#      → 计算中心点和尺寸，扩大 1.2 倍后裁剪
#
#   3. 将裁剪后的图片 resize 到 518×518:
#      → 518 = 37 × 14，因为 DINOv2 使用 14×14 的 patch
#      → 518/14 = 37，所以会产生 37×37 = 1369 个 patch token + 1 个 CLS token
#
#   4. 预乘 alpha (pre-multiply alpha):
#      output = output[:, :, :3] * output[:, :, 3:4]
#      → 背景区域变为纯黑 (0,0,0)，让模型只关注前景物体

preprocessed_image = pipeline.preprocess_image(image)


# ============================================================================
# 阶段 2: 图像特征编码 (pipeline.encode_image → pipeline.get_cond)
# ============================================================================
#
# 内部流程 (trellis/pipelines/trellis_image_to_3d.py 第118-160行):
#
#   步骤 2a: 图像转 Tensor
#     image → resize(518,518) → numpy → torch.Tensor [B, 3, 518, 518]
#
#   步骤 2b: ImageNet 标准化
#     transform = Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
#     → 这是 DINOv2 预训练时使用的标准化参数
#
#   步骤 2c: DINOv2 前向传播
#     features = dinov2_model(image, is_training=True)['x_prenorm']
#     → DINOv2 ViT-Large 将 518×518 图片分成 37×37=1369 个 patch
#     → 每个 patch 14×14 像素，加上 1 个 CLS token = 1370 个 token
#     → 输出 features shape: [B, 1370, 1024]
#     → 'x_prenorm' 指的是最后一层 LayerNorm 之前的特征
#
#   步骤 2d: 对输出做 LayerNorm
#     patchtokens = F.layer_norm(features, features.shape[-1:])
#     → shape: [B, 1370, 1024]
#     → 这 1370 个 token 就是图片的"条件信息"
#
#   步骤 2e: 构造 CFG 所需的负条件
#     neg_cond = torch.zeros_like(cond)
#     → 全零向量作为无条件引导的"空"条件

cond = pipeline.get_cond([preprocessed_image])
# cond = {
#     'cond':     Tensor [1, 1370, 1024],   # 正条件：DINOv2 提取的图像特征
#     'neg_cond': Tensor [1, 1370, 1024],   # 负条件：全零，用于 Classifier-Free Guidance
# }


# ============================================================================
# 阶段 3: 稀疏结构采样 (pipeline.sample_sparse_structure)
# ============================================================================
#
# 目标: 确定 3D 空间中哪些体素位置是"有物体的"（占用/occupancy）
# 输出: 稀疏坐标 coords [N_occupied, 4]  (batch_idx, x, y, z)
#
# ── 3a: 构造初始噪声 ──────────────────────────────────────────────────────
#   flow_model = self.models['sparse_structure_flow_model']
#   → 类型: SparseStructureFlowModel
#   → resolution = 16 (16×16×16 的体素网格，在潜在空间中)
#   → in_channels = 8 (8 通道的潜在向量)
#
#   noise = torch.randn(1, 8, 16, 16, 16)
#   → shape: [B=1, C=8, D=16, H=16, W=16]
#   → 这是一个标准的 3D 体积噪声，作为 Flow Matching 的起点 x_T

torch.manual_seed(1)

# ── 3b: Flow Matching 采样 (FlowEulerGuidanceIntervalSampler) ─────────────
#
#   Flow Matching 的数学原理:
#   x_t = (1 - t) * x_0 + (σ_min + (1 - σ_min) * t) * ε
#   其中 t ∈ [0, 1]，t=1 时完全是噪声，t=0 时是干净数据
#
#   采样过程（Euler 法，从 t=1 到 t=0）:
#     t_seq = [1.0, 0.917, 0.833, ..., 0.0]  (12步等距)
#     for each (t, t_prev) pair:
#
#       步骤 3b-i: 模型预测速度场 v
#         pred_v = model(x_t, t*1000, cond)
#         → SparseStructureFlowModel.forward() 内部:
#           1) patchify: [B,8,16,16,16] → [B, 8*2³=64, 8,8,8] → flatten → [B, 512, 64]
#              将 3D volume 按 patch_size=2 分块，每个 2³=8 个体素合并为 1 个 token
#              结果: 512 个 token，每个 64 维
#           2) input_layer: Linear(64 → model_channels)
#              将 patch 映射到模型隐藏维度（如 1024）
#           3) 加绝对位置编码:
#              h = h + pos_emb  (预计算的 8³=512 个 3D 位置的正弦编码)
#           4) 时间步嵌入:
#              t_emb = TimestepEmbedder(t*1000)
#              → 正弦编码 → MLP(256 → 1024 → 1024)
#           5) 经过 N 个 ModulatedTransformerCrossBlock:
#              每个 block 包含:
#                ├─ AdaLN: 用 t_emb 调制 LayerNorm (scale, shift, gate)
#                ├─ Self-Attention: 512 个 3D-token 之间的全局注意力
#                ├─ Cross-Attention: 512 个 3D-token 对 1370 个图像 token 做交叉注意力
#                │    → 这就是图像条件如何注入 3D 生成过程的关键!
#                └─ FFN: 前馈网络
#           6) out_layer: Linear(model_channels → 64)
#           7) unpatchify: [B, 512, 64] → [B, 8, 16, 16, 16] 还原回 3D volume
#
#       步骤 3b-ii: Classifier-Free Guidance (CFG)
#         对于 GuidanceIntervalSampler:
#           if cfg_interval[0] <= t <= cfg_interval[1]:
#             pred = model(x_t, t, cond)       # 有条件预测
#             neg_pred = model(x_t, t, neg_cond) # 无条件预测 (全零条件)
#             final = (1 + cfg_strength) * pred - cfg_strength * neg_pred
#             → cfg_strength 默认 7.5，增强图像条件的影响
#           else:
#             final = model(x_t, t, cond)       # 不做 guidance
#
#       步骤 3b-iii: 从 v 计算 x_0 和 ε
#         pred_x_0, pred_eps = v_to_xstart_eps(x_t, t, v)
#         → ε = (1-t)*v + x_t
#         → x_0 = (1-σ_min)*x_t - (σ_min + (1-σ_min)*t)*v
#
#       步骤 3b-iv: Euler 步进
#         x_{t-1} = x_t - (t - t_prev) * v
#         → 沿着速度场方向前进一小步
#
#   最终得到: z_s = x_0, shape [1, 8, 16, 16, 16]
#   → 这是稀疏结构的潜在表示

coords = pipeline.sample_sparse_structure(
    cond,
    num_samples=1,
    sampler_params={
        "steps": 12,
        "cfg_strength": 7.5,
    },
)

# ── 3c: 解码占用网格 ──────────────────────────────────────────────────────
#
#   decoder = self.models['sparse_structure_decoder']
#   → 类型: SparseStructureDecoder
#   → 这是一个 3D CNN 解码器 (对应论文中的 D_S)
#   → 内部结构: Conv3d → ResBlock3d × N → Upsample → ... → Conv3d
#   → 将 [1,8,16,16,16] 潜在向量解码为 [1,1,64,64,64] 的占用概率
#     (latent_channels=8 → 逐级上采样 → 最终 out_channels=1)
#
#   coords = torch.argwhere(decoder(z_s) > 0)[:, [0, 2, 3, 4]].int()
#   → 阈值化: 概率 > 0 的位置认为"有物体"
#   → argwhere 返回非零元素的坐标
#   → 取 [batch_idx, x, y, z] 四列
#   → 最终 coords shape: [N_occupied, 4]
#
# 例如一个简单的 "T" 形物体，可能有约 3000-5000 个占用体素

print(f"稀疏结构坐标数量: {coords.shape[0]}")
# 例如输出: 稀疏结构坐标数量: 4218


# ============================================================================
# 阶段 4: 结构化潜在采样 (pipeline.sample_slat)
# ============================================================================
#
# 目标: 在每个占用体素位置上生成"潜在特征向量"（SLat）
# 输入: 稀疏坐标 coords [N, 4] + 图像条件 cond
# 输出: SparseTensor，每个体素有一个特征向量
#
# ── 4a: 构造稀疏噪声 ─────────────────────────────────────────────────────
#   noise = sp.SparseTensor(
#       feats=torch.randn(N, in_channels),  # [N, 8] 随机噪声特征
#       coords=coords,                       # [N, 4] 稀疏坐标
#   )
#   → 与阶段3不同：这里是稀疏 tensor，只在有物体的位置存储数据
#   → in_channels = 8 (SLat 的潜在维度)
#
# ── 4b: Sparse Flow Matching 采样 ────────────────────────────────────────
#
#   使用 SLatFlowModel (sparse_structure_flow.py 中的 SLatFlowModel):
#
#   SLatFlowModel.forward(x: SparseTensor, t, cond):
#
#     1) input_layer: SparseLinear(8 → io_block_channels[0])
#        → 将 8 维潜在向量映射到初始通道数
#
#     2) Input Blocks (SparseResBlock3d × N，带下采样):
#        → 类似 U-Net 的编码器部分
#        → 每个 SparseResBlock3d:
#            ├─ LayerNorm → SiLU → SparseConv3d(3×3×3)
#            ├─ 时间步调制: norm(h) * (1+scale) + shift
#            │   (scale, shift 从 t_emb 通过 MLP 得到)
#            ├─ SiLU → SparseConv3d(3×3×3)
#            └─ 残差连接 skip connection
#        → 每隔几个 block 做一次 SparseDownsample(2)
#        → 保存每层的 skip features 用于后面的解码
#
#     3) 绝对位置编码:
#        h = h + pos_embedder(coords)
#        → 将每个体素的 (x,y,z) 坐标编码为高维向量
#
#     4) Transformer Blocks (ModulatedSparseTransformerCrossBlock × N):
#        → 这是模型的核心！对稀疏体素做 Transformer 注意力
#        → 每个 block:
#            ├─ AdaLN: 用 t_emb 调制（与阶段3的 DiT 相同的 adaptive LayerNorm）
#            ├─ Sparse Self-Attention: 稀疏体素之间的注意力
#            │    → 在这些"有物体"的体素之间建立全局关系
#            ├─ Cross-Attention: 体素 token 对 1370 个图像 token 的交叉注意力
#            │    → 再次注入图像条件信息
#            │    → 让每个体素"看到"图片中对应的外观和纹理信息
#            └─ FFN: 前馈网络
#
#     5) Output Blocks (SparseResBlock3d × N，带上采样):
#        → 类似 U-Net 的解码器部分
#        → 使用 skip connection: concat([h, skip_features])
#        → 每隔几个 block 做一次 SparseUpsample(2)
#
#     6) LayerNorm → out_layer: SparseLinear(model_channels → 8)
#        → 输出与输入同维度的速度场预测 v
#
#   采样过程与阶段3完全相同 (Euler + CFG)，只是数据结构是 SparseTensor:
#     for (t, t_prev) in [(1.0, 0.917), (0.917, 0.833), ...]:
#       v = model(x_t, t, cond)  # 带 CFG
#       x_{t-1} = x_t - (t - t_prev) * v
#
# ── 4c: 反归一化 ─────────────────────────────────────────────────────────
#   训练时对 SLat 做了标准化 (零均值、单位方差)
#   推理时需要还原:
#     slat = slat * std + mean
#   → slat_normalization 中存储了训练集上统计的均值和标准差

slat = pipeline.sample_slat(
    cond,
    coords,
    sampler_params={
        "steps": 12,
        "cfg_strength": 3,
    },
)
# slat 是一个 SparseTensor:
#   - slat.coords: [N, 4] 每个体素的坐标 (batch_idx, x, y, z)
#   - slat.feats:  [N, 8] 每个体素的潜在特征向量
# N ≈ 4000-5000（取决于物体复杂度）


# ============================================================================
# 阶段 5: 多格式解码 (pipeline.decode_slat)
# ============================================================================
#
# SLat 是一个统一的中间表示，可以解码为 3 种不同的 3D 格式。
# 每种格式由一个独立的 Sparse Transformer 解码器处理。
#
# 所有解码器共享相同的基础架构 (SparseTransformerBase):
#   input_layer: SparseLinear(8 → model_channels)
#   → pos_embedder(coords)  # 绝对位置编码
#   → SparseTransformerBlock × N:
#       ├─ Sparse Self-Attention (swin 窗口注意力，窗口大小 8)
#       │    → 交替使用: 不移位窗口 → 移位窗口 → 不移位窗口 → ...
#       │    → 类似 Swin Transformer 的移位窗口策略
#       ├─ LayerNorm
#       └─ FFN
#   → LayerNorm → out_layer

outputs = pipeline.decode_slat(slat, formats=['mesh', 'gaussian', 'radiance_field'])


# ── 5a: 3D Gaussian Splatting 解码 (SLatGaussianDecoder) ─────────────────
#
#   SLatGaussianDecoder.forward(slat):
#     1) SparseTransformerBase.forward(): 稀疏 Transformer 编码
#     2) out_layer: SparseLinear(model_channels → out_channels)
#        out_channels = num_gaussians × (3 + 3 + 3 + 4 + 1) = num_gaussians × 14
#        → 每个体素生成 num_gaussians 个高斯，每个高斯包含:
#            _xyz:         [num_gs, 3]     局部偏移量 (体素内的精确位置)
#            _features_dc: [num_gs, 1, 3]  球谐函数 DC 系数 (颜色)
#            _scaling:     [num_gs, 3]     缩放 (3个轴的大小)
#            _rotation:    [num_gs, 4]     旋转四元数
#            _opacity:     [num_gs, 1]     不透明度
#
#     3) to_representation():
#        → 将体素坐标转换为世界坐标:
#            xyz = (coord + 0.5) / resolution  # 归一化到 [0, 1]
#        → 叠加学习到的局部偏移:
#            offset = tanh(raw_offset) / resolution * 0.5 * voxel_size
#            → Hammersley 低差异序列提供初始扰动
#            → 确保高斯核心分布在体素内
#        → 组装成 Gaussian 对象
#
# 最终每个体素可能包含 32 个小高斯核，总共约 4000×32 = 128000 个高斯

gaussian = outputs['gaussian'][0]
print(f"3D Gaussian 数量: {gaussian._xyz.shape[0]}")


# ── 5b: 辐射场解码 (SLatRadianceFieldDecoder) ───────────────────────────
#
#   SLatRadianceFieldDecoder.forward(slat):
#     1) SparseTransformerBase.forward(): 稀疏 Transformer 编码
#     2) out_layer → to_representation():
#        → 生成 Strivec (Tri-Vector Radiance Field) 表示:
#            trivec:       [rank, 3, dim]  三向量分解的特征
#            density:      [rank]          密度
#            features_dc:  [rank, 1, 3]    颜色
#        → Strivec 是一种高效的辐射场参数化:
#            在每个体素内用 3 个正交方向的向量组合来表示体积数据
#            比 NeRF 的 MLP 更高效

radiance_field = outputs['radiance_field'][0]


# ── 5c: 网格解码 (SLatMeshDecoder) ──────────────────────────────────────
#
#   SLatMeshDecoder.forward(slat):
#     1) SparseTransformerBase.forward(): 稀疏 Transformer 编码
#        → 输入分辨率: 64³ (SLat 的分辨率)
#
#     2) 上采样 (2 个 SparseSubdivideBlock3d):
#        → 64³ → 128³ → 256³
#        → 每个 SparseSubdivideBlock3d:
#            ├─ GroupNorm → SiLU
#            ├─ SparseSubdivide(): 将每个体素细分为 8 个子体素
#            ├─ SparseConv3d(3×3×3) → GroupNorm → SiLU → SparseConv3d(3×3×3)
#            └─ 残差连接
#        → model_channels → model_channels//4 → model_channels//8
#
#     3) out_layer: SparseLinear → 输出 FlexiCubes 所需的特征:
#        → sdf:     [8, 1]    8 个顶点的 SDF 值 (有符号距离场)
#        → deform:  [8, 3]    8 个顶点的变形偏移
#        → weights: [21]      FlexiCubes 的 beta(12) + alpha(8) + gamma(1)
#        → color:   [8, 6]    8 个顶点的颜色+法线 (可选)
#
#     4) SparseFeatures2Mesh (FlexiCubes 提取网格):
#        → sparse_cube2verts(): 将 cube 特征插值到顶点
#        → get_dense_attrs(): 填充到稠密网格
#        → FlexiCubes 算法:
#            a) 基于 SDF 场确定表面位置 (类似 Marching Cubes)
#            b) 用 beta, alpha, gamma 权重调整网格拓扑
#            c) 用 deform 偏移微调顶点位置
#            d) 插值颜色属性到最终顶点
#        → 输出: MeshExtractResult(vertices, faces, vertex_attrs)

mesh = outputs['mesh'][0]
print(f"网格顶点数: {mesh.vertices.shape[0]}, 面片数: {mesh.faces.shape[0]}")


# ============================================================================
# 阶段 6: 后处理与导出
# ============================================================================

# ── 6a: 渲染视频 ─────────────────────────────────────────────────────────
video = render_utils.render_video(gaussian)['color']
imageio.mimsave("sample_gs.mp4", video, fps=30)

video = render_utils.render_video(radiance_field)['color']
imageio.mimsave("sample_rf.mp4", video, fps=30)

video = render_utils.render_video(mesh)['normal']
imageio.mimsave("sample_mesh.mp4", video, fps=30)

# ── 6b: 导出 GLB (textured mesh) ─────────────────────────────────────────
#
# to_glb 内部流程:
#   1) simplify: 使用网格简化算法减少三角面数 (保留 5% = 1-0.95)
#   2) texture baking: 
#      → 用 3D Gaussian Splatting 从多个视角渲染 2D 图像
#      → 将这些渲染结果通过 UV 映射烘焙到网格纹理上
#      → texture_size=1024 表示生成 1024×1024 的纹理贴图
#   3) 组装成 trimesh.Scene → 导出 GLB 格式

glb = postprocessing_utils.to_glb(
    gaussian,
    mesh,
    simplify=0.95,
    texture_size=1024,
)
glb.export("sample.glb")

# ── 6c: 导出 PLY (3D Gaussians) ──────────────────────────────────────────
gaussian.save_ply("sample.ply")

print("全流程完成！生成文件:")
print("  - sample_gs.mp4:   3D Gaussian 渲染视频")
print("  - sample_rf.mp4:   辐射场渲染视频")
print("  - sample_mesh.mp4: 网格法线渲染视频")
print("  - sample.glb:      带纹理的 3D 网格 (可在浏览器中查看)")
print("  - sample.ply:      3D Gaussian 点云")
