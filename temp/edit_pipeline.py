"""
VoxHammer 编辑流程核心文件。

实现论文 Sec.3 中描述的无训练3D局部编辑框架：
  1) 将输入3D资产通过二阶Taylor改进的Euler方案（RF-Solver）反转到终端噪声，
     同时在每个时间步缓存潜在向量和注意力层的K/V张量；
  2) 从反转噪声出发进行去噪，通过掩码引导的潜在替换(Latent Replacement)
     和键值替换(Key-Value Replacement)实现精确的局部编辑。

整个流程分为两个阶段：
  - ST阶段 (Structure Stage)：在 64^3 密集体素网格上操作，预测稀疏结构
  - SLAT阶段 (Sparse Latent Stage)：在稀疏体素上操作，生成精细几何和纹理
"""

import os
os.environ["ATTN_BACKEND"] = "sdpa"
os.environ["SPCONV_ALGO"] = "native"

import gc
import rembg
import psutil
import torch
import utils3d
import numpy as np
import torch.nn.functional as F
import torchvision.transforms as transforms

from typing import *
from PIL import Image
from tqdm import tqdm
from types import MethodType

import trellis.modules.sparse as sp
from trellis.utils import postprocessing_utils
from trellis.modules.spatial import patchify, unpatchify
from trellis.pipelines import TrellisTextTo3DPipeline, TrellisImageTo3DPipeline
from trellis.pipelines.samplers.flow_euler import FlowEulerGuidanceIntervalSampler

##############################################################################
# 数据加载与预处理工具函数
##############################################################################

def ply_to_coords(ply_path):
    """读取PLY文件中的顶点坐标，映射到 64^3 体素网格的整数坐标。"""
    position = utils3d.io.read_ply(ply_path)[0]
    coords = ((torch.tensor(position) + 0.5) * 64).int().contiguous().cuda()
    return coords

def coords_to_voxel(coords):
    """将稀疏体素坐标转换为 [1,1,64,64,64] 的密集二值占用网格。"""
    voxel = torch.zeros(1, 1, 64, 64, 64, dtype=torch.float)
    voxel[:, 0, coords[:, 0], coords[:, 1], coords[:, 2]] = 1
    return voxel.cuda()

def feats_to_slat(pipeline, feats_path):
    """加载预提取的DINOv2特征，通过SLAT编码器生成结构化潜在(Structured Latent)。
    
    结构化潜在 SLAT 是论文中 {(z_i, p_i)} 的表示形式，
    每个活动体素 p_i 关联一个局部潜在向量 z_i ∈ R^C，编码精细几何和外观。
    """
    feats = np.load(feats_path)
    feats_tensor = sp.SparseTensor(
        feats=torch.from_numpy(feats["patchtokens"]).float(),
        coords=torch.cat([torch.zeros(feats["patchtokens"].shape[0], 1).int(), torch.from_numpy(feats["indices"]).int()], dim=1)).cuda()
    feats_encoder = pipeline.models["slat_encoder"]
    slat = feats_encoder(feats_tensor, sample_posterior=False)
    return slat

def image_rgb(img_path):
    """读取图像并确保输出为RGB模式（RGBA图像会合成到白色背景上）。"""
    image = Image.open(img_path)
    if image.mode == "RGB":
        image_rgb = image
    else:
        image = image.convert("RGBA")
        background = Image.new("RGB", image.size, (255, 255, 255))
        background.paste(image, mask=image.split()[3])
        image_rgb = background
    return image_rgb

def get_process_rss_gib():
    """返回当前进程的 RSS（GiB），用于观察系统内存占用。"""
    process = psutil.Process(os.getpid())
    return process.memory_info().rss / 1024**3

def preprocess_image(pipeline, img_src_path, img_tgt_path, img_mask_path):
    """预处理源图像、目标编辑图像和2D掩码。
    
    对源图和目标图进行背景去除(rembg)，基于二者的联合边界框进行
    居中裁剪并缩放到 518×518（DINOv2 输入尺寸），最后用alpha通道
    将背景区域置零。
    """
    img_src = image_rgb(img_src_path)
    img_tgt = image_rgb(img_tgt_path)
    img_mask = Image.open(img_mask_path).convert("L")
    max_size = max(img_src.size)
    scale = min(1, 1024 / max_size)
    resize_size = (int(img_src.width * scale), int(img_src.height * scale))
    if scale < 1:
        img_src = img_src.resize(resize_size, Image.Resampling.LANCZOS)
        img_tgt = img_tgt.resize(resize_size, Image.Resampling.LANCZOS)
        img_mask = img_mask.resize(resize_size, Image.Resampling.LANCZOS)
    if getattr(pipeline, "rembg_session", None) is None:
        pipeline.rembg_session = rembg.new_session("u2net")
    pre_img_src = rembg.remove(img_src, session=pipeline.rembg_session)
    pre_img_tgt = rembg.remove(img_tgt, session=pipeline.rembg_session)
    pre_img_src_np = np.array(pre_img_src)
    pre_img_tgt_np = np.array(pre_img_tgt)
    alpha_src = pre_img_src_np[:, :, 3]
    alpha_tgt = pre_img_tgt_np[:, :, 3]
    bbox_src = np.argwhere(alpha_src > 0.8 * 255)
    bbox_tgt = np.argwhere(alpha_tgt > 0.8 * 255)
    bbox = (
        min(np.min(bbox_src[:, 1]), np.min(bbox_tgt[:, 1])),
        min(np.min(bbox_src[:, 0]), np.min(bbox_tgt[:, 0])),
        max(np.max(bbox_src[:, 1]), np.max(bbox_tgt[:, 1])),
        max(np.max(bbox_src[:, 0]), np.max(bbox_tgt[:, 0])),
    )
    center = (bbox[0] + bbox[2]) / 2, (bbox[1] + bbox[3]) / 2
    size = max(bbox[2] - bbox[0], bbox[3] - bbox[1])
    size = int(size * 1.2)
    bbox = (center[0] - size // 2, center[1] - size // 2, center[0] + size // 2, center[1] + size // 2,)
    pre_img_src = pre_img_src.crop(bbox)
    pre_img_tgt = pre_img_tgt.crop(bbox)
    pre_img_mask = img_mask.crop(bbox)
    pre_img_src = pre_img_src.resize((518, 518), Image.Resampling.LANCZOS)
    pre_img_tgt = pre_img_tgt.resize((518, 518), Image.Resampling.LANCZOS)
    pre_img_mask = pre_img_mask.resize((518, 518), Image.Resampling.LANCZOS)
    pre_img_src = np.array(pre_img_src).astype(np.float32) / 255
    pre_img_tgt = np.array(pre_img_tgt).astype(np.float32) / 255
    pre_img_src = pre_img_src[:, :, :3] * pre_img_src[:, :, 3:4]
    pre_img_tgt = pre_img_tgt[:, :, :3] * pre_img_tgt[:, :, 3:4]
    pre_img_src = Image.fromarray((pre_img_src * 255).astype(np.uint8))
    pre_img_tgt = Image.fromarray((pre_img_tgt * 255).astype(np.uint8))
    return pre_img_src, pre_img_tgt, pre_img_mask

def gaussian_kernel1d(sigma, dtype, device, truncate=3.0):
    """构造归一化的一维高斯核。sigma<=0 时返回长度为1的单位核。"""
    if sigma <= 0:
        return torch.ones(1, dtype=dtype, device=device)
    radius = max(1, int(truncate * sigma + 0.5))
    x = torch.arange(-radius, radius + 1, dtype=dtype, device=device)
    kernel = torch.exp(-(x ** 2) / (2 * sigma ** 2))
    kernel = kernel / kernel.sum()
    return kernel

def gaussian_blur3d(mask, sigma):
    """对 [B,C,D,H,W] 掩码做 3D Gaussian blur。"""
    kernel_1d = gaussian_kernel1d(sigma, mask.dtype, mask.device)
    radius = kernel_1d.shape[0] // 2
    kernel_3d = kernel_1d[:, None, None] * kernel_1d[None, :, None] * kernel_1d[None, None, :]
    kernel_3d = kernel_3d.view(1, 1, *kernel_3d.shape)
    return F.conv3d(mask, kernel_3d, padding=radius)

def make_soft_mask3d(mask, dilation=2, sigma=1.0):
    """将编辑掩码从硬边界转换为 dilation + Gaussian falloff 的 soft mask。"""
    hard_mask = mask.float()
    if dilation > 0:
        kernel_size = 2 * dilation + 1
        dilated_mask = F.max_pool3d(hard_mask, kernel_size=kernel_size, stride=1, padding=dilation)
    else:
        dilated_mask = hard_mask
    if sigma > 0:
        soft_mask = gaussian_blur3d(dilated_mask, sigma)
        # 保持原始编辑区域仍为 1，只在边界外形成平滑过渡带。
        soft_mask = torch.maximum(hard_mask, soft_mask)
    else:
        soft_mask = dilated_mask
    return soft_mask.clamp_(0.0, 1.0)

def coords_to_flat_indices(coords, resolution=64):
    """将稀疏坐标编码为一维整数索引，用于行级集合运算。"""
    coords = coords.long()
    if coords.shape[1] == 3:
        return coords[:, 0] * resolution * resolution + coords[:, 1] * resolution + coords[:, 2]
    if coords.shape[1] == 4:
        return (
            coords[:, 0] * resolution * resolution * resolution +
            coords[:, 1] * resolution * resolution +
            coords[:, 2] * resolution +
            coords[:, 3]
        )
    raise ValueError(f"coords_to_flat_indices 仅支持 [...,3] 或 [...,4] 坐标，收到: {coords.shape}")

def make_slat_soft_weights(coords_tgt, coords_preserve, dilation=2, sigma=1.0, resolution=64):
    """为SLAT阶段的保留区域坐标生成 soft mask 权重。
    
    返回值语义与 ST 阶段一致：编辑权重=1，缓存/源特征权重=(1-weight)。
    这里只为 preserve 坐标返回权重；edit 坐标始终保持完全可编辑。
    """
    if coords_preserve.shape[0] == 0:
        return torch.zeros(0, 1, dtype=torch.float, device=coords_tgt.device)

    preserve_codes = coords_to_flat_indices(coords_preserve, resolution)
    tgt_codes = coords_to_flat_indices(coords_tgt, resolution)
    coords_edit = coords_tgt[~torch.isin(tgt_codes, preserve_codes)]

    if coords_edit.shape[0] == 0:
        return torch.zeros(coords_preserve.shape[0], 1, dtype=torch.float, device=coords_tgt.device)

    edit_mask = torch.zeros(1, 1, resolution, resolution, resolution, dtype=torch.float, device=coords_tgt.device)
    edit_mask[:, 0, coords_edit[:, 1], coords_edit[:, 2], coords_edit[:, 3]] = 1
    soft_mask = make_soft_mask3d(edit_mask, dilation=dilation, sigma=sigma)
    preserve_edit_weights = soft_mask[0, 0, coords_preserve[:, 1], coords_preserve[:, 2], coords_preserve[:, 3]]
    return preserve_edit_weights.unsqueeze(1).contiguous().float()

##############################################################################
# 掩码构建函数 —— 论文 Sec.3.3 中编辑掩码 M^ss 和 W 的实现
##############################################################################

def ply_to_ss_mask(coords_mask, pre_mask, soft_mask=False, soft_mask_dilation=2, soft_mask_sigma=1.0):
    """为ST阶段构建多级掩码。
    
    Args:
        coords_mask: 保留区域(Ωkeep)的体素坐标，即不被编辑的部分
        pre_mask: 2D图像编辑掩码（用于cross-attention的KV替换）
        soft_mask: 是否将 M^ss 替换为 dilation + Gaussian falloff 的 soft mask
        soft_mask_dilation: soft mask 的 3D 膨胀半径（voxel）
        soft_mask_sigma: soft mask 的 3D Gaussian sigma（voxel）
    
    Returns:
        voxel_mask:       [1,1,64,64,64] 体素级掩码，保留=0，编辑=1
                          用于ST去噪后合并：voxel = voxel_new * mask + voxel_src * (1-mask)
        ss_latent_mask:   [1,8,16,16,16] 潜在空间掩码，对应论文公式(4)中的 M^ss
                          经过两次 patchify (64→32→16) 后在 patch 维度取 all()
        ss_self_kv_mask:  [1,16,4096,64] self-attention的KV替换掩码，对应论文公式(6)(7)中的 W
                          形状适配 [batch, num_heads, seq_len, head_dim]
        cross_kv_mask:    [1,16,1374,64] cross-attention的KV替换掩码
                          从518×518图像掩码映射到37×37的DINOv2 patch token空间
    """
    # 体素掩码：保留区域=0，编辑区域=1
    voxel_mask = torch.ones(1, 1, 64, 64, 64, dtype=torch.float)
    voxel_mask[:, 0, coords_mask[:, 0], coords_mask[:, 1], coords_mask[:, 2]] = 0
    voxel_mask = voxel_mask.cuda()

    # 潜在空间掩码 M^ss：从 64^3 体素空间下采样到 16^3 潜在空间
    # 默认保持原论文中的硬掩码逻辑；可选替换为 dilation + Gaussian falloff soft mask。
    ss_latent_mask_hard = voxel_mask.reshape(1, 1, 32, 2, 32, 2, 32, 2)
    ss_latent_mask_hard = ss_latent_mask_hard.permute(0, 1, 3, 5, 7, 2, 4, 6)
    ss_latent_mask_hard = ss_latent_mask_hard.reshape(1, 8, 32, 32, 32)
    ss_latent_mask_hard = ss_latent_mask_hard.reshape(1, 8, 16, 2, 16, 2, 16, 2)
    ss_latent_mask_hard = ss_latent_mask_hard.permute(0, 1, 3, 5, 7, 2, 4, 6)
    ss_latent_mask_hard = ss_latent_mask_hard.reshape(1, 64, 16, 16, 16)
    # 只有patch内所有体素都是编辑区域(=1)时, 该patch才标记为编辑;
    # 只要有任一保留体素(=0), 该patch即为保留(=0)
    ss_latent_mask_hard = ss_latent_mask_hard.all(dim=1, keepdim=True)
    ss_latent_mask_hard = ss_latent_mask_hard.repeat(1, 8, 1, 1, 1).contiguous().float().cuda()
    if soft_mask:
        voxel_mask_soft = make_soft_mask3d(voxel_mask, dilation=soft_mask_dilation, sigma=soft_mask_sigma)
        # 64→16 的 soft 下采样直接用平均池化，保留 [0,1] 连续权重。
        ss_latent_mask = F.avg_pool3d(voxel_mask_soft, kernel_size=4, stride=4)
        ss_latent_mask = ss_latent_mask.repeat(1, 8, 1, 1, 1).contiguous().float()
    else:
        ss_latent_mask = ss_latent_mask_hard

    # Self-attention KV掩码 W^self: 从潜在空间掩码展平为注意力维度
    # soft_mask 仅作用于 M^ss；W 仍保持原始硬掩码实现，避免改变 KV replacement 语义。
    # 形状变换: [1,8,16,16,16] → [1,8,4096] → [1,4096,8] → all → [1,4096,1]
    # → repeat → [1,4096,1024] → reshape 为 [1,16,4096,64] (num_heads=16, head_dim=64)
    ss_self_kv_mask = ss_latent_mask_hard.reshape(1, 8, -1)
    ss_self_kv_mask = ss_self_kv_mask.permute(0, 2, 1)
    ss_self_kv_mask = ss_self_kv_mask.all(dim=2, keepdim=True)
    ss_self_kv_mask = ss_self_kv_mask.repeat(1, 1, 1024)
    ss_self_kv_mask = ss_self_kv_mask.reshape(1, 4096, 16, 64)
    ss_self_kv_mask = ss_self_kv_mask.permute(0, 2, 1, 3).contiguous().float().cuda()

    # Cross-attention KV掩码: 从2D图像掩码 (518×518) 映射到 DINOv2 patch token 空间
    if pre_mask is None:
        cross_kv_mask = None
    else:
        img_mask = transforms.ToTensor()(pre_mask)
        img_mask = (img_mask > 0).float()
        # 518×518 → 37×37 个patch (每个patch 14×14像素)
        cross_kv_mask = img_mask.reshape(37, 14, 37, 14)
        cross_kv_mask = cross_kv_mask.permute(1, 3, 0, 2)
        cross_kv_mask = cross_kv_mask.reshape(1, 196, 37, 37)
        # 只要patch内有任意像素属于编辑区域, 该token就标记为编辑
        cross_kv_mask = cross_kv_mask.any(dim=1, keepdim=True)
        cross_kv_mask = cross_kv_mask.repeat(1, 1024, 1, 1)
        cross_kv_mask = cross_kv_mask.reshape(1, 1024, 1369)
        cross_kv_mask = cross_kv_mask.permute(0, 2, 1)
        # 前5个token是特殊token（cls等），掩码设为1，始终使用目标条件的新KV
        cross_kv_mask = torch.cat((torch.ones(1, 5, 1024), cross_kv_mask), dim=1)
        # reshape 为注意力维度 [1, num_heads=16, seq_len=1374, head_dim=64]
        cross_kv_mask = cross_kv_mask.reshape(1, 1374, 16, 64)
        cross_kv_mask = cross_kv_mask.permute(0, 2, 1, 3).contiguous().float().cuda()
    return voxel_mask, ss_latent_mask, ss_self_kv_mask, cross_kv_mask

def ply_to_slat_mask(coords_tgt, coords_mask):
    """为SLAT阶段构建self-attention的KV替换掩码。
    
    由于SLAT阶段操作在稀疏体素上，KV替换通过坐标匹配实现，
    而非ST阶段的密集掩码乘法。此函数将保留区域坐标下采样到
    SLAT分辨率空间（factor=2的下采样），并去重得到唯一坐标集合。
    
    Returns:
        slat_self_kv_mask: 保留区域在SLAT空间中的坐标，
                           用于在 slat_attn_forward 中匹配并替换KV特征
    """
    coords_mask = torch.cat([torch.zeros(coords_mask.shape[0], 1).int().cuda(), coords_mask], dim=1)
    factor = (2, 2, 2)
    coord_tgt = list(coords_tgt.unbind(dim=-1))
    coord_mask = list(coords_mask.unbind(dim=-1))
    # 按 factor=(2,2,2) 对 xyz 坐标进行下采样
    for i, f in enumerate(factor):
        coord_tgt[i + 1] = coord_tgt[i + 1] // f
        coord_mask[i + 1] = coord_mask[i + 1] // f
    # 将多维坐标编码为一维唯一标识符，然后去重
    MAX = [coord_tgt[i + 1].max().item() + 1 for i in range(3)]
    OFFSET = torch.cumprod(torch.tensor(MAX[::-1]), 0).tolist()[::-1] + [1]
    code_tgt = sum([c * o for c, o in zip(coord_tgt, OFFSET)])
    code_mask = sum([c * o for c, o in zip(coord_mask, OFFSET)])
    code_tgt, idx_tgt = code_tgt.unique(return_inverse=True)
    code_mask, idx_mask = code_mask.unique(return_inverse=True)
    new_coords_tgt = torch.stack([code_tgt // OFFSET[0]] + [(code_tgt // OFFSET[i + 1]) % MAX[i] for i in range(3)], dim=-1)
    new_coords_mask = torch.stack([code_mask // OFFSET[0]] + [(code_mask // OFFSET[i + 1]) % MAX[i] for i in range(3)], dim=-1)
    slat_self_kv_mask = new_coords_mask.contiguous().cuda()
    return slat_self_kv_mask

##############################################################################
# ST阶段 —— 注入的 forward 函数
# 通过 MethodType 在运行时替换原始 TRELLIS 模型的 forward，
# 实现论文 Sec.3.2 的KV缓存和 Sec.3.3 的KV替换，无需重训练或修改权重。
##############################################################################

def ss_attn_forward(self, x, context=None, indices=None, ss_kv=None, kv_mask=None, t_latent=None, order=None, pos=None, layer=None, is_text=False):
    """ST阶段注意力层的注入 forward —— 实现KV缓存与替换。
    
    论文 Sec.3.2 & 3.3 的核心实现之一。
    - 反转时(kv_mask=None)：计算K/V后缓存到 ss_kv 字典中
    - 编辑时(kv_mask≠None)：用掩码混合新K/V和缓存K/V，实现公式(6)(7)
    
    KV字典的key格式: "{t_latent}_{order}_{pos}_{layer}_{attn_type}_k/v"
    对应论文中按"时间、块顺序、位置编码、层ID和注意力类型"索引的 KV^st。
    """
    B, L, C = x.shape
    if self._type == "self":
        qkv = self.to_qkv(x)
        qkv = qkv.reshape(B, L, 3, self.num_heads, -1)
        q, k, v = qkv.unbind(dim=2)
        if self.use_rope:
            q, k = self.rope(q, k, indices)
    else:
        Lkv = context.shape[1]
        q = self.to_q(x)
        kv = self.to_kv(context)
        q = q.reshape(B, L, self.num_heads, -1)
        kv = kv.reshape(B, Lkv, 2, self.num_heads, -1)
        k, v = kv.unbind(dim=2)
    if self.qk_rms_norm:
        q = self.q_rms_norm(q)
        k = self.k_rms_norm(k)
    q = q.permute(0, 2, 1, 3)
    k = k.permute(0, 2, 1, 3)
    v = v.permute(0, 2, 1, 3)
    # 文本条件的cross-attention不参与KV缓存/替换
    if not (is_text and (self._type != "self")):
        if kv_mask is None:
            # 反转阶段: 缓存当前时间步的K/V到CPU内存
            ss_kv[f"{t_latent}_{order}_{pos}_{layer}_{self._type}_k"] = k.cpu()
            ss_kv[f"{t_latent}_{order}_{pos}_{layer}_{self._type}_v"] = v.cpu()
        else:
            # 编辑阶段: 论文公式(6)(7) —— K/V替换
            # K = W ⊙ K_new + (1-W) ⊙ K_cache
            # V = W ⊙ V_new + (1-W) ⊙ V_cache
            # kv_mask 中编辑区域=1(使用新K/V), 保留区域=0(使用缓存K/V)
            k = k * kv_mask + ss_kv[f"{t_latent}_{order}_{pos}_{layer}_{self._type}_k"].cuda() * (1 - kv_mask)
            v = v * kv_mask + ss_kv[f"{t_latent}_{order}_{pos}_{layer}_{self._type}_v"].cuda() * (1 - kv_mask)
            k = k.type(q.dtype)
            v = v.type(q.dtype)
    h = F.scaled_dot_product_attention(q, k, v)
    h = h.permute(0, 2, 1, 3)
    h = h.reshape(B, L, -1)
    h = self.to_out(h)
    return h

def ss_trsfmr_forward(self, x, mod, context, ss_kv, self_kv_mask, cross_kv_mask, t_latent, order, pos, layer, is_text):
    """ST阶段 Transformer 块的注入 forward。
    
    在原始的 ModulatedTransformerCrossBlock 基础上增加了 KV缓存/替换 的参数透传。
    结构: AdaLN → Self-Attn(+KV操作) → Cross-Attn(+KV操作) → FFN
    """
    if self.share_mod:
        shift_msa, scale_msa, gate_msa, shift_mlp, scale_mlp, gate_mlp = mod.chunk(6, dim=1)
    else:
        shift_msa, scale_msa, gate_msa, shift_mlp, scale_mlp, gate_mlp = self.adaLN_modulation(mod).chunk(6, dim=1)
    h = self.norm1(x)
    h = h * (1 + scale_msa.unsqueeze(1)) + shift_msa.unsqueeze(1)
    h = self.self_attn(h, ss_kv=ss_kv, kv_mask=self_kv_mask, t_latent=t_latent, order=order, pos=pos, layer=layer, is_text=is_text)
    h = h * gate_msa.unsqueeze(1)
    x = x + h
    h = self.norm2(x)
    h = self.cross_attn(h, context, ss_kv=ss_kv, kv_mask=cross_kv_mask, t_latent=t_latent, order=order, pos=pos, layer=layer, is_text=is_text)
    x = x + h
    h = self.norm3(x)
    h = h * (1 + scale_mlp.unsqueeze(1)) + shift_mlp.unsqueeze(1)
    h = self.mlp(h)
    h = h * gate_mlp.unsqueeze(1)
    x = x + h
    return x

def ss_flow_forward(self, x, t, cond, ss_kv, self_kv_mask, cross_kv_mask, t_latent, order, pos, is_text):
    """ST阶段 SparseStructureFlowModel 的注入 forward。
    
    替换原始 forward 以将 KV缓存/替换 参数逐层传递给每个 Transformer 块。
    数据流: patchify(64→16) → input_layer → pos_emb → Transformer blocks → out_layer → unpatchify
    """
    assert [*x.shape] == [x.shape[0], self.in_channels, *[self.resolution] * 3], \
        f"Input shape mismatch, got {x.shape}, expected {[x.shape[0], self.in_channels, *[self.resolution] * 3]}"
    h = patchify(x, self.patch_size)
    h = h.view(*h.shape[:2], -1).permute(0, 2, 1).contiguous()
    h = self.input_layer(h)
    h = h + self.pos_emb[None]
    t_emb = self.t_embedder(t)
    if self.share_mod:
        t_emb = self.adaLN_modulation(t_emb)
    t_emb = t_emb.type(self.dtype)
    h = h.type(self.dtype)
    cond = cond.type(self.dtype)
    for layer, block in enumerate(self.blocks):
        h = block(h, t_emb, cond, ss_kv, self_kv_mask, cross_kv_mask, t_latent, order, pos, layer, is_text)
    h = h.type(x.dtype)
    h = F.layer_norm(h, h.shape[-1:])
    h = self.out_layer(h)
    h = h.permute(0, 2, 1).view(h.shape[0], h.shape[2], *[self.resolution // self.patch_size] * 3)
    h = unpatchify(h, self.patch_size).contiguous()
    return h

##############################################################################
# SLAT阶段 —— 注入的 forward 函数
# 与ST阶段类似，但操作在 SparseTensor 上，KV替换需要通过坐标匹配实现。
##############################################################################

def slat_attn_forward(self, x, context=None, slat_kv=None, kv_mask=None, t_latent=None, order=None, pos=None, layer=None, is_text=False):
    """SLAT阶段注意力层的注入 forward —— 稀疏KV缓存与替换。
    
    与 ss_attn_forward 的关键区别:
    - Self-Attention: K/V 是 SparseTensor，替换时需通过坐标匹配
      找到保留区域 Ωkeep 中对应位置的特征，然后逐点替换
    - Cross-Attention: K/V 是密集张量(来自图像条件)，替换方式与ST阶段相同
    """
    if self._type == "self":
        qkv = self._linear(self.to_qkv, x)
        qkv = self._fused_pre(qkv, num_fused=3)
        if self.use_rope:
            qkv = self._rope(qkv)
        q, k, v = qkv.unbind(dim=1)
        if self.qk_rms_norm:
            q = self.q_rms_norm(q)
            k = self.k_rms_norm(k)
        if kv_mask is None:
            # 反转阶段: 缓存稀疏K/V（包含坐标和特征）
            slat_kv[f"{t_latent}_{order}_{pos}_{layer}_{self._type}_k"] = k.cpu()
            slat_kv[f"{t_latent}_{order}_{pos}_{layer}_{self._type}_v"] = v.cpu()
        else:
            if kv_mask.shape[0] != 0 and slat_kv:
                # 编辑阶段: 通过坐标匹配替换保留区域的K特征
                # kv_mask 是保留区域的坐标集合 (非二值掩码)
                # match_1: 当前K的坐标与保留区域坐标的匹配矩阵
                # match_2: 缓存K的坐标与保留区域坐标的匹配矩阵
                match_1_k = (k.coords.unsqueeze(1) == kv_mask.unsqueeze(0)).all(dim=-1)
                match_2_k = (slat_kv[f"{t_latent}_{order}_{pos}_{layer}_{self._type}_k"].coords.cuda().unsqueeze(1) == kv_mask.unsqueeze(0)).all(dim=-1)
                idx_1_k = match_1_k.float().argmax(0)
                idx_2_k = match_2_k.float().argmax(0)
                feats_k = k.feats.clone()
                # 保留区域的K特征用缓存值覆盖
                feats_k[idx_1_k] = slat_kv[f"{t_latent}_{order}_{pos}_{layer}_{self._type}_k"].feats.cuda()[idx_2_k]
                k = k.replace(feats_k)
                k = k.type(q.dtype)

                # 同样替换保留区域的V特征
                match_1_v = (v.coords.unsqueeze(1) == kv_mask.unsqueeze(0)).all(dim=-1)
                match_2_v = (slat_kv[f"{t_latent}_{order}_{pos}_{layer}_{self._type}_v"].coords.cuda().unsqueeze(1) == kv_mask.unsqueeze(0)).all(dim=-1)
                idx_1_v = match_1_v.float().argmax(0)
                idx_2_v = match_2_v.float().argmax(0)
                feats_v = v.feats.clone()
                feats_v[idx_1_v] = slat_kv[f"{t_latent}_{order}_{pos}_{layer}_{self._type}_v"].feats.cuda()[idx_2_v]
                v = v.replace(feats_v)
                v = v.type(q.dtype)
        # 从 SparseTensor 提取特征张量，转换为标准注意力输入格式
        h = q
        q = q.feats
        k = k.feats
        v = v.feats
        q = q.unsqueeze(0)
        k = k.unsqueeze(0)
        v = v.unsqueeze(0)
        q = q.permute(0, 2, 1, 3)
        k = k.permute(0, 2, 1, 3)
        v = v.permute(0, 2, 1, 3)
    else:
        # Cross-Attention: K/V来自图像条件(密集张量)，处理方式与ST阶段类似
        q = self._linear(self.to_q, x)
        q = self._reshape_chs(q, (self.num_heads, -1))
        kv = self._linear(self.to_kv, context)
        kv = self._fused_pre(kv, num_fused=2)
        k, v = kv.unbind(dim=2)
        h = q
        q = q.feats
        q = q.unsqueeze(0)
        q = q.permute(0, 2, 1, 3)
        k = k.permute(0, 2, 1, 3)
        v = v.permute(0, 2, 1, 3)
        if not is_text:
            if kv_mask is None:
                slat_kv[f"{t_latent}_{order}_{pos}_{layer}_{self._type}_k"] = k.cpu()
                slat_kv[f"{t_latent}_{order}_{pos}_{layer}_{self._type}_v"] = v.cpu()
            else:
                # Cross-attn KV替换: 密集掩码乘法，与ST阶段公式(6)(7)相同
                k = k * kv_mask + slat_kv[f"{t_latent}_{order}_{pos}_{layer}_{self._type}_k"].cuda() * (1 - kv_mask)
                v = v * kv_mask + slat_kv[f"{t_latent}_{order}_{pos}_{layer}_{self._type}_v"].cuda() * (1 - kv_mask)
                k = k.type(q.dtype)
                v = v.type(q.dtype)
    out = F.scaled_dot_product_attention(q, k, v)
    out = out.permute(0, 2, 1, 3)[0]
    h = h.replace(out)
    h = self._reshape_chs(h, (-1,))
    h = self._linear(self.to_out, h)
    return h

def slat_trsfmr_forward(self, x, mod, context, slat_kv, self_kv_mask, cross_kv_mask, t_latent, order, pos, layer, is_text):
    """SLAT阶段 Transformer 块的注入 forward。
    
    与 ss_trsfmr_forward 结构相同，但操作在 SparseTensor 上，
    使用 x.replace(norm(x.feats)) 而非直接 norm(x)。
    """
    if self.share_mod:
        shift_msa, scale_msa, gate_msa, shift_mlp, scale_mlp, gate_mlp = mod.chunk(6, dim=1)
    else:
        shift_msa, scale_msa, gate_msa, shift_mlp, scale_mlp, gate_mlp = self.adaLN_modulation(mod).chunk(6, dim=1)
    h = x.replace(self.norm1(x.feats))
    h = h * (1 + scale_msa) + shift_msa
    h = self.self_attn(h, slat_kv=slat_kv, kv_mask=self_kv_mask, t_latent=t_latent, order=order, pos=pos, layer=layer, is_text=is_text)
    h = h * gate_msa
    x = x + h
    h = x.replace(self.norm2(x.feats))
    h = self.cross_attn(h, context, slat_kv=slat_kv, kv_mask=cross_kv_mask, t_latent=t_latent, order=order, pos=pos, layer=layer, is_text=is_text)
    x = x + h
    h = x.replace(self.norm3(x.feats))
    h = h * (1 + scale_mlp) + shift_mlp
    h = self.mlp(h)
    h = h * gate_mlp
    x = x + h
    return x

def slat_flow_forward(self, x, t, cond, slat_kv, self_kv_mask, cross_kv_mask, t_latent, order, pos, is_text):
    """SLAT阶段 SLatFlowModel 的注入 forward。
    
    数据流: input_layer → ResBlock下采样(input_blocks) → pos_emb →
            Transformer blocks(+KV操作) → ResBlock上采样(out_blocks+skip) → out_layer
    """
    h = self.input_layer(x).type(self.dtype)
    t_emb = self.t_embedder(t)
    if self.share_mod:
        t_emb = self.adaLN_modulation(t_emb)
    t_emb = t_emb.type(self.dtype)
    cond = cond.type(self.dtype)
    skips = []
    for block in self.input_blocks:
        h = block(h, t_emb)
        skips.append(h.feats)
    if self.pe_mode == "ape":
        h = h + self.pos_embedder(h.coords[:, 1:]).type(self.dtype)
    for layer, block in enumerate(self.blocks):
        h = block(h, t_emb, cond, slat_kv, self_kv_mask, cross_kv_mask, t_latent, order, pos, layer, is_text)
    for block, skip in zip(self.out_blocks, reversed(skips)):
        if self.use_skip_connection:
            h = block(h.replace(torch.cat([h.feats, skip], dim=1)), t_emb)
        else:
            h = block(h, t_emb)
    h = h.replace(F.layer_norm(h.feats, h.feats.shape[-1:]))
    h = self.out_layer(h.type(x.dtype))
    return h

##############################################################################
# 采样器 —— 论文 Sec.3.1 RF-Solver 和 Sec.3.2/3.3 反转/编辑采样循环
##############################################################################

class InversionFlowEulerGuidanceIntervalSampler(FlowEulerGuidanceIntervalSampler):
    """支持反转和编辑的Flow Euler采样器。
    
    继承自 TRELLIS 的 FlowEulerGuidanceIntervalSampler，
    重写采样逻辑以支持：
    - 二阶Taylor改进的Euler更新（RF-Solver, 论文公式(1)(2)）
    - 双向采样：反转(数据→噪声) 和 去噪(噪声→数据)
    - Late-time CFG: 仅在 t∈[0.5,1.0] 应用分类器无关引导（论文公式(3)）
    - 每步的潜在替换和KV替换
    """

    def _inference_model(self, model, sample, t, cond, kv, self_kv_mask, cross_kv_mask, t_latent, order, pos, is_text):
        """单次模型前向推理，将连续时间 t 转换为模型输入的 1000*t 刻度。"""
        t = torch.tensor([1000 * t] * sample.shape[0], device=sample.device, dtype=torch.float32)
        if cond is not None and cond.shape[0] == 1 and sample.shape[0] > 1:
            cond = cond.repeat(sample.shape[0], *([1] * (len(cond.shape) - 1)))
        return model(sample, t, cond, kv, self_kv_mask, cross_kv_mask, t_latent, order, pos, is_text)

    def inference_model(self, model, sample, t, cond, cfg_strength, kv, self_kv_mask, cross_kv_mask, t_latent, order, is_text):
        """带 Late-time CFG 的模型推理 —— 论文公式(3)。
        
        仅在 t_latent ∈ [0.5, 1.0] 时应用 CFG:
          f_cfg = (1+ω) * f_θ(cond) - ω * f_θ(neg_cond)
        其余时间仅使用正条件，以保留早期步骤的可逆性。
        pos=1 表示正条件，pos=0 表示负条件，用于KV缓存的key区分。
        """
        cfg_interval = [0.5, 1.0]
        if cfg_interval[0] <= t_latent <= cfg_interval[1]:
            pos = 1
            pred = self._inference_model(model, sample, t, cond["cond"], kv, self_kv_mask, cross_kv_mask, t_latent, order, pos, is_text)
            pos = 0
            neg_pred = self._inference_model(model, sample, t, cond["neg_cond"], kv, self_kv_mask, cross_kv_mask, t_latent, order, pos, is_text)
            return (1 + cfg_strength) * pred - cfg_strength * neg_pred
        else:
            pos = 1
            return self._inference_model(model, sample, t, cond["cond"], kv, self_kv_mask, cross_kv_mask, t_latent, order, pos, is_text)

    def sample_once(self, model, sample, t_curr, t_prev, cond, cfg_strength, kv, self_kv_mask, cross_kv_mask, t_latent, is_text):
        """单步采样 —— 二阶Taylor改进的Euler方案（RF-Solver）。
        
        对应论文公式(1)(2)，设 h = t_prev - t_curr:
          x_{t+h} = x_t + h·f_θ(x_t, t) - ½·h²·∂_t f_θ(x_t, t)
        其中 ∂_t f_θ 通过中间点有限差分近似:
          ∂_t f_θ ≈ [f_θ(x_{t+h/2}, t+h/2) - f_θ(x_t, t)] / (h/2)
        
        order=1 用于第一次模型调用(当前点), order=2 用于第二次(中间点),
        二者的KV分别缓存以避免冲突。
        """
        # 第一次调用: 计算当前点预测 f_θ(x_t, t)
        order = 1
        pred = self.inference_model(model, sample, t_curr, cond, cfg_strength, kv, self_kv_mask, cross_kv_mask, t_latent, order, is_text)
        # 计算中间点: x_{t-Δ/2} = x_t + (Δ/2) * f_θ(x_t, t)
        sample_mid = sample + (t_prev - t_curr) / 2 * pred
        t_mid = t_curr + (t_prev - t_curr) / 2
        # 第二次调用: 计算中间点预测 f_θ(x_{t-Δ/2}, t-Δ/2)
        order = 2
        pred_mid = self.inference_model(model, sample_mid, t_mid, cond, cfg_strength, kv, self_kv_mask, cross_kv_mask, t_latent, order, is_text)
        # 有限差分估计时间导数
        first_order = (pred_mid - pred) / ((t_prev - t_curr) / 2)
        # Taylor展开更新: x_{t-Δ} = x_t + Δ·f_θ + ½·Δ²·∂_t f_θ
        sample = sample + (t_prev - t_curr) * pred - 0.5 * (t_prev - t_curr) ** 2 * first_order
        return sample

    @torch.no_grad()
    def sample(self, model, stage, noise, cond, cfg_strength, latent=None, latent_mask=None, kv=None, self_kv_mask=None, cross_kv_mask=None, skip_step=None, noise_init=None, is_text=False):
        """反转/去噪主循环 —— 论文 Sec.3.2 和 Sec.3.3 的核心。
        
        通过 latent_mask 是否为 None 自动区分两种模式:
        - latent_mask=None → 反转模式: 时间从 0→1 (数据→噪声)，缓存每步的latent和KV
        - latent_mask≠None → 编辑模式: 时间从 1→0 (噪声→数据)，每步执行潜在替换
        
        Args:
            stage: 1=ST阶段(密集体素), 2=SLAT阶段(稀疏体素)
            noise: 反转时为编码后的潜在/SLAT, 编辑时为反转得到的终端噪声
            latent: 反转时缓存的每步潜在字典 (编辑模式传入)
            latent_mask: ST阶段为二值/soft掩码张量；
                         SLAT阶段为保留区域坐标，或 {"coords", "edit_weights"} 字典
            kv: 反转时缓存的每步KV字典 (编辑模式传入)
        """
        steps = 25
        rescale_t = 3.0
        sample = noise
        # 生成时间序列并应用非线性重缩放（TRELLIS原始设计）
        t_seq = np.linspace(1, 0, steps + 1)
        t_seq = rescale_t * t_seq / (1 + (rescale_t - 1) * t_seq)
        if latent_mask is None:
            inverse_bool = True
        else:
            inverse_bool = False
        if skip_step is not None:
            t_seq = t_seq[skip_step:]
            steps = steps - skip_step
        # 噪声重初始化选项: 在跳步后用源潜在插值初始化
        if noise_init is not None:
            noise_randn = torch.randn_like(noise)
            t_init = t_seq[0]
            sample = noise_init * (1 - t_init) + noise_randn * t_init
        if inverse_bool:
            # 反转: 反转时间序列方向 (0→1)，初始化缓存字典
            t_seq = t_seq[::-1]
            desc = "Inversing"
            latent = {}
            kv = {}
        else:
            desc = "Sampling"

        slat_latent_mask_coords = latent_mask
        slat_latent_edit_weights = None
        if stage == 2 and isinstance(latent_mask, dict):
            slat_latent_mask_coords = latent_mask["coords"]
            slat_latent_edit_weights = latent_mask.get("edit_weights")

        t_pairs = list((t_seq[i], t_seq[i + 1]) for i in range(steps))
        for t_curr, t_prev in tqdm(t_pairs, desc=desc, disable=False):
            if inverse_bool:
                t_latent = t_prev
            else:
                t_latent = t_curr
                # ===== 潜在替换 (Latent Replacement) =====
                if stage == 1:
                    # ST阶段 —— 论文公式(4): z_t ← M ⊙ z_t + (1-M) ⊙ ẑ_t
                    # 密集掩码乘法，保留区域用缓存的反转潜在覆盖
                    sample = sample * latent_mask + latent[f"{t_latent}"].cuda() * (1 - latent_mask)
                elif stage == 2:
                    # SLAT阶段 —— 论文公式(5): ∀u∈Ωkeep: z_t[u] ← ẑ_t[u]
                    # 可选 soft mask：对保留区域边界做 current/cached 的连续混合。
                    if slat_latent_mask_coords.shape[0] != 0 and latent:
                        match_1 = (sample.coords.unsqueeze(1) == slat_latent_mask_coords.unsqueeze(0)).all(dim=-1)
                        match_2 = (latent[f"{t_latent}"].coords.cuda().unsqueeze(1) == slat_latent_mask_coords.unsqueeze(0)).all(dim=-1)
                        idx_1 = match_1.float().argmax(0)
                        idx_2 = match_2.float().argmax(0)
                        feats = sample.feats.clone()
                        cached_feats = latent[f"{t_latent}"].feats.cuda()[idx_2]
                        if slat_latent_edit_weights is None:
                            feats[idx_1] = cached_feats
                        else:
                            current_feats = feats[idx_1].clone()
                            edit_weights = slat_latent_edit_weights.to(device=feats.device, dtype=feats.dtype)
                            feats[idx_1] = current_feats * edit_weights + cached_feats * (1 - edit_weights)
                        sample = sample.replace(feats)
            # 执行一步采样（Taylor改进的Euler更新）
            sample = self.sample_once(model, sample, t_curr, t_prev, cond, cfg_strength, kv, self_kv_mask, cross_kv_mask, t_latent, is_text)
            if inverse_bool:
                # 反转: 缓存当前步的潜在到CPU（节省GPU内存）
                latent[f"{t_latent}"] = sample.cpu()
        return sample, latent, kv

##############################################################################
# 反转与去噪的流程函数 —— 论文 Sec.3.2 (3D Inversion) 和 Sec.3.3 (3D Editing)
##############################################################################

def sample_sparse_structure_inverse(pipeline, cond_src, voxel_src, cfg_strength_stage_1_inverse, skip_step, is_text):
    """ST阶段反转 —— 论文 Sec.3.2。
    
    将源3D资产的体素占用编码为潜在空间，然后通过反转采样映射到终端噪声。
    同时缓存每个时间步的潜在向量(ss_latent)和注意力K/V张量(ss_kv)。
    """
    stage = 1
    flow_model = pipeline.models["sparse_structure_flow_model"]
    encoder = pipeline.models["sparse_structure_encoder"]
    z_s = encoder(voxel_src)
    sigma_min = pipeline.sparse_structure_sampler.sigma_min
    sparse_structure_sampler = InversionFlowEulerGuidanceIntervalSampler(sigma_min)
    if cfg_strength_stage_1_inverse is None:
        cfg_strength = pipeline.sparse_structure_sampler_params["cfg_strength"]
    else:
        cfg_strength = cfg_strength_stage_1_inverse
    noise, ss_latent, ss_kv = sparse_structure_sampler.sample(flow_model, stage, z_s, cond_src, cfg_strength, skip_step=skip_step, is_text=is_text)
    return noise, ss_latent, ss_kv

def sample_sparse_structure_denoise(pipeline, cond_tgt, noise, voxel_src, voxel_mask, ss_latent, ss_latent_mask, ss_kv, ss_self_kv_mask, ss_cross_kv_mask, cfg_strength_stage_1_forward, skip_step, re_init, is_text):
    """ST阶段去噪/编辑 —— 论文 Sec.3.3。
    
    从反转得到的终端噪声出发，以目标条件(编辑后图像/文本)进行去噪。
    每步执行潜在替换(公式4)和KV替换(公式6,7)以保留未编辑区域。
    去噪后用 voxel_mask 合并编辑区域和源体素，确保未编辑区域完全保留。
    
    re_init=True 时使用源体素编码进行噪声重初始化（消融实验 "w/ Noise Re-init"）。
    """
    stage = 1
    flow_model = pipeline.models["sparse_structure_flow_model"]
    sigma_min = pipeline.sparse_structure_sampler.sigma_min
    sparse_structure_sampler = InversionFlowEulerGuidanceIntervalSampler(sigma_min)
    if cfg_strength_stage_1_forward is None:
        cfg_strength = pipeline.sparse_structure_sampler_params["cfg_strength"]
    else:
        cfg_strength = cfg_strength_stage_1_forward
    if re_init:
        encoder = pipeline.models["sparse_structure_encoder"]
        noise_init = encoder(voxel_src)
    else:
        noise_init = None
    z_s, ss_latent, ss_kv = sparse_structure_sampler.sample(flow_model, stage, noise, cond_tgt, cfg_strength, ss_latent, ss_latent_mask, ss_kv, ss_self_kv_mask, ss_cross_kv_mask, skip_step, noise_init, is_text=is_text)
    decoder = pipeline.models["sparse_structure_decoder"]
    voxel = decoder(z_s)
    # 最终体素合并: 保留区域(mask=0)用源体素, 编辑区域(mask=1)用新生成体素
    voxel = voxel * voxel_mask + voxel_src * (1 - voxel_mask)
    return voxel

def sample_slat_inverse(pipeline, cond_src, slat_src, coords_mask, cfg_strength_stage_2_inverse, is_text):
    """SLAT阶段反转 —— 论文 Sec.3.2。
    
    首先从完整SLAT中提取保留区域 Ωkeep 的特征子集，
    对其归一化后执行反转采样，缓存每步的潜在和KV张量。
    
    论文: "we first extract the preserved set Ωkeep from the decoded output
    of the ST stage by removing the edit voxels, normalize the features,
    and run the same Taylor-improved inversion scheme"
    """
    if coords_mask.shape[0] == 0:
        print("[WARN] coords_preserve is empty; skipping SLAT inverse preservation and continuing with full-region trellis_edit.")
        return {}, {}

    stage = 2
    coords_mask = torch.cat([torch.zeros(coords_mask.shape[0], 1).int().cuda(), coords_mask], dim=1)
    # 提取保留区域 Ωkeep: 找到 slat_src 中属于保留区域的体素
    sparse_tensor_mask = torch.zeros(slat_src.coords.shape[0], dtype=torch.bool, device="cuda")
    for coord in coords_mask:
        sparse_tensor_mask[torch.all(slat_src.coords == coord, dim=1)] = True
    slat_inverse = slat_src.replace(slat_src.feats[sparse_tensor_mask], slat_src.coords[sparse_tensor_mask])
    flow_model = pipeline.models["slat_flow_model"]
    # 归一化: 使用预训练的均值和标准差
    std = torch.tensor(pipeline.slat_normalization["std"], device=pipeline.device)[None]
    mean = torch.tensor(pipeline.slat_normalization["mean"], device=pipeline.device)[None]
    slat_inverse = (slat_inverse - mean) / std
    sigma_min = pipeline.slat_sampler.sigma_min
    slat_sampler = InversionFlowEulerGuidanceIntervalSampler(sigma_min)
    if cfg_strength_stage_2_inverse is None:
        cfg_strength = pipeline.sparse_structure_sampler_params["cfg_strength"]
    else:
        cfg_strength = cfg_strength_stage_2_inverse
    noise, slat_latent, slat_kv = slat_sampler.sample(flow_model, stage, slat_inverse, cond_src, cfg_strength, is_text=is_text)
    return slat_latent, slat_kv

def sample_slat_denoise(
    pipeline,
    cond_tgt,
    coords_tgt,
    slat_src,
    coords_mask,
    slat_latent,
    slat_kv,
    slat_self_kv_mask,
    slat_cross_kv_mask,
    cfg_strength_stage_2_forward,
    is_text,
    slat_soft_mask=False,
    slat_soft_mask_dilation=2,
    slat_soft_mask_sigma=1.0,
):
    """SLAT阶段去噪/编辑 —— 论文 Sec.3.3。
    
    在ST阶段确定的新稀疏结构 coords_tgt 上，以目标条件进行去噪。
    去噪过程中每步执行稀疏潜在替换(公式5)和稀疏KV替换。
    可选 soft mask 会将保留区域边界改为平滑混合，缓解 SLAT 特征接缝。
    """
    stage = 2
    coords_mask = torch.cat([torch.zeros(coords_mask.shape[0], 1).int().cuda(), coords_mask], dim=1)
    slat_latent_mask = coords_mask
    slat_preserve_edit_weights = None
    if slat_soft_mask and coords_mask.shape[0] != 0:
        slat_preserve_edit_weights = make_slat_soft_weights(
            coords_tgt,
            coords_mask,
            dilation=slat_soft_mask_dilation,
            sigma=slat_soft_mask_sigma,
        )
        slat_latent_mask = {
            "coords": coords_mask,
            "edit_weights": slat_preserve_edit_weights,
        }
    flow_model = pipeline.models["slat_flow_model"]
    # 在新的稀疏结构上初始化随机噪声
    noise = sp.SparseTensor(feats=torch.randn(coords_tgt.shape[0], flow_model.in_channels).to(pipeline.device), coords=coords_tgt)
    sigma_min = pipeline.slat_sampler.sigma_min
    slat_sampler = InversionFlowEulerGuidanceIntervalSampler(sigma_min)
    if cfg_strength_stage_2_forward is None:
        cfg_strength = pipeline.sparse_structure_sampler_params["cfg_strength"]
    else:
        cfg_strength = cfg_strength_stage_2_forward
    # 当保留区域为空、SLAT 反转被跳过时，不存在可用于替换的缓存。
    # 此时应退化为全区域编辑，禁用 self/cross KV replacement，
    # 否则 cross-attention 会在空字典上查找缓存 key 并触发 KeyError。
    if not slat_kv:
        slat_self_kv_mask = None
        slat_cross_kv_mask = None
    slat, slat_latent, slat_kv = slat_sampler.sample(flow_model, stage, noise, cond_tgt, cfg_strength, slat_latent, slat_latent_mask, slat_kv, slat_self_kv_mask, slat_cross_kv_mask, is_text=is_text)
    # 反归一化
    std = torch.tensor(pipeline.slat_normalization["std"], device=pipeline.device)[None]
    mean = torch.tensor(pipeline.slat_normalization["mean"], device=pipeline.device)[None]
    slat = slat * std + mean

    # 最终特征覆盖: 保留区域用原始SLAT特征替换去噪结果
    # 确保未编辑区域的几何和纹理完全保真
    if coords_mask.shape[0] == 0:
        return slat

    match_1 = (coords_tgt.unsqueeze(1) == coords_mask.unsqueeze(0)).all(dim=-1)
    match_2 = (slat_src.coords.unsqueeze(1) == coords_mask.unsqueeze(0)).all(dim=-1)
    idx_1 = match_1.float().argmax(0)
    idx_2 = match_2.float().argmax(0)
    feats = slat.feats.clone()
    src_feats = slat_src.feats[idx_2]
    if slat_preserve_edit_weights is None:
        feats[idx_1] = src_feats
    else:
        current_feats = feats[idx_1].clone()
        edit_weights = slat_preserve_edit_weights.to(device=feats.device, dtype=feats.dtype)
        feats[idx_1] = current_feats * edit_weights + src_feats * (1 - edit_weights)
    slat = slat.replace(feats)
    return slat

##############################################################################
# 主编辑函数 —— 论文 Figure 3 所示的完整 VoxHammer 流程
##############################################################################

def run_edit(
    pipeline,
    render_dir,
    output_path,
    image_dir,
    is_text,
    source_prompt,
    target_prompt,
    skip_step=0,
    re_init=False,
    cfg=[5.0, 6.0, 0.0, 0.0],
    coord_match="1d",
    soft_mask=False,
    soft_mask_dilation=2,
    soft_mask_sigma=1.0,
    slat_soft_mask=False,
    slat_soft_mask_dilation=2,
    slat_soft_mask_sigma=1.0,
):
    """VoxHammer 3D编辑主流程。
    
    Args:
        pipeline: TRELLIS pipeline (图像条件或文本条件)
        render_dir: 包含 voxels.ply, features.npz, voxels_delete.ply 的渲染目录
        output_path: 输出GLB文件路径
        image_dir: 包含 2d_render.png, 2d_edit.png, 2d_mask.png 的图像目录
        is_text: True=文本条件, False=图像条件
        skip_step: 跳过的初始步数（用于加速）
        re_init: 是否重初始化噪声（消融实验选项）
        cfg: [ST反转CFG, ST去噪CFG, SLAT反转CFG, SLAT去噪CFG]
        coord_match: 坐标匹配方式，"1d" 使用1D整数索引行级匹配（正确），
                     "elementwise" 使用逐元素 torch.isin + all(dim=1)（有bug，仅供对比）
        soft_mask: 是否对 ST 阶段的 M^ss 启用 soft mask
        soft_mask_dilation: soft mask 的 3D 膨胀半径（voxel）
        soft_mask_sigma: soft mask 的 3D Gaussian sigma（voxel）
        slat_soft_mask: 是否对 SLAT 阶段的潜在替换/最终特征覆盖启用 soft mask
        slat_soft_mask_dilation: SLAT soft mask 的 3D 膨胀半径（voxel）
        slat_soft_mask_sigma: SLAT soft mask 的 3D Gaussian sigma（voxel）
    """

    # ===== Step 0: 动态注入 forward 函数 =====
    # 论文: "All modifications are implemented by dynamically adjusting forward
    # functions at inference time, without retraining or weight updates."
    # 通过 MethodType 将自定义的 forward（含KV缓存/替换逻辑）绑定到模型实例
    ss_flow = pipeline.models["sparse_structure_flow_model"]
    ss_flow.forward = MethodType(ss_flow_forward, ss_flow)
    for block in ss_flow.blocks:
        trsfmr_obj = block
        trsfmr_obj.forward = MethodType(ss_trsfmr_forward, trsfmr_obj)
        self_attn_obj = block.self_attn
        self_attn_obj.forward = MethodType(ss_attn_forward, self_attn_obj)
        cross_attn_obj = block.cross_attn
        cross_attn_obj.forward = MethodType(ss_attn_forward, cross_attn_obj)

    slat_flow = pipeline.models["slat_flow_model"]
    slat_flow.forward = MethodType(slat_flow_forward, slat_flow)
    for block in slat_flow.blocks:
        trsfmr_obj = block
        trsfmr_obj.forward = MethodType(slat_trsfmr_forward, trsfmr_obj)
        self_attn_obj = block.self_attn
        self_attn_obj.forward = MethodType(slat_attn_forward, self_attn_obj)
        cross_attn_obj = block.cross_attn
        cross_attn_obj.forward = MethodType(slat_attn_forward, cross_attn_obj)

    cfg_strength_stage_1_inverse = cfg[0]
    cfg_strength_stage_1_forward = cfg[1]
    cfg_strength_stage_2_inverse = cfg[2]
    cfg_strength_stage_2_forward = cfg[3]

    # ===== Step 1: 加载输入数据 =====
    coords_src = ply_to_coords(os.path.join(render_dir, "voxels.ply"))
    voxel_src = coords_to_voxel(coords_src)
    slat_src = feats_to_slat(pipeline, os.path.join(render_dir, "features.npz"))
    ply_delete_path = os.path.join(render_dir, "voxels_delete.ply")
    if not is_text:
        img_src_path = os.path.join(image_dir, "2d_render.png")
        img_tgt_path = os.path.join(image_dir, "2d_edit.png")
        img_mask_path = os.path.join(image_dir, "2d_mask.png")
        pre_src, pre_tgt, pre_mask = preprocess_image(pipeline, img_src_path, img_tgt_path, img_mask_path)
        cond_src = pipeline.get_cond([pre_src])
        cond_tgt = pipeline.get_cond([pre_tgt])
    else:
        pre_mask = None
        cond_src = pipeline.get_cond([source_prompt])
        cond_tgt = pipeline.get_cond([target_prompt])
    # 计算保留区域坐标: 全部体素 - 待删除体素
    # 注意: torch.isin 是逐元素检查，需将3D坐标编码为1D整数索引做行级别匹配
    
    coords_delete = ply_to_coords(ply_delete_path)

    if coord_match == "1d":
        coords_delete_1d = coords_to_flat_indices(coords_delete)
        coords_src_1d = coords_to_flat_indices(coords_src)
        coords_preserve = coords_src[~torch.isin(coords_src_1d, coords_delete_1d)]
    elif coord_match == "elementwise":
        coords_preserve = coords_src[~torch.isin(coords_src, coords_delete).all(dim=1)]
    else:
        raise ValueError(f"coord_match 必须为 '1d' 或 'elementwise'，收到: {coord_match!r}")

    # DEBUG: 保存坐标供可视化（设置环境变量 VOXHAMMER_DEBUG_COORDS=1 启用）
    if os.environ.get("VOXHAMMER_DEBUG_COORDS"):
        debug_dir = os.path.join(render_dir, "debug_coords")
        os.makedirs(debug_dir, exist_ok=True)

        dbg_delete_1d = coords_to_flat_indices(coords_delete)
        dbg_src_1d = coords_to_flat_indices(coords_src)
        dbg_preserve_1d          = coords_src[~torch.isin(dbg_src_1d, dbg_delete_1d)]
        dbg_preserve_elementwise = coords_src[~torch.isin(coords_src, coords_delete).all(dim=1)]

        np.save(os.path.join(debug_dir, "coords_src.npy"),                    coords_src.cpu().numpy())
        np.save(os.path.join(debug_dir, "coords_delete.npy"),                 coords_delete.cpu().numpy())
        np.save(os.path.join(debug_dir, "coords_preserve_1d.npy"),            dbg_preserve_1d.cpu().numpy())
        np.save(os.path.join(debug_dir, "coords_preserve_elementwise.npy"),   dbg_preserve_elementwise.cpu().numpy())
        print(f"[DEBUG] coords saved → {debug_dir}")
        print(f"  coords_src:                    {len(coords_src):5d} voxels")
        print(f"  coords_delete:                 {len(coords_delete):5d} voxels")
        print(f"  coords_preserve (1d):          {len(dbg_preserve_1d):5d} voxels")
        print(f"  coords_preserve (elementwise): {len(dbg_preserve_elementwise):5d} voxels")

    # ===== Step 2: ST阶段 —— 反转 + 编辑 =====
    voxel_mask, ss_latent_mask, ss_self_kv_mask, cross_kv_mask = ply_to_ss_mask(
        coords_preserve,
        pre_mask,
        soft_mask=soft_mask,
        soft_mask_dilation=soft_mask_dilation,
        soft_mask_sigma=soft_mask_sigma,
    )
    # ST反转: 源体素 → 终端噪声，缓存潜在和KV
    noise, ss_latent, ss_kv = sample_sparse_structure_inverse(pipeline, cond_src, voxel_src, cfg_strength_stage_1_inverse, skip_step, is_text)
    # ST去噪: 终端噪声 → 编辑后体素，通过掩码替换保留未编辑区域
    voxel_tgt = sample_sparse_structure_denoise(pipeline, cond_tgt, noise, voxel_src, voxel_mask, ss_latent, ss_latent_mask, ss_kv, ss_self_kv_mask, cross_kv_mask, cfg_strength_stage_1_forward, skip_step, re_init, is_text)
    # 从编辑后的体素中提取新的稀疏结构坐标
    coords_tgt = torch.argwhere(voxel_tgt > 0)[:, [0, 2, 3, 4]].int()
    print(f"[RSS] ST阶段清理前: {get_process_rss_gib():.2f} GiB")
    del noise
    del ss_latent
    del ss_kv
    del ss_latent_mask
    del ss_self_kv_mask
    del voxel_mask
    del voxel_src
    del voxel_tgt
    gc.collect()
    print(f"[RSS] ST阶段清理后: {get_process_rss_gib():.2f} GiB")

    # ===== Step 3: SLAT阶段 —— 反转 + 编辑 =====
    slat_self_kv_mask = ply_to_slat_mask(coords_tgt, coords_preserve)
    # SLAT反转: 保留区域SLAT → 终端噪声，缓存潜在和KV
    slat_latent, slat_kv = sample_slat_inverse(pipeline, cond_src, slat_src, coords_preserve, cfg_strength_stage_2_inverse, is_text)
    # SLAT去噪: 在新结构上去噪精细几何和纹理
    slat_tgt = sample_slat_denoise(
        pipeline,
        cond_tgt,
        coords_tgt,
        slat_src,
        coords_preserve,
        slat_latent,
        slat_kv,
        slat_self_kv_mask,
        cross_kv_mask,
        cfg_strength_stage_2_forward,
        is_text,
        slat_soft_mask=slat_soft_mask,
        slat_soft_mask_dilation=slat_soft_mask_dilation,
        slat_soft_mask_sigma=slat_soft_mask_sigma,
    )
    print(f"[RSS] SLAT阶段清理前: {get_process_rss_gib():.2f} GiB")
    del slat_latent
    del slat_kv
    del slat_self_kv_mask
    del cross_kv_mask
    del coords_tgt
    del coords_preserve
    del slat_src
    del cond_src
    del cond_tgt
    gc.collect()
    print(f"[RSS] SLAT阶段清理后: {get_process_rss_gib():.2f} GiB")

    # ===== Step 4: 解码并导出 =====
    assets_tgt = pipeline.decode_slat(slat_tgt, ["gaussian", "mesh"])
    torch.set_grad_enabled(True)
    glb_tgt = postprocessing_utils.to_glb(assets_tgt["gaussian"][0], assets_tgt["mesh"][0], simplify=0.95, texture_size=1024)
    torch.set_grad_enabled(False)
    glb_tgt.export(output_path)

if __name__ == "__main__":
    is_text = False
    if is_text:
        pipeline = TrellisTextTo3DPipeline.from_pretrained("microsoft/TRELLIS-text-large")
    else:
        pipeline = TrellisImageTo3DPipeline.from_pretrained("microsoft/TRELLIS-image-large")
    pipeline.cuda()
    run_edit(pipeline, render_dir="/path/to/render", output_path="/path/to/output.glb", image_dir="/path/to/image", is_text=is_text, source_prompt="", target_prompt="")
