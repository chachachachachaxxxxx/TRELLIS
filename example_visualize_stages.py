"""
TRELLIS 全流程中间结果可视化脚本

在 outputs/visualize_stages/<case_name>/ 下保存每个阶段的输入和输出。
命名规则: <阶段号>_<input/output>_<描述>.<格式>
"""

import os
os.environ['SPCONV_ALGO'] = 'native'

import sys
import imageio
import torch
import torch.nn.functional as F
import numpy as np
from PIL import Image
from sklearn.decomposition import PCA
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D

from trellis.pipelines import TrellisImageTo3DPipeline
from trellis.utils import render_utils, postprocessing_utils
from output_layout import build_output_layout


def save_tensor_stats(tensor, path, title="Tensor Statistics"):
    """保存 tensor 的统计信息为文本文件"""
    with open(path, 'w') as f:
        f.write(f"{title}\n{'='*50}\n")
        f.write(f"Shape: {list(tensor.shape)}\n")
        f.write(f"Dtype: {tensor.dtype}\n")
        f.write(f"Device: {tensor.device}\n")
        f.write(f"Min:  {tensor.min().item():.6f}\n")
        f.write(f"Max:  {tensor.max().item():.6f}\n")
        f.write(f"Mean: {tensor.mean().item():.6f}\n")
        f.write(f"Std:  {tensor.std().item():.6f}\n")


def safe_hist(ax, data, **kwargs):
    """自动处理各种退化数据情况的直方图绘制"""
    data = np.asarray(data, dtype=np.float64).ravel()
    data = data[np.isfinite(data)]
    if len(data) == 0:
        ax.text(0.5, 0.5, 'No finite data', transform=ax.transAxes, ha='center')
        return
    kwargs.setdefault('color', 'steelblue')
    kwargs.setdefault('edgecolor', 'black')
    for n_bins in [50, 30, 10]:
        try:
            ax.hist(data, bins=n_bins, **kwargs)
            return
        except ValueError:
            continue
    ax.hist(data, bins=np.linspace(data.min() - 0.5, data.max() + 0.5, 11), **kwargs)


def visualize_feature_pca(features, h, w, path, title="Feature PCA"):
    """将高维 patch features 用 PCA 降到3维并可视化为 RGB 图"""
    feats = features.cpu().numpy()  # [N_tokens, C]
    pca = PCA(n_components=3)
    pca_result = pca.fit_transform(feats)
    pca_min = pca_result.min(axis=0)
    pca_max = pca_result.max(axis=0)
    pca_norm = (pca_result - pca_min) / (pca_max - pca_min + 1e-8)
    pca_img = pca_norm.reshape(h, w, 3)

    fig, ax = plt.subplots(1, 1, figsize=(6, 6))
    ax.imshow(pca_img)
    ax.set_title(title)
    ax.axis('off')
    fig.tight_layout()
    fig.savefig(path, dpi=150, bbox_inches='tight')
    plt.close(fig)


def visualize_3d_coords(coords, path, title="Sparse Structure", color_values=None):
    """用3D散点图可视化稀疏体素坐标"""
    coords_np = coords.cpu().numpy()
    if coords_np.shape[1] == 4:
        x, y, z = coords_np[:, 1], coords_np[:, 2], coords_np[:, 3]
    else:
        x, y, z = coords_np[:, 0], coords_np[:, 1], coords_np[:, 2]

    fig = plt.figure(figsize=(10, 10))

    views = [(30, 45), (30, 135), (75, 45), (0, 0)]
    view_names = ['front-right', 'front-left', 'top', 'front']

    for idx, ((elev, azim), vname) in enumerate(zip(views, view_names)):
        ax = fig.add_subplot(2, 2, idx + 1, projection='3d')
        if color_values is not None:
            scatter = ax.scatter(x, y, z, c=color_values, s=1, alpha=0.6, cmap='viridis')
            plt.colorbar(scatter, ax=ax, shrink=0.5)
        else:
            ax.scatter(x, y, z, s=1, alpha=0.4, c='steelblue')
        ax.set_xlabel('X')
        ax.set_ylabel('Y')
        ax.set_zlabel('Z')
        ax.set_title(f'{title} ({vname})')
        ax.view_init(elev=elev, azim=azim)

    fig.tight_layout()
    fig.savefig(path, dpi=150, bbox_inches='tight')
    plt.close(fig)


def visualize_occupancy_grid(decoder, z_s, path):
    """可视化稀疏结构解码器输出的占用概率体积（取3个正交切片）"""
    with torch.no_grad():
        occ = decoder(z_s)  # [1, 1, 64, 64, 64]
    occ_np = occ[0, 0].cpu().numpy()
    mid = occ_np.shape[0] // 2

    fig, axes = plt.subplots(1, 3, figsize=(15, 5))
    titles = [f'XY slice (z={mid})', f'XZ slice (y={mid})', f'YZ slice (x={mid})']
    slices = [occ_np[:, :, mid], occ_np[:, mid, :], occ_np[mid, :, :]]
    for ax, sl, t in zip(axes, slices, titles):
        im = ax.imshow(sl.T, origin='lower', cmap='RdBu_r', vmin=-3, vmax=3)
        ax.set_title(t)
        ax.axis('off')
        plt.colorbar(im, ax=ax, shrink=0.8)
    fig.suptitle('Occupancy Logits (>0 = occupied)', fontsize=14)
    fig.tight_layout()
    fig.savefig(path, dpi=150, bbox_inches='tight')
    plt.close(fig)


def visualize_latent_volume_slices(z_s, path):
    """可视化 3D 潜在体积的通道切片"""
    vol = z_s[0].cpu().numpy()  # [C, D, H, W]
    n_channels = vol.shape[0]
    mid = vol.shape[1] // 2

    cols = 4
    rows = (n_channels + cols - 1) // cols
    fig, axes = plt.subplots(rows, cols, figsize=(4 * cols, 4 * rows))
    axes = axes.flatten()
    for i in range(n_channels):
        im = axes[i].imshow(vol[i, mid, :, :], cmap='viridis')
        axes[i].set_title(f'Channel {i} (z={mid})')
        axes[i].axis('off')
        plt.colorbar(im, ax=axes[i], shrink=0.8)
    for i in range(n_channels, len(axes)):
        axes[i].axis('off')
    fig.suptitle('Latent Volume Slices (mid-z)', fontsize=14)
    fig.tight_layout()
    fig.savefig(path, dpi=150, bbox_inches='tight')
    plt.close(fig)


def render_snapshot_image(sample, path):
    """渲染4视角快照拼图"""
    res = render_utils.render_snapshot(sample, resolution=512)
    if 'color' in res:
        frames = res['color']
    else:
        frames = res['normal']
    n = len(frames)
    cols = min(n, 4)
    rows = (n + cols - 1) // cols
    fig, axes = plt.subplots(rows, cols, figsize=(5 * cols, 5 * rows))
    if rows == 1 and cols == 1:
        axes = [axes]
    else:
        axes = np.array(axes).flatten()
    for i, frame in enumerate(frames):
        axes[i].imshow(frame)
        axes[i].axis('off')
    for i in range(n, len(axes)):
        axes[i].axis('off')
    fig.tight_layout()
    fig.savefig(path, dpi=150, bbox_inches='tight')
    plt.close(fig)


def run_visualization(image_path: str, case_name: str, seed: int = 1):
    """运行完整 pipeline 并保存每个阶段的输入输出可视化"""

    out_dir = str(build_output_layout("visualize_stages", case_name).case_dir)
    os.makedirs(out_dir, exist_ok=True)
    print(f"[INFO] 输出目录: {out_dir}")

    # 模型前向与可视化渲染都关闭梯度，避免保留 autograd 计算图。
    # 这里使用 no_grad 而不是 inference_mode：
    # inference tensor 会在 to_glb() 内部的自定义 autograd 渲染算子里报错。
    with torch.no_grad():
        # ========== 加载 Pipeline ==========
        print("[阶段0] 加载 pipeline...")
        pipeline = TrellisImageTo3DPipeline.from_pretrained("microsoft/TRELLIS-image-large")
        pipeline.cuda()

        # ========== 阶段 1: 图像预处理 ==========
        print("[阶段1] 图像预处理...")
        image = Image.open(image_path)
        # 原始输入是 PIL.Image，空间尺寸为原图的 (W, H)，这里还不是固定 shape 的张量。
        image.save(os.path.join(out_dir, "1_input_original_image.png"))

        preprocessed_image = pipeline.preprocess_image(image)
        # preprocess_image() 会把前景裁剪后统一 resize 到 518x518。
        # 后续送入 DINOv2 编码器时，对应的张量维度是 [B, 3, 518, 518]。
        preprocessed_image.save(os.path.join(out_dir, "1_output_preprocessed_518x518.png"))
        print(f"  原始图像尺寸: {image.size}, 预处理后: {preprocessed_image.size}")

        # ========== 阶段 2: 图像特征编码 (DINOv2) ==========
        print("[阶段2] DINOv2 特征编码...")
        preprocessed_image.save(os.path.join(out_dir, "2_input_preprocessed_image.png"))

        cond = pipeline.get_cond([preprocessed_image])
        # cond_feat 是 DINOv2 输出的条件 token：
        # [B, N_tokens, C] = [1, 1370, 1024]
        # 其中 1370 = 1 个 CLS token + 37x37=1369 个 patch token，1024 是每个 token 的特征维度。
        cond_feat = cond['cond']  # [1, 1370, 1024]
        neg_feat = cond['neg_cond']

        save_tensor_stats(cond_feat, os.path.join(out_dir, "2_output_cond_features_stats.txt"),
                          title="DINOv2 Condition Features")

        n_patches = 37 * 37  # 518/14 = 37, DINOv2 patch grid
        patch_tokens = cond_feat[0, -n_patches:, :]  # 取最后1369个patch token（跳过CLS和register tokens）
        visualize_feature_pca(patch_tokens, 37, 37,
                              os.path.join(out_dir, "2_output_cond_features_pca.png"),
                              title="DINOv2 Patch Features (PCA→RGB, 37×37)")

        all_tokens = cond_feat[0]
        feat_norm = all_tokens.norm(dim=-1).cpu().numpy()
        fig, axes = plt.subplots(1, 2, figsize=(12, 5))
        safe_hist(axes[0], feat_norm)
        axes[0].set_title('Token L2 Norm Distribution')
        axes[0].set_xlabel('L2 Norm')
        axes[0].set_ylabel('Count')
        patch_norms = feat_norm[-n_patches:].reshape(37, 37)
        im = axes[1].imshow(patch_norms, cmap='hot')
        axes[1].set_title('Patch Token Norm Heatmap (37×37)')
        axes[1].axis('off')
        plt.colorbar(im, ax=axes[1], shrink=0.8)
        fig.suptitle('DINOv2 Feature Analysis', fontsize=14)
        fig.tight_layout()
        fig.savefig(os.path.join(out_dir, "2_output_cond_features_analysis.png"), dpi=150, bbox_inches='tight')
        plt.close(fig)

        print(f"  cond shape: {cond_feat.shape}, neg_cond shape: {neg_feat.shape}")

        # ========== 阶段 3: 稀疏结构采样 ==========
        print("[阶段3] 稀疏结构采样...")
        save_tensor_stats(cond_feat, os.path.join(out_dir, "3_input_cond_features_stats.txt"),
                          title="Stage 3 Input: Condition Features")

        torch.manual_seed(seed)

        flow_model = pipeline.models['sparse_structure_flow_model']
        reso = flow_model.resolution
        # 这里是阶段3真正输入给 sparse_structure_flow_model 的 3D latent 噪声：
        # [B, C, D, H, W] = [1, 8, 16, 16, 16]
        # B=1: 单张图采样一个样本
        # C=8: 稀疏结构 latent 通道数
        # D/H/W=16: 16^3 的潜在体素网格分辨率
        noise_ss = torch.randn(1, flow_model.in_channels, reso, reso, reso).to(pipeline.device)
        print(f"  sparse structure flow input shape: {tuple(noise_ss.shape)} = [B, C, D, H, W]")

        visualize_latent_volume_slices(noise_ss,
                                       os.path.join(out_dir, "3_input_noise_volume_slices.png"))

        sampler_params_ss = {**pipeline.sparse_structure_sampler_params, "steps": 12, "cfg_strength": 7.5}
        z_s = pipeline.sparse_structure_sampler.sample(
            flow_model, noise_ss,
            **cond, **sampler_params_ss, verbose=True
        ).samples

        visualize_latent_volume_slices(z_s,
                                       os.path.join(out_dir, "3_output_latent_volume_slices.png"))

        decoder_ss = pipeline.models['sparse_structure_decoder']
        visualize_occupancy_grid(decoder_ss, z_s,
                                 os.path.join(out_dir, "3_output_occupancy_grid_slices.png"))

        coords = torch.argwhere(decoder_ss(z_s) > 0)[:, [0, 2, 3, 4]].int()

        visualize_3d_coords(coords,
                            os.path.join(out_dir, "3_output_sparse_coords_3d.png"),
                            title=f"Sparse Structure ({coords.shape[0]} voxels)")

        save_tensor_stats(coords.float(), os.path.join(out_dir, "3_output_sparse_coords_stats.txt"),
                          title=f"Sparse Coordinates ({coords.shape[0]} occupied voxels)")
        print(f"  占用体素数量: {coords.shape[0]}")

        # ========== 阶段 4: 结构化潜在采样 (SLat) ==========
        print("[阶段4] SLat 采样...")
        visualize_3d_coords(coords,
                            os.path.join(out_dir, "4_input_sparse_coords_3d.png"),
                            title=f"SLat Input Coords ({coords.shape[0]} voxels)")

        slat = pipeline.sample_slat(cond, coords,
                                    sampler_params={"steps": 12, "cfg_strength": 3})

        slat_feats = slat.feats.cpu().numpy()  # [N, 8]
        slat_coords_np = slat.coords.cpu().numpy()

        save_tensor_stats(slat.feats, os.path.join(out_dir, "4_output_slat_features_stats.txt"),
                          title=f"SLat Features ({slat.feats.shape[0]} voxels × {slat.feats.shape[1]} channels)")

        pca = PCA(n_components=3)
        slat_pca = pca.fit_transform(slat_feats)
        slat_pca_norm = (slat_pca - slat_pca.min(axis=0)) / (slat_pca.max(axis=0) - slat_pca.min(axis=0) + 1e-8)

        if slat_coords_np.shape[1] == 4:
            sx, sy, sz = slat_coords_np[:, 1], slat_coords_np[:, 2], slat_coords_np[:, 3]
        else:
            sx, sy, sz = slat_coords_np[:, 0], slat_coords_np[:, 1], slat_coords_np[:, 2]

        fig = plt.figure(figsize=(16, 8))
        views = [(30, 45), (30, 135), (75, 45), (0, 0)]
        view_names = ['front-right', 'front-left', 'top', 'front']
        for idx, ((elev, azim), vname) in enumerate(zip(views, view_names)):
            ax = fig.add_subplot(2, 4, idx + 1, projection='3d')
            ax.scatter(sx, sy, sz, c=slat_pca_norm, s=1, alpha=0.6)
            ax.set_title(f'PCA Color ({vname})')
            ax.view_init(elev=elev, azim=azim)

        for ch in range(min(4, slat_feats.shape[1])):
            ax = fig.add_subplot(2, 4, 5 + ch, projection='3d')
            scatter = ax.scatter(sx, sy, sz, c=slat_feats[:, ch], s=1, alpha=0.6, cmap='viridis')
            ax.set_title(f'Channel {ch}')
            ax.view_init(elev=30, azim=45)
            plt.colorbar(scatter, ax=ax, shrink=0.5)

        fig.suptitle('SLat Features Visualization', fontsize=14)
        fig.tight_layout()
        fig.savefig(os.path.join(out_dir, "4_output_slat_features_pca.png"), dpi=150, bbox_inches='tight')
        plt.close(fig)

        fig, axes = plt.subplots(2, 4, figsize=(16, 8))
        for ch in range(slat_feats.shape[1]):
            row, col = ch // 4, ch % 4
            safe_hist(axes[row, col], slat_feats[:, ch])
            axes[row, col].set_title(f'Channel {ch}')
            axes[row, col].set_xlabel('Value')
        fig.suptitle('SLat Feature Channel Distributions', fontsize=14)
        fig.tight_layout()
        fig.savefig(os.path.join(out_dir, "4_output_slat_features_distribution.png"), dpi=150, bbox_inches='tight')
        plt.close(fig)
        print(f"  SLat feats shape: {slat.feats.shape}")

        # ========== 阶段 5: 多格式解码 ==========
        print("[阶段5] 多格式解码...")
        save_tensor_stats(slat.feats, os.path.join(out_dir, "5_input_slat_features_stats.txt"),
                          title="Stage 5 Input: SLat Features")

        outputs = pipeline.decode_slat(slat, formats=['mesh', 'gaussian', 'radiance_field'])

        gaussian = outputs['gaussian'][0]
        radiance_field = outputs['radiance_field'][0]
        mesh = outputs['mesh'][0]

        render_snapshot_image(gaussian,
                              os.path.join(out_dir, "5_output_gaussian_snapshot.png"))
        render_snapshot_image(radiance_field,
                              os.path.join(out_dir, "5_output_radiance_field_snapshot.png"))
        render_snapshot_image(mesh,
                              os.path.join(out_dir, "5_output_mesh_snapshot.png"))

        with open(os.path.join(out_dir, "5_output_decode_summary.txt"), 'w') as f:
            f.write(f"Gaussian: {gaussian._xyz.shape[0]} points\n")
            f.write(f"Mesh: {mesh.vertices.shape[0]} vertices, {mesh.faces.shape[0]} faces\n")

        print(f"  Gaussian 数量: {gaussian._xyz.shape[0]}")
        print(f"  Mesh 顶点/面: {mesh.vertices.shape[0]} / {mesh.faces.shape[0]}")

        # ========== 阶段 6: 后处理与导出 ==========
        print("[阶段6] 后处理与导出...")

        print("  渲染 Gaussian 视频...")
        video_gs = render_utils.render_video(gaussian, num_frames=120)['color']
        gs_video_path = os.path.join(out_dir, "6_output_video_gaussian.mp4")
        imageio.mimsave(gs_video_path, video_gs, fps=30)
        Image.fromarray(video_gs[0]).save(os.path.join(out_dir, "6_input_gaussian_frame0.png"))

        print("  渲染 RadianceField 视频...")
        video_rf = render_utils.render_video(radiance_field, num_frames=120)['color']
        rf_video_path = os.path.join(out_dir, "6_output_video_radiance_field.mp4")
        imageio.mimsave(rf_video_path, video_rf, fps=30)
        Image.fromarray(video_rf[0]).save(os.path.join(out_dir, "6_input_radiance_field_frame0.png"))

        print("  渲染 Mesh 视频...")
        video_mesh = render_utils.render_video(mesh, num_frames=120)['normal']
        mesh_video_path = os.path.join(out_dir, "6_output_video_mesh.mp4")
        imageio.mimsave(mesh_video_path, video_mesh, fps=30)
        Image.fromarray(video_mesh[0]).save(os.path.join(out_dir, "6_input_mesh_frame0.png"))

    print("  导出 GLB...")
    mesh.vertices = mesh.vertices.detach()
    if mesh.vertex_attrs is not None:
        mesh.vertex_attrs = mesh.vertex_attrs.detach()
    glb = postprocessing_utils.to_glb(gaussian, mesh, simplify=0.95, texture_size=1024)
    glb.export(os.path.join(out_dir, "6_output_textured_mesh.glb"))

    print("  导出 PLY...")
    gaussian.save_ply(os.path.join(out_dir, "6_output_gaussian.ply"))

    # ========== 生成目录索引 ==========
    index_path = os.path.join(out_dir, "README.txt")
    with open(index_path, 'w') as f:
        f.write(f"TRELLIS Pipeline Visualization: {case_name}\n")
        f.write(f"Input image: {image_path}\n")
        f.write(f"Seed: {seed}\n")
        f.write(f"{'='*60}\n\n")
        f.write("阶段1: 图像预处理\n")
        f.write("  1_input_original_image.png          原始输入图片\n")
        f.write("  1_output_preprocessed_518x518.png   去背景+裁剪+缩放到518×518\n\n")
        f.write("阶段2: DINOv2 图像特征编码\n")
        f.write("  2_input_preprocessed_image.png      预处理后的图片\n")
        f.write("  2_output_cond_features_stats.txt    条件特征统计信息\n")
        f.write("  2_output_cond_features_pca.png      PCA降维可视化(37×37 RGB)\n")
        f.write("  2_output_cond_features_analysis.png 特征分析(norm分布+heatmap)\n\n")
        f.write("阶段3: 稀疏结构采样\n")
        f.write("  3_input_cond_features_stats.txt     输入条件特征统计\n")
        f.write("  3_input_noise_volume_slices.png     初始噪声体积切片\n")
        f.write("  3_output_latent_volume_slices.png   去噪后潜在体积切片\n")
        f.write("  3_output_occupancy_grid_slices.png  占用概率网格切片\n")
        f.write("  3_output_sparse_coords_3d.png       稀疏体素坐标3D散点图\n")
        f.write("  3_output_sparse_coords_stats.txt    稀疏坐标统计信息\n\n")
        f.write("阶段4: 结构化潜在采样 (SLat)\n")
        f.write("  4_input_sparse_coords_3d.png        输入稀疏坐标\n")
        f.write("  4_output_slat_features_stats.txt    SLat特征统计\n")
        f.write("  4_output_slat_features_pca.png      SLat PCA+通道可视化\n")
        f.write("  4_output_slat_features_distribution.png  SLat通道分布\n\n")
        f.write("阶段5: 多格式解码\n")
        f.write("  5_input_slat_features_stats.txt     输入SLat特征统计\n")
        f.write("  5_output_gaussian_snapshot.png       Gaussian 4视角快照\n")
        f.write("  5_output_radiance_field_snapshot.png RadianceField 4视角快照\n")
        f.write("  5_output_mesh_snapshot.png           Mesh 4视角快照\n")
        f.write("  5_output_decode_summary.txt          解码结果摘要\n\n")
        f.write("阶段6: 后处理与导出\n")
        f.write("  6_input_gaussian_frame0.png          Gaussian渲染首帧\n")
        f.write("  6_input_radiance_field_frame0.png    RadianceField渲染首帧\n")
        f.write("  6_input_mesh_frame0.png              Mesh渲染首帧\n")
        f.write("  6_output_video_gaussian.mp4          Gaussian旋转视频\n")
        f.write("  6_output_video_radiance_field.mp4    RadianceField旋转视频\n")
        f.write("  6_output_video_mesh.mp4              Mesh法线旋转视频\n")
        f.write("  6_output_textured_mesh.glb           带纹理网格(GLB)\n")
        f.write("  6_output_gaussian.ply                Gaussian点云(PLY)\n")

    print(f"\n{'='*60}")
    print(f"全流程可视化完成！结果保存在: {out_dir}/")
    print(f"文件索引: {index_path}")
    print(f"{'='*60}")


if __name__ == '__main__':
    image_path = sys.argv[1] if len(sys.argv) > 1 else "assets/example_image/2d_edit.png"
    case_name = sys.argv[2] if len(sys.argv) > 2 else os.path.splitext(os.path.basename(image_path))[0]
    seed = int(sys.argv[3]) if len(sys.argv) > 3 else 1

    print(f"Image: {image_path}")
    print(f"Case:  {case_name}")
    print(f"Seed:  {seed}")
    run_visualization(image_path, case_name, seed)
