#!/usr/bin/env python3
"""
按 VoxHammer edit_pipeline.py 的逻辑读取 source/delete 体素，
计算 preserve 区域，导出 JSON 供 Three.js 可视化。

用法:
    python export_voxel_regions.py \
        --render_dir outputs/image_uniedit_rf_inversion/render \
        --output voxel_regions.json
"""
import argparse
import json
import numpy as np
import torch
import utils3d


def ply_to_coords(ply_path):
    """VoxHammer 原始逻辑：读取 PLY → 64^3 整数体素坐标。"""
    position = utils3d.io.read_ply(ply_path)[0]
    coords = ((torch.tensor(position) + 0.5) * 64).int().contiguous()
    return coords


def coords_to_flat_indices(coords):
    return coords[:, 0] * 64 * 64 + coords[:, 1] * 64 + coords[:, 2]


def coords_to_positions(coords, resolution=64):
    """整数体素坐标 → 归一化浮点坐标（中心对齐），范围 [-0.5, 0.5)。"""
    return (coords.float() + 0.5) / float(resolution) - 0.5


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--render_dir", required=True)
    parser.add_argument("--output", default="voxel_regions.json")
    args = parser.parse_args()

    import os
    voxels_path = os.path.join(args.render_dir, "voxels.ply")
    delete_path = os.path.join(args.render_dir, "voxels_delete.ply")

    coords_src = ply_to_coords(voxels_path)
    coords_delete = ply_to_coords(delete_path)

    src_1d = coords_to_flat_indices(coords_src)
    del_1d = coords_to_flat_indices(coords_delete)
    preserve_mask = ~torch.isin(src_1d, del_1d)
    coords_preserve = coords_src[preserve_mask]

    def to_list(coords):
        pts = coords_to_positions(coords).numpy().tolist()
        return pts

    data = {
        "resolution": 64,
        "source": {
            "count": int(coords_src.shape[0]),
            "positions": to_list(coords_src),
        },
        "delete": {
            "count": int(coords_delete.shape[0]),
            "positions": to_list(coords_delete),
        },
        "preserve": {
            "count": int(coords_preserve.shape[0]),
            "positions": to_list(coords_preserve),
        },
        "stats": {
            "source_total": int(coords_src.shape[0]),
            "delete_total": int(coords_delete.shape[0]),
            "preserve_total": int(coords_preserve.shape[0]),
            "source_in_delete": int(torch.isin(src_1d, del_1d).sum().item()),
        },
    }

    with open(args.output, "w") as f:
        json.dump(data, f)
    print(f"Exported {args.output}")
    for k, v in data["stats"].items():
        print(f"  {k}: {v}")


if __name__ == "__main__":
    main()
