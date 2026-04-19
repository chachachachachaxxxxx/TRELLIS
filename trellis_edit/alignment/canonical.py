from __future__ import annotations

from pathlib import Path

import numpy as np

from trellis_edit.common.glb_postprocess import (
    flatten_glb_scene,
    rewrite_glb_materials_to_matte_nonmetal,
    scene_bounds_from_flat_scene,
    summarize_glb_bounds,
)


def compute_uniform_canonical_scale_transform(
    *,
    input_half_extent: float,
    output_half_extent: float,
) -> tuple[np.ndarray, float]:
    if input_half_extent <= 0.0:
        raise ValueError(f"input_half_extent must be positive, got {input_half_extent}")
    if output_half_extent <= 0.0:
        raise ValueError(f"output_half_extent must be positive, got {output_half_extent}")

    scale = float(output_half_extent) / float(input_half_extent)
    matrix = np.eye(4, dtype=np.float64)
    matrix[0, 0] = scale
    matrix[1, 1] = scale
    matrix[2, 2] = scale
    return matrix, scale


def export_glb_to_canonical_space(
    *,
    input_glb: Path | str,
    output_glb: Path | str,
    input_half_extent: float,
    output_half_extent: float = 0.5,
    apply_matte_nonmetal: bool = True,
    drop_normal: bool = False,
) -> dict:
    input_path = Path(input_glb).expanduser().resolve()
    output_path = Path(output_glb).expanduser().resolve()

    matrix, scale = compute_uniform_canonical_scale_transform(
        input_half_extent=float(input_half_extent),
        output_half_extent=float(output_half_extent),
    )
    scene = flatten_glb_scene(input_path)
    input_min, input_max = scene_bounds_from_flat_scene(scene)
    input_center = (input_min + input_max) * 0.5
    input_extent = input_max - input_min
    for geom in scene.geometry.values():
        geom.apply_transform(matrix)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    scene.export(output_path)

    material_stats = None
    if apply_matte_nonmetal:
        material_stats = rewrite_glb_materials_to_matte_nonmetal(output_path, drop_normal=drop_normal)

    return {
        "input_glb": str(input_path),
        "output_glb": str(output_path),
        "alignment_rule": "fixed_uniform_scale_to_centered_canonical_cube_preserve_origin",
        "material_rule": (
            "preserve_base_color_and_normal_remove_mr_and_emissive_set_metallic0_roughness1"
            if apply_matte_nonmetal
            else "unchanged"
        ),
        "scale": float(scale),
        "translation": [0.0, 0.0, 0.0],
        "input_center_before": [float(value) for value in input_center],
        "input_max_extent_before": float(input_extent.max()),
        "input_half_extent": float(input_half_extent),
        "output_half_extent": float(output_half_extent),
        "material_stats": material_stats,
        "output_bounds_after": summarize_glb_bounds(output_path),
    }
