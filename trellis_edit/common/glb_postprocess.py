from __future__ import annotations

import json
import struct
from pathlib import Path

import numpy as np
import trimesh


GLB_HEADER = struct.Struct("<4sII")
CHUNK_HEADER = struct.Struct("<I4s")
JSON_CHUNK_TYPE = b"JSON"


def load_glb_json(path: Path | str) -> tuple[dict, list[tuple[bytes, bytes]], int]:
    resolved = Path(path)
    data = resolved.read_bytes()
    magic, version, length = GLB_HEADER.unpack_from(data, 0)
    if magic != b"glTF":
        raise ValueError(f"Not a GLB file: {resolved}")
    if length != len(data):
        raise ValueError(f"Corrupt GLB length in {resolved}: header={length}, actual={len(data)}")

    offset = GLB_HEADER.size
    chunks: list[tuple[bytes, bytes]] = []
    gltf: dict | None = None
    while offset < length:
        chunk_length, chunk_type = CHUNK_HEADER.unpack_from(data, offset)
        offset += CHUNK_HEADER.size
        chunk_data = data[offset : offset + chunk_length]
        offset += chunk_length
        chunks.append((chunk_type, chunk_data))
        if chunk_type == JSON_CHUNK_TYPE:
            gltf = json.loads(chunk_data.decode("utf-8"))

    if gltf is None:
        raise ValueError(f"GLB missing JSON chunk: {resolved}")
    return gltf, chunks, version


def dump_glb_json(path: Path | str, gltf: dict, chunks: list[tuple[bytes, bytes]], version: int) -> None:
    resolved = Path(path)
    json_bytes = json.dumps(gltf, ensure_ascii=False, separators=(",", ":")).encode("utf-8")
    json_padding = (4 - len(json_bytes) % 4) % 4
    json_bytes += b" " * json_padding

    payload = bytearray()
    replaced_json = False
    for chunk_type, chunk_data in chunks:
        if chunk_type == JSON_CHUNK_TYPE and not replaced_json:
            chunk_data = json_bytes
            replaced_json = True
        payload += CHUNK_HEADER.pack(len(chunk_data), chunk_type)
        payload += chunk_data

    if not replaced_json:
        raise ValueError(f"GLB document missing JSON chunk while writing {resolved}")

    header = GLB_HEADER.pack(b"glTF", version, GLB_HEADER.size + len(payload))
    resolved.parent.mkdir(parents=True, exist_ok=True)
    resolved.write_bytes(header + payload)


def rewrite_glb_materials_to_matte_nonmetal(path: Path | str, *, drop_normal: bool) -> dict:
    resolved = Path(path)
    gltf, chunks, version = load_glb_json(resolved)
    materials = gltf.get("materials", [])
    removed_mr = 0
    removed_emissive = 0
    removed_normal = 0

    for material in materials:
        pbr = material.setdefault("pbrMetallicRoughness", {})
        if "metallicRoughnessTexture" in pbr:
            pbr.pop("metallicRoughnessTexture", None)
            removed_mr += 1
        pbr["metallicFactor"] = 0.0
        pbr["roughnessFactor"] = 1.0

        if "emissiveTexture" in material:
            material.pop("emissiveTexture", None)
            removed_emissive += 1
        if "emissiveFactor" in material:
            material.pop("emissiveFactor", None)
            removed_emissive += 1

        if drop_normal and "normalTexture" in material:
            material.pop("normalTexture", None)
            removed_normal += 1

    dump_glb_json(resolved, gltf, chunks, version)
    return {
        "materials": len(materials),
        "removed_mr": removed_mr,
        "removed_emissive": removed_emissive,
        "removed_normal": removed_normal,
        "drop_normal": bool(drop_normal),
    }


def flatten_glb_scene(path: Path | str) -> trimesh.Scene:
    resolved = Path(path)
    scene = trimesh.load(resolved, force="scene")
    flat_scene = trimesh.Scene()
    node_count = 0
    for node_name in scene.graph.nodes_geometry:
        transform, geom_name = scene.graph[node_name]
        geom = scene.geometry[geom_name].copy()
        geom.apply_transform(transform)
        flat_scene.add_geometry(geom, node_name=node_name, geom_name=node_name)
        node_count += 1
    if node_count == 0:
        raise RuntimeError(f"No geometry found in GLB scene: {resolved}")
    return flat_scene


def scene_bounds_from_flat_scene(scene: trimesh.Scene) -> tuple[np.ndarray, np.ndarray]:
    mins: list[np.ndarray] = []
    maxs: list[np.ndarray] = []
    for geom in scene.geometry.values():
        mins.append(geom.bounds[0])
        maxs.append(geom.bounds[1])
    if not mins:
        raise RuntimeError("Cannot compute bounds for an empty scene.")
    return np.vstack(mins).min(axis=0), np.vstack(maxs).max(axis=0)


def compute_uniform_bbox_transform(
    input_glb: Path | str,
    reference_glb: Path | str,
) -> tuple[np.ndarray, float, np.ndarray, np.ndarray, np.ndarray, float, float]:
    input_scene = flatten_glb_scene(input_glb)
    reference_scene = flatten_glb_scene(reference_glb)
    input_min, input_max = scene_bounds_from_flat_scene(input_scene)
    reference_min, reference_max = scene_bounds_from_flat_scene(reference_scene)
    input_center = (input_min + input_max) * 0.5
    reference_center = (reference_min + reference_max) * 0.5
    input_extent = float((input_max - input_min).max())
    reference_extent = float((reference_max - reference_min).max())
    scale = reference_extent / input_extent if input_extent > 0 else 1.0
    translation = reference_center - input_center * scale

    matrix = np.eye(4, dtype=np.float64)
    matrix[0, 0] = scale
    matrix[1, 1] = scale
    matrix[2, 2] = scale
    matrix[:3, 3] = translation
    return matrix, scale, translation, input_center, reference_center, input_extent, reference_extent


def summarize_glb_bounds(path: Path | str) -> dict:
    scene = trimesh.load(Path(path), force="scene")
    bounds = scene.bounds
    center = (bounds[0] + bounds[1]) * 0.5
    extent = bounds[1] - bounds[0]
    return {
        "min": bounds[0].tolist(),
        "max": bounds[1].tolist(),
        "center": center.tolist(),
        "extent": extent.tolist(),
        "max_extent": float(extent.max()),
    }


def export_aligned_glb_to_reference(
    *,
    input_glb: Path | str,
    reference_glb: Path | str,
    output_glb: Path | str,
    apply_matte_nonmetal: bool = True,
    drop_normal: bool = False,
) -> dict:
    input_path = Path(input_glb).expanduser().resolve()
    reference_path = Path(reference_glb).expanduser().resolve()
    output_path = Path(output_glb).expanduser().resolve()

    matrix, scale, translation, input_center, reference_center, input_extent, reference_extent = (
        compute_uniform_bbox_transform(input_path, reference_path)
    )
    scene = flatten_glb_scene(input_path)
    for geom in scene.geometry.values():
        geom.apply_transform(matrix)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    scene.export(output_path)

    material_stats = None
    if apply_matte_nonmetal:
        material_stats = rewrite_glb_materials_to_matte_nonmetal(output_path, drop_normal=drop_normal)

    return {
        "reference_glb": str(reference_path),
        "input_glb": str(input_path),
        "output_glb": str(output_path),
        "alignment_rule": "flatten_node_transforms_then_bbox_center_uniform_scale_to_reference",
        "material_rule": (
            "preserve_base_color_and_normal_remove_mr_and_emissive_set_metallic0_roughness1"
            if apply_matte_nonmetal
            else "unchanged"
        ),
        "scale": float(scale),
        "translation": [float(value) for value in translation],
        "input_center_before": [float(value) for value in input_center],
        "reference_center": [float(value) for value in reference_center],
        "input_max_extent_before": float(input_extent),
        "reference_max_extent": float(reference_extent),
        "material_stats": material_stats,
        "reference_bounds_after": summarize_glb_bounds(reference_path),
        "output_bounds_after": summarize_glb_bounds(output_path),
    }
