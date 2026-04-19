#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import struct
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable


GLB_HEADER = struct.Struct("<4sII")
CHUNK_HEADER = struct.Struct("<I4s")
JSON_CHUNK_TYPE = b"JSON"


@dataclass
class GlbDocument:
    version: int
    chunks: list[tuple[bytes, bytes]]
    gltf: dict


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Rewrite GLB PBR materials to matte non-metal variants. "
            "The script preserves base color textures by default, keeps normal maps unless "
            "--drop-normal is passed, removes metallic-roughness and emissive inputs, and "
            "writes new GLB files next to the originals."
        )
    )
    parser.add_argument(
        "inputs",
        nargs="+",
        type=Path,
        help="Input GLB files or directories that contain the target GLB files.",
    )
    parser.add_argument(
        "--match-name",
        action="append",
        default=[],
        help=(
            "When an input is a directory, process files with this name under that directory. "
            "Defaults to source_model.glb and edit.glb."
        ),
    )
    parser.add_argument(
        "--suffix",
        default="_matte_nonmetal",
        help="Suffix added before .glb for rewritten files.",
    )
    parser.add_argument(
        "--drop-normal",
        action="store_true",
        help="Also remove normalTexture references.",
    )
    parser.add_argument(
        "--overwrite",
        action="store_true",
        help="Overwrite output files if they already exist.",
    )
    return parser.parse_args()


def load_glb(path: Path) -> GlbDocument:
    data = path.read_bytes()
    magic, version, length = GLB_HEADER.unpack_from(data, 0)
    if magic != b"glTF":
        raise ValueError(f"Not a GLB file: {path}")
    if length != len(data):
        raise ValueError(f"Corrupt GLB length in {path}: header={length}, actual={len(data)}")

    offset = GLB_HEADER.size
    chunks: list[tuple[bytes, bytes]] = []
    gltf: dict | None = None
    while offset < length:
        chunk_length, chunk_type = CHUNK_HEADER.unpack_from(data, offset)
        offset += CHUNK_HEADER.size
        chunk_data = data[offset:offset + chunk_length]
        offset += chunk_length
        chunks.append((chunk_type, chunk_data))
        if chunk_type == JSON_CHUNK_TYPE:
            gltf = json.loads(chunk_data.decode("utf-8"))

    if gltf is None:
        raise ValueError(f"GLB missing JSON chunk: {path}")
    return GlbDocument(version=version, chunks=chunks, gltf=gltf)


def dump_glb(path: Path, document: GlbDocument) -> None:
    json_bytes = json.dumps(document.gltf, ensure_ascii=False, separators=(",", ":")).encode("utf-8")
    json_padding = (4 - len(json_bytes) % 4) % 4
    json_bytes += b" " * json_padding

    payload = bytearray()
    replaced_json = False
    for chunk_type, chunk_data in document.chunks:
        if chunk_type == JSON_CHUNK_TYPE and not replaced_json:
            chunk_data = json_bytes
            replaced_json = True
        payload += CHUNK_HEADER.pack(len(chunk_data), chunk_type)
        payload += chunk_data

    if not replaced_json:
        raise ValueError(f"GLB document missing JSON chunk while writing {path}")

    header = GLB_HEADER.pack(b"glTF", document.version, GLB_HEADER.size + len(payload))
    path.write_bytes(header + payload)


def iter_target_files(inputs: Iterable[Path], match_names: list[str]) -> list[Path]:
    targets: list[Path] = []
    for raw_path in inputs:
        path = raw_path.expanduser().resolve()
        if path.is_file():
            if path.suffix.lower() != ".glb":
                raise ValueError(f"Only .glb files are supported: {path}")
            targets.append(path)
            continue
        if not path.is_dir():
            raise FileNotFoundError(f"Input path not found: {path}")

        found = False
        for name in match_names:
            candidate = path / name
            if candidate.is_file():
                targets.append(candidate.resolve())
                found = True
        if not found:
            raise FileNotFoundError(
                f"No matching GLB files found in directory {path}. "
                f"Expected one of: {', '.join(match_names)}"
            )
    return targets


def rewrite_materials(gltf: dict, drop_normal: bool) -> dict:
    materials = gltf.get("materials", [])
    changed = 0
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

        changed += 1

    return {
        "materials": len(materials),
        "changed": changed,
        "removed_mr": removed_mr,
        "removed_emissive": removed_emissive,
        "removed_normal": removed_normal,
    }


def output_path_for(input_path: Path, suffix: str) -> Path:
    return input_path.with_name(f"{input_path.stem}{suffix}{input_path.suffix}")


def main() -> None:
    args = parse_args()
    match_names = args.match_name or ["source_model.glb", "edit.glb"]
    targets = iter_target_files(args.inputs, match_names)

    for input_path in targets:
        output_path = output_path_for(input_path, args.suffix)
        if output_path.exists() and not args.overwrite:
            raise FileExistsError(
                f"Refusing to overwrite existing output: {output_path}. "
                "Pass --overwrite to replace it."
            )

        document = load_glb(input_path)
        stats = rewrite_materials(document.gltf, drop_normal=args.drop_normal)
        dump_glb(output_path, document)

        print(f"[OK] {input_path} -> {output_path}")
        print(
            "     "
            f"materials={stats['materials']} "
            f"removed_mr={stats['removed_mr']} "
            f"removed_emissive={stats['removed_emissive']} "
            f"removed_normal={stats['removed_normal']}"
        )


if __name__ == "__main__":
    main()
