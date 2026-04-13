#!/usr/bin/env python3
from __future__ import annotations

import argparse
from dataclasses import asdict, fields, is_dataclass, replace
from pathlib import Path
from typing import Any

import yaml

from trellis_edit.common import build_experiment_output_layout
from trellis_edit.composable import (
    ComposableExperimentRunner,
    InputConfig,
    RuntimeConfig,
    get_entrypoint,
    has_entrypoint,
    list_entrypoints,
)


LEGACY_CONFIG_KEYS = (
    "method",
    "case",
    "method_args",
    "extra_params",
    "asset_dir",
    "render_dir",
    "source_model",
    "source_prompt",
    "edit_prompt",
)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Composable entrypoint for TRELLIS no-training image editing experiments."
    )
    parser.add_argument("--config", type=str, help="YAML config file path (recommended)")
    parser.add_argument("--list-entrypoints", action="store_true", help="List runnable composable entrypoints.")
    parser.add_argument("--entrypoint", default="", help="Composable entrypoint name to run.")
    parser.add_argument("--case-name", default="", help="Optional output case name override.")
    parser.add_argument("--model", default="", help="Pipeline checkpoint or Hugging Face repo override.")
    parser.add_argument("--seed", type=int, default=None, help="Seed override.")
    parser.add_argument("--attn-backend", default="", help="Attention backend override: flash_attn or xformers.")
    parser.add_argument(
        "--sparse-attn-backend",
        default="",
        help="Optional sparse attention backend override. Defaults to the selected attention backend.",
    )
    parser.add_argument(
        "--spconv-algo",
        default="",
        help="spconv algorithm override. Defaults to native unless runtime config overrides it.",
    )
    parser.add_argument("--skip-render", action="store_true", help="Skip preview rendering outputs.")
    parser.add_argument("--skip-glb", action="store_true", help="Skip GLB export.")
    parser.add_argument("--skip-ply", action="store_true", help="Skip PLY export.")
    parser.add_argument("--source-image", default="", help="Aligned source render image.")
    parser.add_argument("--edit-image", default="", help="Edited target image.")
    parser.add_argument("--mask-image", default="", help="2D edit mask.")
    parser.add_argument("--mask-glb", default="", help="3D edit mask GLB/GLTF.")
    parser.add_argument("--source-voxels", default="", help="Source voxel coords path.")
    parser.add_argument("--source-features", default="", help="Source SLAT feature path.")
    parser.add_argument("--edited-coords", default="", help="Edited coords path for SLAT-only runs.")
    parser.add_argument("--ss-steps", type=int, default=None, help="Override sparse structure sampling steps.")
    parser.add_argument("--slat-steps", type=int, default=None, help="Override SLAT sampling steps.")
    parser.add_argument("--num-samples", type=int, default=None, help="Override sample count.")
    parser.add_argument("--output-root", default="", help="Override output root.")
    parser.add_argument("--save-source-outputs", action="store_true", help="Save source decode outputs for SLAT runs.")
    parser.add_argument("--device", type=str, default="cuda:0", help="Device to use (default: cuda:0).")
    parser.add_argument("--dry-run", action="store_true", help="Resolve inputs and print the final experiment config.")
    return parser


def print_entrypoints() -> None:
    for entrypoint in list_entrypoints():
        print(f"{entrypoint.name}: {entrypoint.description}")


def _coerce_config_path(value: str | Path | None) -> Path | None:
    if value in (None, ""):
        return None
    return Path(value).expanduser().resolve()


def _serialize_config_value(value: Any) -> Any:
    if is_dataclass(value):
        return _serialize_config_value(asdict(value))
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, dict):
        return {key: _serialize_config_value(val) for key, val in value.items()}
    if isinstance(value, (list, tuple)):
        return [_serialize_config_value(item) for item in value]
    return value


def _apply_dataclass_overrides(template: Any, overrides: dict[str, Any], label: str) -> Any:
    if not isinstance(overrides, dict):
        raise RuntimeError(f"{label} overrides must be a mapping, got {type(overrides).__name__}")

    valid_fields = {field.name: field for field in fields(template)}
    unknown = sorted(set(overrides) - set(valid_fields))
    if unknown:
        raise RuntimeError(f"{label} has unknown keys: {', '.join(unknown)}")

    updated_values: dict[str, Any] = {}
    for key, value in overrides.items():
        current = getattr(template, key)
        if is_dataclass(current):
            updated_values[key] = _apply_dataclass_overrides(current, value, f"{label}.{key}")
        elif isinstance(current, tuple) and isinstance(value, list):
            updated_values[key] = tuple(value)
        elif isinstance(current, Path):
            updated_values[key] = _coerce_config_path(value)
        else:
            updated_values[key] = value

    return replace(template, **updated_values)


def _build_runtime(args) -> RuntimeConfig:
    output_root = _coerce_config_path(args.output_root) or Path("outputs").resolve()
    return RuntimeConfig(
        model=str(args.model),
        case_name=str(args.case_name),
        seed=int(args.seed if args.seed is not None else 42),
        device=str(args.device),
        output_root=output_root,
        num_samples=int(args.num_samples or 1),
        attn_backend=str(args.attn_backend or ""),
        sparse_attn_backend=str(args.sparse_attn_backend or ""),
        spconv_algo=str(args.spconv_algo or "native"),
        skip_render=bool(args.skip_render),
        skip_glb=bool(args.skip_glb),
        skip_ply=bool(args.skip_ply),
        save_source_outputs=bool(args.save_source_outputs),
    )


def _build_inputs(args) -> InputConfig:
    return InputConfig(
        source_image=_coerce_config_path(args.source_image),
        edit_image=_coerce_config_path(args.edit_image),
        mask_image=_coerce_config_path(args.mask_image),
        mask_glb=_coerce_config_path(args.mask_glb),
        source_voxels=_coerce_config_path(args.source_voxels),
        source_features=_coerce_config_path(args.source_features),
        edited_coords=_coerce_config_path(args.edited_coords),
    )


def _apply_config_overrides(base_config, config_data: dict[str, Any]):
    config = base_config

    preprocess_overrides = config_data.get("preprocess")
    if preprocess_overrides is not None:
        config = replace(
            config,
            preprocess=_apply_dataclass_overrides(config.preprocess, preprocess_overrides, "preprocess"),
        )

    ss_overrides = config_data.get("ss")
    if ss_overrides is not None:
        if config.ss is None:
            raise RuntimeError(f"{config.entry_name} does not expose an ss stage")
        config = replace(
            config,
            ss=replace(
                config.ss,
                config=_apply_dataclass_overrides(config.ss.config, ss_overrides, "ss"),
            ),
        )

    slat_overrides = config_data.get("slat")
    if slat_overrides is not None:
        if config.slat is None:
            raise RuntimeError(f"{config.entry_name} does not expose a slat stage")
        config = replace(
            config,
            slat=replace(
                config.slat,
                config=_apply_dataclass_overrides(config.slat.config, slat_overrides, "slat"),
            ),
        )

    return config


def _reject_legacy_config(config_data: dict[str, Any]) -> None:
    legacy_keys = [key for key in LEGACY_CONFIG_KEYS if key in config_data]
    if legacy_keys:
        raise RuntimeError(
            "Legacy config keys are no longer supported: "
            + ", ".join(sorted(legacy_keys))
            + ". Use entrypoint + runtime/inputs/preprocess/ss/slat."
        )


def _load_config(args, parser: argparse.ArgumentParser) -> dict[str, Any]:
    config_data: dict[str, Any] = {}
    if not args.config:
        return config_data

    config_path = Path(args.config)
    if not config_path.exists():
        parser.error(f"Config file not found: {config_path}")

    with open(config_path, "r", encoding="utf-8") as handle:
        loaded_config = yaml.safe_load(handle) or {}
    if not isinstance(loaded_config, dict):
        raise RuntimeError(f"Config file must deserialize to a mapping, got {type(loaded_config).__name__}")

    _reject_legacy_config(loaded_config)
    config_data = loaded_config
    runtime_config = loaded_config.get("runtime") if isinstance(loaded_config.get("runtime"), dict) else {}
    inputs_config = loaded_config.get("inputs") if isinstance(loaded_config.get("inputs"), dict) else {}

    default_device = parser.get_default("device")

    def nested_value(section: dict[str, Any], key: str):
        if key in section and section[key] not in (None, ""):
            return section[key]
        return None

    if not args.entrypoint and (value := loaded_config.get("entrypoint")) not in (None, ""):
        args.entrypoint = value
    if not args.case_name and (value := nested_value(runtime_config, "case_name")) is not None:
        args.case_name = value
    if not args.model and (value := nested_value(runtime_config, "model")) is not None:
        args.model = value
    if args.seed is None and runtime_config.get("seed") is not None:
        args.seed = runtime_config["seed"]
    if not args.attn_backend and (value := nested_value(runtime_config, "attn_backend")) is not None:
        args.attn_backend = value
    if not args.sparse_attn_backend and (value := nested_value(runtime_config, "sparse_attn_backend")) is not None:
        args.sparse_attn_backend = value
    if not args.spconv_algo and (value := nested_value(runtime_config, "spconv_algo")) is not None:
        args.spconv_algo = value
    if not args.skip_render and bool(runtime_config.get("skip_render")):
        args.skip_render = True
    if not args.skip_glb and bool(runtime_config.get("skip_glb")):
        args.skip_glb = True
    if not args.skip_ply and bool(runtime_config.get("skip_ply")):
        args.skip_ply = True
    if not args.save_source_outputs and bool(runtime_config.get("save_source_outputs")):
        args.save_source_outputs = True
    if args.num_samples is None and runtime_config.get("num_samples") is not None:
        args.num_samples = runtime_config["num_samples"]
    if not args.output_root and (value := nested_value(runtime_config, "output_root")) is not None:
        args.output_root = value
    if args.device == default_device and (value := nested_value(runtime_config, "device")) is not None:
        args.device = value

    for field_name, arg_name in (
        ("source_image", "source_image"),
        ("edit_image", "edit_image"),
        ("mask_image", "mask_image"),
        ("mask_glb", "mask_glb"),
        ("source_voxels", "source_voxels"),
        ("source_features", "source_features"),
        ("edited_coords", "edited_coords"),
    ):
        if getattr(args, arg_name) in ("", None):
            value = nested_value(inputs_config, field_name)
            if value is not None:
                setattr(args, arg_name, value)

    return config_data


def run_entrypoint(entrypoint_name: str, args, config_data: dict[str, Any]) -> int:
    if not args.model:
        raise RuntimeError(f"{entrypoint_name} requires --model or runtime.model")
    if not args.case_name:
        raise RuntimeError(f"{entrypoint_name} requires --case-name or runtime.case_name")

    entrypoint = get_entrypoint(entrypoint_name)
    runtime = _build_runtime(args)
    inputs = _build_inputs(args)
    config = entrypoint.build(runtime=runtime, inputs=inputs)
    config = _apply_config_overrides(config, config_data)

    if args.ss_steps is not None:
        if config.ss is None:
            raise RuntimeError(f"{config.entry_name} does not expose an ss stage")
        config = replace(
            config,
            ss=replace(
                config.ss,
                config=replace(config.ss.config, sampler=replace(config.ss.config.sampler, steps=args.ss_steps)),
            ),
        )

    if args.slat_steps is not None:
        if config.slat is None:
            raise RuntimeError(f"{config.entry_name} does not expose a slat stage")
        config = replace(
            config,
            slat=replace(
                config.slat,
                config=replace(config.slat.config, sampler=replace(config.slat.config.sampler, steps=args.slat_steps)),
            ),
        )

    config.validate_inputs()

    if args.dry_run:
        print(yaml.safe_dump(_serialize_config_value(config), sort_keys=False, allow_unicode=True))
        return 0

    runner = ComposableExperimentRunner(config)
    runner.run()
    layout = build_experiment_output_layout(
        method_name=config.entry_name,
        case_name=config.runtime.case_name,
        outputs_root=config.runtime.output_root,
    )
    print(f"Experiment finished: {layout.case_dir}")
    print(f"Outputs: {layout.edit_dir}")
    return 0


def main() -> int:
    parser = build_parser()
    args = parser.parse_args()
    config_data = _load_config(args, parser)

    if args.list_entrypoints:
        print_entrypoints()
        return 0

    if not args.entrypoint:
        parser.error("--entrypoint is required unless --list-entrypoints is used.")
    if not has_entrypoint(args.entrypoint):
        parser.error(f"Unknown composable entrypoint: {args.entrypoint}")

    return run_entrypoint(args.entrypoint, args, config_data)


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except RuntimeError as exc:
        print(f"Error: {exc}")
        raise SystemExit(1)
