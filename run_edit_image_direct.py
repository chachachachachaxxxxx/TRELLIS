#!/usr/bin/env python3
from __future__ import annotations

import argparse
import os
from dataclasses import asdict, dataclass, field, fields, is_dataclass, replace
from pathlib import Path
from typing import Any

import yaml
from PIL import Image

from trellis_edit.common import (
    build_backend_config,
    build_experiment_output_layout,
    ensure_dir,
    save_outputs,
    write_json,
)


@dataclass(frozen=True)
class DirectRuntimeConfig:
    case_name: str
    model: str = "microsoft/TRELLIS-image-large"
    seed: int = 1
    device: str = "cuda:0"
    output_root: Path = Path("outputs")
    num_samples: int = 1
    attn_backend: str = ""
    sparse_attn_backend: str = ""
    spconv_algo: str = "native"
    skip_render: bool = True
    skip_glb: bool = False
    skip_ply: bool = False


@dataclass(frozen=True)
class DirectInputsConfig:
    edit_image: Path


@dataclass(frozen=True)
class DirectSamplerConfig:
    steps: int | None = None
    cfg_strength: float | None = None
    rescale_t: float | None = None


@dataclass(frozen=True)
class DirectMethodConfig:
    ss_sampler: DirectSamplerConfig = field(default_factory=DirectSamplerConfig)
    slat_sampler: DirectSamplerConfig = field(default_factory=DirectSamplerConfig)
    formats: tuple[str, ...] = ("mesh", "gaussian")
    preprocess_image: bool = True


@dataclass(frozen=True)
class DirectExperimentConfig:
    entrypoint: str
    runtime: DirectRuntimeConfig
    inputs: DirectInputsConfig
    direct: DirectMethodConfig

    def validate(self) -> None:
        if self.entrypoint != "edit_image_direct":
            raise RuntimeError(f"Unsupported direct entrypoint: {self.entrypoint}")
        if not self.inputs.edit_image.is_file():
            raise RuntimeError(f"edit_image does not exist: {self.inputs.edit_image}")


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Generate 3D directly from an edited image.")
    parser.add_argument("--config", type=str, required=True, help="YAML config path.")
    parser.add_argument("--case-name", default="", help="Optional runtime.case_name override.")
    parser.add_argument("--model", default="", help="Optional runtime.model override.")
    parser.add_argument("--seed", type=int, default=None, help="Optional runtime.seed override.")
    parser.add_argument("--device", default="", help="Optional runtime.device override.")
    parser.add_argument("--output-root", default="", help="Optional runtime.output_root override.")
    parser.add_argument("--edit-image", default="", help="Optional inputs.edit_image override.")
    parser.add_argument("--ss-steps", type=int, default=None, help="Optional direct.ss_sampler.steps override.")
    parser.add_argument("--slat-steps", type=int, default=None, help="Optional direct.slat_sampler.steps override.")
    parser.add_argument("--skip-render", action="store_true", help="Override runtime.skip_render=true.")
    parser.add_argument("--skip-glb", action="store_true", help="Override runtime.skip_glb=true.")
    parser.add_argument("--skip-ply", action="store_true", help="Override runtime.skip_ply=true.")
    parser.add_argument("--dry-run", action="store_true", help="Print resolved config without executing.")
    return parser


def _coerce_path(value: str | Path | None) -> Path | None:
    if value in (None, ""):
        return None
    return Path(value).expanduser().resolve()


def _serialize(value: Any) -> Any:
    if is_dataclass(value):
        return _serialize(asdict(value))
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, dict):
        return {key: _serialize(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [_serialize(item) for item in value]
    return value


def _apply_dataclass_overrides(template: Any, overrides: dict[str, Any], label: str) -> Any:
    if not isinstance(overrides, dict):
        raise RuntimeError(f"{label} must be a mapping, got {type(overrides).__name__}")

    valid_fields = {field.name for field in fields(template)}
    unknown = sorted(set(overrides) - valid_fields)
    if unknown:
        raise RuntimeError(f"{label} has unknown keys: {', '.join(unknown)}")

    updated: dict[str, Any] = {}
    for key, value in overrides.items():
        current = getattr(template, key)
        if is_dataclass(current):
            updated[key] = _apply_dataclass_overrides(current, value, f"{label}.{key}")
        elif isinstance(current, tuple) and isinstance(value, list):
            updated[key] = tuple(value)
        elif isinstance(current, Path):
            updated[key] = _coerce_path(value)
        else:
            updated[key] = value
    return replace(template, **updated)


def _sampler_params(config: DirectSamplerConfig) -> dict[str, Any]:
    params: dict[str, Any] = {}
    if config.steps is not None:
        params["steps"] = int(config.steps)
    if config.cfg_strength is not None:
        params["cfg_strength"] = float(config.cfg_strength)
    if config.rescale_t is not None:
        params["rescale_t"] = float(config.rescale_t)
    return params


def load_config(args) -> DirectExperimentConfig:
    config_path = Path(args.config).expanduser().resolve()
    with open(config_path, "r", encoding="utf-8") as handle:
        raw = yaml.safe_load(handle) or {}
    if not isinstance(raw, dict):
        raise RuntimeError(f"Config must deserialize to a mapping, got {type(raw).__name__}")

    entrypoint = str(raw.get("entrypoint") or "edit_image_direct")
    runtime = DirectRuntimeConfig(case_name="direct_edit")
    inputs = DirectInputsConfig(edit_image=Path("."))
    direct = DirectMethodConfig()

    if isinstance(raw.get("runtime"), dict):
        runtime = _apply_dataclass_overrides(runtime, raw["runtime"], "runtime")
    if isinstance(raw.get("inputs"), dict):
        inputs = _apply_dataclass_overrides(inputs, raw["inputs"], "inputs")
    if isinstance(raw.get("direct"), dict):
        direct = _apply_dataclass_overrides(direct, raw["direct"], "direct")

    if args.case_name:
        runtime = replace(runtime, case_name=args.case_name)
    if args.model:
        runtime = replace(runtime, model=args.model)
    if args.seed is not None:
        runtime = replace(runtime, seed=args.seed)
    if args.device:
        runtime = replace(runtime, device=args.device)
    if args.output_root:
        output_root = _coerce_path(args.output_root)
        if output_root is None:
            raise RuntimeError("output_root override cannot be empty")
        runtime = replace(runtime, output_root=output_root)
    if args.skip_render:
        runtime = replace(runtime, skip_render=True)
    if args.skip_glb:
        runtime = replace(runtime, skip_glb=True)
    if args.skip_ply:
        runtime = replace(runtime, skip_ply=True)
    if args.edit_image:
        edit_image = _coerce_path(args.edit_image)
        if edit_image is None:
            raise RuntimeError("edit_image override cannot be empty")
        inputs = replace(inputs, edit_image=edit_image)
    if args.ss_steps is not None:
        direct = replace(direct, ss_sampler=replace(direct.ss_sampler, steps=args.ss_steps))
    if args.slat_steps is not None:
        direct = replace(direct, slat_sampler=replace(direct.slat_sampler, steps=args.slat_steps))

    config = DirectExperimentConfig(
        entrypoint=entrypoint,
        runtime=runtime,
        inputs=inputs,
        direct=direct,
    )
    config.validate()
    return config


def run(config: DirectExperimentConfig) -> Path:
    backend = build_backend_config(
        attn_backend=config.runtime.attn_backend,
        sparse_attn_backend=config.runtime.sparse_attn_backend,
        spconv_algo=config.runtime.spconv_algo,
    )
    for key, value in backend.env.items():
        os.environ[key] = value

    import torch
    from trellis.pipelines import TrellisImageTo3DPipeline

    requested_device = config.runtime.device
    if "CUDA_VISIBLE_DEVICES" in os.environ and requested_device.startswith("cuda:"):
        load_device = "cuda:0"
    else:
        load_device = requested_device

    pipeline = TrellisImageTo3DPipeline.from_pretrained(config.runtime.model)
    pipeline.to(torch.device(load_device))

    with Image.open(config.inputs.edit_image) as handle:
        edit_image = handle.copy()

    outputs = pipeline.run(
        edit_image,
        num_samples=config.runtime.num_samples,
        seed=config.runtime.seed,
        sparse_structure_sampler_params=_sampler_params(config.direct.ss_sampler),
        slat_sampler_params=_sampler_params(config.direct.slat_sampler),
        formats=list(config.direct.formats),
        preprocess_image=config.direct.preprocess_image,
    )

    layout = build_experiment_output_layout(
        method_name=config.entrypoint,
        case_name=config.runtime.case_name,
        outputs_root=config.runtime.output_root,
    )
    ensure_dir(layout.case_dir)
    ensure_dir(layout.edit_dir)
    write_json(layout.case_dir / "experiment_config.json", _serialize(config))
    save_outputs(
        outputs=outputs,
        out_dir=layout.edit_dir,
        skip_render=config.runtime.skip_render,
        skip_glb=config.runtime.skip_glb,
        skip_ply=config.runtime.skip_ply,
    )
    return layout.case_dir


def main() -> int:
    parser = build_parser()
    args = parser.parse_args()
    config = load_config(args)

    if args.dry_run:
        print(yaml.safe_dump(_serialize(config), sort_keys=False, allow_unicode=True))
        return 0

    case_dir = run(config)
    print(f"Direct edit-image generation finished: {case_dir}")
    return 0


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except RuntimeError as exc:
        print(f"Error: {exc}")
        raise SystemExit(1)
