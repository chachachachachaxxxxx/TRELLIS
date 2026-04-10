#!/usr/bin/env python3
from __future__ import annotations

import argparse
import os
import subprocess
import sys
import time
import yaml
from pathlib import Path
from typing import Optional

from PIL import Image

from editing.common import (
    apply_backend_env,
    build_backend_config,
    build_experiment_output_layout,
    ensure_dir,
    sanitize_name,
    utc_now_iso,
    write_json,
)
from editing.io.case_loader import apply_case_overrides, load_case, summarize_case
from editing.methods import EditMethodConfig, EditMethodRunner, get_method, list_methods


REPO_ROOT = Path(__file__).resolve().parent


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Unified entrypoint for TRELLIS no-training image editing experiments."
    )
    parser.add_argument("--config", type=str, help="YAML config file path (recommended)")
    parser.add_argument("--list-methods", action="store_true", help="List registered editing methods and exit.")
    parser.add_argument("--method", default="", help="Registered method name to run.")
    parser.add_argument(
        "--case",
        default="",
        help="Case directory or manifest.json path. When omitted, use CLI asset overrides directly.",
    )
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
        help="spconv algorithm override. Defaults to case/default/native.",
    )
    parser.set_defaults(preprocess=None)
    parser.add_argument(
        "--preprocess",
        dest="preprocess",
        action="store_true",
        help="Force shared source/edit/mask preprocessing on the target method.",
    )
    parser.add_argument(
        "--no-preprocess",
        dest="preprocess",
        action="store_false",
        help="Disable shared preprocessing and rely on already aligned inputs.",
    )
    parser.add_argument("--skip-render", action="store_true", help="Forward --skip-render to the method script.")
    parser.add_argument("--skip-glb", action="store_true", help="Forward --skip-glb to the method script.")
    parser.add_argument("--skip-ply", action="store_true", help="Forward --skip-ply to the method script.")
    parser.add_argument("--asset-dir", default="", help="Override source asset directory (preprocessed 3D assets).")
    parser.add_argument("--render-dir", default="", help="Override render directory that contains voxels/features.")
    parser.add_argument("--source-model", default="", help="Override source model path.")
    parser.add_argument("--source-image", default="", help="Override aligned source render image.")
    parser.add_argument("--edit-image", default="", help="Override edited target image.")
    parser.add_argument("--mask-image", default="", help="Override 2D edit mask.")
    parser.add_argument("--mask-glb", default="", help="Override 3D edit mask GLB/GLTF.")
    parser.add_argument("--source-prompt", default="", help="Override source text prompt.")
    parser.add_argument("--edit-prompt", default="", help="Override edit text prompt.")
    parser.add_argument("--ss-steps", type=int, default=None, help="Override sparse structure sampling steps (default: 25).")
    parser.add_argument("--slat-steps", type=int, default=None, help="Override SLAT sampling steps (default: 25).")
    parser.add_argument("--device", type=str, default="cuda:0", help="Device to use (default: cuda:0).")
    parser.add_argument(
        "--init-case",
        default="",
        help="Create a standard editing case directory with source/, edit/, and manifest.json, then exit.",
    )
    parser.add_argument("--dry-run", action="store_true", help="Resolve inputs and print the method command without running it.")
    return parser


def print_methods() -> None:
    for spec in list_methods():
        print(f"{spec.name}: {spec.description}")


def resolve_effective_value(args_value, case_defaults: dict, key: str, fallback=None):
    if args_value not in (None, ""):
        return args_value
    if key in case_defaults:
        return case_defaults[key]
    return fallback


def normalize_passthrough_args(extra_args: list[str]) -> list[str]:
    if extra_args and extra_args[0] == "--":
        return extra_args[1:]
    return extra_args


def build_case_manifest_template(case_name: str) -> dict:
    return {
        "case_name": sanitize_name(case_name, fallback="case"),
        "notes": "",
        "defaults": {
            "model": "microsoft/TRELLIS-image-large",
            "seed": 1,
            "preprocess": True,
            "spconv_algo": "native",
        },
        "source": {
            "dir": "source",
        },
        "edit": {
            "dir": "edit",
        },
        "methods": {
            spec.name: {
                "args": [],
            }
            for spec in list_methods()
        },
    }


def init_case_directory(case_dir: Path) -> Path:
    case_dir = case_dir.expanduser().resolve()
    manifest_path = case_dir / "manifest.json"
    if manifest_path.exists():
        raise RuntimeError(f"Refusing to overwrite existing manifest: {manifest_path}")

    ensure_dir(case_dir / "source")
    ensure_dir(case_dir / "edit")
    write_json(manifest_path, build_case_manifest_template(case_dir.name))
    return manifest_path


def run_method_class(
    method_spec,
    case,
    effective_case_name: str,
    effective_model: str,
    effective_seed: int,
    effective_preprocess: bool,
    backend,
    skip_render: bool,
    skip_glb: bool,
    skip_ply: bool,
    extra_args: list,
    ss_steps: Optional[int] = None,
    slat_steps: Optional[int] = None,
    device: str = "cuda:0",
) -> int:
    """Run method using method class (new way).

    Args:
        method_spec: MethodSpec with method_class
        case: EditingCase
        effective_case_name: Case name
        effective_model: Model path
        effective_seed: Seed
        effective_preprocess: Preprocess flag
        backend: Backend config
        device: Device to use (e.g., cuda:0)
        skip_render: Skip render flag
        skip_glb: Skip GLB flag
        skip_ply: Skip PLY flag
        extra_args: Extra CLI args

    Returns:
        Exit code (0 for success)
    """
    # Apply backend environment
    for key, value in backend.env.items():
        os.environ[key] = value

    # Determine pipeline type based on method
    is_text_method = "text" in method_spec.name

    # Load pipeline
    print(f"Loading pipeline: {effective_model}")
    print(f"Target device: {device}")
    print(f"CUDA_VISIBLE_DEVICES: {os.environ.get('CUDA_VISIBLE_DEVICES', 'not set')}")

    import torch

    # If CUDA_VISIBLE_DEVICES is set, use cuda:0 (which maps to the visible device)
    # Otherwise use the specified device
    if "CUDA_VISIBLE_DEVICES" in os.environ and device.startswith("cuda:"):
        load_device = "cuda:0"
        print(f"CUDA_VISIBLE_DEVICES is set, loading to cuda:0 (physical device {os.environ['CUDA_VISIBLE_DEVICES']})")
    else:
        load_device = device

    if is_text_method:
        from trellis.pipelines import TrellisTextTo3DPipeline
        pipeline = TrellisTextTo3DPipeline.from_pretrained(effective_model)
    else:
        from trellis.pipelines import TrellisImageTo3DPipeline
        pipeline = TrellisImageTo3DPipeline.from_pretrained(effective_model)

    # Move pipeline to specified device
    pipeline.to(torch.device(load_device))
    print(f"Pipeline loaded on device: {pipeline.device}")

    # Load inputs based on method type
    if is_text_method:
        source_prompt = case.source_prompt
        edit_prompt = case.edit_prompt
        if not source_prompt or not edit_prompt:
            raise RuntimeError("Text method requires source_prompt and edit_prompt")
        source_image = None
        edit_image = None
        mask_image = None
    else:
        source_image = Image.open(case.source_image) if case.source_image else None
        edit_image = Image.open(case.edit_image) if case.edit_image else None
        mask_image = Image.open(case.mask_image) if case.mask_image else None
        # Some methods (like fusion) don't require source_image
        if edit_image is None:
            raise RuntimeError("Image method requires at least edit_image")
        if source_image is None and method_spec.name not in ["image_slat_xor_fusion"]:
            raise RuntimeError(f"Method {method_spec.name} requires source_image")
        source_prompt = None
        edit_prompt = None

    # Get method default config
    method = method_spec.create_method()
    default_config = method.get_default_config()

    # Parse extra args into config (merge with defaults)
    extra_params = {
        "skip_source": False,
    }

    # Apply method defaults
    extra_params.update(default_config)

    # Override with CLI flags if explicitly set
    if skip_render:
        extra_params["skip_render"] = True
    if skip_glb:
        extra_params["skip_glb"] = True
    if skip_ply:
        extra_params["skip_ply"] = True

    # Parse extra args (simple key=value parsing)
    for arg in extra_args:
        if "=" in arg:
            key, value = arg.split("=", 1)
            key = key.lstrip("-").replace("-", "_")
            # Try to parse as number or bool
            if value.lower() in ("true", "false"):
                extra_params[key] = value.lower() == "true"
            elif value.isdigit():
                extra_params[key] = int(value)
            else:
                try:
                    extra_params[key] = float(value)
                except ValueError:
                    extra_params[key] = value

    # Build sampler params
    sparse_structure_sampler_params = {}
    if ss_steps is not None:
        sparse_structure_sampler_params["steps"] = ss_steps

    slat_sampler_params = {}
    if slat_steps is not None:
        slat_sampler_params["steps"] = slat_steps

    # Build config
    config = EditMethodConfig(
        method_name=method_spec.name,
        seed=effective_seed,
        num_samples=1,
        sparse_structure_sampler_params=sparse_structure_sampler_params if sparse_structure_sampler_params else None,
        slat_sampler_params=slat_sampler_params if slat_sampler_params else None,
        extra_params=extra_params,
    )

    # Use the already created method instance
    runner = EditMethodRunner(method, pipeline)

    # Run
    print(f"Running method: {method_spec.name}")
    print(f"Case: {effective_case_name}")
    print(f"Seed: {effective_seed}")

    try:
        # Build extra inputs for text methods
        extra_inputs = {}
        if is_text_method:
            extra_inputs["source_prompt"] = source_prompt
            extra_inputs["edit_prompt"] = edit_prompt

        runner.run(
            source_image=source_image,
            edit_image=edit_image,
            mask_image=mask_image,
            config=config,
            case_name=effective_case_name,
            preprocess=effective_preprocess,
            source_voxels_path=case.asset_dir / "voxels.ply" if case.asset_dir else None,
            source_features_path=case.asset_dir / "features.npz" if case.asset_dir else None,
            mask_glb_path=case.mask_glb,
            asset_dir=case.asset_dir,
            extra_inputs=extra_inputs,
        )
        print(f"✓ Method completed successfully")
        return 0
    except Exception as e:
        print(f"✗ Method failed: {e}")
        import traceback
        traceback.print_exc()
        return 1


def main() -> int:
    parser = build_parser()
    args, extra_args = parser.parse_known_args()
    extra_args = normalize_passthrough_args(list(extra_args))

    # Load config from YAML if provided
    if args.config:
        config_path = Path(args.config)
        if not config_path.exists():
            print(f"[ERROR] Config file not found: {config_path}")
            return 1

        with open(config_path, "r") as f:
            config = yaml.safe_load(f)

        # Apply config values (CLI args override config)
        if not args.method and "method" in config:
            args.method = config["method"]
        if not args.case and "case" in config:
            args.case = config["case"]
        if not args.case_name and "case_name" in config:
            args.case_name = config["case_name"]
        if not args.model and "model" in config:
            args.model = config["model"]
        if args.seed is None and "seed" in config:
            args.seed = config["seed"]
        if not args.attn_backend and "attn_backend" in config:
            args.attn_backend = config["attn_backend"]
        if not args.sparse_attn_backend and "sparse_attn_backend" in config:
            args.sparse_attn_backend = config["sparse_attn_backend"]
        if not args.spconv_algo and "spconv_algo" in config:
            args.spconv_algo = config["spconv_algo"]
        if args.preprocess is None and "preprocess" in config:
            args.preprocess = config["preprocess"]
        if not args.skip_render and config.get("skip_render", False):
            args.skip_render = True
        if not args.skip_glb and config.get("skip_glb", False):
            args.skip_glb = True
        if not args.skip_ply and config.get("skip_ply", False):
            args.skip_ply = True
        if not args.asset_dir and "asset_dir" in config:
            args.asset_dir = config["asset_dir"]
        if not args.render_dir and "render_dir" in config:
            args.render_dir = config["render_dir"]
        if not args.source_model and "source_model" in config:
            args.source_model = config["source_model"]
        if not args.source_image and "source_image" in config:
            args.source_image = config["source_image"]
        if not args.edit_image and "edit_image" in config:
            args.edit_image = config["edit_image"]
        if not args.mask_image and "mask_image" in config:
            args.mask_image = config["mask_image"]
        if not args.mask_glb and "mask_glb" in config:
            args.mask_glb = config["mask_glb"]
        if not args.source_prompt and "source_prompt" in config:
            args.source_prompt = config["source_prompt"]
        if not args.edit_prompt and "edit_prompt" in config:
            args.edit_prompt = config["edit_prompt"]
        if args.ss_steps is None and "ss_steps" in config:
            args.ss_steps = config["ss_steps"]
        if args.slat_steps is None and "slat_steps" in config:
            args.slat_steps = config["slat_steps"]
        if not args.device and "device" in config:
            args.device = config["device"]

        # Load method_args from config if present
        if "method_args" in config and not extra_args:
            method_args_dict = config["method_args"]
            for key, value in method_args_dict.items():
                extra_args.append(f"--{key}")
                # 布尔值 True 转换为标志参数（不带值）
                # None 或 False 跳过
                # 其他值（包括空字符串 ""）都添加
                if value is True:
                    continue  # 标志参数，不添加值
                elif value is not None and value is not False:
                    extra_args.append(str(value))

    if args.list_methods:
        print_methods()
        return 0

    if args.init_case:
        manifest_path = init_case_directory(Path(args.init_case))
        print(f"Created editing case template: {manifest_path}")
        print(f"Put source assets under: {manifest_path.parent / 'source'}")
        print(f"Put edit assets under: {manifest_path.parent / 'edit'}")
        return 0

    if not args.method:
        parser.error("--method is required unless --list-methods or --init-case is used.")

    method = get_method(args.method)
    case = load_case(args.case or None)
    case = apply_case_overrides(
        case,
        case_name=args.case_name,
        seed=args.seed,
        asset_dir=args.asset_dir,
        render_dir=args.render_dir,
        source_model=args.source_model,
        source_image=args.source_image,
        edit_image=args.edit_image,
        mask_image=args.mask_image,
        mask_glb=args.mask_glb,
        source_prompt=args.source_prompt,
        edit_prompt=args.edit_prompt,
    )
    method.validate_case(case)

    effective_case_name = sanitize_name(case.case_name, fallback=method.name)

    # Choose default model based on method type
    default_model = "microsoft/TRELLIS-text-large" if "text" in method.name else "microsoft/TRELLIS-image-large"
    effective_model = resolve_effective_value(args.model, case.defaults, "model", default_model)
    effective_seed = resolve_effective_value(case.seed, case.defaults, "seed")
    if args.seed is not None:
        effective_seed = args.seed
    effective_preprocess = args.preprocess
    if effective_preprocess is None and "preprocess" in case.defaults:
        effective_preprocess = bool(case.defaults["preprocess"])

    effective_attn_backend = resolve_effective_value(args.attn_backend, case.defaults, "attn_backend", "")
    effective_sparse_attn_backend = resolve_effective_value(
        args.sparse_attn_backend,
        case.defaults,
        "sparse_attn_backend",
        "",
    )
    effective_spconv_algo = resolve_effective_value(args.spconv_algo, case.defaults, "spconv_algo", "native")
    backend = build_backend_config(
        attn_backend=str(effective_attn_backend or ""),
        sparse_attn_backend=str(effective_sparse_attn_backend or ""),
        spconv_algo=str(effective_spconv_algo or "native"),
    )

    method_defaults = case.method_defaults(method.name)
    manifest_extra_args = method_defaults.get("args", [])
    if manifest_extra_args and not isinstance(manifest_extra_args, list):
        raise RuntimeError(
            f"Manifest methods.{method.name}.args must be a list, got {type(manifest_extra_args).__name__}."
        )

    layout = build_experiment_output_layout(method.name, effective_case_name)

    # Check if method has a method class implementation
    if method.has_method_class():
        print(f"Using method class for: {method.name}")
        return run_method_class(
            method_spec=method,
            case=case,
            effective_case_name=effective_case_name,
            effective_model=str(effective_model),
            effective_seed=effective_seed,
            effective_preprocess=effective_preprocess,
            backend=backend,
            skip_render=bool(args.skip_render),
            skip_glb=bool(args.skip_glb),
            skip_ply=bool(args.skip_ply),
            extra_args=extra_args,
            ss_steps=args.ss_steps,
            slat_steps=args.slat_steps,
            device=args.device,
        )

    # Fallback to script-based execution (legacy)
    print(f"Using legacy script for: {method.name}")
    command = [
        sys.executable,
        str(method.script_path),
        *method.build_command_args(
            case=case,
            case_name=effective_case_name,
            model=str(effective_model),
            seed=effective_seed,
            preprocess=effective_preprocess,
            attn_backend=backend.attn_backend,
            skip_render=bool(args.skip_render),
            skip_glb=bool(args.skip_glb),
            skip_ply=bool(args.skip_ply),
            extra_args=extra_args,
            manifest_extra_args=manifest_extra_args,
        ),
    ]

    runner_config = {
        "runner": "run_edit_experiment.py",
        "method_name": method.name,
        "method_description": method.description,
        "case_name": effective_case_name,
        "model": str(effective_model),
        "seed": effective_seed,
        "preprocess": effective_preprocess,
        "backend": backend.env,
        "skip_render": bool(args.skip_render),
        "skip_glb": bool(args.skip_glb),
        "skip_ply": bool(args.skip_ply),
        "case": summarize_case(case),
        "manifest_method_args": manifest_extra_args,
        "cli_passthrough_args": list(extra_args),
        "resolved_command": command,
        "output_case_dir": str(layout.case_dir),
        "output_edit_dir": str(layout.edit_dir),
        "output_artifacts_dir": str(layout.artifacts_dir),
        "output_logs_dir": str(layout.logs_dir),
    }

    if args.dry_run:
        print("Resolved case:")
        for key, value in summarize_case(case).items():
            print(f"  {key}: {value}")
        print(f"Planned output: {layout.case_dir}")
        print("Command:")
        print("  " + " ".join(command))
        return 0

    ensure_dir(layout.case_dir)
    ensure_dir(layout.artifacts_dir)
    ensure_dir(layout.logs_dir)
    write_json(layout.case_dir / "config.json", runner_config)
    write_json(layout.artifacts_dir / "case_snapshot.json", case.snapshot())
    write_json(layout.artifacts_dir / "runner_invocation.json", runner_config)
    stdout_log = layout.logs_dir / "runner.stdout.log"
    stderr_log = layout.logs_dir / "runner.stderr.log"
    started_at = utc_now_iso()
    write_json(
        layout.case_dir / "status.json",
        {
            "status": "running",
            "started_at": started_at,
            "finished_at": None,
            "elapsed_seconds": None,
            "method_name": method.name,
            "case_name": effective_case_name,
            "output_case_dir": str(layout.case_dir),
            "stdout_log": str(stdout_log),
            "stderr_log": str(stderr_log),
        },
    )

    env = apply_backend_env(backend)
    start_time = time.time()
    with stdout_log.open("w", encoding="utf-8") as stdout_handle, stderr_log.open("w", encoding="utf-8") as stderr_handle:
        process = subprocess.run(
            command,
            cwd=str(REPO_ROOT),
            env=env,
            stdout=stdout_handle,
            stderr=stderr_handle,
            check=False,
            text=True,
        )
    elapsed = round(time.time() - start_time, 3)
    status = "succeeded" if process.returncode == 0 else "failed"
    finished_at = utc_now_iso()
    write_json(
        layout.case_dir / "status.json",
        {
            "status": status,
            "started_at": started_at,
            "finished_at": finished_at,
            "elapsed_seconds": elapsed,
            "exit_code": process.returncode,
            "method_name": method.name,
            "case_name": effective_case_name,
            "output_case_dir": str(layout.case_dir),
            "output_edit_dir": str(layout.edit_dir),
            "stdout_log": str(stdout_log),
            "stderr_log": str(stderr_log),
        },
    )

    if process.returncode == 0:
        print(f"Experiment finished: {layout.case_dir}")
        print(f"Method outputs: {layout.edit_dir}")
        print(f"Status file: {layout.case_dir / 'status.json'}")
        return 0

    print(f"Experiment failed: {layout.case_dir}", file=sys.stderr)
    print(f"Check logs: {stdout_log} and {stderr_log}", file=sys.stderr)
    return process.returncode


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except RuntimeError as exc:
        print(f"Error: {exc}", file=sys.stderr)
        raise SystemExit(1)
