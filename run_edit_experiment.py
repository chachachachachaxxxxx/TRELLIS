#!/usr/bin/env python3
from __future__ import annotations

import argparse
import subprocess
import sys
import time
from pathlib import Path

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
from editing.methods import get_method, list_methods


REPO_ROOT = Path(__file__).resolve().parent


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Unified entrypoint for TRELLIS no-training image editing experiments."
    )
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
    parser.add_argument("--source-model", default="", help="Override source RF asset directory or file.")
    parser.add_argument("--render-dir", default="", help="Override render directory that contains voxels/features.")
    parser.add_argument("--input-model", default="", help="Override compatibility input model path.")
    parser.add_argument("--source-image", default="", help="Override aligned source render image.")
    parser.add_argument("--edit-image", default="", help="Override edited target image.")
    parser.add_argument("--mask-image", default="", help="Override 2D edit mask.")
    parser.add_argument("--mask-glb", default="", help="Override 3D edit mask GLB/GLTF.")
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


def main() -> int:
    parser = build_parser()
    args, extra_args = parser.parse_known_args()
    extra_args = normalize_passthrough_args(list(extra_args))

    if args.list_methods:
        print_methods()
        return 0

    if not args.method:
        parser.error("--method is required unless --list-methods is used.")

    method = get_method(args.method)
    case = load_case(args.case or None)
    case = apply_case_overrides(
        case,
        case_name=args.case_name,
        seed=args.seed,
        source_model=args.source_model,
        render_dir=args.render_dir,
        input_model=args.input_model,
        source_image=args.source_image,
        edit_image=args.edit_image,
        mask_image=args.mask_image,
        mask_glb=args.mask_glb,
    )
    method.validate_case(case)

    effective_case_name = sanitize_name(case.case_name, fallback=method.name)
    effective_model = resolve_effective_value(args.model, case.defaults, "model", "microsoft/TRELLIS-image-large")
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
