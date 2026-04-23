#!/usr/bin/env python3
"""
批量运行 composable 编辑实验并评测。

推荐使用 `--config edit_configs/template.config` 一类的结构化 YAML：
- 顶层使用 `entrypoint`
- 运行参数使用 `runtime`
- 算法参数使用 `preprocess` / `ss` / `slat`
- 批量评测参数使用 `batch`
"""

from __future__ import annotations

import argparse
import json
import os
import shutil
import subprocess
import sys
import time
from copy import deepcopy
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Any

import yaml

from trellis_edit.common import build_experiment_output_layout, ensure_dir, write_json
from trellis_edit.benchmarking import (
    DEFAULT_BENCHMARK_ROOT,
    build_daily_index,
    build_focus_index,
    create_daily_benchmark_bundle,
)
from trellis_edit.composable import get_entrypoint, has_entrypoint
from trellis_edit.utils.voxel_mesh_converter import (
    load_coords_from_file,
    save_voxel_mesh,
    voxel_mesh_glb_transform_payload,
)


DEFAULT_GT_ROOT = Path("/home/wangxinxing/code/Edit3Dpp/data")
DEFAULT_PRED_DIRNAME = "pred"
DEFAULT_METRICS = [
    "psnr",
    "ssim",
    "lpips",
    "fid",
    "dino_if_max",
    "dino_if_mean",
    "chamfer",
    "clip_t",
]
DEFAULT_BATCH_INPUT_TEMPLATES = {
    "source_image": "{gt_root}/{dataset}/{object_name}/prompt_{prompt_id}/2d_render.png",
    "edit_image": "{gt_root}/{dataset}/{object_name}/prompt_{prompt_id}/2d_edit.png",
    "mask_image": "{gt_root}/{dataset}/{object_name}/prompt_{prompt_id}/2d_mask.png",
    "mask_glb": "{gt_root}/{dataset}/{object_name}/prompt_{prompt_id}/3d_edit_region.glb",
    "source_voxels": "{assets_root}/{dataset}/{object_name}/voxels.ply",
    "source_features": "{assets_root}/{dataset}/{object_name}/features.npz",
    "edited_coords": "",
}
INPUT_KEYS = (
    "source_image",
    "edit_image",
    "mask_image",
    "mask_glb",
    "source_voxels",
    "source_features",
    "edited_coords",
)

@dataclass(frozen=True)
class PreparedCaseRun:
    dataset: str
    object_name: str
    prompt_id: int
    case_name: str
    config_path: Path
    log_path: Path
    expected_glb: Path
    final_glb: Path
    case_output_dir: Path
    source_voxels_path: Path | None


SS_ARTIFACT_FILENAMES = (
    "coords.ply",
    "voxel_mesh.glb",
    "voxel_mesh_transform.json",
    "ss_metadata.json",
)


def _default_gpu_ids(*, fallback_device: str) -> list[str]:
    visible_devices = os.environ.get("CUDA_VISIBLE_DEVICES")
    if visible_devices:
        gpu_list = [item.strip() for item in visible_devices.split(",") if item.strip()]
        if gpu_list:
            return gpu_list

    if fallback_device.startswith("cuda:"):
        try:
            import torch

            device_count = int(torch.cuda.device_count())
        except Exception:
            device_count = 0
        if device_count > 0:
            return [str(index) for index in range(device_count)]
        return [fallback_device.split(":", 1)[1]]
    return []


def parse_gpu_list(gpus_value: Any, *, fallback_device: str) -> list[str]:
    if gpus_value in (None, ""):
        return _default_gpu_ids(fallback_device=fallback_device)
    if isinstance(gpus_value, str):
        gpu_list = [item.strip() for item in gpus_value.split(",") if item.strip()]
    elif isinstance(gpus_value, (list, tuple)):
        gpu_list = [str(item).strip() for item in gpus_value if str(item).strip()]
    else:
        raise RuntimeError("batch.gpus must be a comma-separated string or list.")
    if not gpu_list:
        raise RuntimeError("GPU list is empty after parsing.")
    return gpu_list


def load_edit3d_metadata(gt_root: Path) -> list[dict[str, Any]]:
    metadata_path = gt_root / "metadata.json"
    with open(metadata_path, "r", encoding="utf-8") as handle:
        return json.load(handle)


def parse_case_id(case_id_text: str) -> tuple[str, str, int]:
    raw_text = str(case_id_text or "").strip().strip("/")
    parts = [part.strip() for part in raw_text.split("/") if part.strip()]
    if len(parts) < 3:
        raise RuntimeError(
            f"Invalid case id '{case_id_text}'. Expected '<dataset>/<object_name>/prompt_<id>'."
        )
    dataset = parts[0]
    prompt_name = parts[-1]
    object_name = "/".join(parts[1:-1]).strip()
    if not dataset or not object_name or not prompt_name.startswith("prompt_"):
        raise RuntimeError(
            f"Invalid case id '{case_id_text}'. Expected '<dataset>/<object_name>/prompt_<id>'."
        )
    try:
        prompt_id = int(prompt_name.rsplit("_", 1)[-1])
    except ValueError as exc:
        raise RuntimeError(
            f"Invalid case id '{case_id_text}'. Prompt suffix must be an integer."
        ) from exc
    return dataset, object_name, prompt_id


def _normalize_case_spec(case_spec: Any) -> tuple[str, str, int]:
    if isinstance(case_spec, str):
        return parse_case_id(case_spec)
    if not isinstance(case_spec, dict):
        raise RuntimeError(
            "Selected case entries must be strings like "
            "'GSO/Dog/prompt_1' or mappings with dataset/object_name/prompt_id."
        )
    if case_spec.get("case_id"):
        return parse_case_id(str(case_spec["case_id"]))
    dataset = str(case_spec.get("dataset") or "").strip()
    object_name = str(
        case_spec.get("object_name")
        or case_spec.get("source_model")
        or ""
    ).strip()
    prompt_id_raw = case_spec.get("prompt_id")
    if not dataset or not object_name or prompt_id_raw is None:
        raise RuntimeError(
            "Selected case mappings must contain either 'case_id' or "
            "'dataset', 'object_name', and 'prompt_id'."
        )
    try:
        prompt_id = int(prompt_id_raw)
    except (TypeError, ValueError) as exc:
        raise RuntimeError(f"Invalid prompt_id in selected case: {prompt_id_raw!r}") from exc
    return dataset, object_name, prompt_id


def _load_selected_case_specs(case_file: Path) -> list[Any]:
    if not case_file.exists():
        raise RuntimeError(f"Selected cases file not found: {case_file}")
    payload = yaml.safe_load(case_file.read_text(encoding="utf-8"))
    if isinstance(payload, list):
        return payload
    if isinstance(payload, dict):
        for key in ("cases", "selected_cases"):
            value = payload.get(key)
            if isinstance(value, list):
                return value
    raise RuntimeError(
        f"Selected cases file must be a YAML list or mapping with 'cases': {case_file}"
    )


def resolve_selected_cases(
    *,
    inline_cases: Any,
    case_file_value: Any,
    config_dir: Path,
) -> list[tuple[str, str, int]]:
    raw_specs: list[Any] = []
    if inline_cases is not None:
        if not isinstance(inline_cases, list):
            raise RuntimeError("batch.selected_cases must be a YAML list.")
        raw_specs.extend(inline_cases)
    if case_file_value:
        case_file = Path(str(case_file_value)).expanduser()
        if not case_file.is_absolute():
            case_file = (config_dir / case_file).resolve()
        raw_specs.extend(_load_selected_case_specs(case_file))

    resolved: list[tuple[str, str, int]] = []
    seen: set[tuple[str, str, int]] = set()
    for raw_spec in raw_specs:
        case_tuple = _normalize_case_spec(raw_spec)
        if case_tuple in seen:
            continue
        seen.add(case_tuple)
        resolved.append(case_tuple)
    return resolved


def available_prompt_cases(metadata: list[dict[str, Any]]) -> set[tuple[str, str, int]]:
    cases: set[tuple[str, str, int]] = set()
    for entry in metadata:
        dataset = str(entry.get("dataset") or "").strip()
        object_name = str(entry.get("source_model") or "").strip()
        if not dataset or not object_name:
            continue
        for candidate_prompt_id in (1, 2, 3):
            if entry.get(f"prompt_{candidate_prompt_id}"):
                cases.add((dataset, object_name, candidate_prompt_id))
    return cases


def run_command(
    cmd: list[str],
    *,
    cwd: Path | None = None,
    env: dict[str, str] | None = None,
    timeout: int = 3600,
) -> tuple[int, str, str]:
    print(f"[CMD] {' '.join(cmd)}")
    try:
        result = subprocess.run(
            cmd,
            cwd=cwd,
            env=env,
            capture_output=True,
            text=True,
            timeout=timeout,
        )
    except subprocess.TimeoutExpired:
        return -1, "", f"Timeout after {timeout}s"
    except Exception as exc:  # pragma: no cover - shell execution wrapper
        return -1, "", str(exc)
    return result.returncode, result.stdout, result.stderr


def format_time(seconds: float) -> str:
    hours = int(seconds // 3600)
    minutes = int((seconds % 3600) // 60)
    secs = int(seconds % 60)
    if hours > 0:
        return f"{hours}h {minutes}m {secs}s"
    if minutes > 0:
        return f"{minutes}m {secs}s"
    return f"{secs}s"


def estimate_remaining_time(*, elapsed_seconds: float, finished_cases: int, total_cases: int) -> str:
    if finished_cases <= 0:
        return "estimating"
    remaining_cases = max(total_cases - finished_cases, 0)
    if remaining_cases == 0:
        return "0s"
    throughput = finished_cases / max(elapsed_seconds, 1e-6)
    eta_seconds = remaining_cases / max(throughput, 1e-6)
    return format_time(eta_seconds)


def metrics_require_video(metrics: list[str]) -> bool:
    requested = {metric.strip().lower() for metric in metrics}
    return "fvd" in requested


def case_identifier(dataset: str, object_name: str, prompt_id: int) -> str:
    safe_object_name = object_name.replace("/", "_").replace(" ", "_")
    return f"{dataset}__{safe_object_name}__prompt_{prompt_id}"


def resolve_template_value(template: str | None, variables: dict[str, str | None]) -> str | None:
    if template in (None, ""):
        return None
    for key, value in variables.items():
        token = "{" + key + "}"
        if token in template and value in (None, ""):
            return None
    try:
        return template.format_map(variables)
    except KeyError as exc:
        raise RuntimeError(f"Unknown batch input template key: {exc.args[0]} in '{template}'") from exc


def coerce_existing_file(path_text: str | None) -> str | None:
    if path_text in (None, ""):
        return None
    path = Path(path_text).expanduser().resolve()
    if path.is_file():
        return str(path)
    return None


def stage_output_dir_name(entrypoint_name: str) -> str:
    run_mode = get_entrypoint(entrypoint_name).run_mode
    if run_mode == "ss":
        raise RuntimeError(
            f"Batch evaluation requires an entrypoint that produces final GLB outputs; '{entrypoint_name}' is ss-only."
        )
    return "slat"


def case_output_method_name(*, runtime: dict[str, Any], entrypoint_name: str) -> str:
    output_group = runtime.get("output_group")
    if isinstance(output_group, str) and output_group.strip():
        return output_group.strip()
    return entrypoint_name


def build_case_config(
    *,
    base_config: dict[str, Any],
    entrypoint_name: str,
    gt_root: Path,
    pred_root: Path,
    assets_root: Path | None,
    dataset: str,
    object_name: str,
    prompt_id: int,
    model_override: str | None,
    seed_override: int | None,
    device_override: str | None,
) -> tuple[dict[str, Any], str, Path]:
    config = deepcopy(base_config)
    batch_section = dict(config.get("batch") or {})
    user_input_templates = dict(batch_section.get("input_templates") or {})
    input_templates = dict(DEFAULT_BATCH_INPUT_TEMPLATES)
    input_templates.update(user_input_templates)

    config.pop("batch", None)
    config.pop("method", None)
    config["entrypoint"] = entrypoint_name

    work_root = (pred_root / "_runs").resolve()
    case_name = case_identifier(dataset, object_name, prompt_id)
    prompt_dir = gt_root / dataset / object_name / f"prompt_{prompt_id}"
    variables = {
        "gt_root": str(gt_root.resolve()),
        "pred_root": str(pred_root.resolve()),
        "assets_root": str(assets_root.resolve()) if assets_root is not None else None,
        "dataset": dataset,
        "object_name": object_name,
        "prompt_id": str(prompt_id),
        "prompt_dir": str(prompt_dir.resolve()),
        "case_name": case_name,
    }

    runtime = dict(config.get("runtime") or {})
    if model_override:
        runtime["model"] = model_override
    runtime.setdefault("model", "microsoft/TRELLIS-image-large")
    runtime["case_name"] = case_name
    runtime["output_root"] = str(work_root)
    if seed_override is not None:
        runtime["seed"] = int(seed_override)
    runtime.setdefault("seed", 1)
    if device_override:
        runtime["device"] = device_override
    runtime.setdefault("device", "cuda:0")
    runtime.setdefault("spconv_algo", "native")
    runtime.setdefault("skip_render", True)
    runtime.setdefault("skip_glb", False)
    runtime.setdefault("skip_ply", False)
    runtime.setdefault("save_source_outputs", False)
    if runtime["skip_glb"]:
        raise RuntimeError("Batch evaluation requires runtime.skip_glb=false so that edit.glb is produced.")
    config["runtime"] = runtime

    explicit_inputs = dict(config.get("inputs") or {})
    resolved_inputs: dict[str, str | None] = {}
    for key in INPUT_KEYS:
        if key in user_input_templates:
            resolved_inputs[key] = coerce_existing_file(
                resolve_template_value(user_input_templates.get(key), variables)
            )
            continue
        explicit_value = explicit_inputs.get(key)
        if explicit_value not in (None, ""):
            resolved_inputs[key] = coerce_existing_file(str(explicit_value))
            continue
        template_value = input_templates.get(key)
        resolved_inputs[key] = coerce_existing_file(resolve_template_value(template_value, variables))
    config["inputs"] = resolved_inputs

    layout = build_experiment_output_layout(
        case_output_method_name(runtime=runtime, entrypoint_name=entrypoint_name),
        case_name,
        outputs_root=work_root,
    )
    expected_glb = layout.edit_dir / stage_output_dir_name(entrypoint_name) / "sample_00.glb"
    return config, case_name, expected_glb


def recover_existing_glb(*, expected_glb: Path, final_glb: Path) -> bool:
    if not expected_glb.is_file():
        return False
    ensure_dir(final_glb.parent)
    shutil.copy2(expected_glb, final_glb)
    print(f"[RECOVER] 使用已有单 case GLB: {final_glb}")
    return True


def copy_ss_artifacts(
    *,
    case_output_dir: Path,
    final_case_dir: Path,
) -> list[Path]:
    ss_source_dir = case_output_dir / "edit" / "ss"
    if not ss_source_dir.is_dir():
        return []

    copied: list[Path] = []
    ss_target_dir = ensure_dir(final_case_dir / "ss")
    for filename in SS_ARTIFACT_FILENAMES:
        source_path = ss_source_dir / filename
        if not source_path.is_file():
            continue
        target_path = ss_target_dir / filename
        shutil.copy2(source_path, target_path)
        copied.append(target_path)
    return copied


def copy_source_voxelmesh_artifacts(
    *,
    source_voxels_path: Path | None,
    final_case_dir: Path,
) -> list[Path]:
    if source_voxels_path is None or not source_voxels_path.is_file():
        return []

    coords = load_coords_from_file(source_voxels_path, device="cpu")
    source_target_dir = ensure_dir(final_case_dir / "source_voxelmesh")
    voxel_mesh_path = source_target_dir / "voxel_mesh.glb"
    transform_path = source_target_dir / "voxel_mesh_transform.json"
    save_voxel_mesh(coords, voxel_mesh_path, resolution=64)
    write_json(transform_path, voxel_mesh_glb_transform_payload())
    return [voxel_mesh_path, transform_path]


def render_all_results(
    output_root: Path,
    *,
    gpu_ids: list[str],
    metrics: list[str],
) -> bool:
    render_dir = Path("VoxHammer/Edit3D-Bench")
    if not render_dir.exists():
        render_dir = Path("../VoxHammer/Edit3D-Bench")
    if not render_dir.exists():
        print("[ERROR] 找不到渲染脚本")
        return False

    if not gpu_ids:
        raise RuntimeError("Render stage requires at least one GPU id.")

    skip_video = not metrics_require_video(metrics)
    render_logs_dir = ensure_dir(output_root / "_render_logs")
    processes: list[tuple[str, subprocess.Popen[str], Any, Path]] = []
    shard_count = len(gpu_ids)

    print(
        f"[Render] Using {shard_count} GPU shard(s): "
        + ", ".join(f"cuda:{gpu_id}" for gpu_id in gpu_ids)
    )
    print(f"[Render] Skip video: {'yes' if skip_video else 'no'}")

    for shard_id, gpu_id in enumerate(gpu_ids):
        env = os.environ.copy()
        env["CUDA_VISIBLE_DEVICES"] = gpu_id
        env.setdefault("PYTHONUNBUFFERED", "1")

        cmd = [
            sys.executable,
            "render.py",
            "--base_dir",
            str(output_root.resolve()),
            "--num_shards",
            str(shard_count),
            "--shard_id",
            str(shard_id),
            "--quiet_blender",
            "--skip_existing",
        ]
        if skip_video:
            cmd.append("--skip_video")

        log_path = render_logs_dir / f"render_shard_{shard_id}.log"
        log_handle = open(log_path, "w", encoding="utf-8")
        process = subprocess.Popen(
            cmd,
            cwd=render_dir,
            env=env,
            stdout=log_handle,
            stderr=subprocess.STDOUT,
            text=True,
        )
        processes.append((gpu_id, process, log_handle, log_path))
        print(f"[Render] Launch shard {shard_id + 1}/{shard_count} on cuda:{gpu_id} -> {log_path}")

    success = True
    for gpu_id, process, log_handle, log_path in processes:
        returncode = process.wait(timeout=7200)
        log_handle.close()
        if returncode != 0:
            print(f"[Render] Shard on cuda:{gpu_id} failed -> {log_path}")
            success = False
            continue

        tail = ""
        try:
            lines = log_path.read_text(encoding="utf-8").splitlines()
            tail = "\n".join(lines[-4:])
        except Exception:
            tail = ""
        print(f"[Render] Shard on cuda:{gpu_id} finished")
        if tail:
            print(tail)

    return success


def run_evaluation(
    *,
    gt_root: Path,
    pred_root: Path,
    metrics: list[str],
    output_dir: Path,
    device: str = "cuda:0",
) -> tuple[bool, dict[str, Any] | None]:
    eval_script = Path("VoxHammer/Edit3D-Bench/eval_main.py")
    if not eval_script.exists():
        eval_script = Path("../VoxHammer/Edit3D-Bench/eval_main.py")
    if not eval_script.exists():
        print("[ERROR] 找不到评测脚本")
        return False, None

    cmd = [
        sys.executable,
        str(eval_script),
        "--gt_root",
        str(gt_root),
        "--pred_root",
        str(pred_root),
        "--metrics",
        *metrics,
        "--device",
        device,
        "--batch_size",
        "32",
        "--output_dir",
        str(output_dir),
    ]
    returncode, stdout, stderr = run_command(cmd, timeout=7200)
    if stdout:
        print(stdout)
    if stderr:
        print(f"[STDERR] {stderr}")
    if returncode != 0:
        return False, None

    summary_file = output_dir / "summary.json"
    if not summary_file.exists():
        return False, None
    with open(summary_file, "r", encoding="utf-8") as handle:
        return True, json.load(handle)


def save_results(
    *,
    output_root: Path,
    entrypoint_name: str,
    config_name: str,
    run_group: str | None,
    gt_root: Path,
    cases: list[tuple[str, str, int]],
    requested_metrics: list[str],
    benchmark_root: Path,
    skip_benchmark_render: bool,
    results: dict[str, Any] | None,
    total_time: float,
) -> None:
    if results is None:
        return

    print("\n[4/4] 保存 benchmark daily 结果...")
    bundle_root = create_daily_benchmark_bundle(
        benchmark_root=benchmark_root,
        entrypoint_name=entrypoint_name,
        config_name=config_name,
        run_group=run_group,
        gt_root=gt_root,
        pred_root=output_root,
        cases=cases,
        requested_metrics=requested_metrics,
        summary_results=results,
        total_time_seconds=total_time,
        skip_benchmark_render=skip_benchmark_render,
    )
    daily_index = build_daily_index(benchmark_root)
    focus_index = build_focus_index(benchmark_root)

    print(f"[INFO] daily 结果已保存到: {bundle_root}")
    print(f"[INFO] - 总览页面: {bundle_root / 'index.html'}")
    if daily_index is not None:
        print(f"[INFO] - daily 集合页: {daily_index}")
    if focus_index is not None:
        print(f"[INFO] - focus 总览: {focus_index}")
    print(f"[INFO] - 原始评测数据: {output_root}")


def run_single_edit(
    *,
    gt_root: Path,
    pred_root: Path,
    entrypoint_name: str,
    dataset: str,
    object_name: str,
    prompt_id: int,
    base_config: dict[str, Any],
    assets_root: Path | None,
    skip_exists: bool,
    keep_case_outputs: bool,
    model_override: str | None,
    seed_override: int | None,
    device_override: str | None,
    dry_run: bool,
) -> tuple[bool, bool]:
    prompt_eval_dir = pred_root / dataset / object_name / f"prompt_{prompt_id}"
    ensure_dir(prompt_eval_dir)
    final_glb = prompt_eval_dir / "edit.glb"
    if skip_exists and final_glb.exists():
        print(f"[SKIP] edit.glb 已存在: {final_glb}")
        return True, True

    case_config, case_name, expected_glb = build_case_config(
        base_config=base_config,
        entrypoint_name=entrypoint_name,
        gt_root=gt_root,
        pred_root=pred_root,
        assets_root=assets_root,
        dataset=dataset,
        object_name=object_name,
        prompt_id=prompt_id,
        model_override=model_override,
        seed_override=seed_override,
        device_override=device_override,
    )
    case_output_dir = expected_glb.parents[2]
    source_voxels_text = case_config.get("inputs", {}).get("source_voxels")
    source_voxels_path = Path(source_voxels_text) if source_voxels_text else None

    if skip_exists and recover_existing_glb(expected_glb=expected_glb, final_glb=final_glb):
        copy_ss_artifacts(
            case_output_dir=case_output_dir,
            final_case_dir=prompt_eval_dir,
        )
        copy_source_voxelmesh_artifacts(
            source_voxels_path=source_voxels_path,
            final_case_dir=prompt_eval_dir,
        )
        return True, True

    config_dir = ensure_dir(pred_root / "_batch_configs")
    config_path = config_dir / f"{case_name}.yaml"
    config_path.write_text(
        yaml.safe_dump(case_config, sort_keys=False, allow_unicode=True),
        encoding="utf-8",
    )

    if dry_run:
        print(f"[DRY-RUN] Case config for {case_name}:")
        print(config_path.read_text(encoding="utf-8"))
        return True, False

    if case_output_dir.exists() and not skip_exists:
        shutil.rmtree(case_output_dir)

    command = [sys.executable, "run_edit_experiment.py", "--config", str(config_path)]
    returncode, stdout, stderr = run_command(command, timeout=7200)
    if returncode != 0:
        print(f"[ERROR] 编辑实验失败: {dataset}/{object_name}/prompt_{prompt_id}")
        if stdout:
            print(stdout)
        if stderr:
            print(stderr)
        return False, False

    if stdout:
        print(stdout)
    if stderr:
        print(stderr)

    if not expected_glb.is_file():
        print(f"[ERROR] 找不到生成的 GLB: {expected_glb}")
        return False, False

    shutil.copy2(expected_glb, final_glb)
    print(f"[SUCCESS] GLB 已保存: {final_glb}")
    copied_ss_artifacts = copy_ss_artifacts(
        case_output_dir=case_output_dir,
        final_case_dir=prompt_eval_dir,
    )
    copy_source_voxelmesh_artifacts(
        source_voxels_path=source_voxels_path,
        final_case_dir=prompt_eval_dir,
    )
    if copied_ss_artifacts:
        print(f"[SUCCESS] 已保留 SS artifacts: {len(copied_ss_artifacts)} 个")
    if not keep_case_outputs and case_output_dir.exists():
        shutil.rmtree(case_output_dir)
        print(f"[CLEANUP] 已删除单 case 完整输出: {case_output_dir}")
    return True, False


def prepare_case_run(
    *,
    gt_root: Path,
    pred_root: Path,
    entrypoint_name: str,
    dataset: str,
    object_name: str,
    prompt_id: int,
    base_config: dict[str, Any],
    assets_root: Path | None,
    skip_exists: bool,
    model_override: str | None,
    seed_override: int | None,
    device_override: str | None,
) -> tuple[PreparedCaseRun | None, bool]:
    prompt_eval_dir = pred_root / dataset / object_name / f"prompt_{prompt_id}"
    ensure_dir(prompt_eval_dir)
    final_glb = prompt_eval_dir / "edit.glb"
    if skip_exists and final_glb.exists():
        print(f"[SKIP] edit.glb 已存在: {final_glb}")
        return None, True

    case_config, case_name, expected_glb = build_case_config(
        base_config=base_config,
        entrypoint_name=entrypoint_name,
        gt_root=gt_root,
        pred_root=pred_root,
        assets_root=assets_root,
        dataset=dataset,
        object_name=object_name,
        prompt_id=prompt_id,
        model_override=model_override,
        seed_override=seed_override,
        device_override=device_override,
    )
    case_output_dir = expected_glb.parents[2]
    source_voxels_text = case_config.get("inputs", {}).get("source_voxels")
    source_voxels_path = Path(source_voxels_text) if source_voxels_text else None

    if skip_exists and recover_existing_glb(expected_glb=expected_glb, final_glb=final_glb):
        copy_ss_artifacts(
            case_output_dir=case_output_dir,
            final_case_dir=prompt_eval_dir,
        )
        copy_source_voxelmesh_artifacts(
            source_voxels_path=source_voxels_path,
            final_case_dir=prompt_eval_dir,
        )
        return None, True

    config_dir = ensure_dir(pred_root / "_batch_configs")
    config_path = config_dir / f"{case_name}.yaml"
    config_path.write_text(
        yaml.safe_dump(case_config, sort_keys=False, allow_unicode=True),
        encoding="utf-8",
    )

    if case_output_dir.exists() and not skip_exists:
        shutil.rmtree(case_output_dir)

    log_dir = ensure_dir(pred_root / "_batch_logs")
    log_path = log_dir / f"{case_name}.log"
    return (
        PreparedCaseRun(
            dataset=dataset,
            object_name=object_name,
            prompt_id=prompt_id,
            case_name=case_name,
            config_path=config_path,
            log_path=log_path,
            expected_glb=expected_glb,
            final_glb=final_glb,
            case_output_dir=case_output_dir,
            source_voxels_path=source_voxels_path,
        ),
        False,
    )


def run_editing_jobs_parallel(
    *,
    gt_root: Path,
    pred_root: Path,
    entrypoint_name: str,
    base_config: dict[str, Any],
    cases: list[tuple[str, str, int]],
    assets_root: Path | None,
    gpu_ids: list[str],
    skip_exists: bool,
    keep_case_outputs: bool,
    model_override: str | None,
    seed_override: int | None,
    dry_run: bool,
) -> tuple[int, int, int]:
    if not gpu_ids:
        raise RuntimeError("At least one GPU id is required for batch editing.")

    if dry_run:
        dataset, object_name, prompt_id = cases[0]
        success, _ = run_single_edit(
            gt_root=gt_root,
            pred_root=pred_root,
            entrypoint_name=entrypoint_name,
            dataset=dataset,
            object_name=object_name,
            prompt_id=prompt_id,
            base_config=base_config,
            assets_root=assets_root,
            skip_exists=skip_exists,
            keep_case_outputs=keep_case_outputs,
            model_override=model_override,
            seed_override=seed_override,
            device_override="cuda:0",
            dry_run=True,
        )
        return (1 if success else 0), 0, 0

    pending_jobs: list[PreparedCaseRun] = []
    skip_count = 0
    for dataset, object_name, prompt_id in cases:
        prepared, skipped = prepare_case_run(
            gt_root=gt_root,
            pred_root=pred_root,
            entrypoint_name=entrypoint_name,
            dataset=dataset,
            object_name=object_name,
            prompt_id=prompt_id,
            base_config=base_config,
            assets_root=assets_root,
            skip_exists=skip_exists,
            model_override=model_override,
            seed_override=seed_override,
            device_override="cuda:0",
        )
        if skipped:
            skip_count += 1
            continue
        if prepared is not None:
            pending_jobs.append(prepared)

    if not pending_jobs:
        return 0, skip_count, 0

    total_cases = len(cases)
    finished_cases = skip_count
    success_count = 0
    failure_count = 0
    gpu_pool = list(gpu_ids)
    running: list[dict[str, Any]] = []
    start_time = time.time()
    last_status_time = 0.0

    while pending_jobs or running:
        while pending_jobs and gpu_pool:
            job = pending_jobs.pop(0)
            gpu_id = gpu_pool.pop(0)
            env = os.environ.copy()
            env["CUDA_VISIBLE_DEVICES"] = gpu_id
            env.setdefault("PYTHONUNBUFFERED", "1")
            log_handle = open(job.log_path, "w", encoding="utf-8")
            cmd = [sys.executable, "run_edit_experiment.py", "--config", str(job.config_path)]
            process = subprocess.Popen(
                cmd,
                cwd=Path(__file__).resolve().parent,
                env=env,
                stdout=log_handle,
                stderr=subprocess.STDOUT,
            )
            running.append(
                {
                    "job": job,
                    "gpu_id": gpu_id,
                    "process": process,
                    "log_handle": log_handle,
                    "start_time": time.time(),
                }
            )
            elapsed = time.time() - start_time
            eta = estimate_remaining_time(
                elapsed_seconds=elapsed,
                finished_cases=finished_cases,
                total_cases=total_cases,
            )
            print(
                f"[LAUNCH] {job.case_name} -> cuda:{gpu_id} | "
                f"running={len(running)} pending={len(pending_jobs)} | ETA {eta}"
            )

        if not running:
            continue

        time.sleep(5.0)
        now = time.time()
        still_running: list[dict[str, Any]] = []
        for item in running:
            returncode = item["process"].poll()
            if returncode is None:
                still_running.append(item)
                continue

            item["log_handle"].close()
            gpu_pool.append(item["gpu_id"])
            gpu_pool.sort(key=int)
            finished_cases += 1
            job = item["job"]
            duration = now - item["start_time"]

            if returncode == 0 and job.expected_glb.is_file():
                shutil.copy2(job.expected_glb, job.final_glb)
                copy_ss_artifacts(
                    case_output_dir=job.case_output_dir,
                    final_case_dir=job.final_glb.parent,
                )
                copy_source_voxelmesh_artifacts(
                    source_voxels_path=job.source_voxels_path,
                    final_case_dir=job.final_glb.parent,
                )
                if not keep_case_outputs and job.case_output_dir.exists():
                    shutil.rmtree(job.case_output_dir)
                success_count += 1
                state = "SUCCESS"
            else:
                failure_count += 1
                state = "FAIL"

            elapsed = now - start_time
            eta = estimate_remaining_time(
                elapsed_seconds=elapsed,
                finished_cases=finished_cases,
                total_cases=total_cases,
            )
            print(
                f"[{state}] {job.case_name} | gpu=cuda:{item['gpu_id']} | "
                f"case_time={format_time(duration)} | "
                f"done={finished_cases}/{total_cases} success={success_count} "
                f"failed={failure_count} skipped={skip_count} | "
                f"running={len(still_running)} pending={len(pending_jobs)} | "
                f"elapsed={format_time(elapsed)} ETA={eta}"
            )
            if state == "FAIL":
                print(f"[FAIL] 日志: {job.log_path}")

        running = still_running
        if running and now - last_status_time >= 60.0:
            elapsed = now - start_time
            eta = estimate_remaining_time(
                elapsed_seconds=elapsed,
                finished_cases=finished_cases,
                total_cases=total_cases,
            )
            print(
                f"[STATUS] done={finished_cases}/{total_cases} success={success_count} "
                f"failed={failure_count} skipped={skip_count} | "
                f"running={len(running)} pending={len(pending_jobs)} | "
                f"elapsed={format_time(elapsed)} ETA={eta}"
            )
            last_status_time = now

    return success_count, skip_count, failure_count


def run_editing_and_eval(
    *,
    gt_root: Path,
    pred_root: Path,
    entrypoint_name: str,
    config_name: str,
    base_config: dict[str, Any],
    cases: list[tuple[str, str, int]],
    assets_root: Path | None,
    metrics: list[str],
    device: str,
    gpu_ids: list[str],
    skip_benchmark_render: bool,
    skip_exists: bool,
    keep_case_outputs: bool,
    model_override: str | None,
    seed_override: int | None,
    dry_run: bool,
) -> tuple[bool, dict[str, Any] | None]:
    if not dry_run and not skip_exists and pred_root.exists():
        print(f"[INFO] 清理旧的输出目录: {pred_root}")
        shutil.rmtree(pred_root)
    ensure_dir(pred_root)

    print("\n" + "=" * 80)
    print(f"输出目录: {pred_root}")
    print(f"Entrypoint: {entrypoint_name}")
    print(f"配置: {config_name}")
    print(f"案例数量: {len(cases)}")
    print(f"编辑使用 GPU: {', '.join(f'cuda:{gpu_id}' for gpu_id in gpu_ids)}")
    print(f"跳过已存在: {'是' if skip_exists else '否'}")
    print(f"保留单 case 输出: {'是' if keep_case_outputs else '否'}")
    print("=" * 80)

    print("\n" + "=" * 80)
    print("步骤 1/4: 批量运行 composable 编辑实验")
    print("=" * 80)
    success_count, skip_count, failure_count = run_editing_jobs_parallel(
        gt_root=gt_root,
        pred_root=pred_root,
        entrypoint_name=entrypoint_name,
        base_config=base_config,
        cases=cases,
        assets_root=assets_root,
        gpu_ids=gpu_ids,
        skip_exists=skip_exists,
        keep_case_outputs=keep_case_outputs,
        model_override=model_override,
        seed_override=seed_override,
        dry_run=dry_run,
    )
    if dry_run:
        print("[DRY-RUN] 仅展示首个 case 的解析配置，提前结束。")
        return True, None

    print(f"\n[INFO] 编辑完成: {success_count}/{len(cases)} 成功")
    if skip_count > 0:
        print(f"[INFO] 跳过已存在: {skip_count} 个")
    if failure_count > 0:
        print(f"[WARNING] 失败案例: {failure_count} 个")
    if success_count == 0 and skip_count == 0:
        print("[ERROR] 没有成功的编辑结果")
        return False, None
    if success_count == 0 and skip_count > 0:
        print("[INFO] 没有新生成结果，使用已存在的 edit.glb 继续后续渲染与评测")

    if skip_benchmark_render:
        print("\n[INFO] 跳过 Edit3D-Bench 渲染步骤")
    else:
        print("\n" + "=" * 80)
        print("步骤 2/4: 统一渲染所有结果")
        print("=" * 80)
        render_success = render_all_results(
            pred_root,
            gpu_ids=gpu_ids,
            metrics=metrics,
        )
        if not render_success:
            print("[WARNING] 渲染失败")

    print("\n" + "=" * 80)
    print("步骤 3/4: 运行评测")
    print("=" * 80)
    eval_output_dir = pred_root / "evaluation_output"
    success, results = run_evaluation(
        gt_root=gt_root,
        pred_root=pred_root,
        metrics=metrics,
        output_dir=eval_output_dir,
        device=device,
    )
    if not success:
        print("[WARNING] 评测失败")
        return True, None
    return True, results


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="批量运行 composable 编辑实验并评测",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument("--config", type=str, help="结构化 YAML 配置文件路径（推荐）")
    parser.add_argument("--entrypoint", type=str, help="Composable entrypoint 名称")
    parser.add_argument("--gt-root", type=str, help="Edit3D-Bench GT 数据根目录")
    parser.add_argument("--config-name", type=str, help="本次批量评测的配置标识")
    parser.add_argument("--assets-root", type=str, help="源 3D 资产根目录，通常包含 voxels.ply 和 features.npz")
    parser.add_argument("--pred-root", type=str, help="批量评测输出根目录")
    parser.add_argument("--dataset", type=str, help="数据集过滤，例如 GSO")
    parser.add_argument("--object", type=str, help="物体名称过滤")
    parser.add_argument("--prompt-id", type=int, choices=[1, 2, 3], help="提示 ID 过滤")
    parser.add_argument("--max-cases", type=int, help="限制处理的案例数量")
    parser.add_argument("--model", type=str, help="覆盖 runtime.model")
    parser.add_argument("--seed", type=int, help="覆盖 runtime.seed")
    parser.add_argument("--device", type=str, help="覆盖 runtime.device 和评测 device")
    parser.add_argument("--gpus", type=str, help="编辑阶段使用的 GPU 列表，例如 '1,2'")
    parser.add_argument("--group", type=str, help="写入 daily/focus 的运行分组名")
    parser.add_argument("--metrics", nargs="+", help="覆盖评测指标")
    parser.add_argument("--skip-render", action="store_true", help="跳过 Edit3D-Bench 渲染步骤")
    parser.add_argument("--benchmark-root", type=str, help="benchmark daily/focus 根目录")
    skip_exists_group = parser.add_mutually_exclusive_group()
    skip_exists_group.add_argument("--skip-exists", dest="skip_exists", action="store_true", help="跳过已存在的 edit.glb")
    skip_exists_group.add_argument(
        "--no-skip-exists",
        dest="skip_exists",
        action="store_false",
        help="不跳过已存在的文件，重新生成所有结果",
    )
    parser.set_defaults(skip_exists=None)
    keep_outputs_group = parser.add_mutually_exclusive_group()
    keep_outputs_group.add_argument(
        "--keep-case-outputs",
        dest="keep_case_outputs",
        action="store_true",
        help="保留 _runs/<entrypoint>/<case>/ 下的完整单 case 输出",
    )
    keep_outputs_group.add_argument(
        "--drop-case-outputs",
        dest="keep_case_outputs",
        action="store_false",
        help="复制最终 edit.glb 后删除单 case 完整输出，仅保留评测所需文件",
    )
    parser.set_defaults(keep_case_outputs=None)
    parser.add_argument("--dry-run", action="store_true", help="只打印首个 case 解析后的 run_edit_experiment 配置")
    return parser


def main() -> int:
    parser = build_parser()
    args = parser.parse_args()

    config_data: dict[str, Any] = {}
    batch_config: dict[str, Any] = {}
    config_dir = Path.cwd()
    if args.config:
        config_path = Path(args.config)
        if not config_path.exists():
            print(f"[ERROR] 配置文件不存在: {config_path}")
            return 1
        config_dir = config_path.expanduser().resolve().parent
        with open(config_path, "r", encoding="utf-8") as handle:
            config_data = yaml.safe_load(handle) or {}
        if not isinstance(config_data, dict):
            raise RuntimeError("批量配置文件必须是 YAML mapping")
        batch_config = dict(config_data.get("batch") or {})
        legacy_keys = [key for key in ("method_name", "method_args") if key in config_data]
        if legacy_keys:
            raise RuntimeError(
                "Legacy batch config keys are no longer supported: "
                + ", ".join(legacy_keys)
                + ". Use entrypoint + runtime/inputs/preprocess/ss/slat + batch."
            )

    entrypoint_name = args.entrypoint or config_data.get("entrypoint")
    if not entrypoint_name:
        print("[ERROR] 必须提供 entrypoint（CLI 或 YAML 顶层 entrypoint）")
        return 1
    if not has_entrypoint(entrypoint_name):
        print(f"[ERROR] Unknown composable entrypoint: {entrypoint_name}")
        return 1

    gt_root = Path(args.gt_root or batch_config.get("gt_root") or DEFAULT_GT_ROOT).expanduser().resolve()
    config_name = args.config_name or batch_config.get("config_name") or entrypoint_name
    run_group = args.group or batch_config.get("group")
    dataset = args.dataset or batch_config.get("dataset")
    object_name = args.object or batch_config.get("object")
    prompt_id = args.prompt_id or batch_config.get("prompt_id")
    max_cases = args.max_cases if args.max_cases is not None else batch_config.get("max_cases")
    benchmark_root_text = args.benchmark_root or batch_config.get("benchmark_root")
    benchmark_root = Path(benchmark_root_text).expanduser().resolve() if benchmark_root_text else DEFAULT_BENCHMARK_ROOT
    assets_root_text = args.assets_root or batch_config.get("assets_root")
    assets_root = Path(assets_root_text).expanduser().resolve() if assets_root_text else None
    pred_root_text = args.pred_root or batch_config.get("pred_root")
    pred_root = Path(pred_root_text).expanduser().resolve() if pred_root_text else (
        benchmark_root / DEFAULT_PRED_DIRNAME / f"{entrypoint_name}_{config_name}"
    ).resolve()
    metrics = args.metrics or batch_config.get("metrics") or DEFAULT_METRICS
    device = args.device or config_data.get("runtime", {}).get("device") or batch_config.get("device") or "cuda:0"
    gpu_ids = parse_gpu_list(args.gpus or batch_config.get("gpus"), fallback_device=device)
    skip_benchmark_render = args.skip_render or bool(batch_config.get("skip_benchmark_render", False))
    skip_exists = args.skip_exists
    if skip_exists is None:
        skip_exists = bool(batch_config.get("skip_exists", True))
    keep_case_outputs = args.keep_case_outputs
    if keep_case_outputs is None:
        keep_case_outputs = bool(batch_config.get("keep_case_outputs", False))
    selected_cases = resolve_selected_cases(
        inline_cases=batch_config.get("selected_cases"),
        case_file_value=batch_config.get("selected_cases_file"),
        config_dir=config_dir,
    )

    if not gt_root.exists():
        print(f"[ERROR] GT 数据根目录不存在: {gt_root}")
        return 1

    runtime_defaults = dict(config_data.get("runtime") or {})
    model_override = args.model or runtime_defaults.get("model")
    seed_override = args.seed if args.seed is not None else runtime_defaults.get("seed")
    if seed_override is not None:
        seed_override = int(seed_override)

    print("=" * 80)
    print("批量运行 composable 编辑实验并评测")
    print("=" * 80)
    print(f"Entrypoint: {entrypoint_name}")
    print(f"配置: {config_name}")
    if run_group:
        print(f"运行分组: {run_group}")
    print(f"GT 数据: {gt_root}")
    print(f"预测输出: {pred_root}")
    print(f"benchmark 根目录: {benchmark_root}")
    if assets_root is not None:
        print(f"源资产: {assets_root}")
    if dataset:
        print(f"数据集过滤: {dataset}")
    if object_name:
        print(f"物体过滤: {object_name}")
    if prompt_id:
        print(f"提示 ID 过滤: {prompt_id}")
    if max_cases:
        print(f"案例限制: {max_cases}")
    print(f"评测指标: {metrics}")
    print(f"编辑 GPU 池: {', '.join(f'cuda:{gpu_id}' for gpu_id in gpu_ids)}")
    print(f"评测 GPU: {device}")
    print("=" * 80)

    metadata = load_edit3d_metadata(gt_root)
    if selected_cases:
        known_cases = available_prompt_cases(metadata)
        missing_cases = [case for case in selected_cases if case not in known_cases]
        if missing_cases:
            formatted = ", ".join(
                f"{dataset}/{object_name}/prompt_{prompt_id}"
                for dataset, object_name, prompt_id in missing_cases
            )
            raise RuntimeError(f"Unknown or unavailable selected cases: {formatted}")
        cases = list(selected_cases)
        if dataset:
            cases = [case for case in cases if case[0] == dataset]
        if object_name:
            cases = [case for case in cases if case[1] == object_name]
        if prompt_id:
            cases = [case for case in cases if case[2] == prompt_id]
        print(f"显式案例集: {len(selected_cases)} 个")
    else:
        cases = []
        for entry in metadata:
            entry_dataset = entry.get("dataset")
            entry_object_name = entry.get("source_model")
            if not entry_dataset or not entry_object_name:
                continue
            if dataset and entry_dataset != dataset:
                continue
            if object_name and entry_object_name != object_name:
                continue
            for candidate_prompt_id in (1, 2, 3):
                if prompt_id and candidate_prompt_id != prompt_id:
                    continue
                prompt_key = f"prompt_{candidate_prompt_id}"
                if entry.get(prompt_key):
                    cases.append((entry_dataset, entry_object_name, candidate_prompt_id))

    if max_cases:
        cases = cases[: int(max_cases)]
    print(f"[INFO] 找到 {len(cases)} 个案例")
    if not cases:
        print("[ERROR] 没有匹配的案例")
        return 1

    ensure_dir(pred_root)
    (pred_root / "_batch_selected_cases.json").write_text(
        json.dumps(
            [
                {
                    "dataset": dataset_name,
                    "object_name": object_name_text,
                    "prompt_id": prompt_id_value,
                    "case_id": f"{dataset_name}/{object_name_text}/prompt_{prompt_id_value}",
                }
                for dataset_name, object_name_text, prompt_id_value in cases
            ],
            indent=2,
            ensure_ascii=False,
        ),
        encoding="utf-8",
    )
    (pred_root / "_batch_source_config.yaml").write_text(
        yaml.safe_dump(config_data or {"entrypoint": entrypoint_name, "batch": batch_config}, sort_keys=False, allow_unicode=True),
        encoding="utf-8",
    )

    start_time = time.time()
    success, results = run_editing_and_eval(
        gt_root=gt_root,
        pred_root=pred_root,
        entrypoint_name=entrypoint_name,
        config_name=config_name,
        base_config=config_data or {"entrypoint": entrypoint_name},
        cases=cases,
        assets_root=assets_root,
        metrics=list(metrics),
        device=device,
        skip_benchmark_render=skip_benchmark_render,
        skip_exists=skip_exists,
        keep_case_outputs=keep_case_outputs,
        model_override=model_override,
        seed_override=seed_override,
        gpu_ids=gpu_ids,
        dry_run=args.dry_run,
    )
    if not success:
        print("[ERROR] 处理失败")
        return 1

    total_time = time.time() - start_time
    if results is not None:
        save_results(
            output_root=pred_root,
            entrypoint_name=entrypoint_name,
            config_name=config_name,
            run_group=run_group,
            gt_root=gt_root,
            cases=cases,
            requested_metrics=list(metrics),
            benchmark_root=benchmark_root,
            skip_benchmark_render=skip_benchmark_render,
            results=results,
            total_time=total_time,
        )

    print("\n" + "=" * 80)
    print("批量处理完成")
    print("=" * 80)
    print(f"总耗时: {format_time(total_time)}")
    print(f"输出目录: {pred_root}")
    print("=" * 80)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
