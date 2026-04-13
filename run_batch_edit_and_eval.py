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
from datetime import datetime
from pathlib import Path
from typing import Any

import yaml

from trellis_edit.common import build_experiment_output_layout, ensure_dir
from trellis_edit.composable import get_entrypoint, has_entrypoint


DEFAULT_GT_ROOT = Path("/home/wangxinxing/code/Edit3Dpp/data")
DEFAULT_PRED_ROOT = Path("/cache/wangxinxing/data/temp")
DEFAULT_METRICS = ["psnr", "ssim", "lpips", "fid", "dino_if", "chamfer", "clip_t"]
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


def load_edit3d_metadata(gt_root: Path) -> list[dict[str, Any]]:
    metadata_path = gt_root / "metadata.json"
    with open(metadata_path, "r", encoding="utf-8") as handle:
        return json.load(handle)


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
        runtime["seed"] = seed_override
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

    layout = build_experiment_output_layout(entrypoint_name, case_name, outputs_root=work_root)
    expected_glb = layout.edit_dir / stage_output_dir_name(entrypoint_name) / "sample_00.glb"
    return config, case_name, expected_glb


def render_all_results(output_root: Path, device: str = "cuda:0") -> bool:
    render_dir = Path("VoxHammer/Edit3D-Bench")
    if not render_dir.exists():
        render_dir = Path("../VoxHammer/Edit3D-Bench")
    if not render_dir.exists():
        print("[ERROR] 找不到渲染脚本")
        return False

    env = os.environ.copy()
    if device.startswith("cuda:"):
        env["CUDA_VISIBLE_DEVICES"] = device.replace("cuda:", "")

    cmd = [sys.executable, "render.py", "--base_dir", str(output_root.resolve())]
    returncode, stdout, stderr = run_command(cmd, cwd=render_dir, env=env, timeout=7200)
    if stdout:
        print(stdout)
    if stderr:
        print(f"[STDERR] {stderr}")
    return returncode == 0


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
    results: dict[str, Any] | None,
    total_time: float,
) -> None:
    print("\n[4/4] 保存结果...")
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    results_dir = Path("outputs/results") / f"{entrypoint_name}_{config_name}_{timestamp}"
    ensure_dir(results_dir)

    full_results = {
        "entrypoint": entrypoint_name,
        "config_name": config_name,
        "timestamp": timestamp,
        "datetime": datetime.now().isoformat(),
        "total_time_seconds": total_time,
        "total_time_formatted": format_time(total_time),
        "output_root": str(output_root),
        "evaluation_results": results,
    }
    with open(results_dir / "evaluation_results.json", "w", encoding="utf-8") as handle:
        json.dump(full_results, handle, indent=2, ensure_ascii=False)

    report_file = results_dir / "report.txt"
    with open(report_file, "w", encoding="utf-8") as handle:
        handle.write("=" * 80 + "\n")
        handle.write("Edit3D-Bench 评测报告\n")
        handle.write("=" * 80 + "\n\n")
        handle.write(f"Entrypoint: {entrypoint_name}\n")
        handle.write(f"配置: {config_name}\n")
        handle.write(f"评测时间: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
        handle.write(f"总耗时: {format_time(total_time)}\n")
        handle.write(f"数据位置: {output_root}\n\n")

        if results and "results" in results:
            handle.write("=" * 80 + "\n")
            handle.write("评测指标结果\n")
            handle.write("=" * 80 + "\n\n")
            for metric, values in results["results"].items():
                handle.write(f"{metric.upper()}:\n")
                if isinstance(values, dict):
                    if values.get("mean") is not None:
                        handle.write(f"  均值: {values['mean']:.4f}\n")
                    if values.get("std") is not None:
                        handle.write(f"  标准差: {values['std']:.4f}\n")
                    if values.get("count") is not None:
                        handle.write(f"  样本数: {values['count']}\n")
                elif values is not None:
                    handle.write(f"  值: {values:.4f}\n")
                else:
                    handle.write("  值: N/A\n")
                handle.write("\n")

    print(f"[INFO] 结果已保存到: {results_dir}")
    print(f"[INFO] - 报告: {report_file}")
    print(f"[INFO] - 数据: {output_root}")


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

    case_output_dir = expected_glb.parents[2]
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
    if not keep_case_outputs and case_output_dir.exists():
        shutil.rmtree(case_output_dir)
        print(f"[CLEANUP] 已删除单 case 完整输出: {case_output_dir}")
    return True, False


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
    skip_benchmark_render: bool,
    skip_exists: bool,
    keep_case_outputs: bool,
    model_override: str | None,
    seed_override: int | None,
    dry_run: bool,
) -> tuple[bool, dict[str, Any] | None]:
    if not skip_exists and pred_root.exists():
        print(f"[INFO] 清理旧的输出目录: {pred_root}")
        shutil.rmtree(pred_root)
    ensure_dir(pred_root)

    print("\n" + "=" * 80)
    print(f"输出目录: {pred_root}")
    print(f"Entrypoint: {entrypoint_name}")
    print(f"配置: {config_name}")
    print(f"案例数量: {len(cases)}")
    print(f"跳过已存在: {'是' if skip_exists else '否'}")
    print(f"保留单 case 输出: {'是' if keep_case_outputs else '否'}")
    print("=" * 80)

    success_count = 0
    skip_count = 0
    print("\n" + "=" * 80)
    print("步骤 1/4: 批量运行 composable 编辑实验")
    print("=" * 80)

    for index, (dataset, object_name, prompt_id) in enumerate(cases, start=1):
        print(f"\n[{index}/{len(cases)}] 处理: {dataset}/{object_name}/prompt_{prompt_id}")
        success, skipped = run_single_edit(
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
            device_override=device,
            dry_run=dry_run,
        )
        if not success:
            continue
        success_count += 1
        if skipped:
            skip_count += 1
        if dry_run:
            print("[DRY-RUN] 仅展示首个 case 的解析配置，提前结束。")
            return True, None

    print(f"\n[INFO] 编辑完成: {success_count}/{len(cases)} 成功")
    if skip_count > 0:
        print(f"[INFO] 跳过已存在: {skip_count} 个")
    if success_count == 0:
        print("[ERROR] 没有成功的编辑结果")
        return False, None

    if skip_benchmark_render:
        print("\n[INFO] 跳过 Edit3D-Bench 渲染步骤")
    else:
        print("\n" + "=" * 80)
        print("步骤 2/4: 统一渲染所有结果")
        print("=" * 80)
        render_success = render_all_results(pred_root, device)
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
    parser.add_argument("--metrics", nargs="+", help="覆盖评测指标")
    parser.add_argument("--skip-render", action="store_true", help="跳过 Edit3D-Bench 渲染步骤")
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
    if args.config:
        config_path = Path(args.config)
        if not config_path.exists():
            print(f"[ERROR] 配置文件不存在: {config_path}")
            return 1
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
    dataset = args.dataset or batch_config.get("dataset")
    object_name = args.object or batch_config.get("object")
    prompt_id = args.prompt_id or batch_config.get("prompt_id")
    max_cases = args.max_cases if args.max_cases is not None else batch_config.get("max_cases")
    assets_root_text = args.assets_root or batch_config.get("assets_root")
    assets_root = Path(assets_root_text).expanduser().resolve() if assets_root_text else None
    pred_root_text = args.pred_root or batch_config.get("pred_root")
    pred_root = Path(pred_root_text).expanduser().resolve() if pred_root_text else (
        DEFAULT_PRED_ROOT / f"{entrypoint_name}_{config_name}"
    ).resolve()
    metrics = args.metrics or batch_config.get("metrics") or DEFAULT_METRICS
    device = args.device or config_data.get("runtime", {}).get("device") or batch_config.get("device") or "cuda:0"
    skip_benchmark_render = args.skip_render or bool(batch_config.get("skip_benchmark_render", False))
    skip_exists = args.skip_exists
    if skip_exists is None:
        skip_exists = bool(batch_config.get("skip_exists", True))
    keep_case_outputs = args.keep_case_outputs
    if keep_case_outputs is None:
        keep_case_outputs = bool(batch_config.get("keep_case_outputs", False))

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
    print(f"GT 数据: {gt_root}")
    print(f"预测输出: {pred_root}")
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
    print("=" * 80)

    metadata = load_edit3d_metadata(gt_root)
    cases: list[tuple[str, str, int]] = []
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
