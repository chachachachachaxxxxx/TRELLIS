#!/usr/bin/env python3
"""
批量运行编辑实验并评测

直接从 Edit3D-Bench 数据集读取测试案例，运行编辑方法，生成结果并评测。
结果直接保存到 /cache/wangxinxing/data/temp/{method_name}_{config_name}_{timestamp}/
"""

import os
import sys
import json
import time
import shutil
import argparse
import subprocess
import yaml
from pathlib import Path
from datetime import datetime
from typing import Dict, List, Optional


def load_edit3d_metadata(gt_root: Path) -> List[Dict]:
    """加载 Edit3D-Bench metadata"""
    metadata_path = gt_root / "metadata.json"
    with open(metadata_path, "r") as f:
        return json.load(f)


def run_command(cmd: List[str], cwd: Optional[Path] = None, timeout: int = 3600) -> tuple[int, str, str]:
    """运行命令"""
    print(f"[CMD] {' '.join(cmd)}")
    try:
        result = subprocess.run(cmd, cwd=cwd, capture_output=True, text=True, timeout=timeout)
        return result.returncode, result.stdout, result.stderr
    except subprocess.TimeoutExpired:
        return -1, "", f"Timeout after {timeout}s"
    except Exception as e:
        return -1, "", str(e)


def format_time(seconds: float) -> str:
    """格式化时间"""
    hours = int(seconds // 3600)
    minutes = int((seconds % 3600) // 60)
    secs = int(seconds % 60)
    if hours > 0:
        return f"{hours}h {minutes}m {secs}s"
    elif minutes > 0:
        return f"{minutes}m {secs}s"
    else:
        return f"{secs}s"


def run_editing_and_eval(
    gt_root: Path,
    method_name: str,
    config_name: str,
    cases: List[tuple],
    method_args: List[str],
    seed: int,
    device: str,
    metrics: List[str],
    skip_render: bool = False,
    skip_exists: bool = True,
    assets_root: Optional[Path] = None,
) -> tuple[bool, Optional[Dict]]:
    """批量运行编辑实验并评测"""

    # 使用固定的输出目录（不带时间戳）
    output_root = Path("/cache/wangxinxing/data/temp") / f"{method_name}_{config_name}"

    # 如果 skip_exists 为 False，清理旧数据
    if not skip_exists and output_root.exists():
        print(f"[INFO] 清理旧的输出目录: {output_root}")
        shutil.rmtree(output_root)

    output_root.mkdir(parents=True, exist_ok=True)

    print("\n" + "="*80)
    print(f"输出目录: {output_root}")
    print(f"方法: {method_name}")
    print(f"配置: {config_name}")
    print(f"案例数量: {len(cases)}")
    print(f"跳过已存在: {'是' if skip_exists else '否'}")
    print("="*80)

    # 步骤 1: 批量运行所有编辑实验
    print("\n" + "="*80)
    print("步骤 1/4: 批量运行编辑实验")
    print("="*80)

    success_count = 0
    skip_count = 0
    for i, (dataset, object_name, prompt_id) in enumerate(cases, 1):
        print(f"\n[{i}/{len(cases)}] 处理: {dataset}/{object_name}/prompt_{prompt_id}")

        success = run_single_edit(
            gt_root=gt_root,
            output_root=output_root,
            method_name=method_name,
            dataset=dataset,
            object_name=object_name,
            prompt_id=prompt_id,
            method_args=method_args,
            seed=seed,
            device=device,
            skip_exists=skip_exists,
            assets_root=assets_root,
        )

        if success:
            success_count += 1
            # 检查是否是跳过的（已存在）
            edit_glb = output_root / dataset / object_name / f"prompt_{prompt_id}" / "edit.glb"
            if skip_exists and edit_glb.exists():
                # 检查文件修改时间，如果是刚创建的则不算跳过
                import time
                if time.time() - edit_glb.stat().st_mtime > 60:  # 超过1分钟前创建的
                    skip_count += 1

    print(f"\n[INFO] 编辑完成: {success_count}/{len(cases)} 成功")
    if skip_count > 0:
        print(f"[INFO] 跳过已存在: {skip_count} 个")

    if success_count == 0:
        print("[ERROR] 没有成功的编辑结果")
        return False, None

    # 步骤 2: 统一渲染所有结果
    if not skip_render:
        print("\n" + "="*80)
        print("步骤 2/4: 统一渲染所有结果")
        print("="*80)

        render_success = render_all_results(output_root, device)
        if not render_success:
            print("[WARNING] 渲染失败")
    else:
        print("\n[INFO] 跳过渲染步骤")

    # 步骤 3: 统一评测
    print("\n" + "="*80)
    print("步骤 3/4: 运行评测")
    print("="*80)

    eval_output_dir = output_root / "evaluation_output"
    success, results = run_evaluation(
        gt_root=gt_root,
        pred_root=output_root,
        metrics=metrics,
        output_dir=eval_output_dir,
        device=device,
    )

    if not success:
        print("[WARNING] 评测失败")
        return True, None

    return True, results


def run_single_edit(
    gt_root: Path,
    output_root: Path,
    method_name: str,
    dataset: str,
    object_name: str,
    prompt_id: int,
    method_args: List[str],
    seed: int,
    device: str = "cuda:0",
    assets_root: Optional[Path] = None,
    skip_exists: bool = True,
) -> bool:
    """运行单个编辑实验"""

    # 构建输出路径
    dataset_dir = output_root / dataset
    object_dir = dataset_dir / object_name
    prompt_dir = object_dir / f"prompt_{prompt_id}"
    prompt_dir.mkdir(parents=True, exist_ok=True)

    # 检查是否已经生成过 edit.glb
    edit_glb = prompt_dir / "edit.glb"
    if skip_exists and edit_glb.exists():
        print(f"[SKIP] edit.glb 已存在: {edit_glb}")
        return True

    # 获取输入文件路径
    gt_object_dir = gt_root / dataset / object_name
    source_image = gt_object_dir / f"prompt_{prompt_id}" / "2d_render.png"
    edit_image = gt_object_dir / f"prompt_{prompt_id}" / "2d_edit.png"
    mask_image = gt_object_dir / f"prompt_{prompt_id}" / "2d_mask.png"
    mask_glb = gt_object_dir / f"prompt_{prompt_id}" / "3d_edit_region.glb"

    # 检查文件是否存在
    if not all([source_image.exists(), edit_image.exists(), mask_image.exists()]):
        print(f"[ERROR] 输入文件不存在")
        return False

    # 使用固定的临时案例名称
    temp_case_name = "temp_edit"

    # 构建命令
    cmd = [
        "python", "run_edit_experiment.py",
        "--method", method_name,
        "--case-name", temp_case_name,
        "--source-image", str(source_image),
        "--edit-image", str(edit_image),
        "--mask-image", str(mask_image),
        "--seed", str(seed),
    ]

    # 使用已有的 assets 或预处理
    if assets_root:
        asset_dir = assets_root / dataset / object_name
        if asset_dir.exists():
            cmd.extend(["--asset-dir", str(asset_dir)])
        else:
            print(f"[WARNING] Assets 不存在: {asset_dir}，将使用预处理")
            source_model = gt_object_dir / "source_model" / "model.glb"
            if source_model.exists():
                cmd.extend(["--source-model", str(source_model), "--preprocess"])
            else:
                print(f"[ERROR] 找不到 source model: {source_model}")
                return False
    else:
        source_model = gt_object_dir / "source_model" / "model.glb"
        if source_model.exists():
            cmd.extend(["--source-model", str(source_model), "--preprocess"])
        else:
            print(f"[ERROR] 找不到 source model: {source_model}")
            return False

    if mask_glb.exists():
        cmd.extend(["--mask-glb", str(mask_glb)])

    # 添加设备参数
    cmd.extend(["--device", device])

    # 添加方法参数
    cmd.extend(method_args)

    # 清理临时输出
    temp_output_dir = Path(f"outputs/{method_name}/{temp_case_name}")
    if temp_output_dir.exists():
        shutil.rmtree(temp_output_dir)

    returncode, stdout, stderr = run_command(cmd, timeout=7200)

    if returncode != 0:
        print(f"[ERROR] 编辑实验失败")
        return False

    # 找到生成的 GLB
    temp_glb = Path(f"outputs/{method_name}/{temp_case_name}/edit/sample_00.glb")
    if not temp_glb.exists():
        print(f"[ERROR] 找不到生成的 GLB: {temp_glb}")
        return False

    # 复制到评测目录
    edit_glb = prompt_dir / "edit.glb"
    shutil.copy2(temp_glb, edit_glb)
    print(f"[SUCCESS] GLB 已保存: {edit_glb}")

    # 清理临时输出
    if temp_output_dir.exists():
        shutil.rmtree(temp_output_dir)

    return True


def render_all_results(output_root: Path, device: str = "cuda:0") -> bool:
    """统一渲染所有结果"""

    render_dir = Path("VoxHammer/Edit3D-Bench")
    if not render_dir.exists():
        render_dir = Path("../VoxHammer/Edit3D-Bench")

    if not render_dir.exists():
        print("[ERROR] 找不到渲染脚本")
        return False

    env = os.environ.copy()
    env["CUDA_VISIBLE_DEVICES"] = device.replace("cuda:", "")

    cmd = ["python", "render.py", "--base_dir", str(output_root.resolve())]
    returncode, stdout, stderr = run_command(cmd, cwd=render_dir, timeout=7200)

    print(stdout)
    if stderr:
        print(f"[STDERR] {stderr}")

    return returncode == 0


def run_evaluation(
    gt_root: Path,
    pred_root: Path,
    metrics: List[str],
    output_dir: Path,
    device: str = "cuda:0",
) -> tuple[bool, Optional[Dict]]:
    """运行评测"""

    eval_script = Path("VoxHammer/Edit3D-Bench/eval_main.py")
    if not eval_script.exists():
        eval_script = Path("../VoxHammer/Edit3D-Bench/eval_main.py")

    if not eval_script.exists():
        print("[ERROR] 找不到评测脚本")
        return False, None

    cmd = [
        "python", str(eval_script),
        "--gt_root", str(gt_root),
        "--pred_root", str(pred_root),
        "--metrics", *metrics,
        "--device", device,
        "--batch_size", "32",
        "--output_dir", str(output_dir),
    ]

    returncode, stdout, stderr = run_command(cmd, timeout=7200)

    print(stdout)
    if stderr:
        print(f"[STDERR] {stderr}")

    if returncode != 0:
        return False, None

    summary_file = output_dir / "summary.json"
    if summary_file.exists():
        with open(summary_file, "r") as f:
            results = json.load(f)
        return True, results

    return False, None


def save_results(
    output_root: Path,
    method_name: str,
    config_name: str,
    results: Optional[Dict],
    total_time: float,
):
    """保存评测结果"""
    print("\n[4/4] 保存结果...")

    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    results_dir = Path("outputs/results") / f"{method_name}_{config_name}_{timestamp}"
    results_dir.mkdir(parents=True, exist_ok=True)

    # 保存完整结果
    full_results = {
        "method_name": method_name,
        "config_name": config_name,
        "timestamp": timestamp,
        "datetime": datetime.now().isoformat(),
        "total_time_seconds": total_time,
        "total_time_formatted": format_time(total_time),
        "output_root": str(output_root),
        "evaluation_results": results,
    }

    results_file = results_dir / "evaluation_results.json"
    with open(results_file, "w") as f:
        json.dump(full_results, f, indent=2, ensure_ascii=False)

    # 生成可读报告
    report_file = results_dir / "report.txt"
    with open(report_file, "w") as f:
        f.write("="*80 + "\n")
        f.write("Edit3D-Bench 评测报告\n")
        f.write("="*80 + "\n\n")

        f.write(f"方法: {method_name}\n")
        f.write(f"配置: {config_name}\n")
        f.write(f"评测时间: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
        f.write(f"总耗时: {format_time(total_time)}\n")
        f.write(f"数据位置: {output_root}\n\n")

        if results and "results" in results:
            f.write("="*80 + "\n")
            f.write("评测指标结果\n")
            f.write("="*80 + "\n\n")

            for metric, values in results["results"].items():
                f.write(f"{metric.upper()}:\n")
                if isinstance(values, dict):
                    if "mean" in values and values["mean"] is not None:
                        f.write(f"  均值: {values['mean']:.4f}\n")
                    if "std" in values and values["std"] is not None:
                        f.write(f"  标准差: {values['std']:.4f}\n")
                    if "count" in values and values["count"] is not None:
                        f.write(f"  样本数: {values['count']}\n")
                elif values is not None:
                    f.write(f"  值: {values:.4f}\n")
                else:
                    f.write(f"  值: N/A\n")
                f.write("\n")

    print(f"[INFO] 结果已保存到: {results_dir}")
    print(f"[INFO] - 报告: {report_file}")
    print(f"[INFO] - 数据: {output_root}")


def main():
    parser = argparse.ArgumentParser(
        description="批量运行编辑实验并评测",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )

    # 配置文件或命令行参数
    parser.add_argument("--config", type=str,
                        help="YAML 配置文件路径（推荐）")

    # 必需参数（如果不使用配置文件）
    parser.add_argument("--gt-root", type=str,
                        help="Edit3D-Bench GT 数据根目录")
    parser.add_argument("--method-name", type=str,
                        help="方法名称")
    parser.add_argument("--config-name", type=str,
                        help="配置名称（用于标识不同的参数配置）")

    # 数据过滤
    parser.add_argument("--dataset", type=str,
                        help="数据集过滤（例如：GSO）")
    parser.add_argument("--object", type=str,
                        help="物体名称过滤")
    parser.add_argument("--prompt-id", type=int, choices=[1, 2, 3],
                        help="提示ID过滤")
    parser.add_argument("--max-cases", type=int,
                        help="限制处理的案例数量")

    # 方法参数
    parser.add_argument("--seed", type=int, default=1,
                        help="随机种子")
    parser.add_argument("--assets-root", type=str,
                        help="预处理的 assets 根目录（如 /cache/wangxinxing/data/temp/renders）")
    parser.add_argument("--method-args", nargs=argparse.REMAINDER,
                        help="传递给编辑方法的参数")

    # 评测参数
    parser.add_argument("--metrics", nargs="+",
                        default=["psnr", "ssim", "lpips", "fid", "dino_if", "chamfer", "clip_t"],
                        help="评测指标")
    parser.add_argument("--device", type=str, default="cuda:0",
                        help="计算设备")
    parser.add_argument("--skip-render", action="store_true",
                        help="跳过渲染")
    parser.add_argument("--skip-exists", action="store_true", default=True,
                        help="跳过已存在的 edit.glb 文件（默认启用）")
    parser.add_argument("--no-skip-exists", dest="skip_exists", action="store_false",
                        help="不跳过已存在的文件，重新生成所有结果")

    args = parser.parse_args()

    # 如果提供了配置文件，从配置文件加载参数
    if args.config:
        config_path = Path(args.config)
        if not config_path.exists():
            print(f"[ERROR] 配置文件不存在: {config_path}")
            return 1

        with open(config_path, "r") as f:
            config = yaml.safe_load(f)

        # 从配置文件读取参数（命令行参数优先）
        gt_root = Path(config.get("gt_root", "/home/wangxinxing/code/Edit3Dpp/data"))
        method_name = config.get("method_name")
        config_name = config.get("config_name")
        dataset = config.get("dataset")
        object_name = config.get("object")
        prompt_id = config.get("prompt_id")
        max_cases = config.get("max_cases")
        seed = config.get("seed", 1)
        assets_root = Path(config["assets_root"]) if config.get("assets_root") else None
        metrics = config.get("metrics", ["psnr", "ssim", "lpips", "fid", "dino_if", "chamfer", "clip_t"])
        device = config.get("device", "cuda:0")
        skip_render = config.get("skip_render", False)
        skip_exists = config.get("skip_exists", True)

        # 转换 method_args 从字典到命令行参数列表
        method_args = []
        if "method_args" in config:
            for key, value in config["method_args"].items():
                method_args.append(f"--{key}")
                # 布尔值 True 转换为标志参数（不带值）
                # None 或 False 跳过
                # 其他值（包括空字符串 ""）都添加
                if value is True:
                    continue  # 标志参数，不添加值
                elif value is not None and value is not False:
                    method_args.append(str(value))

        # 命令行参数覆盖配置文件
        if args.gt_root:
            gt_root = Path(args.gt_root)
        if args.method_name:
            method_name = args.method_name
        if args.config_name:
            config_name = args.config_name
        if args.dataset:
            dataset = args.dataset
        if args.object:
            object_name = args.object
        if args.prompt_id:
            prompt_id = args.prompt_id
        if args.max_cases:
            max_cases = args.max_cases
        if args.assets_root:
            assets_root = Path(args.assets_root)
        if args.method_args:
            method_args = args.method_args

    else:
        # 使用命令行参数
        if not all([args.gt_root, args.method_name, args.config_name]):
            print("[ERROR] 必须提供 --config 或 (--gt-root, --method-name, --config-name)")
            return 1

        gt_root = Path(args.gt_root)
        method_name = args.method_name
        config_name = args.config_name
        dataset = args.dataset
        object_name = args.object
        prompt_id = args.prompt_id
        max_cases = args.max_cases
        seed = args.seed
        assets_root = Path(args.assets_root) if args.assets_root else None
        metrics = args.metrics
        device = args.device
        skip_render = args.skip_render
        skip_exists = args.skip_exists
        method_args = args.method_args or []

    print("="*80)
    print("批量运行编辑实验并评测")
    print("="*80)
    print(f"方法: {method_name}")
    print(f"配置: {config_name}")
    print(f"GT 数据: {gt_root}")
    if dataset:
        print(f"数据集过滤: {dataset}")
    if object_name:
        print(f"物体过滤: {object_name}")
    if prompt_id:
        print(f"提示ID过滤: {prompt_id}")
    if max_cases:
        print(f"案例限制: {max_cases}")
    print(f"方法参数: {method_args or '(默认)'}")
    print("="*80)

    # 加载 metadata
    print(f"\n[INFO] 加载 metadata...")
    metadata = load_edit3d_metadata(gt_root)

    # 过滤案例
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

        # 处理三个 prompt
        for pid in [1, 2, 3]:
            if prompt_id and pid != prompt_id:
                continue

            prompt_key = f"prompt_{pid}"
            if prompt_key in entry and entry[prompt_key]:
                cases.append((entry_dataset, entry_object_name, pid))

    if max_cases:
        cases = cases[:max_cases]

    print(f"[INFO] 找到 {len(cases)} 个案例")

    # 处理案例
    start_time = time.time()

    success, results = run_editing_and_eval(
        gt_root=gt_root,
        method_name=method_name,
        config_name=config_name,
        cases=cases,
        method_args=method_args,
        seed=seed,
        device=device,
        metrics=metrics,
        skip_render=skip_render,
        assets_root=assets_root,
    )

    if not success:
        print("[ERROR] 处理失败")
        return 1

    total_time = time.time() - start_time

    # 保存结果
    output_root = Path("/cache/wangxinxing/data/temp") / f"{method_name}_{config_name}"

    if results:
        save_results(
            output_root=output_root,
            method_name=method_name,
            config_name=config_name,
            results=results,
            total_time=total_time,
        )

    # 打印总结
    print("\n" + "="*80)
    print("批量处理完成")
    print("="*80)
    print(f"总耗时: {format_time(total_time)}")
    print(f"输出目录: {output_root}")
    print("="*80)

    return 0


if __name__ == "__main__":
    sys.exit(main())
