from __future__ import annotations

import json
from pathlib import Path
from typing import Any

from .edit3d_bench import EvaluationConfig, EvaluationManager
from .edit3d_bench import render as render_module
from .edit3d_bench.eval_modules.geometry_metrics import GeometryMetricsEvaluator
from .edit3d_bench.eval_modules.image_metrics import ImageMetricsEvaluator
from .edit3d_bench.eval_modules.mask_utils import build_include_mask, load_prepared_rgb_image


def get_edit3d_bench_render_script() -> Path:
    return (Path(__file__).resolve().parent / "edit3d_bench" / "render.py").resolve()


def evaluate_single_view_predictions(
    *,
    gt_root: Path,
    pred_root: Path,
    metrics: list[str],
    output_dir: Path,
    device: str = "cuda:0",
    batch_size: int = 32,
    num_workers: int = 4,
    image_size: tuple[int, int] = (512, 512),
    ignore_mask: bool = False,
    save_detailed: bool = True,
) -> dict[str, Any]:
    config = EvaluationConfig(
        gt_root=str(gt_root),
        pred_root=str(pred_root),
        metrics=list(metrics),
        device=device,
        image_size=image_size,
        batch_size=batch_size,
        num_workers=num_workers,
        output_dir=str(output_dir),
        save_detailed=save_detailed,
        ignore_mask=ignore_mask,
    )
    evaluator = EvaluationManager(config)
    evaluator.run_evaluation()

    summary_path = output_dir / "summary.json"
    if not summary_path.is_file():
        raise RuntimeError(f"Single-view evaluation did not produce summary.json: {summary_path}")
    return json.loads(summary_path.read_text(encoding="utf-8"))
