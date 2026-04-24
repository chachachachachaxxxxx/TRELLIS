#!/usr/bin/env python3
"""
Batch-generate TRELLIS multi-view baselines from benchmark target views.

This script reads benchmark cases under:
    <dataset_root>/cases/<case_id>/

For each case, it collects the multi-view conditioning images from
`case.json -> input_views[*].target_path`, runs TRELLIS multi-image inference,
and writes:
    <output_root>/<run_name>/cases/<case_id>/edit.glb

The script supports both TRELLIS multi-image modes:
    - stochastic
    - multidiffusion
    - view_aligned_stochastic
    - canonical_weighted_stochastic
"""

from __future__ import annotations

import argparse
from collections import deque
from dataclasses import dataclass
from datetime import datetime
import json
import os
from pathlib import Path
import subprocess
import sys
import time
import traceback
from typing import Any, Deque, Dict, List, Optional, Sequence, Tuple


REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO_ROOT))

from trellis_edit.common.backend_config import build_backend_config


DEFAULT_DATASET_ROOT = Path(
    "/cache/wangxinxing/data/trellis_edit_benchmark/edit3d_mv_pseudosource_micro10"
)
DEFAULT_OUTPUT_ROOT = Path(
    "/cache/wangxinxing/data/trellis_edit_benchmark/pred_mv"
)
ENTRYPOINT_NAME = "trellis_mv_batch_generate_from_targets"


@dataclass(frozen=True)
class ViewSpec:
    key: str
    label: str
    azimuth: float
    elevation: float
    relative_path: str
    absolute_path: Path


@dataclass(frozen=True)
class CaseSpec:
    case_id: str
    case_dir: Path
    case_json_path: Path
    dataset: str
    source_model_name: str
    prompt_id: Optional[int]
    prompt_text: str
    edit_instruction: str
    view_kind: str
    views: Sequence[ViewSpec]


def now_iso() -> str:
    return datetime.now().astimezone().isoformat()


def read_json(path: Path) -> Any:
    with path.open("r", encoding="utf-8") as handle:
        return json.load(handle)


def write_json(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as handle:
        json.dump(payload, handle, indent=2, ensure_ascii=False)
        handle.write("\n")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Run TRELLIS multi-view baselines from benchmark target views"
    )
    parser.add_argument(
        "--dataset-root",
        type=Path,
        default=DEFAULT_DATASET_ROOT,
        help="Benchmark dataset root containing cases/",
    )
    parser.add_argument(
        "--output-root",
        type=Path,
        default=DEFAULT_OUTPUT_ROOT,
        help="Root directory that will contain <run_name>/",
    )
    parser.add_argument(
        "--run-name",
        type=str,
        default="",
        help="Run directory name under output-root",
    )
    parser.add_argument(
        "--mode",
        choices=(
            "stochastic",
            "multidiffusion",
            "view_aligned_stochastic",
            "canonical_weighted_stochastic",
        ),
        default="stochastic",
        help="TRELLIS multi-image conditioning mode",
    )
    parser.add_argument(
        "--view-kind",
        choices=("target", "source"),
        default="target",
        help="Which input_views path field to use from case.json",
    )
    parser.add_argument(
        "--view-order",
        choices=(
            "azimuth",
            "reverse_azimuth",
            "front_first_clockwise",
            "front_first_counterclockwise",
            "custom",
        ),
        default="azimuth",
        help="How to order multi-view conditioning images before TRELLIS inference",
    )
    parser.add_argument(
        "--custom-view-order",
        type=str,
        default="",
        help="Comma-separated input_views keys used when --view-order=custom",
    )
    parser.add_argument(
        "--view-keys",
        type=str,
        default="",
        help="Optional comma-separated subset of input_views keys to use",
    )
    parser.add_argument(
        "--conditioning-sequence",
        type=str,
        default="",
        help=(
            "Optional comma-separated ordered input_views keys used for conditioning. "
            "Unlike --view-keys, this sequence may repeat keys (for example: front,front,back). "
            "When set, it overrides --view-order/--custom-view-order/--view-keys."
        ),
    )
    parser.add_argument(
        "--model-name",
        type=str,
        default="microsoft/TRELLIS-image-large",
        help="HF model repo or local model path",
    )
    parser.add_argument("--seed", type=int, default=1, help="Random seed")
    parser.add_argument(
        "--ss-steps",
        type=int,
        default=12,
        help="Sparse-structure sampler steps",
    )
    parser.add_argument(
        "--ss-cfg-strength",
        type=float,
        default=7.5,
        help="Sparse-structure sampler cfg strength",
    )
    parser.add_argument(
        "--slat-steps",
        type=int,
        default=12,
        help="SLAT sampler steps",
    )
    parser.add_argument(
        "--slat-cfg-strength",
        type=float,
        default=3.0,
        help="SLAT sampler cfg strength",
    )
    parser.add_argument(
        "--canonical-weight-floor",
        type=float,
        default=0.15,
        help="Minimum spatial weight used by canonical_weighted_stochastic",
    )
    parser.add_argument(
        "--canonical-weight-ss-nonreference-scale",
        type=float,
        default=0.25,
        help="Redistribution strength for non-reference views in sparse-structure stage",
    )
    parser.add_argument(
        "--canonical-weight-slat-nonreference-scale",
        type=float,
        default=0.60,
        help="Redistribution strength for non-reference views in SLAT stage",
    )
    parser.add_argument(
        "--canonical-weight-ss-sharpness",
        type=float,
        default=6.0,
        help="Spatial mask sharpness for sparse-structure stage",
    )
    parser.add_argument(
        "--canonical-weight-slat-sharpness",
        type=float,
        default=10.0,
        help="Spatial mask sharpness for SLAT stage",
    )
    parser.add_argument(
        "--texture-size",
        type=int,
        default=1024,
        help="Texture size for GLB export",
    )
    parser.add_argument(
        "--simplify",
        type=float,
        default=0.95,
        help="Mesh simplification ratio for GLB export",
    )
    parser.add_argument(
        "--parallel",
        action="store_true",
        help="Run one worker per GPU with queue-based scheduling",
    )
    parser.add_argument(
        "--gpus",
        type=str,
        default="0",
        help="Comma-separated GPU ids used when --parallel is set",
    )
    parser.add_argument(
        "--case-id",
        action="append",
        default=[],
        help="Restrict to one or more specific case ids",
    )
    parser.add_argument(
        "--limit",
        type=int,
        default=None,
        help="Optional case limit after sorting",
    )
    parser.add_argument(
        "--overwrite",
        action="store_true",
        help="Regenerate cases even when edit.glb already exists",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Only validate case discovery and write manifest",
    )
    parser.add_argument(
        "--save-videos",
        action="store_true",
        help="Also render gaussian/mesh preview videos into cases/<id>/videos/",
    )
    parser.add_argument(
        "--save-gaussian-ply",
        action="store_true",
        help="Also save gaussian point cloud as cases/<id>/gaussian.ply",
    )
    parser.add_argument(
        "--fail-fast",
        action="store_true",
        help="Stop immediately on the first failed case in sequential mode",
    )
    parser.add_argument(
        "--attn-backend",
        type=str,
        default=os.environ.get("ATTN_BACKEND", ""),
        help="Attention backend to export before importing TRELLIS",
    )
    parser.add_argument(
        "--sparse-attn-backend",
        type=str,
        default=os.environ.get("SPARSE_ATTN_BACKEND", ""),
        help="Sparse attention backend to export before importing TRELLIS",
    )
    parser.add_argument(
        "--spconv-algo",
        type=str,
        default=os.environ.get("SPCONV_ALGO", "native"),
        help="SPCONV_ALGO value to export before importing TRELLIS",
    )

    # Internal worker mode used by the queue-based launcher.
    parser.add_argument(
        "--worker",
        action="store_true",
        help=argparse.SUPPRESS,
    )
    return parser.parse_args()


def resolve_run_name(args: argparse.Namespace) -> str:
    if args.run_name:
        return args.run_name
    return "trellis_mv_{view_kind}_{mode}_{view_order}_seed{seed}".format(
        view_kind=args.view_kind,
        mode=args.mode,
        view_order=args.view_order,
        seed=args.seed,
    )


def sort_input_view_items(
    input_views: Dict[str, Dict[str, Any]],
    view_order: str,
    custom_view_order: str,
) -> List[Tuple[str, Dict[str, Any]]]:
    items = list(input_views.items())
    azimuth_sorted = sorted(
        items,
        key=lambda item: (
            float(item[1].get("azimuth", 0.0)),
            float(item[1].get("elevation", 0.0)),
            item[0],
        ),
    )
    if view_order == "azimuth":
        return azimuth_sorted
    if view_order == "reverse_azimuth":
        return list(reversed(azimuth_sorted))
    if view_order == "custom":
        requested_keys = [key.strip() for key in custom_view_order.split(",") if key.strip()]
        if not requested_keys:
            raise ValueError("--custom-view-order is required when --view-order=custom")
        if len(requested_keys) != len(items):
            raise ValueError(
                "Custom view order must list exactly {0} keys, got {1}".format(
                    len(items),
                    len(requested_keys),
                )
            )
        missing = sorted(set(input_views) - set(requested_keys))
        extras = sorted(set(requested_keys) - set(input_views))
        duplicates = sorted({key for key in requested_keys if requested_keys.count(key) > 1})
        if missing or extras or duplicates:
            raise ValueError(
                "Invalid custom view order. missing={0}, extras={1}, duplicates={2}".format(
                    missing,
                    extras,
                    duplicates,
                )
            )
        return [(key, input_views[key]) for key in requested_keys]

    label_priority_map = {
        "front_first_clockwise": {
            "front": 0,
            "right": 1,
            "back": 2,
            "left": 3,
        },
        "front_first_counterclockwise": {
            "front": 0,
            "left": 1,
            "back": 2,
            "right": 3,
        },
    }
    priority_map = label_priority_map.get(view_order)
    if priority_map is None:
        raise ValueError("Unsupported view order: {0}".format(view_order))

    return sorted(
        items,
        key=lambda item: (
            priority_map.get(str(item[1].get("label") or "").lower(), len(priority_map)),
            float(item[1].get("azimuth", 0.0)),
            float(item[1].get("elevation", 0.0)),
            item[0],
        ),
    )


def filter_input_views(
    input_views: Dict[str, Dict[str, Any]],
    view_keys: str,
) -> Dict[str, Dict[str, Any]]:
    requested_keys = [key.strip() for key in view_keys.split(",") if key.strip()]
    if not requested_keys:
        return dict(input_views)

    available_keys = set(input_views)
    missing = [key for key in requested_keys if key not in available_keys]
    duplicates = sorted({key for key in requested_keys if requested_keys.count(key) > 1})
    if missing or duplicates:
        raise ValueError(
            "Invalid --view-keys. missing={0}, duplicates={1}".format(
                missing,
                duplicates,
            )
        )
    return {key: input_views[key] for key in requested_keys}


def build_conditioning_view_items(
    input_views: Dict[str, Dict[str, Any]],
    *,
    view_order: str,
    custom_view_order: str,
    view_keys: str,
    conditioning_sequence: str,
) -> List[Tuple[str, Dict[str, Any]]]:
    requested_sequence = [key.strip() for key in conditioning_sequence.split(",") if key.strip()]
    if requested_sequence:
        available_keys = set(input_views)
        missing = [key for key in requested_sequence if key not in available_keys]
        if missing:
            raise ValueError(
                "Invalid --conditioning-sequence. missing={0}".format(sorted(set(missing)))
            )
        return [(key, input_views[key]) for key in requested_sequence]

    filtered_views = filter_input_views(input_views, view_keys)
    return sort_input_view_items(
        input_views=filtered_views,
        view_order=view_order,
        custom_view_order=custom_view_order,
    )


def build_case_specs(
    dataset_root: Path,
    view_kind: str,
    view_order: str,
    custom_view_order: str,
    view_keys: str,
    conditioning_sequence: str,
    requested_case_ids: Sequence[str],
    limit: Optional[int],
) -> List[CaseSpec]:
    cases_root = dataset_root / "cases"
    if not cases_root.is_dir():
        raise FileNotFoundError("Missing cases directory: {0}".format(cases_root))

    requested = set(requested_case_ids)
    specs: List[CaseSpec] = []

    for case_dir in sorted(path for path in cases_root.iterdir() if path.is_dir()):
        case_json_path = case_dir / "case.json"
        if not case_json_path.is_file():
            continue

        payload = read_json(case_json_path)
        case_id = str(payload.get("case_id") or case_dir.name)
        if requested and case_id not in requested:
            continue

        input_views = payload.get("input_views") or {}
        view_field = "{0}_path".format(view_kind)
        ordered_views: List[ViewSpec] = []
        for view_key, view_payload in build_conditioning_view_items(
            input_views=input_views,
            view_order=view_order,
            custom_view_order=custom_view_order,
            view_keys=view_keys,
            conditioning_sequence=conditioning_sequence,
        ):
            relative_path = view_payload.get(view_field)
            if not relative_path:
                raise ValueError(
                    "Case {0} is missing input_views[{1}].{2}".format(
                        case_id,
                        view_key,
                        view_field,
                    )
                )
            absolute_path = (case_dir / relative_path).resolve()
            if not absolute_path.is_file():
                raise FileNotFoundError(
                    "Case {0} view {1} does not exist: {2}".format(
                        case_id,
                        view_key,
                        absolute_path,
                    )
                )
            ordered_views.append(
                ViewSpec(
                    key=view_key,
                    label=str(view_payload.get("label") or view_key),
                    azimuth=float(view_payload.get("azimuth", 0.0)),
                    elevation=float(view_payload.get("elevation", 0.0)),
                    relative_path=str(relative_path),
                    absolute_path=absolute_path,
                )
            )

        if not ordered_views:
            raise ValueError("Case {0} has no usable input views".format(case_id))

        specs.append(
            CaseSpec(
                case_id=case_id,
                case_dir=case_dir.resolve(),
                case_json_path=case_json_path.resolve(),
                dataset=str(payload.get("dataset") or ""),
                source_model_name=str(payload.get("source_model_name") or ""),
                prompt_id=payload.get("prompt_id"),
                prompt_text=str(payload.get("prompt_text") or ""),
                edit_instruction=str(payload.get("edit_instruction") or ""),
                view_kind=view_kind,
                views=tuple(ordered_views),
            )
        )

    if requested:
        found_ids = {spec.case_id for spec in specs}
        missing = sorted(requested - found_ids)
        if missing:
            raise ValueError(
                "Requested case ids were not found under {0}: {1}".format(
                    cases_root,
                    ", ".join(missing),
                )
            )

    if limit is not None:
        specs = specs[:limit]
    return specs


def case_output_dir(run_dir: Path, case_id: str) -> Path:
    return run_dir / "cases" / case_id


def case_output_glb(run_dir: Path, case_id: str) -> Path:
    return case_output_dir(run_dir, case_id) / "edit.glb"


def build_manifest(
    *,
    args: argparse.Namespace,
    run_dir: Path,
    cases: Sequence[CaseSpec],
    case_statuses: Dict[str, Dict[str, Any]],
    started_at: str,
    completed_at: Optional[str] = None,
) -> Dict[str, Any]:
    total_time_seconds: float | None = None
    if completed_at:
        total_time_seconds = (
            datetime.fromisoformat(completed_at) - datetime.fromisoformat(started_at)
        ).total_seconds()

    counts: Dict[str, int] = {}
    for status_info in case_statuses.values():
        status = str(status_info.get("status", "pending"))
        counts[status] = counts.get(status, 0) + 1

    case_entries: List[Dict[str, Any]] = []
    for case in cases:
        status_info = case_statuses.get(case.case_id, {})
        case_entries.append(
            {
                "case_id": case.case_id,
                "dataset": case.dataset,
                "source_model_name": case.source_model_name,
                "prompt_id": case.prompt_id,
                "prompt_text": case.prompt_text,
                "edit_instruction": case.edit_instruction,
                "case_dir": str(case.case_dir),
                "case_json_path": str(case.case_json_path),
                "view_kind": case.view_kind,
                "views": [
                    {
                        "key": view.key,
                        "label": view.label,
                        "azimuth": view.azimuth,
                        "elevation": view.elevation,
                        "relative_path": view.relative_path,
                        "absolute_path": str(view.absolute_path),
                    }
                    for view in case.views
                ],
                "status": status_info.get("status", "pending"),
                "output_glb": str(case_output_glb(run_dir, case.case_id)),
                "output_relpath": "cases/{0}/edit.glb".format(case.case_id),
                "error": status_info.get("error"),
                "gpu_id": status_info.get("gpu_id"),
            }
        )

    return {
        "entrypoint": ENTRYPOINT_NAME,
        "config_name": run_dir.name,
        "group": "mv",
        "created_at": started_at,
        "completed_at": completed_at,
        "total_time_seconds": total_time_seconds,
        "dataset_root": str(args.dataset_root.resolve()),
        "output_root": str(args.output_root.resolve()),
        "run_dir": str(run_dir.resolve()),
        "mode": args.mode,
        "view_kind": args.view_kind,
        "view_order": args.view_order,
        "custom_view_order": args.custom_view_order,
        "view_keys": args.view_keys,
        "conditioning_sequence": args.conditioning_sequence,
        "model_name": args.model_name,
        "seed": args.seed,
        "sampler_params": {
            "sparse_structure_sampler": {
                "steps": args.ss_steps,
                "cfg_strength": args.ss_cfg_strength,
            },
            "slat_sampler": {
                "steps": args.slat_steps,
                "cfg_strength": args.slat_cfg_strength,
            },
        },
        "canonical_view_weighting": {
            "floor": args.canonical_weight_floor,
            "ss_nonreference_scale": args.canonical_weight_ss_nonreference_scale,
            "slat_nonreference_scale": args.canonical_weight_slat_nonreference_scale,
            "ss_sharpness": args.canonical_weight_ss_sharpness,
            "slat_sharpness": args.canonical_weight_slat_sharpness,
        },
        "glb_export": {
            "texture_size": args.texture_size,
            "simplify": args.simplify,
        },
        "backend": {
            "attn_backend": os.environ.get("ATTN_BACKEND", ""),
            "sparse_attn_backend": os.environ.get("SPARSE_ATTN_BACKEND", ""),
            "spconv_algo": os.environ.get("SPCONV_ALGO", ""),
        },
        "options": {
            "parallel": bool(args.parallel),
            "gpus": args.gpus,
            "overwrite": bool(args.overwrite),
            "save_videos": bool(args.save_videos),
            "save_gaussian_ply": bool(args.save_gaussian_ply),
            "dry_run": bool(args.dry_run),
        },
        "summary": {
            "total_cases": len(cases),
            "status_counts": counts,
        },
        "cases": case_entries,
    }


def write_manifest(
    args: argparse.Namespace,
    run_dir: Path,
    cases: Sequence[CaseSpec],
    case_statuses: Dict[str, Dict[str, Any]],
    started_at: str,
    completed_at: Optional[str] = None,
) -> None:
    manifest = build_manifest(
        args=args,
        run_dir=run_dir,
        cases=cases,
        case_statuses=case_statuses,
        started_at=started_at,
        completed_at=completed_at,
    )
    write_json(run_dir / "manifest.json", manifest)


def configure_backend_env(args: argparse.Namespace) -> None:
    backend_config = build_backend_config(
        attn_backend=args.attn_backend,
        sparse_attn_backend=args.sparse_attn_backend,
        spconv_algo=args.spconv_algo,
    )
    os.environ.update(backend_config.env)


def load_pipeline(model_name: str):
    from trellis.pipelines import TrellisImageTo3DPipeline

    pipeline = TrellisImageTo3DPipeline.from_pretrained(model_name)
    pipeline.cuda()
    return pipeline


def load_images(case: CaseSpec):
    from PIL import Image

    images = []
    for view in case.views:
        with Image.open(view.absolute_path) as image:
            images.append(image.convert("RGBA"))
    return images


def generate_case(
    *,
    case: CaseSpec,
    args: argparse.Namespace,
    run_dir: Path,
    pipeline,
) -> Tuple[str, Optional[str]]:
    from trellis.utils import postprocessing_utils

    case_dir = case_output_dir(run_dir, case.case_id)
    case_dir.mkdir(parents=True, exist_ok=True)
    error_path = case_dir / "error.json"
    tmp_glb_path = case_dir / "edit.tmp.glb"
    final_glb_path = case_dir / "edit.glb"

    if error_path.exists():
        error_path.unlink()
    if tmp_glb_path.exists():
        tmp_glb_path.unlink()

    try:
        images = load_images(case)
        front_reference_azimuth = next(
            (
                view.azimuth
                for view in case.views
                if str(view.label).strip().lower() == "front"
            ),
            case.views[0].azimuth,
        )
        outputs = pipeline.run_multi_image(
            images,
            seed=args.seed,
            sparse_structure_sampler_params={
                "steps": args.ss_steps,
                "cfg_strength": args.ss_cfg_strength,
            },
            slat_sampler_params={
                "steps": args.slat_steps,
                "cfg_strength": args.slat_cfg_strength,
            },
            formats=["mesh", "gaussian"],
            mode=args.mode,
            view_azimuths=[view.azimuth for view in case.views],
            view_reference_azimuth=front_reference_azimuth,
            canonical_view_weighting=(
                {
                    "floor": args.canonical_weight_floor,
                    "ss_nonreference_scale": args.canonical_weight_ss_nonreference_scale,
                    "slat_nonreference_scale": args.canonical_weight_slat_nonreference_scale,
                    "ss_sharpness": args.canonical_weight_ss_sharpness,
                    "slat_sharpness": args.canonical_weight_slat_sharpness,
                }
                if args.mode == "canonical_weighted_stochastic"
                else None
            ),
        )

        glb = postprocessing_utils.to_glb(
            outputs["gaussian"][0],
            outputs["mesh"][0],
            simplify=args.simplify,
            texture_size=args.texture_size,
        )
        glb.export(str(tmp_glb_path))
        tmp_glb_path.replace(final_glb_path)

        if args.save_gaussian_ply:
            outputs["gaussian"][0].save_ply(case_dir / "gaussian.ply")

        if args.save_videos:
            from trellis.utils import render_utils
            import imageio

            videos_dir = case_dir / "videos"
            videos_dir.mkdir(parents=True, exist_ok=True)
            gaussian_video = render_utils.render_video(outputs["gaussian"][0])["color"]
            mesh_video = render_utils.render_video(outputs["mesh"][0])["normal"]
            imageio.mimsave(videos_dir / "gaussian.mp4", gaussian_video, fps=30)
            imageio.mimsave(videos_dir / "mesh.mp4", mesh_video, fps=30)

        return "ok", None
    except Exception as exc:  # noqa: BLE001
        if tmp_glb_path.exists():
            tmp_glb_path.unlink()
        error_payload = {
            "case_id": case.case_id,
            "failed_at": now_iso(),
            "error": str(exc),
            "traceback": traceback.format_exc(),
        }
        write_json(error_path, error_payload)
        return "failed", str(exc)


def worker_main(args: argparse.Namespace) -> int:
    run_dir = args.output_root.resolve() / resolve_run_name(args)
    cases = build_case_specs(
        dataset_root=args.dataset_root.resolve(),
        view_kind=args.view_kind,
        view_order=args.view_order,
        custom_view_order=args.custom_view_order,
        view_keys=args.view_keys,
        conditioning_sequence=args.conditioning_sequence,
        requested_case_ids=args.case_id,
        limit=args.limit,
    )
    if len(cases) != 1:
        raise ValueError("Worker mode expects exactly one case, got {0}".format(len(cases)))

    configure_backend_env(args)
    pipeline = load_pipeline(args.model_name)
    status, error = generate_case(
        case=cases[0],
        args=args,
        run_dir=run_dir,
        pipeline=pipeline,
    )
    if status == "ok":
        return 0
    print("Failed case {0}: {1}".format(cases[0].case_id, error), file=sys.stderr)
    return 1


def build_worker_command(
    *,
    script_path: Path,
    args: argparse.Namespace,
    run_name: str,
    case_id: str,
) -> List[str]:
    command = [
        sys.executable,
        str(script_path),
        "--worker",
        "--dataset-root",
        str(args.dataset_root),
        "--output-root",
        str(args.output_root),
        "--run-name",
        run_name,
        "--mode",
        args.mode,
        "--view-kind",
        args.view_kind,
        "--view-order",
        args.view_order,
        "--view-keys",
        args.view_keys,
        "--conditioning-sequence",
        args.conditioning_sequence,
        "--model-name",
        args.model_name,
        "--seed",
        str(args.seed),
        "--ss-steps",
        str(args.ss_steps),
        "--ss-cfg-strength",
        str(args.ss_cfg_strength),
        "--slat-steps",
        str(args.slat_steps),
        "--slat-cfg-strength",
        str(args.slat_cfg_strength),
        "--canonical-weight-floor",
        str(args.canonical_weight_floor),
        "--canonical-weight-ss-nonreference-scale",
        str(args.canonical_weight_ss_nonreference_scale),
        "--canonical-weight-slat-nonreference-scale",
        str(args.canonical_weight_slat_nonreference_scale),
        "--canonical-weight-ss-sharpness",
        str(args.canonical_weight_ss_sharpness),
        "--canonical-weight-slat-sharpness",
        str(args.canonical_weight_slat_sharpness),
        "--texture-size",
        str(args.texture_size),
        "--simplify",
        str(args.simplify),
        "--case-id",
        case_id,
        "--attn-backend",
        args.attn_backend,
        "--sparse-attn-backend",
        args.sparse_attn_backend,
        "--spconv-algo",
        args.spconv_algo,
    ]
    if args.view_order == "custom":
        command.extend(["--custom-view-order", args.custom_view_order])

    if args.overwrite:
        command.append("--overwrite")
    if args.save_videos:
        command.append("--save-videos")
    if args.save_gaussian_ply:
        command.append("--save-gaussian-ply")
    return command


def read_case_error(run_dir: Path, case_id: str) -> Optional[str]:
    error_path = case_output_dir(run_dir, case_id) / "error.json"
    if not error_path.is_file():
        return None
    try:
        payload = read_json(error_path)
    except Exception:  # noqa: BLE001
        return None
    error_text = payload.get("error")
    if isinstance(error_text, str) and error_text.strip():
        return error_text.strip()
    return None


def parse_gpu_ids(gpu_text: str) -> List[str]:
    gpu_ids = [item.strip() for item in gpu_text.split(",") if item.strip()]
    if not gpu_ids:
        raise ValueError("No GPU ids provided")
    return gpu_ids


def run_parallel(
    *,
    args: argparse.Namespace,
    run_dir: Path,
    run_name: str,
    all_cases: Sequence[CaseSpec],
    pending_cases: Sequence[CaseSpec],
    case_statuses: Dict[str, Dict[str, Any]],
    started_at: str,
) -> int:
    gpu_queue: Deque[str] = deque(parse_gpu_ids(args.gpus))
    pending_queue: Deque[CaseSpec] = deque(pending_cases)
    active: List[Tuple[subprocess.Popen, CaseSpec, str]] = []
    script_path = Path(__file__).resolve()
    failures = 0

    while pending_queue or active:
        while pending_queue and gpu_queue:
            gpu_id = gpu_queue.popleft()
            case = pending_queue.popleft()
            env = os.environ.copy()
            env["CUDA_VISIBLE_DEVICES"] = gpu_id
            command = build_worker_command(
                script_path=script_path,
                args=args,
                run_name=run_name,
                case_id=case.case_id,
            )
            print("Launching {0} on GPU {1}".format(case.case_id, gpu_id))
            case_statuses[case.case_id] = {
                "status": "running",
                "gpu_id": gpu_id,
            }
            write_manifest(args, run_dir, all_cases, case_statuses, started_at)
            process = subprocess.Popen(command, env=env)
            active.append((process, case, gpu_id))

        if not active:
            continue

        time.sleep(5)
        still_active: List[Tuple[subprocess.Popen, CaseSpec, str]] = []
        for process, case, gpu_id in active:
            return_code = process.poll()
            if return_code is None:
                still_active.append((process, case, gpu_id))
                continue

            gpu_queue.append(gpu_id)
            if return_code == 0:
                print("✓ {0} completed on GPU {1}".format(case.case_id, gpu_id))
                case_statuses[case.case_id] = {
                    "status": "ok",
                    "gpu_id": gpu_id,
                }
            else:
                failures += 1
                error_text = read_case_error(run_dir, case.case_id)
                print(
                    "✗ {0} failed on GPU {1}{2}".format(
                        case.case_id,
                        gpu_id,
                        ": {0}".format(error_text) if error_text else "",
                    ),
                    file=sys.stderr,
                )
                case_statuses[case.case_id] = {
                    "status": "failed",
                    "gpu_id": gpu_id,
                    "error": error_text or "worker exit code {0}".format(return_code),
                }
            write_manifest(args, run_dir, all_cases, case_statuses, started_at)

        active = still_active

    return 1 if failures else 0


def main() -> int:
    args = parse_args()
    run_name = resolve_run_name(args)
    dataset_root = args.dataset_root.resolve()
    output_root = args.output_root.resolve()
    run_dir = output_root / run_name

    if args.worker:
        return worker_main(args)

    cases = build_case_specs(
        dataset_root=dataset_root,
        view_kind=args.view_kind,
        view_order=args.view_order,
        custom_view_order=args.custom_view_order,
        view_keys=args.view_keys,
        conditioning_sequence=args.conditioning_sequence,
        requested_case_ids=args.case_id,
        limit=args.limit,
    )
    if not cases:
        raise ValueError("No cases matched the requested filters")

    output_root.mkdir(parents=True, exist_ok=True)
    (run_dir / "cases").mkdir(parents=True, exist_ok=True)

    started_at = now_iso()
    case_statuses: Dict[str, Dict[str, Any]] = {}
    for case in cases:
        existing_glb = case_output_glb(run_dir, case.case_id)
        if existing_glb.is_file() and not args.overwrite:
            case_statuses[case.case_id] = {"status": "skipped"}
        else:
            case_statuses[case.case_id] = {"status": "pending"}

    configure_backend_env(args)
    write_manifest(args, run_dir, cases, case_statuses, started_at)

    print("Run directory: {0}".format(run_dir))
    print("Discovered {0} cases".format(len(cases)))
    print("Mode: {0}".format(args.mode))
    print("View kind: {0}".format(args.view_kind))

    if args.dry_run:
        print("Dry run complete. Manifest written to: {0}".format(run_dir / "manifest.json"))
        write_manifest(
            args,
            run_dir,
            cases,
            case_statuses,
            started_at,
            completed_at=now_iso(),
        )
        return 0

    pending_cases = [case for case in cases if case_statuses[case.case_id]["status"] == "pending"]
    if not pending_cases:
        print("All requested cases already have edit.glb outputs. Nothing to do.")
        write_manifest(
            args,
            run_dir,
            cases,
            case_statuses,
            started_at,
            completed_at=now_iso(),
        )
        return 0

    if args.parallel:
        exit_code = run_parallel(
            args=args,
            run_dir=run_dir,
            run_name=run_name,
            all_cases=cases,
            pending_cases=pending_cases,
            case_statuses=case_statuses,
            started_at=started_at,
        )
        write_manifest(
            args,
            run_dir,
            cases,
            case_statuses,
            started_at,
            completed_at=now_iso(),
        )
        return exit_code

    pipeline = load_pipeline(args.model_name)
    failures = 0
    for case in pending_cases:
        print("\n{0}".format("=" * 80))
        print("Processing {0}".format(case.case_id))
        print("{0}".format("=" * 80))
        case_statuses[case.case_id] = {"status": "running"}
        write_manifest(args, run_dir, cases, case_statuses, started_at)
        status, error = generate_case(
            case=case,
            args=args,
            run_dir=run_dir,
            pipeline=pipeline,
        )
        case_statuses[case.case_id] = {
            "status": status,
            "error": error,
        }
        write_manifest(args, run_dir, cases, case_statuses, started_at)
        if status == "ok":
            print("✓ Completed {0}".format(case.case_id))
            continue

        failures += 1
        print("✗ Failed {0}: {1}".format(case.case_id, error), file=sys.stderr)
        if args.fail_fast:
            break

    write_manifest(
        args,
        run_dir,
        cases,
        case_statuses,
        started_at,
        completed_at=now_iso(),
    )
    return 1 if failures else 0


if __name__ == "__main__":
    raise SystemExit(main())
