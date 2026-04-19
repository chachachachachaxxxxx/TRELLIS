#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import shutil
import subprocess
import sys
import time
import traceback
from pathlib import Path
from typing import Any

from trellis_edit.common import ensure_dir, utc_now_iso, write_json
from trellis_edit.common.external_3d import (
    DEFAULT_HUNYUAN_MODEL,
    DEFAULT_ULTRASHAPE_CKPT,
    DEFAULT_ULTRASHAPE_CONFIG,
)


REPO_ROOT = Path(__file__).resolve().parent
DEFAULT_GT_ROOT = Path("/cache/wangxinxing/data/trellis_edit_benchmark/edit3d_data/data")
DEFAULT_DIRECT_OUTPUT_ROOT = Path(
    "/cache/wangxinxing/data/trellis_edit_benchmark/pred/ultrashape_direct_material_first12"
)
DEFAULT_AUTO_OUTPUT_ROOT = Path(
    "/cache/wangxinxing/data/trellis_edit_benchmark/pred/ultrashape_autoencode_material_first12"
)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=(
            "Run UltraShape material benchmark generation on Edit3D-Bench by batching the "
            "existing single-case smoke scripts."
        ),
    )
    parser.add_argument("--mode", choices=("direct", "autoencode"), required=True)
    parser.add_argument("--gt-root", type=Path, default=DEFAULT_GT_ROOT)
    parser.add_argument("--output-root", type=Path, default=None)
    parser.add_argument("--device", type=str, default="cuda:0")
    parser.add_argument("--limit", type=int, default=12, help="Number of prompt-level cases to include.")
    parser.add_argument("--seed", type=int, default=1)
    parser.add_argument("--resume", action="store_true", help="Skip prompt outputs that already exist.")
    parser.add_argument(
        "--case-shard-count",
        type=int,
        default=1,
        help="Split prompt-level generation into this many disjoint shards.",
    )
    parser.add_argument(
        "--case-shard-index",
        type=int,
        default=0,
        help="Zero-based case shard index to run.",
    )
    parser.add_argument(
        "--max-new-cases",
        type=int,
        default=0,
        help="Stop after generating this many new prompt cases. 0 means no limit.",
    )
    parser.add_argument(
        "--continue-on-case-error",
        action="store_true",
        help="Log prompt-level failures and continue instead of aborting the whole run.",
    )
    parser.add_argument("--config-name", type=str, default="")
    parser.add_argument("--run-group", type=str, default="baseline")
    parser.add_argument("--hunyuan-model", type=str, default=DEFAULT_HUNYUAN_MODEL)
    parser.add_argument("--ultrashape-config", type=Path, default=DEFAULT_ULTRASHAPE_CONFIG)
    parser.add_argument("--ultrashape-ckpt", type=Path, default=DEFAULT_ULTRASHAPE_CKPT)
    parser.add_argument("--steps", type=int, default=4, help="UltraShape refine steps for direct mode.")
    parser.add_argument(
        "--num-latents",
        type=int,
        default=8192,
        help="UltraShape direct mode latent count.",
    )
    parser.add_argument(
        "--chunk-size",
        type=int,
        default=0,
        help="Chunk size override. 0 keeps the per-mode default.",
    )
    parser.add_argument("--octree-res", type=int, default=256)
    parser.add_argument("--drop-normal", action="store_true")
    return parser


def _default_output_root(mode: str) -> Path:
    if mode == "direct":
        return DEFAULT_DIRECT_OUTPUT_ROOT
    return DEFAULT_AUTO_OUTPUT_ROOT


def _default_config_name(mode: str) -> str:
    if mode == "direct":
        return "ultrashape_direct_material_first12"
    return "ultrashape_autoencode_material_first12"


def _manifest_filename(mode: str) -> str:
    if mode == "direct":
        return "direct_manifest.json"
    return "autoencode_manifest.json"


def _failures_filename(mode: str) -> str:
    if mode == "direct":
        return "direct_failures.json"
    return "autoencode_failures.json"


def _method_name(mode: str) -> str:
    if mode == "direct":
        return "ultrashape_direct_material"
    return "ultrashape_autoencode_material"


def _script_path(mode: str) -> Path:
    if mode == "direct":
        return REPO_ROOT / "run_ultrashape_direct_edit_material_smoke.py"
    return REPO_ROOT / "run_ultrashape_autoencode_material_smoke.py"


def _required_prompt_asset(mode: str) -> str:
    if mode == "direct":
        return "2d_edit.png"
    return "2d_render.png"


def _select_cases(
    metadata: list[dict[str, Any]],
    *,
    gt_root: Path,
    limit: int,
    mode: str,
) -> list[tuple[str, str, int]]:
    cases: list[tuple[str, str, int]] = []
    required_asset = _required_prompt_asset(mode)
    for row in metadata:
        dataset = str(row["dataset"])
        object_name = str(row["source_model"])
        reference_glb = gt_root / dataset / object_name / "source_model" / "model.glb"
        if not reference_glb.is_file():
            continue
        for prompt_id in (1, 2, 3):
            prompt_dir = gt_root / dataset / object_name / f"prompt_{prompt_id}"
            prompt_asset = prompt_dir / required_asset
            if not prompt_dir.is_dir() or not prompt_asset.is_file():
                continue
            cases.append((dataset, object_name, prompt_id))
            if len(cases) >= limit:
                return cases
    return cases


def _select_case_shard(
    cases: list[tuple[str, str, int]],
    *,
    shard_count: int,
    shard_index: int,
) -> list[tuple[str, str, int]]:
    if shard_count <= 1:
        return cases
    if shard_index < 0 or shard_index >= shard_count:
        raise ValueError(f"Invalid shard index {shard_index} for shard count {shard_count}.")
    return [case for idx, case in enumerate(cases) if idx % shard_count == shard_index]


def _case_identifier(dataset: str, object_name: str, prompt_id: int) -> str:
    return f"{dataset}/{object_name}/prompt_{prompt_id}"


def _final_glb(output_root: Path, dataset: str, object_name: str, prompt_id: int) -> Path:
    return output_root / dataset / object_name / f"prompt_{prompt_id}" / "edit.glb"


def _run_case(
    *,
    args: argparse.Namespace,
    output_root: Path,
    dataset: str,
    object_name: str,
    prompt_id: int,
    case_seed: int,
) -> dict[str, Any]:
    output_glb = _final_glb(output_root, dataset, object_name, prompt_id)
    if args.resume and output_glb.is_file():
        return {
            "dataset": dataset,
            "object_name": object_name,
            "prompt_id": prompt_id,
            "glb_path": str(output_glb),
            "resumed": True,
            "created_at": utc_now_iso(),
        }

    cmd = [
        sys.executable,
        str(_script_path(args.mode)),
        "--gt-root",
        str(args.gt_root),
        "--output-root",
        str(output_root),
        "--dataset",
        dataset,
        "--object-name",
        object_name,
        "--prompt-id",
        str(prompt_id),
        "--device",
        args.device,
        "--ultrashape-config",
        str(args.ultrashape_config),
        "--ultrashape-ckpt",
        str(args.ultrashape_ckpt),
        "--octree-res",
        str(args.octree_res),
    ]
    if args.drop_normal:
        cmd.append("--drop-normal")

    chunk_size = args.chunk_size
    if args.mode == "direct":
        cmd.extend(
            [
                "--hunyuan-model",
                args.hunyuan_model,
                "--seed",
                str(case_seed),
                "--steps",
                str(args.steps),
                "--num-latents",
                str(args.num_latents),
                "--chunk-size",
                str(chunk_size or 2048),
            ]
        )
    else:
        cmd.extend(["--chunk-size", str(chunk_size or 8000)])

    started_at = time.time()
    result = subprocess.run(cmd, cwd=REPO_ROOT, text=True)
    if result.returncode != 0:
        raise RuntimeError(f"Case command failed with exit code {result.returncode}")
    if not output_glb.is_file():
        raise FileNotFoundError(f"Expected edit.glb was not created: {output_glb}")

    case_dir = output_glb.parent
    case_payload_path = case_dir / "run.json"
    case_payload: dict[str, Any] = {
        "dataset": dataset,
        "object_name": object_name,
        "prompt_id": prompt_id,
        "glb_path": str(output_glb),
        "total_seconds": round(time.time() - started_at, 3),
        "resumed": False,
        "created_at": utc_now_iso(),
    }
    if case_payload_path.is_file():
        try:
            existing_payload = json.loads(case_payload_path.read_text(encoding="utf-8"))
            if isinstance(existing_payload, dict):
                case_payload.update(existing_payload)
                case_payload["resumed"] = False
        except Exception:
            pass
    return case_payload


def main() -> None:
    args = build_parser().parse_args()
    args.gt_root = args.gt_root.expanduser().resolve()
    args.output_root = (args.output_root or _default_output_root(args.mode)).expanduser().resolve()
    if not args.config_name:
        args.config_name = _default_config_name(args.mode)
    if args.case_shard_count < 1:
        raise ValueError("--case-shard-count must be >= 1.")
    if args.case_shard_count > 1 and not args.resume:
        raise RuntimeError(
            "Sharded generation should start from a prepared output dir. "
            "Clean it once, then launch all shards with --resume."
        )

    metadata_path = args.gt_root / "metadata.json"
    metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
    cases = _select_cases(metadata, gt_root=args.gt_root, limit=args.limit, mode=args.mode)
    if not cases:
        raise RuntimeError("No benchmark cases selected.")
    shard_cases = _select_case_shard(
        cases,
        shard_count=args.case_shard_count,
        shard_index=args.case_shard_index,
    )
    if not shard_cases:
        raise RuntimeError(
            f"No cases selected for shard {args.case_shard_index}/{args.case_shard_count}."
        )

    if args.output_root.exists() and not args.resume:
        shutil.rmtree(args.output_root)
    ensure_dir(args.output_root)

    manifest_path = args.output_root / _manifest_filename(args.mode)
    failures_path = args.output_root / _failures_filename(args.mode)
    manifest_cases: dict[str, Any] = {}
    failed_cases: dict[str, Any] = {}
    if args.resume and manifest_path.is_file():
        existing_manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
        manifest_cases.update(existing_manifest.get("generated_cases") or {})
        failed_cases.update(existing_manifest.get("failed_cases") or {})
    if args.resume and failures_path.is_file():
        existing_failures = json.loads(failures_path.read_text(encoding="utf-8"))
        failed_cases.update(existing_failures.get("cases") or {})

    start_time = time.time()
    new_case_count = 0
    total_cases = len(shard_cases)
    method_name = _method_name(args.mode)
    print(
        f"[Shard] case shard {args.case_shard_index}/{args.case_shard_count} "
        f"selected {total_cases}/{len(cases)} cases (device={args.device})"
    )

    for case_index, (dataset, object_name, prompt_id) in enumerate(shard_cases, start=1):
        case_id = _case_identifier(dataset, object_name, prompt_id)
        case_seed = args.seed + case_index - 1
        print(f"[UltraShape/{args.mode}] {case_index}/{total_cases} {case_id}")
        try:
            case_result = _run_case(
                args=args,
                output_root=args.output_root,
                dataset=dataset,
                object_name=object_name,
                prompt_id=prompt_id,
                case_seed=case_seed,
            )
        except Exception as exc:
            failed_cases[case_id] = {
                "dataset": dataset,
                "object_name": object_name,
                "prompt_id": prompt_id,
                "error": str(exc),
                "traceback": traceback.format_exc(),
                "created_at": utc_now_iso(),
            }
            write_json(failures_path, {"cases": failed_cases})
            if not args.continue_on_case_error:
                raise
            print(f"[Skip] {case_id} failed: {exc}")
            continue

        manifest_cases[case_id] = case_result
        failed_cases.pop(case_id, None)
        if not case_result.get("resumed", False):
            new_case_count += 1

        if failed_cases:
            write_json(failures_path, {"cases": failed_cases})
        elif failures_path.exists():
            failures_path.unlink()

        write_json(
            manifest_path,
            {
                "config_name": args.config_name,
                "run_group": args.run_group,
                "run_name": args.output_root.name,
                "method": method_name,
                "device": args.device,
                "mode": args.mode,
                "seed": args.seed,
                "case_shard_count": args.case_shard_count,
                "case_shard_index": args.case_shard_index,
                "drop_normal": bool(args.drop_normal),
                "failed_cases": failed_cases,
                "cases": [
                    {
                        "dataset": case_dataset,
                        "object_name": case_object_name,
                        "prompt_id": case_prompt_id,
                    }
                    for case_dataset, case_object_name, case_prompt_id in cases
                ],
                "generated_cases": manifest_cases,
                "created_at": utc_now_iso(),
            },
        )
        if args.max_new_cases > 0 and new_case_count >= args.max_new_cases:
            print(f"[Stop] Reached max new cases for this run: {new_case_count}/{args.max_new_cases}")
            break

    summary = {
        "config_name": args.config_name,
        "run_group": args.run_group,
        "run_name": args.output_root.name,
        "method": method_name,
        "device": args.device,
        "mode": args.mode,
        "num_selected_cases": len(cases),
        "num_shard_cases": total_cases,
        "num_generated_cases": len(manifest_cases),
        "num_failed_cases": len(failed_cases),
        "new_cases_this_run": new_case_count,
        "drop_normal": bool(args.drop_normal),
        "total_seconds": round(time.time() - start_time, 3),
        "created_at": utc_now_iso(),
    }
    write_json(args.output_root / "summary.json", summary)
    print(
        f"[Done] UltraShape {args.mode} generation saved {len(manifest_cases)}/{total_cases} cases "
        f"(new this run: {new_case_count})"
    )
    print(f"[Done] Pred root: {args.output_root}")


if __name__ == "__main__":
    main()
