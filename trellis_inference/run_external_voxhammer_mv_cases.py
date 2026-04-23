#!/usr/bin/env python3
from __future__ import annotations

import argparse
import contextlib
import json
import os
import random
import shutil
import sys
from pathlib import Path
from typing import TextIO

import numpy as np
import torch


os.environ.setdefault("ATTN_BACKEND", "flash_attn")
os.environ.setdefault("SPARSE_ATTN_BACKEND", "flash_attn")
os.environ.setdefault("SPCONV_ALGO", "native")


DEFAULT_DATASET_ROOT = Path(
    "/cache/wangxinxing/data/trellis_edit_benchmark/edit3d_mv_pseudosource_micro10"
)
DEFAULT_SHARED_RENDER_ROOT = Path(
    "/cache/wangxinxing/data/trellis_edit_benchmark/edit3d_pseudosource_micro10_renders"
)
DEFAULT_VOXHAMMER_ROOT = Path("/home/wangxinxing/3dlocaledit/VoxHammer")


class _Tee(TextIO):
    def __init__(self, *streams: TextIO):
        self._streams = streams

    def write(self, data: str) -> int:
        for stream in self._streams:
            stream.write(data)
            stream.flush()
        return len(data)

    def flush(self) -> None:
        for stream in self._streams:
            stream.flush()


def _set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def _ensure_symlink(src: Path, dst: Path) -> None:
    if dst.is_symlink() or dst.exists():
        dst.unlink()
    dst.symlink_to(src.resolve())


def _prepare_render_dir(shared_render_dir: Path, render_dir: Path) -> None:
    render_dir.mkdir(parents=True, exist_ok=True)
    for name in ("transforms.json", "mesh.ply", "voxels.ply", "features.npz"):
        src = shared_render_dir / name
        if not src.is_file():
            raise FileNotFoundError(f"Missing shared render asset: {src}")
        _ensure_symlink(src, render_dir / name)


def _prepare_image_dir(case_root: Path, case_payload: dict, view_key: str, image_dir: Path) -> None:
    view_payload = case_payload["input_views"][view_key]
    image_dir.mkdir(parents=True, exist_ok=True)
    mapping = {
        "2d_render.png": case_root / view_payload["source_path"],
        "2d_edit.png": case_root / view_payload["target_path"],
        "2d_mask.png": case_root / view_payload["mask_path"],
    }
    for dst_name, src in mapping.items():
        if not src.is_file():
            raise FileNotFoundError(f"Missing image input: {src}")
        shutil.copy2(src, image_dir / dst_name)


def _load_case_payload(dataset_root: Path, case_id: str) -> tuple[Path, dict]:
    case_root = dataset_root / "cases" / case_id
    if not case_root.is_dir():
        raise FileNotFoundError(f"Case root does not exist: {case_root}")
    payload = json.loads((case_root / "case.json").read_text(encoding="utf-8"))
    return case_root, payload


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Run external VoxHammer on selected MV benchmark cases.")
    parser.add_argument("--case-id", action="append", required=True, help="Case id, repeatable.")
    parser.add_argument("--pred-root", type=Path, required=True, help="Output prediction root.")
    parser.add_argument("--tmp-root", type=Path, required=True, help="Temporary workspace root.")
    parser.add_argument("--dataset-root", type=Path, default=DEFAULT_DATASET_ROOT)
    parser.add_argument("--shared-render-root", type=Path, default=DEFAULT_SHARED_RENDER_ROOT)
    parser.add_argument("--voxhammer-root", type=Path, default=DEFAULT_VOXHAMMER_ROOT)
    parser.add_argument("--view-key", type=str, default="cardinal_neg90")
    parser.add_argument("--model", type=str, default="microsoft/TRELLIS-image-large")
    parser.add_argument("--seed", type=int, default=1)
    parser.add_argument("--skip-existing", action="store_true", default=True)
    parser.add_argument("--no-skip-existing", dest="skip_existing", action="store_false")
    return parser


def main() -> int:
    args = build_parser().parse_args()
    dataset_root = args.dataset_root.expanduser().resolve()
    shared_render_root = args.shared_render_root.expanduser().resolve()
    voxhammer_root = args.voxhammer_root.expanduser().resolve()
    pred_root = args.pred_root.expanduser().resolve()
    tmp_root = args.tmp_root.expanduser().resolve()

    if str(voxhammer_root) not in sys.path:
        sys.path.insert(0, str(voxhammer_root))

    from inference import run_3d_editing, run_voxel_masking
    from trellis.pipelines import TrellisImageTo3DPipeline

    pred_root.mkdir(parents=True, exist_ok=True)
    tmp_root.mkdir(parents=True, exist_ok=True)

    print(f"[load] pipeline={args.model}")
    pipeline = TrellisImageTo3DPipeline.from_pretrained(args.model)
    pipeline.cuda()
    print("[ready] external VoxHammer pipeline loaded")

    completed_cases: list[str] = []
    for case_id in args.case_id:
        case_root, case_payload = _load_case_payload(dataset_root, case_id)
        dataset_name = str(case_payload["dataset"])
        object_name = str(case_payload["source_model_name"])
        prompt_id = int(case_payload["prompt_id"])
        shared_render_dir = shared_render_root / dataset_name / object_name
        if not shared_render_dir.is_dir():
            raise FileNotFoundError(f"Missing shared render dir: {shared_render_dir}")

        case_tmp_root = tmp_root / case_id
        render_dir = case_tmp_root / "render"
        image_dir = case_tmp_root / "images"
        output_dir = pred_root / "cases" / case_id
        output_dir.mkdir(parents=True, exist_ok=True)
        output_glb = output_dir / "edit.glb"
        log_path = output_dir / "run.log"

        if args.skip_existing and output_glb.is_file():
            print(f"[skip] {case_id} -> {output_glb}")
            completed_cases.append(case_id)
            continue

        if render_dir.exists():
            shutil.rmtree(render_dir)
        if image_dir.exists():
            shutil.rmtree(image_dir)
        _prepare_render_dir(shared_render_dir, render_dir)
        _prepare_image_dir(case_root, case_payload, args.view_key, image_dir)

        mask_glb = case_root / str(case_payload["edit_region_path"])
        source_prompt = str(case_payload.get("source_prompt") or "")
        target_prompt = str(case_payload.get("prompt_text") or case_payload.get("edit_instruction") or "")

        print(f"[case] {case_id}")
        with log_path.open("w", encoding="utf-8") as log_fp:
            tee = _Tee(sys.stdout, log_fp)
            with contextlib.redirect_stdout(tee), contextlib.redirect_stderr(tee):
                print(f"case_id={case_id}")
                print(f"dataset={dataset_name} object={object_name} prompt_id={prompt_id}")
                print(f"view_key={args.view_key}")
                print(f"shared_render_dir={shared_render_dir}")
                print(f"mask_glb={mask_glb}")
                print(f"output_glb={output_glb}")
                _set_seed(args.seed)
                run_voxel_masking(str(mask_glb), str(render_dir))
                _set_seed(args.seed)
                run_3d_editing(
                    pipeline=pipeline,
                    render_dir=str(render_dir),
                    output_path=str(output_glb),
                    image_dir=str(image_dir),
                    is_text=False,
                    source_prompt=source_prompt,
                    target_prompt=target_prompt,
                )
        case_meta = {
            "case_id": case_id,
            "dataset": dataset_name,
            "object_name": object_name,
            "prompt_id": prompt_id,
            "view_key": args.view_key,
            "source_prompt": source_prompt,
            "target_prompt": target_prompt,
            "mask_glb": str(mask_glb),
            "shared_render_dir": str(shared_render_dir),
            "tmp_render_dir": str(render_dir),
            "tmp_image_dir": str(image_dir),
            "output_glb": str(output_glb),
        }
        (output_dir / "case_meta.json").write_text(
            json.dumps(case_meta, indent=2, ensure_ascii=False) + "\n",
            encoding="utf-8",
        )
        completed_cases.append(case_id)
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

    manifest = {
        "run_name": pred_root.name,
        "config_name": pred_root.name,
        "method": "external_voxhammer",
        "group": "mv",
        "dataset_root": str(dataset_root),
        "shared_render_root": str(shared_render_root),
        "voxhammer_root": str(voxhammer_root),
        "view_key": args.view_key,
        "model": args.model,
        "seed": args.seed,
        "cases": completed_cases,
    }
    (pred_root / "manifest.json").write_text(
        json.dumps(manifest, indent=2, ensure_ascii=False) + "\n",
        encoding="utf-8",
    )
    print(f"[done] wrote {len(completed_cases)} case(s) to {pred_root}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
