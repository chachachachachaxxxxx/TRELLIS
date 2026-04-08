from __future__ import annotations

import gc
import sys
from datetime import datetime, timezone
import json
from pathlib import Path
from typing import Any

import torch


def ensure_dir(path: Path | str) -> Path:
    resolved = Path(path)
    resolved.mkdir(parents=True, exist_ok=True)
    return resolved


def write_json(path: Path | str, payload: Any) -> Path:
    resolved = Path(path)
    ensure_dir(resolved.parent)
    resolved.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
    return resolved


def read_json(path: Path | str) -> Any:
    return json.loads(Path(path).read_text(encoding="utf-8"))


def utc_now_iso() -> str:
    return datetime.now(timezone.utc).isoformat().replace("+00:00", "Z")


def release_cuda_memory() -> None:
    """Release CUDA memory by running garbage collection and emptying cache."""
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
        torch.cuda.synchronize()


def offload_models_to_cpu(pipeline) -> None:
    """Offload pipeline models to CPU to free GPU memory before GLB export.

    Args:
        pipeline: The pipeline object with models dict
    """
    if hasattr(pipeline, 'models'):
        for model in pipeline.models.values():
            if hasattr(model, 'cpu'):
                model.cpu()
    release_cuda_memory()


def save_preview_video(render_utils, imageio, sample, out_path: Path, channel: str, label: str) -> bool:
    """Save a preview video for a sample.

    Args:
        render_utils: Render utilities module
        imageio: imageio module
        sample: Sample to render
        out_path: Output path for the video
        channel: Channel to render (e.g., "color", "normal")
        label: Label for error messages (e.g., "gaussian", "radiance_field", "mesh")

    Returns:
        True if successful, False otherwise
    """
    release_cuda_memory()
    try:
        video = render_utils.render_video(sample)[channel]
        imageio.mimsave(str(out_path), video, fps=30)
        return True
    except Exception as exc:
        message = str(exc)
        message_lower = message.lower()
        extra = ""
        if "out of memory" in message_lower:
            extra = (
                " This preview step ran out of VRAM. "
                "Re-run with --skip-radiance-field-render or --skip-render if you only need the generated assets."
            )
        elif "operation not supported on global/shared address space" in message_lower and label == "radiance_field":
            extra = (
                " This often points to a diffoctreerast/CUDA compatibility issue. "
                "Re-run with --skip-radiance-field-render or --skip-render if you only need the generated assets."
            )
        print(
            f"Warning: failed to render {label} preview at {out_path.name}: {message}.{extra}",
            file=sys.stderr,
        )
        return False
    finally:
        if "video" in locals():
            del video
        release_cuda_memory()


def save_outputs(
    outputs: dict,
    out_dir: Path,
    skip_render: bool,
    skip_radiance_field_render: bool = False,
    skip_glb: bool = False,
    skip_ply: bool = False,
) -> None:
    """Save outputs from TRELLIS pipeline.

    Args:
        outputs: Dictionary containing mesh, gaussian, radiance_field outputs
        out_dir: Output directory
        skip_render: Skip rendering preview videos
        skip_radiance_field_render: Skip radiance field preview rendering
        skip_glb: Skip GLB export
        skip_ply: Skip PLY export
    """
    imageio = None
    render_utils = None
    postprocessing_utils = None
    if not skip_render:
        import imageio as _imageio
        from trellis.utils import render_utils as _render_utils

        imageio = _imageio
        render_utils = _render_utils
    if not skip_glb:
        from trellis.utils import postprocessing_utils as _postprocessing_utils

        postprocessing_utils = _postprocessing_utils

    num_samples = 0
    for key in ("mesh", "gaussian", "radiance_field"):
        if key in outputs:
            num_samples = max(num_samples, len(outputs[key]))
    num_samples = max(num_samples, 1)

    print(f"Saving outputs: {list(outputs.keys())}, num_samples={num_samples}")
    print(f"skip_glb={skip_glb}, skip_ply={skip_ply}, skip_render={skip_render}")

    for sample_idx in range(num_samples):
        prefix = f"sample_{sample_idx:02d}"

        if not skip_render:
            if "gaussian" in outputs:
                save_preview_video(
                    render_utils,
                    imageio,
                    outputs["gaussian"][sample_idx],
                    out_dir / f"{prefix}_gs.mp4",
                    "color",
                    "gaussian",
                )
            if "radiance_field" in outputs and not skip_radiance_field_render:
                save_preview_video(
                    render_utils,
                    imageio,
                    outputs["radiance_field"][sample_idx],
                    out_dir / f"{prefix}_rf.mp4",
                    "color",
                    "radiance_field",
                )
            if "mesh" in outputs:
                save_preview_video(
                    render_utils,
                    imageio,
                    outputs["mesh"][sample_idx],
                    out_dir / f"{prefix}_mesh.mp4",
                    "normal",
                    "mesh",
                )

        if not skip_glb and "gaussian" in outputs and "mesh" in outputs:
            print(f"Exporting GLB for sample {sample_idx}...")
            release_cuda_memory()
            glb = postprocessing_utils.to_glb(
                outputs["gaussian"][sample_idx],
                outputs["mesh"][sample_idx],
                simplify=0.95,
                texture_size=1024,
            )
            glb_path = out_dir / f"{prefix}.glb"
            glb.export(str(glb_path))
            print(f"✓ Saved GLB: {glb_path}")
            del glb
            release_cuda_memory()

        if not skip_ply and "gaussian" in outputs:
            print(f"Exporting PLY for sample {sample_idx}...")
            release_cuda_memory()
            ply_path = out_dir / f"{prefix}.ply"
            outputs["gaussian"][sample_idx].save_ply(str(ply_path))
            print(f"✓ Saved PLY: {ply_path}")
