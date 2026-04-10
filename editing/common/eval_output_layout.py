"""
Edit3D-Bench evaluation output layout.

Provides utilities to organize editing results in the format required by Edit3D-Bench evaluation system.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path


@dataclass(frozen=True)
class EvalOutputLayout:
    """
    Edit3D-Bench evaluation output structure:

    exp_comparison/
    └── {method_name}/
        └── {dataset}/                    # e.g., GSO, PartObjaverse-Tiny
            └── {object_name}/            # e.g., Toy_Poodle
                └── prompt_{1,2,3}/       # Three editing prompts
                    ├── edit.glb          # Edited 3D model (required)
                    ├── images/           # Rendered multi-view images (required)
                    │   └── render_XXXX.png  # 16 views (0000-0015)
                    └── videos/           # Rendered videos (optional)
                        └── video_rgb.mp4
    """
    method_name: str
    dataset: str
    object_name: str
    prompt_id: int

    root_dir: Path
    method_dir: Path
    dataset_dir: Path
    object_dir: Path
    prompt_dir: Path

    # Required outputs
    edit_glb: Path
    images_dir: Path

    # Optional outputs
    videos_dir: Path
    video_rgb: Path


def build_eval_output_layout(
    method_name: str,
    dataset: str,
    object_name: str,
    prompt_id: int,
    eval_root: Path | str = "exp_comparison",
) -> EvalOutputLayout:
    """
    Build Edit3D-Bench evaluation output layout.

    Args:
        method_name: Name of the editing method (e.g., "VoxHammer", "TRELLIS_EDIT")
        dataset: Dataset name (e.g., "GSO", "PartObjaverse-Tiny")
        object_name: Object name (e.g., "Toy_Poodle")
        prompt_id: Prompt ID (1, 2, or 3)
        eval_root: Root directory for evaluation outputs

    Returns:
        EvalOutputLayout with all paths configured
    """
    root_dir = Path(eval_root)
    method_dir = root_dir / method_name
    dataset_dir = method_dir / dataset
    object_dir = dataset_dir / object_name
    prompt_dir = object_dir / f"prompt_{prompt_id}"

    images_dir = prompt_dir / "images"
    videos_dir = prompt_dir / "videos"

    return EvalOutputLayout(
        method_name=method_name,
        dataset=dataset,
        object_name=object_name,
        prompt_id=prompt_id,
        root_dir=root_dir,
        method_dir=method_dir,
        dataset_dir=dataset_dir,
        object_dir=object_dir,
        prompt_dir=prompt_dir,
        edit_glb=prompt_dir / "edit.glb",
        images_dir=images_dir,
        videos_dir=videos_dir,
        video_rgb=videos_dir / "video_rgb.mp4",
    )


def get_render_view_path(layout: EvalOutputLayout, view_idx: int) -> Path:
    """Get path for a specific rendered view (0-15)."""
    return layout.images_dir / f"render_{view_idx:04d}.png"
