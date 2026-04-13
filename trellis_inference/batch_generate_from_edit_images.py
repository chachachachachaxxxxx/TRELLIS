#!/usr/bin/env python3
"""
Batch generate 3D models from 2d_edit.png images in asset directories.
Supports parallel execution across multiple GPUs.

Usage:
    # Single GPU
    python batch_generate_from_edit_images.py --asset-dir assets/hard5 --output-dir outputs/hard5_edit_direct

    # Multi-GPU parallel (one case per GPU)
    python batch_generate_from_edit_images.py --asset-dir assets/hard5 --output-dir outputs/hard5_edit_direct --parallel
"""

import os
import sys
from pathlib import Path

os.environ['SPCONV_ALGO'] = 'native'

# Add parent directory to path for imports
sys.path.insert(0, str(Path(__file__).parent.parent))

import argparse
import subprocess
from typing import List, Tuple
import imageio
from PIL import Image
from trellis.pipelines import TrellisImageTo3DPipeline
from trellis.utils import render_utils, postprocessing_utils


def find_edit_images(asset_dir: Path) -> List[Tuple[Path, str]]:
    """Find all 2d_edit.png images in asset directory structure.

    Returns:
        List of (image_path, case_name) tuples
    """
    results = []
    for case_dir in sorted(asset_dir.iterdir()):
        if not case_dir.is_dir():
            continue

        edit_image = case_dir / "images" / "2d_edit.png"
        if edit_image.exists():
            results.append((edit_image, case_dir.name))

    return results


def generate_single_case(image_path: Path, case_name: str, output_dir: Path, seed: int = 1):
    """Generate 3D model from a single edit image."""
    print(f"\n{'='*80}")
    print(f"Processing: {case_name}")
    print(f"Image: {image_path}")
    print(f"{'='*80}\n")

    # Load pipeline
    pipeline = TrellisImageTo3DPipeline.from_pretrained("microsoft/TRELLIS-image-large")
    pipeline.cuda()

    # Load image
    image = Image.open(image_path)

    # Create output directory
    case_output_dir = output_dir / case_name
    case_output_dir.mkdir(parents=True, exist_ok=True)

    # Run pipeline
    outputs = pipeline.run(image, seed=seed)

    # Render videos
    print(f"Rendering videos for {case_name}...")
    video = render_utils.render_video(outputs['gaussian'][0])['color']
    imageio.mimsave(case_output_dir / "gaussian.mp4", video, fps=30)

    video = render_utils.render_video(outputs['radiance_field'][0])['color']
    imageio.mimsave(case_output_dir / "radiance_field.mp4", video, fps=30)

    video = render_utils.render_video(outputs['mesh'][0])['normal']
    imageio.mimsave(case_output_dir / "mesh.mp4", video, fps=30)

    # Export GLB
    print(f"Exporting GLB for {case_name}...")
    glb = postprocessing_utils.to_glb(
        outputs['gaussian'][0],
        outputs['mesh'][0],
        simplify=0.95,
        texture_size=1024,
    )
    glb.export(case_output_dir / "edit.glb")

    # Save Gaussians
    outputs['gaussian'][0].save_ply(case_output_dir / "gaussian.ply")

    print(f"\n✓ Completed: {case_name}")
    print(f"  Output: {case_output_dir}")


def run_parallel_worker(image_path: str, case_name: str, output_dir: str, gpu_id: int, seed: int):
    """Worker function for parallel execution on specific GPU."""
    cmd = [
        "python", "-c",
        f"""
import os
os.environ['CUDA_VISIBLE_DEVICES'] = '{gpu_id}'
os.environ['SPCONV_ALGO'] = 'native'

from pathlib import Path
from batch_generate_from_edit_images import generate_single_case

generate_single_case(
    Path('{image_path}'),
    '{case_name}',
    Path('{output_dir}'),
    seed={seed}
)
"""
    ]

    subprocess.run(cmd, check=True)


def main():
    parser = argparse.ArgumentParser(description="Batch generate 3D from edit images")
    parser.add_argument("--asset-dir", type=Path, required=True,
                        help="Directory containing case subdirectories with images/2d_edit.png")
    parser.add_argument("--output-dir", type=Path, required=True,
                        help="Output directory for generated models")
    parser.add_argument("--seed", type=int, default=1,
                        help="Random seed")
    parser.add_argument("--parallel", action="store_true",
                        help="Run cases in parallel across GPUs")
    parser.add_argument("--gpus", type=str, default="0,1,2,3,4,5,6,7",
                        help="Comma-separated GPU IDs to use")

    args = parser.parse_args()

    # Find all edit images
    cases = find_edit_images(args.asset_dir)
    print(f"Found {len(cases)} cases to process:")
    for img_path, case_name in cases:
        print(f"  - {case_name}")
    print()

    if not cases:
        print("No 2d_edit.png images found!")
        return

    # Create output directory
    args.output_dir.mkdir(parents=True, exist_ok=True)

    if args.parallel:
        # Parallel execution
        gpu_ids = [int(x.strip()) for x in args.gpus.split(",")]
        print(f"Running in parallel mode using GPUs: {gpu_ids}\n")

        processes = []
        for i, (img_path, case_name) in enumerate(cases):
            gpu_id = gpu_ids[i % len(gpu_ids)]
            print(f"Launching {case_name} on GPU {gpu_id}")

            cmd = [
                "python", "-c",
                f"""
import os
import sys
os.environ['CUDA_VISIBLE_DEVICES'] = '{gpu_id}'
os.environ['SPCONV_ALGO'] = 'native'

from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from trellis_inference.batch_generate_from_edit_images import generate_single_case

generate_single_case(
    Path('{img_path}'),
    '{case_name}',
    Path('{args.output_dir}'),
    seed={args.seed}
)
"""
            ]

            proc = subprocess.Popen(cmd)
            processes.append((proc, case_name, gpu_id))

        # Wait for all processes
        print(f"\nWaiting for {len(processes)} processes to complete...\n")
        for proc, case_name, gpu_id in processes:
            proc.wait()
            if proc.returncode == 0:
                print(f"✓ {case_name} (GPU {gpu_id}) completed successfully")
            else:
                print(f"✗ {case_name} (GPU {gpu_id}) failed with code {proc.returncode}")

    else:
        # Sequential execution
        print("Running in sequential mode\n")
        for img_path, case_name in cases:
            generate_single_case(img_path, case_name, args.output_dir, args.seed)

    print(f"\n{'='*80}")
    print(f"All cases completed!")
    print(f"Results saved to: {args.output_dir}")
    print(f"{'='*80}")


if __name__ == "__main__":
    main()
