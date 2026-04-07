"""
Generate a reference image with FLUX.1 Krea [dev], then use that image as
the input of TRELLIS-image-large.

Before running:
1. Accept the FLUX.1 Krea [dev] license on Hugging Face.
2. Login locally with `huggingface-cli login`.
3. Install the extra text-to-image dependencies:
   `pip install -U diffusers accelerate sentencepiece`
"""

import argparse
import gc
import os
from pathlib import Path

# os.environ["ATTN_BACKEND"] = "xformers"
os.environ["SPCONV_ALGO"] = "native"

import imageio
import torch

from trellis.pipelines import TrellisImageTo3DPipeline
from trellis.utils import postprocessing_utils, render_utils


DEFAULT_FLUX_MODEL = "black-forest-labs/FLUX.1-Krea-dev"
DEFAULT_TRELLIS_MODEL = "microsoft/TRELLIS-image-large"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Generate a FLUX.1 Krea [dev] reference image and feed it into TRELLIS."
    )
    parser.add_argument("prompt", type=str, help="Text prompt for FLUX.1 Krea [dev].")
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("outputs/flux_krea_to_3d/strawberries"),
        help="Directory to save the reference image and 3D outputs.",
    )
    parser.add_argument("--flux-model", type=str, default=DEFAULT_FLUX_MODEL)
    parser.add_argument("--trellis-model", type=str, default=DEFAULT_TRELLIS_MODEL)
    parser.add_argument("--image-seed", type=int, default=0)
    parser.add_argument("--trellis-seed", type=int, default=None)
    parser.add_argument("--flux-steps", type=int, default=50)
    parser.add_argument("--flux-guidance", type=float, default=4.5)
    parser.add_argument("--flux-height", type=int, default=1024)
    parser.add_argument("--flux-width", type=int, default=1024)
    parser.add_argument("--ss-steps", type=int, default=12)
    parser.add_argument("--ss-cfg", type=float, default=7.5)
    parser.add_argument("--slat-steps", type=int, default=12)
    parser.add_argument("--slat-cfg", type=float, default=3.0)
    return parser.parse_args()


def cleanup(*objects) -> None:
    for obj in objects:
        del obj
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()


def resolve_flux_dtype() -> torch.dtype:
    if torch.cuda.is_available() and getattr(torch.cuda, "is_bf16_supported", lambda: False)():
        return torch.bfloat16
    return torch.float16


def load_flux_pipeline(model_name: str):
    try:
        from diffusers import FluxPipeline
    except ImportError as exc:
        raise ImportError(
            "Missing optional dependencies for FLUX. Install them with "
            "`pip install -U diffusers accelerate sentencepiece`."
        ) from exc

    try:
        pipeline = FluxPipeline.from_pretrained(
            model_name,
            torch_dtype=resolve_flux_dtype(),
        )
    except Exception as exc:
        raise RuntimeError(
            f"Failed to load {model_name}. Make sure you accepted the model license "
            "on Hugging Face and logged in with `huggingface-cli login`."
        ) from exc

    pipeline.enable_model_cpu_offload()
    return pipeline


def save_outputs(output_dir: Path, reference_image, processed_image, outputs) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)
    reference_image.save(output_dir / "reference.png")
    processed_image.save(output_dir / "reference_preprocessed.png")

    video = render_utils.render_video(outputs["gaussian"][0])["color"]
    imageio.mimsave(output_dir / "sample_gs.mp4", video, fps=30)

    video = render_utils.render_video(outputs["radiance_field"][0])["color"]
    imageio.mimsave(output_dir / "sample_rf.mp4", video, fps=30)

    video = render_utils.render_video(outputs["mesh"][0])["normal"]
    imageio.mimsave(output_dir / "sample_mesh.mp4", video, fps=30)

    glb = postprocessing_utils.to_glb(
        outputs["gaussian"][0],
        outputs["mesh"][0],
        simplify=0.95,
        texture_size=1024,
    )
    glb.export(output_dir / "sample.glb")

    outputs["gaussian"][0].save_ply(output_dir / "sample.ply")


def main() -> None:
    args = parse_args()

    if not torch.cuda.is_available():
        raise RuntimeError("This script requires CUDA because TRELLIS image-to-3D inference runs on GPU.")

    args.output_dir.mkdir(parents=True, exist_ok=True)
    (args.output_dir / "prompt.txt").write_text(args.prompt + "\n", encoding="utf-8")

    print("Stage 1/2: generating a reference image with FLUX.1 Krea [dev]...")
    flux_pipeline = load_flux_pipeline(args.flux_model)
    generator = torch.Generator(device="cpu").manual_seed(args.image_seed)
    reference_image = flux_pipeline(
        prompt=args.prompt,
        height=args.flux_height,
        width=args.flux_width,
        guidance_scale=args.flux_guidance,
        num_inference_steps=args.flux_steps,
        generator=generator,
    ).images[0]
    cleanup(flux_pipeline)

    print("Stage 2/2: converting the reference image to 3D with TRELLIS...")
    trellis_pipeline = TrellisImageTo3DPipeline.from_pretrained(args.trellis_model)
    trellis_pipeline.cuda()

    processed_image = trellis_pipeline.preprocess_image(reference_image)
    outputs = trellis_pipeline.run(
        processed_image,
        seed=args.image_seed if args.trellis_seed is None else args.trellis_seed,
        preprocess_image=False,
        sparse_structure_sampler_params={
            "steps": args.ss_steps,
            "cfg_strength": args.ss_cfg,
        },
        slat_sampler_params={
            "steps": args.slat_steps,
            "cfg_strength": args.slat_cfg,
        },
    )
    cleanup(trellis_pipeline)

    save_outputs(args.output_dir, reference_image, processed_image, outputs)
    print(f"Done. Outputs saved to: {args.output_dir}")


if __name__ == "__main__":
    main()
