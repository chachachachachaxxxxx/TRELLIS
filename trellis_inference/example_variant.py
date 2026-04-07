import os
from pathlib import Path
# os.environ['ATTN_BACKEND'] = 'xformers'   # Can be 'flash-attn' or 'xformers', default is 'flash-attn'
os.environ['SPCONV_ALGO'] = 'native'        # Can be 'native' or 'auto', default is 'auto'.
                                            # 'auto' is faster but will do benchmarking at the beginning.
                                            # Recommended to set to 'native' if run only once.

import imageio
import numpy as np
import open3d as o3d
from trellis.pipelines import TrellisTextTo3DPipeline
from trellis.utils import render_utils
from output_layout import build_output_layout

# Load a pipeline from a model folder or a Hugging Face model hub.
pipeline = TrellisTextTo3DPipeline.from_pretrained("microsoft/TRELLIS-text-xlarge")
pipeline.cuda()

# Load mesh to make variants
mesh_path = Path("assets/T.ply")
base_mesh = o3d.io.read_triangle_mesh(str(mesh_path))
sample_name = f"{mesh_path.stem.lower()}_variant"
output_dir = build_output_layout("variant_to_3d", sample_name).case_dir
output_dir.mkdir(parents=True, exist_ok=True)

# Run the pipeline
outputs = pipeline.run_variant(
    base_mesh,
    "Rugged, metallic texture with orange and white paint finish, suggesting a durable, industrial feel.",
    seed=1,
    # Optional parameters
    # slat_sampler_params={
    #     "steps": 12,
    #     "cfg_strength": 7.5,
    # },
)
# outputs is a dictionary containing generated 3D assets in different formats:
# - outputs['gaussian']: a list of 3D Gaussians
# - outputs['radiance_field']: a list of radiance fields
# - outputs['mesh']: a list of meshes

# Render the outputs
video_gs = render_utils.render_video(outputs['gaussian'][0])['color']
video_mesh = render_utils.render_video(outputs['mesh'][0])['normal']
video = [np.concatenate([frame_gs, frame_mesh], axis=1) for frame_gs, frame_mesh in zip(video_gs, video_mesh)]
imageio.mimsave(output_dir / "sample_variant.mp4", video, fps=30)

print(f"Saved outputs to: {output_dir}")
