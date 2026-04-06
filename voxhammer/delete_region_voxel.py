import os
import bpy

from voxhammer.util_voxel_filtering import process_voxels_with_improved_filtering


def glb_to_ply(input_glb_path, input_ply_path):
    if not os.path.exists(input_glb_path):
        raise FileNotFoundError(f"Input GLB file not found: {input_glb_path}")
    bpy.ops.object.select_all(action="SELECT")
    bpy.ops.object.delete(use_global=False)
    bpy.context.scene.render.engine = "CYCLES"
    bpy.ops.import_scene.gltf(filepath=input_glb_path)
    bpy.ops.wm.ply_export(filepath=input_ply_path, export_normals=True, ascii_format=True)

def process_delete_ply(input_glb_path, render_dir, filter_method="volume", voxel_size=1/64):
    preset_voxel_path = "assets/preset/preset_grid64.ply"
    input_ply_path = os.path.join(render_dir, "mesh_delete.ply")
    output_ply_path = os.path.join(render_dir, "voxels_delete.ply")
    glb_to_ply(input_glb_path, input_ply_path)
    outside_voxel_points, additional_info = process_voxels_with_improved_filtering(
        preset_voxel_path,
        input_ply_path,
        output_ply_path,
        method=filter_method,
        voxel_size=voxel_size,
        inside=True,
    )

if __name__ == "__main__":
    render_dir = "/render/path"
    input_glb_path = "/path/to/your/input.glb"
    process_delete_ply(input_glb_path, render_dir)
