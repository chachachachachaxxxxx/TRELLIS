import numpy as np
import open3d as o3d
import trimesh
from pysdf import SDF


def sample_out_points(model_path, mask_path, num_points=20480):
    model = load_trimesh(model_path)
    mask = load_trimesh(mask_path)

    surface_pts = trimesh.sample.sample_surface(model, num_points)[0]
    sdf = load_and_create_sdf(mask)
    inside_mask, distances = check_points_with_sdf(surface_pts, sdf)
    outside_pts = surface_pts[~inside_mask]

    return outside_pts


def load_trimesh(model_path):
    scene = trimesh.load(model_path, force="mesh", process=False)

    if isinstance(scene, trimesh.Trimesh):
        mesh = scene
    elif isinstance(scene, trimesh.scene.Scene):
        mesh = trimesh.Trimesh()
        for obj in scene.geometry.values():
            mesh = trimesh.util.concatenate([mesh, obj])

    return mesh


def load_and_create_sdf(trimesh_mesh):
    sdf = SDF(trimesh_mesh.vertices, trimesh_mesh.faces)
    return sdf


def check_points_with_sdf(points, sdf):
    distances = sdf(points)
    inside_mask = distances > 0  # NOTE that inside = sdf > 0
    return inside_mask, distances


def sample_uniform_points(min_bounds=-0.5, max_bounds=0.5, num_points=4096):
    points = np.random.uniform(min_bounds, max_bounds, (num_points, 3))
    return points


def save_colored_point_cloud(points, inside_mask, output_path):
    pcd = o3d.geometry.PointCloud()
    pcd.points = o3d.utility.Vector3dVector(points)

    # green inside, red outside
    colors = np.zeros((len(points), 3))
    colors[inside_mask] = [0, 1, 0]  # green
    colors[~inside_mask] = [1, 0, 0]  # red

    pcd.colors = o3d.utility.Vector3dVector(colors)

    o3d.io.write_point_cloud(output_path, pcd)


if __name__ == "__main__":
    source_model = "GSO/3D_Dollhouse_Happy_Brother/source_model/model.glb"
    mask_model = "GSO/3D_Dollhouse_Happy_Brother/prompt_1/3d_edit_region.glb"

    mask_mesh = trimesh.load(mask_model, process=False)

    sampled_points = sample_uniform_points(mask_mesh, num_points=4096)  # [N, 3]

    ### NOTE: key!!!
    sdf = load_and_create_sdf(mask_model)
    inside_mask, distances = check_points_with_sdf(
        sampled_points, sdf
    )  # [N,], inside if `inside_mask` is True
    ###

    output_path = "mask_sampled_points.ply"
    save_colored_point_cloud(sampled_points, inside_mask, output_path)
