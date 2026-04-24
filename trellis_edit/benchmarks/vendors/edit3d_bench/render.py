import argparse
import contextlib
import os
import sys
import time
from glob import glob
from pathlib import Path

import imageio.v2 as imageio
import numpy as np


def _bootstrap_bpyrenderer():
    candidates = []

    env_path = os.environ.get("BPY_RENDERER_SRC")
    if env_path:
        candidates.append(Path(env_path).expanduser())

    this_file = Path(__file__).resolve()
    for parent in this_file.parents:
        candidates.append(parent / "bpy-renderer" / "src")

    for candidate in candidates:
        if candidate.exists() and str(candidate) not in sys.path:
            sys.path.insert(0, str(candidate))
            break


_bootstrap_bpyrenderer()

try:
    import bpy
except ImportError:
    bpy = None

from bpyrenderer import SceneManager
from bpyrenderer.camera import add_camera
from bpyrenderer.camera.layout import get_camera_positions_on_sphere
from bpyrenderer.engine import init_render_engine
from bpyrenderer.environment import set_env_map
from bpyrenderer.importer import load_file
from bpyrenderer.render_output import enable_color_output


ENV_MAP_PATH = str((Path(__file__).resolve().parent / "assets" / "brown_photostudio_02_1k.exr").resolve())
IMAGE_ELEVATIONS = [15, 30]
IMAGE_AZIMUTHS = [item - 90 for item in [0, 45, 90, 135, 180, 225, 270, 315]]
EXPECTED_IMAGE_COUNT = len(IMAGE_ELEVATIONS) * len(IMAGE_AZIMUTHS)
BACKEND_PRIORITY = ["OPTIX", "CUDA", "HIP", "METAL", "ONEAPI"]


def configure_cycles_backend(requested_backend: str) -> str:
    if bpy is None:
        return "unavailable"

    if requested_backend == "CPU":
        bpy.context.scene.cycles.device = "CPU"
        return "CPU"

    cycles_addon = bpy.context.preferences.addons.get("cycles")
    if cycles_addon is None:
        return "missing"

    preferences = cycles_addon.preferences
    refresh_devices = getattr(preferences, "refresh_devices", None)
    if refresh_devices is None:
        refresh_devices = getattr(preferences, "get_devices", None)
    if callable(refresh_devices):
        refresh_devices()

    if requested_backend == "AUTO":
        backends = BACKEND_PRIORITY
    else:
        backends = [requested_backend]

    for backend in backends:
        try:
            preferences.compute_device_type = backend
            if callable(refresh_devices):
                refresh_devices()
        except Exception:
            continue

        matched_devices = 0
        for device in preferences.devices:
            should_use = getattr(device, "type", "").upper() == backend
            try:
                device.use = should_use
            except Exception:
                pass
            matched_devices += int(should_use)

        if matched_devices > 0:
            bpy.context.scene.cycles.device = "GPU"
            return backend

    # Last fallback: enable any visible non-CPU device.
    visible_gpu_count = 0
    for device in preferences.devices:
        should_use = getattr(device, "type", "").upper() != "CPU"
        try:
            device.use = should_use
        except Exception:
            pass
        visible_gpu_count += int(should_use)

    if visible_gpu_count > 0:
        bpy.context.scene.cycles.device = "GPU"
        return "VISIBLE_GPUS"

    bpy.context.scene.cycles.device = "CPU"
    return "CPU"


def init_engine(engine: str, samples: int, cycles_backend: str) -> None:
    init_render_engine(engine, render_samples=samples)

    if bpy is None:
        return

    if hasattr(bpy.context.scene.render, "use_persistent_data"):
        bpy.context.scene.render.use_persistent_data = True

    if engine == "CYCLES":
        resolved_backend = configure_cycles_backend(cycles_backend)
        print(f"Cycles backend: {resolved_backend}")


@contextlib.contextmanager
def suppress_blender_output(enabled: bool):
    if not enabled:
        yield
        return

    sys.stdout.flush()
    sys.stderr.flush()

    stdout_fd = os.dup(1)
    stderr_fd = os.dup(2)

    try:
        with open(os.devnull, "w") as devnull:
            os.dup2(devnull.fileno(), 1)
            os.dup2(devnull.fileno(), 2)
            yield
    finally:
        os.dup2(stdout_fd, 1)
        os.dup2(stderr_fd, 2)
        os.close(stdout_fd)
        os.close(stderr_fd)


def remove_camera_objects() -> None:
    if bpy is None:
        return

    bpy.context.scene.camera = None
    for obj in list(bpy.data.objects):
        if obj.type == "CAMERA":
            bpy.data.objects.remove(obj, do_unlink=True)


def reset_render_pass(scene_manager: SceneManager) -> None:
    remove_camera_objects()
    scene_manager.clear(clear_objects=False, clear_nodes=True, reset_keyframes=True)


def clear_render_sequence(output_dir: str) -> None:
    for file_path in glob(os.path.join(output_dir, "render_*.png")):
        os.remove(file_path)


def add_camera_path(camera_mats) -> None:
    for i, camera_mat in enumerate(camera_mats):
        add_camera(camera_mat, "PERSP", add_frame=i < len(camera_mats) - 1)


def prepare_scene(input_model: str, engine: str, samples: int, cycles_backend: str):
    init_engine(engine, samples, cycles_backend)

    scene_manager = SceneManager()
    scene_manager.clear(reset_keyframes=True)

    load_file(input_model)
    scene_manager.clear_normal_map()
    scene_manager.set_material_transparency(False)
    scene_manager.set_materials_opaque()
    set_env_map(ENV_MAP_PATH)

    return scene_manager


def render_images(
    scene_manager: SceneManager,
    output_dir: str,
    width: int,
    height: int,
    quiet_blender: bool,
):
    reset_render_pass(scene_manager)
    clear_render_sequence(output_dir)

    _, cam_mats, _, _ = get_camera_positions_on_sphere(
        center=(0, 0, 0),
        radius=1.8,
        elevations=IMAGE_ELEVATIONS,
        azimuths=IMAGE_AZIMUTHS,
    )
    add_camera_path(cam_mats)

    enable_color_output(
        width,
        height,
        output_dir,
        file_format="PNG",
        mode="IMAGE",
        film_transparent=True,
    )
    with suppress_blender_output(enabled=quiet_blender):
        scene_manager.render()


def render_video_frames(
    scene_manager: SceneManager,
    output_dir: str,
    width: int,
    height: int,
    num_frames: int,
    quiet_blender: bool,
):
    reset_render_pass(scene_manager)
    clear_render_sequence(output_dir)

    _, cam_mats, _, _ = get_camera_positions_on_sphere(
        center=(0, 0, 0),
        radius=1.8,
        elevations=[15],
        num_camera_per_layer=num_frames,
        azimuth_offset=-90,
    )
    add_camera_path(cam_mats)

    enable_color_output(
        width,
        height,
        output_dir,
        file_format="PNG",
        mode="IMAGE",
        film_transparent=True,
    )
    with suppress_blender_output(enabled=quiet_blender):
        scene_manager.render()


def build_videos(
    output_dir: str,
    fps: int,
    save_video_mask: bool,
) -> None:
    render_files = sorted(glob(os.path.join(output_dir, "render_*.png")))
    if not render_files:
        return

    white_video_path = os.path.join(output_dir, "video_rgb.mp4")
    mask_video_path = os.path.join(output_dir, "video_mask.mp4")

    for stale_path in [white_video_path, mask_video_path]:
        if os.path.exists(stale_path):
            os.remove(stale_path)

    white_writer = imageio.get_writer(white_video_path, fps=fps)
    mask_writer = imageio.get_writer(mask_video_path, fps=fps) if save_video_mask else None

    try:
        for file_path in render_files:
            image = imageio.imread(file_path)
            frame_height, frame_width = image.shape[:2]

            if image.ndim == 2:
                image = np.repeat(image[..., None], 4, axis=2)
            elif image.shape[2] == 3:
                alpha = np.full((frame_height, frame_width, 1), 255, dtype=np.uint8)
                image = np.concatenate([image, alpha], axis=2)

            mask = image[:, :, 3]
            alpha = image[:, :, 3:4].astype(np.float32) / 255.0
            white_bg = np.full((frame_height, frame_width, 3), 255, dtype=np.uint8)
            white_image = image[:, :, :3].astype(np.float32) * alpha + white_bg * (1 - alpha)

            white_writer.append_data(white_image.astype(np.uint8))
            if mask_writer is not None:
                mask_writer.append_data(mask)

            os.remove(file_path)
    finally:
        white_writer.close()
        if mask_writer is not None:
            mask_writer.close()


def find_all_edit_glb_files(base_dir: str, file_name: str = "edit.glb"):
    pattern = os.path.join(base_dir, "**", file_name)
    return sorted(glob(pattern, recursive=True))


def create_output_dir(model_path: str, mode: str) -> str:
    model_dir = os.path.dirname(model_path)
    output_dir = os.path.join(model_dir, mode)
    os.makedirs(output_dir, exist_ok=True)
    return output_dir


def images_complete(output_dir: str) -> bool:
    return len(glob(os.path.join(output_dir, "render_*.png"))) == EXPECTED_IMAGE_COUNT


def videos_complete(output_dir: str, save_video_mask: bool) -> bool:
    required_files = [os.path.join(output_dir, "video_rgb.mp4")]
    if save_video_mask:
        required_files.append(os.path.join(output_dir, "video_mask.mp4"))
    return all(os.path.exists(file_path) for file_path in required_files)


def shard_models(all_models, num_shards: int, shard_id: int):
    if num_shards <= 1:
        return all_models
    return all_models[shard_id::num_shards]


def render_model(model_path: str, args) -> None:
    model_start = time.perf_counter()
    stage_times = {}

    images_dir = create_output_dir(model_path, mode="images")
    videos_dir = create_output_dir(model_path, mode="videos")

    need_images = (not args.skip_images) and not (args.skip_existing and images_complete(images_dir))
    need_videos = (not args.skip_video) and not (
        args.skip_existing and videos_complete(videos_dir, args.save_video_mask)
    )

    if not need_images and not need_videos:
        print(f"[SKIP] {model_path}")
        return

    t0 = time.perf_counter()
    scene_manager = prepare_scene(
        model_path,
        engine=args.engine,
        samples=args.samples,
        cycles_backend=args.cycles_backend,
    )
    stage_times["prepare_scene"] = time.perf_counter() - t0

    try:
        if need_images:
            t0 = time.perf_counter()
            render_images(
                scene_manager,
                images_dir,
                args.image_size,
                args.image_size,
                args.quiet_blender,
            )
            stage_times["render_images"] = time.perf_counter() - t0

        if need_videos:
            t0 = time.perf_counter()
            render_video_frames(
                scene_manager,
                videos_dir,
                args.video_size,
                args.video_size,
                args.video_frames,
                args.quiet_blender,
            )
            stage_times["render_video_frames"] = time.perf_counter() - t0

            t0 = time.perf_counter()
            build_videos(
                videos_dir,
                args.fps,
                args.save_video_mask,
            )
            stage_times["build_videos"] = time.perf_counter() - t0
    finally:
        scene_manager.clear(reset_keyframes=True)
        scene_manager.gc()

    total_time = time.perf_counter() - model_start
    timing_summary = ", ".join(
        f"{name}={value:.1f}s" for name, value in stage_times.items()
    )
    print(f"Timing: {timing_summary}, total={total_time:.1f}s")


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--base_dir", type=str, required=True)
    parser.add_argument(
        "--engine",
        type=str,
        default="CYCLES",
        choices=["CYCLES", "BLENDER_EEVEE"],
    )
    parser.add_argument("--samples", type=int, default=64)
    parser.add_argument("--image_size", type=int, default=1024)
    parser.add_argument("--video_size", type=int, default=1024)
    parser.add_argument("--video_frames", type=int, default=120)
    parser.add_argument("--fps", type=int, default=24)
    parser.add_argument(
        "--cycles_backend",
        type=str,
        default="AUTO",
        choices=["AUTO", "OPTIX", "CUDA", "HIP", "METAL", "ONEAPI", "CPU"],
    )
    parser.add_argument("--skip_existing", action="store_true")
    parser.add_argument("--skip_images", action="store_true")
    parser.add_argument("--skip_video", action="store_true")
    parser.add_argument("--save_video_mask", action="store_true")
    parser.add_argument("--quiet_blender", action="store_true")
    parser.add_argument("--num_shards", type=int, default=1)
    parser.add_argument("--shard_id", type=int, default=0)
    parser.add_argument("--limit", type=int, default=0)
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()

    if args.num_shards < 1:
        raise ValueError("--num_shards must be >= 1")
    if not 0 <= args.shard_id < args.num_shards:
        raise ValueError("--shard_id must satisfy 0 <= shard_id < num_shards")

    print("Loading models...")
    all_models = find_all_edit_glb_files(args.base_dir, file_name="edit.glb")
    print(f"Found {len(all_models)} models")

    if len(all_models) == 0:
        print("No models found, please check the path")
        raise SystemExit(1)

    worker_models = shard_models(all_models, args.num_shards, args.shard_id)
    if args.limit > 0:
        worker_models = worker_models[: args.limit]
    print(
        f"Shard {args.shard_id + 1}/{args.num_shards}: "
        f"processing {len(worker_models)} models"
    )

    failed_models = []
    succeeded = 0

    for idx, model_path in enumerate(worker_models, start=1):
        print(f"[{idx}/{len(worker_models)}] {model_path}")
        try:
            render_model(model_path, args)
            succeeded += 1
        except Exception as exc:
            print(f"Render failed: {model_path} - Error: {exc}")
            failed_models.append(model_path)

    print("\nRender completed!")
    print(f"Success: {succeeded}")
    print(f"Failed: {len(failed_models)}")
    print("Render results saved to images/videos folder under each model directory")
    print(f"Failed models: {failed_models}")
