from __future__ import annotations

from dataclasses import dataclass
import importlib.util

import numpy as np
import torch
from PIL import Image

from trellis_edit.common import release_cuda_memory
from trellis_edit.preprocess import save_preprocessed_inputs
from trellis_edit.preprocess.image_alignment import (
    PROC_IMAGE_SIZE,
    CropContext,
    PreparedInputs,
    composite_rgb_from_rgba,
    extract_foreground_rgba,
    prepare_rgb_for_rembg,
    scale_image,
)
from trellis_edit.preprocess.foreground_scale_selection import (
    build_square_crop_box,
    compute_active_bbox,
    compute_sparse_structure_metrics,
    normalize_sparse_coords,
    select_crop_scale_from_linear_fit,
    select_crop_scale_from_probes,
)
from trellis_edit.preprocess.mask_utils import build_auto_mask, build_blank_mask, binarize_mask_image, extract_mask_channel
from trellis_edit.utils.voxel_mesh_converter import load_coords_from_file

from .artifacts import PreprocessArtifact
from .base import ExperimentContext, PreprocessPlugin
from .config import AdaptiveForegroundScaleConfig, MaskPolicy, PreprocessConfig


@dataclass(frozen=True)
class _PreparedForeground:
    image: Image.Image
    mask: np.ndarray


class SharedImagePreprocessPlugin(PreprocessPlugin):
    name = "shared_image"

    @staticmethod
    def _ensure_rembg_session(pipeline):
        if importlib.util.find_spec("rembg") is None:
            raise RuntimeError(
                "rembg is not installed in the current environment. "
                "VoxHammer-style preprocessing requires rembg."
            )
        import rembg

        if getattr(pipeline, "rembg_session", None) is None:
            pipeline.rembg_session = rembg.new_session("u2net")
        return rembg, pipeline.rembg_session

    @staticmethod
    def _voxhammer_image_rgb(image: Image.Image) -> Image.Image:
        return prepare_rgb_for_rembg(image)

    def _prepare_voxhammer_foreground(
        self,
        image: Image.Image,
        scale: float,
        pipeline,
    ) -> _PreparedForeground:
        rembg, session = self._ensure_rembg_session(pipeline)
        scaled = scale_image(self._voxhammer_image_rgb(image), scale, Image.Resampling.LANCZOS)
        rgba = rembg.remove(scaled, session=session)
        if not isinstance(rgba, Image.Image):
            raise RuntimeError("rembg did not return a PIL image as expected.")
        rgba = rgba.convert("RGBA")
        alpha_mask = np.asarray(rgba)[:, :, 3] > int(0.8 * 255)
        return _PreparedForeground(image=rgba, mask=alpha_mask)

    @staticmethod
    def _render_voxhammer_output(image: Image.Image) -> Image.Image:
        rgba = np.asarray(image.convert("RGBA")).astype(np.float32) / 255.0
        rgb = rgba[:, :, :3] * rgba[:, :, 3:4]
        return Image.fromarray((rgb * 255).astype(np.uint8), mode="RGB")

    def _resolve_mask(self, context: ExperimentContext, config: PreprocessConfig) -> Image.Image:
        source_image = context.loaded_inputs.source_image
        edit_image = context.loaded_inputs.edit_image
        mask_image = context.loaded_inputs.mask_image

        policy: MaskPolicy = config.mask_policy
        if policy == "provided":
            if mask_image is None:
                raise RuntimeError("mask_policy='provided' requires inputs.mask_image")
            return mask_image
        if policy == "auto_diff":
            if source_image is None:
                raise RuntimeError("mask_policy='auto_diff' requires inputs.source_image")
            return build_auto_mask(
                source_image=source_image,
                edit_image=edit_image,
                threshold=config.auto_mask_diff_threshold,
                max_filter=config.auto_mask_max_filter,
            )
        if policy == "blank":
            return build_blank_mask(source_image.size)
        raise RuntimeError(f"Unknown mask policy: {policy}")

    def _prepare_foreground(
        self,
        image: Image.Image,
        scale: float,
        pipeline,
    ) -> _PreparedForeground:
        rgba = extract_foreground_rgba(image, pipeline, scale)
        alpha_mask = np.asarray(rgba)[:, :, 3] > int(0.8 * 255)
        return _PreparedForeground(image=rgba, mask=alpha_mask)

    @staticmethod
    def _resolve_active_mask(
        source_fg: _PreparedForeground,
        edit_fg: _PreparedForeground,
        mask_binary: np.ndarray,
        config: PreprocessConfig,
    ) -> np.ndarray:
        if config.crop_policy == "union_crop":
            if config.style == "voxhammer":
                return source_fg.mask | edit_fg.mask
            return source_fg.mask | edit_fg.mask | mask_binary
        if config.crop_policy == "mask_crop":
            return mask_binary
        raise RuntimeError(f"Unknown crop policy: {config.crop_policy}")

    @staticmethod
    def _build_crop_context_from_scale(active: np.ndarray, scale: float, crop_scale: float) -> CropContext:
        active_bbox = compute_active_bbox(active)
        return CropContext(scale=scale, bbox=build_square_crop_box(active_bbox, crop_scale))

    @staticmethod
    def _build_voxhammer_crop_context_from_scale(active: np.ndarray, scale: float, crop_scale: float) -> CropContext:
        bbox_pixels = np.argwhere(active)
        if bbox_pixels.size == 0:
            return CropContext(scale=scale, bbox=(0, 0, 1, 1))

        x_min = float(np.min(bbox_pixels[:, 1]))
        y_min = float(np.min(bbox_pixels[:, 0]))
        x_max = float(np.max(bbox_pixels[:, 1]))
        y_max = float(np.max(bbox_pixels[:, 0]))
        center_x = (x_min + x_max) / 2.0
        center_y = (y_min + y_max) / 2.0
        size = max(x_max - x_min, y_max - y_min)
        size = max(int(size * float(crop_scale)), 1)
        bbox = (
            center_x - size // 2,
            center_y - size // 2,
            center_x + size // 2,
            center_y + size // 2,
        )
        return CropContext(scale=scale, bbox=bbox)

    @staticmethod
    def _resolve_probe_sampler_params(context: ExperimentContext) -> tuple[int, dict]:
        seed = int(context.config.runtime.seed)
        ss_spec = context.config.ss
        if ss_spec is None:
            return seed, {}
        sampler_config = getattr(ss_spec, "sampler", None)
        if sampler_config is None:
            return seed, {}

        sampler_params: dict[str, float | int] = {}
        if sampler_config.steps is not None:
            sampler_params["steps"] = int(sampler_config.steps)
        if sampler_config.cfg_strength is not None:
            sampler_params["cfg_strength"] = float(sampler_config.cfg_strength)
        if sampler_config.rescale_t is not None:
            sampler_params["rescale_t"] = float(sampler_config.rescale_t)
        return seed, sampler_params

    def _sample_probe_bbox_volume_ratio(
        self,
        context: ExperimentContext,
        image: Image.Image,
        sampler_params: dict,
        resolution: int,
        seed: int,
    ) -> dict:
        pipeline = context.pipeline
        torch.manual_seed(seed)
        np.random.seed(seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(seed)

        cond = pipeline.get_cond([image])
        coords = pipeline.sample_sparse_structure(
            cond,
            num_samples=1,
            sampler_params=sampler_params,
        )
        coords_xyz = normalize_sparse_coords(coords, resolution=resolution)
        metrics = compute_sparse_structure_metrics(coords_xyz, resolution=resolution)
        release_cuda_memory()
        return metrics

    @staticmethod
    def _select_global_linear_crop_scale(
        adaptive: AdaptiveForegroundScaleConfig,
        target_bbox_volume_ratio: float,
    ) -> tuple[float, dict]:
        selected_scale, selection_meta = select_crop_scale_from_linear_fit(
            target_bbox_volume_ratio=target_bbox_volume_ratio,
            slope=adaptive.linear_bbox_slope,
            intercept=adaptive.linear_bbox_intercept,
            fallback_scale=adaptive.fallback_scale,
            min_scale=adaptive.min_scale,
            max_scale=adaptive.max_scale,
        )
        return selected_scale, {
            **selection_meta,
            "linear_fit_source_run": (
                "exp/foreground_crop_sweep_ss_size/results/full_300x5_seed1_default"
            ),
            "linear_fit_filter_rule": (
                "exclude cases with case-wise Spearman(crop_scale,bbox_volume_ratio) >= 0 "
                "or any consecutive bbox_volume_ratio increase >= 0.15"
            ),
            "linear_fit_kept_cases": 264,
            "linear_fit_removed_cases": 36,
        }

    def _select_adaptive_crop_scale(
        self,
        context: ExperimentContext,
        config: PreprocessConfig,
        *,
        active: np.ndarray,
        source_fg: _PreparedForeground,
        edit_fg: _PreparedForeground,
        downscale: float,
    ) -> tuple[float, dict]:
        adaptive: AdaptiveForegroundScaleConfig = config.adaptive_foreground_scale
        default_scale = float(adaptive.fallback_scale)

        if not adaptive.enabled:
            return default_scale, {
                "enabled": False,
                "strategy": adaptive.strategy,
                "mode": "fixed_default",
                "selected_scale": default_scale,
            }
        if config.crop_policy != "union_crop":
            return default_scale, {
                "enabled": False,
                "strategy": adaptive.strategy,
                "mode": "fixed_default",
                "reason": "adaptive_selection_only_supports_union_crop",
                "selected_scale": default_scale,
            }
        if context.config.run_mode not in {"ss", "full"}:
            return default_scale, {
                "enabled": False,
                "strategy": adaptive.strategy,
                "mode": "fixed_default",
                "reason": "adaptive_selection_requires_stage1_run_mode",
                "selected_scale": default_scale,
            }
        source_voxels_path = context.config.inputs.source_voxels
        if source_voxels_path is None:
            return default_scale, {
                "enabled": False,
                "strategy": adaptive.strategy,
                "mode": "fixed_default",
                "reason": "source_voxels_missing",
                "selected_scale": default_scale,
            }
        if len(adaptive.probe_scales) != 2:
            raise RuntimeError(
                "preprocess.adaptive_foreground_scale.probe_scales must contain exactly two values"
            )

        pipeline = context.pipeline
        resolution = int(getattr(pipeline, "sparse_structure_sampler_params", {}).get("grid_size", 64))
        source_coords = load_coords_from_file(source_voxels_path, pipeline.device)
        source_coords_xyz = normalize_sparse_coords(source_coords, resolution=resolution)
        source_metrics = compute_sparse_structure_metrics(source_coords_xyz, resolution=resolution)
        target_bbox_volume_ratio = float(source_metrics["bbox_volume_ratio"])

        if adaptive.strategy == "global_linear":
            selected_scale, selection_meta = self._select_global_linear_crop_scale(
                adaptive,
                target_bbox_volume_ratio,
            )
            return selected_scale, {
                "enabled": True,
                "strategy": adaptive.strategy,
                "resolution": resolution,
                "source_voxels_path": str(source_voxels_path),
                "source_metrics": source_metrics,
                **selection_meta,
            }

        if adaptive.strategy != "two_probe":
            raise RuntimeError(
                f"Unknown preprocess.adaptive_foreground_scale.strategy: {adaptive.strategy}"
            )

        seed, sampler_params = self._resolve_probe_sampler_params(context)
        probe_results: list[dict] = []
        for probe_scale in sorted(float(value) for value in adaptive.probe_scales):
            crop_context = self._build_crop_context_from_scale(active, downscale, probe_scale)
            probe_image = self._render_output_image(
                source_fg.image.crop(crop_context.bbox).resize((PROC_IMAGE_SIZE, PROC_IMAGE_SIZE), Image.Resampling.LANCZOS)
            )
            probe_metrics = self._sample_probe_bbox_volume_ratio(
                context=context,
                image=probe_image,
                sampler_params=sampler_params,
                resolution=resolution,
                seed=seed,
            )
            probe_results.append(
                {
                    "crop_scale": float(probe_scale),
                    "crop_bbox_xyxy": [int(value) for value in crop_context.bbox],
                    **probe_metrics,
                }
            )

        selected_scale, selection_meta = select_crop_scale_from_probes(
            target_bbox_volume_ratio=target_bbox_volume_ratio,
            probe_results=probe_results,
            fallback_scale=adaptive.fallback_scale,
            min_scale=adaptive.min_scale,
            max_scale=adaptive.max_scale,
            min_bbox_volume_delta=adaptive.min_bbox_volume_delta,
        )
        return selected_scale, {
            "enabled": True,
            "strategy": adaptive.strategy,
            "probe_image": "source",
            "resolution": resolution,
            "source_voxels_path": str(source_voxels_path),
            "source_metrics": source_metrics,
            "probe_results": probe_results,
            **selection_meta,
        }

    def _build_crop_context(
        self,
        context: ExperimentContext,
        source_fg: _PreparedForeground,
        edit_fg: _PreparedForeground,
        mask_image: Image.Image,
        original_size: tuple[int, int],
        config: PreprocessConfig,
        scale: float,
    ) -> tuple[CropContext, dict]:
        if config.crop_policy == "disabled":
            return CropContext(scale=1.0, bbox=(0, 0, original_size[0], original_size[1])), {
                "enabled": False,
                "strategy": "disabled",
                "mode": "disabled",
                "selected_scale": 1.0,
            }

        scaled_mask = scale_image(
            extract_mask_channel(mask_image),
            scale,
            Image.Resampling.NEAREST,
        )
        mask_binary = np.asarray(scaled_mask) > int(config.mask_threshold)
        active = self._resolve_active_mask(source_fg, edit_fg, mask_binary, config)
        selected_crop_scale, selection_meta = self._select_adaptive_crop_scale(
            context,
            config,
            active=active,
            source_fg=source_fg,
            edit_fg=edit_fg,
            downscale=scale,
        )
        if config.style == "voxhammer":
            crop_context = self._build_voxhammer_crop_context_from_scale(active, scale, selected_crop_scale)
        else:
            crop_context = self._build_crop_context_from_scale(active, scale, selected_crop_scale)
        selection_meta = {
            **selection_meta,
            "selected_crop_bbox_xyxy": [int(value) for value in crop_context.bbox],
        }
        return crop_context, selection_meta

    @staticmethod
    def _render_output_image(image: Image.Image) -> Image.Image:
        if image.mode == "RGBA":
            return composite_rgb_from_rgba(image)
        return image.convert("RGB")

    def run(
        self,
        context: ExperimentContext,
        config: PreprocessConfig,
    ) -> PreprocessArtifact:
        source_image = context.loaded_inputs.source_image
        edit_image = context.loaded_inputs.edit_image
        source_reference = source_image or edit_image

        if source_image is not None and source_image.size != edit_image.size:
            raise RuntimeError(
                f"source_image and edit_image must have the same size, got {source_image.size} and {edit_image.size}"
            )

        mask_image = self._resolve_mask(context, config)

        if config.crop_policy == "disabled":
            source = self._render_output_image(
                source_reference.resize((PROC_IMAGE_SIZE, PROC_IMAGE_SIZE), Image.Resampling.LANCZOS)
            )
            edit = self._render_output_image(
                edit_image.resize((PROC_IMAGE_SIZE, PROC_IMAGE_SIZE), Image.Resampling.LANCZOS)
            )
            mask = binarize_mask_image(
                extract_mask_channel(mask_image).resize((PROC_IMAGE_SIZE, PROC_IMAGE_SIZE), Image.Resampling.NEAREST),
                threshold=config.mask_threshold,
            )
            prepared = PreparedInputs(
                source=source,
                edit=edit,
                mask=mask,
                meta={
                    "style": config.style,
                    "crop_policy": config.crop_policy,
                    "mask_policy": config.mask_policy,
                    "mask_threshold": config.mask_threshold,
                    "proc_size": [PROC_IMAGE_SIZE, PROC_IMAGE_SIZE],
                    "scale": 1.0,
                    "crop_scale": 1.0,
                    "crop_bbox_xyxy": [0, 0, int(source_reference.width), int(source_reference.height)],
                    "foreground_scale_selection": {
                        "enabled": False,
                        "mode": "disabled",
                        "selected_scale": 1.0,
                    },
                },
            )
            return PreprocessArtifact(
                prepared_inputs=prepared,
                artifact_paths=save_preprocessed_inputs(context.preprocess_dir, prepared, include_mask=True),
            )

        scale = min(1.0, 1024.0 / float(max(edit_image.size)))
        if config.style == "voxhammer":
            if source_image is None:
                raise RuntimeError("preprocess.style='voxhammer' requires inputs.source_image")
            source_fg = self._prepare_voxhammer_foreground(source_reference, scale, context.pipeline)
            edit_fg = self._prepare_voxhammer_foreground(edit_image, scale, context.pipeline)
        else:
            source_fg = self._prepare_foreground(source_reference, scale, context.pipeline)
            edit_fg = self._prepare_foreground(edit_image, scale, context.pipeline)
        crop_context, crop_meta = self._build_crop_context(
            context,
            source_fg=source_fg,
            edit_fg=edit_fg,
            mask_image=mask_image,
            original_size=source_reference.size,
            config=config,
            scale=scale,
        )

        mask_resample = Image.Resampling.LANCZOS if config.style == "voxhammer" else Image.Resampling.NEAREST
        scaled_mask = scale_image(
            extract_mask_channel(mask_image),
            crop_context.scale,
            mask_resample,
        )

        crop_box = crop_context.bbox
        if config.style == "voxhammer":
            source = self._render_voxhammer_output(
                source_fg.image.crop(crop_box).resize((PROC_IMAGE_SIZE, PROC_IMAGE_SIZE), Image.Resampling.LANCZOS)
            )
            edit = self._render_voxhammer_output(
                edit_fg.image.crop(crop_box).resize((PROC_IMAGE_SIZE, PROC_IMAGE_SIZE), Image.Resampling.LANCZOS)
            )
            mask = scaled_mask.crop(crop_box).resize((PROC_IMAGE_SIZE, PROC_IMAGE_SIZE), Image.Resampling.LANCZOS)
        else:
            source = self._render_output_image(
                source_fg.image.crop(crop_box).resize((PROC_IMAGE_SIZE, PROC_IMAGE_SIZE), Image.Resampling.LANCZOS)
            )
            edit = self._render_output_image(
                edit_fg.image.crop(crop_box).resize((PROC_IMAGE_SIZE, PROC_IMAGE_SIZE), Image.Resampling.LANCZOS)
            )
            mask = binarize_mask_image(
                scaled_mask.crop(crop_box).resize((PROC_IMAGE_SIZE, PROC_IMAGE_SIZE), Image.Resampling.NEAREST),
                threshold=config.mask_threshold,
            )

        prepared = PreparedInputs(
            source=source,
            edit=edit,
            mask=mask,
            meta={
                "style": config.style,
                "crop_policy": config.crop_policy,
                "mask_policy": config.mask_policy,
                "mask_threshold": config.mask_threshold,
                "proc_size": [PROC_IMAGE_SIZE, PROC_IMAGE_SIZE],
                "scale": float(crop_context.scale),
                "crop_scale": float(crop_meta["selected_scale"]),
                "crop_bbox_xyxy": [int(value) for value in crop_context.bbox],
                "foreground_scale_selection": crop_meta,
            },
        )
        return PreprocessArtifact(
            prepared_inputs=prepared,
            artifact_paths=save_preprocessed_inputs(context.preprocess_dir, prepared, include_mask=True),
        )
