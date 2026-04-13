from __future__ import annotations

import gc
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
import torch
from PIL import Image

from trellis_edit.common import save_outputs, write_json
from trellis_edit.inversion import invert_slat, invert_sparse_structure
from trellis_edit.inversion.uniedit_sampler import UniEditRFSolver
from trellis_edit.composable.p2p_common import P2PLatentBlendCommonMixin
from trellis_edit.preprocess.asset_3d import (
    coords_to_voxel,
    feats_to_slat,
    load_mask_glb_coords,
    ply_to_coords,
    project_sparse_terminal_noise,
)
from trellis_edit.samplers import LatentBlendFlowEulerGuidanceIntervalSampler
from trellis_edit.utils import save_mask_overlay_preview
from trellis_edit.utils.uniedit_utils import (
    build_sparse_replace_index_map,
    build_stage2_selector,
    compose_stage1_coords,
    coords3d_to_batched,
)
from trellis_edit.utils.voxel_mesh_converter import (
    coords_to_cubic_mesh,
    load_coords_from_file,
    save_coords_to_file,
    save_voxel_mesh,
)

from .artifacts import SLATArtifact, SSArtifact
from .base import ExperimentContext, SLATStagePlugin, SSStagePlugin
from .config import (
    P2PLatentBlendSLATConfig,
    P2PLatentBlendSSConfig,
    RuntimeConfig,
    SamplerOverrideConfig,
    UniEditSLATConfig,
    UniEditSSConfig,
)


def _sampler_params(config: SamplerOverrideConfig) -> dict[str, Any]:
    params: dict[str, Any] = {}
    if config.steps is not None:
        params["steps"] = config.steps
    if config.cfg_strength is not None:
        params["cfg_strength"] = config.cfg_strength
    if config.rescale_t is not None:
        params["rescale_t"] = config.rescale_t
    return params


def _prepared_images(preprocess) -> tuple[Image.Image, Image.Image, Image.Image]:
    prepared = preprocess.prepared_inputs
    return prepared.source, prepared.edit, prepared.mask


def _source_asset_dir(context: ExperimentContext) -> Path | None:
    for candidate in (
        context.config.inputs.source_voxels,
        context.config.inputs.source_features,
        context.config.inputs.edited_coords,
    ):
        if candidate is not None:
            return candidate.parent
    return None


def _release_cuda_memory() -> None:
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()


def _serialize_json(value: Any) -> Any:
    if isinstance(value, dict):
        return {str(key): _serialize_json(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [_serialize_json(item) for item in value]
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, torch.Tensor):
        return value.detach().cpu().tolist()
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, np.generic):
        return value.item()
    return value


def _write_metadata(path: Path, metadata: dict[str, Any]) -> Path:
    return write_json(path, _serialize_json(metadata))


def _save_ss_artifacts(
    artifact: SSArtifact,
    out_dir: Path,
    *,
    output_format: str,
    save_mesh: bool,
) -> dict[str, str]:
    artifact_paths: dict[str, str] = {}
    resolution = int(artifact.metadata.get("resolution", 64))

    if output_format in {"ply", "both"}:
        coords_ply_path = out_dir / "coords.ply"
        save_coords_to_file(artifact.coords, coords_ply_path, format="ply", resolution=resolution)
        artifact_paths["coords_ply"] = str(coords_ply_path)

    if output_format in {"pt", "both"}:
        coords_pt_path = out_dir / "coords.pt"
        save_coords_to_file(artifact.coords, coords_pt_path, format="pt", resolution=resolution)
        artifact_paths["coords_pt"] = str(coords_pt_path)

    if save_mesh and artifact.voxel_mesh is not None:
        voxel_mesh_path = out_dir / "voxel_mesh.glb"
        save_voxel_mesh(artifact.coords, voxel_mesh_path, resolution)
        artifact_paths["voxel_mesh"] = str(voxel_mesh_path)

    metadata_path = _write_metadata(out_dir / "ss_metadata.json", artifact.metadata)
    artifact_paths["metadata"] = str(metadata_path)
    return artifact_paths


def _save_slat_metadata(artifact: SLATArtifact, out_dir: Path) -> dict[str, str]:
    if not artifact.metadata:
        return {}
    metadata_path = _write_metadata(out_dir / "slat_metadata.json", artifact.metadata)
    return {"metadata": str(metadata_path)}


def _runtime_output_params(runtime: RuntimeConfig) -> dict[str, bool]:
    return {
        "skip_render": runtime.skip_render,
        "skip_glb": runtime.skip_glb,
        "skip_ply": runtime.skip_ply,
    }


def _resolve_decode_modes(
    requested_modes: tuple[str, ...] | list[str],
    *,
    skip_render: bool,
    skip_glb: bool,
    skip_ply: bool,
) -> list[str]:
    requested = list(dict.fromkeys(requested_modes))
    resolved: list[str] = []
    for mode in requested:
        if mode == "mesh":
            if not skip_render or not skip_glb:
                resolved.append(mode)
            continue
        if mode == "gaussian":
            if not skip_render or not skip_ply or not skip_glb:
                resolved.append(mode)
            continue
        if mode == "radiance_field":
            if not skip_render:
                resolved.append(mode)
            continue
        resolved.append(mode)
    return resolved


def _move_mesh_outputs_to_cpu(outputs: dict[str, Any]) -> None:
    if "mesh" not in outputs:
        return
    for mesh in outputs["mesh"]:
        mesh.vertices = mesh.vertices.cpu()
        mesh.faces = mesh.faces.cpu()
        if mesh.vertex_attrs is not None:
            mesh.vertex_attrs = mesh.vertex_attrs.cpu()
        if mesh.face_normal is not None:
            mesh.face_normal = mesh.face_normal.cpu()


def _decode_outputs_with_flow_offload(
    pipeline,
    edit_slat,
    decode_modes: tuple[str, ...] | list[str],
):
    models_to_cpu = {}
    try:
        for key in ("sparse_structure_flow_model", "slat_flow_model"):
            if key in pipeline.models:
                models_to_cpu[key] = pipeline.models[key]
                pipeline.models[key] = pipeline.models[key].cpu()

        _release_cuda_memory()
        outputs = pipeline.decode_slat(edit_slat, list(decode_modes))
        _move_mesh_outputs_to_cpu(outputs)
        return outputs
    finally:
        for key, model in models_to_cpu.items():
            pipeline.models[key] = model.to(pipeline.device)


@dataclass(frozen=True)
class _P2PHookInputs:
    source_image: Image.Image
    edit_image: Image.Image
    mask_image: Image.Image


@dataclass(frozen=True)
class _StageRunConfig:
    seed: int
    num_samples: int
    sparse_structure_sampler_params: dict[str, Any] | None = None
    slat_sampler_params: dict[str, Any] | None = None
    extra_params: dict[str, Any] | None = None


class UniEditSSAdapter(SSStagePlugin):
    name = "uniedit_ss"

    def run(
        self,
        context: ExperimentContext,
        preprocess,
        config: UniEditSSConfig,
    ) -> SSArtifact:
        source_voxels_path = context.config.inputs.source_voxels
        if source_voxels_path is None:
            raise RuntimeError("UniEdit SS requires inputs.source_voxels")

        pipeline = context.pipeline
        resolution = pipeline.sparse_structure_sampler_params.get("grid_size", 64)
        ss_params = {
            **getattr(pipeline, "sparse_structure_sampler_params", {}),
            **_sampler_params(config.sampler),
        }

        source_coords = ply_to_coords(source_voxels_path, pipeline.device, resolution)
        mask_coords = None
        if context.config.inputs.mask_glb is not None:
            mask_result = load_mask_glb_coords(
                mask_glb=str(context.config.inputs.mask_glb),
                device=pipeline.device,
                resolution=resolution,
                asset_dir=_source_asset_dir(context),
            )
            mask_coords = mask_result.coords

        source_image, edit_image, _ = _prepared_images(preprocess)
        source_cond_dict = pipeline.get_cond([source_image])
        edit_cond_dict = pipeline.get_cond([edit_image])
        source_cond = source_cond_dict["cond"]
        edit_cond = edit_cond_dict["cond"]
        neg_cond = source_cond_dict["neg_cond"]

        _release_cuda_memory()

        print("Stage 0: Inverting source sparse structure...")
        source_voxel = coords_to_voxel(source_coords, pipeline.device, resolution)
        ss_terminal_noise = invert_sparse_structure(
            pipeline=pipeline,
            cond_src={"cond": source_cond, "neg_cond": neg_cond},
            voxel_src=source_voxel,
            params=ss_params,
            cfg_interval=config.cfg_interval,
            verbose=True,
        )

        _release_cuda_memory()

        print(f"SS Stage: Editing sparse structure (omega={config.omega})...")
        coords_ss_raw = self._denoise_sparse_structure_uniedit(
            pipeline=pipeline,
            source_cond={"cond": source_cond, "neg_cond": neg_cond},
            target_cond={"cond": edit_cond, "neg_cond": neg_cond},
            terminal_noise=ss_terminal_noise,
            params=ss_params,
            cfg_interval=config.cfg_interval,
            omega=config.omega,
        )

        coords_ss_masked, _, ss_meta = compose_stage1_coords(
            coords_source=source_coords,
            coords_stage1_raw=coords_ss_raw,
            mask_coords=mask_coords,
        )
        print(f"SS Stage complete: {ss_meta['stage1_masked_voxel_count']} voxels")

        _release_cuda_memory()

        voxel_mesh = None
        if config.save_voxel_mesh:
            voxel_mesh = coords_to_cubic_mesh(coords_ss_masked, resolution)
            print(f"Generated voxel mesh: {len(voxel_mesh.faces)} faces")

        return SSArtifact(
            plugin_name=self.name,
            coords=coords_ss_masked,
            voxel_mesh=voxel_mesh,
            metadata={
                "ss_meta": ss_meta,
                "ss_omega": config.omega,
                "cfg_interval": config.cfg_interval,
                "resolution": resolution,
                "voxel_count": int(len(coords_ss_masked)),
            },
        )

    def save(self, artifact: SSArtifact, out_dir: Path, config: UniEditSSConfig) -> dict[str, str]:
        return _save_ss_artifacts(
            artifact,
            out_dir,
            output_format=config.output_format,
            save_mesh=config.save_voxel_mesh,
        )

    @staticmethod
    def _denoise_sparse_structure_uniedit(
        pipeline,
        source_cond: dict,
        target_cond: dict,
        terminal_noise: torch.Tensor,
        params: dict,
        cfg_interval: tuple[float, float],
        omega: float,
    ) -> torch.Tensor:
        flow_model = pipeline.models["sparse_structure_flow_model"]
        decoder = pipeline.models["sparse_structure_decoder"]
        sampler = UniEditRFSolver()

        z_tgt = sampler.sample(
            model=flow_model,
            sample=terminal_noise,
            source_cond=source_cond["cond"],
            target_cond=target_cond["cond"],
            neg_cond=target_cond["neg_cond"],
            steps=params["steps"],
            rescale_t=params["rescale_t"],
            cfg_strength=params["cfg_strength"],
            cfg_interval=cfg_interval,
            omega=float(omega),
            selector=None,
            mode="full_uniedit",
            verbose=True,
        )

        voxel = decoder(z_tgt)
        coords = torch.argwhere(voxel > 0)[:, [0, 2, 3, 4]].int()
        if coords.shape[0] == 0:
            raise RuntimeError("SS stage UniEdit denoising produced an empty structure.")
        return coords


class _P2PLatentBlendPluginBase(P2PLatentBlendCommonMixin):
    def __init__(self) -> None:
        self.hook = None

    def cleanup(self) -> None:
        if self.hook is not None:
            self.hook.restore()
            self.hook = None

    @staticmethod
    def _build_coord_hash_set(coords: torch.Tensor) -> set[str]:
        return {
            f"{int(x)}_{int(y)}_{int(z)}"
            for _, x, y, z in coords.cpu().tolist()
        }

    @staticmethod
    def _build_mask_hash_set(mask_coords: torch.Tensor) -> set[str]:
        mask_hash_set = set()
        for coord in mask_coords.cpu().tolist():
            if len(coord) == 4:
                _, x, y, z = coord
            else:
                x, y, z = coord
            mask_hash_set.add(f"{int(x)}_{int(y)}_{int(z)}")
        return mask_hash_set

    @staticmethod
    def _move_mesh_outputs_to_cpu(outputs: dict[str, Any]) -> None:
        if "mesh" not in outputs:
            return
        for mesh in outputs["mesh"]:
            mesh.vertices = mesh.vertices.cpu()
            mesh.faces = mesh.faces.cpu()
            if mesh.vertex_attrs is not None:
                mesh.vertex_attrs = mesh.vertex_attrs.cpu()
            if mesh.face_normal is not None:
                mesh.face_normal = mesh.face_normal.cpu()

    @staticmethod
    def _patch_stage_hooks(pipeline, stage_configs: dict[str, Any], hook) -> None:
        if hook is None:
            return
        if "ss" in stage_configs:
            hook.patch_model(pipeline.models["sparse_structure_flow_model"], "ss")
        if "slat" in stage_configs:
            hook.patch_model(pipeline.models["slat_flow_model"], "slat")

    def _prepare_ss_source(
        self,
        pipeline,
        source_coords: torch.Tensor,
        source_cond_dict: dict[str, Any],
        inversion_mode: str,
        config: _StageRunConfig,
        resolution: int,
    ) -> dict[str, torch.Tensor]:
        if torch.cuda.is_available():
            print(f"[Memory] Before inversion - Allocated: {torch.cuda.memory_allocated()/1024**3:.2f} GB")
        latent_cache = self._prepare_ss_latent_from_coords(
            pipeline,
            source_coords,
            source_cond_dict,
            inversion_mode,
            config,
            resolution,
        )
        if torch.cuda.is_available():
            print(f"[Memory] After inversion - Allocated: {torch.cuda.memory_allocated()/1024**3:.2f} GB")
        print(f"[Memory] Latent cache size: {len(latent_cache)} timesteps")
        return latent_cache

    @staticmethod
    def _blend_coords_with_mask(
        source_coords: torch.Tensor,
        edit_coords: torch.Tensor,
        mask_coords: torch.Tensor,
    ) -> torch.Tensor:
        mask_hash_set = _P2PLatentBlendPluginBase._build_mask_hash_set(mask_coords)
        blended_set = set()
        blended_coords_list = []

        for coord in source_coords.cpu():
            b, x, y, z = coord.tolist()
            coord_hash = f"{int(x)}_{int(y)}_{int(z)}"
            if coord_hash not in mask_hash_set and coord_hash not in blended_set:
                blended_set.add(coord_hash)
                blended_coords_list.append([b, x, y, z])

        for coord in edit_coords.cpu():
            b, x, y, z = coord.tolist()
            coord_hash = f"{int(x)}_{int(y)}_{int(z)}"
            if coord_hash in mask_hash_set and coord_hash not in blended_set:
                blended_set.add(coord_hash)
                blended_coords_list.append([b, x, y, z])

        if not blended_coords_list:
            return torch.zeros(0, 4, dtype=source_coords.dtype, device=source_coords.device)

        return torch.tensor(
            blended_coords_list,
            dtype=source_coords.dtype,
            device=source_coords.device,
        )

    def _build_slat_preserve_coords_from_mask(
        self,
        source_slat,
        mask_coords: torch.Tensor,
    ) -> torch.Tensor:
        source_coords = source_slat.coords
        mask_hash_set = set()
        for coord in mask_coords.cpu():
            if len(coord) == 4:
                b, x, y, z = coord.tolist()
            else:
                x, y, z = coord.tolist()
                b = 0
            mask_hash_set.add(f"{int(b)}_{int(x)}_{int(y)}_{int(z)}")

        preserve_coords = []
        for coord in source_coords:
            b, x, y, z = coord.tolist()
            coord_hash = f"{int(b)}_{int(x)}_{int(y)}_{int(z)}"
            if coord_hash not in mask_hash_set:
                preserve_coords.append(coord.tolist())

        if not preserve_coords:
            return torch.zeros(0, 4, dtype=torch.int32, device=source_coords.device)

        return torch.tensor(preserve_coords, dtype=torch.int32, device=source_coords.device)

    @staticmethod
    def _resolve_decode_modes(decode_modes: tuple[str, ...] | list[str]) -> list[str]:
        return list(decode_modes)

    def _prepare_slat_source(
        self,
        pipeline,
        source_slat,
        source_cond_dict: dict[str, Any],
        inversion_mode: str,
        config: _StageRunConfig,
    ) -> dict[str, Any]:
        std = torch.tensor(pipeline.slat_normalization["std"], device=pipeline.device)[None]
        mean = torch.tensor(pipeline.slat_normalization["mean"], device=pipeline.device)[None]
        source_slat_normalized = (source_slat - mean) / std
        print(
            "Running "
            f"{'simple Euler' if inversion_mode == 'simple' else 'RF-Solver'} inversion for SLAT..."
        )
        steps = (config.slat_sampler_params or {}).get("steps", 25)
        return self._invert_sample(
            model=pipeline.models["slat_flow_model"],
            sample=source_slat_normalized,
            cond_dict=source_cond_dict,
            steps=steps,
            stage_prefix="slat",
            config=config,
            inversion_mode=inversion_mode,
        )

    def _run_slat_stage(
        self,
        pipeline,
        *,
        source_slat,
        source_cond_dict: dict[str, Any],
        edit_cond_dict: dict[str, Any],
        edit_coords: torch.Tensor,
        mask_coords: torch.Tensor | None,
        blend_slat_enabled: bool,
        inversion_mode: str,
        config: _StageRunConfig,
        verbose: bool,
    ):
        extra = config.extra_params or {}
        slat_params = self._resolve_stage_sampler_params(
            "slat",
            config.slat_sampler_params,
            extra,
        )

        source_slat_latent_cache = None
        original_slat_sampler = pipeline.slat_sampler
        try:
            if blend_slat_enabled:
                print("Preparing source SLAT latent via inversion...")
                _release_cuda_memory()
                source_slat_latent_cache = self._prepare_slat_source(
                    pipeline,
                    source_slat,
                    source_cond_dict,
                    inversion_mode,
                    config,
                )
                if verbose:
                    print(f"Cached {len(source_slat_latent_cache)} SLAT latent timesteps")
                _release_cuda_memory()

            if blend_slat_enabled and source_slat_latent_cache is not None and mask_coords is not None:
                print("Setting up SLAT latent blending...")
                slat_sampler = LatentBlendFlowEulerGuidanceIntervalSampler(
                    sigma_min=original_slat_sampler.sigma_min
                )
                slat_preserve_coords = self._build_slat_preserve_coords_from_mask(
                    source_slat,
                    mask_coords,
                )
                slat_sampler.set_blend_source(
                    source_latent_cache=source_slat_latent_cache,
                    latent_mask=slat_preserve_coords,
                    is_sparse=True,
                )
                pipeline.slat_sampler = slat_sampler
            else:
                print("SLAT blending disabled")

            print("Generating edit SLAT with P2P + SLAT blending...")
            edit_slat = pipeline.sample_slat(
                edit_cond_dict,
                edit_coords,
                sampler_params=slat_params,
            )
            if verbose:
                print(f"Edit SLAT: {edit_slat.coords.shape[0]} voxels")
            return edit_slat
        finally:
            pipeline.slat_sampler = original_slat_sampler
            if source_slat_latent_cache is not None:
                del source_slat_latent_cache
            gc.collect()
            if torch.cuda.is_available():
                torch.cuda.empty_cache()

    def _decode_outputs(
        self,
        pipeline,
        *,
        source_slat,
        edit_slat,
        decode_modes: tuple[str, ...] | list[str],
        skip_source_decode: bool,
        verbose: bool,
    ) -> tuple[dict[str, Any], dict[str, Any] | None]:
        resolved_decode_modes = self._resolve_decode_modes(decode_modes)

        if verbose:
            print("Moving source SLAT to CPU...")
        source_slat_cpu = source_slat.to("cpu")

        models_to_cpu = {}
        try:
            if verbose:
                print("Moving flow models to CPU...")
            for key in ["sparse_structure_flow_model", "slat_flow_model"]:
                if key in pipeline.models:
                    models_to_cpu[key] = pipeline.models[key]
                    pipeline.models[key] = pipeline.models[key].cpu()

            gc.collect()
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
            print("Decoding result...")
            outputs = pipeline.decode_slat(edit_slat, resolved_decode_modes)
            del edit_slat
            self._move_mesh_outputs_to_cpu(outputs)
            gc.collect()
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
                torch.cuda.synchronize()
                torch.cuda.empty_cache()

            if skip_source_decode:
                print("Skipping source decode to save memory")
                source_outputs = None
            else:
                print("Decoding source SLAT...")
                source_slat_gpu = source_slat_cpu.to(pipeline.device)
                source_outputs = pipeline.decode_slat(source_slat_gpu, resolved_decode_modes)
                del source_slat_gpu
                self._move_mesh_outputs_to_cpu(source_outputs)

            return outputs, source_outputs
        finally:
            if verbose:
                print("Restoring flow models to GPU...")
            for key, model in models_to_cpu.items():
                pipeline.models[key] = model.to(pipeline.device)


class P2PLatentBlendSSAdapter(_P2PLatentBlendPluginBase, SSStagePlugin):
    name = "p2p_latent_blend_ss"

    def run(
        self,
        context: ExperimentContext,
        preprocess,
        config: P2PLatentBlendSSConfig,
    ) -> SSArtifact:
        source_voxels_path = context.config.inputs.source_voxels
        if source_voxels_path is None:
            raise RuntimeError("P2P latent-blend SS requires inputs.source_voxels")
        if context.config.inputs.mask_glb is None:
            raise RuntimeError("P2P latent-blend SS requires inputs.mask_glb")

        pipeline = context.pipeline
        resolution = pipeline.sparse_structure_sampler_params.get("grid_size", 64)
        source_coords = ply_to_coords(source_voxels_path, pipeline.device, resolution)
        print(f"Source coords: {len(source_coords)} voxels")

        mask_result = load_mask_glb_coords(
            mask_glb=str(context.config.inputs.mask_glb),
            device=pipeline.device,
            resolution=resolution,
            asset_dir=_source_asset_dir(context),
        )
        mask_coords = mask_result.coords
        print(f"Mask coords: {len(mask_coords)} voxels")

        source_image, edit_image, mask_image = _prepared_images(preprocess)
        source_cond_dict = pipeline.get_cond([source_image])
        edit_cond_dict = pipeline.get_cond([edit_image])

        extra = {
            "ss_t_start": config.hook.t_start,
            "ss_t_end": config.hook.t_end,
            "ss_strength": config.hook.strength,
            "patch_coverage_threshold": config.patch_coverage_threshold,
            "query_chunk": config.query_chunk,
            "ss_denoise_cfg_strength": config.denoise_cfg_strength,
            "ss_denoise_cfg_interval_start": config.denoise_cfg_interval[0],
            "ss_denoise_cfg_interval_end": config.denoise_cfg_interval[1],
            "ss_inversion_cfg_strength": config.inversion_cfg_strength,
            "ss_inversion_cfg_interval_start": config.inversion_cfg_interval[0],
            "ss_inversion_cfg_interval_end": config.inversion_cfg_interval[1],
        }
        hook_inputs = _P2PHookInputs(
            source_image=source_image,
            edit_image=edit_image,
            mask_image=mask_image,
        )
        self.hook, _, stage_configs = self._create_p2p_hook(
            inputs=hook_inputs,
            extra=extra,
            source_cond_dict=source_cond_dict,
            edit_cond_dict=edit_cond_dict,
            enabled_stages=["ss"],
        )

        stage_config = _StageRunConfig(
            seed=context.config.runtime.seed,
            num_samples=context.config.runtime.num_samples,
            sparse_structure_sampler_params=_sampler_params(config.sampler),
            extra_params=extra,
        )

        inversion_cfg = self._resolve_inversion_cfg("ss", extra)
        effective_sampler_params = {
            **getattr(pipeline, "sparse_structure_sampler_params", {}),
            **self._resolve_stage_sampler_params("ss", stage_config.sparse_structure_sampler_params, extra),
        }

        print(f"Step 1: Inverting source SS (mode={config.inversion_mode})...")
        latent_cache = self._prepare_ss_source(
            pipeline,
            source_coords,
            source_cond_dict,
            config.inversion_mode,
            stage_config,
            resolution,
        )

        _release_cuda_memory()

        print("Step 2: Building 3D latent mask...")
        model = pipeline.models["sparse_structure_flow_model"]
        latent_mask = self._build_ss_latent_mask(
            mask_coords,
            resolution=resolution,
            channels=model.in_channels,
            latent_resolution=model.resolution,
        )

        print("Step 3: Running SS denoising with P2P + latent blending...")
        self._patch_stage_hooks(pipeline, stage_configs, self.hook)
        sample = self._denoise_ss_with_step_blend(
            pipeline,
            edit_cond_dict,
            stage_config,
            source_latent_cache=latent_cache if config.blend_enabled else None,
            latent_mask=latent_mask,
            blend_strength=config.blend_strength,
            verbose=True,
        )

        decoder = pipeline.models["sparse_structure_decoder"]
        if torch.cuda.is_available():
            print(f"[Memory] Before decode - Allocated: {torch.cuda.memory_allocated()/1024**3:.2f} GB")
        voxel = decoder(sample)
        coords_edited = torch.argwhere(voxel > 0)[:, [0, 2, 3, 4]].int()
        del sample, voxel
        _release_cuda_memory()
        if torch.cuda.is_available():
            print(f"[Memory] After decode - Allocated: {torch.cuda.memory_allocated()/1024**3:.2f} GB")

        if coords_edited.shape[0] == 0:
            raise RuntimeError("SS stage produced empty structure")

        print(f"SS Stage complete: {len(coords_edited)} voxels")

        voxel_mesh = None
        if config.save_voxel_mesh:
            voxel_mesh = coords_to_cubic_mesh(coords_edited, resolution)
            print(f"Generated voxel mesh: {len(voxel_mesh.faces)} faces")

        return SSArtifact(
            plugin_name=self.name,
            coords=coords_edited,
            voxel_mesh=voxel_mesh,
            metadata={
                "inversion_mode": config.inversion_mode,
                "blend_enabled": config.blend_enabled,
                "blend_strength": config.blend_strength,
                "ss_denoise_cfg_strength": float(effective_sampler_params.get("cfg_strength", 0.0)),
                "ss_denoise_cfg_interval": tuple(effective_sampler_params.get("cfg_interval", (0.0, 1.0))),
                "ss_inversion_cfg_strength": inversion_cfg["cfg_strength"],
                "ss_inversion_cfg_interval": inversion_cfg["cfg_interval"],
                "resolution": resolution,
                "voxel_count": int(len(coords_edited)),
            },
        )

    def save(self, artifact: SSArtifact, out_dir: Path, config: P2PLatentBlendSSConfig) -> dict[str, str]:
        return _save_ss_artifacts(
            artifact,
            out_dir,
            output_format=config.output_format,
            save_mesh=config.save_voxel_mesh,
        )


class UniEditSLATAdapter(SLATStagePlugin):
    name = "uniedit_slat"

    def run(
        self,
        context: ExperimentContext,
        preprocess,
        config: UniEditSLATConfig,
        ss_artifact: SSArtifact | None,
    ) -> SLATArtifact:
        self._output_options = _runtime_output_params(context.config.runtime)
        edited_coords_path = context.config.inputs.edited_coords
        if ss_artifact is not None:
            edited_coords_path = ss_artifact.primary_coords_path
        if edited_coords_path is None:
            raise RuntimeError("SLAT stage requires explicit edited_coords input or a preceding SS stage output")

        source_features_path = context.config.inputs.source_features
        if source_features_path is None:
            raise RuntimeError("UniEdit SLAT requires inputs.source_features")

        pipeline = context.pipeline
        resolution = pipeline.sparse_structure_sampler_params.get("grid_size", 64)
        slat_params = {
            **getattr(pipeline, "slat_sampler_params", {}),
            **_sampler_params(config.sampler),
        }

        coords_edited_raw = load_coords_from_file(edited_coords_path, pipeline.device)
        coords_edited = coords3d_to_batched(coords_edited_raw, batch_idx=0)
        print(f"Loaded edited coords: {len(coords_edited_raw)} voxels")
        _release_cuda_memory()

        from trellis.modules import sparse as sp

        source_slat = feats_to_slat(pipeline, source_features_path, sp.SparseTensor)
        print(f"Loaded source SLAT: {source_slat.coords.shape[0]} features")
        _release_cuda_memory()

        mask_coords = None
        if context.config.inputs.mask_glb is not None:
            mask_result = load_mask_glb_coords(
                mask_glb=str(context.config.inputs.mask_glb),
                device=pipeline.device,
                resolution=resolution,
                asset_dir=_source_asset_dir(context),
            )
            mask_coords = mask_result.coords

        source_coords = None
        if context.config.inputs.source_voxels is not None:
            source_coords = load_coords_from_file(context.config.inputs.source_voxels, pipeline.device)

        source_image, edit_image, _ = _prepared_images(preprocess)
        source_cond_dict = pipeline.get_cond([source_image])
        edit_cond_dict = pipeline.get_cond([edit_image])
        source_cond = source_cond_dict["cond"]
        edit_cond = edit_cond_dict["cond"]
        neg_cond = source_cond_dict["neg_cond"]

        _release_cuda_memory()

        print("Stage 0: Inverting source SLAT...")
        slat_latent_cache = None
        if config.stage2_variant == "latent_replace_union":
            slat_terminal_noise, slat_latent_cache = self._invert_slat_with_cache(
                pipeline=pipeline,
                cond_src={"cond": source_cond, "neg_cond": neg_cond},
                slat_src=source_slat,
                params=slat_params,
                cfg_interval=config.cfg_interval,
            )
        else:
            slat_terminal_noise = invert_slat(
                pipeline=pipeline,
                cond_src={"cond": source_cond, "neg_cond": neg_cond},
                slat_src=source_slat,
                params=slat_params,
                cfg_interval=config.cfg_interval,
                verbose=True,
            )

        _release_cuda_memory()

        print("Projecting SLAT noise to edited coordinates...")
        projected_slat_noise = project_sparse_terminal_noise(
            source_noise=slat_terminal_noise,
            target_coords=coords_edited.to(device=pipeline.device),
            device=pipeline.device,
            SparseTensor=type(slat_terminal_noise),
            resolution=resolution,
        )

        stage2_selector = None
        if source_coords is not None and mask_coords is not None:
            _, coords_preserve, _ = compose_stage1_coords(
                coords_source=source_coords,
                coords_stage1_raw=coords_edited,
                mask_coords=mask_coords,
            )
            stage2_selector = build_stage2_selector(
                coords_edited,
                coords3d_to_batched(coords_preserve, batch_idx=0),
            )
            print(f"Built stage2 selector: {stage2_selector.sum().item()} preserve voxels")

        print(f"SLAT Stage: Editing SLAT features (variant={config.stage2_variant}, omega={config.omega})...")
        if config.stage2_variant == "preserve_uniedit":
            slat_tgt = self._denoise_slat_variant(
                pipeline=pipeline,
                source_cond={"cond": source_cond, "neg_cond": neg_cond},
                target_cond={"cond": edit_cond, "neg_cond": neg_cond},
                terminal_noise=projected_slat_noise,
                params=slat_params,
                cfg_interval=config.cfg_interval,
                omega=config.omega,
                selector=stage2_selector,
                mode="preserve_overlap",
            )
        elif config.stage2_variant == "free_target":
            slat_tgt = self._denoise_slat_variant(
                pipeline=pipeline,
                source_cond={"cond": source_cond, "neg_cond": neg_cond},
                target_cond={"cond": edit_cond, "neg_cond": neg_cond},
                terminal_noise=projected_slat_noise,
                params=slat_params,
                cfg_interval=config.cfg_interval,
                omega=config.omega,
                selector=None,
                mode="target_only",
            )
        elif config.stage2_variant == "latent_replace_union":
            if stage2_selector is None:
                raise RuntimeError(
                    "UniEdit SLAT stage2_variant='latent_replace_union' requires both source_voxels and mask_glb"
                )
            stage2_replace_target_idx, stage2_replace_source_idx = build_sparse_replace_index_map(
                coords_target=projected_slat_noise.coords,
                coords_source=slat_terminal_noise.coords,
                selector=stage2_selector,
            )
            slat_tgt = self._denoise_slat_latent_replace(
                pipeline=pipeline,
                target_cond={"cond": edit_cond, "neg_cond": neg_cond},
                terminal_noise=projected_slat_noise,
                params=slat_params,
                cfg_interval=config.cfg_interval,
                latent_cache=slat_latent_cache,
                replace_target_indices=stage2_replace_target_idx,
                replace_source_indices=stage2_replace_source_idx,
            )
        else:
            raise NotImplementedError(
                f"Stage 2 variant '{config.stage2_variant}' not implemented. "
                "Supported: preserve_uniedit, free_target, latent_replace_union."
            )

        del slat_terminal_noise, projected_slat_noise, source_slat
        del source_cond, edit_cond, neg_cond, source_cond_dict, edit_cond_dict
        if slat_latent_cache is not None:
            del slat_latent_cache
        if source_coords is not None:
            del source_coords
        if mask_coords is not None:
            del mask_coords
        if stage2_selector is not None:
            del stage2_selector
        _release_cuda_memory()

        decode_modes = _resolve_decode_modes(
            config.decode_modes,
            skip_render=self._output_options["skip_render"],
            skip_glb=self._output_options["skip_glb"],
            skip_ply=self._output_options["skip_ply"],
        )
        print("Decoding final result...")
        print(f"Decode modes: {decode_modes}")
        outputs = _decode_outputs_with_flow_offload(
            pipeline,
            slat_tgt,
            decode_modes,
        )
        return SLATArtifact(
            plugin_name=self.name,
            outputs=outputs,
            source_outputs=None,
            metadata={
                "stage_mode": "slat_only",
                "stage2_variant": config.stage2_variant,
                "slat_omega": config.omega,
                "cfg_interval": config.cfg_interval,
                "edited_voxel_count": int(len(coords_edited_raw)),
            },
        )

    def save(self, artifact: SLATArtifact, out_dir: Path, config: UniEditSLATConfig) -> dict[str, str]:
        output_options = getattr(self, "_output_options", {"skip_render": True, "skip_glb": False, "skip_ply": False})
        save_outputs(
            outputs=artifact.outputs,
            out_dir=out_dir,
            skip_render=output_options["skip_render"],
            skip_glb=output_options["skip_glb"],
            skip_ply=output_options["skip_ply"],
        )
        return _save_slat_metadata(artifact, out_dir)

    @staticmethod
    def _denoise_slat_variant(
        pipeline,
        source_cond: dict,
        target_cond: dict,
        terminal_noise,
        params: dict,
        cfg_interval: tuple[float, float],
        omega: float,
        selector,
        mode: str,
    ):
        flow_model = pipeline.models["slat_flow_model"]
        sampler = UniEditRFSolver()

        slat_normalized = sampler.sample(
            model=flow_model,
            sample=terminal_noise,
            source_cond=source_cond["cond"],
            target_cond=target_cond["cond"],
            neg_cond=target_cond["neg_cond"],
            steps=params["steps"],
            rescale_t=params["rescale_t"],
            cfg_strength=params["cfg_strength"],
            cfg_interval=cfg_interval,
            omega=float(omega),
            selector=selector,
            mode=mode,
            verbose=True,
        )

        from trellis_edit.inversion.rf_inversion import get_slat_norm_tensors

        mean, std = get_slat_norm_tensors(
            pipeline,
            slat_normalized.device,
            slat_normalized.feats.dtype,
        )
        return slat_normalized * std + mean

    @staticmethod
    def _invert_slat_with_cache(
        pipeline,
        cond_src: dict,
        slat_src,
        params: dict,
        cfg_interval: tuple[float, float],
    ):
        from trellis_edit.inversion.latent_replace_sampler import SparseLatentReplaceRFSolver
        from trellis_edit.inversion.rf_inversion import get_slat_norm_tensors

        flow_model = pipeline.models["slat_flow_model"]
        mean, std = get_slat_norm_tensors(pipeline, slat_src.device, slat_src.feats.dtype)
        slat_normalized = (slat_src - mean) / std
        sampler = SparseLatentReplaceRFSolver()
        return sampler.invert_with_cache(
            model=flow_model,
            sample=slat_normalized,
            cond_dict=cond_src,
            steps=params["steps"],
            rescale_t=params["rescale_t"],
            cfg_strength=params["cfg_strength"],
            cfg_interval=cfg_interval,
            verbose=True,
        )

    @staticmethod
    def _denoise_slat_latent_replace(
        pipeline,
        target_cond: dict,
        terminal_noise,
        params: dict,
        cfg_interval: tuple[float, float],
        latent_cache: dict,
        replace_target_indices: torch.Tensor,
        replace_source_indices: torch.Tensor,
    ):
        from trellis_edit.inversion.latent_replace_sampler import SparseLatentReplaceRFSolver
        from trellis_edit.inversion.rf_inversion import get_slat_norm_tensors

        flow_model = pipeline.models["slat_flow_model"]
        sampler = SparseLatentReplaceRFSolver()
        slat_normalized = sampler.sample_with_replacement(
            model=flow_model,
            sample=terminal_noise,
            cond_dict=target_cond,
            steps=params["steps"],
            rescale_t=params["rescale_t"],
            cfg_strength=params["cfg_strength"],
            cfg_interval=cfg_interval,
            latent_cache=latent_cache,
            replace_target_indices=replace_target_indices,
            replace_source_indices=replace_source_indices,
            verbose=True,
        )
        mean, std = get_slat_norm_tensors(
            pipeline,
            slat_normalized.device,
            slat_normalized.feats.dtype,
        )
        return slat_normalized * std + mean


class P2PLatentBlendSLATAdapter(_P2PLatentBlendPluginBase, SLATStagePlugin):
    name = "p2p_latent_blend_slat"

    def run(
        self,
        context: ExperimentContext,
        preprocess,
        config: P2PLatentBlendSLATConfig,
        ss_artifact: SSArtifact | None,
    ) -> SLATArtifact:
        self._output_options = _runtime_output_params(context.config.runtime)
        edited_coords_path = context.config.inputs.edited_coords
        if ss_artifact is not None:
            edited_coords_path = ss_artifact.primary_coords_path
        if edited_coords_path is None:
            raise RuntimeError("SLAT stage requires explicit edited_coords input or a preceding SS stage output")

        source_features_path = context.config.inputs.source_features
        if source_features_path is None:
            raise RuntimeError("P2P latent-blend SLAT requires inputs.source_features")

        pipeline = context.pipeline
        resolution = pipeline.sparse_structure_sampler_params.get("grid_size", 64)
        source_image, edit_image, mask_image = _prepared_images(preprocess)
        source_cond_dict = pipeline.get_cond([source_image])
        edit_cond_dict = pipeline.get_cond([edit_image])

        edited_coords_raw = load_coords_from_file(edited_coords_path, pipeline.device)
        edited_coords = coords3d_to_batched(edited_coords_raw, batch_idx=0)
        print(f"Loaded edited coords: {edited_coords.shape[0]} voxels from {edited_coords_path}")

        from trellis.modules.sparse.basic import SparseTensor

        print(f"Loading source SLAT features from: {source_features_path}")
        source_slat = feats_to_slat(
            pipeline=pipeline,
            feats_path=source_features_path,
            SparseTensor=SparseTensor,
        )

        mask_coords = None
        if context.config.inputs.mask_glb is not None:
            mask_result = load_mask_glb_coords(
                mask_glb=str(context.config.inputs.mask_glb),
                device=pipeline.device,
                resolution=resolution,
                asset_dir=_source_asset_dir(context),
            )
            mask_coords = mask_result.coords
            print(f"Mask coords: {mask_coords.shape[0]} voxels")
        elif config.blend_enabled:
            raise RuntimeError("P2P latent-blend SLAT requires inputs.mask_glb when blending is enabled")

        extra = {
            "slat_t_start": config.hook.t_start,
            "slat_t_end": config.hook.t_end,
            "slat_strength": config.hook.strength,
            "patch_coverage_threshold": config.patch_coverage_threshold,
            "query_chunk": config.query_chunk,
            "skip_source_decode": config.skip_source_decode and not context.config.runtime.save_source_outputs,
        }
        hook_inputs = _P2PHookInputs(
            source_image=source_image,
            edit_image=edit_image,
            mask_image=mask_image,
        )
        self.hook, token_meta, stage_configs = self._create_p2p_hook(
            inputs=hook_inputs,
            extra=extra,
            source_cond_dict=source_cond_dict,
            edit_cond_dict=edit_cond_dict,
            enabled_stages=["slat"],
        )

        stage_config = _StageRunConfig(
            seed=context.config.runtime.seed,
            num_samples=context.config.runtime.num_samples,
            slat_sampler_params=_sampler_params(config.sampler),
            extra_params=extra,
        )

        grad_enabled = torch.is_grad_enabled()
        torch.set_grad_enabled(False)
        try:
            self._patch_stage_hooks(pipeline, stage_configs, self.hook)
            edit_slat = self._run_slat_stage(
                pipeline,
                source_slat=source_slat,
                source_cond_dict=source_cond_dict,
                edit_cond_dict=edit_cond_dict,
                edit_coords=edited_coords,
                mask_coords=mask_coords,
                blend_slat_enabled=config.blend_enabled,
                inversion_mode=config.inversion_mode,
                config=stage_config,
                verbose=False,
            )
            outputs, source_outputs = self._decode_outputs(
                pipeline,
                source_slat=source_slat,
                edit_slat=edit_slat,
                decode_modes=("mesh", "gaussian"),
                skip_source_decode=extra["skip_source_decode"],
                verbose=False,
            )
        finally:
            torch.set_grad_enabled(grad_enabled)

        return SLATArtifact(
            plugin_name=self.name,
            outputs=outputs,
            source_outputs=source_outputs,
            metadata=token_meta,
        )

    def save(self, artifact: SLATArtifact, out_dir: Path, config: P2PLatentBlendSLATConfig) -> dict[str, str]:
        output_options = getattr(self, "_output_options", {"skip_render": True, "skip_glb": False, "skip_ply": False})
        save_outputs(
            outputs=artifact.outputs,
            out_dir=out_dir,
            skip_render=output_options["skip_render"],
            skip_glb=output_options["skip_glb"],
            skip_ply=output_options["skip_ply"],
        )

        artifact_paths = _save_slat_metadata(artifact, out_dir)
        edit_img_path = out_dir / "edit_preprocessed.png"
        mask_img_path = out_dir / "mask_preprocessed.png"
        if edit_img_path.is_file() and mask_img_path.is_file() and artifact.metadata:
            save_mask_overlay_preview(
                edit_image=Image.open(edit_img_path),
                mask_image=Image.open(mask_img_path),
                token_meta=artifact.metadata,
                path=out_dir / "mask_token_overlay.png",
            )
            artifact_paths["mask_overlay"] = str(out_dir / "mask_token_overlay.png")
        return artifact_paths


SS_PLUGIN_REGISTRY: dict[str, type[SSStagePlugin]] = {
    "uniedit_ss": UniEditSSAdapter,
    "p2p_latent_blend_ss": P2PLatentBlendSSAdapter,
}

SLAT_PLUGIN_REGISTRY: dict[str, type[SLATStagePlugin]] = {
    "uniedit_slat": UniEditSLATAdapter,
    "p2p_latent_blend_slat": P2PLatentBlendSLATAdapter,
}
