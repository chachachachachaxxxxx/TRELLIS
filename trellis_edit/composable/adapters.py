from __future__ import annotations

import gc
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
import torch
from PIL import Image

from trellis_edit.common import ensure_pipeline_encoders, save_outputs, write_json
from trellis_edit.inversion.uniedit_sampler import UniEditRFSolver
from trellis_edit.inversion.rf_sampler import RFSolverSampler, resolve_inversion_steps
from trellis_edit.composable.p2p_common import ControlledDenoiseCommonMixin
from trellis_edit.composable.runtime import SourceTrace
from trellis_edit.preprocess.asset_3d import (
    coords_to_flat_indices,
    coords_to_voxel,
    feats_to_slat,
    load_mask_glb_coords,
    load_source_voxel_normalization,
    ply_to_coords,
    project_sparse_terminal_noise,
)
from trellis_edit.samplers import (
    AnchorFlowSampler,
    FlowEditSampler,
    LatentBlendFlowEulerGuidanceIntervalSampler,
    SparseLatentBlendMask,
    blend_sparse_features,
)
from trellis_edit.utils.uniedit_utils import (
    build_stage2_selector,
    compose_stage1_coords,
    compose_stage1_coords_restore_all_outside_mask,
    compose_stage1_coords_boundary_band_restore,
    coords3d_to_batched,
)
from trellis_edit.utils.voxel_mesh_converter import (
    coords_to_cubic_mesh,
    load_coords_from_file,
    save_coords_to_file,
    save_voxel_mesh,
    voxel_mesh_glb_transform_payload,
)

from .artifacts import SLATArtifact, SSArtifact
from .base import ExperimentContext, SLATStagePlugin, SSStagePlugin
from .config import (
    RuntimeConfig,
    SamplerOverrideConfig,
    SLATStageConfig,
    SSStageConfig,
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


def _mask_cache_dir(context: ExperimentContext) -> Path | None:
    mask_glb = context.config.inputs.mask_glb
    source_asset_dir = _source_asset_dir(context)
    if mask_glb is None:
        return source_asset_dir

    if source_asset_dir is not None:
        case_cache_dir = source_asset_dir / mask_glb.parent.name
        if case_cache_dir.is_dir():
            return case_cache_dir

    return mask_glb.parent


def _mask_source_normalization(context: ExperimentContext):
    source_asset_dir = _source_asset_dir(context)
    if source_asset_dir is None:
        return None
    return load_source_voxel_normalization(source_asset_dir)


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
        voxel_mesh_transform_path = _write_metadata(
            out_dir / "voxel_mesh_transform.json",
            voxel_mesh_glb_transform_payload(),
        )
        artifact_paths["voxel_mesh_transform"] = str(voxel_mesh_transform_path)

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


def _ss_voxel_mesh_export_meta(enabled: bool) -> dict[str, Any]:
    if not enabled:
        return {}
    return {"voxel_mesh_export": voxel_mesh_glb_transform_payload()}


def _resolve_ss_edit_region_mode(config: SSStageConfig) -> str:
    mode_sources: list[tuple[str, str]] = []
    if config.controls.kv_blend.enabled:
        mode_sources.append(
            ("ss.controls.kv_blend.edit_region_mode", str(config.controls.kv_blend.edit_region_mode))
        )
    if config.controls.latent_blend.enabled:
        mode_sources.append(
            ("ss.controls.latent_blend.edit_region_mode", str(config.controls.latent_blend.edit_region_mode))
        )
    if not mode_sources:
        return "mask_only"

    resolved_modes = {mode for _, mode in mode_sources}
    if len(resolved_modes) != 1:
        details = ", ".join(f"{field}={mode}" for field, mode in mode_sources)
        raise RuntimeError(
            "SS blend controls must use the same edit_region_mode when enabled together. "
            f"Got: {details}"
        )
    return mode_sources[0][1]


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


def _load_mask_coords(
    context: ExperimentContext,
    pipeline,
    *,
    resolution: int,
) -> torch.Tensor | None:
    if context.config.inputs.mask_glb is None:
        return None
    mask_result = load_mask_glb_coords(
        mask_glb=str(context.config.inputs.mask_glb),
        device=pipeline.device,
        resolution=resolution,
        asset_dir=_mask_cache_dir(context),
        source_normalization=_mask_source_normalization(context),
    )
    return mask_result.coords


def _apply_ss_postprocess(
    *,
    mode: str,
    source_coords: torch.Tensor | None,
    coords_stage1_raw: torch.Tensor,
    mask_coords: torch.Tensor | None,
    boundary_band_width_voxels: int,
    boundary_band_target_neighbor_threshold: int,
) -> tuple[torch.Tensor, torch.Tensor, dict[str, Any]]:
    if mode == "none":
        coords_ref = source_coords if source_coords is not None else coords_stage1_raw
        return compose_stage1_coords(
            coords_source=coords_ref,
            coords_stage1_raw=coords_stage1_raw,
            mask_coords=None,
        )
    if mode == "restore_source_outside_mask":
        if source_coords is None:
            raise RuntimeError("ss.postprocess.mode='restore_source_outside_mask' requires source voxels.")
        if mask_coords is None:
            raise RuntimeError("ss.postprocess.mode='restore_source_outside_mask' requires mask_glb.")
        return compose_stage1_coords(
            coords_source=source_coords,
            coords_stage1_raw=coords_stage1_raw,
            mask_coords=mask_coords,
        )
    if mode == "restore_all_outside_mask":
        if source_coords is None:
            raise RuntimeError("ss.postprocess.mode='restore_all_outside_mask' requires source voxels.")
        if mask_coords is None:
            raise RuntimeError("ss.postprocess.mode='restore_all_outside_mask' requires mask_glb.")
        return compose_stage1_coords_restore_all_outside_mask(
            coords_source=source_coords,
            coords_stage1_raw=coords_stage1_raw,
            mask_coords=mask_coords,
        )
    if mode == "boundary_band_restore":
        if source_coords is None:
            raise RuntimeError("ss.postprocess.mode='boundary_band_restore' requires source voxels.")
        if mask_coords is None:
            raise RuntimeError("ss.postprocess.mode='boundary_band_restore' requires mask_glb.")
        return compose_stage1_coords_boundary_band_restore(
            coords_source=source_coords,
            coords_stage1_raw=coords_stage1_raw,
            mask_coords=mask_coords,
            band_width_voxels=boundary_band_width_voxels,
            band_target_neighbor_threshold=boundary_band_target_neighbor_threshold,
        )
    raise RuntimeError(f"Unsupported ss.postprocess.mode: {mode}")


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


def _get_slat_norm_tensors(
    pipeline,
    device: torch.device,
    dtype: torch.dtype,
) -> tuple[torch.Tensor, torch.Tensor]:
    norm = pipeline.slat_normalization
    if isinstance(norm, dict):
        mean_val = norm["mean"]
        std_val = norm["std"]
        mean = (
            torch.tensor(mean_val, device=device, dtype=dtype)
            if isinstance(mean_val, (list, tuple))
            else mean_val.to(device=device, dtype=dtype)
        )
        std = (
            torch.tensor(std_val, device=device, dtype=dtype)
            if isinstance(std_val, (list, tuple))
            else std_val.to(device=device, dtype=dtype)
        )
        return mean, std
    if isinstance(norm, (list, tuple)):
        return (
            torch.tensor(norm[0], device=device, dtype=dtype),
            torch.tensor(norm[1], device=device, dtype=dtype),
        )
    raise TypeError(f"Unexpected slat_normalization type: {type(norm)}")


def _invert_sparse_structure_with_rf(
    pipeline,
    *,
    cond_src: dict[str, Any],
    voxel_src: torch.Tensor,
    params: dict[str, Any],
    cfg_interval: tuple[float, float],
    start_step: int | None = None,
    verbose: bool = False,
) -> torch.Tensor:
    encoder = pipeline.models["sparse_structure_encoder"]
    flow_model = pipeline.models["sparse_structure_flow_model"]
    z_src = encoder(voxel_src)
    sampler = RFSolverSampler()
    return sampler.sample(
        model=flow_model,
        sample=z_src,
        cond_dict=cond_src,
        steps=params["steps"],
        rescale_t=params["rescale_t"],
        cfg_strength=params["cfg_strength"],
        cfg_interval=cfg_interval,
        inverse=True,
        start_step=start_step,
        verbose=verbose,
    )


def _invert_slat_with_rf(
    pipeline,
    *,
    cond_src: dict[str, Any],
    slat_src,
    params: dict[str, Any],
    cfg_interval: tuple[float, float],
    start_step: int | None = None,
    verbose: bool = False,
):
    flow_model = pipeline.models["slat_flow_model"]
    mean, std = _get_slat_norm_tensors(pipeline, slat_src.device, slat_src.feats.dtype)
    slat_normalized = (slat_src - mean) / std
    sampler = RFSolverSampler()
    return sampler.sample(
        model=flow_model,
        sample=slat_normalized,
        cond_dict=cond_src,
        steps=params["steps"],
        rescale_t=params["rescale_t"],
        cfg_strength=params["cfg_strength"],
        cfg_interval=cfg_interval,
        inverse=True,
        start_step=start_step,
        verbose=verbose,
    )


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


class _ControlledDenoiseHookSupportMixin(ControlledDenoiseCommonMixin):
    def __init__(self) -> None:
        self.hook = None

    def cleanup(self) -> None:
        if self.hook is not None:
            self.hook.restore()
            self.hook = None

    @staticmethod
    def _patch_stage_hooks(pipeline, stage_configs: dict[str, Any], hook) -> None:
        if hook is None:
            return
        if "ss" in stage_configs:
            hook.patch_model(pipeline.models["sparse_structure_flow_model"], "ss")
        if "slat" in stage_configs:
            hook.patch_model(pipeline.models["slat_flow_model"], "slat")


class UniEditSSAdapter(_ControlledDenoiseHookSupportMixin, SSStagePlugin):
    name = "uniedit"

    def run(
        self,
        context: ExperimentContext,
        preprocess,
        config: SSStageConfig,
    ) -> SSArtifact:
        if not config.controls.uniedit.enabled:
            raise RuntimeError("UniEdit SS adapter requires ss.controls.uniedit.enabled=true")
        source_voxels_path = context.config.inputs.source_voxels
        if source_voxels_path is None:
            raise RuntimeError("UniEdit SS requires inputs.source_voxels")

        pipeline = context.pipeline
        resolution = pipeline.sparse_structure_sampler_params.get("grid_size", 64)
        ss_params = {
            **getattr(pipeline, "sparse_structure_sampler_params", {}),
            **_sampler_params(config.sampler),
        }
        ss_inversion_steps = resolve_inversion_steps(
            total_steps=int(ss_params.get("steps", 25)),
            inversion_steps=config.inversion.inversion_steps,
        )

        source_coords = ply_to_coords(source_voxels_path, pipeline.device, resolution)
        mask_coords = _load_mask_coords(context, pipeline, resolution=resolution)

        source_image, edit_image, _ = _prepared_images(preprocess)
        source_cond_dict = pipeline.get_cond([source_image])
        edit_cond_dict = pipeline.get_cond([edit_image])
        source_cond = source_cond_dict["cond"]
        edit_cond = edit_cond_dict["cond"]
        neg_cond = source_cond_dict["neg_cond"]

        token_meta = None
        if config.controls.p2p.enabled:
            extra = {
                "ss_t_start": config.controls.p2p.t_start,
                "ss_t_end": config.controls.p2p.t_end,
                "ss_strength": config.controls.p2p.strength,
                "patch_coverage_threshold": config.controls.p2p.patch_coverage_threshold,
            }
            hook_inputs = _P2PHookInputs(
                source_image=source_image,
                edit_image=edit_image,
                mask_image=preprocess.prepared_inputs.mask,
            )
            self.hook, token_meta, stage_configs = self._create_p2p_hook(
                inputs=hook_inputs,
                extra=extra,
                source_cond_dict=source_cond_dict,
                edit_cond_dict=edit_cond_dict,
                enabled_stages=["ss"],
            )
            self._patch_stage_hooks(pipeline, stage_configs, self.hook)

        _release_cuda_memory()

        print("Stage 0: Inverting source sparse structure...")
        source_voxel = coords_to_voxel(source_coords, pipeline.device, resolution)
        ss_terminal_noise = _invert_sparse_structure_with_rf(
            pipeline=pipeline,
            cond_src={"cond": source_cond, "neg_cond": neg_cond},
            voxel_src=source_voxel,
            params=ss_params,
            cfg_interval=config.controls.uniedit.cfg_interval,
            start_step=ss_inversion_steps,
            verbose=True,
        )

        _release_cuda_memory()

        print(f"SS Stage: Editing sparse structure (omega={config.controls.uniedit.omega})...")
        coords_ss_raw = self._denoise_sparse_structure_uniedit(
            pipeline=pipeline,
            source_cond={"cond": source_cond, "neg_cond": neg_cond},
            target_cond={"cond": edit_cond, "neg_cond": neg_cond},
            terminal_noise=ss_terminal_noise,
            params=ss_params,
            cfg_interval=config.controls.uniedit.cfg_interval,
            omega=config.controls.uniedit.omega,
            start_step=ss_inversion_steps,
        )

        coords_ss_masked, _, ss_meta = _apply_ss_postprocess(
            mode=config.postprocess.mode,
            source_coords=source_coords,
            coords_stage1_raw=coords_ss_raw,
            mask_coords=mask_coords,
            boundary_band_width_voxels=config.postprocess.boundary_band.band_width_voxels,
            boundary_band_target_neighbor_threshold=(
                config.postprocess.boundary_band.band_target_neighbor_threshold
            ),
        )
        print(f"SS Stage complete: {ss_meta['stage1_masked_voxel_count']} voxels")

        _release_cuda_memory()

        voxel_mesh = None
        if config.output.save_voxel_mesh:
            voxel_mesh = coords_to_cubic_mesh(coords_ss_masked, resolution)
            print(f"Generated voxel mesh: {len(voxel_mesh.faces)} faces")

        return SSArtifact(
            plugin_name=self.name,
            coords=coords_ss_masked,
            voxel_mesh=voxel_mesh,
            metadata={
                "ss_meta": ss_meta,
                "ss_controls": {
                    "p2p_enabled": config.controls.p2p.enabled,
                    "latent_blend_enabled": config.controls.latent_blend.enabled,
                    "uniedit_enabled": True,
                },
                "ss_omega": config.controls.uniedit.omega,
                "cfg_interval": config.controls.uniedit.cfg_interval,
                "ss_postprocess_mode": config.postprocess.mode,
                "ss_total_steps": int(ss_params.get("steps", 25)),
                "ss_inversion_steps": ss_inversion_steps,
                "resolution": resolution,
                "voxel_count": int(len(coords_ss_masked)),
                **_ss_voxel_mesh_export_meta(config.output.save_voxel_mesh),
                **({"token_meta": token_meta} if token_meta is not None else {}),
            },
        )

    def save(self, artifact: SSArtifact, out_dir: Path, config: SSStageConfig) -> dict[str, str]:
        return _save_ss_artifacts(
            artifact,
            out_dir,
            output_format=config.output.output_format,
            save_mesh=config.output.save_voxel_mesh,
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
        start_step: int | None = None,
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
            start_step=start_step,
            selector=None,
            mode="full_uniedit",
            verbose=True,
        )

        voxel = decoder(z_tgt)
        coords = torch.argwhere(voxel > 0)[:, [0, 2, 3, 4]].int()
        if coords.shape[0] == 0:
            raise RuntimeError("SS stage UniEdit denoising produced an empty structure.")
        return coords


class FlowEditSSAdapter(SSStagePlugin):
    name = "flowedit"

    def run(
        self,
        context: ExperimentContext,
        preprocess,
        config: SSStageConfig,
    ) -> SSArtifact:
        if context.config.runtime.num_samples != 1:
            raise RuntimeError(
                "ss.method='flowedit' currently supports runtime.num_samples=1 only."
            )

        source_voxels_path = context.config.inputs.source_voxels
        if source_voxels_path is None:
            raise RuntimeError("FlowEdit SS requires inputs.source_voxels")

        pipeline = context.pipeline
        resolution = int(pipeline.sparse_structure_sampler_params.get("grid_size", 64))
        ss_params = {
            **getattr(pipeline, "sparse_structure_sampler_params", {}),
            **_sampler_params(config.sampler),
        }

        ensure_pipeline_encoders(
            pipeline,
            model_root=context.config.runtime.model,
            require_sparse_structure_encoder=True,
        )

        source_coords = ply_to_coords(source_voxels_path, pipeline.device, resolution)
        print(f"Loaded source coords: {source_coords.shape[0]} voxels")
        mask_coords = _load_mask_coords(context, pipeline, resolution=resolution)
        if mask_coords is not None:
            print(f"Loaded SS postprocess mask: {mask_coords.shape[0]} voxels")

        source_image, edit_image, _ = _prepared_images(preprocess)
        source_cond_dict = pipeline.get_cond([source_image])
        edit_cond_dict = pipeline.get_cond([edit_image])

        source_voxel = coords_to_voxel(source_coords, pipeline.device, resolution)
        encoder = pipeline.models["sparse_structure_encoder"]
        flow_model = pipeline.models["sparse_structure_flow_model"]
        decoder = pipeline.models["sparse_structure_decoder"]

        _release_cuda_memory()
        print("Encoding source sparse structure...")
        source_latent = encoder(source_voxel)

        flowedit_sampler = FlowEditSampler()
        print("Running FlowEdit sparse-structure editing...")
        edited_latent = flowedit_sampler.sample(
            sampler=pipeline.sparse_structure_sampler,
            model=flow_model,
            source_latent=source_latent,
            source_cond=source_cond_dict["cond"],
            target_cond=edit_cond_dict["cond"],
            neg_cond=source_cond_dict["neg_cond"],
            steps=int(ss_params.get("steps", 25)),
            rescale_t=float(ss_params.get("rescale_t", 1.0)),
            start_step=int(config.flowedit.start_step),
            n_avg=int(config.flowedit.n_avg),
            source_cfg_strength=float(config.flowedit.src_cfg_strength),
            target_cfg_strength=float(config.flowedit.tar_cfg_strength),
            cfg_interval=tuple(config.flowedit.cfg_interval),
            verbose=True,
        )

        _release_cuda_memory()
        print("Decoding FlowEdit sparse-structure result...")
        voxel = decoder(edited_latent)
        coords_edited_raw = torch.argwhere(voxel > 0)[:, [0, 2, 3, 4]].int()
        del voxel, edited_latent, source_latent, source_voxel
        _release_cuda_memory()

        if coords_edited_raw.shape[0] == 0:
            raise RuntimeError("FlowEdit SS produced an empty structure.")

        coords_edited, _, ss_meta = _apply_ss_postprocess(
            mode=config.postprocess.mode,
            source_coords=source_coords,
            coords_stage1_raw=coords_edited_raw,
            mask_coords=mask_coords,
            boundary_band_width_voxels=config.postprocess.boundary_band.band_width_voxels,
            boundary_band_target_neighbor_threshold=(
                config.postprocess.boundary_band.band_target_neighbor_threshold
            ),
        )
        print(f"FlowEdit SS complete: {ss_meta['stage1_masked_voxel_count']} voxels")

        voxel_mesh = None
        if config.output.save_voxel_mesh:
            voxel_mesh = coords_to_cubic_mesh(coords_edited, resolution)
            print(f"Generated voxel mesh: {len(voxel_mesh.faces)} faces")

        return SSArtifact(
            plugin_name=self.name,
            coords=coords_edited,
            voxel_mesh=voxel_mesh,
            metadata={
                "ss_method": self.name,
                "ss_postprocess_mode": config.postprocess.mode,
                "ss_total_steps": int(ss_params.get("steps", 25)),
                "ss_flowedit_n_avg": int(config.flowedit.n_avg),
                "ss_flowedit_start_step": int(config.flowedit.start_step),
                "ss_flowedit_src_cfg_strength": float(config.flowedit.src_cfg_strength),
                "ss_flowedit_tar_cfg_strength": float(config.flowedit.tar_cfg_strength),
                "ss_flowedit_cfg_interval": tuple(config.flowedit.cfg_interval),
                "resolution": resolution,
                "voxel_count": int(coords_edited.shape[0]),
                "ss_meta": ss_meta,
                **_ss_voxel_mesh_export_meta(config.output.save_voxel_mesh),
            },
        )

    def save(self, artifact: SSArtifact, out_dir: Path, config: SSStageConfig) -> dict[str, str]:
        return _save_ss_artifacts(
            artifact,
            out_dir,
            output_format=config.output.output_format,
            save_mesh=config.output.save_voxel_mesh,
        )


class AnchorFlowSSAdapter(SSStagePlugin):
    name = "anchorflow"

    def run(
        self,
        context: ExperimentContext,
        preprocess,
        config: SSStageConfig,
    ) -> SSArtifact:
        if context.config.runtime.num_samples != 1:
            raise RuntimeError(
                "ss.method='anchorflow' currently supports runtime.num_samples=1 only."
            )

        source_voxels_path = context.config.inputs.source_voxels
        if source_voxels_path is None:
            raise RuntimeError("AnchorFlow SS requires inputs.source_voxels")

        pipeline = context.pipeline
        resolution = int(pipeline.sparse_structure_sampler_params.get("grid_size", 64))
        ss_params = {
            **getattr(pipeline, "sparse_structure_sampler_params", {}),
            **_sampler_params(config.sampler),
        }

        ensure_pipeline_encoders(
            pipeline,
            model_root=context.config.runtime.model,
            require_sparse_structure_encoder=True,
        )

        source_coords = ply_to_coords(source_voxels_path, pipeline.device, resolution)
        print(f"Loaded source coords: {source_coords.shape[0]} voxels")
        mask_coords = _load_mask_coords(context, pipeline, resolution=resolution)
        if mask_coords is not None:
            print(f"Loaded SS postprocess mask: {mask_coords.shape[0]} voxels")

        source_image, edit_image, _ = _prepared_images(preprocess)
        source_cond_dict = pipeline.get_cond([source_image])
        edit_cond_dict = pipeline.get_cond([edit_image])

        source_voxel = coords_to_voxel(source_coords, pipeline.device, resolution)
        encoder = pipeline.models["sparse_structure_encoder"]
        flow_model = pipeline.models["sparse_structure_flow_model"]
        decoder = pipeline.models["sparse_structure_decoder"]

        _release_cuda_memory()
        print("Encoding source sparse structure...")
        source_latent = encoder(source_voxel)

        anchorflow_sampler = AnchorFlowSampler()
        print("Running AnchorFlow sparse-structure editing...")
        edited_latent = anchorflow_sampler.sample(
            sampler=pipeline.sparse_structure_sampler,
            model=flow_model,
            source_latent=source_latent,
            source_cond=source_cond_dict["cond"],
            target_cond=edit_cond_dict["cond"],
            neg_cond=source_cond_dict["neg_cond"],
            steps=int(ss_params.get("steps", 25)),
            rescale_t=float(ss_params.get("rescale_t", 1.0)),
            n_max=int(config.anchorflow.n_max),
            n_avg=int(config.anchorflow.n_avg),
            source_cfg_strength=float(config.anchorflow.src_cfg_strength),
            target_cfg_strength=float(config.anchorflow.tar_cfg_strength),
            cfg_interval=tuple(config.anchorflow.cfg_interval),
            center_weight=float(config.anchorflow.center_weight),
            band_weight=float(config.anchorflow.band_weight),
            ortho_weight=float(config.anchorflow.ortho_weight),
            residual_weight=float(config.anchorflow.residual_weight),
            band_min_ratio=float(config.anchorflow.band_min_ratio),
            band_max_ratio=float(config.anchorflow.band_max_ratio),
            margin_scale=float(config.anchorflow.margin_scale),
            direction_gate_tau=float(config.anchorflow.direction_gate_tau),
            eps=float(config.anchorflow.eps),
            anchor_noise=bool(config.anchorflow.anchor_noise),
            verbose=True,
        )

        _release_cuda_memory()
        print("Decoding AnchorFlow sparse-structure result...")
        voxel = decoder(edited_latent)
        coords_edited_raw = torch.argwhere(voxel > 0)[:, [0, 2, 3, 4]].int()
        del voxel, edited_latent, source_latent, source_voxel
        _release_cuda_memory()

        if coords_edited_raw.shape[0] == 0:
            raise RuntimeError("AnchorFlow SS produced an empty structure.")

        coords_edited, _, ss_meta = _apply_ss_postprocess(
            mode=config.postprocess.mode,
            source_coords=source_coords,
            coords_stage1_raw=coords_edited_raw,
            mask_coords=mask_coords,
            boundary_band_width_voxels=config.postprocess.boundary_band.band_width_voxels,
            boundary_band_target_neighbor_threshold=(
                config.postprocess.boundary_band.band_target_neighbor_threshold
            ),
        )
        print(f"AnchorFlow SS complete: {ss_meta['stage1_masked_voxel_count']} voxels")

        voxel_mesh = None
        if config.output.save_voxel_mesh:
            voxel_mesh = coords_to_cubic_mesh(coords_edited, resolution)
            print(f"Generated voxel mesh: {len(voxel_mesh.faces)} faces")

        return SSArtifact(
            plugin_name=self.name,
            coords=coords_edited,
            voxel_mesh=voxel_mesh,
            metadata={
                "ss_method": self.name,
                "ss_postprocess_mode": config.postprocess.mode,
                "ss_total_steps": int(ss_params.get("steps", 25)),
                "ss_anchorflow_n_avg": int(config.anchorflow.n_avg),
                "ss_anchorflow_n_max": int(config.anchorflow.n_max),
                "ss_anchorflow_src_cfg_strength": float(config.anchorflow.src_cfg_strength),
                "ss_anchorflow_tar_cfg_strength": float(config.anchorflow.tar_cfg_strength),
                "ss_anchorflow_cfg_interval": tuple(config.anchorflow.cfg_interval),
                "ss_anchorflow_center_weight": float(config.anchorflow.center_weight),
                "ss_anchorflow_band_weight": float(config.anchorflow.band_weight),
                "ss_anchorflow_ortho_weight": float(config.anchorflow.ortho_weight),
                "ss_anchorflow_residual_weight": float(config.anchorflow.residual_weight),
                "ss_anchorflow_band_min_ratio": float(config.anchorflow.band_min_ratio),
                "ss_anchorflow_band_max_ratio": float(config.anchorflow.band_max_ratio),
                "ss_anchorflow_margin_scale": float(config.anchorflow.margin_scale),
                "ss_anchorflow_direction_gate_tau": float(config.anchorflow.direction_gate_tau),
                "ss_anchorflow_eps": float(config.anchorflow.eps),
                "ss_anchorflow_anchor_noise": bool(config.anchorflow.anchor_noise),
                "resolution": resolution,
                "voxel_count": int(coords_edited.shape[0]),
                "ss_meta": ss_meta,
                **_ss_voxel_mesh_export_meta(config.output.save_voxel_mesh),
            },
        )

    def save(self, artifact: SSArtifact, out_dir: Path, config: SSStageConfig) -> dict[str, str]:
        return _save_ss_artifacts(
            artifact,
            out_dir,
            output_format=config.output.output_format,
            save_mesh=config.output.save_voxel_mesh,
        )


class _ControlledDenoisePluginBase(_ControlledDenoiseHookSupportMixin):

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
        return_terminal_noise: bool = False,
        hook: Any | None = None,
    ) -> SourceTrace | tuple[SourceTrace, torch.Tensor]:
        if torch.cuda.is_available():
            print(f"[Memory] Before inversion - Allocated: {torch.cuda.memory_allocated()/1024**3:.2f} GB")
        terminal_noise = None
        if return_terminal_noise:
            source_trace, terminal_noise = self._prepare_ss_latent_and_terminal_from_coords(
                pipeline,
                source_coords,
                source_cond_dict,
                inversion_mode,
                config,
                resolution,
                hook=hook,
            )
        else:
            source_trace = self._prepare_ss_latent_from_coords(
                pipeline,
                source_coords,
                source_cond_dict,
                inversion_mode,
                config,
                resolution,
                hook=hook,
            )
        if torch.cuda.is_available():
            print(f"[Memory] After inversion - Allocated: {torch.cuda.memory_allocated()/1024**3:.2f} GB")
        print(f"[Memory] Source trace size: {len(source_trace)} timesteps")
        if return_terminal_noise:
            return source_trace, terminal_noise
        return source_trace

    @staticmethod
    def _blend_coords_with_mask(
        source_coords: torch.Tensor,
        edit_coords: torch.Tensor,
        mask_coords: torch.Tensor,
    ) -> torch.Tensor:
        mask_hash_set = _ControlledDenoisePluginBase._build_mask_hash_set(mask_coords)
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

    def _build_slat_blend_mask(
        self,
        *,
        source_slat,
        edit_coords: torch.Tensor,
        mask_coords: torch.Tensor,
        resolution: int,
        soft_mask_enabled: bool,
        soft_mask_dilation: int,
        soft_mask_sigma: float,
    ) -> torch.Tensor | SparseLatentBlendMask:
        preserve_coords = self._build_slat_preserve_coords_from_mask(source_slat, mask_coords)
        if not soft_mask_enabled:
            return preserve_coords

        edit_weights = self._build_slat_soft_edit_weights(
            edit_coords,
            preserve_coords,
            resolution=resolution,
            dilation=soft_mask_dilation,
            sigma=soft_mask_sigma,
        )
        return SparseLatentBlendMask(
            coords=preserve_coords,
            edit_weights=edit_weights,
        )

    def _build_slat_nano3d_replace_coords(
        self,
        *,
        source_slat,
        edit_coords: torch.Tensor,
        mask_coords: torch.Tensor,
        resolution: int,
    ) -> torch.Tensor:
        preserve_coords = self._build_slat_preserve_coords_from_mask(source_slat, mask_coords)
        if preserve_coords.shape[0] == 0:
            return preserve_coords

        preserve_codes = coords_to_flat_indices(preserve_coords, resolution)
        edit_codes = coords_to_flat_indices(edit_coords, resolution)
        overlap_mask = torch.isin(preserve_codes, edit_codes)
        if not overlap_mask.any():
            return preserve_coords[:0]
        return preserve_coords[overlap_mask]

    @staticmethod
    def _apply_final_slat_feature_blend(
        edit_slat,
        source_slat,
        blend_mask: torch.Tensor | SparseLatentBlendMask,
    ) -> tuple[Any, dict[str, int]]:
        return blend_sparse_features(edit_slat, source_slat, blend_mask)

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
        resolution: int,
        inversion_scope: str = "full_source",
        preserve_coords: torch.Tensor | None = None,
        return_terminal_noise: bool = False,
        hook: Any | None = None,
    ) -> SourceTrace | tuple[SourceTrace, Any]:
        source_for_inversion = source_slat
        if inversion_scope == "preserve_only":
            if preserve_coords is None:
                raise RuntimeError("SLAT inversion scope 'preserve_only' requires preserve coordinates.")
            if preserve_coords.shape[0] == 0:
                print("[WARN] SLAT preserve set is empty; skipping SLAT inversion cache generation.")
                if return_terminal_noise:
                    return SourceTrace(), None
                return SourceTrace()
            source_codes = coords_to_flat_indices(source_slat.coords, resolution)
            preserve_codes = coords_to_flat_indices(preserve_coords, resolution)
            keep_mask = torch.isin(source_codes, preserve_codes)
            if not keep_mask.any():
                print("[WARN] No overlapping preserve coordinates found in source SLAT; skipping inversion cache generation.")
                if return_terminal_noise:
                    return SourceTrace(), None
                return SourceTrace()
            source_for_inversion = source_slat.replace(
                source_slat.feats[keep_mask],
                source_slat.coords[keep_mask],
            )
        std = torch.tensor(pipeline.slat_normalization["std"], device=pipeline.device)[None]
        mean = torch.tensor(pipeline.slat_normalization["mean"], device=pipeline.device)[None]
        source_slat_normalized = (source_for_inversion - mean) / std
        print(
            "Running "
            f"{self._solver_display_name(inversion_mode)} inversion for SLAT "
            f"(scope={inversion_scope})..."
        )
        extra = config.extra_params or {}
        stage_params = self._resolve_stage_sampler_params(
            "slat",
            config.slat_sampler_params,
            extra,
        )
        sampler_params = {**getattr(pipeline, "slat_sampler_params", {}), **stage_params}
        return self._invert_sample(
            model=pipeline.models["slat_flow_model"],
            sample=source_slat_normalized,
            cond_dict=source_cond_dict,
            steps=int(sampler_params.get("steps", 25)),
            rescale_t=float(sampler_params.get("rescale_t", 3.0)),
            stage_prefix="slat",
            config=config,
            inversion_mode=inversion_mode,
            return_terminal_noise=return_terminal_noise,
            hook=hook,
        )

    @staticmethod
    def _denormalize_slat_sample(pipeline, slat_normalized):
        device = getattr(slat_normalized, "device", None)
        if device is None:
            device = slat_normalized.feats.device
        std = torch.tensor(pipeline.slat_normalization["std"], device=device)[None]
        mean = torch.tensor(pipeline.slat_normalization["mean"], device=device)[None]
        return slat_normalized * std + mean

    @staticmethod
    def _denoise_slat_from_terminal_noise(
        pipeline,
        *,
        terminal_noise,
        edit_cond_dict: dict[str, Any],
        slat_params: dict[str, Any],
    ):
        flow_model = pipeline.models["slat_flow_model"]
        slat_normalized = pipeline.slat_sampler.sample(
            flow_model,
            terminal_noise,
            **edit_cond_dict,
            **slat_params,
            verbose=True,
        ).samples

        return _ControlledDenoisePluginBase._denormalize_slat_sample(pipeline, slat_normalized)

    def _run_slat_stage(
        self,
        pipeline,
        *,
        source_slat,
        source_coords: torch.Tensor | None,
        source_cond_dict: dict[str, Any] | None,
        edit_cond_dict: dict[str, Any],
        edit_coords: torch.Tensor,
        mask_coords: torch.Tensor | None,
        resolution: int,
        blend_slat_enabled: bool,
        nano3d_replace_enabled: bool,
        kv_blend_enabled: bool,
        inversion_enabled: bool,
        inversion_mode: str,
        soft_mask_enabled: bool,
        soft_mask_dilation: int,
        soft_mask_sigma: float,
        config: _StageRunConfig,
        verbose: bool,
        hook: Any | None = None,
    ) -> tuple[Any, dict[str, int] | None]:
        extra = config.extra_params or {}
        slat_params = self._resolve_stage_sampler_params(
            "slat",
            config.slat_sampler_params,
            extra,
        )
        denoise_init = str(extra.get("slat_denoise_init", "terminal_noise"))
        inversion_scope = str(extra.get("slat_inversion_scope", "full_source"))

        source_slat_trace = None
        source_slat_terminal_noise = None
        original_slat_sampler = pipeline.slat_sampler
        slat_blend_mask: torch.Tensor | SparseLatentBlendMask | None = None
        nano3d_replace_coords: torch.Tensor | None = None
        final_blend_stats: dict[str, int] | None = None
        projected_slat_noise = None
        try:
            preserve_coords = None
            if source_slat is not None and mask_coords is not None:
                preserve_coords = self._build_slat_preserve_coords_from_mask(source_slat, mask_coords)

            if nano3d_replace_enabled:
                if source_slat is None:
                    raise RuntimeError("Nano3D SLAT replace requires inputs.source_features.")
                if mask_coords is None:
                    raise RuntimeError("Nano3D SLAT replace requires inputs.mask_glb.")
                nano3d_replace_coords = self._build_slat_nano3d_replace_coords(
                    source_slat=source_slat,
                    edit_coords=edit_coords,
                    mask_coords=mask_coords,
                    resolution=resolution,
                )
                print(
                    "Prepared Nano3D SLAT final replacement set: "
                    f"{nano3d_replace_coords.shape[0]} overlapping outside-mask coords"
                )

            if denoise_init not in {"terminal_noise", "random_noise"}:
                raise ValueError(f"Unknown slat_denoise_init: {denoise_init}")

            if inversion_enabled:
                if source_slat is None or source_cond_dict is None:
                    raise RuntimeError("SLAT inversion requires source SLAT features and source conditioning.")
                inversion_label = (
                    "source SLAT terminal noise"
                    if denoise_init == "terminal_noise"
                    else "source SLAT inversion cache"
                )
                print(f"Preparing {inversion_label} via inversion...")
                _release_cuda_memory()
                prepare_result = self._prepare_slat_source(
                    pipeline,
                    source_slat,
                    source_cond_dict,
                    inversion_mode,
                    config,
                    resolution=resolution,
                    inversion_scope=inversion_scope,
                    preserve_coords=preserve_coords,
                    return_terminal_noise=denoise_init == "terminal_noise",
                    hook=hook,
                )
                if denoise_init == "terminal_noise":
                    source_slat_trace, source_slat_terminal_noise = prepare_result
                else:
                    source_slat_trace = prepare_result
                    source_slat_terminal_noise = None
                if verbose:
                    print(f"Built {len(source_slat_trace)} SLAT source-trace entries")
                _release_cuda_memory()
            else:
                print("Skipping SLAT source inversion; using pure target denoising.")

            if denoise_init == "terminal_noise":
                if source_slat_terminal_noise is None:
                    raise RuntimeError(
                        "slat.inversion.denoise_init='terminal_noise' requires a terminal noise cache."
                    )
                print("Projecting source SLAT terminal noise to edited coordinates...")
                projected_slat_noise = project_sparse_terminal_noise(
                    source_noise=source_slat_terminal_noise,
                    target_coords=edit_coords.to(device=pipeline.device),
                    preserve_coords=preserve_coords,
                    device=pipeline.device,
                    SparseTensor=type(source_slat_terminal_noise),
                    resolution=resolution,
                )
                _release_cuda_memory()
            else:
                print("Using random-noise SLAT init.")

            if blend_slat_enabled and source_slat_trace is not None and mask_coords is not None:
                print("Setting up SLAT latent blending...")
                slat_blend_mask = self._build_slat_blend_mask(
                    source_slat=source_slat,
                    edit_coords=edit_coords,
                    mask_coords=mask_coords,
                    resolution=resolution,
                    soft_mask_enabled=soft_mask_enabled,
                    soft_mask_dilation=soft_mask_dilation,
                    soft_mask_sigma=soft_mask_sigma,
                )
                if not (kv_blend_enabled or self._is_rf_family_solver(inversion_mode)):
                    slat_sampler = LatentBlendFlowEulerGuidanceIntervalSampler(
                        sigma_min=original_slat_sampler.sigma_min
                    )
                    slat_sampler.set_blend_source(
                        source_trace=source_slat_trace,
                        latent_mask=slat_blend_mask,
                        is_sparse=True,
                    )
                    pipeline.slat_sampler = slat_sampler
            else:
                print("SLAT blending disabled")

            slat_total_steps = int(slat_params.get("steps", 25))
            slat_denoise_start_step = (
                self._resolve_inversion_steps("slat", slat_total_steps, extra)
                if denoise_init == "terminal_noise"
                else slat_total_steps
            )
            use_custom_slat_loop = bool(
                kv_blend_enabled
                or self._is_rf_family_solver(inversion_mode)
                or slat_denoise_start_step < slat_total_steps
            )
            if denoise_init == "terminal_noise":
                print("Generating edit SLAT from projected terminal noise...")
                if use_custom_slat_loop:
                    edit_slat_normalized = self._denoise_slat_with_step_blend(
                        pipeline,
                        edit_cond_dict,
                        config,
                        edit_coords=edit_coords,
                        initial_sample=projected_slat_noise,
                        source_trace=(
                            source_slat_trace
                            if (blend_slat_enabled or kv_blend_enabled)
                            else None
                        ),
                        source_cond_dict=source_cond_dict,
                        latent_mask=slat_blend_mask,
                        solver_mode=inversion_mode,
                        verbose=True,
                        hook=hook,
                    )
                    edit_slat = self._denormalize_slat_sample(pipeline, edit_slat_normalized)
                else:
                    edit_slat = self._denoise_slat_from_terminal_noise(
                        pipeline,
                        terminal_noise=projected_slat_noise,
                        edit_cond_dict=edit_cond_dict,
                        slat_params=slat_params,
                    )
            else:
                print("Generating edit SLAT from random noise...")
                if use_custom_slat_loop:
                    edit_slat_normalized = self._denoise_slat_with_step_blend(
                        pipeline,
                        edit_cond_dict,
                        config,
                        edit_coords=edit_coords,
                        initial_sample=None,
                        source_trace=source_slat_trace if kv_blend_enabled else None,
                        source_cond_dict=source_cond_dict,
                        latent_mask=None,
                        solver_mode=inversion_mode,
                        verbose=True,
                        hook=hook,
                    )
                    edit_slat = self._denormalize_slat_sample(pipeline, edit_slat_normalized)
                else:
                    edit_slat = pipeline.sample_slat(
                        edit_cond_dict,
                        edit_coords,
                        sampler_params=slat_params,
                    )
            if verbose:
                print(f"Edit SLAT: {edit_slat.coords.shape[0]} voxels")
            if slat_blend_mask is not None:
                print("Applying final SLAT feature merge...")
                edit_slat, final_blend_stats = self._apply_final_slat_feature_blend(
                    edit_slat,
                    source_slat,
                    slat_blend_mask,
                )
                print(
                    "Final SLAT feature merge matched "
                    f"{final_blend_stats['valid_matches']} preserve voxels"
                )
            elif nano3d_replace_coords is not None:
                print("Applying final Nano3D SLAT feature replacement...")
                edit_slat, final_blend_stats = self._apply_final_slat_feature_blend(
                    edit_slat,
                    source_slat,
                    nano3d_replace_coords,
                )
                print(
                    "Final Nano3D SLAT replacement matched "
                    f"{final_blend_stats['valid_matches']} overlapping outside-mask voxels"
                )

            return edit_slat, final_blend_stats
        finally:
            pipeline.slat_sampler = original_slat_sampler
            if source_slat_terminal_noise is not None:
                del source_slat_terminal_noise
            if projected_slat_noise is not None:
                del projected_slat_noise
            if source_slat_trace is not None:
                del source_slat_trace
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

        source_slat_cpu = None
        if verbose and source_slat is not None:
            print("Moving source SLAT to CPU...")
        if source_slat is not None:
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

            if skip_source_decode or source_slat_cpu is None:
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


class ControlledDenoiseSSAdapter(_ControlledDenoisePluginBase, SSStagePlugin):
    name = "controlled_denoise"

    def run(
        self,
        context: ExperimentContext,
        preprocess,
        config: SSStageConfig,
    ) -> SSArtifact:
        if config.controls.uniedit.enabled:
            raise RuntimeError("controlled_denoise SS adapter does not accept ss.controls.uniedit.enabled=true")
        source_voxels_path = context.config.inputs.source_voxels
        source_coords = None
        mask_coords = None

        pipeline = context.pipeline
        resolution = pipeline.sparse_structure_sampler_params.get("grid_size", 64)
        if source_voxels_path is not None:
            source_coords = ply_to_coords(source_voxels_path, pipeline.device, resolution)
            print(f"Source coords: {len(source_coords)} voxels")

        mask_coords = _load_mask_coords(context, pipeline, resolution=resolution)
        if mask_coords is not None:
            print(f"Mask coords: {len(mask_coords)} voxels")

        source_image, edit_image, mask_image = _prepared_images(preprocess)
        edit_cond_dict = pipeline.get_cond([edit_image])
        source_cond_dict = None
        ss_edit_region_mode = _resolve_ss_edit_region_mode(config)
        if config.inversion.enabled or config.controls.p2p.enabled or config.controls.kv_blend.enabled:
            source_cond_dict = pipeline.get_cond([source_image])

        extra = {
            "ss_t_start": config.controls.p2p.t_start,
            "ss_t_end": config.controls.p2p.t_end,
            "ss_strength": config.controls.p2p.strength,
            "patch_coverage_threshold": config.controls.p2p.patch_coverage_threshold,
            "ss_kv_t_start": config.controls.kv_blend.t_start,
            "ss_kv_t_end": config.controls.kv_blend.t_end,
            "ss_kv_self_attention": config.controls.kv_blend.self_attention,
            "ss_kv_cross_attention": config.controls.kv_blend.cross_attention,
            "ss_predictor_corrector_steps": int(config.inversion.predictor_corrector_steps),
            "ss_inversion_steps": config.inversion.inversion_steps,
            "ss_denoise_cfg_strength": config.inversion.denoise_cfg_strength,
            "ss_denoise_cfg_interval_start": config.inversion.denoise_cfg_interval[0],
            "ss_denoise_cfg_interval_end": config.inversion.denoise_cfg_interval[1],
            "ss_inversion_cfg_strength": config.inversion.inversion_cfg_strength,
            "ss_inversion_cfg_interval_start": config.inversion.inversion_cfg_interval[0],
            "ss_inversion_cfg_interval_end": config.inversion.inversion_cfg_interval[1],
            "ss_edit_region_mode": ss_edit_region_mode,
        }
        token_meta = None
        stage_configs = {}
        if config.controls.kv_blend.enabled:
            hook_inputs = _P2PHookInputs(
                source_image=source_image,
                edit_image=edit_image,
                mask_image=mask_image,
            )
            stage_masks = {
                "ss": {
                    "self": self._build_ss_self_kv_token_mask(
                        mask_coords,
                        source_coords=source_coords,
                        resolution=resolution,
                        latent_resolution=pipeline.models["sparse_structure_flow_model"].resolution,
                        hard_mask_mode=config.controls.latent_blend.hard_mask_mode,
                        edit_region_mode=ss_edit_region_mode,
                        soft_mask_enabled=config.controls.kv_blend.soft_mask.enabled,
                        soft_mask_dilation=config.controls.kv_blend.soft_mask.dilation,
                        soft_mask_sigma=config.controls.kv_blend.soft_mask.sigma,
                    ),
                    "cross": None,
                }
            }
            self.hook, token_meta, stage_configs = self._create_kv_blend_hook(
                inputs=hook_inputs,
                cond_dict=edit_cond_dict,
                enabled_stages=["ss"],
                stage_masks=stage_masks,
                extra=extra,
            )
            self.hook.stage_masks["ss"]["cross"] = self._build_cross_kv_token_mask(
                token_meta,
                soft_mask_enabled=config.controls.kv_blend.soft_mask.enabled,
                soft_mask_dilation=config.controls.kv_blend.soft_mask.dilation,
                soft_mask_sigma=config.controls.kv_blend.soft_mask.sigma,
            )
            self._patch_stage_hooks(pipeline, stage_configs, self.hook)
        elif config.controls.p2p.enabled:
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
        effective_ss_inversion_steps = (
            resolve_inversion_steps(
                total_steps=int(effective_sampler_params.get("steps", 25)),
                inversion_steps=config.inversion.inversion_steps,
            )
            if config.inversion.enabled
            else None
        )

        source_trace = None
        terminal_noise = None
        if config.inversion.enabled:
            print(f"Step 1: Inverting source SS (mode={config.inversion.solver})...")
            source_trace, terminal_noise = self._prepare_ss_source(
                pipeline,
                source_coords,
                source_cond_dict,
                config.inversion.solver,
                stage_config,
                resolution,
                return_terminal_noise=True,
                hook=self.hook,
            )
            _release_cuda_memory()
        else:
            print("Step 1: Skipping SS source inversion; using random noise init.")

        latent_mask = None
        if config.controls.latent_blend.enabled:
            print("Step 2: Building 3D latent mask...")
            model = pipeline.models["sparse_structure_flow_model"]
            latent_mask = self._build_ss_latent_mask(
                mask_coords,
                source_coords=source_coords,
                resolution=resolution,
                channels=model.in_channels,
                latent_resolution=model.resolution,
                hard_mask_mode=config.controls.latent_blend.hard_mask_mode,
                edit_region_mode=ss_edit_region_mode,
                soft_mask_enabled=config.controls.latent_blend.soft_mask.enabled,
                soft_mask_dilation=config.controls.latent_blend.soft_mask.dilation,
                soft_mask_sigma=config.controls.latent_blend.soft_mask.sigma,
            )
            print("Step 3: Running SS denoising with latent blending from inverted terminal noise...")
        elif config.inversion.enabled:
            print("Step 2: SS blend disabled; using inverted terminal noise as denoising init.")
            print("Step 3: Running SS denoising from inverted terminal noise...")
        else:
            print("Step 2: SS blend disabled.")
            print("Step 3: Running SS denoising from random noise...")
        if config.controls.p2p.enabled:
            self._patch_stage_hooks(pipeline, stage_configs, self.hook)
        sample = self._denoise_ss_with_step_blend(
            pipeline,
            edit_cond_dict,
            stage_config,
            initial_sample=terminal_noise,
            source_trace=source_trace if config.controls.latent_blend.enabled or config.controls.kv_blend.enabled else None,
            source_cond_dict=source_cond_dict,
            latent_mask=latent_mask,
            blend_strength=config.controls.latent_blend.strength,
            solver_mode=config.inversion.solver,
            verbose=True,
            hook=self.hook,
        )

        decoder = pipeline.models["sparse_structure_decoder"]
        if torch.cuda.is_available():
            print(f"[Memory] Before decode - Allocated: {torch.cuda.memory_allocated()/1024**3:.2f} GB")
        voxel = decoder(sample)
        coords_edited_raw = torch.argwhere(voxel > 0)[:, [0, 2, 3, 4]].int()
        del sample, voxel
        _release_cuda_memory()
        if torch.cuda.is_available():
            print(f"[Memory] After decode - Allocated: {torch.cuda.memory_allocated()/1024**3:.2f} GB")

        if coords_edited_raw.shape[0] == 0:
            raise RuntimeError("SS stage produced empty structure")

        coords_edited, _, ss_meta = _apply_ss_postprocess(
            mode=config.postprocess.mode,
            source_coords=source_coords,
            coords_stage1_raw=coords_edited_raw,
            mask_coords=mask_coords,
            boundary_band_width_voxels=config.postprocess.boundary_band.band_width_voxels,
            boundary_band_target_neighbor_threshold=(
                config.postprocess.boundary_band.band_target_neighbor_threshold
            ),
        )
        print(f"SS Stage complete: {ss_meta['stage1_masked_voxel_count']} voxels")
        voxel_mesh = None
        if config.output.save_voxel_mesh:
            voxel_mesh = coords_to_cubic_mesh(coords_edited, resolution)
            print(f"Generated voxel mesh: {len(voxel_mesh.faces)} faces")

        return SSArtifact(
            plugin_name=self.name,
            coords=coords_edited,
            voxel_mesh=voxel_mesh,
            metadata={
                "ss_meta": ss_meta,
                "ss_controls": {
                    "p2p_enabled": config.controls.p2p.enabled,
                    "kv_blend_enabled": config.controls.kv_blend.enabled,
                    "latent_blend_enabled": config.controls.latent_blend.enabled,
                    "uniedit_enabled": False,
                },
                "inversion_enabled": config.inversion.enabled,
                "inversion_mode": config.inversion.solver,
                "ss_total_steps": int(effective_sampler_params.get("steps", 25)),
                "ss_inversion_steps": effective_ss_inversion_steps,
                "ss_predictor_corrector_steps": int(config.inversion.predictor_corrector_steps),
                "blend_enabled": config.controls.latent_blend.enabled,
                "blend_strength": config.controls.latent_blend.strength,
                "ss_hard_mask_mode": config.controls.latent_blend.hard_mask_mode,
                "ss_edit_region_mode": ss_edit_region_mode,
                "ss_denoise_init": config.inversion.denoise_init,
                "ss_denoise_solver_mode": config.inversion.solver,
                "ss_kv_blend_enabled": config.controls.kv_blend.enabled,
                "ss_kv_self_attention": config.controls.kv_blend.self_attention,
                "ss_kv_cross_attention": config.controls.kv_blend.cross_attention,
                "ss_kv_soft_mask_enabled": config.controls.kv_blend.soft_mask.enabled,
                "ss_kv_soft_mask_dilation": int(config.controls.kv_blend.soft_mask.dilation),
                "ss_kv_soft_mask_sigma": float(config.controls.kv_blend.soft_mask.sigma),
                "ss_soft_mask_enabled": config.controls.latent_blend.soft_mask.enabled,
                "ss_soft_mask_dilation": int(config.controls.latent_blend.soft_mask.dilation),
                "ss_soft_mask_sigma": float(config.controls.latent_blend.soft_mask.sigma),
                "ss_postprocess_mode": config.postprocess.mode,
                "ss_denoise_cfg_strength": float(effective_sampler_params.get("cfg_strength", 0.0)),
                "ss_denoise_cfg_interval": tuple(effective_sampler_params.get("cfg_interval", (0.0, 1.0))),
                "ss_inversion_cfg_strength": inversion_cfg["cfg_strength"],
                "ss_inversion_cfg_interval": inversion_cfg["cfg_interval"],
                "resolution": resolution,
                "voxel_count": int(len(coords_edited)),
                **_ss_voxel_mesh_export_meta(config.output.save_voxel_mesh),
                **({"token_meta": token_meta} if token_meta is not None else {}),
            },
        )

    def save(self, artifact: SSArtifact, out_dir: Path, config: SSStageConfig) -> dict[str, str]:
        return _save_ss_artifacts(
            artifact,
            out_dir,
            output_format=config.output.output_format,
            save_mesh=config.output.save_voxel_mesh,
        )


class UniEditSLATAdapter(_ControlledDenoisePluginBase, SLATStagePlugin):
    name = "uniedit"

    def run(
        self,
        context: ExperimentContext,
        preprocess,
        config: SLATStageConfig,
        ss_artifact: SSArtifact | None,
    ) -> SLATArtifact:
        if not config.controls.uniedit.enabled:
            raise RuntimeError("UniEdit SLAT adapter requires slat.controls.uniedit.enabled=true")
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
        slat_inversion_steps = resolve_inversion_steps(
            total_steps=int(slat_params.get("steps", 25)),
            inversion_steps=config.inversion.inversion_steps,
        )

        coords_edited_raw = load_coords_from_file(edited_coords_path, pipeline.device)
        coords_edited = coords3d_to_batched(coords_edited_raw, batch_idx=0)
        print(f"Loaded edited coords: {len(coords_edited_raw)} voxels")
        _release_cuda_memory()

        ensure_pipeline_encoders(
            pipeline,
            model_root=context.config.runtime.model,
            require_slat_encoder=True,
        )
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
                asset_dir=_mask_cache_dir(context),
                source_normalization=_mask_source_normalization(context),
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

        token_meta = None
        if config.controls.p2p.enabled:
            extra = {
                "slat_t_start": config.controls.p2p.t_start,
                "slat_t_end": config.controls.p2p.t_end,
                "slat_strength": config.controls.p2p.strength,
                "patch_coverage_threshold": config.controls.p2p.patch_coverage_threshold,
            }
            hook_inputs = _P2PHookInputs(
                source_image=source_image,
                edit_image=edit_image,
                mask_image=preprocess.prepared_inputs.mask,
            )
            self.hook, token_meta, stage_configs = self._create_p2p_hook(
                inputs=hook_inputs,
                extra=extra,
                source_cond_dict=source_cond_dict,
                edit_cond_dict=edit_cond_dict,
                enabled_stages=["slat"],
            )
            self._patch_stage_hooks(pipeline, stage_configs, self.hook)

        _release_cuda_memory()

        print("Stage 0: Inverting source SLAT...")
        slat_terminal_noise = _invert_slat_with_rf(
            pipeline=pipeline,
            cond_src={"cond": source_cond, "neg_cond": neg_cond},
            slat_src=source_slat,
            params=slat_params,
            cfg_interval=config.controls.uniedit.cfg_interval,
            start_step=slat_inversion_steps,
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

        print(
            "SLAT Stage: Editing SLAT features "
            f"(variant={config.controls.uniedit.stage2_variant}, omega={config.controls.uniedit.omega})..."
        )
        if config.controls.uniedit.stage2_variant == "preserve_uniedit":
            slat_tgt = self._denoise_slat_variant(
                pipeline=pipeline,
                source_cond={"cond": source_cond, "neg_cond": neg_cond},
                target_cond={"cond": edit_cond, "neg_cond": neg_cond},
                terminal_noise=projected_slat_noise,
                params=slat_params,
                cfg_interval=config.controls.uniedit.cfg_interval,
                omega=config.controls.uniedit.omega,
                start_step=slat_inversion_steps,
                selector=stage2_selector,
                mode="preserve_overlap",
            )
        elif config.controls.uniedit.stage2_variant == "free_target":
            slat_tgt = self._denoise_slat_variant(
                pipeline=pipeline,
                source_cond={"cond": source_cond, "neg_cond": neg_cond},
                target_cond={"cond": edit_cond, "neg_cond": neg_cond},
                terminal_noise=projected_slat_noise,
                params=slat_params,
                cfg_interval=config.controls.uniedit.cfg_interval,
                omega=config.controls.uniedit.omega,
                start_step=slat_inversion_steps,
                selector=None,
                mode="target_only",
            )
        else:
            raise NotImplementedError(
                f"Stage 2 variant '{config.controls.uniedit.stage2_variant}' not implemented. "
                "Supported: preserve_uniedit, free_target."
            )

        final_blend_stats = None
        if config.postprocess.mode == "restore_source_outside_mask":
            if mask_coords is None:
                raise RuntimeError("slat.postprocess.mode='restore_source_outside_mask' requires inputs.mask_glb.")
            nano3d_replace_coords = self._build_slat_nano3d_replace_coords(
                source_slat=source_slat,
                edit_coords=coords_edited,
                mask_coords=mask_coords,
                resolution=resolution,
            )
            print(
                "Applying final SLAT restore_source_outside_mask postprocess: "
                f"{nano3d_replace_coords.shape[0]} overlapping outside-mask coords"
            )
            slat_tgt, final_blend_stats = self._apply_final_slat_feature_blend(
                slat_tgt,
                source_slat,
                nano3d_replace_coords,
            )

        del slat_terminal_noise, projected_slat_noise, source_slat
        del source_cond, edit_cond, neg_cond, source_cond_dict, edit_cond_dict
        if source_coords is not None:
            del source_coords
        if mask_coords is not None:
            del mask_coords
        if stage2_selector is not None:
            del stage2_selector
        _release_cuda_memory()

        decode_modes = _resolve_decode_modes(
            config.decode.modes,
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
                "stage2_variant": config.controls.uniedit.stage2_variant,
                "slat_controls": {
                    "p2p_enabled": config.controls.p2p.enabled,
                    "latent_blend_enabled": config.controls.latent_blend.enabled,
                    "restore_source_outside_mask_enabled": config.postprocess.mode == "restore_source_outside_mask",
                    "uniedit_enabled": True,
                },
                "slat_omega": config.controls.uniedit.omega,
                "cfg_interval": config.controls.uniedit.cfg_interval,
                "slat_total_steps": int(slat_params.get("steps", 25)),
                "slat_inversion_steps": slat_inversion_steps,
                "edited_voxel_count": int(len(coords_edited_raw)),
                "slat_postprocess_mode": config.postprocess.mode,
                **({"slat_final_blend_stats": final_blend_stats} if final_blend_stats is not None else {}),
                **({"token_meta": token_meta} if token_meta is not None else {}),
            },
        )

    def save(self, artifact: SLATArtifact, out_dir: Path, config: SLATStageConfig) -> dict[str, str]:
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
        start_step: int | None = None,
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
            start_step=start_step,
            selector=selector,
            mode=mode,
            verbose=True,
        )

        mean, std = _get_slat_norm_tensors(
            pipeline,
            slat_normalized.device,
            slat_normalized.feats.dtype,
        )
        return slat_normalized * std + mean


class DirectTargetSLATAdapter(_ControlledDenoisePluginBase, SLATStagePlugin):
    name = "direct_target"

    def run(
        self,
        context: ExperimentContext,
        preprocess,
        config: SLATStageConfig,
        ss_artifact: SSArtifact | None,
    ) -> SLATArtifact:
        self._output_options = _runtime_output_params(context.config.runtime)
        edited_coords_path = context.config.inputs.edited_coords
        if ss_artifact is not None:
            edited_coords_path = ss_artifact.primary_coords_path
        if edited_coords_path is None:
            raise RuntimeError("SLAT stage requires explicit edited_coords input or a preceding SS stage output")

        pipeline = context.pipeline
        resolution = int(pipeline.sparse_structure_sampler_params.get("grid_size", 64))
        slat_params = {
            **getattr(pipeline, "slat_sampler_params", {}),
            **_sampler_params(config.sampler),
        }

        _, edit_image, _ = _prepared_images(preprocess)
        edit_cond_dict = pipeline.get_cond([edit_image])

        edited_coords_raw = load_coords_from_file(edited_coords_path, pipeline.device)
        edited_coords = coords3d_to_batched(edited_coords_raw, batch_idx=0)
        print(f"Loaded edited coords: {edited_coords.shape[0]} voxels from {edited_coords_path}")

        restore_source_outside_mask = config.postprocess.mode == "restore_source_outside_mask"
        source_slat = None
        mask_coords = None
        final_blend_stats = None
        if restore_source_outside_mask:
            source_features_path = context.config.inputs.source_features
            if source_features_path is None:
                raise RuntimeError("slat.postprocess.mode='restore_source_outside_mask' requires inputs.source_features.")
            ensure_pipeline_encoders(
                pipeline,
                model_root=context.config.runtime.model,
                require_slat_encoder=True,
            )
            from trellis.modules.sparse.basic import SparseTensor

            source_slat = feats_to_slat(
                pipeline=pipeline,
                feats_path=source_features_path,
                SparseTensor=SparseTensor,
            )
            print(f"Loaded source SLAT: {source_slat.coords.shape[0]} features")
            mask_coords = _load_mask_coords(context, pipeline, resolution=resolution)
            if mask_coords is None:
                raise RuntimeError("slat.postprocess.mode='restore_source_outside_mask' requires inputs.mask_glb.")
            print(f"Loaded SLAT postprocess mask: {mask_coords.shape[0]} voxels")

        grad_enabled = torch.is_grad_enabled()
        torch.set_grad_enabled(False)
        try:
            print("Generating edit SLAT from random noise...")
            edit_slat = pipeline.sample_slat(
                edit_cond_dict,
                edited_coords,
                sampler_params=slat_params,
            )
            print(f"Edit SLAT: {edit_slat.coords.shape[0]} voxels")

            if restore_source_outside_mask:
                nano3d_replace_coords = self._build_slat_nano3d_replace_coords(
                    source_slat=source_slat,
                    edit_coords=edited_coords,
                    mask_coords=mask_coords,
                    resolution=resolution,
                )
                print(
                    "Applying SLAT restore_source_outside_mask postprocess: "
                    f"{nano3d_replace_coords.shape[0]} overlapping outside-mask coords"
                )
                edit_slat, final_blend_stats = self._apply_final_slat_feature_blend(
                    edit_slat,
                    source_slat,
                    nano3d_replace_coords,
                )

            outputs, source_outputs = self._decode_outputs(
                pipeline,
                source_slat=source_slat,
                edit_slat=edit_slat,
                decode_modes=config.decode.modes,
                skip_source_decode=(
                    config.decode.skip_source_decode
                    and not context.config.runtime.save_source_outputs
                ) or source_slat is None,
                verbose=False,
            )
        finally:
            torch.set_grad_enabled(grad_enabled)

        metadata: dict[str, Any] = {
            "slat_method": self.name,
            "slat_postprocess_mode": config.postprocess.mode,
            "slat_total_steps": int(slat_params.get("steps", 25)),
            "slat_denoise_init": "random_noise",
            "edited_voxel_count": int(edited_coords_raw.shape[0]),
            "slat_controls": {
                "p2p_enabled": False,
                "kv_blend_enabled": False,
                "latent_blend_enabled": False,
                "nano3d_replace_enabled": restore_source_outside_mask,
                "restore_source_outside_mask_enabled": restore_source_outside_mask,
                "uniedit_enabled": False,
            },
        }
        if final_blend_stats is not None:
            metadata["slat_final_blend_stats"] = final_blend_stats
            metadata["slat_final_blend_mode"] = "restore_source_outside_mask"

        return SLATArtifact(
            plugin_name=self.name,
            outputs=outputs,
            source_outputs=source_outputs,
            metadata=metadata,
        )

    def save(self, artifact: SLATArtifact, out_dir: Path, config: SLATStageConfig) -> dict[str, str]:
        output_options = getattr(self, "_output_options", {"skip_render": True, "skip_glb": False, "skip_ply": False})
        save_outputs(
            outputs=artifact.outputs,
            out_dir=out_dir,
            skip_render=output_options["skip_render"],
            skip_glb=output_options["skip_glb"],
            skip_ply=output_options["skip_ply"],
        )
        return _save_slat_metadata(artifact, out_dir)


class ControlledDenoiseSLATAdapter(_ControlledDenoisePluginBase, SLATStagePlugin):
    name = "controlled_denoise"

    def run(
        self,
        context: ExperimentContext,
        preprocess,
        config: SLATStageConfig,
        ss_artifact: SSArtifact | None,
    ) -> SLATArtifact:
        if config.controls.uniedit.enabled:
            raise RuntimeError("controlled_denoise SLAT adapter does not accept slat.controls.uniedit.enabled=true")
        self._output_options = _runtime_output_params(context.config.runtime)
        edited_coords_path = context.config.inputs.edited_coords
        if ss_artifact is not None:
            edited_coords_path = ss_artifact.primary_coords_path
        if edited_coords_path is None:
            raise RuntimeError("SLAT stage requires explicit edited_coords input or a preceding SS stage output")

        source_features_path = context.config.inputs.source_features
        source_voxels_path = context.config.inputs.source_voxels

        pipeline = context.pipeline
        resolution = pipeline.sparse_structure_sampler_params.get("grid_size", 64)
        source_image, edit_image, mask_image = _prepared_images(preprocess)
        edit_cond_dict = pipeline.get_cond([edit_image])
        source_cond_dict = None
        if config.inversion.enabled or config.controls.p2p.enabled or config.controls.kv_blend.enabled:
            source_cond_dict = pipeline.get_cond([source_image])

        edited_coords_raw = load_coords_from_file(edited_coords_path, pipeline.device)
        edited_coords = coords3d_to_batched(edited_coords_raw, batch_idx=0)
        print(f"Loaded edited coords: {edited_coords.shape[0]} voxels from {edited_coords_path}")

        source_slat = None
        if source_features_path is not None:
            ensure_pipeline_encoders(
                pipeline,
                model_root=context.config.runtime.model,
                require_slat_encoder=True,
            )
            from trellis.modules.sparse.basic import SparseTensor

            print(f"Loading source SLAT features from: {source_features_path}")
            source_slat = feats_to_slat(
                pipeline=pipeline,
                feats_path=source_features_path,
                SparseTensor=SparseTensor,
            )

        source_coords = None
        if source_voxels_path is not None:
            source_coords_raw = load_coords_from_file(source_voxels_path, pipeline.device)
            source_coords = coords3d_to_batched(source_coords_raw, batch_idx=0)

        mask_coords = None
        if context.config.inputs.mask_glb is not None:
            mask_result = load_mask_glb_coords(
                mask_glb=str(context.config.inputs.mask_glb),
                device=pipeline.device,
                resolution=resolution,
                asset_dir=_mask_cache_dir(context),
                source_normalization=_mask_source_normalization(context),
            )
            mask_coords = mask_result.coords
            print(f"Mask coords: {mask_coords.shape[0]} voxels")
        elif config.controls.latent_blend.enabled:
            raise RuntimeError("controlled_denoise SLAT requires inputs.mask_glb when blending is enabled")

        extra = {
            "slat_t_start": config.controls.p2p.t_start,
            "slat_t_end": config.controls.p2p.t_end,
            "slat_strength": config.controls.p2p.strength,
            "patch_coverage_threshold": config.controls.p2p.patch_coverage_threshold,
            "slat_kv_t_start": config.controls.kv_blend.t_start,
            "slat_kv_t_end": config.controls.kv_blend.t_end,
            "slat_kv_self_attention": config.controls.kv_blend.self_attention,
            "slat_kv_cross_attention": config.controls.kv_blend.cross_attention,
            "slat_predictor_corrector_steps": int(config.inversion.predictor_corrector_steps),
            "slat_inversion_steps": config.inversion.inversion_steps,
            "slat_denoise_init": config.inversion.denoise_init,
            "slat_inversion_scope": config.inversion.scope,
            "skip_source_decode": config.decode.skip_source_decode and not context.config.runtime.save_source_outputs,
        }
        if config.inversion.denoise_cfg_strength is not None:
            extra["slat_denoise_cfg_strength"] = float(config.inversion.denoise_cfg_strength)
        if config.inversion.denoise_cfg_interval is not None:
            extra["slat_denoise_cfg_interval_start"] = float(config.inversion.denoise_cfg_interval[0])
            extra["slat_denoise_cfg_interval_end"] = float(config.inversion.denoise_cfg_interval[1])
        if config.inversion.inversion_cfg_strength is not None:
            extra["slat_inversion_cfg_strength"] = float(config.inversion.inversion_cfg_strength)
        if config.inversion.inversion_cfg_interval is not None:
            extra["slat_inversion_cfg_interval_start"] = float(config.inversion.inversion_cfg_interval[0])
            extra["slat_inversion_cfg_interval_end"] = float(config.inversion.inversion_cfg_interval[1])
        token_meta = {}
        stage_configs = {}
        if config.controls.kv_blend.enabled:
            if source_coords is None:
                raise RuntimeError("controlled_denoise SLAT with kv_blend requires inputs.source_voxels")
            if mask_coords is None:
                raise RuntimeError("controlled_denoise SLAT with kv_blend requires inputs.mask_glb")
            _, preserve_coords, _ = compose_stage1_coords(
                coords_source=source_coords,
                coords_stage1_raw=edited_coords_raw,
                mask_coords=mask_coords,
            )
            hook_inputs = _P2PHookInputs(
                source_image=source_image,
                edit_image=edit_image,
                mask_image=mask_image,
            )
            stage_masks = {
                "slat": {
                    "self": self._build_slat_self_kv_mask(
                        edited_coords,
                        preserve_coords,
                        resolution=resolution,
                        soft_mask_enabled=config.controls.kv_blend.soft_mask.enabled,
                        soft_mask_dilation=config.controls.kv_blend.soft_mask.dilation,
                        soft_mask_sigma=config.controls.kv_blend.soft_mask.sigma,
                    ),
                    "cross": None,
                }
            }
            self.hook, token_meta, stage_configs = self._create_kv_blend_hook(
                inputs=hook_inputs,
                cond_dict=edit_cond_dict,
                enabled_stages=["slat"],
                stage_masks=stage_masks,
                extra=extra,
            )
            self.hook.stage_masks["slat"]["cross"] = self._build_cross_kv_token_mask(
                token_meta,
                soft_mask_enabled=config.controls.kv_blend.soft_mask.enabled,
                soft_mask_dilation=config.controls.kv_blend.soft_mask.dilation,
                soft_mask_sigma=config.controls.kv_blend.soft_mask.sigma,
            )
            self._patch_stage_hooks(pipeline, stage_configs, self.hook)
        elif config.controls.p2p.enabled:
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
        restore_source_outside_mask_enabled = (
            config.controls.nano3d_replace.enabled
            or config.postprocess.mode == "restore_source_outside_mask"
        )
        effective_slat_sampler_params = {
            **getattr(pipeline, "slat_sampler_params", {}),
            **self._resolve_stage_sampler_params("slat", stage_config.slat_sampler_params, extra),
        }
        effective_slat_inversion_steps = (
            resolve_inversion_steps(
                total_steps=int(effective_slat_sampler_params.get("steps", 25)),
                inversion_steps=config.inversion.inversion_steps,
            )
            if config.inversion.enabled
            else None
        )

        grad_enabled = torch.is_grad_enabled()
        torch.set_grad_enabled(False)
        try:
            if config.controls.p2p.enabled:
                self._patch_stage_hooks(pipeline, stage_configs, self.hook)
            edit_slat, final_blend_stats = self._run_slat_stage(
                pipeline,
                source_slat=source_slat,
                source_coords=source_coords,
                source_cond_dict=source_cond_dict,
                edit_cond_dict=edit_cond_dict,
                edit_coords=edited_coords,
                mask_coords=mask_coords,
                resolution=resolution,
                blend_slat_enabled=config.controls.latent_blend.enabled,
                nano3d_replace_enabled=restore_source_outside_mask_enabled,
                kv_blend_enabled=config.controls.kv_blend.enabled,
                inversion_enabled=config.inversion.enabled,
                inversion_mode=config.inversion.solver,
                soft_mask_enabled=config.controls.latent_blend.soft_mask.enabled,
                soft_mask_dilation=config.controls.latent_blend.soft_mask.dilation,
                soft_mask_sigma=config.controls.latent_blend.soft_mask.sigma,
                config=stage_config,
                verbose=False,
                hook=self.hook,
            )
            outputs, source_outputs = self._decode_outputs(
                pipeline,
                source_slat=source_slat,
                edit_slat=edit_slat,
                decode_modes=config.decode.modes,
                skip_source_decode=extra["skip_source_decode"],
                verbose=False,
            )
        finally:
            torch.set_grad_enabled(grad_enabled)

        token_meta = dict(token_meta)
        token_meta.update(
            {
                "slat_controls": {
                    "p2p_enabled": config.controls.p2p.enabled,
                    "kv_blend_enabled": config.controls.kv_blend.enabled,
                    "latent_blend_enabled": config.controls.latent_blend.enabled,
                    "nano3d_replace_enabled": restore_source_outside_mask_enabled,
                    "restore_source_outside_mask_enabled": restore_source_outside_mask_enabled,
                    "uniedit_enabled": False,
                },
                "inversion_enabled": config.inversion.enabled,
                "slat_kv_blend_enabled": config.controls.kv_blend.enabled,
                "slat_kv_self_attention": config.controls.kv_blend.self_attention,
                "slat_kv_cross_attention": config.controls.kv_blend.cross_attention,
                "slat_kv_soft_mask_enabled": config.controls.kv_blend.soft_mask.enabled,
                "slat_kv_soft_mask_dilation": int(config.controls.kv_blend.soft_mask.dilation),
                "slat_kv_soft_mask_sigma": float(config.controls.kv_blend.soft_mask.sigma),
                "slat_soft_mask_enabled": config.controls.latent_blend.soft_mask.enabled,
                "slat_soft_mask_dilation": int(config.controls.latent_blend.soft_mask.dilation),
                "slat_soft_mask_sigma": float(config.controls.latent_blend.soft_mask.sigma),
                "slat_postprocess_mode": config.postprocess.mode,
                "slat_nano3d_replace_enabled": restore_source_outside_mask_enabled,
                "slat_total_steps": int(effective_slat_sampler_params.get("steps", 25)),
                "slat_inversion_steps": effective_slat_inversion_steps,
                "slat_predictor_corrector_steps": int(config.inversion.predictor_corrector_steps),
                "slat_denoise_init": config.inversion.denoise_init,
                "slat_inversion_scope": config.inversion.scope,
                "slat_denoise_cfg_strength": config.inversion.denoise_cfg_strength,
                "slat_denoise_cfg_interval": config.inversion.denoise_cfg_interval,
                "slat_inversion_cfg_strength": config.inversion.inversion_cfg_strength,
                "slat_inversion_cfg_interval": config.inversion.inversion_cfg_interval,
            }
        )
        if final_blend_stats is not None:
            token_meta["slat_final_blend_stats"] = final_blend_stats
            token_meta["slat_final_blend_mode"] = (
                "restore_source_outside_mask" if restore_source_outside_mask_enabled else "latent_blend"
            )

        return SLATArtifact(
            plugin_name=self.name,
            outputs=outputs,
            source_outputs=source_outputs,
            metadata=token_meta,
        )

    def save(self, artifact: SLATArtifact, out_dir: Path, config: SLATStageConfig) -> dict[str, str]:
        output_options = getattr(self, "_output_options", {"skip_render": True, "skip_glb": False, "skip_ply": False})
        save_outputs(
            outputs=artifact.outputs,
            out_dir=out_dir,
            skip_render=output_options["skip_render"],
            skip_glb=output_options["skip_glb"],
            skip_ply=output_options["skip_ply"],
        )

        return _save_slat_metadata(artifact, out_dir)


SS_PLUGIN_REGISTRY: dict[str, type[SSStagePlugin]] = {
    "anchorflow": AnchorFlowSSAdapter,
    "flowedit": FlowEditSSAdapter,
    "uniedit": UniEditSSAdapter,
    "controlled_denoise": ControlledDenoiseSSAdapter,
}

SLAT_PLUGIN_REGISTRY: dict[str, type[SLATStagePlugin]] = {
    "direct_target": DirectTargetSLATAdapter,
    "uniedit": UniEditSLATAdapter,
    "controlled_denoise": ControlledDenoiseSLATAdapter,
}
