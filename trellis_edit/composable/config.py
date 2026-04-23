from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Literal


RunMode = Literal["ss", "slat", "full"]
CropPolicy = Literal["disabled", "union_crop", "mask_crop"]
PreprocessStyle = Literal["shared", "voxhammer"]
MaskPolicy = Literal["provided", "auto_diff", "blank"]
AdaptiveForegroundScaleStrategy = Literal["two_probe", "global_linear"]
SolverMode = Literal["simple", "rf_solver", "voxhammer_rf_solver"]
InheritableSolverMode = Literal["inherit", "simple", "rf_solver", "voxhammer_rf_solver"]
SSDenoiseInit = Literal["random_noise", "inverted_terminal_noise"]
SLATDenoiseInit = Literal["random_noise", "terminal_noise"]
SLATInversionScope = Literal["full_source", "preserve_only"]
SSBlendRegionMode = Literal["mask_only", "preserve_complement"]
KVSourceBundleMode = Literal["solver_aligned", "external_voxhammer"]
SSMethod = Literal["controlled_denoise", "uniedit", "flowedit", "anchorflow"]
SLATMethod = Literal["controlled_denoise", "uniedit", "direct_target"]
StagePostprocessMode = Literal[
    "none",
    "restore_source_outside_mask",
    "restore_all_outside_mask",
    "boundary_band_restore",
]

_SOLVER_MODE_VALUES = {"simple", "rf_solver", "voxhammer_rf_solver"}
_INHERITABLE_SOLVER_MODE_VALUES = {"inherit", *tuple(_SOLVER_MODE_VALUES)}


def validate_solver_mode_value(solver_mode: str, *, field_name: str) -> None:
    if solver_mode not in _SOLVER_MODE_VALUES:
        valid = ", ".join(sorted(_SOLVER_MODE_VALUES))
        raise RuntimeError(f"{field_name} must be one of: {valid}. Got {solver_mode!r}.")


def validate_inheritable_solver_mode_value(solver_mode: str, *, field_name: str) -> None:
    if solver_mode not in _INHERITABLE_SOLVER_MODE_VALUES:
        valid = ", ".join(sorted(_INHERITABLE_SOLVER_MODE_VALUES))
        raise RuntimeError(f"{field_name} must be one of: {valid}. Got {solver_mode!r}.")


def resolve_inheritable_solver_mode(
    solver_mode: SolverMode | InheritableSolverMode,
    *,
    inherit_from: SolverMode,
) -> SolverMode:
    if solver_mode == "inherit":
        return inherit_from
    return solver_mode


def solver_predictor_eval_count(solver_mode: SolverMode) -> int:
    return 2 if solver_mode in {"rf_solver", "voxhammer_rf_solver"} else 1


@dataclass(frozen=True)
class InputConfig:
    source_image: Path | None = None
    edit_image: Path | None = None
    mask_image: Path | None = None
    mask_glb: Path | None = None
    source_voxels: Path | None = None
    source_features: Path | None = None
    edited_coords: Path | None = None


@dataclass(frozen=True)
class RuntimeConfig:
    model: str
    case_name: str
    seed: int
    device: str = "cuda:0"
    output_root: Path = Path("outputs")
    output_group: str = ""
    num_samples: int = 1
    attn_backend: str = ""
    sparse_attn_backend: str = ""
    spconv_algo: str = "native"
    skip_render: bool = True
    skip_glb: bool = False
    skip_ply: bool = False
    save_source_outputs: bool = False


@dataclass(frozen=True)
class PreprocessConfig:
    crop_policy: CropPolicy
    mask_policy: MaskPolicy
    style: PreprocessStyle = "shared"
    mask_threshold: int = 128
    auto_mask_max_filter: int = 5
    auto_mask_diff_threshold: int = 24
    adaptive_foreground_scale: "AdaptiveForegroundScaleConfig" = field(
        default_factory=lambda: AdaptiveForegroundScaleConfig()
    )


@dataclass(frozen=True)
class AdaptiveForegroundScaleConfig:
    enabled: bool = False
    strategy: AdaptiveForegroundScaleStrategy = "two_probe"
    probe_scales: tuple[float, float] = (1.10, 1.35)
    min_scale: float = 1.0
    max_scale: float = 1.5
    min_bbox_volume_delta: float = 0.02
    fallback_scale: float = 1.2
    linear_bbox_slope: float = 0.5449373200077813
    linear_bbox_intercept: float = 0.29052836699859885


@dataclass(frozen=True)
class SamplerOverrideConfig:
    steps: int | None = None
    cfg_strength: float | None = None
    rescale_t: float | None = None


@dataclass(frozen=True)
class SoftMaskConfig:
    enabled: bool = False
    dilation: int = 2
    sigma: float = 1.0


@dataclass(frozen=True)
class KVBlendControlConfig:
    enabled: bool = False
    t_start: float = 1.0
    t_end: float = 0.0
    self_attention: bool = True
    cross_attention: bool = True
    source_bundle_mode: KVSourceBundleMode = "solver_aligned"
    soft_mask: SoftMaskConfig = field(default_factory=SoftMaskConfig)


@dataclass(frozen=True)
class SSKVBlendControlConfig(KVBlendControlConfig):
    t_start: float = 1.0
    t_end: float = 0.0
    edit_region_mode: SSBlendRegionMode = "mask_only"


@dataclass(frozen=True)
class SLATKVBlendControlConfig(KVBlendControlConfig):
    t_start: float = 1.0
    t_end: float = 0.0


@dataclass(frozen=True)
class P2PControlConfig:
    enabled: bool = False
    t_start: float = 1.0
    t_end: float = 0.3
    strength: float = 1.0
    patch_coverage_threshold: float = 0.0


@dataclass(frozen=True)
class SSP2PControlConfig(P2PControlConfig):
    t_start: float = 1.0
    t_end: float = 0.3


@dataclass(frozen=True)
class SLATP2PControlConfig(P2PControlConfig):
    t_start: float = 0.8
    t_end: float = 0.0


@dataclass(frozen=True)
class SSLatentBlendControlConfig:
    enabled: bool = False
    strength: float = 1.0
    hard_mask_mode: Literal["edit_any", "edit_all"] = "edit_all"
    edit_region_mode: SSBlendRegionMode = "mask_only"
    soft_mask: SoftMaskConfig = field(default_factory=SoftMaskConfig)


@dataclass(frozen=True)
class SLATLatentBlendControlConfig:
    enabled: bool = False
    soft_mask: SoftMaskConfig = field(default_factory=SoftMaskConfig)


@dataclass(frozen=True)
class SLATNano3DReplaceControlConfig:
    enabled: bool = False


@dataclass(frozen=True)
class SSUniEditControlConfig:
    enabled: bool = False
    omega: float = 1.0
    cfg_interval: tuple[float, float] = (0.5, 1.0)


@dataclass(frozen=True)
class SLATUniEditControlConfig:
    enabled: bool = False
    omega: float = 1.0
    cfg_interval: tuple[float, float] = (0.5, 1.0)
    stage2_variant: Literal["preserve_uniedit", "free_target"] = "preserve_uniedit"


@dataclass(frozen=True)
class SSControlsConfig:
    p2p: SSP2PControlConfig = field(default_factory=SSP2PControlConfig)
    kv_blend: SSKVBlendControlConfig = field(default_factory=SSKVBlendControlConfig)
    latent_blend: SSLatentBlendControlConfig = field(default_factory=SSLatentBlendControlConfig)
    uniedit: SSUniEditControlConfig = field(default_factory=SSUniEditControlConfig)


@dataclass(frozen=True)
class SLATControlsConfig:
    p2p: SLATP2PControlConfig = field(default_factory=SLATP2PControlConfig)
    kv_blend: SLATKVBlendControlConfig = field(default_factory=SLATKVBlendControlConfig)
    latent_blend: SLATLatentBlendControlConfig = field(default_factory=SLATLatentBlendControlConfig)
    nano3d_replace: SLATNano3DReplaceControlConfig = field(default_factory=SLATNano3DReplaceControlConfig)
    uniedit: SLATUniEditControlConfig = field(default_factory=SLATUniEditControlConfig)


@dataclass(frozen=True)
class SSDenoiseConfig:
    solver: InheritableSolverMode = "inherit"


@dataclass(frozen=True)
class SLATDenoiseConfig:
    solver: InheritableSolverMode = "inherit"


@dataclass(frozen=True)
class SSInversionConfig:
    enabled: bool = True
    denoise_init: SSDenoiseInit = "inverted_terminal_noise"
    solver: SolverMode = "simple"
    refinement_steps: int = 0
    refinement_solver: InheritableSolverMode = "inherit"
    inversion_steps: int | None = None
    denoise_cfg_strength: float = 5.0
    denoise_cfg_interval: tuple[float, float] = (0.5, 1.0)
    inversion_cfg_strength: float = 6.0
    inversion_cfg_interval: tuple[float, float] = (0.5, 1.0)
    save_artifacts: bool = False


@dataclass(frozen=True)
class SLATInversionConfig:
    enabled: bool = True
    denoise_init: SLATDenoiseInit = "terminal_noise"
    scope: SLATInversionScope = "full_source"
    solver: SolverMode = "simple"
    refinement_steps: int = 0
    refinement_solver: InheritableSolverMode = "inherit"
    inversion_steps: int | None = None
    denoise_cfg_strength: float | None = None
    denoise_cfg_interval: tuple[float, float] | None = None
    inversion_cfg_strength: float | None = None
    inversion_cfg_interval: tuple[float, float] | None = None
    save_artifacts: bool = False


@dataclass(frozen=True)
class SSOutputConfig:
    output_format: Literal["ply", "pt", "both"] = "both"
    save_voxel_mesh: bool = True


@dataclass(frozen=True)
class SLATDecodeConfig:
    skip_source_decode: bool = True
    modes: tuple[str, ...] = ("gaussian", "mesh")


@dataclass(frozen=True)
class SSFlowEditConfig:
    n_avg: int = 5
    start_step: int = 12
    src_cfg_strength: float = 1.5
    tar_cfg_strength: float = 5.5
    cfg_interval: tuple[float, float] = (0.0, 1.0)


@dataclass(frozen=True)
class SSAnchorFlowConfig:
    n_avg: int = 1
    n_max: int = 31
    src_cfg_strength: float = 3.5
    tar_cfg_strength: float = 5.0
    cfg_interval: tuple[float, float] = (0.0, 1.0)
    center_weight: float = 0.10
    band_weight: float = 0.75
    ortho_weight: float = 0.20
    residual_weight: float = 0.05
    band_min_ratio: float = 0.60
    band_max_ratio: float = 1.40
    margin_scale: float = 0.80
    direction_gate_tau: float = 0.05
    eps: float = 1.0e-6
    anchor_noise: bool = False


@dataclass(frozen=True)
class SSBoundaryBandRestoreConfig:
    band_width_voxels: int = 1
    band_target_neighbor_threshold: int = 1


@dataclass(frozen=True)
class SSPostprocessConfig:
    mode: StagePostprocessMode = "restore_all_outside_mask"
    boundary_band: SSBoundaryBandRestoreConfig = field(default_factory=SSBoundaryBandRestoreConfig)


@dataclass(frozen=True)
class SLATPostprocessConfig:
    mode: StagePostprocessMode = "none"


@dataclass(frozen=True)
class SSStageConfig:
    method: SSMethod = "controlled_denoise"
    sampler: SamplerOverrideConfig = field(default_factory=SamplerOverrideConfig)
    inversion: SSInversionConfig = field(default_factory=SSInversionConfig)
    denoise: SSDenoiseConfig = field(default_factory=SSDenoiseConfig)
    flowedit: SSFlowEditConfig = field(default_factory=SSFlowEditConfig)
    anchorflow: SSAnchorFlowConfig = field(default_factory=SSAnchorFlowConfig)
    controls: SSControlsConfig = field(default_factory=SSControlsConfig)
    postprocess: SSPostprocessConfig = field(default_factory=SSPostprocessConfig)
    output: SSOutputConfig = field(default_factory=SSOutputConfig)

    def requires_source_image(self) -> bool:
        if self.method in {"flowedit", "anchorflow"}:
            return True
        return (
            self.inversion.enabled
            or self.controls.p2p.enabled
            or self.controls.kv_blend.enabled
            or self.controls.uniedit.enabled
        )

    def requires_source_voxels(self) -> bool:
        if self.method in {"flowedit", "anchorflow"}:
            return True
        return (
            self.inversion.enabled
            or self.controls.latent_blend.enabled
            or self.controls.uniedit.enabled
            or self.postprocess.mode != "none"
        )

    def requires_mask_glb(self) -> bool:
        return (
            self.controls.latent_blend.enabled
            or self.controls.kv_blend.enabled
            or self.postprocess.mode != "none"
        )

    def validate(self, label: str = "ss") -> None:
        from .validation_rules import validate_ss_stage_config

        validate_ss_stage_config(self, label)
        validate_solver_mode_value(self.inversion.solver, field_name=f"{label}.inversion.solver")
        validate_inheritable_solver_mode_value(
            self.inversion.refinement_solver,
            field_name=f"{label}.inversion.refinement_solver",
        )
        validate_inheritable_solver_mode_value(self.denoise.solver, field_name=f"{label}.denoise.solver")
        enabled_modes = []
        if self.controls.kv_blend.enabled:
            enabled_modes.append(
                ("controls.kv_blend.edit_region_mode", self.controls.kv_blend.edit_region_mode)
            )
        if self.controls.latent_blend.enabled:
            enabled_modes.append(
                ("controls.latent_blend.edit_region_mode", self.controls.latent_blend.edit_region_mode)
            )
        if len({mode for _, mode in enabled_modes}) > 1:
            details = ", ".join(f"{field}={mode}" for field, mode in enabled_modes)
            raise RuntimeError(
                f"{label} enabled blend controls must share the same edit_region_mode. Got: {details}"
            )
        resolved_denoise_solver = resolve_inheritable_solver_mode(
            self.denoise.solver,
            inherit_from=self.inversion.solver,
        )
        if (
            self.inversion.enabled
            and self.controls.kv_blend.enabled
            and self.controls.kv_blend.source_bundle_mode == "external_voxhammer"
            and solver_predictor_eval_count(self.inversion.solver)
            != solver_predictor_eval_count(resolved_denoise_solver)
        ):
            raise RuntimeError(
                f"{label}.controls.kv_blend.source_bundle_mode='external_voxhammer' requires "
                f"{label}.inversion.solver and {label}.denoise.solver to share the same predictor "
                f"slot layout. Resolved inversion={self.inversion.solver!r}, "
                f"denoise={resolved_denoise_solver!r}."
            )


@dataclass(frozen=True)
class SLATStageConfig:
    method: SLATMethod = "controlled_denoise"
    sampler: SamplerOverrideConfig = field(default_factory=SamplerOverrideConfig)
    inversion: SLATInversionConfig = field(default_factory=SLATInversionConfig)
    denoise: SLATDenoiseConfig = field(default_factory=SLATDenoiseConfig)
    controls: SLATControlsConfig = field(default_factory=SLATControlsConfig)
    postprocess: SLATPostprocessConfig = field(default_factory=SLATPostprocessConfig)
    decode: SLATDecodeConfig = field(default_factory=SLATDecodeConfig)

    def requires_source_image(self) -> bool:
        if self.method == "direct_target":
            return False
        return (
            self.inversion.enabled
            or self.controls.p2p.enabled
            or self.controls.kv_blend.enabled
            or self.controls.uniedit.enabled
        )

    def requires_source_features(self) -> bool:
        if self.method == "direct_target":
            return self.postprocess.mode == "restore_source_outside_mask"
        return (
            self.inversion.enabled
            or self.controls.latent_blend.enabled
            or self.controls.nano3d_replace.enabled
            or self.controls.uniedit.enabled
            or self.postprocess.mode == "restore_source_outside_mask"
        )

    def requires_source_voxels(self) -> bool:
        return self.controls.kv_blend.enabled

    def requires_mask_glb(self) -> bool:
        return (
            self.controls.latent_blend.enabled
            or self.controls.nano3d_replace.enabled
            or self.controls.kv_blend.enabled
            or self.requires_source_voxels()
            or (self.inversion.enabled and self.inversion.scope == "preserve_only")
            or self.postprocess.mode == "restore_source_outside_mask"
        )

    def validate(self, label: str = "slat") -> None:
        from .validation_rules import validate_slat_stage_config

        validate_slat_stage_config(self, label)
        validate_solver_mode_value(self.inversion.solver, field_name=f"{label}.inversion.solver")
        validate_inheritable_solver_mode_value(
            self.inversion.refinement_solver,
            field_name=f"{label}.inversion.refinement_solver",
        )
        validate_inheritable_solver_mode_value(self.denoise.solver, field_name=f"{label}.denoise.solver")
        resolved_denoise_solver = resolve_inheritable_solver_mode(
            self.denoise.solver,
            inherit_from=self.inversion.solver,
        )
        if (
            self.inversion.enabled
            and self.controls.kv_blend.enabled
            and self.controls.kv_blend.source_bundle_mode == "external_voxhammer"
            and solver_predictor_eval_count(self.inversion.solver)
            != solver_predictor_eval_count(resolved_denoise_solver)
        ):
            raise RuntimeError(
                f"{label}.controls.kv_blend.source_bundle_mode='external_voxhammer' requires "
                f"{label}.inversion.solver and {label}.denoise.solver to share the same predictor "
                f"slot layout. Resolved inversion={self.inversion.solver!r}, "
                f"denoise={resolved_denoise_solver!r}."
            )


def default_ss_stage() -> SSStageConfig:
    return SSStageConfig()


def default_slat_stage() -> SLATStageConfig:
    return SLATStageConfig()


@dataclass(frozen=True)
class ExperimentConfig:
    entry_name: str
    preset_name: str
    run_mode: RunMode
    runtime: RuntimeConfig
    inputs: InputConfig
    preprocess: PreprocessConfig
    ss: SSStageConfig | None = None
    slat: SLATStageConfig | None = None

    def required_input_fields(self) -> tuple[str, ...]:
        required: set[str] = {"edit_image"}

        if self.preprocess.style == "voxhammer":
            required.add("source_image")

        if self.preprocess.mask_policy == "provided":
            required.add("mask_image")
        if self.preprocess.mask_policy == "auto_diff":
            required.add("source_image")

        if self.run_mode in {"ss", "full"}:
            if self.ss is None:
                raise RuntimeError("run_mode requires an ss stage, but config.ss is None")
            self.ss.validate("ss")
            if self.ss.requires_source_image():
                required.add("source_image")
            if self.ss.requires_source_voxels():
                required.add("source_voxels")
            if self.ss.requires_mask_glb():
                required.add("mask_glb")

        if self.run_mode in {"slat", "full"}:
            if self.slat is None:
                raise RuntimeError("run_mode requires a slat stage, but config.slat is None")
            self.slat.validate("slat")
            if self.slat.requires_source_image():
                required.add("source_image")
            if self.slat.requires_source_features():
                required.add("source_features")
            if self.slat.requires_source_voxels():
                required.add("source_voxels")
            if self.slat.requires_mask_glb():
                required.add("mask_glb")
            if self.run_mode == "slat":
                required.add("edited_coords")

        return tuple(sorted(required))

    def validate_inputs(self) -> None:
        missing = [field for field in self.required_input_fields() if getattr(self.inputs, field) is None]
        if missing:
            missing_labels = ", ".join(missing)
            raise RuntimeError(f"{self.entry_name} requires explicit input paths for: {missing_labels}")

        for field_name in (
            "source_image",
            "edit_image",
            "mask_image",
            "mask_glb",
            "source_voxels",
            "source_features",
            "edited_coords",
        ):
            value = getattr(self.inputs, field_name)
            if value is not None and not value.is_file():
                raise RuntimeError(f"{self.entry_name} input path does not exist: {field_name}={value}")
