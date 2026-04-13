from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Literal


RunMode = Literal["ss", "slat", "full"]
CropPolicy = Literal["disabled", "union_crop", "mask_crop"]
ForegroundPolicy = Literal["raw_rgb", "alpha_only", "alpha_or_rembg"]
MaskPolicy = Literal["provided", "auto_diff", "blank"]


@dataclass(frozen=True)
class InputConfig:
    source_image: Path
    edit_image: Path
    mask_image: Path | None = None
    # 这里可以放一个source_model_path，用于指定source model的路径，然后source_voxels和source_feature都可以通过trelli的slat encoder获取得到
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
    foreground_policy: ForegroundPolicy
    mask_policy: MaskPolicy
    mask_threshold: int = 128
    auto_mask_max_filter: int = 5
    auto_mask_diff_threshold: int = 24


@dataclass(frozen=True)
class SamplerOverrideConfig:
    steps: int | None = None
    cfg_strength: float | None = None
    rescale_t: float | None = None


@dataclass(frozen=True)
class P2PHookConfig:
    t_start: float
    t_end: float
    strength: float


@dataclass(frozen=True)
class UniEditSSConfig:
    sampler: SamplerOverrideConfig = field(default_factory=SamplerOverrideConfig)
    omega: float = 1.0
    cfg_interval: tuple[float, float] = (0.5, 1.0)
    output_format: Literal["ply", "pt", "both"] = "both"
    save_voxel_mesh: bool = True


@dataclass(frozen=True)
class P2PLatentBlendSSConfig:
    sampler: SamplerOverrideConfig = field(default_factory=SamplerOverrideConfig)
    inversion_mode: Literal["simple", "rf_solver"] = "simple"
    blend_enabled: bool = True
    blend_strength: float = 1.0
    hook: P2PHookConfig = field(default_factory=lambda: P2PHookConfig(1.0, 0.3, 1.0))
    patch_coverage_threshold: float = 0.0
    query_chunk: int = 1024
    denoise_cfg_strength: float = 5.0
    denoise_cfg_interval: tuple[float, float] = (0.5, 1.0)
    inversion_cfg_strength: float = 6.0
    inversion_cfg_interval: tuple[float, float] = (0.5, 1.0)
    output_format: Literal["ply", "pt", "both"] = "both"
    save_voxel_mesh: bool = True


@dataclass(frozen=True)
class UniEditSLATConfig:
    sampler: SamplerOverrideConfig = field(default_factory=SamplerOverrideConfig)
    omega: float = 1.0
    cfg_interval: tuple[float, float] = (0.5, 1.0)
    stage2_variant: Literal["preserve_uniedit", "free_target", "latent_replace_union"] = "preserve_uniedit"
    decode_modes: tuple[str, ...] = ("gaussian", "mesh")


@dataclass(frozen=True)
class P2PLatentBlendSLATConfig:
    sampler: SamplerOverrideConfig = field(default_factory=SamplerOverrideConfig)
    inversion_mode: Literal["simple", "rf_solver"] = "simple"
    blend_enabled: bool = True
    hook: P2PHookConfig = field(default_factory=lambda: P2PHookConfig(0.8, 0.0, 1.0))
    patch_coverage_threshold: float = 0.0
    query_chunk: int = 1024
    skip_source_decode: bool = True


@dataclass(frozen=True)
class SSStageSpec:
    plugin_name: Literal["uniedit_ss", "p2p_latent_blend_ss"]
    config: UniEditSSConfig | P2PLatentBlendSSConfig


@dataclass(frozen=True)
class SLATStageSpec:
    plugin_name: Literal["uniedit_slat", "p2p_latent_blend_slat"]
    config: UniEditSLATConfig | P2PLatentBlendSLATConfig


@dataclass(frozen=True)
class ExperimentConfig:
    entry_name: str
    preset_name: str
    run_mode: RunMode
    runtime: RuntimeConfig
    inputs: InputConfig
    preprocess: PreprocessConfig
    ss: SSStageSpec | None = None
    slat: SLATStageSpec | None = None

    def required_input_fields(self) -> tuple[str, ...]:
        required: set[str] = {"source_image", "edit_image"}

        if self.preprocess.mask_policy == "provided":
            required.add("mask_image")

        if self.run_mode in {"ss", "full"}:
            required.add("source_voxels")
            if self.ss is not None and self.ss.plugin_name == "p2p_latent_blend_ss":
                required.add("mask_glb")

        if self.run_mode in {"slat", "full"}:
            required.add("source_features")
            if self.run_mode == "slat":
                required.add("edited_coords")
            if (
                self.slat is not None
                and self.slat.plugin_name == "uniedit_slat"
                and self.slat.config.stage2_variant == "latent_replace_union"
            ):
                required.update({"source_voxels", "mask_glb"})
            if (
                self.slat is not None
                and self.slat.plugin_name == "p2p_latent_blend_slat"
                and self.slat.config.blend_enabled
            ):
                required.add("mask_glb")

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
