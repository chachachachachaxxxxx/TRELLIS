from __future__ import annotations

from dataclasses import dataclass

from .config import (
    ExperimentConfig,
    P2PLatentBlendSLATConfig,
    P2PLatentBlendSSConfig,
    PreprocessConfig,
    RuntimeConfig,
    SLATStageSpec,
    SSStageSpec,
    UniEditSLATConfig,
    UniEditSSConfig,
)


@dataclass(frozen=True)
class PresetDefinition:
    name: str
    preprocess: PreprocessConfig
    ss: SSStageSpec | None
    slat: SLATStageSpec | None

    def build(
        self,
        *,
        run_mode: str,
        runtime: RuntimeConfig,
        inputs,
        entry_name: str | None = None,
    ) -> ExperimentConfig:
        return ExperimentConfig(
            entry_name=entry_name or self.name,
            preset_name=self.name,
            run_mode=run_mode,
            runtime=runtime,
            inputs=inputs,
            preprocess=self.preprocess,
            ss=self.ss,
            slat=self.slat,
        )


PRESETS: dict[str, PresetDefinition] = {
    "uniedit_full": PresetDefinition(
        name="uniedit_full",
        preprocess=PreprocessConfig(
            crop_policy="union_crop",
            foreground_policy="alpha_or_rembg",
            mask_policy="provided",
        ),
        ss=SSStageSpec(
            plugin_name="uniedit_ss",
            config=UniEditSSConfig(),
        ),
        slat=SLATStageSpec(
            plugin_name="uniedit_slat",
            config=UniEditSLATConfig(),
        ),
    ),
    "p2p_latent_blend_full": PresetDefinition(
        name="p2p_latent_blend_full",
        preprocess=PreprocessConfig(
            crop_policy="union_crop",
            foreground_policy="alpha_or_rembg",
            mask_policy="provided",
        ),
        ss=SSStageSpec(
            plugin_name="p2p_latent_blend_ss",
            config=P2PLatentBlendSSConfig(),
        ),
        slat=SLATStageSpec(
            plugin_name="p2p_latent_blend_slat",
            config=P2PLatentBlendSLATConfig(),
        ),
    ),
    "ss_p2p_slat_uniedit": PresetDefinition(
        name="ss_p2p_slat_uniedit",
        preprocess=PreprocessConfig(
            crop_policy="union_crop",
            foreground_policy="alpha_or_rembg",
            mask_policy="provided",
        ),
        ss=SSStageSpec(
            plugin_name="p2p_latent_blend_ss",
            config=P2PLatentBlendSSConfig(),
        ),
        slat=SLATStageSpec(
            plugin_name="uniedit_slat",
            config=UniEditSLATConfig(),
        ),
    ),
    "ss_uniedit_slat_p2p": PresetDefinition(
        name="ss_uniedit_slat_p2p",
        preprocess=PreprocessConfig(
            crop_policy="union_crop",
            foreground_policy="alpha_or_rembg",
            mask_policy="provided",
        ),
        ss=SSStageSpec(
            plugin_name="uniedit_ss",
            config=UniEditSSConfig(),
        ),
        slat=SLATStageSpec(
            plugin_name="p2p_latent_blend_slat",
            config=P2PLatentBlendSLATConfig(),
        ),
    ),
}


@dataclass(frozen=True)
class EntrypointDefinition:
    name: str
    description: str
    preset_name: str
    run_mode: str

    def build(self, *, runtime: RuntimeConfig, inputs) -> ExperimentConfig:
        preset = get_preset(self.preset_name)
        return preset.build(
            entry_name=self.name,
            run_mode=self.run_mode,
            runtime=runtime,
            inputs=inputs,
        )


ENTRYPOINTS: dict[str, EntrypointDefinition] = {
    "uniedit_full": EntrypointDefinition(
        name="uniedit_full",
        description="Composable UniEdit pipeline with SS and SLAT stages.",
        preset_name="uniedit_full",
        run_mode="full",
    ),
    "uniedit_ss": EntrypointDefinition(
        name="uniedit_ss",
        description="Composable UniEdit sparse-structure stage only.",
        preset_name="uniedit_full",
        run_mode="ss",
    ),
    "uniedit_slat": EntrypointDefinition(
        name="uniedit_slat",
        description="Composable UniEdit SLAT stage only.",
        preset_name="uniedit_full",
        run_mode="slat",
    ),
    "p2p_latent_blend_full": EntrypointDefinition(
        name="p2p_latent_blend_full",
        description="Composable P2P latent-blend pipeline with SS and SLAT stages.",
        preset_name="p2p_latent_blend_full",
        run_mode="full",
    ),
    "p2p_latent_blend_ss": EntrypointDefinition(
        name="p2p_latent_blend_ss",
        description="Composable P2P latent-blend sparse-structure stage only.",
        preset_name="p2p_latent_blend_full",
        run_mode="ss",
    ),
    "p2p_latent_blend_slat": EntrypointDefinition(
        name="p2p_latent_blend_slat",
        description="Composable P2P latent-blend SLAT stage only.",
        preset_name="p2p_latent_blend_full",
        run_mode="slat",
    ),
    "ss_p2p_slat_uniedit": EntrypointDefinition(
        name="ss_p2p_slat_uniedit",
        description="Composable mixed pipeline: P2P SS followed by UniEdit SLAT.",
        preset_name="ss_p2p_slat_uniedit",
        run_mode="full",
    ),
    "ss_uniedit_slat_p2p": EntrypointDefinition(
        name="ss_uniedit_slat_p2p",
        description="Composable mixed pipeline: UniEdit SS followed by P2P SLAT.",
        preset_name="ss_uniedit_slat_p2p",
        run_mode="full",
    ),
}


def get_preset(name: str) -> PresetDefinition:
    try:
        return PRESETS[name]
    except KeyError as exc:
        raise RuntimeError(f"Unknown preset: {name}") from exc


def list_presets() -> list[PresetDefinition]:
    return [PRESETS[name] for name in sorted(PRESETS)]


def has_entrypoint(name: str) -> bool:
    return name in ENTRYPOINTS


def get_entrypoint(name: str) -> EntrypointDefinition:
    try:
        return ENTRYPOINTS[name]
    except KeyError as exc:
        raise RuntimeError(f"Unknown composable entrypoint: {name}") from exc


def list_entrypoints() -> list[EntrypointDefinition]:
    return [ENTRYPOINTS[name] for name in sorted(ENTRYPOINTS)]
