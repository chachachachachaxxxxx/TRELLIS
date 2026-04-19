from __future__ import annotations

from dataclasses import dataclass

from .config import (
    AdaptiveForegroundScaleConfig,
    ExperimentConfig,
    PreprocessConfig,
    RuntimeConfig,
    default_slat_stage,
    default_ss_stage,
)


@dataclass(frozen=True)
class PresetDefinition:
    name: str
    preprocess: PreprocessConfig

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
            ss=default_ss_stage(),
            slat=default_slat_stage(),
        )


PRESETS: dict[str, PresetDefinition] = {
    "edit_default": PresetDefinition(
        name="edit_default",
        preprocess=PreprocessConfig(
            crop_policy="union_crop",
            mask_policy="provided",
            adaptive_foreground_scale=AdaptiveForegroundScaleConfig(enabled=True),
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
    "edit_full": EntrypointDefinition(
        name="edit_full",
        description="Composable full pipeline; stage behavior is configured in ss/slat.",
        preset_name="edit_default",
        run_mode="full",
    ),
    "edit_ss": EntrypointDefinition(
        name="edit_ss",
        description="Composable sparse-structure stage only; behavior is configured in ss.",
        preset_name="edit_default",
        run_mode="ss",
    ),
    "edit_slat": EntrypointDefinition(
        name="edit_slat",
        description="Composable SLAT stage only; behavior is configured in slat.",
        preset_name="edit_default",
        run_mode="slat",
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
