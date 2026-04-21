from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Literal


ConditionOp = Literal[
    "eq",
    "ne",
    "is_none",
    "not_none",
    "gt",
    "ge",
    "lt",
    "le",
    "in",
    "not_in",
    "first_gt_second",
    "gt_path",
]


@dataclass(frozen=True)
class ConditionSpec:
    path: str
    op: ConditionOp
    value: Any = None


@dataclass(frozen=True)
class ValidationRule:
    when: tuple[ConditionSpec, ...]
    message: str
    require: tuple[ConditionSpec, ...] = ()


def _resolve_path(obj: Any, path: str) -> Any:
    value = obj
    for part in path.split("."):
        value = getattr(value, part)
    return value


def _matches_condition(obj: Any, condition: ConditionSpec) -> bool:
    value = _resolve_path(obj, condition.path)
    if condition.op == "eq":
        return value == condition.value
    if condition.op == "ne":
        return value != condition.value
    if condition.op == "is_none":
        return value is None
    if condition.op == "not_none":
        return value is not None
    if condition.op == "gt":
        return value > condition.value
    if condition.op == "ge":
        return value >= condition.value
    if condition.op == "lt":
        return value < condition.value
    if condition.op == "le":
        return value <= condition.value
    if condition.op == "in":
        return value in condition.value
    if condition.op == "not_in":
        return value not in condition.value
    if condition.op == "first_gt_second":
        return value[0] > value[1]
    if condition.op == "gt_path":
        return value > _resolve_path(obj, condition.value)
    raise ValueError(f"Unsupported validation op: {condition.op}")


def _evaluate_rules(config: Any, label: str, rules: tuple[ValidationRule, ...]) -> None:
    for rule in rules:
        if not all(_matches_condition(config, condition) for condition in rule.when):
            continue
        if rule.require and all(_matches_condition(config, condition) for condition in rule.require):
            continue
        raise RuntimeError(rule.message.format(label=label))


SS_VALIDATION_RULES: tuple[ValidationRule, ...] = (
    ValidationRule(
        when=(ConditionSpec("method", "eq", "uniedit"),),
        require=(ConditionSpec("controls.uniedit.enabled", "eq", True),),
        message="{label}.method='uniedit' requires {label}.controls.uniedit.enabled=true.",
    ),
    ValidationRule(
        when=(ConditionSpec("controls.uniedit.enabled", "eq", True),),
        require=(ConditionSpec("method", "eq", "uniedit"),),
        message="{label}.controls.uniedit.enabled=true requires {label}.method='uniedit'.",
    ),
    ValidationRule(
        when=(ConditionSpec("method", "eq", "flowedit"),),
        require=(ConditionSpec("inversion.enabled", "eq", False),),
        message="{label}.method='flowedit' requires {label}.inversion.enabled=false.",
    ),
    ValidationRule(
        when=(
            ConditionSpec("method", "eq", "flowedit"),
            ConditionSpec("controls.p2p.enabled", "eq", True),
        ),
        message="{label}.method='flowedit' does not support {label}.controls.p2p.enabled=true.",
    ),
    ValidationRule(
        when=(
            ConditionSpec("method", "eq", "flowedit"),
            ConditionSpec("controls.kv_blend.enabled", "eq", True),
        ),
        message="{label}.method='flowedit' does not support {label}.controls.kv_blend.enabled=true.",
    ),
    ValidationRule(
        when=(
            ConditionSpec("method", "eq", "flowedit"),
            ConditionSpec("controls.latent_blend.enabled", "eq", True),
        ),
        message="{label}.method='flowedit' does not support {label}.controls.latent_blend.enabled=true.",
    ),
    ValidationRule(
        when=(ConditionSpec("flowedit.n_avg", "le", 0),),
        message="{label}.flowedit.n_avg must be > 0.",
    ),
    ValidationRule(
        when=(ConditionSpec("flowedit.start_step", "lt", 0),),
        message="{label}.flowedit.start_step must be >= 0.",
    ),
    ValidationRule(
        when=(ConditionSpec("flowedit.cfg_interval", "first_gt_second"),),
        message="{label}.flowedit.cfg_interval must be ordered as [start, end].",
    ),
    ValidationRule(
        when=(ConditionSpec("method", "eq", "anchorflow"),),
        require=(ConditionSpec("inversion.enabled", "eq", False),),
        message="{label}.method='anchorflow' requires {label}.inversion.enabled=false.",
    ),
    ValidationRule(
        when=(
            ConditionSpec("method", "eq", "anchorflow"),
            ConditionSpec("controls.p2p.enabled", "eq", True),
        ),
        message="{label}.method='anchorflow' does not support {label}.controls.p2p.enabled=true.",
    ),
    ValidationRule(
        when=(
            ConditionSpec("method", "eq", "anchorflow"),
            ConditionSpec("controls.kv_blend.enabled", "eq", True),
        ),
        message="{label}.method='anchorflow' does not support {label}.controls.kv_blend.enabled=true.",
    ),
    ValidationRule(
        when=(
            ConditionSpec("method", "eq", "anchorflow"),
            ConditionSpec("controls.latent_blend.enabled", "eq", True),
        ),
        message="{label}.method='anchorflow' does not support {label}.controls.latent_blend.enabled=true.",
    ),
    ValidationRule(
        when=(ConditionSpec("anchorflow.n_avg", "le", 0),),
        message="{label}.anchorflow.n_avg must be > 0.",
    ),
    ValidationRule(
        when=(ConditionSpec("anchorflow.n_max", "le", 0),),
        message="{label}.anchorflow.n_max must be > 0.",
    ),
    ValidationRule(
        when=(ConditionSpec("anchorflow.cfg_interval", "first_gt_second"),),
        message="{label}.anchorflow.cfg_interval must be ordered as [start, end].",
    ),
    ValidationRule(
        when=(ConditionSpec("anchorflow.center_weight", "lt", 0),),
        message="{label}.anchorflow.center_weight must be >= 0.",
    ),
    ValidationRule(
        when=(ConditionSpec("anchorflow.band_weight", "lt", 0),),
        message="{label}.anchorflow.band_weight must be >= 0.",
    ),
    ValidationRule(
        when=(ConditionSpec("anchorflow.ortho_weight", "lt", 0),),
        message="{label}.anchorflow.ortho_weight must be >= 0.",
    ),
    ValidationRule(
        when=(ConditionSpec("anchorflow.residual_weight", "lt", 0),),
        message="{label}.anchorflow.residual_weight must be >= 0.",
    ),
    ValidationRule(
        when=(ConditionSpec("anchorflow.band_min_ratio", "lt", 0),),
        message="{label}.anchorflow.band_min_ratio must be >= 0.",
    ),
    ValidationRule(
        when=(ConditionSpec("anchorflow.band_max_ratio", "lt", 0),),
        message="{label}.anchorflow.band_max_ratio must be >= 0.",
    ),
    ValidationRule(
        when=(ConditionSpec("anchorflow.band_min_ratio", "gt_path", "anchorflow.band_max_ratio"),),
        message="{label}.anchorflow.band_min_ratio must be <= {label}.anchorflow.band_max_ratio.",
    ),
    ValidationRule(
        when=(ConditionSpec("anchorflow.margin_scale", "le", 0),),
        message="{label}.anchorflow.margin_scale must be > 0.",
    ),
    ValidationRule(
        when=(ConditionSpec("anchorflow.direction_gate_tau", "lt", 0),),
        message="{label}.anchorflow.direction_gate_tau must be >= 0.",
    ),
    ValidationRule(
        when=(ConditionSpec("anchorflow.eps", "le", 0),),
        message="{label}.anchorflow.eps must be > 0.",
    ),
    ValidationRule(
        when=(
            ConditionSpec("postprocess.mode", "eq", "boundary_band_restore"),
            ConditionSpec("postprocess.boundary_band.band_width_voxels", "le", 0),
        ),
        message="{label}.postprocess.boundary_band.band_width_voxels must be > 0 for boundary_band_restore.",
    ),
    ValidationRule(
        when=(
            ConditionSpec("postprocess.mode", "eq", "boundary_band_restore"),
            ConditionSpec("postprocess.boundary_band.band_target_neighbor_threshold", "le", 0),
        ),
        message=(
            "{label}.postprocess.boundary_band.band_target_neighbor_threshold must be > 0 "
            "for boundary_band_restore."
        ),
    ),
    ValidationRule(
        when=(
            ConditionSpec("postprocess.mode", "eq", "boundary_band_restore"),
            ConditionSpec("postprocess.boundary_band.band_target_neighbor_threshold", "gt", 6),
        ),
        message=(
            "{label}.postprocess.boundary_band.band_target_neighbor_threshold must be <= 6 "
            "for boundary_band_restore."
        ),
    ),
    ValidationRule(
        when=(
            ConditionSpec("controls.latent_blend.enabled", "eq", True),
            ConditionSpec("controls.uniedit.enabled", "eq", True),
        ),
        message="{label}.controls.latent_blend and {label}.controls.uniedit cannot both be enabled.",
    ),
    ValidationRule(
        when=(
            ConditionSpec("controls.kv_blend.enabled", "eq", True),
            ConditionSpec("controls.p2p.enabled", "eq", True),
        ),
        message="{label}.controls.kv_blend and {label}.controls.p2p cannot both be enabled yet.",
    ),
    ValidationRule(
        when=(
            ConditionSpec("controls.kv_blend.enabled", "eq", True),
            ConditionSpec("controls.uniedit.enabled", "eq", True),
        ),
        message="{label}.controls.kv_blend and {label}.controls.uniedit cannot both be enabled.",
    ),
    ValidationRule(
        when=(ConditionSpec("inversion.enabled", "eq", False),),
        require=(ConditionSpec("inversion.denoise_init", "eq", "random_noise"),),
        message="{label}.inversion.enabled=false requires denoise_init=random_noise.",
    ),
    ValidationRule(
        when=(
            ConditionSpec("inversion.enabled", "eq", False),
            ConditionSpec("inversion.inversion_steps", "not_none"),
        ),
        message="{label}.inversion.inversion_steps requires {label}.inversion.enabled=true.",
    ),
    ValidationRule(
        when=(
            ConditionSpec("inversion.enabled", "eq", True),
            ConditionSpec("inversion.denoise_init", "eq", "random_noise"),
        ),
        message="{label}.inversion.enabled=true requires denoise_init=inverted_terminal_noise.",
    ),
    ValidationRule(
        when=(
            ConditionSpec("controls.latent_blend.enabled", "eq", True),
            ConditionSpec("inversion.enabled", "eq", False),
        ),
        message="{label}.controls.latent_blend requires {label}.inversion.enabled=true.",
    ),
    ValidationRule(
        when=(
            ConditionSpec("controls.kv_blend.enabled", "eq", True),
            ConditionSpec("inversion.enabled", "eq", False),
        ),
        message="{label}.controls.kv_blend requires {label}.inversion.enabled=true.",
    ),
    ValidationRule(
        when=(
            ConditionSpec("controls.uniedit.enabled", "eq", True),
            ConditionSpec("inversion.enabled", "eq", False),
        ),
        message="{label}.controls.uniedit currently requires {label}.inversion.enabled=true.",
    ),
    ValidationRule(
        when=(ConditionSpec("inversion.predictor_corrector_steps", "lt", 0),),
        message="{label}.inversion.predictor_corrector_steps must be >= 0.",
    ),
    ValidationRule(
        when=(
            ConditionSpec("inversion.inversion_steps", "not_none"),
            ConditionSpec("inversion.inversion_steps", "le", 0),
        ),
        message="{label}.inversion.inversion_steps must be > 0 when set.",
    ),
)


SLAT_VALIDATION_RULES: tuple[ValidationRule, ...] = (
    ValidationRule(
        when=(ConditionSpec("method", "eq", "uniedit"),),
        require=(ConditionSpec("controls.uniedit.enabled", "eq", True),),
        message="{label}.method='uniedit' requires {label}.controls.uniedit.enabled=true.",
    ),
    ValidationRule(
        when=(ConditionSpec("controls.uniedit.enabled", "eq", True),),
        require=(ConditionSpec("method", "eq", "uniedit"),),
        message="{label}.controls.uniedit.enabled=true requires {label}.method='uniedit'.",
    ),
    ValidationRule(
        when=(ConditionSpec("method", "eq", "direct_target"),),
        require=(ConditionSpec("inversion.enabled", "eq", False),),
        message="{label}.method='direct_target' requires {label}.inversion.enabled=false.",
    ),
    ValidationRule(
        when=(
            ConditionSpec("method", "eq", "direct_target"),
            ConditionSpec("controls.p2p.enabled", "eq", True),
        ),
        message="{label}.method='direct_target' does not support {label}.controls.p2p.enabled=true.",
    ),
    ValidationRule(
        when=(
            ConditionSpec("method", "eq", "direct_target"),
            ConditionSpec("controls.kv_blend.enabled", "eq", True),
        ),
        message="{label}.method='direct_target' does not support {label}.controls.kv_blend.enabled=true.",
    ),
    ValidationRule(
        when=(
            ConditionSpec("method", "eq", "direct_target"),
            ConditionSpec("controls.latent_blend.enabled", "eq", True),
        ),
        message="{label}.method='direct_target' does not support {label}.controls.latent_blend.enabled=true.",
    ),
    ValidationRule(
        when=(
            ConditionSpec("method", "eq", "direct_target"),
            ConditionSpec("controls.nano3d_replace.enabled", "eq", True),
        ),
        message=(
            "{label}.method='direct_target' uses {label}.postprocess instead of "
            "{label}.controls.nano3d_replace.enabled=true."
        ),
    ),
    ValidationRule(
        when=(
            ConditionSpec("controls.nano3d_replace.enabled", "eq", True),
            ConditionSpec("postprocess.mode", "ne", "none"),
        ),
        message="{label}.controls.nano3d_replace.enabled and {label}.postprocess.mode cannot both be enabled.",
    ),
    ValidationRule(
        when=(ConditionSpec("postprocess.mode", "eq", "boundary_band_restore"),),
        message="{label}.postprocess.mode='boundary_band_restore' is only supported for ss.",
    ),
    ValidationRule(
        when=(ConditionSpec("postprocess.mode", "eq", "restore_all_outside_mask"),),
        message="{label}.postprocess.mode='restore_all_outside_mask' is only supported for ss.",
    ),
    ValidationRule(
        when=(
            ConditionSpec("controls.latent_blend.enabled", "eq", True),
            ConditionSpec("controls.uniedit.enabled", "eq", True),
        ),
        message="{label}.controls.latent_blend and {label}.controls.uniedit cannot both be enabled.",
    ),
    ValidationRule(
        when=(
            ConditionSpec("controls.latent_blend.enabled", "eq", True),
            ConditionSpec("controls.nano3d_replace.enabled", "eq", True),
        ),
        message="{label}.controls.latent_blend and {label}.controls.nano3d_replace cannot both be enabled.",
    ),
    ValidationRule(
        when=(
            ConditionSpec("controls.nano3d_replace.enabled", "eq", True),
            ConditionSpec("controls.uniedit.enabled", "eq", True),
        ),
        message="{label}.controls.nano3d_replace and {label}.controls.uniedit cannot both be enabled.",
    ),
    ValidationRule(
        when=(
            ConditionSpec("controls.kv_blend.enabled", "eq", True),
            ConditionSpec("controls.p2p.enabled", "eq", True),
        ),
        message="{label}.controls.kv_blend and {label}.controls.p2p cannot both be enabled yet.",
    ),
    ValidationRule(
        when=(
            ConditionSpec("controls.kv_blend.enabled", "eq", True),
            ConditionSpec("controls.uniedit.enabled", "eq", True),
        ),
        message="{label}.controls.kv_blend and {label}.controls.uniedit cannot both be enabled.",
    ),
    ValidationRule(
        when=(ConditionSpec("inversion.enabled", "eq", False),),
        require=(ConditionSpec("inversion.denoise_init", "eq", "random_noise"),),
        message="{label}.inversion.enabled=false requires denoise_init=random_noise.",
    ),
    ValidationRule(
        when=(
            ConditionSpec("inversion.enabled", "eq", False),
            ConditionSpec("inversion.inversion_steps", "not_none"),
        ),
        message="{label}.inversion.inversion_steps requires {label}.inversion.enabled=true.",
    ),
    ValidationRule(
        when=(ConditionSpec("inversion.scope", "not_in", {"full_source", "preserve_only"}),),
        message=(
            "{label}.inversion.scope must be one of ('full_source', 'preserve_only'), "
            "got invalid value."
        ),
    ),
    ValidationRule(
        when=(
            ConditionSpec("controls.latent_blend.enabled", "eq", True),
            ConditionSpec("inversion.enabled", "eq", False),
        ),
        message="{label}.controls.latent_blend requires {label}.inversion.enabled=true.",
    ),
    ValidationRule(
        when=(
            ConditionSpec("controls.kv_blend.enabled", "eq", True),
            ConditionSpec("inversion.enabled", "eq", False),
        ),
        message="{label}.controls.kv_blend requires {label}.inversion.enabled=true.",
    ),
    ValidationRule(
        when=(
            ConditionSpec("controls.uniedit.enabled", "eq", True),
            ConditionSpec("inversion.enabled", "eq", False),
        ),
        message="{label}.controls.uniedit currently requires {label}.inversion.enabled=true.",
    ),
    ValidationRule(
        when=(
            ConditionSpec("inversion.scope", "eq", "preserve_only"),
            ConditionSpec("inversion.enabled", "eq", False),
        ),
        message="{label}.inversion.scope='preserve_only' requires {label}.inversion.enabled=true.",
    ),
    ValidationRule(
        when=(ConditionSpec("inversion.predictor_corrector_steps", "lt", 0),),
        message="{label}.inversion.predictor_corrector_steps must be >= 0.",
    ),
    ValidationRule(
        when=(
            ConditionSpec("inversion.inversion_steps", "not_none"),
            ConditionSpec("inversion.inversion_steps", "le", 0),
        ),
        message="{label}.inversion.inversion_steps must be > 0 when set.",
    ),
)


def validate_ss_stage_config(config: Any, label: str) -> None:
    _evaluate_rules(config, label, SS_VALIDATION_RULES)


def validate_slat_stage_config(config: Any, label: str) -> None:
    _evaluate_rules(config, label, SLAT_VALIDATION_RULES)
