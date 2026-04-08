from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Optional, Sequence, Type

from editing.io.case_loader import EditingCase


REPO_ROOT = Path(__file__).resolve().parents[2]


@dataclass(frozen=True)
class MethodSpec:
    name: str
    script_path: Path
    description: str
    required_fields: tuple[str, ...] = ()
    path_args: tuple[tuple[str, str], ...] = ()
    requires_source_assets: bool = False
    source_assets_description: str = "RF inversion assets"
    method_class: Optional[Type] = None  # New: optional method class

    def has_method_class(self) -> bool:
        """Check if this method has a method class implementation."""
        return self.method_class is not None

    def create_method(self):
        """Create method instance if method_class is available."""
        if self.method_class:
            return self.method_class()
        return None

    def validate_case(self, case: EditingCase) -> None:
        if self.requires_source_assets:
            if case.asset_dir is None and case.render_dir is None:
                raise RuntimeError(
                    f"Method '{self.name}' requires either asset_dir or render_dir for "
                    f"{self.source_assets_description}."
                )

        missing = [field for field in self.required_fields if getattr(case, field) is None]
        if missing:
            raise RuntimeError(
                f"Method '{self.name}' requires {', '.join(missing)}. "
                "Provide them via manifest or CLI overrides."
            )

    def build_command_args(
        self,
        *,
        case: EditingCase,
        case_name: str,
        model: str,
        seed: int | None,
        preprocess: bool | None,
        attn_backend: str,
        skip_render: bool,
        skip_glb: bool,
        skip_ply: bool,
        extra_args: Sequence[str],
        manifest_extra_args: Sequence[str],
    ) -> list[str]:
        args: list[str] = []
        if model:
            args.extend(["--model", model])
        if seed is not None:
            args.extend(["--seed", str(seed)])
        if case_name:
            args.extend(["--case-name", case_name])
        if attn_backend:
            args.extend(["--attn-backend", attn_backend])
        if preprocess is True:
            args.append("--preprocess")
        elif preprocess is False:
            args.append("--no-preprocess")
        if skip_render:
            args.append("--skip-render")
        if skip_glb:
            args.append("--skip-glb")
        if skip_ply:
            args.append("--skip-ply")

        if self.requires_source_assets:
            if case.render_dir is not None:
                args.extend(["--render_dir", str(case.render_dir)])
            elif case.asset_dir is not None:
                args.extend(["--asset-dir", str(case.asset_dir)])

        for field_name, flag in self.path_args:
            value = getattr(case, field_name)
            if value is not None:
                args.extend([flag, str(value)])

        args.extend(str(arg) for arg in manifest_extra_args)
        args.extend(str(arg) for arg in extra_args)
        return args


# Import method classes
try:
    from .image_prompt_to_prompt import ImagePromptToPromptMethod
except ImportError:
    ImagePromptToPromptMethod = None

try:
    from .text_prompt_to_prompt import TextPromptToPromptMethod
except ImportError:
    TextPromptToPromptMethod = None

try:
    from .image_slat_xor_fusion import ImageSlatXorFusionMethod
except ImportError:
    ImageSlatXorFusionMethod = None

try:
    from .image_prompt_to_prompt_rf_inversion import ImagePromptToPromptRFInversionMethod
except ImportError:
    ImagePromptToPromptRFInversionMethod = None

try:
    from .image_uniedit_rf_inversion import ImageUniEditRFInversionMethod
except ImportError:
    ImageUniEditRFInversionMethod = None

try:
    from .image_uniedit_p2p_hybrid import ImageUniEditP2PHybridMethod
except ImportError:
    ImageUniEditP2PHybridMethod = None

try:
    from .image_uniedit_euler import ImageUniEditEulerMethod
except ImportError:
    ImageUniEditEulerMethod = None

try:
    from .image_p2p_latent_blend import ImageP2PLatentBlendMethod
except ImportError:
    ImageP2PLatentBlendMethod = None


METHODS = {
    "image_prompt_to_prompt": MethodSpec(
        name="image_prompt_to_prompt",
        script_path=REPO_ROOT / "example_image_prompt_to_prompt.py",
        description="Pure image Prompt-to-Prompt editing on aligned source/edit/mask inputs.",
        required_fields=("source_image", "edit_image", "mask_image"),
        path_args=(
            ("source_image", "--source-image"),
            ("edit_image", "--edit-image"),
            ("mask_image", "--mask-image"),
        ),
        method_class=ImagePromptToPromptMethod,
    ),
    "text_prompt_to_prompt": MethodSpec(
        name="text_prompt_to_prompt",
        script_path=REPO_ROOT / "example_text_prompt_to_prompt.py",
        description="Text Prompt-to-Prompt editing with token alignment.",
        required_fields=("source_prompt", "edit_prompt"),
        path_args=(
            ("source_prompt", "--source-prompt"),
            ("edit_prompt", "--edit-prompt"),
        ),
        method_class=TextPromptToPromptMethod,
    ),
    "image_prompt_to_prompt_rf_inversion": MethodSpec(
        name="image_prompt_to_prompt_rf_inversion",
        script_path=REPO_ROOT / "example_image_prompt_to_prompt_rf_inversion.py",
        description="RF inversion initialization plus image Prompt-to-Prompt cross-attention injection.",
        required_fields=("edit_image",),
        path_args=(
            ("source_model", "--source_model"),
            ("source_image", "--source-image"),
            ("edit_image", "--edit-image"),
            ("mask_image", "--mask-image"),
        ),
        requires_source_assets=True,
        method_class=ImagePromptToPromptRFInversionMethod,
    ),
    "image_uniedit_rf_inversion": MethodSpec(
        name="image_uniedit_rf_inversion",
        script_path=REPO_ROOT / "example_image_uniedit_rf_inversion.py",
        description="RF inversion initialization plus UniEdit-style two-stage voxel editing.",
        required_fields=("edit_image", "mask_glb"),
        path_args=(
            ("source_model", "--source_model"),
            ("source_image", "--source-image"),
            ("edit_image", "--edit-image"),
            ("mask_glb", "--mask_glb"),
        ),
        requires_source_assets=True,
        method_class=ImageUniEditRFInversionMethod,
    ),
    "image_slat_xor_fusion": MethodSpec(
        name="image_slat_xor_fusion",
        script_path=REPO_ROOT / "example_image_slat_xor_fusion.py",
        description="Post-hoc SLAT block fusion that reuses source overlap and keeps target-only edit voxels.",
        required_fields=("edit_image",),
        path_args=(("edit_image", "--edit-image"),),
        requires_source_assets=True,
        source_assets_description="source SLAT assets",
        method_class=ImageSlatXorFusionMethod,
    ),
    "image_uniedit_p2p_hybrid": MethodSpec(
        name="image_uniedit_p2p_hybrid",
        script_path=REPO_ROOT / "example_image_uniedit_p2p_hybrid.py",
        description="Hybrid method combining UniEdit latent replacement with P2P attention injection during denoising.",
        required_fields=("edit_image", "mask_glb", "mask_image"),
        path_args=(
            ("source_model", "--source_model"),
            ("source_image", "--source-image"),
            ("edit_image", "--edit-image"),
            ("mask_glb", "--mask_glb"),
            ("mask_image", "--mask-image"),
        ),
        requires_source_assets=True,
        method_class=ImageUniEditP2PHybridMethod,
    ),
    "image_p2p_latent_blend": MethodSpec(
        name="image_p2p_latent_blend",
        script_path=REPO_ROOT / "example_image_p2p_latent_blend.py",  # Legacy script (not used)
        description="P2P with per-step latent blending during denoising. Uses real source SLAT features.",
        required_fields=("source_image", "edit_image", "mask_image"),
        path_args=(
            ("source_image", "--source-image"),
            ("edit_image", "--edit-image"),
            ("mask_image", "--mask-image"),
        ),
        requires_source_assets=True,  # Needs source SLAT features
        source_assets_description="source SLAT features (features.npz)",
        method_class=ImageP2PLatentBlendMethod,
    ),
    "image_uniedit_euler": MethodSpec(
        name="image_uniedit_euler",
        script_path=REPO_ROOT / "example_image_uniedit_euler.py",
        description="UniEdit with first-order Euler + predictor-corrector (simplified version without RF-Solver).",
        required_fields=("edit_image", "mask_glb"),
        path_args=(
            ("source_model", "--source_model"),
            ("source_image", "--source-image"),
            ("edit_image", "--edit-image"),
            ("mask_glb", "--mask_glb"),
        ),
        requires_source_assets=True,
        method_class=ImageUniEditEulerMethod,
    ),
}


def list_methods() -> list[MethodSpec]:
    return [METHODS[name] for name in sorted(METHODS)]


def get_method(name: str) -> MethodSpec:
    try:
        return METHODS[name]
    except KeyError as exc:
        available = ", ".join(sorted(METHODS))
        raise RuntimeError(f"Unknown method '{name}'. Available methods: {available}.") from exc
