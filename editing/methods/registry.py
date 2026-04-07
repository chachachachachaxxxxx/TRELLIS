from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Sequence

from editing.io.case_loader import EditingCase


REPO_ROOT = Path(__file__).resolve().parents[2]


@dataclass(frozen=True)
class MethodSpec:
    name: str
    script_path: Path
    description: str

    def validate_case(self, case: EditingCase) -> None:
        if self.name == "image_prompt_to_prompt":
            missing = []
            if case.source_image is None:
                missing.append("source_image")
            if case.edit_image is None:
                missing.append("edit_image")
            if case.mask_image is None:
                missing.append("mask_image")
            if missing:
                raise RuntimeError(
                    f"Method '{self.name}' requires {', '.join(missing)}. "
                    "Provide them via manifest or CLI overrides."
                )
            return

        if self.name == "image_prompt_to_prompt_rf_inversion":
            if case.source_model is None and case.render_dir is None:
                raise RuntimeError(
                    f"Method '{self.name}' requires either source_model or render_dir for RF inversion assets."
                )
            if case.edit_image is None:
                raise RuntimeError(
                    f"Method '{self.name}' requires edit_image. Provide it via manifest or CLI override."
                )
            return

        if self.name == "image_uniedit_rf_inversion":
            if case.source_model is None and case.render_dir is None:
                raise RuntimeError(
                    f"Method '{self.name}' requires either source_model or render_dir for RF inversion assets."
                )
            if case.edit_image is None:
                raise RuntimeError(
                    f"Method '{self.name}' requires edit_image. Provide it via manifest or CLI override."
                )
            if case.mask_glb is None:
                raise RuntimeError(
                    f"Method '{self.name}' requires mask_glb. Provide it via manifest or CLI override."
                )
            return

        raise RuntimeError(f"Unknown method registry validation rule for: {self.name}")

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

        if self.name == "image_prompt_to_prompt":
            args.extend(["--source-image", str(case.source_image)])
            args.extend(["--edit-image", str(case.edit_image)])
            args.extend(["--mask-image", str(case.mask_image)])
        elif self.name in {"image_prompt_to_prompt_rf_inversion", "image_uniedit_rf_inversion"}:
            if case.render_dir is not None:
                args.extend(["--render_dir", str(case.render_dir)])
            elif case.source_model is not None:
                args.extend(["--source-model", str(case.source_model)])

            if case.input_model is not None:
                args.extend(["--input_model", str(case.input_model)])
            if case.source_image is not None:
                args.extend(["--source-image", str(case.source_image)])
            if case.edit_image is not None:
                args.extend(["--edit-image", str(case.edit_image)])
            if self.name == "image_prompt_to_prompt_rf_inversion" and case.mask_image is not None:
                args.extend(["--mask-image", str(case.mask_image)])
            if self.name == "image_uniedit_rf_inversion" and case.mask_glb is not None:
                args.extend(["--mask_glb", str(case.mask_glb)])
        else:
            raise RuntimeError(f"Unknown method: {self.name}")

        args.extend(str(arg) for arg in manifest_extra_args)
        args.extend(str(arg) for arg in extra_args)
        return args


METHODS = {
    "image_prompt_to_prompt": MethodSpec(
        name="image_prompt_to_prompt",
        script_path=REPO_ROOT / "example_image_prompt_to_prompt.py",
        description="Pure image Prompt-to-Prompt editing on aligned source/edit/mask inputs.",
    ),
    "image_prompt_to_prompt_rf_inversion": MethodSpec(
        name="image_prompt_to_prompt_rf_inversion",
        script_path=REPO_ROOT / "example_image_prompt_to_prompt_rf_inversion.py",
        description="RF inversion initialization plus image Prompt-to-Prompt cross-attention injection.",
    ),
    "image_uniedit_rf_inversion": MethodSpec(
        name="image_uniedit_rf_inversion",
        script_path=REPO_ROOT / "example_image_uniedit_rf_inversion.py",
        description="RF inversion initialization plus UniEdit-style two-stage voxel editing.",
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
