from __future__ import annotations

from pathlib import Path

from .canonical import export_glb_to_canonical_space

HUNYUAN21_NATIVE_HALF_EXTENT = 1.01


def export_hunyuan21_glb_to_canonical_space(
    *,
    input_glb: Path | str,
    output_glb: Path | str,
    apply_matte_nonmetal: bool = True,
    drop_normal: bool = False,
) -> dict:
    return export_glb_to_canonical_space(
        input_glb=input_glb,
        output_glb=output_glb,
        input_half_extent=HUNYUAN21_NATIVE_HALF_EXTENT,
        apply_matte_nonmetal=apply_matte_nonmetal,
        drop_normal=drop_normal,
    )
