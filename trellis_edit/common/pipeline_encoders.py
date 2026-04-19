from __future__ import annotations

from typing import Any


def _pretrained_ckpt_path(model_root: str, ckpt_name: str) -> str:
    model_root = str(model_root).rstrip("/")
    return f"{model_root}/ckpts/{ckpt_name}"


def ensure_pipeline_encoders(
    pipeline: Any,
    *,
    model_root: str,
    require_sparse_structure_encoder: bool = False,
    require_slat_encoder: bool = False,
) -> None:
    """Load auxiliary VAE encoders into a TRELLIS pipeline on demand.

    TRELLIS image pipelines expose the flow models and decoders needed for
    generation, but local editing methods also need the VAE encoders to lift
    source voxels/features back into latent space.
    """
    if not (require_sparse_structure_encoder or require_slat_encoder):
        return

    from trellis import models

    device = pipeline.device

    if require_sparse_structure_encoder and "sparse_structure_encoder" not in pipeline.models:
        encoder = models.from_pretrained(_pretrained_ckpt_path(model_root, "ss_enc_conv3d_16l8_fp16"))
        encoder.eval()
        encoder.to(device)
        pipeline.models["sparse_structure_encoder"] = encoder

    if require_slat_encoder and "slat_encoder" not in pipeline.models:
        encoder = models.from_pretrained(_pretrained_ckpt_path(model_root, "slat_enc_swin8_B_64l8_fp16"))
        encoder.eval()
        encoder.to(device)
        pipeline.models["slat_encoder"] = encoder
