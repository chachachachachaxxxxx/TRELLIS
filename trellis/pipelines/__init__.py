from . import samplers

__all__ = [
    "samplers",
    "TrellisImageTo3DPipeline",
    "TrellisTextTo3DPipeline",
    "from_pretrained",
]


def __getattr__(name: str):
    if name == "TrellisImageTo3DPipeline":
        from .trellis_image_to_3d import TrellisImageTo3DPipeline

        return TrellisImageTo3DPipeline
    if name == "TrellisTextTo3DPipeline":
        from .trellis_text_to_3d import TrellisTextTo3DPipeline

        return TrellisTextTo3DPipeline
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")


def from_pretrained(path: str):
    """
    Load a pipeline from a model folder or a Hugging Face model hub.

    Args:
        path: The path to the model. Can be either local path or a Hugging Face model name.
    """
    import os
    import json
    is_local = os.path.exists(f"{path}/pipeline.json")

    if is_local:
        config_file = f"{path}/pipeline.json"
    else:
        from huggingface_hub import hf_hub_download
        config_file = hf_hub_download(path, "pipeline.json")

    with open(config_file, 'r') as f:
        config = json.load(f)

    name = config["name"]
    if name == "TrellisImageTo3DPipeline":
        from .trellis_image_to_3d import TrellisImageTo3DPipeline

        pipeline_cls = TrellisImageTo3DPipeline
    elif name == "TrellisTextTo3DPipeline":
        from .trellis_text_to_3d import TrellisTextTo3DPipeline

        pipeline_cls = TrellisTextTo3DPipeline
    else:
        raise ValueError(f"Unknown pipeline class: {name}")

    return pipeline_cls.from_pretrained(path)
