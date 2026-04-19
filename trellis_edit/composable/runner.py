from __future__ import annotations

import os
from dataclasses import asdict, is_dataclass
from pathlib import Path
from typing import Any

from PIL import Image

from trellis_edit.common import build_backend_config, build_experiment_output_layout, ensure_dir, save_outputs, write_json
from trellis_edit.preprocess import save_preprocessed_inputs

from .adapters import SLAT_PLUGIN_REGISTRY, SS_PLUGIN_REGISTRY
from .artifacts import ExperimentResult
from .base import ExperimentContext, LoadedInputs
from .config import ExperimentConfig
from .preprocess import SharedImagePreprocessPlugin


def _serialize(value: Any) -> Any:
    if is_dataclass(value):
        return _serialize(asdict(value))
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, dict):
        return {key: _serialize(val) for key, val in value.items()}
    if isinstance(value, (list, tuple)):
        return [_serialize(item) for item in value]
    return value


def _load_required_image(path: Path, label: str) -> Image.Image:
    if not path.is_file():
        raise RuntimeError(f"{label} does not exist: {path}")
    return Image.open(path)


class ComposableExperimentRunner:
    def __init__(self, config: ExperimentConfig):
        self.config = config

    def _load_pipeline(self):
        backend = build_backend_config(
            attn_backend=self.config.runtime.attn_backend,
            sparse_attn_backend=self.config.runtime.sparse_attn_backend,
            spconv_algo=self.config.runtime.spconv_algo,
        )
        for key, value in backend.env.items():
            os.environ[key] = value

        import torch
        from trellis.pipelines import TrellisImageTo3DPipeline

        requested_device = self.config.runtime.device
        if "CUDA_VISIBLE_DEVICES" in os.environ and requested_device.startswith("cuda:"):
            load_device = "cuda:0"
        else:
            load_device = requested_device

        pipeline = TrellisImageTo3DPipeline.from_pretrained(self.config.runtime.model)
        pipeline.to(torch.device(load_device))
        return pipeline

    def _load_inputs(self) -> LoadedInputs:
        source_image = None
        if self.config.inputs.source_image is not None:
            source_image = _load_required_image(self.config.inputs.source_image, "source_image")
        edit_image = _load_required_image(self.config.inputs.edit_image, "edit_image")
        mask_image = None
        if self.config.inputs.mask_image is not None:
            mask_image = _load_required_image(self.config.inputs.mask_image, "mask_image")
        return LoadedInputs(
            source_image=source_image,
            edit_image=edit_image,
            mask_image=mask_image,
        )

    def _write_config(self, layout) -> None:
        write_json(layout.case_dir / "experiment_config.json", _serialize(self.config))

    @staticmethod
    def _primary_coords_path(ss_artifact) -> Path:
        for key in ("coords_pt", "coords_ply"):
            if key in ss_artifact.artifact_paths:
                return Path(ss_artifact.artifact_paths[key])
        raise RuntimeError("SS stage did not emit coords_pt or coords_ply")

    def run(self) -> ExperimentResult:
        self.config.validate_inputs()
        if self.config.run_mode in {"ss", "full"} and self.config.ss is None:
            raise RuntimeError("run_mode requires an ss stage, but config.ss is None")
        if self.config.run_mode in {"slat", "full"} and self.config.slat is None:
            raise RuntimeError("run_mode requires a slat stage, but config.slat is None")

        pipeline = self._load_pipeline()
        loaded_inputs = self._load_inputs()
        layout = build_experiment_output_layout(
            method_name=self.config.runtime.output_group or self.config.entry_name,
            case_name=self.config.runtime.case_name,
            outputs_root=self.config.runtime.output_root,
        )

        ensure_dir(layout.case_dir)
        ensure_dir(layout.edit_dir)
        ensure_dir(layout.artifacts_dir)
        self._write_config(layout)

        context = ExperimentContext(
            config=self.config,
            pipeline=pipeline,
            layout=layout,
            loaded_inputs=loaded_inputs,
        )

        preprocess_plugin = SharedImagePreprocessPlugin()
        preprocess_artifact = preprocess_plugin.run(context, self.config.preprocess)
        result = ExperimentResult(preprocess=preprocess_artifact)

        ss_artifact = None
        if self.config.run_mode in {"ss", "full"}:
            if self.config.ss is None:
                raise RuntimeError("Missing ss config")
            ensure_dir(context.ss_dir)
            save_preprocessed_inputs(context.ss_dir, preprocess_artifact.prepared_inputs, include_mask=True)
            ss_plugin = SS_PLUGIN_REGISTRY[self.config.ss.method]()
            try:
                ss_artifact = ss_plugin.run(context, preprocess_artifact, self.config.ss)
                ss_artifact.artifact_paths = ss_plugin.save(ss_artifact, context.ss_dir, self.config.ss)
                ss_artifact.primary_coords_path = self._primary_coords_path(ss_artifact)
                result.ss = ss_artifact
            finally:
                ss_plugin.cleanup()

        if self.config.run_mode in {"slat", "full"}:
            if self.config.slat is None:
                raise RuntimeError("Missing slat config")
            ensure_dir(context.slat_dir)
            save_preprocessed_inputs(context.slat_dir, preprocess_artifact.prepared_inputs, include_mask=True)
            slat_plugin = SLAT_PLUGIN_REGISTRY[self.config.slat.method]()
            try:
                slat_artifact = slat_plugin.run(context, preprocess_artifact, self.config.slat, ss_artifact)
                slat_artifact.artifact_paths = slat_plugin.save(slat_artifact, context.slat_dir, self.config.slat)
                if self.config.runtime.save_source_outputs and slat_artifact.source_outputs is not None:
                    source_dir = ensure_dir(layout.source_original_dir)
                    save_outputs(
                        outputs=slat_artifact.source_outputs,
                        out_dir=source_dir,
                        skip_render=self.config.runtime.skip_render,
                        skip_glb=self.config.runtime.skip_glb,
                        skip_ply=self.config.runtime.skip_ply,
                    )
                result.slat = slat_artifact
            finally:
                slat_plugin.cleanup()

        return result
