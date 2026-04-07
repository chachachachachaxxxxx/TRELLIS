#!/usr/bin/env python3
from __future__ import annotations

import argparse
import os
import queue
import sys
import threading
import traceback
from pathlib import Path
from typing import Any, Optional


def _peek_arg(flag: str, default: str = "") -> str:
    if flag not in sys.argv:
        return default
    idx = sys.argv.index(flag)
    if idx + 1 >= len(sys.argv):
        return default
    return sys.argv[idx + 1]


_attn_backend = _peek_arg("--attn-backend", os.environ.get("ATTN_BACKEND", ""))
if _attn_backend:
    os.environ["ATTN_BACKEND"] = _attn_backend

os.environ.setdefault("SPCONV_ALGO", "native")

import gradio as gr
from gradio_litmodel3d import LitModel3D

from uniedit_trace_core import (
    TraceConfig,
    artifact_choices,
    load_manifest,
    preview_gallery_items,
    resolve_artifact,
    run_uniedit_trace,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Stream and browse UniEdit intermediate traces in a Gradio web app.")
    parser.add_argument("--server-name", default="127.0.0.1", help="Bind address for the Gradio server.")
    parser.add_argument("--server-port", type=int, default=7861, help="Port for the Gradio server.")
    parser.add_argument("--share", action="store_true", help="Enable Gradio share link.")
    parser.add_argument(
        "--attn-backend",
        default=_attn_backend or "",
        help="Attention backend override applied before TRELLIS import.",
    )
    return parser.parse_args()


def format_status(log_lines: list[str]) -> str:
    recent = log_lines[-16:]
    if not recent:
        return "等待任务开始。"
    body = "\n".join(f"- {line}" for line in recent)
    return f"**Status**\n{body}"


def summarize_manifest(manifest: Optional[dict]) -> dict:
    if not manifest:
        return {"status": "empty"}
    return {
        "method_name": manifest.get("method_name"),
        "case_name": manifest.get("case_name"),
        "output_dir": manifest.get("output_dir"),
        "num_steps": len(manifest.get("steps", [])),
        "num_artifacts": len(manifest.get("artifacts", [])),
        "last_artifact": manifest.get("artifacts", [{}])[-1].get("label") if manifest.get("artifacts") else None,
        "steps": [
            f"{step['index']:02d}. {step['title']} ({len(step.get('artifacts', []))} artifacts)"
            for step in manifest.get("steps", [])
        ],
    }


def _image_like(path: Optional[str]) -> Optional[str]:
    if not path:
        return None
    if Path(path).suffix.lower() in {".png", ".jpg", ".jpeg", ".webp"}:
        return path
    return None


def _video_like(path: Optional[str]) -> Optional[str]:
    if not path:
        return None
    if Path(path).suffix.lower() in {".mp4", ".webm", ".mov"}:
        return path
    return None


def _model_like(path: Optional[str]) -> Optional[str]:
    if not path:
        return None
    if Path(path).suffix.lower() in {".glb", ".gltf", ".ply", ".obj"}:
        return path
    return None


def resolve_view(manifest: Optional[dict], artifact_id: Optional[str]) -> tuple[str, Any, Any, Any, Any, dict, list[tuple[str, str]], dict]:
    if not manifest:
        return (
            "暂无工件。",
            None,
            None,
            None,
            None,
            {},
            [],
            summarize_manifest(manifest),
        )

    artifact = resolve_artifact(manifest, artifact_id)
    gallery = preview_gallery_items(manifest)
    if artifact is None:
        return (
            "当前 trace 还没有产物。",
            None,
            None,
            None,
            None,
            {},
            gallery,
            summarize_manifest(manifest),
        )

    primary_path = artifact.get("absolute_path")
    preview_path = artifact.get("absolute_preview_path")
    video_path = artifact.get("absolute_video_path")
    download_path = artifact.get("absolute_download_path")
    title = (
        f"**{artifact['label']}**\n\n"
        f"- media: `{artifact['media_type']}`\n"
        f"- viewer: `{artifact['viewer_hint']}`\n"
        f"- file: `{primary_path}`"
    )

    image_value = _image_like(preview_path) or _image_like(primary_path)
    video_value = _video_like(video_path) or _video_like(primary_path)
    model_value = _model_like(primary_path) if artifact["media_type"].startswith("model/") else None

    return (
        title,
        image_value,
        video_value,
        model_value,
        download_path,
        artifact.get("metadata", {}),
        gallery,
        summarize_manifest(manifest),
    )


def build_ui_state(logs: list[str], manifest: Optional[dict], selected_artifact_id: Optional[str]):
    output_dir = manifest.get("output_dir", "") if manifest else ""
    choices = artifact_choices(manifest) if manifest else []
    if choices and selected_artifact_id is None:
        selected_artifact_id = choices[-1][1]
    title, image_value, video_value, model_value, download_value, metadata_value, gallery, summary = resolve_view(
        manifest, selected_artifact_id
    )
    return (
        format_status(logs),
        output_dir,
        manifest,
        gr.update(choices=choices, value=selected_artifact_id),
        title,
        image_value,
        video_value,
        model_value,
        download_value,
        metadata_value,
        gallery,
        summary,
    )


def on_artifact_change(artifact_id: str, manifest: Optional[dict], status_markdown: str):
    logs = []
    if status_markdown:
        logs = [line[2:] for line in status_markdown.splitlines() if line.startswith("- ")]
    return build_ui_state(logs, manifest, artifact_id)[4:]


def load_existing_trace(manifest_path: str):
    path = manifest_path.strip()
    if not path:
        raise gr.Error("请提供 trace_manifest.json 路径，或者它所在的输出目录。")
    manifest = load_manifest(path)
    return build_ui_state([f"Loaded existing trace from {path}."], manifest, None)


def run_trace_stream(
    model: str,
    source_model: str,
    render_dir: str,
    image_dir: str,
    source_image: str,
    edit_image: str,
    mask_image: str,
    mask_glb: str,
    case_name: str,
    output_dir: str,
    seed: int,
    ss_steps: float,
    slat_steps: float,
    ss_cfg: float,
    slat_cfg: float,
    ss_inverse_cfg: float,
    slat_inverse_cfg: float,
    ss_omega: float,
    slat_omega: float,
    cfg_interval_start: float,
    cfg_interval_end: float,
    preprocess: bool,
    mask_threshold: float,
    auto_mask_threshold: float,
    auto_mask_max_filter: float,
    trace_every: float,
    render_videos: bool,
    export_glb: bool,
) :
    config = TraceConfig(
        model=model.strip(),
        source_model=source_model.strip(),
        render_dir=render_dir.strip(),
        image_dir=image_dir.strip(),
        source_image=source_image.strip(),
        edit_image=edit_image.strip(),
        mask_image=mask_image.strip(),
        mask_glb=mask_glb.strip(),
        case_name=case_name.strip(),
        output_dir=output_dir.strip(),
        seed=int(seed),
        ss_steps=None if int(ss_steps) <= 0 else int(ss_steps),
        slat_steps=None if int(slat_steps) <= 0 else int(slat_steps),
        ss_cfg=None if ss_cfg < 0 else float(ss_cfg),
        slat_cfg=None if slat_cfg < 0 else float(slat_cfg),
        ss_inverse_cfg=None if ss_inverse_cfg < 0 else float(ss_inverse_cfg),
        slat_inverse_cfg=None if slat_inverse_cfg < 0 else float(slat_inverse_cfg),
        ss_omega=float(ss_omega),
        slat_omega=float(slat_omega),
        cfg_interval_start=float(cfg_interval_start),
        cfg_interval_end=float(cfg_interval_end),
        preprocess=bool(preprocess),
        mask_threshold=int(mask_threshold),
        auto_mask_threshold=int(auto_mask_threshold),
        auto_mask_max_filter=int(auto_mask_max_filter),
        trace_every=max(int(trace_every), 1),
        render_videos=bool(render_videos),
        export_glb=bool(export_glb),
        quiet=True,
        attn_backend=os.environ.get("ATTN_BACKEND", ""),
    )

    event_queue: queue.Queue = queue.Queue()
    result: dict[str, Any] = {"manifest": None, "error": None}

    def worker() -> None:
        try:
            manifest = run_uniedit_trace(config, on_event=lambda event: event_queue.put(event))
            result["manifest"] = manifest
            event_queue.put(
                {
                    "type": "done",
                    "message": "Trace finished successfully.",
                    "manifest": manifest,
                    "selected_artifact_id": None,
                }
            )
        except Exception as exc:
            result["error"] = exc
            event_queue.put(
                {
                    "type": "error",
                    "message": f"{exc}",
                    "traceback": traceback.format_exc(),
                    "manifest": result.get("manifest"),
                    "selected_artifact_id": None,
                }
            )

    thread = threading.Thread(target=worker, daemon=True)
    thread.start()

    logs = ["Trace job submitted."]
    manifest = None
    selected_artifact_id = None
    yield build_ui_state(logs, manifest, selected_artifact_id)

    while thread.is_alive() or not event_queue.empty():
        try:
            event = event_queue.get(timeout=0.5)
        except queue.Empty:
            continue

        if event.get("message"):
            logs.append(event["message"])
        if event.get("manifest"):
            manifest = event["manifest"]
        if event.get("selected_artifact_id"):
            selected_artifact_id = event["selected_artifact_id"]

        if event["type"] == "error":
            logs.append("Traceback:")
            logs.extend(line for line in event.get("traceback", "").splitlines() if line.strip())
            yield build_ui_state(logs, manifest, selected_artifact_id)
            return

        yield build_ui_state(logs, manifest, selected_artifact_id)

    if result.get("manifest") is not None:
        manifest = result["manifest"]
    yield build_ui_state(logs, manifest, selected_artifact_id)


CSS = """
.trace-shell {max-width: 1600px; margin: 0 auto;}
.trace-shell .gradio-container {max-width: 1600px;}
"""


with gr.Blocks(css=CSS, title="UniEdit Trace Viewer", theme=gr.themes.Soft()) as demo:
    manifest_state = gr.State(value=None)

    with gr.Column(elem_classes=["trace-shell"]):
        gr.Markdown(
            """
            # UniEdit Trace Viewer
            输入 TRELLIS UniEdit 所需资产后，页面会边跑边推送中间变量，并支持在浏览器里查看 2D 预览和 3D 工件。
            """
        )

        with gr.Row():
            with gr.Column(scale=1):
                model = gr.Textbox(label="Model", value="microsoft/TRELLIS-image-large")
                source_model = gr.Textbox(
                    label="Source Model Dir",
                    placeholder="包含 voxels.ply / features.npz 的目录，或目录中的任意文件",
                )
                render_dir = gr.Textbox(
                    label="Render Dir",
                    placeholder="可选。若填写则优先从这里读取 voxels.ply / features.npz",
                )
                image_dir = gr.Textbox(
                    label="Image Dir",
                    placeholder="可选。若填写则自动读取 2d_render.png / 2d_edit.png / 2d_mask.png",
                )
                source_image = gr.Textbox(label="Source Image", placeholder="可选。显式指定 source 图")
                edit_image = gr.Textbox(label="Edit Image", placeholder="可选。显式指定 edit 图")
                mask_image = gr.Textbox(label="Mask Image", placeholder="可选。2D mask 路径")
                mask_glb = gr.Textbox(label="Mask GLB", placeholder="可选。3D 局部编辑区域")
                case_name = gr.Textbox(label="Case Name", placeholder="可选。默认自动生成")
                output_dir = gr.Textbox(label="Output Dir", placeholder="可选。默认写到 outputs/image_uniedit_rf_inversion_trace/<case>/edit/")

                with gr.Accordion("Advanced", open=False):
                    seed = gr.Number(label="Seed", value=1, precision=0)
                    ss_steps = gr.Number(label="SS Steps", value=20, precision=0, info="填 0 表示沿用 pipeline 默认值")
                    slat_steps = gr.Number(label="SLat Steps", value=20, precision=0, info="填 0 表示沿用 pipeline 默认值")
                    ss_cfg = gr.Number(label="SS CFG", value=-1, info="填负数表示沿用默认值")
                    slat_cfg = gr.Number(label="SLat CFG", value=-1, info="填负数表示沿用默认值")
                    ss_inverse_cfg = gr.Number(label="SS Inverse CFG", value=-1, info="填负数表示沿用默认值")
                    slat_inverse_cfg = gr.Number(label="SLat Inverse CFG", value=-1, info="填负数表示沿用默认值")
                    ss_omega = gr.Slider(label="SS Omega", minimum=0.0, maximum=3.0, step=0.05, value=1.0)
                    slat_omega = gr.Slider(label="SLat Omega", minimum=0.0, maximum=3.0, step=0.05, value=1.0)
                    cfg_interval_start = gr.Slider(label="CFG Interval Start", minimum=0.0, maximum=1.0, step=0.01, value=0.5)
                    cfg_interval_end = gr.Slider(label="CFG Interval End", minimum=0.0, maximum=1.0, step=0.01, value=1.0)
                    mask_threshold = gr.Slider(label="Mask Threshold", minimum=0, maximum=255, step=1, value=127)
                    auto_mask_threshold = gr.Slider(label="Auto Mask Threshold", minimum=0, maximum=255, step=1, value=24)
                    auto_mask_max_filter = gr.Slider(label="Auto Mask Max Filter", minimum=1, maximum=31, step=2, value=7)
                    trace_every = gr.Slider(label="Trace Every N Steps", minimum=1, maximum=20, step=1, value=5)
                    preprocess = gr.Checkbox(label="Shared Preprocess", value=True)
                    render_videos = gr.Checkbox(label="Render MP4 Previews", value=False)
                    export_glb = gr.Checkbox(label="Export Final GLB", value=True)

                run_btn = gr.Button("Run Trace", variant="primary")

                with gr.Accordion("Load Existing Trace", open=False):
                    existing_trace = gr.Textbox(
                        label="Manifest Or Output Dir",
                        placeholder="填 trace_manifest.json 路径，或它所在的输出目录",
                    )
                    load_btn = gr.Button("Load Trace")

            with gr.Column(scale=2):
                status_markdown = gr.Markdown(value="等待任务开始。")
                output_dir_box = gr.Textbox(label="Current Output Dir", interactive=False)
                artifact_dropdown = gr.Dropdown(label="Artifacts", choices=[], value=None)
                trace_summary = gr.JSON(label="Trace Summary")

                with gr.Row():
                    preview_image = gr.Image(label="Preview", height=360, interactive=False)
                    preview_video = gr.Video(label="Video Preview", height=360, interactive=False)

                model_viewer = LitModel3D(label="3D Viewer", exposure=10.0, height=520)
                artifact_title = gr.Markdown(value="暂无工件。")
                artifact_metadata = gr.JSON(label="Artifact Metadata")
                artifact_download = gr.File(label="Artifact Download", interactive=False)
                gallery = gr.Gallery(label="Preview Gallery", height=240, object_fit="contain")

    run_inputs = [
        model,
        source_model,
        render_dir,
        image_dir,
        source_image,
        edit_image,
        mask_image,
        mask_glb,
        case_name,
        output_dir,
        seed,
        ss_steps,
        slat_steps,
        ss_cfg,
        slat_cfg,
        ss_inverse_cfg,
        slat_inverse_cfg,
        ss_omega,
        slat_omega,
        cfg_interval_start,
        cfg_interval_end,
        preprocess,
        mask_threshold,
        auto_mask_threshold,
        auto_mask_max_filter,
        trace_every,
        render_videos,
        export_glb,
    ]

    run_outputs = [
        status_markdown,
        output_dir_box,
        manifest_state,
        artifact_dropdown,
        artifact_title,
        preview_image,
        preview_video,
        model_viewer,
        artifact_download,
        artifact_metadata,
        gallery,
        trace_summary,
    ]

    run_btn.click(
        run_trace_stream,
        inputs=run_inputs,
        outputs=run_outputs,
        api_name="run_uniedit_trace",
        show_progress="minimal",
    )

    load_btn.click(
        load_existing_trace,
        inputs=[existing_trace],
        outputs=run_outputs,
        api_name="load_uniedit_trace",
        show_progress="minimal",
    )

    artifact_dropdown.change(
        on_artifact_change,
        inputs=[artifact_dropdown, manifest_state, status_markdown],
        outputs=[
            artifact_title,
            preview_image,
            preview_video,
            model_viewer,
            artifact_download,
            artifact_metadata,
            gallery,
            trace_summary,
        ],
        show_progress="hidden",
    )

demo.queue(default_concurrency_limit=1)


if __name__ == "__main__":
    args = parse_args()
    if args.attn_backend:
        os.environ["ATTN_BACKEND"] = args.attn_backend
    demo.launch(server_name=args.server_name, server_port=args.server_port, share=args.share)
