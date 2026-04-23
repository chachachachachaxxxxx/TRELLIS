#!/usr/bin/env python3
from __future__ import annotations

import argparse
import html
import json
from pathlib import Path

from PIL import Image, ImageDraw, ImageFont, ImageOps


DEFAULT_DATASET_ROOT = Path(
    "/cache/wangxinxing/data/trellis_edit_benchmark/edit3d_mv_pseudosource_micro10"
)
DEFAULT_OUT_DIR = Path(
    "/cache/wangxinxing/data/trellis_edit_benchmark/daily/mv_random3_analysis_20260420"
)
DEFAULT_VIEW_KEY = "cardinal_neg90"
DEFAULT_METHODS = [
    (
        "Direct Front",
        "/cache/wangxinxing/data/trellis_edit_benchmark/pred_mv/trellis_mv_target_direct_frontonly_first10_seed1",
    ),
    (
        "Stochastic F1B1",
        "/cache/wangxinxing/data/trellis_edit_benchmark/pred_mv/trellis_mv_target_stochastic_front1_back1_first10_seed1",
    ),
    (
        "CW F2B1",
        "/cache/wangxinxing/data/trellis_edit_benchmark/pred_mv/trellis_mv_target_canonical_weighted_front2_back1_first10_seed1_axisfix_redistribute",
    ),
    (
        "CW F1B1",
        "/cache/wangxinxing/data/trellis_edit_benchmark/pred_mv/trellis_mv_target_canonical_weighted_front1_back1_first10_seed1_axisfix_redistribute",
    ),
    (
        "ViewAligned F1B1",
        "/cache/wangxinxing/data/trellis_edit_benchmark/pred_mv/trellis_mv_target_view_aligned_front1_back1_first10_seed1_axisfix",
    ),
    (
        "Ablation1 Front",
        "/cache/wangxinxing/data/trellis_edit_benchmark/pred_mv/ablation1_08_ss_inversion_kv_latent__slat_inversion__nano3d_replace_frontonly_first10_seed1",
    ),
]


def _load_font(size: int, *, bold: bool = False) -> ImageFont.FreeTypeFont | ImageFont.ImageFont:
    candidates = []
    if bold:
        candidates.extend(
            [
                "/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf",
                "/usr/share/fonts/truetype/liberation2/LiberationSans-Bold.ttf",
            ]
        )
    else:
        candidates.extend(
            [
                "/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf",
                "/usr/share/fonts/truetype/liberation2/LiberationSans-Regular.ttf",
            ]
        )
    for path in candidates:
        font_path = Path(path)
        if font_path.is_file():
            return ImageFont.truetype(str(font_path), size=size)
    return ImageFont.load_default()


def _prepare_image(path: Path, size: tuple[int, int]) -> Image.Image:
    image = Image.open(path).convert("RGBA")
    white = Image.new("RGBA", image.size, (255, 255, 255, 255))
    image = Image.alpha_composite(white, image).convert("RGB")
    fitted = ImageOps.contain(image, size, method=Image.Resampling.LANCZOS)
    canvas = Image.new("RGB", size, "white")
    offset = ((size[0] - fitted.width) // 2, (size[1] - fitted.height) // 2)
    canvas.paste(fitted, offset)
    return canvas


def _read_metrics(pred_root: Path) -> dict:
    metrics_path = pred_root / "mv_metrics.json"
    if not metrics_path.is_file():
        return {}
    return json.loads(metrics_path.read_text(encoding="utf-8")).get("cases", {})


def _parse_methods(raw_methods: list[str] | None) -> list[tuple[str, Path]]:
    if not raw_methods:
        return [(label, Path(path)) for label, path in DEFAULT_METHODS]
    parsed = []
    for item in raw_methods:
        if "::" not in item:
            raise ValueError(f"--method must look like 'Label::/abs/pred_root', got: {item!r}")
        label, path = item.split("::", 1)
        parsed.append((label.strip(), Path(path.strip())))
    return parsed


def _draw_text_block(draw: ImageDraw.ImageDraw, xy: tuple[int, int], width: int, lines: list[str], font, fill):
    x, y = xy
    for line in lines:
        draw.text((x, y), line, font=font, fill=fill)
        y += font.size + 4


def build_sheet(
    *,
    case_root: Path,
    case_payload: dict,
    case_id: str,
    methods: list[tuple[str, Path]],
    out_path: Path,
    view_key: str,
) -> None:
    view_payload = case_payload["input_views"][view_key]
    headers = [
        ("Source", case_root / view_payload["source_path"]),
        ("Edit", case_root / view_payload["target_path"]),
        ("Front", None),
        ("Right", None),
        ("Back", None),
        ("Left", None),
    ]
    render_ids = ["0000", "0001", "0002", "0003"]

    metrics_by_method = {label: _read_metrics(pred_root) for label, pred_root in methods}

    label_w = 280
    cell_w = 210
    cell_h = 210
    row_h = 250
    header_h = 76
    pad = 20
    total_w = pad * 2 + label_w + len(headers) * cell_w
    total_h = pad * 2 + header_h + len(methods) * row_h

    canvas = Image.new("RGB", (total_w, total_h), "#f8fafc")
    draw = ImageDraw.Draw(canvas)
    title_font = _load_font(22, bold=True)
    header_font = _load_font(20, bold=True)
    method_font = _load_font(18, bold=True)
    body_font = _load_font(15)

    draw.rounded_rectangle((0, 0, total_w - 1, total_h - 1), radius=18, fill="#f8fafc")

    start_x = pad + label_w
    for idx, (header, _) in enumerate(headers):
        x0 = start_x + idx * cell_w
        draw.text((x0 + 12, pad + 18), header, font=header_font, fill="#0f172a")

    for row_idx, (label, pred_root) in enumerate(methods):
        y0 = pad + header_h + row_idx * row_h
        card = (pad, y0, total_w - pad, y0 + row_h - 16)
        draw.rounded_rectangle(card, radius=16, fill="white", outline="#e2e8f0")

        case_metrics = metrics_by_method.get(label, {}).get(case_id, {})
        metric_lines = [
            label,
            f"LPIPS novel: {case_metrics.get('lpips_in_novel', 'n/a')}",
            f"DINO novel: {case_metrics.get('dino_if_novel', 'n/a')}",
            f"Chamfer: {case_metrics.get('chamfer_target', 'n/a')}",
        ]
        _draw_text_block(draw, (pad + 16, y0 + 16), label_w - 28, metric_lines[:1], method_font, "#0f172a")
        _draw_text_block(draw, (pad + 16, y0 + 48), label_w - 28, metric_lines[1:], body_font, "#334155")

        source_img = _prepare_image(headers[0][1], (cell_w - 20, cell_h - 20))
        edit_img = _prepare_image(headers[1][1], (cell_w - 20, cell_h - 20))
        images = [source_img, edit_img]
        for render_id in render_ids:
            render_path = pred_root / "cases" / case_id / "mv_eval" / "seen_views" / f"render_{render_id}.png"
            if render_path.is_file():
                images.append(_prepare_image(render_path, (cell_w - 20, cell_h - 20)))
            else:
                placeholder = Image.new("RGB", (cell_w - 20, cell_h - 20), "white")
                ph_draw = ImageDraw.Draw(placeholder)
                ph_draw.rectangle((0, 0, placeholder.width - 1, placeholder.height - 1), outline="#cbd5e1")
                ph_draw.text((16, 16), "missing", font=body_font, fill="#64748b")
                images.append(placeholder)

        for col_idx, image in enumerate(images):
            x0 = start_x + col_idx * cell_w
            frame = (x0, y0 + 14, x0 + cell_w - 10, y0 + 14 + cell_h)
            draw.rounded_rectangle(frame, radius=14, fill="white", outline="#cbd5e1")
            canvas.paste(image, (x0 + 10, y0 + 24))

    canvas.save(out_path)


def build_index(out_dir: Path, case_items: list[tuple[str, dict]]) -> None:
    sections = []
    for case_id, case_payload in case_items:
        sections.append(
            "<section class='case'>"
            f"<h2>{html.escape(case_id)}</h2>"
            f"<p><strong>Prompt:</strong> {html.escape(str(case_payload.get('prompt_text') or ''))}</p>"
            f"<p><strong>Edit:</strong> {html.escape(str(case_payload.get('edit_instruction') or ''))}</p>"
            f"<img src='{html.escape(case_id)}.png' alt='{html.escape(case_id)}'>"
            "</section>"
        )
    html_text = f"""<!DOCTYPE html>
<html lang="zh">
<head>
<meta charset="UTF-8">
<meta name="viewport" content="width=device-width, initial-scale=1.0">
<title>MV Random 3 Analysis</title>
<style>
body {{ font-family: "Segoe UI", sans-serif; margin: 0; background: #f8fafc; color: #0f172a; }}
.page {{ width: min(1600px, calc(100vw - 32px)); margin: 24px auto 48px; }}
.hero, .case {{ background: white; border: 1px solid #e2e8f0; border-radius: 16px; padding: 20px; box-shadow: 0 4px 18px rgba(15,23,42,0.04); }}
.case + .case {{ margin-top: 20px; }}
.hero {{ margin-bottom: 20px; }}
img {{ width: 100%; border-radius: 12px; border: 1px solid #cbd5e1; background: white; }}
p {{ line-height: 1.5; }}
</style>
</head>
<body>
<div class='page'>
<section class='hero'>
<h1>MV Random 3 Analysis</h1>
<p>Columns are Source, Edit, and final Front/Right/Back/Left renders. Rows are methods, with per-case LPIPS novel, DINO novel, and Chamfer target shown in the method cell.</p>
</section>
{''.join(sections)}
</div>
</body>
</html>"""
    (out_dir / "index.html").write_text(html_text, encoding="utf-8")


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Build a white-background MV random-case comparison page.")
    parser.add_argument("--case-id", action="append", required=True, help="Case id, repeatable.")
    parser.add_argument("--dataset-root", type=Path, default=DEFAULT_DATASET_ROOT)
    parser.add_argument("--out-dir", type=Path, default=DEFAULT_OUT_DIR)
    parser.add_argument("--view-key", type=str, default=DEFAULT_VIEW_KEY)
    parser.add_argument(
        "--method",
        action="append",
        help="Method spec as 'Label::/abs/pred_root'. Repeatable. Defaults to the built-in method set.",
    )
    return parser


def main() -> int:
    args = build_parser().parse_args()
    dataset_root = args.dataset_root.expanduser().resolve()
    out_dir = args.out_dir.expanduser().resolve()
    out_dir.mkdir(parents=True, exist_ok=True)
    methods = _parse_methods(args.method)

    case_items: list[tuple[str, dict]] = []
    for case_id in args.case_id:
        case_root = dataset_root / "cases" / case_id
        case_payload = json.loads((case_root / "case.json").read_text(encoding="utf-8"))
        build_sheet(
            case_root=case_root,
            case_payload=case_payload,
            case_id=case_id,
            methods=methods,
            out_path=out_dir / f"{case_id}.png",
            view_key=args.view_key,
        )
        case_items.append((case_id, case_payload))
    build_index(out_dir, case_items)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
