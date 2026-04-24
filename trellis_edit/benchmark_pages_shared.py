from __future__ import annotations

from pathlib import Path
from typing import Any

from .benchmark_pages_common import _load_json, _slug


def _load_case_payload(bundle_root: Path, case_record: dict[str, Any]) -> dict[str, Any]:
    detail_relpath = str(case_record.get("detail_path") or "").strip()
    case_payload = _load_json(bundle_root / detail_relpath) if detail_relpath else {}
    if case_payload:
        return case_payload
    return {
        "case_id": case_record.get("case_id"),
        "dataset": case_record.get("dataset"),
        "object_name": case_record.get("object_name"),
        "prompt_id": case_record.get("prompt_id"),
        "sample_name": case_record.get("sample_name"),
        "display_name": case_record.get("display_name"),
        "status": case_record.get("status"),
        "prompt_text": "",
        "metrics": case_record.get("metrics") or {},
        "artifacts": {},
    }


def _case_prompt_label(payload: dict[str, Any]) -> str:
    display_name = str(payload.get("display_name") or payload.get("sample_name") or "").strip()
    if display_name:
        return display_name
    prompt_id = payload.get("prompt_id")
    if prompt_id not in (None, ""):
        try:
            return f"prompt_{int(prompt_id)}"
        except (TypeError, ValueError):
            pass
    case_id = str(payload.get("case_id") or "").strip()
    if case_id:
        return case_id
    return "case"


def _case_record_page_path(bundle_root: Path, case_record: dict[str, Any]) -> Path:
    page_relpath = str(case_record.get("page_path") or "").strip()
    if page_relpath:
        return bundle_root / page_relpath
    dataset = str(case_record.get("dataset") or "").strip()
    object_name = str(case_record.get("object_name") or "").strip()
    prompt_id = int(case_record.get("prompt_id") or 0)
    return bundle_root / "pages" / dataset / object_name / f"prompt_{prompt_id}.html"


def _run_dataset_gallery_page_path(bundle_root: Path, dataset: str) -> Path:
    return bundle_root / "datasets" / f"{_slug(dataset)}.html"


__all__ = [
    "_load_case_payload",
    "_case_prompt_label",
    "_case_record_page_path",
    "_run_dataset_gallery_page_path",
]
