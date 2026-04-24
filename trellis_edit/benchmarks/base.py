from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Sequence


@dataclass(frozen=True)
class CaseIdentity:
    case_id: str
    dataset: str | None = None
    object_name: str | None = None
    prompt_id: int | None = None
    sample_name: str | None = None
    display_name: str | None = None
    path_tokens: tuple[str, ...] = ()

    def resolved_path_tokens(self) -> tuple[str, ...]:
        if self.path_tokens:
            return self.path_tokens

        tokens: list[str] = []
        if self.dataset:
            tokens.append(self.dataset)
        if self.object_name:
            tokens.extend(part for part in self.object_name.split("/") if part)
        if self.prompt_id is not None:
            tokens.append(f"prompt_{self.prompt_id}")
        elif self.sample_name:
            tokens.append(self.sample_name)
        else:
            tokens.extend(part for part in self.case_id.strip("/").split("/") if part)
        return tuple(tokens)


@dataclass(frozen=True)
class CaseResult:
    identity: CaseIdentity
    status: str
    metrics: dict[str, Any]
    artifacts: dict[str, Any]
    prompt_text: str | None = None
    task_meta: dict[str, Any] | None = None


@dataclass(frozen=True)
class RunResult:
    task_name: str
    requested_metrics: list[str]
    summary_metrics: dict[str, Any]
    case_results: list[CaseResult]
    manifest: dict[str, Any] = field(default_factory=dict)
    run_artifacts: dict[str, Any] = field(default_factory=dict)


@dataclass(frozen=True)
class BenchmarkContext:
    pred_root: Path
    requested_metrics: tuple[str, ...] = ()
    device: str = "cuda:0"
    output_root: Path | None = None
    task_config: dict[str, Any] = field(default_factory=dict)
    runtime: dict[str, Any] = field(default_factory=dict)


class BenchmarkTask(ABC):
    name = "benchmark"

    @abstractmethod
    def discover_cases(self, context: BenchmarkContext) -> Sequence[Any]:
        raise NotImplementedError

    def prepare_cases(
        self,
        cases: Sequence[Any],
        context: BenchmarkContext,
    ) -> Sequence[Any]:
        return list(cases)

    @abstractmethod
    def evaluate_cases(
        self,
        prepared_cases: Sequence[Any],
        context: BenchmarkContext,
    ) -> Sequence[CaseResult]:
        raise NotImplementedError

    def summarize_run(
        self,
        case_results: Sequence[CaseResult],
        context: BenchmarkContext,
    ) -> dict[str, Any]:
        return {}

    def build_manifest(
        self,
        case_results: Sequence[CaseResult],
        context: BenchmarkContext,
    ) -> dict[str, Any]:
        return {
            "task_name": self.name,
            "pred_root": str(context.pred_root),
            "requested_metrics": list(context.requested_metrics),
        }

    def collect_run_artifacts(
        self,
        case_results: Sequence[CaseResult],
        context: BenchmarkContext,
    ) -> dict[str, Any]:
        return {}
