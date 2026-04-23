from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any


def _time_key(t_value: float) -> str:
    return f"{float(t_value):.10f}"


@dataclass(frozen=True)
class StepContext:
    stage: str
    phase: str
    step_index: int
    t_curr: float
    t_next: float
    logical_t: float


@dataclass(frozen=True)
class TraceEvalState:
    actual_t: float
    logical_t: float
    eval_idx: int
    sample: Any


@dataclass(frozen=True)
class PredictorStepResult:
    x_pred: Any
    eval_states: tuple[TraceEvalState, ...] = ()


@dataclass(frozen=True)
class RefinementResult:
    x_next: Any
    eval_states: tuple[TraceEvalState, ...] = ()


@dataclass(frozen=True)
class SolverStepResult:
    x_next: Any
    predictor_states: tuple[TraceEvalState, ...] = ()
    refinement_states: tuple[TraceEvalState, ...] = ()


@dataclass(frozen=True)
class SourceTraceEntry:
    step_index: int
    logical_t: float
    sample: Any
    predictor_states: tuple[TraceEvalState, ...] = ()
    refinement_states: tuple[TraceEvalState, ...] = ()


@dataclass
class SourceTrace:
    entries: list[SourceTraceEntry] = field(default_factory=list)
    _by_logical_t: dict[str, SourceTraceEntry] = field(default_factory=dict, init=False, repr=False)

    def __post_init__(self) -> None:
        for entry in self.entries:
            self._by_logical_t[_time_key(entry.logical_t)] = entry

    def add_entry(
        self,
        *,
        step_index: int,
        logical_t: float,
        sample: Any,
        predictor_states: tuple[TraceEvalState, ...] = (),
        refinement_states: tuple[TraceEvalState, ...] = (),
    ) -> None:
        entry = SourceTraceEntry(
            step_index=step_index,
            logical_t=float(logical_t),
            sample=sample,
            predictor_states=tuple(predictor_states),
            refinement_states=tuple(refinement_states),
        )
        self.entries.append(entry)
        self._by_logical_t[_time_key(entry.logical_t)] = entry

    def get_entry(self, logical_t: float) -> SourceTraceEntry | None:
        return self._by_logical_t.get(_time_key(logical_t))

    def require_entry(self, logical_t: float) -> SourceTraceEntry:
        entry = self.get_entry(logical_t)
        if entry is None:
            raise KeyError(f"Missing source trace entry for logical_t={float(logical_t):.10f}")
        return entry

    def get_sample(self, logical_t: float) -> Any | None:
        entry = self.get_entry(logical_t)
        if entry is None:
            return None
        return entry.sample

    def __len__(self) -> int:
        return len(self.entries)
