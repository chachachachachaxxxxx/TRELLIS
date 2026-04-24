from __future__ import annotations

from .base import BenchmarkContext, BenchmarkTask, RunResult


class BenchmarkRunner:
    def run(self, task: BenchmarkTask, context: BenchmarkContext) -> RunResult:
        cases = list(task.discover_cases(context))
        prepared_cases = list(task.prepare_cases(cases, context))
        case_results = list(task.evaluate_cases(prepared_cases, context))
        summary_metrics = task.summarize_run(case_results, context)
        manifest = task.build_manifest(case_results, context)
        run_artifacts = task.collect_run_artifacts(case_results, context)
        return RunResult(
            task_name=task.name,
            requested_metrics=list(context.requested_metrics),
            summary_metrics=summary_metrics,
            case_results=case_results,
            manifest=manifest,
            run_artifacts=run_artifacts,
        )
