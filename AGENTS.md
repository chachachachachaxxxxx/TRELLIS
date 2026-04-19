# Repository Guidelines

## Project Structure & Module Organization
`trellis/` contains the core pipelines, models, renderers, trainers, and utilities for image-to-3D and text-to-3D. Put no-training local editing work in `trellis_edit/`, especially `trellis_edit/composable/`, `trellis_edit/preprocess/`, `trellis_edit/hooks/`, `trellis_edit/inversion/`, `trellis_edit/samplers/`, and `trellis_edit/common/`; use [`run_edit_experiment.py`](/home/wangxinxing/3dlocaledit/TRELLIS_EDIT/run_edit_experiment.py) as the unified entrypoint. Keep smoke examples in `trellis_inference/`, tracing tools in `vis/`, dataset scripts in `dataset_toolkits/`, configs in `edit_configs/`, and assets in `assets/`. Treat `trellis/representations/mesh/flexicubes/` as vendored code: do not edit it unless the change is intentional.

## Build, Test, and Development Commands
Work in the expected Conda env: `conda activate hammer`.

- `python run_edit_experiment.py --list-entrypoints`: list runnable composable entrypoints.
- `python run_edit_experiment.py --config edit_configs/template.config --dry-run`: validate structured composable config parsing.
- `python run_batch_edit_and_eval.py --config edit_configs/quick_test_batch.yaml --dry-run`: validate batch composable config expansion for the first case.
- `python -m compileall trellis trellis_edit run_edit_experiment.py run_batch_edit_and_eval.py`: syntax-check Python changes.
- `python trellis_inference/example.py` or `python trellis_inference/example_text.py`: core pipeline smoke tests.

## Coding Style & Naming Conventions
Use 4-space indentation, PEP 8 spacing, and local import ordering. Match the surrounding file style and avoid unrelated reformatting. Prefer `snake_case` for functions, files, and CLI flags; use `PascalCase` for classes. Add type hints to new Python APIs when practical. Set `ATTN_BACKEND`, `SPARSE_ATTN_BACKEND`, and `SPCONV_ALGO` before importing `trellis` modules.

## Visualization Preference
For voxel visualizations, prefer cubic voxel rendering in the style of [`vis/visualize_boundary_alignment.py`](/home/wangxinxing/3dlocaledit/TRELLIS_EDIT/vis/visualize_boundary_alignment.py), using voxel cubes rather than point-cloud scatter plots, unless the user explicitly asks for a point-based view. By default, voxel cubes should be opaque, tightly packed without gaps, and rendered with edge lines in a distinct color from the cube faces.

## Current Migration Rule
For the current stage-separation and P2P cleanup work, do not add compatibility layers for old method names, old parameter names, or old stage names. When a naming scheme changes, update the codebase to the new naming directly and delete the old path instead of mapping or aliasing it.
For the current editing-config refactor work, do not keep or read stale configs, stale generated configs, or stale result directories after the new path replaces them. Delete old configs and old results directly unless the user explicitly asks to preserve or inspect them.

## Testing Guidelines
This repo relies on smoke tests more than a full `pytest` suite. Add validators as `test_<area>.py`, and keep GPU regressions reproducible through example scripts or shell runners. For editing changes, verify the narrowest affected flow first, then record the command, backend, seed, and output path.

## Architecture Wiki Maintenance
Use `wiki/` to preserve durable code understanding, not transient notes. This repo no longer treats `docs/` as the source of truth for editing refactors; put durable guidance in `wiki/` instead. Read existing pages first, update them instead of creating duplicates, and keep entries short, practical, and grounded in code. Capture architecture overviews, module summaries, flow traces, recurring patterns, durable Q&A, and dated change notes when the finding took effort or will matter again. Cross-link related pages, append concise updates to `wiki/log.md`, and track gaps or conflicts under `wiki/maintenance/`. Skip wiki churn for tiny isolated edits.

## Commit & Pull Request Guidelines
Recent history uses conventional prefixes such as `feat:`, `fix:`, `perf:`, `docs:`, and `chore:`; keep subjects short and imperative. PRs should state the impacted pipeline or method, list validation commands, note GPU/backend assumptions, and include screenshots or output paths for rendering or visualization changes. Never commit `outputs/`, temporary artifacts, model weights, or large exported `.glb`, `.ply`, or video files.
