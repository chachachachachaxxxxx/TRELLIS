# Repository Guidelines

## Project Structure & Module Organization
`trellis/` contains the core pipelines, models, renderers, trainers, and utilities for image-to-3D and text-to-3D. Put no-training editing work in `editing/`, especially `editing/methods/`, `editing/preprocess/`, `editing/hooks/`, and `editing/inversion/`; use [`run_edit_experiment.py`](/home/wangxinxing/code/TRELLIS/run_edit_experiment.py) as the unified entrypoint. Keep smoke examples in `trellis_inference/`, tracing tools in `vis/`, dataset scripts in `dataset_toolkits/`, configs in `configs/`, and assets in `assets/`. Treat `trellis/representations/mesh/flexicubes/` as vendored code: do not edit it unless the change is intentional.

## Build, Test, and Development Commands
Work in the expected Conda env: `conda activate hammer`.

- `python run_edit_experiment.py --list-methods`: list registered editing methods.
- `python test_preprocess_validation.py`: quick validation for preprocessing and path resolution.
- `python -m compileall trellis editing run_edit_experiment.py`: syntax-check Python changes.
- `python trellis_inference/example.py` or `python trellis_inference/example_text.py`: core pipeline smoke tests.
- `python generate_source_assets.py`: prepare source assets required by editing workflows.
- `bash test_all_methods.sh`: GPU-heavy end-to-end smoke test for migrated editing methods; update the `GPU` variable first.

## Coding Style & Naming Conventions
Use 4-space indentation, PEP 8 spacing, and local import ordering. Match the surrounding file style and avoid unrelated reformatting. Prefer `snake_case` for functions, files, and CLI flags; use `PascalCase` for classes. Add type hints to new Python APIs when practical. Set `ATTN_BACKEND`, `SPARSE_ATTN_BACKEND`, and `SPCONV_ALGO` before importing `trellis` modules.

## Testing Guidelines
This repo relies on smoke tests more than a full `pytest` suite. Add validators as `test_<area>.py`, and keep GPU regressions reproducible through example scripts or shell runners. For editing changes, verify the narrowest affected flow first, then record the command, backend, seed, and output path.

## Architecture Wiki Maintenance
Use `wiki/` to preserve durable code understanding, not transient notes. Read existing pages first, update them instead of creating duplicates, and keep entries short, practical, and grounded in code. Capture architecture overviews, module summaries, flow traces, recurring patterns, durable Q&A, and dated change notes when the finding took effort or will matter again. Cross-link related pages, append concise updates to `wiki/log.md`, and track gaps or conflicts under `wiki/maintenance/`. Skip wiki churn for tiny isolated edits.

## Commit & Pull Request Guidelines
Recent history uses conventional prefixes such as `feat:`, `fix:`, `perf:`, `docs:`, and `chore:`; keep subjects short and imperative. PRs should state the impacted pipeline or method, list validation commands, note GPU/backend assumptions, and include screenshots or output paths for rendering or visualization changes. Never commit `outputs/`, temporary artifacts, model weights, or large exported `.glb`, `.ply`, or video files.
