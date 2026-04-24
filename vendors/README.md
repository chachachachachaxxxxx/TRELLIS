Vendor Layout
=============

This repository keeps third-party or upstream-derived code in two forms:

1. In-place vendor
   `trellis/` is treated as the vendored TRELLIS core.
   It stays at the repository root to keep the existing Python import path stable,
   preserve the current `python -m compileall trellis ...` workflow, and avoid
   breaking the `trellis/representations/mesh/flexicubes` submodule path.

2. Nested vendor
   New external repositories that do not need to keep a historical top-level path
   should live under `vendors/<name>/`.

Current vendor map
------------------

- `trellis/`
  In-place vendored TRELLIS core. Keep local editing methods, benchmark wrappers,
  and experiment-specific glue out of this tree unless the change is a deliberate
  core patch.

- `trellis/representations/mesh/flexicubes/`
  Vendored upstream submodule. Treat it as external code and avoid edits unless
  they are intentional.

Repository boundary
-------------------

- Put local editing and evaluation work in `trellis_edit/`.
- Put smoke examples in `trellis_inference/`.
- Prefer vendoring new external baselines or model repos under `vendors/`
  instead of mixing them into `trellis/`.

This layout is intentional groundwork for future integration of additional
upstream codebases such as `TRELLIS.2` without collapsing their boundaries
into the local editing framework.
