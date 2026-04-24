Vendored TRELLIS Core
=====================

`trellis/` is an in-place vendor in this repository.

Why it stays in place
---------------------

- The repo has a large existing import surface that expects `import trellis`.
- Development commands and docs already refer to the top-level `trellis/` path.
- `trellis/representations/mesh/flexicubes/` is mounted as a git submodule under
  the current path, so hard-moving the tree would create unnecessary churn.

Patch policy
------------

- Keep this tree upstream-like.
- Put local editing workflows, experiment orchestration, benchmark glue, and
  repo-specific utilities in `trellis_edit/` instead of here.
- Change files under `trellis/` only when the behavior truly belongs to the
  vendored core.
- Treat `trellis/representations/mesh/flexicubes/` as separately vendored code.

For new external repositories, prefer `vendors/<name>/` instead of extending
this in-place vendor boundary.
