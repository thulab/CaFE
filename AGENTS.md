# Repository Guidelines

## Current Scope

The repository is currently focused on the paper's synthetic time-series
mechanism benchmark. The FastAPI/Vue MVP remains in the repository but is
legacy context unless a task explicitly targets it.

Paper v8 is the only active experiment pipeline. Do not extend old v2-v7
scripts when implementing new protocol behavior.

## Paper v8 Core

The formal flow is:

```text
real-data calibration
  -> deterministic synthetic generation
  -> mechanism validation
  -> model inference
  -> capability and stability analysis
```

Primary files:

- `scripts/paper_v8_pipeline_common.py`
- `scripts/paper_v8_features.py`
- `scripts/calibrate_paper_v8.py`
- `scripts/generate_paper_v8_samples.py`
- `scripts/validate_paper_v8_samples.py`
- `scripts/run_paper_v8_inference.py`
- `scripts/analyze_paper_v8.py`
- `scripts/paper_v8_structured_baselines.py`
- `backend/app/services/synthetic_v8_generation.py`
- `backend/app/services/synthetic_v8_feature_gate.py`
- `docs/superpowers/specs/2026-07-24-paper-v8-full-pipeline-review.md`

The current protocol uses a 168-point real calibration history, a 336-point
synthetic master history, a 48-point horizon, and 96/168/336 inference views.
The fixed-context main table uses 168; oracle-context selects among the three
views. Real-anchor forecasts are an auxiliary real-data table and never enter
the synthetic mechanism ranking.

## Experiment Rules

- Real data supplies empirical feature references and generator nuisance
  parameters. Family-level intensity uses the usable real/generator overlap
  only when it spans enough controllable dose; otherwise it records a
  generator-relative fallback. Synthetic futures in the main benchmark are
  deterministic.
- Compute calibration and realized features from history only.
- Preserve paired seeds, anchors, nuisance paths, and normalization statistics
  across intensity or counterfactual members.
- Record every real-feature mapping and protocol fallback explicitly; never
  hide a missing structural feature behind an unlabelled default.
- Keep structural positive controls separate from foundation-model rankings.
- Treat manifests as immutable protocol records. A schema or protocol change
  requires a new experiment directory and regenerated artifacts.
- `runtime/` is ignored but may contain expensive user experiments. Never
  delete or overwrite it broadly.

## Editing and Testing

Python uses 4-space indentation, type hints, and snake_case names. Keep changes
inside v8 files unless a shared helper must be corrected.

Paper v8 scripts use lightweight research modules and must not depend on the
database, API, or persistence service import graph. The repository has no
root Python project, so use the backend uv project only as the dependency
environment while keeping the working directory at the repository root:

```bash
uv run --project backend python scripts/calibrate_paper_v8.py ...
uv run --project backend python scripts/generate_paper_v8_samples.py ...
uv run --project backend python scripts/validate_paper_v8_samples.py ...
uv run --project backend python scripts/run_paper_v8_pipeline.py ...
```

If a Paper v8 script unexpectedly requires `sqlmodel`, a database session, or
an application store merely to import, treat that as a layering bug and
remove the dependency instead of changing the execution environment.

Backend pytest still uses the backend uv project. Run focused tests first, for
example:

```bash
cd backend
uv run pytest tests/unit/test_paper_v8_pipeline_script.py
uv run pytest tests/unit/test_synthetic_v8_generation.py
uv run pytest tests/unit/test_paper_v8_structured_baselines.py
```

Do not run the full backend suite, frontend tests, or start the web platform
unless the user explicitly asks. For generator changes, also run the relevant
`test_synthetic_formula_*.py` files and a small calibration/generation pilot.

## Long-Running Jobs

When starting a long-running or background experiment through a Codex tool
shell, use a uniquely named detached `tmux` session. Do not use `nohup ... &`:
the tool runtime may clean up that process tree when its parent shell exits,
even when `nohup` is present.

Redirect the command to an explicit log under `runtime/`, enable
`remain-on-exit` when useful, and verify all three after launch:

- `tmux list-panes` reports a live pane;
- the expected experiment process is running;
- the log and status manifest are advancing without errors.

## Git and Safety

Keep commits concise and task-scoped. Do not commit runtime datasets,
predictions, service logs, model files, secrets, `.venv/`, or `node_modules/`.
Preserve unrelated user changes in a dirty worktree. Avoid destructive git or
filesystem commands.

Multi-agent work is appropriate when write sets are disjoint. Assign ownership
by pipeline layer and report changed paths plus verification commands.
