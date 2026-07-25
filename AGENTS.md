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

- Real data calibrates empirical feature ranges and generator nuisance
  parameters. Synthetic futures in the main benchmark are deterministic.
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

Run focused tests first, for example:

```bash
cd backend
uv run pytest tests/unit/test_paper_v8_pipeline_script.py
uv run pytest tests/unit/test_synthetic_v8_generation.py
uv run pytest tests/unit/test_paper_v8_structured_baselines.py
```

Do not run the full backend suite, frontend tests, or start the web platform
unless the user explicitly asks. For generator changes, also run the relevant
`test_synthetic_formula_*.py` files and a small calibration/generation pilot.

## Git and Safety

Keep commits concise and task-scoped. Do not commit runtime datasets,
predictions, service logs, model files, secrets, `.venv/`, or `node_modules/`.
Preserve unrelated user changes in a dirty worktree. Avoid destructive git or
filesystem commands.

Multi-agent work is appropriate when write sets are disjoint. Assign ownership
by pipeline layer and report changed paths plus verification commands.
