# Repository Guidelines

## Scope

CaFE is a standalone paper experiment repository. It contains real-data
calibration, deterministic synthetic generation, mechanism validation, model
inference, and capability/stability analysis. Do not introduce web platform,
database, API, or persistence-service dependencies.

## Scientific protocol

- Real calibration history: 168.
- Synthetic master history: 336.
- Forecast horizon: 48.
- Inference views: 96, 168, and 336.
- Fixed-context main table: 168.
- Synthetic futures are deterministic.
- Features and normalization statistics use history only.
- Paired members preserve seeds, anchors, nuisance paths, and normalization.
- Real-anchor forecasts are auxiliary and never enter synthetic rankings.

## Stage contracts

`experiment.json` is identity-only. Each stage freezes its own immutable
contract under `stage_contracts/` when it starts. Downstream stages may be
created by later Git revisions, but must hash and validate their upstream
contract. Never overwrite completed artifacts or redefine an existing stage
contract.

## Development

Use Python type hints, 4-space indentation, and snake_case names.

```bash
uv sync --extra test
uv run pytest tests/test_generation.py
uv run pytest tests/test_pipeline.py
```

Run focused tests first. Do not start model services or long-running
experiments unless explicitly requested. Long jobs must use a uniquely named
detached tmux session with logs under `runtime/`.

`runtime/` can contain expensive experiments. Never delete or overwrite it
broadly.

## Git

Keep commits task-scoped. Do not commit runtime artifacts, predictions, logs,
datasets, model weights, secrets, virtual environments, or caches. Preserve
the `monorepo-cutover-2026-07-28` tag and ancestor history.
