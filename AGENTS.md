# Repository Guidelines

## Project

CaFE is a Capability-Focused Extension of existing forecasting benchmarks.
The current adapter reads GIFT-Eval's official test instances and adds paired
capability treatments directly to their authentic histories and futures.

## Architecture

The active pipeline has four stages:

```text
official GIFT-Eval instances
  → generation
  → validation
  → inference
  → analysis
```

Generation preserves native target dimensions and official forecast origins.
Treatments cover the complete official input history, while inference applies
each model's maximum-context truncation afterward. Five capability levels use
ordered, non-overlapping parameter intervals and deterministic randomness from
the official instance id, capability, level, and augmentation seed.

## Stage contracts

`experiment.json` stores experiment identity. Stage contracts live under
`stage_contracts/`, record code and configuration, and hash upstream artifacts.
Completed experiments remain immutable; a changed protocol uses a new
experiment id.

## Development

Python uses type hints, 4-space indentation, and snake_case names. Focused
tests live under `tests/`.

```bash
uv sync --extra test
uv run pytest
```

Runtime experiments live under `runtime/`. Source datasets live under `data/`.
Both stay outside Git commits.

## Project decisions

User-confirmed project decisions are recorded in `项目决策.md` with their date
and rationale.
