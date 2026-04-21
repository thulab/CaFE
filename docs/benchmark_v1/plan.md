# TSBenchmark v1 Integration Plan

This document records the `data-xmy` branch plan as integrated into the current
FastAPI + Flask system.

## Scope

- v1 is a synthetic zero-shot univariate forecasting benchmark.
- It has two tracks:
  - Anchor Track: samples near real time-series structural feature prototypes.
  - Diagnostic Track: controlled samples for five capability families.
- The integrated artifact root is `runtime/generated/benchmark_v1/`.

## Algorithm Shape

- Real or bootstrap corpus -> structural features -> k-medoids anchor prototypes.
- Five diagnostic generators:
  - `trend`
  - `multi_seasonal`
  - `regime_switching`
  - `long_memory_nonlinear`
  - `intermittent_heteroskedastic`
- Proxy difficulty calibration uses:
  - `last_value`
  - `seasonal_naive`
  - `auto_theta`
  - `ridge_ar`
- Main metric is `MASE`; secondary metric is `sMAPE`; reports also include
  `relative_skill = 1 - MASE_model / MASE_baseline`.

## Known v1 Limitations

- Anchor Track still borrows the diagnostic generators.
- Gaussian copula anchor prior is not implemented in the integrated v1.
- Difficulty and realism checks are heuristic and are written into artifact
  metadata as limitations.

## Main Workflow

1. Build anchor stats with optional local GIFT-Eval / TFB roots.
2. Build benchmark parquet and metadata.
3. Run one model evaluation at a time.
4. Generate aggregate report artifacts.

