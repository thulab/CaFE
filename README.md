# CaFE

CaFE (Capability-Focused Extension) extends existing time-series benchmarks
with controlled capability treatments. Its active implementation starts from
GIFT-Eval's official test instances, modifies their authentic paths, and asks
forecasting models to predict the corresponding treated futures.

## Pipeline

```text
GIFT-Eval official native test instances
  → capability generation
  → pair and provenance validation
  → model inference
  → baseline/treatment accuracy, capability-effect, and input-attribution analysis
```

There is no calibration stage and no standalone synthetic-curve generator.
The adapter reproduces GIFT-Eval's prediction length, test-window count,
split offset, and rolling origin distance. Native multivariate targets remain
one benchmark sample. A univariate-only model is adapted during inference and
its channel forecasts are reassembled.

Every supported instance×capability cell produces five treatments. Parameter
draws come from ordered, non-overlapping level intervals and are reproducible
from `official_instance_id`, capability, level, and `augmentation_seed`.
Changing the augmentation seed creates another treatment batch over the same
official samples.

The source Arrow files remain the only stored copy of authentic paths.
Generation writes compact replay contracts to ZSTD Parquet; it does not copy
full targets or calendar covariates. Inference rebuilds bounded batches in
memory, sends MessagePack bulk requests across compatible endpoints/GPUs, and
writes source-sharded float32 Parquet forecasts. No model-specific task JSONL
is materialized. Analysis scans one prediction shard at a time.

Treatments modify the complete retained official history. Model-specific
context truncation happens afterward. Amplitude levels are calibrated by the
full-history macro normalized RMS. Qualification checks the distinct contexts
actually received by the seven evaluated models: every model-context macro
distance must be in `[0.10, 2.0]`, and every affected channel must remain at or
below `3.0`. These checks reject a treatment group but never rescale it.

## Capabilities

The GIFT-Eval adapter attempts these nine generatable mechanisms per official
instance:

- whole-history linear trend in the sample's own trend direction;
- independent secondary seasonality;
- constrained time-varying seasonal amplitude;
- regime change with level-controlled change location;
- nonlinear persistence;
- predictable intermittency with level-controlled event sparsity;
- native-panel common factor;
- directed predictive cross-series transfer;
- response to deterministic known-future calendar covariates.

Availability is instance-specific. Short or structurally unsuitable samples
remain in the official baseline table and record a capability-unavailable
reason. Hierarchical coherence is currently qualification-only and produces
no ranked treatments.

Analysis keeps three result families separate: baseline and treatment
MASE/MAE; paired effect NRMSE, correlation, and amplitude ratio; and a
common/cross input-ablation table. The ablation keeps the assessed target
history and treatment future fixed while temporally misaligning auxiliary
channels, so it measures whether intact panel inputs improve the forecast.

## Install and test

```bash
uv sync --extra test
uv run pytest
```

## Smoke preparation

This runs generation and validation without starting model services:

```bash
uv run cafe run \
  --experiment-id gift-v7-smoke \
  --dataset-id gift_ett1_h \
  --max-instances 2 \
  --augmentation-seed 2026081601 \
  --stop-after validation
```

`--max-instances` selects a non-formal source-order prefix. Omitting it uses
all official GIFT-Eval test instances.

Validation defaults to the research policy: it scans every treatment distance
gate in parallel and writes the acceptance report required by inference. Add
`--validation-mode publication` for full manifest/hash checks and exact replay
of every contract. `--validation-workers` controls the per-dataset process
pool. When it is greater than one, datasets run sequentially to avoid nested
process pools; setting it to one lets `--validation-dataset-workers` control
concurrent lightweight dataset scans.

## Formal run

```bash
uv run cafe run \
  --experiment-id gift-v7-formal \
  --dataset-ids gift_electricity_h gift_ett1_h gift_jena_weather_h \
  --augmentation-seed 2026081601 \
  --models Timer-4.0 Chronos-2 timesfm2.5 tirex2 moirai2 Timer-3.5 toto2.0 \
  --endpoints http://100.102.176.45:10810 \
  --generation-workers 8 \
  --preprocess-workers 8 \
  --disk-budget-gb 40
```

Artifacts use this layout:

```text
<experiment>/
├── experiment.json
├── stage_contracts/
└── <dataset_id>/
    ├── 01_generation/
    ├── 02_validation/
    ├── 03_inference/
    └── 04_analysis/
```

See [docs/protocol.md](docs/protocol.md) for the frozen scientific protocol
and [docs/real_anchored_ten_capability_design.md](docs/real_anchored_ten_capability_design.md)
for mechanism formulas. Four-card Timer Service bulk and concurrency settings
are recorded in
[docs/inference_throughput_4x_rtx5090.md](docs/inference_throughput_4x_rtx5090.md).

## History

CaFE was extracted from TSBenchmark at commit `21b8452`. The
`monorepo-cutover-2026-07-28` tag and ancestor history remain available.
