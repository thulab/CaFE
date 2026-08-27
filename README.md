# CaFE

CaFE (Capability-Focused Extension) extends existing time-series benchmarks
with controlled capability treatments. Its benchmark-neutral native-window
contract currently supports GIFT-Eval and the official FEV v0.8.0 Mini-20
suite, modifies their authentic paths, and asks forecasting models to predict
the corresponding treated futures.

## Pipeline

```text
benchmark-official native forecast windows
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

Every supported instance×capability cell produces five treatments. Each
capability has one explicit level coordinate: amplitude for amplitude-controlled
mechanisms, period count for multi-seasonality, change location for regime
switching, and event spacing for intermittency. Each seed first selects one
capability structure per official instance. That structure is shared by all
five levels, while level-specific draws only control the declared difficulty
coordinate. Changing `augmentation_seed` therefore creates a structurally
different treatment batch over the same official samples.

The source Arrow files remain the only stored copy of authentic paths.
Generation writes compact replay contracts to ZSTD Parquet; it does not copy
full targets or native covariates. Inference rebuilds bounded batches in
memory, sends MessagePack bulk requests across compatible endpoints/GPUs, and
writes source-sharded float32 Parquet forecasts. No model-specific task JSONL
is materialized. Analysis scans one prediction shard at a time.

Multi-host inference is optional.  Supplying one `--distributed-worker`
mapping per endpoint moves source-shard replay, bulk construction, the local
service request, and prediction writing onto that endpoint's host.  Omitting
the mappings retains the ordinary single-orchestrator path.  For example:

```bash
--endpoints http://192.168.99.90:10810 http://192.168.99.92:10810 \
--distributed-worker http://192.168.99.90:10810=192.168.99.90 \
--distributed-worker http://192.168.99.92:10810=local \
--distributed-repo-root /data/xmy/CaFE
```

Generation keeps the requested source-shard size as a cap and plans roughly
three shards for smaller datasets.  This layout is topology-independent: one
host can consume every shard, while two or three near-endpoint workers avoid a
single-shard tail without fragmenting large datasets into thousands of files.

Treatments modify the complete retained official history. Model-specific
context truncation happens afterward. Amplitude-controlled levels are calibrated
by the full-history macro normalized RMS. Multi-seasonality instead keeps one
shared full-history RMS across all five levels so that only the number of periods
changes. Qualification checks the distinct contexts actually received by the
seven evaluated models: every model-context macro distance must be in
`[0.10, 2.0]`, and every affected channel must remain at or below `3.0`. These
checks reject a treatment group but never rescale it.

## Capabilities

By default, the GIFT-Eval adapter attempts these eight generatable mechanisms
per official instance:

- sampled linear, delayed, curved, or recurring piecewise-linear trend in the
  sample's own stable trend direction;
- controlled multi-period extrapolation: levels contain 2–6 independent
  periods at fixed total RMS, sampling a stable real anchor among the top three
  history spectrum candidates or a protocol anchor, with sampled sinusoidal or
  two-harmonic waveforms;
- sampled carrier amplitude modulation or periodic phase drift;
- sampled step, ramp, or sigmoid regime change with level-controlled location;
- sampled pulse shape, sign, channel mask, and bounded recurring jitter with
  level-controlled event sparsity;
- sampled native-panel PC loading mixture and latent carrier;
- sampled top-pool directed predictive cross-series transfer;
- sampled causal impulse-response kernel, target, sign, timing, and legal
  benchmark-native covariate, preserving each source field's past-only or
  known-future visibility.

Availability is instance-specific. Short or structurally unsuitable samples
remain in the official baseline table and record a capability-unavailable
reason. Hierarchical coherence is currently qualification-only and produces
no ranked treatments. State-dependent persistence remains implemented for
explicit research runs, but is excluded from the default capability set while
its treatment and scoring design are reconsidered.

Analysis keeps three result families separate: baseline and treatment
MASE/MAE; MASE-standardized pooled effect NRMSE, correlation, coverage, and
amplitude ratio; and a
common/cross/covariate input-ablation table. The ablation keeps the assessed
target history and treatment future fixed while temporally misaligning only
the relevant auxiliary treatment signal, so it measures whether intact inputs
improve the forecast.

The analysis stage also writes a suite-level task-equal summary. A GIFT
dataset and an FEV task each contribute one value, and model comparisons use a
paired nonparametric bootstrap over their common tasks. This prevents large
datasets or high-window-count tasks from dominating the benchmark result.

## Install and test

```bash
uv sync --extra test
uv run pytest
```

Install FEV support and freeze the exact Mini-20 suite definition with:

```bash
uv sync --extra test --extra fev
uv run python tools/snapshot_fev_mini.py
```

The snapshot pins FEV `v0.8.0`, upstream commit
`f1afffbf97bc51a4a233080d331633c6f7ab32f6`, the suite SHA-256, and the
`autogluon/fev_datasets` commit
`f71c0fff4cf81283a2c43e7f3a73aa4f9826aef8`. The 20 required Parquet files
occupy about 19 MiB and form the single authentic-data copy; generation
manifests freeze both file hashes and each selected task's dataset fingerprint.

## Smoke preparation

This runs generation and validation without starting model services:

```bash
uv run cafe run \
  --experiment-id gift-v13-smoke \
  --dataset-id gift_ett1_h \
  --max-instances 2 \
  --augmentation-seed 2026081601 \
  --stop-after validation
```

`--max-instances` selects a non-formal source-order prefix. Omitting it uses
all official GIFT-Eval test instances.

An FEV research smoke run uses the same four stages and queries the configured
service for the selected models' actual context/output limits:

```bash
uv run python -m cafe.fev_pipeline \
  --experiment-id fev-mini-smoke \
  --task-id fev__ETT_1H \
  --models Timer-4.0 \
  --endpoints http://100.102.176.45:10810 \
  --capabilities trend \
  --max-instances 1
```

Omit `--task-id` and `--max-instances` for all 20 official tasks. FEV smoke
runs intentionally use research validation; GIFT publication replay remains a
separate stricter validation mode.

Validation defaults to the research policy: it scans every treatment distance
and mechanism-scoring gate in parallel and writes the acceptance report
required by inference. Add
`--validation-mode publication` for full manifest/hash checks and exact replay
of every contract. `--validation-workers` controls the per-dataset process
pool. When it is greater than one, datasets run sequentially to avoid nested
process pools; setting it to one lets `--validation-dataset-workers` control
concurrent lightweight dataset scans.

## Formal run

```bash
uv run cafe run \
  --experiment-id gift-v13-formal \
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
├── 04_analysis_suite/
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
