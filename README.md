# CaFE

CaFE (Capability-Focused Extension) is a benchmark-extension suite for
time-series forecasting capability analysis. It keeps three scientifically
separate tracks:

- real-data forecast accuracy;
- real-path-anchored capability counterfactuals;
- deterministic synthetic mechanism tests.

The canonical flow is:

```text
real-data calibration and authentic background windows
  → real-anchored counterfactuals + deterministic synthetic generation
  → mechanism validation
  → model inference
  → track-separated capability and stability analysis
```

The synthetic track uses a 168-point real calibration history, a 336-point
master history, a 48-point horizon, and 96/168/336 inference views. The
real-anchored track fits decomposition contracts on L504 history, exposes only
the trailing L336 to the model, retains an observed real H48, and ranks only
the fixed L168 view. Pair members share the unmodified real L336
normalization/MASE reference. The two mechanism tracks never share scores or
rankings.

The v5 real-anchored implementation supports nine potentially rankable mechanisms:
local nonlinear trend, independent multi-seasonality, constrained carrier AM,
observed regime level shifts, nonlinear persistence, predictable intermittency,
common factor, directed cross-series predictive transfer, and known-future
covariate response. Availability remains dataset-specific. Formal panel tasks
require native synchronized `D>=3` records; two-channel panels are sensitivity
only and, when at least two distinct donor backgrounds exist, are emitted as a
separately analyzed, never-ranked auxiliary track.
The nonlinear history-residual replay is likewise emitted only as an auxiliary
sensitivity track; zero-future-innovation remains the formal estimand.
Hierarchical coherence is qualification-only until a raw-support/
nonnegativity policy is frozen. Common/cross input ablations are mandatory
attribution audits but never receive weight in the main score.

Qualification thresholds are protocol-declared and hash-frozen against a
source-time-disjoint reference bank. The same bank freezes the dose targets,
solver, caps, and coverage policy. Each evaluation contract resolves its own
physical multiplier grid from history-only response, while the reported
strength coordinate remains `lambda = 0.2, 0.4, 0.6, 0.8, 1.0`. Every
treatment must be at least 0.10 frozen-scale RMS away from its authentic source
on trailing L168 and remain within the local-augmentation budget; the exact
baseline is exempt and adjacent-level distance is diagnostic only. Final
evaluation targets cannot tune this policy, and reference windows are never
emitted as inference tasks.

## Install

```bash
uv sync --extra test
```

Plotting tools additionally require:

```bash
uv sync --extra plots
```

## Run

CaFE exposes one command with stage subcommands:

```bash
uv run cafe calibrate ...
uv run cafe generate ...
uv run cafe validate ...
uv run cafe infer ...
uv run cafe analyze ...
uv run cafe run ...
```

A small preparation run looks like:

```bash
uv run cafe run \
  --experiment-id smoke \
  --dataset-id gift_electricity_h \
  --seed-count 1 \
  --max-anchors 12 \
  --calibration-seeds 2 \
  --max-calibration-seeds 2 \
  --capabilities trend \
  --stop-after validation
```

Runtime artifacts are written under `runtime/experiments/` and are ignored by
Git.

GIFT-Eval assets downloaded under `data/gift-eval/` are used by default. The
offline real-anchored qualification does not start model services or create
pipeline artifacts:

```bash
uv run python tools/qualification/real_anchored.py \
  --dataset-id gift_electricity_h \
  --dataset-id gift_jena_weather_h \
  --maximum-backgrounds 32
```

GIFT-Eval's official short-term test tail is removed before any CaFE window is
sampled. M4 Hourly uses its official single H48 tail rule.
Consequently these are GIFT-derived CaFE extension tasks, not a replacement
for the official GIFT-Eval test-set leaderboard.

### FEV-Bench pilot data

The first FEV-Bench integration pins nine representative configurations and
downloads their Parquet assets into the ignored `data/fev-bench/` directory:

```bash
uv run python tools/data/download_fev_bench.py
```

Registered dataset IDs start with `fev_`. The adapter preserves native
multivariate targets and exposes only task-declared known dynamic columns as
future covariates. Past-only and static columns remain frozen in provenance but
are not passed to the CaFE v1 inference contract. Calibration still uses the
CaFE 168-point history and 48-point horizon, so these are FEV-derived CaFE
calibrations rather than official FEV-Bench evaluations.

For example:

```bash
uv run cafe run \
  --experiment-id fev-pilot-smoke \
  --dataset-id fev_solar_with_weather_1h \
  --source-root data/fev-bench \
  --seed-count 1 \
  --max-anchors 12 \
  --calibration-seeds 2 \
  --max-calibration-seeds 2 \
  --capabilities covariate_response \
  --stop-after generation
```

Before expanding beyond the pilot, the metadata-only Phase 1 audit can inspect
all 100 official task views without downloading the full Parquet corpus or
starting calibration:

```bash
uv run python tools/data/audit_fev_bench.py \
  --audit-id fev-full-phase1-YYYYMMDD
```

The immutable report under `runtime/fev_bench_audits/<audit-id>/` contains the
96-config inventory, task-by-capability candidate matrix, pinned download
manifest, duplicate-source flags, and workload estimates. A candidate cell is
not a calibration result: actual sequence lengths, missingness, categorical
levels, finite-window support, and structural feature support remain Phase 2
checks.

Phase 2 downloads and checksum-verifies the frozen 96-file manifest, then
reuses CaFE's exact anchor and real-feature extraction path to resolve those
checks:

```bash
uv run python tools/data/qualify_fev_bench.py \
  --qualification-id fev-full-phase2-YYYYMMDD
```

The immutable report under `runtime/fev_bench_qualifications/` records actual
length and missingness statistics, discovered categorical levels, usable
anchor counts, and the data-qualified task-by-capability matrix. This stage
does not run response-curve calibration or synthetic generation; an eligible
cell can still fail the later calibration reachability gates.

## Stage provenance

`experiment.json` fixes only the experiment identity and directory layout. It
does not freeze code or options for stages that have not run.

Each stage gets an immutable contract when that stage starts:

```text
stage_contracts/
├── calibration.json
├── generation.json
├── validation.json
├── inference.json
└── analysis.json
```

A contract records that stage's config and Git provenance and hashes the
upstream stage contract. This permits a later Git revision to run inference or
analysis against already frozen generation artifacts without rewriting their
history. Re-running an existing stage with different code or config is
rejected; use a new experiment id for that branch of the experiment.

See [docs/protocol.md](docs/protocol.md) for the scientific protocol and
[docs/migration.md](docs/migration.md) for history and path mapping.

## Repository layout

```text
src/cafe/
├── data/          # adapters, real records, imputation
├── features/      # history-only feature profile and primitives
├── calibration/   # response curves and conditioning
├── generation/    # deterministic mechanism families
├── validation/    # mechanism and realism gates
├── inference/     # model-service client and scheduling
├── analysis/      # metrics, baselines, reports
├── provenance.py  # stage contracts
├── protocol.py    # shared scientific protocol
└── pipeline.py    # orchestration
```

## History

CaFE was extracted from TSBenchmark at commit `21b8452`, tagged
`monorepo-cutover-2026-07-28`. The full ancestor history is retained. Files
were moved with Git renames, so generator history remains available with:

```bash
git log --follow -- src/cafe/generation/families.py
```
