# CaFE

CaFE (Capability-Focused Extension) is the standalone research repository for
the paper's deterministic synthetic time-series mechanism benchmark.

The canonical flow is:

```text
real-data calibration
  → deterministic synthetic generation
  → mechanism validation
  → model inference
  → capability and stability analysis
```

The benchmark uses a 168-point real calibration history, a 336-point synthetic
master history, a 48-point horizon, and 96/168/336 inference views. The fixed
main table uses context 168; oracle context selects among the three views.

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
