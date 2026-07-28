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
