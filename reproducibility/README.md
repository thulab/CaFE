# Reproducing the CaFE experiments

This directory is the reviewer-facing entry point for the experiments reported
in the paper. It separates three activities that have very different resource
requirements:

1. **Artifact verification** checks the frozen tables and figures in seconds.
2. **Analysis reconstruction** recomputes tables and figures from compact local
   snapshots without running forecasting models.
3. **End-to-end reproduction** regenerates treatments, runs model inference,
   and performs analysis. This requires the benchmark sources, model weights,
   Linux GPU workers, and substantially more time.

The paper reports only publicly available models. The machine-readable configs
and the commands emitted here therefore omit unreleased systems.

## Quick verification

Use Python 3.11 or newer:

```bash
python reproducibility/reproduce.py verify
```

The command checks SHA-256 digests and row counts for the representative
paper tables and figures listed in `expected_results.json`.

## Reconstruct tables and figures

Install the plotting and test dependencies:

```bash
uv sync --extra plots --extra test
```

Then reconstruct one analysis block or all four:

```bash
python reproducibility/reproduce.py figures main
python reproducibility/reproduce.py figures ablation
python reproducibility/reproduce.py figures stability
python reproducibility/reproduce.py figures finetuning
```

The scripts consume frozen snapshots below `paper_results/work/`. They do not
contact the experiment server. The main analysis applies task-equal aggregation;
the fine-tuning analysis first aggregates within `(dataset, capability, level)`
strata. These two MASE values are intentionally not compared numerically.

Some immutable raw snapshots retain an internal, unreleased model for audit
provenance. The reconstruction scripts remove that model before computing any
reviewer-facing table, rank, or figure. The outputs covered by
`expected_results.json` are public-model-only.

## Experiment map

| Paper block | Frozen experiment(s) | Deterministic control |
|---|---|---|
| Main benchmark | GIFT-Eval short, medium, long; FEV | augmentation seed in `configs/paper_experiments.json` |
| Auxiliary-input ablation | GIFT-Eval short and FEV target-only summaries | same treatment contracts as the corresponding main run |
| Stability | fixed ten-task GIFT-Eval short panel | seeds `2026082701`–`2026082710` |
| Fine-tuning | two GIFT-Eval short treatment batches | full selection and training configs in `chronos2_finetuning/config.json` |

The exact experiment identifiers, upstream revisions, capabilities, public
model list, and FEV snapshot revisions are stored in
`configs/paper_experiments.json`.

## End-to-end CaFE runs

The GIFT-Eval and FEV pipelines use the same four immutable stages:

```text
generation -> validation -> inference -> analysis
```

Prepare the repository and source data as described in the root `README.md`.
Set the following paths for the local or shared worker installation:

```bash
export GIFT_EVAL_DIR=/path/to/gift-eval
export FEV_DATA_ROOT=/path/to/fev-mini-v0.8.0
export CAFE_MODEL_ROOT=/path/to/runtime/models
export CAFE_MODEL_CODE_ROOT=/path/to/runtime/model_runtime
export CAFE_OUTPUT_ROOT=/path/to/cafe-experiments
export CUDA_DEVICES=0,1,2,3
```

Print the frozen command for any main experiment:

```bash
python reproducibility/reproduce.py commands gift_short
python reproducibility/reproduce.py commands gift_medium
python reproducibility/reproduce.py commands gift_long
python reproducibility/reproduce.py commands fev
```

Print all ten stability runs with the fixed task panel:

```bash
python reproducibility/reproduce.py commands stability
```

For multi-host runs, append `--worker-host` and
`--distributed-repo-root` according to `docs/native_inference.md`. Host names,
IP addresses, and storage paths are deployment details and are deliberately not
part of the scientific config.

The released configuration excludes the private model used during internal
development. Its maximum context duplicated a public model's qualification
context; it is not needed for the reported public-model tables.

## Auxiliary-input ablation

The current generation code emits target-only views for `common_factor` and
`cross_series_dependence`, and removes the treated carrier for
`covariate_impulse_response`. In every case the assessed target history and
treatment future are held fixed. Consequently the paired MASE difference tests
the value of the removed auxiliary input rather than the effect of a shifted
synthetic path.

The ablation is produced during a normal full pipeline run. The compact
cross-suite summaries used by the paper are reconstructed with:

```bash
python reproducibility/reproduce.py figures ablation
```

## Stability

The stability run changes only the augmentation seed over a fixed ten-task
GIFT-Eval short panel. Each seed therefore selects different treatment
structures over the same official instances. Run the ten commands emitted by
`commands stability`, then collect each
`04_analysis_suite/task_equal_summary.json` and rerun the stability analysis.

The Git bundle already contains the ten compact suite summaries needed to
reconstruct the reported stability figures. Full per-seed experiments are
listed as optional audit artifacts in `DATA_ARTIFACTS.md`.

## Chronos-2 fine-tuning

The fine-tuning code is an extension of the official Chronos repository rather
than part of the CaFE runtime. It is frozen as:

- an exact upstream Chronos commit;
- a small patch to the Chronos dataset and fit pipeline;
- the data preparation, training, checkpoint evaluation, and aggregation
  scripts used by the runs;
- exact base-model revision and file hashes;
- exact data selection and optimizer configs.

See `chronos2_finetuning/README.md` for the full workflow. The two training
recipes are reproduced as protocols, not described as a loss-only ablation:
they differ in objective, training representation, forecast origin, learning
rate, and effective data exposure.

## Numerical expectations

Generation and selection are deterministic for the recorded seed. CPU-derived
tables should match the committed files exactly in the recorded environment.
GPU forecasts and training can vary slightly across CUDA, driver, and kernel
versions. Reproductions should therefore compare aggregate metrics with a
documented floating-point tolerance rather than require byte-identical model
weights.

## Data availability

Git contains code, compact manifests, analysis snapshots, tables, and figures.
Source benchmark files, replay contracts, predictions, and optional adapters
belong in the external artifact archive. `DATA_ARTIFACTS.md` gives the proposed
minimal and extended archive contents, their measured sizes, and why each item
is or is not needed.
