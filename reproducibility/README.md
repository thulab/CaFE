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

## Software and hardware environment

Three reproduction levels have different requirements:

| Level | Required environment | Compute and storage |
|---|---|---|
| Verify frozen artifacts | Linux or macOS, Python 3.11+ | CPU only, less than 1 GB RAM and disk |
| Reconstruct tables and figures | Linux or macOS, Python 3.11+, `uv`, dependencies from `pyproject.toml`/`uv.lock` | CPU only; 4 cores, 8 GB RAM, and 5 GB free disk recommended |
| End-to-end CaFE experiments | Linux x86-64, Python 3.12, NVIDIA driver compatible with CUDA 12.8, `uv sync --extra fev --extra inference --extra plots` | one or more NVIDIA GPUs; 32 GB VRAM per worker, 16 CPU cores, 64 GB RAM, and 100 GB free disk recommended |

The recorded CaFE runs used Ubuntu 24.04.4 (kernel 6.8), Python 3.12.3,
`uv` 0.12.0, NVIDIA driver 575.57.08, CUDA 12.8, and a primary worker with
an Intel Xeon Gold 6530, 48 logical CPUs, 188 GiB RAM, and four RTX 5090 GPUs
with 32 GB VRAM each. Distributed inference used two or three workers depending
on the experiment block. Worker count changes throughput, not the experiment
definition or deterministic treatment contracts.

The principal recorded Python packages were PyTorch 2.10.0+cu128, NumPy 2.5.1,
PyArrow 25.0.0, pandas 2.3.3, SciPy 1.18.1, Matplotlib 3.11.1, FEV 0.8.0,
GluonTS 0.17.0, datasets 4.8.5, huggingface-hub 1.28.0, and safetensors
0.8.0. `uv.lock` remains the authoritative CaFE dependency lock; the versions
above document the machine that produced the archived outputs.

Four GPUs and multiple workers are not correctness requirements. A reviewer
may run models sequentially on fewer devices, provided each model's context
and memory requirements are met. CPU-derived verification and plotting require
no GPU.

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
included in the external reviewer artifact described by `DATA_ARTIFACTS.md`.

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

The acceptance thresholds are frozen before reviewer replay as follows. Every
comparison uses the displayed aggregation unit and the common set of
identifier keys; coverage counts and missingness states must match exactly.

| Replayed block | Comparison unit | Accepted numerical deviation |
|---|---|---:|
| Main and stability Reference/probe MASE and paired NRMSE | each shared suite-model-capability-level cell; suite-model cell for Reference MASE | `max(1e-4, 1% of abs(reference))` |
| Auxiliary-input removal | each shared suite-model-capability cell in `Delta MASE` | absolute deviation at most `1e-3` |
| Fine-tuning checkpoint evaluation | each objective-checkpoint-metric cell | `max(5e-4, 1% of abs(reference))` |
| Fine-tuning replay from step 0 | each objective-checkpoint-metric cell | `max(2e-3, 2% of abs(reference))` |

A replay fails if any required key is absent, a previously numeric cell becomes
nonnumeric, an unsupported or metric-undefined state changes category, a
coverage count changes, or any cell exceeds its applicable threshold. Derived
ranks and figures are regenerated only after the cell-level gate passes; ranks
are not used as a substitute tolerance for nearly tied values.

## End-to-end GPU smoke test

After staging the native model runtime, the following bounded run exercises one
official instance, one five-level treatment group, one Chronos-2 request stream,
and compact analysis:

```bash
uv run cafe run \
  --experiment-id reviewer-smoke-2026082701 \
  --dataset-id gift_ett1_h \
  --term short \
  --gift-eval-dir "$GIFT_EVAL_DIR" \
  --output-root "$CAFE_OUTPUT_ROOT" \
  --augmentation-seed 2026082701 \
  --capabilities trend \
  --models Chronos-2 \
  --backend native \
  --model-root "$CAFE_MODEL_ROOT" \
  --model-code-root "$CAFE_MODEL_CODE_ROOT" \
  --devices 0 \
  --max-instances 1 \
  --generation-workers 1 \
  --validation-workers 1 \
  --validation-dataset-workers 1 \
  --preprocess-workers 1 \
  --analysis-workers 1 \
  --disk-budget-gb 5

python reproducibility/reproduce.py smoke-check \
  "$CAFE_OUTPUT_ROOT/reviewer-smoke-2026082701"
```

On success, the final command prints:

```text
smoke check passed: dataset=gift_ett1_h, instances=1, treatments=5, models=1, stages=4
```

Runtime depends mainly on the first Chronos-2 load and local download cache;
the smoke scope, rather than a wall-clock promise, is the acceptance contract.

## Recovery and cleanup

`experiment.json` fixes the experiment identity. Each file in
`stage_contracts/` fixes that stage's code, configuration, and upstream hashes.
Before resuming, compare the requested arguments with those records; a mismatch
requires a new experiment ID.

- Generation, validation, and analysis are immutable after their terminal
  manifest or report exists. Reuse them with `--start-at` set to the next stage.
  If an interruption occurred before the terminal file was written, move only
  that exact incomplete stage directory to a quarantine location and rerun the
  stage with the same experiment ID and arguments. Never merge files from two
  experiment roots.
- Inference is the resumable stage. Rerun the original command with
  `--start-at inference --resume-inference`; completed model shards are accepted
  only when their recorded hashes still match, and pending models are issued
  again. Temporary `*.tmp` files inside the affected prediction directory may
  be removed after the process has stopped.
- Analysis scratch data under `04_analysis/.source_shard_parts/` are disposable
  only when `04_analysis/manifest.json` is absent. Completed analysis outputs
  and every upstream generation, validation, and inference artifact remain
  immutable.
- Fine-tuning preparation caches are reusable when their local manifest exists.
  An incomplete cache is quarantined and rebuilt from the CaFE contracts.
  Training does not claim optimizer-state recovery: restart an interrupted
  recipe in a new empty model directory. Completed LoRA checkpoints may be
  evaluated independently, and incomplete metric-part directories may be
  regenerated without rerunning training.

After any recovery, rerun `smoke-check` or the full frozen verification and
confirm that every stage contract points to the same experiment ID and upstream
hashes. Deleting an entire output root, completed stage, contract, checkpoint,
or source snapshot is never part of the recovery procedure.

## Data availability

Git contains code, compact manifests, analysis snapshots, tables, and figures.
Source benchmark files, replay contracts, predictions, and fine-tuned adapters
belong in the external artifact archive. `DATA_ARTIFACTS.md` defines its single
complete reviewer-facing layout. The designated distribution folder is
[Tsinghua Cloud](https://cloud.tsinghua.edu.cn/d/77e1d26573a347e89b6b/); use the
byte size and SHA-256 recorded in `DATA_ARTIFACTS.md` to identify and verify the
uploaded archive.
