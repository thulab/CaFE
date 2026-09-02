# Chronos-2 fine-tuning reproduction

This package reconstructs the two Chronos-2 fine-tuning protocols reported in
the paper. It does not vendor the upstream repository. Instead, it materializes
the exact official Chronos commit and applies the frozen CaFE overlay.

## Frozen versions

- Upstream: `amazon-science/chronos-forecasting`
- Commit: `8589d1988e9676817548e9626738ff06b6ca6370`
- Base model: `amazon/chronos-2`
- Model revision: `29ec3766d36d6f73f0696f85560a422f50e8498c`
- Python used for the recorded runs: 3.12.3
- GPU used for the recorded runs: NVIDIA GeForce RTX 5090 (32 GB)

`config.json` records the complete environment and SHA-256 digests of the base
model config and weights. The server's local model directory was verified to
be byte-identical to this Hugging Face revision.

## 1. Materialize Chronos

From the CaFE repository root:

```bash
python reproducibility/chronos2_finetuning/bootstrap_chronos.py \
  ../chronos-cafe-artifact
```

The bootstrapper clones Chronos, checks out the pinned commit, verifies and
applies `patches/chronos2-cafe.patch`, copies the overlay scripts and tests,
and writes `.cafe-finetuning-overlay.json`. It refuses to overwrite an
existing directory. Verify an existing materialization with:

```bash
python reproducibility/chronos2_finetuning/bootstrap_chronos.py \
  ../chronos-cafe-artifact --check
```

## 2. Create the environment

Inside the materialized Chronos repository:

```bash
uv venv --python 3.12
uv pip install -e ".[dev,extras]" \
  --constraint /path/to/CaFE/reproducibility/chronos2_finetuning/constraints.txt
```

The exact package versions observed on the training server are recorded in
`constraints.txt`. Install a CUDA-compatible PyTorch wheel for the target
machine, then constrain the remaining packages as closely as the platform
allows. If the selected PyTorch wheel must come from a platform-specific index,
install that wheel first and use the same constraint file for the editable
Chronos installation. Exact CUDA index commands are intentionally not
hard-coded because they depend on the reviewer's driver and package index.

Run the overlay tests:

```bash
.venv/bin/python -m pytest \
  test/test_chronos2.py \
  test/test_cafe_seed_transfer.py \
  test/test_cafe_effect_finetune.py
```

## 3. Required CaFE sources

The workflow consumes these two deterministic CaFE contract trees:

- `gift-v15-short-qualified-feasible-moirai16k-seed2026082701-r1`
- `gift-v15-short-qualified-feasible-moirai16k-seed2026082702-r1`

They must appear below `<cafe-root>/runtime/experiments/`. Both replay the
official GIFT-Eval Arrow source under `<cafe-root>/data/gift-eval/`.

The fit corpus contains 50,535 eligible treatments. The evaluation corpus
contains 48,365. Exact fold salts, counts, capabilities, levels, horizon, and
contexts are machine-readable in `config.json`; no source-order or random
subsampling is left implicit.

## 4. Inspect the exact commands

Before allocating GPUs, print the complete workflow without executing it:

```bash
python reproducibility/chronos2_finetuning/run_finetuning.py all \
  --cafe-root "$PWD" \
  --chronos-root ../chronos-cafe-artifact \
  --work-root /path/to/chronos-cafe-run \
  --dry-run
```

The runner downloads the pinned base-model snapshot on a real run and verifies
both model file hashes before training.

## 5. Prepare, train, and evaluate

Run each phase separately for easier recovery:

```bash
python reproducibility/chronos2_finetuning/run_finetuning.py prepare \
  --cafe-root "$PWD" \
  --chronos-root ../chronos-cafe-artifact \
  --work-root /path/to/chronos-cafe-run

python reproducibility/chronos2_finetuning/run_finetuning.py train \
  --cafe-root "$PWD" \
  --chronos-root ../chronos-cafe-artifact \
  --work-root /path/to/chronos-cafe-run \
  --device-index 0

python reproducibility/chronos2_finetuning/run_finetuning.py evaluate \
  --cafe-root "$PWD" \
  --chronos-root ../chronos-cafe-artifact \
  --work-root /path/to/chronos-cafe-run \
  --gpus 0 1 2 3
```

The recorded runs use LoRA, 40,000 optimizer steps, checkpoints every 4,000
steps, training seed `2026082701`, a context of 2,048, and a series budget of
32. Checkpoint evaluation uses float32, context 8,192, and the median forecast.

The standard protocol uses Chronos' multi-quantile objective with learning rate
`1e-4`. The paired protocol optimizes squared, MASE-standardized treatment
effect NRMSE with learning rate `1e-5`. Both are evaluated with the same MASE
and effect-NRMSE implementation.

## 6. Outputs

The runner creates:

```text
<work-root>/
├── data/
│   ├── fit/{treatments,direct-evaluation,effect-pairs}
│   └── evaluation/{treatments,direct-evaluation}
├── models/{default,effect-nrmse}/checkpoint-*
└── results/{default,effect-nrmse}/
    ├── metric-parts/
    ├── curve.json
    └── curve.csv
```

On the recorded run, the materialized fit/evaluation treatment datasets are
about 1.8 GB each, the paired fit dataset is 3.6 GB, and the direct-evaluation
metadata total about 169 MB. The LoRA checkpoint trees are about 52 MB per
protocol. Materialized datasets are caches and need not be archived if the CaFE
contract trees are available.

To reconstruct the paper-facing summaries and figures from curve and metric
parts, place them under the layout described by
`paper_results/work/finetuning/README.md` and run:

```bash
python reproducibility/reproduce.py figures finetuning
```
