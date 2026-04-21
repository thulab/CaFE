# TSBenchmark v1 Algorithm Notes

## Feature Space

The integrated v1 keeps the 8-dimensional structural feature vector from
`data-xmy`:

- `trend_strength`
- `seasonal_strength`
- `spectral_entropy`
- `acf_half_life`
- `changepoint_density`
- `variance_shift`
- `intermittency`
- `outlier_rate`

The feature implementation lives in
`backend/app/datasets/benchmark_v1/features.py`.

## Anchor Prior

Anchor stats are built by scanning local GIFT-Eval and TFB directories for
numeric univariate series. If neither root is provided or no usable series is
found, the builder falls back to a bootstrap corpus and writes
`anchor_mode=bootstrap` into metadata.

The current prior is a discrete k-medoids prototype table with empirical
cluster weights. This preserves `data-xmy` behavior and deliberately does not
claim Gaussian copula support.

## Generation

Benchmark samples are generated from anchor prototypes plus one of the five
diagnostic families. Horizon and context follow:

- `horizon_ratio in {0.25, 0.5, 1.0}`
- `H = clip(round(r * dominant_scale), 12, 96)`
- `context = min(8H, 512)`
- `burn_in = max(4 * dominant_scale, 200)`

Default full-scale quota:

- Anchor Track: `2000`
- Diagnostic Track: `5 x 5 x 3 x 100 = 7500`
- Total: `9500`

## Evaluation

The v1 runner supports built-in baselines and external official adapters:

- `timesfm_2_5_200m`
- `chronos_bolt_base`
- `sundial_base_128m`
- `moirai_moe_base`
- `lag_llama`

Official adapters run in a separate interpreter selected by
`TSBENCHMARK_MODEL_PYTHON`. The main application does not download weights and
does not install the heavy model stacks by default.
