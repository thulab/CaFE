# Synthetic v2 Validation Definition Experiment

日期：2026-07-07

## Purpose

为论文中的真实分布抽取、特征维度和生成后检验规则提供可复现实验证据；本实验不包含 discriminative score 或 predictive score。

## Config

- Anchor: M4 Hourly, windows=2000 train=1600 holdout=400
- Window: context=168, horizon=24, season=24
- Synthetic: 48 samples per capability/difficulty

## Real Anchor Feature Quantiles

| Feature | p05 | p50 | p95 |
| --- | ---: | ---: | ---: |
| `trend_strength` | 0 | 0.1659 | 0.7714 |
| `seasonal_strength` | 0.5768 | 0.9129 | 0.9961 |
| `acf_abs_mean` | 0.2515 | 0.5201 | 0.5574 |
| `level_shift_strength` | 0.3115 | 0.5132 | 0.9885 |
| `burst_rate` | 0 | 0 | 0.01042 |

## Synthetic Difficulty Evidence

| Capability | Feature | d1 | d3 | d5 | Spearman |
| --- | --- | ---: | ---: | ---: | ---: |
| `trend` | `trend_strength` | 0.002674 | 0.2572 | 0.6078 | 1 |
| `multi_seasonal` | `seasonal_strength` | 0.9762 | 0.8829 | 0.6786 | -1 |
| `time_varying_seasonality` | `seasonal_strength` | 0.8897 | 0.6982 | 0.3479 | -1 |
| `regime_switching` | `level_shift_strength` | 1.675 | 1.544 | 1.402 | -0.7 |
| `long_memory_nonlinear` | `acf_abs_mean` | 0.4727 | 0.7534 | 0.3893 | -0.6 |
| `intermittent_heteroskedastic` | `burst_rate` | 0.01888 | 0.06738 | 0.1095 | 1 |
| `common_factor` | `pca_top1_explained` | 0.738 | 0.6478 | 0.6352 | -0.8 |
| `lead_lag_coupling` | `lead_lag_peak_abs` | 0.905 | 0.822 | 0.803 | -1 |
| `hierarchical_coherence` | `hierarchy_residual_mean_abs` | 7.815e-17 | 8.132e-17 | 8.516e-17 | 0.9 |
| `covariate_response` | `avg_abs_covariate_target_corr` | 0.3896 | 0.5123 | 0.5926 | 1 |

## Novelty Calibration

- real holdout raw DCR q05/median: `0.01401` / `0.09725`
- real holdout feature DCR q05/median: `0.004623` / `0.03811`

| Capability | raw novelty q05 | feature novelty q05 | raw near-dup | feature near-dup | NNDR q05 |
| --- | ---: | ---: | ---: | ---: | ---: |
| `covariate_response` | 31.29 | 17.26 | 0 | 0 | 0.9263 |
| `intermittent_heteroskedastic` | 31.33 | 54.38 | 0 | 0 | 0.9715 |
| `long_memory_nonlinear` | 25.77 | 14.54 | 0 | 0 | 0.904 |
| `multi_seasonal` | 6.487 | 4.113 | 0 | 0.004167 | 0.9179 |
| `regime_switching` | 27.93 | 32.99 | 0 | 0 | 0.8029 |
| `time_varying_seasonality` | 10.69 | 7.542 | 0 | 0 | 0.8907 |
| `trend` | 21.78 | 6.286 | 0 | 0 | 0.9482 |

## Controlled Distribution Distances

- real-vs-real MMD / SWD reference: `0.000261` / `0.04537`

| Capability | MMD vs real | SWD vs real |
| --- | ---: | ---: |
| `covariate_response` | 0.5042 | 0.9999 |
| `intermittent_heteroskedastic` | 0.9645 | 1.704 |
| `long_memory_nonlinear` | 0.3367 | 0.9139 |
| `multi_seasonal` | 0.1521 | 0.4862 |
| `regime_switching` | 0.768 | 1.651 |
| `time_varying_seasonality` | 0.2511 | 0.596 |
| `trend` | 0.1162 | 0.4523 |

## Recommended Feature Set

- Univariate: `trend_strength`, `seasonal_strength`, `acf_abs_mean`, `level_shift_strength`, `burst_rate`
- Multi/covariate: `pca_top1_explained`, `lead_lag_peak_abs`, `avg_abs_covariate_target_corr`, `hierarchy_residual_mean_abs`

Full JSON summary: `runtime/research/synthetic-v2-validation-definition-experiment/summary.json`.
