# Stability paper tables (generated)

## Macro effect NRMSE across augmentation seeds

| Model | Mean ± seed SD | Seed 95% empirical interval | 95% CI of seed mean | Mean rank (range) | Top-1 | NRMSE<1 |
|---|---:|---:|---:|---:|---:|---:|
| Chronos-2 | 0.529 ± 0.020 | [0.501, 0.555] | [0.515, 0.543] | 1.0 (1–1) | 10/10 | 92.2% |
| timesfm2.5 | 0.563 ± 0.012 | [0.549, 0.577] | [0.554, 0.572] | 2.3 (2–3) | 0/10 | 87.5% |
| Timer-3.5 | 0.568 ± 0.011 | [0.555, 0.585] | [0.560, 0.576] | 2.7 (2–3) | 0/10 | 90.0% |
| tirex2 | 0.630 ± 0.022 | [0.597, 0.657] | [0.614, 0.645] | 4.4 (4–6) | 0/10 | 82.5% |
| moirai2 | 0.638 ± 0.009 | [0.629, 0.656] | [0.632, 0.645] | 4.8 (4–6) | 0/10 | 77.5% |
| toto2.0 | 0.671 ± 0.024 | [0.634, 0.708] | [0.653, 0.688] | 5.8 (5–6) | 0/10 | 75.8% |

## Capability-level stability (equal model × level cells)

| Capability | Mean ± seed SD | CV | NRMSE<1 | Crossing cells | Seed SD / task SE (median) | 10-unique structures |
|---|---:|---:|---:|---:|---:|---:|
| Trend | 0.286 ± 0.026 | 9.0% | 100.0% | 0/35 | 0.253 | 100.0% |
| Multi-seasonal | 0.878 ± 0.030 | 3.5% | 79.7% | 8/35 | 0.275 | 56.9% |
| TV seasonality | 0.523 ± 0.016 | 3.1% | 98.0% | 1/35 | 0.198 | 2.7% |
| Regime switching | 0.126 ± 0.011 | 8.4% | 100.0% | 0/35 | 0.272 | 91.8% |
| Intermittency | 1.008 ± 0.036 | 3.6% | 46.7% | 7/35 | 0.596 | 100.0% |
| Common factor | 0.308 ± 0.021 | 6.8% | 100.0% | 0/35 | 0.314 | 1.5% |
| Cross-series | 1.043 ± 0.048 | 4.6% | 58.3% | 17/35 | 0.379 | 0.2% |
| Covariate impulse | 0.627 ± 0.028 | 4.4% | 91.3% | 1/35 | 0.364 | 0.0% |

## Adjacent model gaps versus paired seed variation

| Better mean | Next model | Mean gap | Paired seed SD | SD / gap | Better seeds | Gap 95% CI |
|---|---|---:|---:|---:|---:|---:|
| Chronos-2 | timesfm2.5 | 0.0338 | 0.0133 | 0.39 | 10/10 | [0.0243, 0.0433] |
| timesfm2.5 | Timer-3.5 | 0.0050 | 0.0067 | 1.34 | 7/10 | [0.0002, 0.0098] |
| Timer-3.5 | tirex2 | 0.0614 | 0.0146 | 0.24 | 10/10 | [0.0509, 0.0718] |
| tirex2 | moirai2 | 0.0089 | 0.0217 | 2.44 | 7/10 | [-0.0066, 0.0244] |
| moirai2 | toto2.0 | 0.0323 | 0.0256 | 0.79 | 9/10 | [0.0140, 0.0506] |

## Rank agreement and cell winners

- Kendall's W: 0.921.
- Pairwise seed Spearman: mean 0.912, minimum 0.771.
- Pairwise seed Kendall tau: mean 0.822, minimum 0.600.
- Unanimous capability × level winner: 12/40 (30.0%).
- Mean modal-winner consistency: 79.2%.
