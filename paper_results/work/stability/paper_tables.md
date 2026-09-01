# Stability paper tables (generated)

## Macro effect NRMSE across augmentation seeds

| Model | Mean ± seed SD | Seed 95% empirical interval | 95% CI of seed mean | Mean rank (range) | Top-1 | NRMSE<1 |
|---|---:|---:|---:|---:|---:|---:|
| Chronos-2 | 0.529 ± 0.020 | [0.501, 0.555] | [0.515, 0.543] | 1.0 (1–1) | 10/10 | 92.2% |
| timesfm2.5 | 0.563 ± 0.012 | [0.549, 0.577] | [0.554, 0.572] | 2.3 (2–3) | 0/10 | 87.5% |
| Timer-3.5 | 0.568 ± 0.011 | [0.555, 0.585] | [0.560, 0.576] | 2.7 (2–3) | 0/10 | 90.0% |
| tirex2 | 0.630 ± 0.022 | [0.597, 0.657] | [0.614, 0.645] | 4.6 (4–7) | 0/10 | 82.5% |
| Timer-4.0 | 0.638 ± 0.022 | [0.604, 0.665] | [0.622, 0.654] | 5.4 (4–7) | 0/10 | 81.5% |
| moirai2 | 0.638 ± 0.009 | [0.629, 0.656] | [0.632, 0.645] | 5.3 (4–6) | 0/10 | 77.5% |
| toto2.0 | 0.671 ± 0.024 | [0.634, 0.708] | [0.653, 0.688] | 6.7 (5–7) | 0/10 | 75.8% |

## Capability-level stability (equal model × level cells)

| Capability | Mean ± seed SD | CV | NRMSE<1 | Crossing cells | Seed SD / task SE (median) | 10-unique structures |
|---|---:|---:|---:|---:|---:|---:|
| Trend | 0.307 ± 0.024 | 7.9% | 100.0% | 0/35 | 0.226 | 100.0% |
| Multi-seasonal | 0.891 ± 0.033 | 3.7% | 76.6% | 12/35 | 0.299 | 56.9% |
| TV seasonality | 0.521 ± 0.017 | 3.4% | 98.3% | 1/35 | 0.217 | 2.7% |
| Regime switching | 0.128 ± 0.011 | 8.4% | 100.0% | 0/35 | 0.353 | 91.8% |
| Intermittency | 1.006 ± 0.039 | 3.9% | 46.3% | 8/35 | 0.623 | 100.0% |
| Common factor | 0.311 ± 0.019 | 6.2% | 100.0% | 0/35 | 0.250 | 1.5% |
| Cross-series | 1.033 ± 0.047 | 4.6% | 60.6% | 22/35 | 0.383 | 0.2% |
| Covariate impulse | 0.646 ± 0.030 | 4.6% | 89.1% | 3/35 | 0.396 | 0.0% |

## Adjacent model gaps versus paired seed variation

| Better mean | Next model | Mean gap | Paired seed SD | SD / gap | Better seeds | Gap 95% CI |
|---|---|---:|---:|---:|---:|---:|
| Chronos-2 | timesfm2.5 | 0.0338 | 0.0133 | 0.39 | 10/10 | [0.0243, 0.0433] |
| timesfm2.5 | Timer-3.5 | 0.0050 | 0.0067 | 1.34 | 7/10 | [0.0002, 0.0098] |
| Timer-3.5 | tirex2 | 0.0614 | 0.0146 | 0.24 | 10/10 | [0.0509, 0.0718] |
| tirex2 | Timer-4.0 | 0.0086 | 0.0143 | 1.66 | 8/10 | [-0.0016, 0.0188] |
| Timer-4.0 | moirai2 | 0.0003 | 0.0210 | 79.45 | 5/10 | [-0.0148, 0.0153] |
| moirai2 | toto2.0 | 0.0323 | 0.0256 | 0.79 | 9/10 | [0.0140, 0.0506] |

## Rank agreement and cell winners

- Kendall's W: 0.889.
- Pairwise seed Spearman: mean 0.876, minimum 0.643.
- Pairwise seed Kendall tau: mean 0.767, minimum 0.429.
- Unanimous capability × level winner: 12/40 (30.0%).
- Mean modal-winner consistency: 79.0%.
