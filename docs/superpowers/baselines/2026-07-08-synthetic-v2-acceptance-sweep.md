# Synthetic v2 Acceptance Threshold Sweep

日期：2026-07-08

## Purpose

比较 hard acceptance 阈值策略对生成通过率、重采样成本和失败特征的影响，为论文阶段固定真实分布验收阈值提供依据。

## Design

- Samples: 48 base samples per capability/intensity, 12 deterministic attempts per sample.
- Window: context=168, horizon=24, requested multi target_dim=3, frequency=h.
- Each strategy is evaluated on the same raw attempt pool, so strategy differences come only from thresholds.
- `current` is the backend cap set. `profile_m*_event*` rebuilds caps from real-profile p95 values; bounded features are clipped at 1.0.
- Operational screen used for the automatic recommendation: min acceptance >= 0.90, at most two capability/intensity cells below 0.95, and max mean attempts <= 3.

## Strategy Summary

| Strategy | min acc | p10 acc | median acc | cells <0.95 | max mean attempts | top terminal failures |
| --- | ---: | ---: | ---: | ---: | ---: | --- |
| `current` | 1.000 | 1.000 | 1.000 | 0 | 1.042 | - |
| `profile_m1_event5` | 0 | 0.3458 | 1.000 | 10 | 5.756 | noise_ratio:268, burst_rate:49, outlier_rate:47, acf_abs_mean:42 |
| `profile_m1_25_event5` | 0 | 0.9792 | 1.000 | 5 | 7.600 | noise_ratio:163, outlier_rate:21, burst_rate:21, acf_abs_mean:16 |
| `profile_m1_5_event2` | 0.8958 | 1.000 | 1.000 | 1 | 5.279 | noise_ratio:5, burst_rate:1, outlier_rate:1 |
| `profile_m1_5_event3` | 0.8958 | 1.000 | 1.000 | 1 | 5.279 | noise_ratio:5, burst_rate:1, outlier_rate:1 |
| `profile_m1_5_event5` | 0.8958 | 1.000 | 1.000 | 1 | 5.279 | noise_ratio:5, burst_rate:1, outlier_rate:1 |
| `profile_m1_5_event7` | 0.8958 | 1.000 | 1.000 | 1 | 5.279 | noise_ratio:5, burst_rate:1, outlier_rate:1 |
| `profile_m2_event5` | 1.000 | 1.000 | 1.000 | 0 | 1.104 | - |
| `profile_m2_5_event5` | 1.000 | 1.000 | 1.000 | 0 | 1.000 | - |

## Recommendation

- Recommended strategy: `profile_m2_event5`.
- Reason: smallest profile-derived strategy satisfying min acceptance >= 0.90, at most two cells below 0.95, and max mean attempts <= 3.

## Capability Detail

下表只展示 `current`、推荐策略，以及相邻的 profile-derived 策略，便于判断是否过紧或过松。

### `current`

| Capability | min acc | i1 acc | i3 acc | i5 acc | max mean attempts | main terminal failures |
| --- | ---: | ---: | ---: | ---: | ---: | --- |
| `coherent_regime_shift` | 1.000 | 1.000 | 1.000 | 1.000 | 1.000 | - |
| `common_factor` | 1.000 | 1.000 | 1.000 | 1.000 | 1.000 | - |
| `covariate_response` | 1.000 | 1.000 | 1.000 | 1.000 | 1.000 | - |
| `hierarchical_coherence` | 1.000 | 1.000 | 1.000 | 1.000 | 1.000 | - |
| `intermittent_heteroskedastic` | 1.000 | 1.000 | 1.000 | 1.000 | 1.021 | - |
| `lead_lag_coupling` | 1.000 | 1.000 | 1.000 | 1.000 | 1.000 | - |
| `long_memory_nonlinear` | 1.000 | 1.000 | 1.000 | 1.000 | 1.042 | - |
| `multi_seasonal` | 1.000 | 1.000 | 1.000 | 1.000 | 1.000 | - |
| `regime_switching` | 1.000 | 1.000 | 1.000 | 1.000 | 1.000 | - |
| `time_varying_seasonality` | 1.000 | 1.000 | 1.000 | 1.000 | 1.000 | - |
| `trend` | 1.000 | 1.000 | 1.000 | 1.000 | 1.000 | - |

### `profile_m2_event5`

| Capability | min acc | i1 acc | i3 acc | i5 acc | max mean attempts | main terminal failures |
| --- | ---: | ---: | ---: | ---: | ---: | --- |
| `coherent_regime_shift` | 1.000 | 1.000 | 1.000 | 1.000 | 1.021 | - |
| `common_factor` | 1.000 | 1.000 | 1.000 | 1.000 | 1.000 | - |
| `covariate_response` | 1.000 | 1.000 | 1.000 | 1.000 | 1.000 | - |
| `hierarchical_coherence` | 1.000 | 1.000 | 1.000 | 1.000 | 1.000 | - |
| `intermittent_heteroskedastic` | 1.000 | 1.000 | 1.000 | 1.000 | 1.000 | - |
| `lead_lag_coupling` | 1.000 | 1.000 | 1.000 | 1.000 | 1.000 | - |
| `long_memory_nonlinear` | 1.000 | 1.000 | 1.000 | 1.000 | 1.000 | - |
| `multi_seasonal` | 1.000 | 1.000 | 1.000 | 1.000 | 1.000 | - |
| `regime_switching` | 1.000 | 1.000 | 1.000 | 1.000 | 1.104 | - |
| `time_varying_seasonality` | 1.000 | 1.000 | 1.000 | 1.000 | 1.000 | - |
| `trend` | 1.000 | 1.000 | 1.000 | 1.000 | 1.000 | - |

### `profile_m1_25_event5`

| Capability | min acc | i1 acc | i3 acc | i5 acc | max mean attempts | main terminal failures |
| --- | ---: | ---: | ---: | ---: | ---: | --- |
| `coherent_regime_shift` | 1.000 | 1.000 | 1.000 | 1.000 | 1.312 | - |
| `common_factor` | 1.000 | 1.000 | 1.000 | 1.000 | 1.688 | - |
| `covariate_response` | 1.000 | 1.000 | 1.000 | 1.000 | 1.000 | - |
| `hierarchical_coherence` | 1.000 | 1.000 | 1.000 | 1.000 | 1.000 | - |
| `intermittent_heteroskedastic` | 0 | 1.000 | 0.1042 | 0 | 7.600 | noise_ratio:155, outlier_rate:21, burst_rate:21, spike_rate:5 |
| `lead_lag_coupling` | 1.000 | 1.000 | 1.000 | 1.000 | 1.583 | - |
| `long_memory_nonlinear` | 0.6667 | 1.000 | 0.6667 | 1.000 | 5.938 | acf_abs_mean:16, noise_ratio:8 |
| `multi_seasonal` | 1.000 | 1.000 | 1.000 | 1.000 | 1.000 | - |
| `regime_switching` | 0.9792 | 0.9792 | 0.9792 | 1.000 | 3.191 | change_point_shift_energy:2, level_shift_strength:2, volatility_shift_strength:1 |
| `time_varying_seasonality` | 1.000 | 1.000 | 1.000 | 1.000 | 1.000 | - |
| `trend` | 1.000 | 1.000 | 1.000 | 1.000 | 1.292 | - |

### `profile_m1_5_event5`

| Capability | min acc | i1 acc | i3 acc | i5 acc | max mean attempts | main terminal failures |
| --- | ---: | ---: | ---: | ---: | ---: | --- |
| `coherent_regime_shift` | 1.000 | 1.000 | 1.000 | 1.000 | 1.188 | - |
| `common_factor` | 1.000 | 1.000 | 1.000 | 1.000 | 1.188 | - |
| `covariate_response` | 1.000 | 1.000 | 1.000 | 1.000 | 1.000 | - |
| `hierarchical_coherence` | 1.000 | 1.000 | 1.000 | 1.000 | 1.000 | - |
| `intermittent_heteroskedastic` | 0.8958 | 1.000 | 1.000 | 0.8958 | 5.279 | noise_ratio:5, burst_rate:1, outlier_rate:1 |
| `lead_lag_coupling` | 1.000 | 1.000 | 1.000 | 1.000 | 1.292 | - |
| `long_memory_nonlinear` | 1.000 | 1.000 | 1.000 | 1.000 | 1.042 | - |
| `multi_seasonal` | 1.000 | 1.000 | 1.000 | 1.000 | 1.000 | - |
| `regime_switching` | 1.000 | 1.000 | 1.000 | 1.000 | 2.146 | - |
| `time_varying_seasonality` | 1.000 | 1.000 | 1.000 | 1.000 | 1.000 | - |
| `trend` | 1.000 | 1.000 | 1.000 | 1.000 | 1.000 | - |


## Seasonality Resolution

| Capability | season_length | source | candidates |
| --- | ---: | --- | --- |
| `trend` | 24 | `profile_bucket` | 24 |
| `multi_seasonal` | 24 | `profile_bucket` | 24 |
| `regime_switching` | 24 | `profile_bucket` | 24 |
| `time_varying_seasonality` | 24 | `profile_bucket` | 24 |
| `long_memory_nonlinear` | 24 | `profile_bucket` | 24 |
| `intermittent_heteroskedastic` | 24 | `profile_bucket` | 24 |
| `common_factor` | 24 | `profile_bucket` | 24 |
| `lead_lag_coupling` | 24 | `profile_bucket` | 24 |
| `coherent_regime_shift` | 24 | `profile_bucket` | 24 |
| `hierarchical_coherence` | 7 | `profile_bucket` | 7 |
| `covariate_response` | 24 | `profile_bucket` | 24 |

## Notes

- `event_lift_abs` is swept separately because the M5 event profile is sparse and would otherwise dominate covariate acceptance.
- `hierarchy_residual_mean_abs` keeps a fixed floating-point tolerance floor of `1e-6` even though the real M5 p95 is 0.
- This is a first-pass operational sweep at the recorded sample size. Before freezing paper thresholds, rerun with a larger cached attempt pool and keep the same report schema.
- This sweep evaluates feature-threshold acceptance. Near-distance DCR/NNDR acceptance is calibrated by the separate real-holdout experiment and enforced online through the generated reference artifact.

Full JSON summary: `runtime/research/synthetic-v2-acceptance-sweep/summary.json`.
