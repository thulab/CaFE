# Synthetic v2 全能力真实模型实验表

日期：2026-07-01

## 输入实验

| Source | Capabilities | Target dim | Covariate dim | Samples / difficulty | Selected models |
| --- | --- | ---: | ---: | ---: | --- |
| `runtime/research/synthetic-v2-univariate-capabilities-experiment/summary.json` | `trend`, `multi_seasonal`, `regime_switching`, `long_memory_nonlinear`, `intermittent_heteroskedastic` | 1 | 0 | 12 | `Timer-3.5`, `Timer-3.0`, `Chronos-2`, `moirai2`, `toto2.0`, `timesfm2.5` |
| `runtime/research/synthetic-v2-time-varying-seasonality-experiment/summary.json` | `time_varying_seasonality` | 1 | 0 | 12 | `Timer-3.5`, `Timer-3.0`, `Chronos-2`, `moirai2`, `toto2.0`, `timesfm2.5` |
| `runtime/research/synthetic-v2-multitarget-capabilities-experiment/summary.json` | `common_factor`, `lead_lag_coupling`, `coherent_regime_shift` | 3 | 0 | 12 | `toto2.0` |
| `runtime/research/synthetic-v2-hierarchical-coherence-experiment/summary.json` | `hierarchical_coherence` | 3 | 0 | 12 | `toto2.0` |
| `runtime/research/synthetic-v2-covariate-capabilities-experiment/summary.json` | `covariate_response` | 1 | 2 | 12 | `Chronos-2` |

## 模型支持边界

| Experiment | Requested model | Status | Reason |
| --- | --- | --- | --- |
| trend, multi_seasonal, regime_switching, long_memory_nonlinear, intermittent_heteroskedastic | `Timer-3.5` | selected | - |
| trend, multi_seasonal, regime_switching, long_memory_nonlinear, intermittent_heteroskedastic | `Timer-3.0` | selected | - |
| trend, multi_seasonal, regime_switching, long_memory_nonlinear, intermittent_heteroskedastic | `Chronos-2` | selected | - |
| trend, multi_seasonal, regime_switching, long_memory_nonlinear, intermittent_heteroskedastic | `moirai2` | selected | - |
| trend, multi_seasonal, regime_switching, long_memory_nonlinear, intermittent_heteroskedastic | `toto2.0` | selected | - |
| trend, multi_seasonal, regime_switching, long_memory_nonlinear, intermittent_heteroskedastic | `timesfm2.5` | selected | - |
| time_varying_seasonality | `Timer-3.5` | selected | - |
| time_varying_seasonality | `Timer-3.0` | selected | - |
| time_varying_seasonality | `Chronos-2` | selected | - |
| time_varying_seasonality | `moirai2` | selected | - |
| time_varying_seasonality | `toto2.0` | selected | - |
| time_varying_seasonality | `timesfm2.5` | selected | - |
| common_factor, lead_lag_coupling, coherent_regime_shift | `Timer-3.5` | skipped | target_dim_unsupported |
| common_factor, lead_lag_coupling, coherent_regime_shift | `Timer-3.0` | skipped | target_dim_unsupported |
| common_factor, lead_lag_coupling, coherent_regime_shift | `Chronos-2` | skipped | target_dim_unsupported |
| common_factor, lead_lag_coupling, coherent_regime_shift | `moirai2` | skipped | target_dim_unsupported |
| common_factor, lead_lag_coupling, coherent_regime_shift | `toto2.0` | selected | - |
| common_factor, lead_lag_coupling, coherent_regime_shift | `timesfm2.5` | skipped | target_dim_unsupported |
| hierarchical_coherence | `Timer-3.5` | skipped | target_dim_unsupported |
| hierarchical_coherence | `Timer-3.0` | skipped | target_dim_unsupported |
| hierarchical_coherence | `Chronos-2` | skipped | target_dim_unsupported |
| hierarchical_coherence | `moirai2` | skipped | target_dim_unsupported |
| hierarchical_coherence | `toto2.0` | selected | - |
| hierarchical_coherence | `timesfm2.5` | skipped | target_dim_unsupported |
| covariate_response | `Timer-3.5` | skipped | covariate_dim_unsupported |
| covariate_response | `Timer-3.0` | skipped | covariate_dim_unsupported |
| covariate_response | `Chronos-2` | selected | - |
| covariate_response | `moirai2` | skipped | covariate_dim_unsupported |
| covariate_response | `toto2.0` | skipped | covariate_dim_unsupported |
| covariate_response | `timesfm2.5` | skipped | covariate_dim_unsupported |

## 主要观察

- `trend`：平均 MAE 最低的是 `Chronos-2`（0.3433）。
- `multi_seasonal`：平均 MAE 最低的是 `toto2.0`（0.1008）。
- `time_varying_seasonality`：平均 MAE 最低的是 `toto2.0`（0.1847）。
- `regime_switching`：平均 MAE 最低的是 `Timer-3.0`（1.2165）。
- `long_memory_nonlinear`：平均 MAE 最低的是 `Timer-3.5`（0.4122）。
- `intermittent_heteroskedastic`：平均 MAE 最低的是 `Chronos-2`（0.4461）。
- `common_factor`：平均 MAE 最低的是 `toto2.0`（0.2389）。
- `lead_lag_coupling`：平均 MAE 最低的是 `toto2.0`（0.2355）。
- `coherent_regime_shift`：平均 MAE 最低的是 `toto2.0`（1.6006）。
- `hierarchical_coherence`：平均 MAE 最低的是 `toto2.0`（0.1609）。
- `covariate_response`：平均 MAE 最低的是 `Chronos-2`（0.3104）。
- 单变量 6 个维度本轮 6 个真实模型全部成功；Timer 修复后 `regime_switching` 不再 failed request。
- 新增 `time_varying_seasonality` 和 `hierarchical_coherence` 都能跑通；后者额外记录 `coherence_mae`，用于检查预测是否满足 parent-child 加总关系。
- 多目标维度当前只有 `toto2.0` 声明支持 `target_dim=3`，因此这些维度更像 toto 与 naive baselines 的 sanity check，还不能做横向模型排名。
- `covariate_response` 当前按单目标 known-future covariates 跑，只有 `Chronos-2` 纳入主实验；AutoARIMA/Holt-Winters 小样本 dry run 显示慢或失败，未进入主表。
- `regime_switching` 与 `coherent_regime_shift` 的 horizon 内切换带有不可预测成分，适合测鲁棒性和快速适应；如果论文要强调可预测能力，需要补先兆信号或 covariate shock 版本。
- `long_memory_nonlinear` 当前生成的是非线性持久性，不是严格 fractional long-memory；论文命名应避免过度声称。
- `intermittent_heteroskedastic` 的 spike/outlier/noise 随难度增强明显，但 `target_max_abs` 会偏高；后续需要用真实 intermittent demand 分布重新定 cap。

## 指标长表

| Capability | Difficulty | Model | Target dim | Cov dim | Samples | Fail | MAE | MASE | MSE | MAE / SNaive | Coherence MAE |
| --- | ---: | --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| `trend` | 1 | `naive` | 1 | 0 | 12 | 0 | 0.9327 | 2.1433 | 1.2629 | - | - |
| `trend` | 1 | `seasonal_naive` | 1 | 0 | 12 | 0 | 0.4991 | 1.1457 | 0.3818 | - | - |
| `trend` | 1 | `Timer-3.5` | 1 | 0 | 12 | 0 | 0.3287 | 0.7559 | 0.159 | 0.6585 | - |
| `trend` | 1 | `Timer-3.0` | 1 | 0 | 12 | 0 | 0.3231 | 0.7422 | 0.1555 | 0.6474 | - |
| `trend` | 1 | `Chronos-2` | 1 | 0 | 12 | 0 | 0.3126 | 0.7199 | 0.1458 | 0.6263 | - |
| `trend` | 1 | `moirai2` | 1 | 0 | 12 | 0 | 0.3754 | 0.8634 | 0.209 | 0.752 | - |
| `trend` | 1 | `toto2.0` | 1 | 0 | 12 | 0 | 0.3138 | 0.7217 | 0.1428 | 0.6287 | - |
| `trend` | 1 | `timesfm2.5` | 1 | 0 | 12 | 0 | 0.3162 | 0.728 | 0.1478 | 0.6335 | - |
| `trend` | 2 | `naive` | 1 | 0 | 12 | 0 | 0.9185 | 2.1208 | 1.2332 | - | - |
| `trend` | 2 | `seasonal_naive` | 1 | 0 | 12 | 0 | 0.4795 | 1.1018 | 0.3517 | - | - |
| `trend` | 2 | `Timer-3.5` | 1 | 0 | 12 | 0 | 0.3747 | 0.8615 | 0.2114 | 0.7814 | - |
| `trend` | 2 | `Timer-3.0` | 1 | 0 | 12 | 0 | 0.3597 | 0.827 | 0.1963 | 0.7503 | - |
| `trend` | 2 | `Chronos-2` | 1 | 0 | 12 | 0 | 0.3293 | 0.7565 | 0.1656 | 0.6868 | - |
| `trend` | 2 | `moirai2` | 1 | 0 | 12 | 0 | 0.4017 | 0.9247 | 0.2328 | 0.8377 | - |
| `trend` | 2 | `toto2.0` | 1 | 0 | 12 | 0 | 0.3288 | 0.7549 | 0.1694 | 0.6859 | - |
| `trend` | 2 | `timesfm2.5` | 1 | 0 | 12 | 0 | 0.3444 | 0.7904 | 0.1795 | 0.7182 | - |
| `trend` | 3 | `naive` | 1 | 0 | 12 | 0 | 0.8316 | 2.0557 | 0.9982 | - | - |
| `trend` | 3 | `seasonal_naive` | 1 | 0 | 12 | 0 | 0.4779 | 1.1811 | 0.3521 | - | - |
| `trend` | 3 | `Timer-3.5` | 1 | 0 | 12 | 0 | 0.3867 | 0.9516 | 0.2335 | 0.8093 | - |
| `trend` | 3 | `Timer-3.0` | 1 | 0 | 12 | 0 | 0.3721 | 0.9165 | 0.2172 | 0.7787 | - |
| `trend` | 3 | `Chronos-2` | 1 | 0 | 12 | 0 | 0.3248 | 0.8018 | 0.1693 | 0.6798 | - |
| `trend` | 3 | `moirai2` | 1 | 0 | 12 | 0 | 0.3888 | 0.9586 | 0.2413 | 0.8136 | - |
| `trend` | 3 | `toto2.0` | 1 | 0 | 12 | 0 | 0.3349 | 0.8295 | 0.1729 | 0.7009 | - |
| `trend` | 3 | `timesfm2.5` | 1 | 0 | 12 | 0 | 0.3445 | 0.8501 | 0.1833 | 0.7208 | - |
| `trend` | 4 | `naive` | 1 | 0 | 12 | 0 | 0.9385 | 2.4276 | 1.2876 | - | - |
| `trend` | 4 | `seasonal_naive` | 1 | 0 | 12 | 0 | 0.5011 | 1.2898 | 0.3767 | - | - |
| `trend` | 4 | `Timer-3.5` | 1 | 0 | 12 | 0 | 0.5293 | 1.3598 | 0.4349 | 1.0563 | - |
| `trend` | 4 | `Timer-3.0` | 1 | 0 | 12 | 0 | 0.5118 | 1.3107 | 0.4008 | 1.0214 | - |
| `trend` | 4 | `Chronos-2` | 1 | 0 | 12 | 0 | 0.3749 | 0.9662 | 0.2289 | 0.7482 | - |
| `trend` | 4 | `moirai2` | 1 | 0 | 12 | 0 | 0.4723 | 1.2138 | 0.3442 | 0.9426 | - |
| `trend` | 4 | `toto2.0` | 1 | 0 | 12 | 0 | 0.3828 | 0.991 | 0.2312 | 0.7641 | - |
| `trend` | 4 | `timesfm2.5` | 1 | 0 | 12 | 0 | 0.4621 | 1.1904 | 0.3229 | 0.9224 | - |
| `trend` | 5 | `naive` | 1 | 0 | 12 | 0 | 0.8593 | 2.3645 | 1.104 | - | - |
| `trend` | 5 | `seasonal_naive` | 1 | 0 | 12 | 0 | 0.4641 | 1.2787 | 0.3423 | - | - |
| `trend` | 5 | `Timer-3.5` | 1 | 0 | 12 | 0 | 0.4682 | 1.2742 | 0.3735 | 1.0087 | - |
| `trend` | 5 | `Timer-3.0` | 1 | 0 | 12 | 0 | 0.4437 | 1.2121 | 0.33 | 0.9559 | - |
| `trend` | 5 | `Chronos-2` | 1 | 0 | 12 | 0 | 0.3746 | 1.0277 | 0.2232 | 0.8071 | - |
| `trend` | 5 | `moirai2` | 1 | 0 | 12 | 0 | 0.4071 | 1.1154 | 0.2761 | 0.8771 | - |
| `trend` | 5 | `toto2.0` | 1 | 0 | 12 | 0 | 0.3874 | 1.0619 | 0.2503 | 0.8347 | - |
| `trend` | 5 | `timesfm2.5` | 1 | 0 | 12 | 0 | 0.386 | 1.0503 | 0.2518 | 0.8317 | - |
| `multi_seasonal` | 1 | `naive` | 1 | 0 | 12 | 0 | 1.1247 | 4.2286 | 1.9445 | - | - |
| `multi_seasonal` | 1 | `seasonal_naive` | 1 | 0 | 12 | 0 | 0.1549 | 0.5833 | 0.0385 | - | - |
| `multi_seasonal` | 1 | `Timer-3.5` | 1 | 0 | 12 | 0 | 0.1124 | 0.4232 | 0.0197 | 0.7254 | - |
| `multi_seasonal` | 1 | `Timer-3.0` | 1 | 0 | 12 | 0 | 0.1108 | 0.4173 | 0.02 | 0.7151 | - |
| `multi_seasonal` | 1 | `Chronos-2` | 1 | 0 | 12 | 0 | 0.1125 | 0.4235 | 0.0195 | 0.7263 | - |
| `multi_seasonal` | 1 | `moirai2` | 1 | 0 | 12 | 0 | 0.1341 | 0.5033 | 0.0276 | 0.8658 | - |
| `multi_seasonal` | 1 | `toto2.0` | 1 | 0 | 12 | 0 | 0.1048 | 0.3949 | 0.0173 | 0.6765 | - |
| `multi_seasonal` | 1 | `timesfm2.5` | 1 | 0 | 12 | 0 | 0.1103 | 0.4161 | 0.0193 | 0.7121 | - |
| `multi_seasonal` | 2 | `naive` | 1 | 0 | 12 | 0 | 1.0117 | 3.9273 | 1.4729 | - | - |
| `multi_seasonal` | 2 | `seasonal_naive` | 1 | 0 | 12 | 0 | 0.3035 | 1.1819 | 0.1258 | - | - |
| `multi_seasonal` | 2 | `Timer-3.5` | 1 | 0 | 12 | 0 | 0.1246 | 0.4849 | 0.0242 | 0.4106 | - |
| `multi_seasonal` | 2 | `Timer-3.0` | 1 | 0 | 12 | 0 | 0.1482 | 0.5759 | 0.0385 | 0.4883 | - |
| `multi_seasonal` | 2 | `Chronos-2` | 1 | 0 | 12 | 0 | 0.1211 | 0.4712 | 0.0229 | 0.3991 | - |
| `multi_seasonal` | 2 | `moirai2` | 1 | 0 | 12 | 0 | 0.2579 | 1.0069 | 0.0995 | 0.8497 | - |
| `multi_seasonal` | 2 | `toto2.0` | 1 | 0 | 12 | 0 | 0.118 | 0.4594 | 0.0216 | 0.389 | - |
| `multi_seasonal` | 2 | `timesfm2.5` | 1 | 0 | 12 | 0 | 0.1186 | 0.4617 | 0.0222 | 0.391 | - |
| `multi_seasonal` | 3 | `naive` | 1 | 0 | 12 | 0 | 1.021 | 4.1555 | 1.5201 | - | - |
| `multi_seasonal` | 3 | `seasonal_naive` | 1 | 0 | 12 | 0 | 0.6126 | 2.5044 | 0.4906 | - | - |
| `multi_seasonal` | 3 | `Timer-3.5` | 1 | 0 | 12 | 0 | 0.1085 | 0.4417 | 0.0183 | 0.1771 | - |
| `multi_seasonal` | 3 | `Timer-3.0` | 1 | 0 | 12 | 0 | 0.124 | 0.5048 | 0.0247 | 0.2024 | - |
| `multi_seasonal` | 3 | `Chronos-2` | 1 | 0 | 12 | 0 | 0.1084 | 0.4434 | 0.0179 | 0.177 | - |
| `multi_seasonal` | 3 | `moirai2` | 1 | 0 | 12 | 0 | 0.2768 | 1.1263 | 0.1162 | 0.4518 | - |
| `multi_seasonal` | 3 | `toto2.0` | 1 | 0 | 12 | 0 | 0.1052 | 0.4299 | 0.0167 | 0.1718 | - |
| `multi_seasonal` | 3 | `timesfm2.5` | 1 | 0 | 12 | 0 | 0.1182 | 0.4823 | 0.0205 | 0.193 | - |
| `multi_seasonal` | 4 | `naive` | 1 | 0 | 12 | 0 | 1.1009 | 4.7472 | 1.8447 | - | - |
| `multi_seasonal` | 4 | `seasonal_naive` | 1 | 0 | 12 | 0 | 0.8223 | 3.5672 | 0.8443 | - | - |
| `multi_seasonal` | 4 | `Timer-3.5` | 1 | 0 | 12 | 0 | 0.0946 | 0.4085 | 0.0144 | 0.1151 | - |
| `multi_seasonal` | 4 | `Timer-3.0` | 1 | 0 | 12 | 0 | 0.1082 | 0.4675 | 0.0179 | 0.1316 | - |
| `multi_seasonal` | 4 | `Chronos-2` | 1 | 0 | 12 | 0 | 0.0986 | 0.4263 | 0.0149 | 0.1199 | - |
| `multi_seasonal` | 4 | `moirai2` | 1 | 0 | 12 | 0 | 0.1639 | 0.7048 | 0.0512 | 0.1994 | - |
| `multi_seasonal` | 4 | `toto2.0` | 1 | 0 | 12 | 0 | 0.0925 | 0.4004 | 0.0133 | 0.1125 | - |
| `multi_seasonal` | 4 | `timesfm2.5` | 1 | 0 | 12 | 0 | 0.1004 | 0.4339 | 0.0163 | 0.1221 | - |
| `multi_seasonal` | 5 | `naive` | 1 | 0 | 12 | 0 | 1.0905 | 4.7452 | 1.6959 | - | - |
| `multi_seasonal` | 5 | `seasonal_naive` | 1 | 0 | 12 | 0 | 0.9894 | 4.3119 | 1.2208 | - | - |
| `multi_seasonal` | 5 | `Timer-3.5` | 1 | 0 | 12 | 0 | 0.0937 | 0.4076 | 0.0136 | 0.0947 | - |
| `multi_seasonal` | 5 | `Timer-3.0` | 1 | 0 | 12 | 0 | 0.0948 | 0.4117 | 0.0143 | 0.0958 | - |
| `multi_seasonal` | 5 | `Chronos-2` | 1 | 0 | 12 | 0 | 0.0902 | 0.3919 | 0.0122 | 0.0912 | - |
| `multi_seasonal` | 5 | `moirai2` | 1 | 0 | 12 | 0 | 0.1093 | 0.4739 | 0.0196 | 0.1104 | - |
| `multi_seasonal` | 5 | `toto2.0` | 1 | 0 | 12 | 0 | 0.0834 | 0.3617 | 0.0113 | 0.0843 | - |
| `multi_seasonal` | 5 | `timesfm2.5` | 1 | 0 | 12 | 0 | 0.0877 | 0.3811 | 0.0118 | 0.0886 | - |
| `time_varying_seasonality` | 1 | `naive` | 1 | 0 | 12 | 0 | 1.2653 | 3.9722 | 2.4179 | - | - |
| `time_varying_seasonality` | 1 | `seasonal_naive` | 1 | 0 | 12 | 0 | 0.3754 | 1.1746 | 0.2057 | - | - |
| `time_varying_seasonality` | 1 | `Timer-3.5` | 1 | 0 | 12 | 0 | 0.3401 | 1.0642 | 0.171 | 0.9059 | - |
| `time_varying_seasonality` | 1 | `Timer-3.0` | 1 | 0 | 12 | 0 | 0.3256 | 1.0194 | 0.1603 | 0.8673 | - |
| `time_varying_seasonality` | 1 | `Chronos-2` | 1 | 0 | 12 | 0 | 0.2526 | 0.7903 | 0.1 | 0.6729 | - |
| `time_varying_seasonality` | 1 | `moirai2` | 1 | 0 | 12 | 0 | 0.348 | 1.086 | 0.1775 | 0.9269 | - |
| `time_varying_seasonality` | 1 | `toto2.0` | 1 | 0 | 12 | 0 | 0.2305 | 0.7231 | 0.0883 | 0.6139 | - |
| `time_varying_seasonality` | 1 | `timesfm2.5` | 1 | 0 | 12 | 0 | 0.2426 | 0.7594 | 0.0911 | 0.6461 | - |
| `time_varying_seasonality` | 2 | `naive` | 1 | 0 | 12 | 0 | 1.4474 | 5.0224 | 3.1799 | - | - |
| `time_varying_seasonality` | 2 | `seasonal_naive` | 1 | 0 | 12 | 0 | 0.3631 | 1.2559 | 0.1894 | - | - |
| `time_varying_seasonality` | 2 | `Timer-3.5` | 1 | 0 | 12 | 0 | 0.3087 | 1.0686 | 0.1405 | 0.85 | - |
| `time_varying_seasonality` | 2 | `Timer-3.0` | 1 | 0 | 12 | 0 | 0.4565 | 1.5787 | 0.2904 | 1.2571 | - |
| `time_varying_seasonality` | 2 | `Chronos-2` | 1 | 0 | 12 | 0 | 0.1971 | 0.6806 | 0.0618 | 0.5429 | - |
| `time_varying_seasonality` | 2 | `moirai2` | 1 | 0 | 12 | 0 | 0.5284 | 1.8277 | 0.3995 | 1.4551 | - |
| `time_varying_seasonality` | 2 | `toto2.0` | 1 | 0 | 12 | 0 | 0.2127 | 0.7346 | 0.069 | 0.5858 | - |
| `time_varying_seasonality` | 2 | `timesfm2.5` | 1 | 0 | 12 | 0 | 0.2302 | 0.794 | 0.0808 | 0.6338 | - |
| `time_varying_seasonality` | 3 | `naive` | 1 | 0 | 12 | 0 | 1.4691 | 5.2679 | 3.3473 | - | - |
| `time_varying_seasonality` | 3 | `seasonal_naive` | 1 | 0 | 12 | 0 | 0.3959 | 1.4157 | 0.2286 | - | - |
| `time_varying_seasonality` | 3 | `Timer-3.5` | 1 | 0 | 12 | 0 | 0.2183 | 0.7779 | 0.0746 | 0.5513 | - |
| `time_varying_seasonality` | 3 | `Timer-3.0` | 1 | 0 | 12 | 0 | 0.3206 | 1.147 | 0.1467 | 0.8097 | - |
| `time_varying_seasonality` | 3 | `Chronos-2` | 1 | 0 | 12 | 0 | 0.151 | 0.5388 | 0.0359 | 0.3813 | - |
| `time_varying_seasonality` | 3 | `moirai2` | 1 | 0 | 12 | 0 | 0.558 | 1.9921 | 0.4441 | 1.4094 | - |
| `time_varying_seasonality` | 3 | `toto2.0` | 1 | 0 | 12 | 0 | 0.1809 | 0.6465 | 0.05 | 0.4568 | - |
| `time_varying_seasonality` | 3 | `timesfm2.5` | 1 | 0 | 12 | 0 | 0.2101 | 0.7508 | 0.0648 | 0.5307 | - |
| `time_varying_seasonality` | 4 | `naive` | 1 | 0 | 12 | 0 | 1.4434 | 5.4342 | 3.1686 | - | - |
| `time_varying_seasonality` | 4 | `seasonal_naive` | 1 | 0 | 12 | 0 | 0.5258 | 1.9848 | 0.3649 | - | - |
| `time_varying_seasonality` | 4 | `Timer-3.5` | 1 | 0 | 12 | 0 | 0.2345 | 0.8859 | 0.0859 | 0.4461 | - |
| `time_varying_seasonality` | 4 | `Timer-3.0` | 1 | 0 | 12 | 0 | 0.3838 | 1.4491 | 0.216 | 0.73 | - |
| `time_varying_seasonality` | 4 | `Chronos-2` | 1 | 0 | 12 | 0 | 0.1647 | 0.6203 | 0.0414 | 0.3132 | - |
| `time_varying_seasonality` | 4 | `moirai2` | 1 | 0 | 12 | 0 | 0.5231 | 1.961 | 0.391 | 0.9949 | - |
| `time_varying_seasonality` | 4 | `toto2.0` | 1 | 0 | 12 | 0 | 0.1617 | 0.6106 | 0.0416 | 0.3076 | - |
| `time_varying_seasonality` | 4 | `timesfm2.5` | 1 | 0 | 12 | 0 | 0.2167 | 0.8156 | 0.0697 | 0.4122 | - |
| `time_varying_seasonality` | 5 | `naive` | 1 | 0 | 12 | 0 | 1.5269 | 5.9425 | 3.6339 | - | - |
| `time_varying_seasonality` | 5 | `seasonal_naive` | 1 | 0 | 12 | 0 | 0.6642 | 2.5851 | 0.5811 | - | - |
| `time_varying_seasonality` | 5 | `Timer-3.5` | 1 | 0 | 12 | 0 | 0.1888 | 0.7338 | 0.0554 | 0.2843 | - |
| `time_varying_seasonality` | 5 | `Timer-3.0` | 1 | 0 | 12 | 0 | 0.3951 | 1.5373 | 0.2371 | 0.5948 | - |
| `time_varying_seasonality` | 5 | `Chronos-2` | 1 | 0 | 12 | 0 | 0.1998 | 0.7783 | 0.0583 | 0.3008 | - |
| `time_varying_seasonality` | 5 | `moirai2` | 1 | 0 | 12 | 0 | 0.563 | 2.1889 | 0.4367 | 0.8476 | - |
| `time_varying_seasonality` | 5 | `toto2.0` | 1 | 0 | 12 | 0 | 0.1377 | 0.5355 | 0.0285 | 0.2073 | - |
| `time_varying_seasonality` | 5 | `timesfm2.5` | 1 | 0 | 12 | 0 | 0.2711 | 1.0561 | 0.1017 | 0.4082 | - |
| `regime_switching` | 1 | `naive` | 1 | 0 | 12 | 0 | 1.4687 | 5.975 | 4.246 | - | - |
| `regime_switching` | 1 | `seasonal_naive` | 1 | 0 | 12 | 0 | 1.5278 | 6.1776 | 4.9493 | - | - |
| `regime_switching` | 1 | `Timer-3.5` | 1 | 0 | 12 | 0 | 1.367 | 5.3809 | 4.7837 | 0.8948 | - |
| `regime_switching` | 1 | `Timer-3.0` | 1 | 0 | 12 | 0 | 1.3245 | 5.172 | 4.4702 | 0.8669 | - |
| `regime_switching` | 1 | `Chronos-2` | 1 | 0 | 12 | 0 | 1.3614 | 5.4436 | 4.5917 | 0.8911 | - |
| `regime_switching` | 1 | `moirai2` | 1 | 0 | 12 | 0 | 1.3648 | 5.276 | 4.7846 | 0.8933 | - |
| `regime_switching` | 1 | `toto2.0` | 1 | 0 | 12 | 0 | 1.2891 | 5.0248 | 4.296 | 0.8438 | - |
| `regime_switching` | 1 | `timesfm2.5` | 1 | 0 | 12 | 0 | 1.3988 | 5.5999 | 4.7393 | 0.9155 | - |
| `regime_switching` | 2 | `naive` | 1 | 0 | 12 | 0 | 1.5955 | 4.8291 | 4.2831 | - | - |
| `regime_switching` | 2 | `seasonal_naive` | 1 | 0 | 12 | 0 | 1.5653 | 4.5722 | 4.588 | - | - |
| `regime_switching` | 2 | `Timer-3.5` | 1 | 0 | 12 | 0 | 1.2758 | 3.6033 | 3.2878 | 0.815 | - |
| `regime_switching` | 2 | `Timer-3.0` | 1 | 0 | 12 | 0 | 1.2946 | 3.5546 | 3.4592 | 0.8271 | - |
| `regime_switching` | 2 | `Chronos-2` | 1 | 0 | 12 | 0 | 1.3814 | 4.2714 | 3.7312 | 0.8825 | - |
| `regime_switching` | 2 | `moirai2` | 1 | 0 | 12 | 0 | 1.4755 | 4.4321 | 4.01 | 0.9426 | - |
| `regime_switching` | 2 | `toto2.0` | 1 | 0 | 12 | 0 | 1.3956 | 4.3855 | 3.7912 | 0.8916 | - |
| `regime_switching` | 2 | `timesfm2.5` | 1 | 0 | 12 | 0 | 1.42 | 4.3587 | 4.0202 | 0.9072 | - |
| `regime_switching` | 3 | `naive` | 1 | 0 | 12 | 0 | 1.0929 | 3.6103 | 1.9379 | - | - |
| `regime_switching` | 3 | `seasonal_naive` | 1 | 0 | 12 | 0 | 1.1549 | 3.7663 | 2.2128 | - | - |
| `regime_switching` | 3 | `Timer-3.5` | 1 | 0 | 12 | 0 | 0.9784 | 3.1812 | 1.647 | 0.8472 | - |
| `regime_switching` | 3 | `Timer-3.0` | 1 | 0 | 12 | 0 | 0.8715 | 2.8017 | 1.4577 | 0.7546 | - |
| `regime_switching` | 3 | `Chronos-2` | 1 | 0 | 12 | 0 | 0.939 | 3.0328 | 1.5828 | 0.8131 | - |
| `regime_switching` | 3 | `moirai2` | 1 | 0 | 12 | 0 | 0.9288 | 2.9745 | 1.5414 | 0.8042 | - |
| `regime_switching` | 3 | `toto2.0` | 1 | 0 | 12 | 0 | 0.9571 | 3.1512 | 1.5999 | 0.8288 | - |
| `regime_switching` | 3 | `timesfm2.5` | 1 | 0 | 12 | 0 | 0.9891 | 3.2113 | 1.7484 | 0.8564 | - |
| `regime_switching` | 4 | `naive` | 1 | 0 | 12 | 0 | 1.5919 | 5.2806 | 4.0044 | - | - |
| `regime_switching` | 4 | `seasonal_naive` | 1 | 0 | 12 | 0 | 1.5052 | 5.0143 | 3.6324 | - | - |
| `regime_switching` | 4 | `Timer-3.5` | 1 | 0 | 12 | 0 | 1.2927 | 4.284 | 2.8469 | 0.8588 | - |
| `regime_switching` | 4 | `Timer-3.0` | 1 | 0 | 12 | 0 | 1.3037 | 4.3354 | 2.8485 | 0.8661 | - |
| `regime_switching` | 4 | `Chronos-2` | 1 | 0 | 12 | 0 | 1.4008 | 4.6728 | 3.3256 | 0.9306 | - |
| `regime_switching` | 4 | `moirai2` | 1 | 0 | 12 | 0 | 1.3249 | 4.4382 | 2.9752 | 0.8802 | - |
| `regime_switching` | 4 | `toto2.0` | 1 | 0 | 12 | 0 | 1.3677 | 4.5682 | 3.3639 | 0.9086 | - |
| `regime_switching` | 4 | `timesfm2.5` | 1 | 0 | 12 | 0 | 1.3561 | 4.531 | 3.2057 | 0.9009 | - |
| `regime_switching` | 5 | `naive` | 1 | 0 | 12 | 0 | 1.5035 | 4.2798 | 3.8252 | - | - |
| `regime_switching` | 5 | `seasonal_naive` | 1 | 0 | 12 | 0 | 1.5785 | 4.4143 | 3.9604 | - | - |
| `regime_switching` | 5 | `Timer-3.5` | 1 | 0 | 12 | 0 | 1.3905 | 3.9835 | 3.2192 | 0.8809 | - |
| `regime_switching` | 5 | `Timer-3.0` | 1 | 0 | 12 | 0 | 1.288 | 3.6787 | 2.9116 | 0.816 | - |
| `regime_switching` | 5 | `Chronos-2` | 1 | 0 | 12 | 0 | 1.3941 | 3.996 | 3.3594 | 0.8832 | - |
| `regime_switching` | 5 | `moirai2` | 1 | 0 | 12 | 0 | 1.2995 | 3.7774 | 3.0422 | 0.8233 | - |
| `regime_switching` | 5 | `toto2.0` | 1 | 0 | 12 | 0 | 1.4143 | 4.1461 | 3.6739 | 0.896 | - |
| `regime_switching` | 5 | `timesfm2.5` | 1 | 0 | 12 | 0 | 1.3888 | 4.0448 | 3.4725 | 0.8798 | - |
| `long_memory_nonlinear` | 1 | `naive` | 1 | 0 | 12 | 0 | 1.5715 | 6.9754 | 3.1413 | - | - |
| `long_memory_nonlinear` | 1 | `seasonal_naive` | 1 | 0 | 12 | 0 | 0.9254 | 4.128 | 0.9914 | - | - |
| `long_memory_nonlinear` | 1 | `Timer-3.5` | 1 | 0 | 12 | 0 | 0.2831 | 1.2514 | 0.1243 | 0.306 | - |
| `long_memory_nonlinear` | 1 | `Timer-3.0` | 1 | 0 | 12 | 0 | 0.4713 | 2.0979 | 0.3167 | 0.5093 | - |
| `long_memory_nonlinear` | 1 | `Chronos-2` | 1 | 0 | 12 | 0 | 0.3035 | 1.3414 | 0.1636 | 0.328 | - |
| `long_memory_nonlinear` | 1 | `moirai2` | 1 | 0 | 12 | 0 | 0.7185 | 3.1994 | 0.6422 | 0.7764 | - |
| `long_memory_nonlinear` | 1 | `toto2.0` | 1 | 0 | 12 | 0 | 0.4343 | 1.921 | 0.2855 | 0.4693 | - |
| `long_memory_nonlinear` | 1 | `timesfm2.5` | 1 | 0 | 12 | 0 | 0.2782 | 1.2307 | 0.1218 | 0.3006 | - |
| `long_memory_nonlinear` | 2 | `naive` | 1 | 0 | 12 | 0 | 1.6301 | 8.9277 | 3.2216 | - | - |
| `long_memory_nonlinear` | 2 | `seasonal_naive` | 1 | 0 | 12 | 0 | 1.0982 | 6.0449 | 1.5207 | - | - |
| `long_memory_nonlinear` | 2 | `Timer-3.5` | 1 | 0 | 12 | 0 | 0.3999 | 2.205 | 0.2547 | 0.3641 | - |
| `long_memory_nonlinear` | 2 | `Timer-3.0` | 1 | 0 | 12 | 0 | 0.4053 | 2.1938 | 0.3007 | 0.3691 | - |
| `long_memory_nonlinear` | 2 | `Chronos-2` | 1 | 0 | 12 | 0 | 0.404 | 2.2058 | 0.299 | 0.3679 | - |
| `long_memory_nonlinear` | 2 | `moirai2` | 1 | 0 | 12 | 0 | 0.8362 | 4.5967 | 0.9686 | 0.7614 | - |
| `long_memory_nonlinear` | 2 | `toto2.0` | 1 | 0 | 12 | 0 | 0.4806 | 2.6071 | 0.4402 | 0.4376 | - |
| `long_memory_nonlinear` | 2 | `timesfm2.5` | 1 | 0 | 12 | 0 | 0.4636 | 2.5214 | 0.3783 | 0.4221 | - |
| `long_memory_nonlinear` | 3 | `naive` | 1 | 0 | 12 | 0 | 1.1172 | 8.3011 | 1.7428 | - | - |
| `long_memory_nonlinear` | 3 | `seasonal_naive` | 1 | 0 | 12 | 0 | 0.7586 | 5.7181 | 1.0186 | - | - |
| `long_memory_nonlinear` | 3 | `Timer-3.5` | 1 | 0 | 12 | 0 | 0.5159 | 3.8228 | 0.4868 | 0.68 | - |
| `long_memory_nonlinear` | 3 | `Timer-3.0` | 1 | 0 | 12 | 0 | 0.5056 | 3.716 | 0.4232 | 0.6664 | - |
| `long_memory_nonlinear` | 3 | `Chronos-2` | 1 | 0 | 12 | 0 | 0.5461 | 4.0233 | 0.5206 | 0.7199 | - |
| `long_memory_nonlinear` | 3 | `moirai2` | 1 | 0 | 12 | 0 | 0.5998 | 4.4025 | 0.6771 | 0.7907 | - |
| `long_memory_nonlinear` | 3 | `toto2.0` | 1 | 0 | 12 | 0 | 0.5947 | 4.4084 | 0.593 | 0.784 | - |
| `long_memory_nonlinear` | 3 | `timesfm2.5` | 1 | 0 | 12 | 0 | 0.6268 | 4.683 | 0.6537 | 0.8262 | - |
| `long_memory_nonlinear` | 4 | `naive` | 1 | 0 | 12 | 0 | 1.2225 | 4.5151 | 2.1709 | - | - |
| `long_memory_nonlinear` | 4 | `seasonal_naive` | 1 | 0 | 12 | 0 | 0.7755 | 2.8984 | 0.8949 | - | - |
| `long_memory_nonlinear` | 4 | `Timer-3.5` | 1 | 0 | 12 | 0 | 0.4806 | 1.9436 | 0.3861 | 0.6198 | - |
| `long_memory_nonlinear` | 4 | `Timer-3.0` | 1 | 0 | 12 | 0 | 0.4716 | 2.0191 | 0.3464 | 0.6082 | - |
| `long_memory_nonlinear` | 4 | `Chronos-2` | 1 | 0 | 12 | 0 | 0.4938 | 1.9139 | 0.407 | 0.6368 | - |
| `long_memory_nonlinear` | 4 | `moirai2` | 1 | 0 | 12 | 0 | 0.5452 | 2.1494 | 0.459 | 0.703 | - |
| `long_memory_nonlinear` | 4 | `toto2.0` | 1 | 0 | 12 | 0 | 0.525 | 2.0054 | 0.4461 | 0.677 | - |
| `long_memory_nonlinear` | 4 | `timesfm2.5` | 1 | 0 | 12 | 0 | 0.5402 | 2.0949 | 0.4809 | 0.6966 | - |
| `long_memory_nonlinear` | 5 | `naive` | 1 | 0 | 12 | 0 | 0.7469 | 2.2576 | 0.8846 | - | - |
| `long_memory_nonlinear` | 5 | `seasonal_naive` | 1 | 0 | 12 | 0 | 0.5138 | 1.5236 | 0.4028 | - | - |
| `long_memory_nonlinear` | 5 | `Timer-3.5` | 1 | 0 | 12 | 0 | 0.3817 | 1.174 | 0.2256 | 0.7428 | - |
| `long_memory_nonlinear` | 5 | `Timer-3.0` | 1 | 0 | 12 | 0 | 0.4079 | 1.2628 | 0.256 | 0.7939 | - |
| `long_memory_nonlinear` | 5 | `Chronos-2` | 1 | 0 | 12 | 0 | 0.3524 | 1.076 | 0.1894 | 0.6859 | - |
| `long_memory_nonlinear` | 5 | `moirai2` | 1 | 0 | 12 | 0 | 0.4036 | 1.2452 | 0.2475 | 0.7854 | - |
| `long_memory_nonlinear` | 5 | `toto2.0` | 1 | 0 | 12 | 0 | 0.3651 | 1.0951 | 0.2074 | 0.7105 | - |
| `long_memory_nonlinear` | 5 | `timesfm2.5` | 1 | 0 | 12 | 0 | 0.4209 | 1.3223 | 0.2831 | 0.8192 | - |
| `intermittent_heteroskedastic` | 1 | `naive` | 1 | 0 | 12 | 0 | 0.7274 | 1.4473 | 2.4686 | - | - |
| `intermittent_heteroskedastic` | 1 | `seasonal_naive` | 1 | 0 | 12 | 0 | 0.5796 | 1.1425 | 2.5263 | - | - |
| `intermittent_heteroskedastic` | 1 | `Timer-3.5` | 1 | 0 | 12 | 0 | 0.5084 | 1.0073 | 2.3811 | 0.8772 | - |
| `intermittent_heteroskedastic` | 1 | `Timer-3.0` | 1 | 0 | 12 | 0 | 0.5048 | 0.9997 | 2.3319 | 0.8709 | - |
| `intermittent_heteroskedastic` | 1 | `Chronos-2` | 1 | 0 | 12 | 0 | 0.4485 | 0.8863 | 2.2673 | 0.7738 | - |
| `intermittent_heteroskedastic` | 1 | `moirai2` | 1 | 0 | 12 | 0 | 0.5409 | 1.0697 | 2.4264 | 0.9331 | - |
| `intermittent_heteroskedastic` | 1 | `toto2.0` | 1 | 0 | 12 | 0 | 0.4493 | 0.8875 | 2.2616 | 0.7752 | - |
| `intermittent_heteroskedastic` | 1 | `timesfm2.5` | 1 | 0 | 12 | 0 | 0.5461 | 1.0816 | 2.4113 | 0.9422 | - |
| `intermittent_heteroskedastic` | 2 | `naive` | 1 | 0 | 12 | 0 | 1.3032 | 2.2454 | 7.9954 | - | - |
| `intermittent_heteroskedastic` | 2 | `seasonal_naive` | 1 | 0 | 12 | 0 | 0.6484 | 1.0384 | 1.7968 | - | - |
| `intermittent_heteroskedastic` | 2 | `Timer-3.5` | 1 | 0 | 12 | 0 | 0.414 | 0.6531 | 0.79 | 0.6384 | - |
| `intermittent_heteroskedastic` | 2 | `Timer-3.0` | 1 | 0 | 12 | 0 | 0.3931 | 0.6223 | 0.7199 | 0.6062 | - |
| `intermittent_heteroskedastic` | 2 | `Chronos-2` | 1 | 0 | 12 | 0 | 0.386 | 0.6082 | 0.7406 | 0.5953 | - |
| `intermittent_heteroskedastic` | 2 | `moirai2` | 1 | 0 | 12 | 0 | 0.4174 | 0.6595 | 0.803 | 0.6438 | - |
| `intermittent_heteroskedastic` | 2 | `toto2.0` | 1 | 0 | 12 | 0 | 0.3922 | 0.6193 | 0.7524 | 0.6049 | - |
| `intermittent_heteroskedastic` | 2 | `timesfm2.5` | 1 | 0 | 12 | 0 | 0.4233 | 0.6689 | 0.8029 | 0.6528 | - |
| `intermittent_heteroskedastic` | 3 | `naive` | 1 | 0 | 12 | 0 | 0.8851 | 1.2878 | 2.5195 | - | - |
| `intermittent_heteroskedastic` | 3 | `seasonal_naive` | 1 | 0 | 12 | 0 | 0.7062 | 1.0301 | 2.3731 | - | - |
| `intermittent_heteroskedastic` | 3 | `Timer-3.5` | 1 | 0 | 12 | 0 | 0.4248 | 0.6159 | 1.3932 | 0.6015 | - |
| `intermittent_heteroskedastic` | 3 | `Timer-3.0` | 1 | 0 | 12 | 0 | 0.4507 | 0.6533 | 1.3445 | 0.6382 | - |
| `intermittent_heteroskedastic` | 3 | `Chronos-2` | 1 | 0 | 12 | 0 | 0.4086 | 0.5919 | 1.3574 | 0.5786 | - |
| `intermittent_heteroskedastic` | 3 | `moirai2` | 1 | 0 | 12 | 0 | 0.4383 | 0.6349 | 1.4229 | 0.6206 | - |
| `intermittent_heteroskedastic` | 3 | `toto2.0` | 1 | 0 | 12 | 0 | 0.4091 | 0.5925 | 1.3819 | 0.5792 | - |
| `intermittent_heteroskedastic` | 3 | `timesfm2.5` | 1 | 0 | 12 | 0 | 0.4282 | 0.6203 | 1.4026 | 0.6063 | - |
| `intermittent_heteroskedastic` | 4 | `naive` | 1 | 0 | 12 | 0 | 0.6046 | 0.8067 | 1.3702 | - | - |
| `intermittent_heteroskedastic` | 4 | `seasonal_naive` | 1 | 0 | 12 | 0 | 0.8705 | 1.161 | 2.4384 | - | - |
| `intermittent_heteroskedastic` | 4 | `Timer-3.5` | 1 | 0 | 12 | 0 | 0.5156 | 0.6875 | 1.4267 | 0.5923 | - |
| `intermittent_heteroskedastic` | 4 | `Timer-3.0` | 1 | 0 | 12 | 0 | 0.5567 | 0.743 | 1.3278 | 0.6395 | - |
| `intermittent_heteroskedastic` | 4 | `Chronos-2` | 1 | 0 | 12 | 0 | 0.5125 | 0.6829 | 1.454 | 0.5888 | - |
| `intermittent_heteroskedastic` | 4 | `moirai2` | 1 | 0 | 12 | 0 | 0.5105 | 0.681 | 1.4293 | 0.5864 | - |
| `intermittent_heteroskedastic` | 4 | `toto2.0` | 1 | 0 | 12 | 0 | 0.4975 | 0.6634 | 1.3884 | 0.5715 | - |
| `intermittent_heteroskedastic` | 4 | `timesfm2.5` | 1 | 0 | 12 | 0 | 0.5053 | 0.674 | 1.4074 | 0.5804 | - |
| `intermittent_heteroskedastic` | 5 | `naive` | 1 | 0 | 12 | 0 | 0.8848 | 1.1671 | 2.4274 | - | - |
| `intermittent_heteroskedastic` | 5 | `seasonal_naive` | 1 | 0 | 12 | 0 | 0.7316 | 0.9709 | 1.934 | - | - |
| `intermittent_heteroskedastic` | 5 | `Timer-3.5` | 1 | 0 | 12 | 0 | 0.4814 | 0.6403 | 1.426 | 0.6581 | - |
| `intermittent_heteroskedastic` | 5 | `Timer-3.0` | 1 | 0 | 12 | 0 | 0.5834 | 0.7719 | 1.373 | 0.7974 | - |
| `intermittent_heteroskedastic` | 5 | `Chronos-2` | 1 | 0 | 12 | 0 | 0.4749 | 0.6315 | 1.4461 | 0.6491 | - |
| `intermittent_heteroskedastic` | 5 | `moirai2` | 1 | 0 | 12 | 0 | 0.4882 | 0.6497 | 1.4324 | 0.6673 | - |
| `intermittent_heteroskedastic` | 5 | `toto2.0` | 1 | 0 | 12 | 0 | 0.4825 | 0.6422 | 1.4647 | 0.6596 | - |
| `intermittent_heteroskedastic` | 5 | `timesfm2.5` | 1 | 0 | 12 | 0 | 0.4837 | 0.6433 | 1.4311 | 0.6611 | - |
| `common_factor` | 1 | `naive` | 3 | 0 | 12 | 0 | 0.9075 | 3.2628 | 1.1339 | - | - |
| `common_factor` | 1 | `seasonal_naive` | 3 | 0 | 12 | 0 | 1.1511 | 4.137 | 1.6685 | - | - |
| `common_factor` | 1 | `toto2.0` | 3 | 0 | 12 | 0 | 0.185 | 0.6544 | 0.0715 | 0.1607 | - |
| `common_factor` | 2 | `naive` | 3 | 0 | 12 | 0 | 0.9585 | 3.2036 | 1.2339 | - | - |
| `common_factor` | 2 | `seasonal_naive` | 3 | 0 | 12 | 0 | 1.0584 | 3.612 | 1.5098 | - | - |
| `common_factor` | 2 | `toto2.0` | 3 | 0 | 12 | 0 | 0.1953 | 0.6377 | 0.0761 | 0.1846 | - |
| `common_factor` | 3 | `naive` | 3 | 0 | 12 | 0 | 0.8143 | 3.0418 | 0.9382 | - | - |
| `common_factor` | 3 | `seasonal_naive` | 3 | 0 | 12 | 0 | 1.0664 | 4.0462 | 1.5705 | - | - |
| `common_factor` | 3 | `toto2.0` | 3 | 0 | 12 | 0 | 0.2868 | 0.9694 | 0.1769 | 0.2689 | - |
| `common_factor` | 4 | `naive` | 3 | 0 | 12 | 0 | 0.7669 | 2.6219 | 0.854 | - | - |
| `common_factor` | 4 | `seasonal_naive` | 3 | 0 | 12 | 0 | 0.7945 | 2.6945 | 0.9581 | - | - |
| `common_factor` | 4 | `toto2.0` | 3 | 0 | 12 | 0 | 0.257 | 0.8183 | 0.1236 | 0.3234 | - |
| `common_factor` | 5 | `naive` | 3 | 0 | 12 | 0 | 0.8562 | 2.806 | 1.0334 | - | - |
| `common_factor` | 5 | `seasonal_naive` | 3 | 0 | 12 | 0 | 1.1669 | 3.8316 | 1.7907 | - | - |
| `common_factor` | 5 | `toto2.0` | 3 | 0 | 12 | 0 | 0.2706 | 0.8383 | 0.1701 | 0.2319 | - |
| `lead_lag_coupling` | 1 | `naive` | 3 | 0 | 12 | 0 | 0.931 | 3.732 | 1.4858 | - | - |
| `lead_lag_coupling` | 1 | `seasonal_naive` | 3 | 0 | 12 | 0 | 1.0483 | 4.2235 | 1.5052 | - | - |
| `lead_lag_coupling` | 1 | `toto2.0` | 3 | 0 | 12 | 0 | 0.1649 | 0.6193 | 0.0622 | 0.1573 | - |
| `lead_lag_coupling` | 2 | `naive` | 3 | 0 | 12 | 0 | 1.0066 | 3.3492 | 1.5589 | - | - |
| `lead_lag_coupling` | 2 | `seasonal_naive` | 3 | 0 | 12 | 0 | 1.0559 | 3.8998 | 1.5141 | - | - |
| `lead_lag_coupling` | 2 | `toto2.0` | 3 | 0 | 12 | 0 | 0.2001 | 0.6234 | 0.0884 | 0.1895 | - |
| `lead_lag_coupling` | 3 | `naive` | 3 | 0 | 12 | 0 | 0.9254 | 3.5893 | 1.3668 | - | - |
| `lead_lag_coupling` | 3 | `seasonal_naive` | 3 | 0 | 12 | 0 | 0.9234 | 3.777 | 1.2013 | - | - |
| `lead_lag_coupling` | 3 | `toto2.0` | 3 | 0 | 12 | 0 | 0.1985 | 0.7278 | 0.08 | 0.2149 | - |
| `lead_lag_coupling` | 4 | `naive` | 3 | 0 | 12 | 0 | 0.9121 | 2.8332 | 1.3323 | - | - |
| `lead_lag_coupling` | 4 | `seasonal_naive` | 3 | 0 | 12 | 0 | 0.905 | 2.9197 | 1.2112 | - | - |
| `lead_lag_coupling` | 4 | `toto2.0` | 3 | 0 | 12 | 0 | 0.2931 | 0.8568 | 0.1867 | 0.3239 | - |
| `lead_lag_coupling` | 5 | `naive` | 3 | 0 | 12 | 0 | 0.9955 | 2.843 | 1.5587 | - | - |
| `lead_lag_coupling` | 5 | `seasonal_naive` | 3 | 0 | 12 | 0 | 0.9217 | 2.6246 | 1.2085 | - | - |
| `lead_lag_coupling` | 5 | `toto2.0` | 3 | 0 | 12 | 0 | 0.3207 | 0.8276 | 0.1897 | 0.3479 | - |
| `coherent_regime_shift` | 1 | `naive` | 3 | 0 | 12 | 0 | 1.6361 | 4.7544 | 4.3757 | - | - |
| `coherent_regime_shift` | 1 | `seasonal_naive` | 3 | 0 | 12 | 0 | 1.4678 | 4.2651 | 3.7753 | - | - |
| `coherent_regime_shift` | 1 | `toto2.0` | 3 | 0 | 12 | 0 | 1.1432 | 3.3219 | 3.3535 | 0.7789 | - |
| `coherent_regime_shift` | 2 | `naive` | 3 | 0 | 12 | 0 | 1.7955 | 4.6142 | 6.0768 | - | - |
| `coherent_regime_shift` | 2 | `seasonal_naive` | 3 | 0 | 12 | 0 | 1.6502 | 4.2451 | 5.6881 | - | - |
| `coherent_regime_shift` | 2 | `toto2.0` | 3 | 0 | 12 | 0 | 1.3979 | 3.5948 | 4.7375 | 0.8471 | - |
| `coherent_regime_shift` | 3 | `naive` | 3 | 0 | 12 | 0 | 1.7802 | 4.1326 | 5.4965 | - | - |
| `coherent_regime_shift` | 3 | `seasonal_naive` | 3 | 0 | 12 | 0 | 1.5378 | 3.5638 | 4.5948 | - | - |
| `coherent_regime_shift` | 3 | `toto2.0` | 3 | 0 | 12 | 0 | 1.3256 | 3.0702 | 4.4583 | 0.862 | - |
| `coherent_regime_shift` | 4 | `naive` | 3 | 0 | 12 | 0 | 2.6183 | 5.5608 | 12.2326 | - | - |
| `coherent_regime_shift` | 4 | `seasonal_naive` | 3 | 0 | 12 | 0 | 2.5812 | 5.4888 | 12.9441 | - | - |
| `coherent_regime_shift` | 4 | `toto2.0` | 3 | 0 | 12 | 0 | 2.3386 | 4.9677 | 12.0836 | 0.906 | - |
| `coherent_regime_shift` | 5 | `naive` | 3 | 0 | 12 | 0 | 2.2806 | 4.4347 | 12.0051 | - | - |
| `coherent_regime_shift` | 5 | `seasonal_naive` | 3 | 0 | 12 | 0 | 2.062 | 4.0086 | 10.8 | - | - |
| `coherent_regime_shift` | 5 | `toto2.0` | 3 | 0 | 12 | 0 | 1.7976 | 3.4906 | 9.8782 | 0.8718 | - |
| `hierarchical_coherence` | 1 | `naive` | 3 | 0 | 12 | 0 | 0.7938 | 3.6964 | 0.9386 | - | 0 |
| `hierarchical_coherence` | 1 | `seasonal_naive` | 3 | 0 | 12 | 0 | 0.6721 | 3.1253 | 0.7016 | - | 0 |
| `hierarchical_coherence` | 1 | `toto2.0` | 3 | 0 | 12 | 0 | 0.1165 | 0.5394 | 0.0237 | 0.1733 | 0.0892 |
| `hierarchical_coherence` | 2 | `naive` | 3 | 0 | 12 | 0 | 0.7767 | 3.5907 | 0.9808 | - | 0 |
| `hierarchical_coherence` | 2 | `seasonal_naive` | 3 | 0 | 12 | 0 | 0.6406 | 2.9622 | 0.6619 | - | 0 |
| `hierarchical_coherence` | 2 | `toto2.0` | 3 | 0 | 12 | 0 | 0.1518 | 0.7006 | 0.0471 | 0.2369 | 0.1384 |
| `hierarchical_coherence` | 3 | `naive` | 3 | 0 | 12 | 0 | 0.7826 | 3.5829 | 0.9568 | - | 0 |
| `hierarchical_coherence` | 3 | `seasonal_naive` | 3 | 0 | 12 | 0 | 0.7004 | 3.2099 | 0.7713 | - | 0 |
| `hierarchical_coherence` | 3 | `toto2.0` | 3 | 0 | 12 | 0 | 0.1631 | 0.7475 | 0.0453 | 0.2329 | 0.1372 |
| `hierarchical_coherence` | 4 | `naive` | 3 | 0 | 12 | 0 | 0.7894 | 3.6407 | 1.0875 | - | 0 |
| `hierarchical_coherence` | 4 | `seasonal_naive` | 3 | 0 | 12 | 0 | 0.6763 | 3.1279 | 0.7258 | - | 0 |
| `hierarchical_coherence` | 4 | `toto2.0` | 3 | 0 | 12 | 0 | 0.1583 | 0.7298 | 0.0491 | 0.234 | 0.149 |
| `hierarchical_coherence` | 5 | `naive` | 3 | 0 | 12 | 0 | 0.8994 | 3.929 | 1.3463 | - | 0 |
| `hierarchical_coherence` | 5 | `seasonal_naive` | 3 | 0 | 12 | 0 | 0.7349 | 3.1987 | 0.8632 | - | 0 |
| `hierarchical_coherence` | 5 | `toto2.0` | 3 | 0 | 12 | 0 | 0.2149 | 0.935 | 0.1005 | 0.2924 | 0.2141 |
| `covariate_response` | 1 | `naive` | 1 | 2 | 12 | 0 | 0.8719 | 2.4858 | 1.2078 | - | - |
| `covariate_response` | 1 | `seasonal_naive` | 1 | 2 | 12 | 0 | 1.1389 | 3.3139 | 1.7885 | - | - |
| `covariate_response` | 1 | `Chronos-2` | 1 | 2 | 12 | 0 | 0.2571 | 0.7473 | 0.0997 | 0.2257 | - |
| `covariate_response` | 2 | `naive` | 1 | 2 | 12 | 0 | 0.9012 | 2.3119 | 1.6289 | - | - |
| `covariate_response` | 2 | `seasonal_naive` | 1 | 2 | 12 | 0 | 1.0154 | 2.6192 | 1.9295 | - | - |
| `covariate_response` | 2 | `Chronos-2` | 1 | 2 | 12 | 0 | 0.2948 | 0.7577 | 0.1587 | 0.2903 | - |
| `covariate_response` | 3 | `naive` | 1 | 2 | 12 | 0 | 0.8245 | 2.2142 | 1.372 | - | - |
| `covariate_response` | 3 | `seasonal_naive` | 1 | 2 | 12 | 0 | 0.9987 | 2.7915 | 1.9253 | - | - |
| `covariate_response` | 3 | `Chronos-2` | 1 | 2 | 12 | 0 | 0.2974 | 0.7794 | 0.1613 | 0.2977 | - |
| `covariate_response` | 4 | `naive` | 1 | 2 | 12 | 0 | 1.0406 | 2.6761 | 1.9113 | - | - |
| `covariate_response` | 4 | `seasonal_naive` | 1 | 2 | 12 | 0 | 1.3046 | 3.5199 | 2.7266 | - | - |
| `covariate_response` | 4 | `Chronos-2` | 1 | 2 | 12 | 0 | 0.3487 | 0.9045 | 0.1969 | 0.2673 | - |
| `covariate_response` | 5 | `naive` | 1 | 2 | 12 | 0 | 1.065 | 2.6484 | 2.0468 | - | - |
| `covariate_response` | 5 | `seasonal_naive` | 1 | 2 | 12 | 0 | 1.2624 | 3.0088 | 2.4621 | - | - |
| `covariate_response` | 5 | `Chronos-2` | 1 | 2 | 12 | 0 | 0.3539 | 0.8539 | 0.1984 | 0.2803 | - |
