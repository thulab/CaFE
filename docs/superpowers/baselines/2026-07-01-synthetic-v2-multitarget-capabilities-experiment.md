# Synthetic v2 真实模型响应实验

日期：2026-07-01

## 目的

用本机 `timer-rest-service` 的真实模型验证 synthetic v2 `common_factor` / `lead_lag_coupling` / `coherent_regime_shift` probe 是否能呈现模型能力差异。实验直接调用 `http://127.0.0.1:10810/ai/api/v1/forecast`，并保留 naive / seasonal naive 作为基线。

## 配置

- 服务：`http://127.0.0.1:10810`
- context / horizon / season：`168 / 24 / 24`
- 每个能力每个难度样本数：`12`
- batch size：`6`
- 能力维度：`common_factor` / `lead_lag_coupling` / `coherent_regime_shift`
- required target / covariate dim：`3 / 0`
- requested 模型：Timer-3.5, Timer-3.0, Chronos-2, moirai2, toto2.0, timesfm2.5
- 参评 active 模型：toto2.0
- 跳过模型：Timer-3.5 (target_dim_unsupported), Timer-3.0 (target_dim_unsupported), Chronos-2 (target_dim_unsupported), moirai2 (target_dim_unsupported), timesfm2.5 (target_dim_unsupported)
- runtime 输出：`runtime/research/synthetic-v2-multitarget-capabilities-experiment`

## 模型运行状态

- `toto2.0`: `succeeded`, failed=0, elapsed=16.939s

## 结果汇总

### `common_factor`

| Model | Fail | MAE d1 | MAE d3 | MAE d5 | MASE d1 | MASE d3 | MASE d5 | MAE d5 / SNaive d5 |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| naive | 0 | 0.9075 | 0.8143 | 0.8562 | 3.2628 | 3.0418 | 2.806 | - |
| seasonal_naive | 0 | 1.1511 | 1.0664 | 1.1669 | 4.137 | 4.0462 | 3.8316 | - |
| toto2.0 | 0 | 0.185 | 0.2868 | 0.2706 | 0.6544 | 0.9694 | 0.8383 | 0.2319 |

### `lead_lag_coupling`

| Model | Fail | MAE d1 | MAE d3 | MAE d5 | MASE d1 | MASE d3 | MASE d5 | MAE d5 / SNaive d5 |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| naive | 0 | 0.931 | 0.9254 | 0.9955 | 3.732 | 3.5893 | 2.843 | - |
| seasonal_naive | 0 | 1.0483 | 0.9234 | 0.9217 | 4.2235 | 3.777 | 2.6246 | - |
| toto2.0 | 0 | 0.1649 | 0.1985 | 0.3207 | 0.6193 | 0.7278 | 0.8276 | 0.3479 |

### `coherent_regime_shift`

| Model | Fail | MAE d1 | MAE d3 | MAE d5 | MASE d1 | MASE d3 | MASE d5 | MAE d5 / SNaive d5 |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| naive | 0 | 1.6361 | 1.7802 | 2.2806 | 4.7544 | 4.1326 | 4.4347 | - |
| seasonal_naive | 0 | 1.4678 | 1.5378 | 2.062 | 4.2651 | 3.5638 | 4.0086 | - |
| toto2.0 | 0 | 1.1432 | 1.3256 | 1.7976 | 3.3219 | 3.0702 | 3.4906 | 0.8718 |


## 初步观察

- `common_factor`：平均 MAE 最低的是 `toto2.0`（0.2389）。
- `lead_lag_coupling`：平均 MAE 最低的是 `toto2.0`（0.2355）。
- `coherent_regime_shift`：平均 MAE 最低的是 `toto2.0`（1.6006）。
- 本轮跳过模型：`Timer-3.5`（target_dim_unsupported）, `Timer-3.0`（target_dim_unsupported）, `Chronos-2`（target_dim_unsupported）, `moirai2`（target_dim_unsupported）, `timesfm2.5`（target_dim_unsupported）。
- 这份结果用于观察真实模型响应，不替代后续更大样本、多随机种子和更多能力维度的论文主实验。

## 复现

```bash
cd backend && PYTHONPATH=.:../scripts uv run python ../scripts/run_synthetic_v2_real_model_experiment.py --models Timer-3.5 Timer-3.0 Chronos-2 moirai2 toto2.0 timesfm2.5 --capabilities common_factor lead_lag_coupling coherent_regime_shift --sample-count 12 --batch-size 6
```
