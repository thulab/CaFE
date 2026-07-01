# Synthetic v2 真实模型响应实验

日期：2026-07-01

## 目的

用本机 `timer-rest-service` 的真实模型验证 synthetic v2 `time_varying_seasonality` probe 是否能呈现模型能力差异。实验直接调用 `http://127.0.0.1:10810/ai/api/v1/forecast`，并保留 naive / seasonal naive 作为基线。

## 配置

- 服务：`http://127.0.0.1:10810`
- context / horizon / season：`168 / 24 / 24`
- 每个能力每个难度样本数：`12`
- batch size：`6`
- 能力维度：`time_varying_seasonality`
- required target / covariate dim：`1 / 0`
- requested 模型：Timer-3.5, Timer-3.0, Chronos-2, moirai2, toto2.0, timesfm2.5
- 参评 active 模型：Timer-3.5, Timer-3.0, Chronos-2, moirai2, toto2.0, timesfm2.5
- 跳过模型：none
- runtime 输出：`runtime/research/synthetic-v2-time-varying-seasonality-experiment`

## 模型运行状态

- `Timer-3.5`: `succeeded`, failed=0, elapsed=10.647s
- `Timer-3.0`: `succeeded`, failed=0, elapsed=4.18s
- `Chronos-2`: `succeeded`, failed=0, elapsed=2.035s
- `moirai2`: `succeeded`, failed=0, elapsed=1.544s
- `toto2.0`: `succeeded`, failed=0, elapsed=5.474s
- `timesfm2.5`: `succeeded`, failed=0, elapsed=27.284s

## 结果汇总

### `time_varying_seasonality`

| Model | Fail | MAE d1 | MAE d3 | MAE d5 | MASE d1 | MASE d3 | MASE d5 | MAE d5 / SNaive d5 |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| naive | 0 | 1.2653 | 1.4691 | 1.5269 | 3.9722 | 5.2679 | 5.9425 | - |
| seasonal_naive | 0 | 0.3754 | 0.3959 | 0.6642 | 1.1746 | 1.4157 | 2.5851 | - |
| Timer-3.5 | 0 | 0.3401 | 0.2183 | 0.1888 | 1.0642 | 0.7779 | 0.7338 | 0.2843 |
| Timer-3.0 | 0 | 0.3256 | 0.3206 | 0.3951 | 1.0194 | 1.147 | 1.5373 | 0.5948 |
| Chronos-2 | 0 | 0.2526 | 0.151 | 0.1998 | 0.7903 | 0.5388 | 0.7783 | 0.3008 |
| moirai2 | 0 | 0.348 | 0.558 | 0.563 | 1.086 | 1.9921 | 2.1889 | 0.8476 |
| toto2.0 | 0 | 0.2305 | 0.1809 | 0.1377 | 0.7231 | 0.6465 | 0.5355 | 0.2073 |
| timesfm2.5 | 0 | 0.2426 | 0.2101 | 0.2711 | 0.7594 | 0.7508 | 1.0561 | 0.4082 |


## 初步观察

- `time_varying_seasonality`：平均 MAE 最低的是 `toto2.0`（0.1847）。
- 这份结果用于观察真实模型响应，不替代后续更大样本、多随机种子和更多能力维度的论文主实验。

## 复现

```bash
cd backend && PYTHONPATH=.:../scripts uv run python ../scripts/run_synthetic_v2_real_model_experiment.py --models Timer-3.5 Timer-3.0 Chronos-2 moirai2 toto2.0 timesfm2.5 --capabilities time_varying_seasonality --sample-count 12 --batch-size 6
```
