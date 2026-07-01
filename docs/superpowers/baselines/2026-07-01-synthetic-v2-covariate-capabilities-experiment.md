# Synthetic v2 真实模型响应实验

日期：2026-07-01

## 目的

用本机 `timer-rest-service` 的真实模型验证 synthetic v2 `covariate_response` probe 是否能呈现模型能力差异。实验直接调用 `http://127.0.0.1:10810/ai/api/v1/forecast`，并保留 naive / seasonal naive 作为基线。

## 配置

- 服务：`http://127.0.0.1:10810`
- context / horizon / season：`168 / 24 / 24`
- 每个能力每个难度样本数：`12`
- batch size：`6`
- 能力维度：`covariate_response`
- required target / covariate dim：`1 / 2`
- requested 模型：Timer-3.5, Timer-3.0, Chronos-2, moirai2, toto2.0, timesfm2.5
- 参评 active 模型：Chronos-2
- 跳过模型：Timer-3.5 (covariate_dim_unsupported), Timer-3.0 (covariate_dim_unsupported), moirai2 (covariate_dim_unsupported), toto2.0 (covariate_dim_unsupported), timesfm2.5 (covariate_dim_unsupported)
- runtime 输出：`runtime/research/synthetic-v2-covariate-capabilities-experiment`

## 模型运行状态

- `Chronos-2`: `succeeded`, failed=0, elapsed=2.218s

## 结果汇总

### `covariate_response`

| Model | Fail | MAE d1 | MAE d3 | MAE d5 | MASE d1 | MASE d3 | MASE d5 | MAE d5 / SNaive d5 |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| naive | 0 | 0.8719 | 0.8245 | 1.065 | 2.4858 | 2.2142 | 2.6484 | - |
| seasonal_naive | 0 | 1.1389 | 0.9987 | 1.2624 | 3.3139 | 2.7915 | 3.0088 | - |
| Chronos-2 | 0 | 0.2571 | 0.2974 | 0.3539 | 0.7473 | 0.7794 | 0.8539 | 0.2803 |


## 初步观察

- `covariate_response`：平均 MAE 最低的是 `Chronos-2`（0.3104）。
- 本轮跳过模型：`Timer-3.5`（covariate_dim_unsupported）, `Timer-3.0`（covariate_dim_unsupported）, `moirai2`（covariate_dim_unsupported）, `toto2.0`（covariate_dim_unsupported）, `timesfm2.5`（covariate_dim_unsupported）。
- 这份结果用于观察真实模型响应，不替代后续更大样本、多随机种子和更多能力维度的论文主实验。

## 复现

```bash
cd backend && PYTHONPATH=.:../scripts uv run python ../scripts/run_synthetic_v2_real_model_experiment.py --models Timer-3.5 Timer-3.0 Chronos-2 moirai2 toto2.0 timesfm2.5 --capabilities covariate_response --sample-count 12 --batch-size 6
```
