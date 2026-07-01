# Synthetic v2 真实模型响应实验

日期：2026-07-01

## 目的

用本机 `timer-rest-service` 的真实模型验证 synthetic v2 `hierarchical_coherence` probe 是否能呈现模型能力差异。实验直接调用 `http://127.0.0.1:10810/ai/api/v1/forecast`，并保留 naive / seasonal naive 作为基线。

## 配置

- 服务：`http://127.0.0.1:10810`
- context / horizon / season：`168 / 24 / 24`
- 每个能力每个难度样本数：`12`
- batch size：`6`
- 能力维度：`hierarchical_coherence`
- required target / covariate dim：`3 / 0`
- requested 模型：Timer-3.5, Timer-3.0, Chronos-2, moirai2, toto2.0, timesfm2.5
- 参评 active 模型：toto2.0
- 跳过模型：Timer-3.5 (target_dim_unsupported), Timer-3.0 (target_dim_unsupported), Chronos-2 (target_dim_unsupported), moirai2 (target_dim_unsupported), timesfm2.5 (target_dim_unsupported)
- runtime 输出：`runtime/research/synthetic-v2-hierarchical-coherence-experiment`

## 模型运行状态

- `toto2.0`: `succeeded`, failed=0, elapsed=5.673s

## 结果汇总

### `hierarchical_coherence`

| Model | Fail | MAE d1 | MAE d3 | MAE d5 | MASE d1 | MASE d3 | MASE d5 | MAE d5 / SNaive d5 |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| naive | 0 | 0.7938 | 0.7826 | 0.8994 | 3.6964 | 3.5829 | 3.929 | - |
| seasonal_naive | 0 | 0.6721 | 0.7004 | 0.7349 | 3.1253 | 3.2099 | 3.1987 | - |
| toto2.0 | 0 | 0.1165 | 0.1631 | 0.2149 | 0.5394 | 0.7475 | 0.935 | 0.2924 |


## 初步观察

- `hierarchical_coherence`：平均 MAE 最低的是 `toto2.0`（0.1609）。
- 本轮跳过模型：`Timer-3.5`（target_dim_unsupported）, `Timer-3.0`（target_dim_unsupported）, `Chronos-2`（target_dim_unsupported）, `moirai2`（target_dim_unsupported）, `timesfm2.5`（target_dim_unsupported）。
- 这份结果用于观察真实模型响应，不替代后续更大样本、多随机种子和更多能力维度的论文主实验。

## 复现

```bash
cd backend && PYTHONPATH=.:../scripts uv run python ../scripts/run_synthetic_v2_real_model_experiment.py --models Timer-3.5 Timer-3.0 Chronos-2 moirai2 toto2.0 timesfm2.5 --capabilities hierarchical_coherence --sample-count 12 --batch-size 6
```
