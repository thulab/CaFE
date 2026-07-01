# Synthetic v2 真实模型响应实验

日期：2026-07-01

## 目的

用本机 `timer-rest-service` 的真实模型验证 synthetic v2 `trend` / `multi_seasonal` probe 是否能呈现模型能力差异。实验直接调用 `http://127.0.0.1:10810/ai/api/v1/forecast`，并保留 naive / seasonal naive 作为基线。

## 配置

- 服务：`http://127.0.0.1:10810`
- context / horizon / season：`168 / 24 / 24`
- 每个能力每个难度样本数：`12`
- batch size：`6`
- requested 模型：Timer-3.5, Timer-3.0, Chronos-2, timesfm2.5
- 参评 active 模型：Timer-3.5, Timer-3.0, Chronos-2
- 跳过模型：timesfm2.5 (inactive)
- runtime 输出：`runtime/research/synthetic-v2-real-model-experiment`

## 模型运行状态

- `Timer-3.5`: `succeeded`, failed=0, elapsed=33.967s
- `Timer-3.0`: `succeeded`, failed=0, elapsed=20.864s
- `Chronos-2`: `succeeded`, failed=0, elapsed=14.794s

## 结果汇总

### `trend`

| Model | Fail | MAE d1 | MAE d3 | MAE d5 | MASE d1 | MASE d3 | MASE d5 | MAE d5 / SNaive d5 |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| naive | 0 | 0.9327 | 0.8316 | 0.8593 | 2.1433 | 2.0557 | 2.3645 | - |
| seasonal_naive | 0 | 0.4991 | 0.4779 | 0.4641 | 1.1457 | 1.1811 | 1.2787 | - |
| Timer-3.5 | 0 | 0.3286 | 0.3866 | 0.468 | 0.7557 | 0.9511 | 1.2737 | 1.0083 |
| Timer-3.0 | 0 | 0.3249 | 0.3738 | 0.4347 | 0.7467 | 0.9207 | 1.1886 | 0.9366 |
| Chronos-2 | 0 | 0.3126 | 0.3248 | 0.3746 | 0.7199 | 0.8018 | 1.0277 | 0.8071 |

### `multi_seasonal`

| Model | Fail | MAE d1 | MAE d3 | MAE d5 | MASE d1 | MASE d3 | MASE d5 | MAE d5 / SNaive d5 |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| naive | 0 | 1.1247 | 1.021 | 1.0905 | 4.2286 | 4.1555 | 4.7452 | - |
| seasonal_naive | 0 | 0.1549 | 0.6126 | 0.9894 | 0.5833 | 2.5044 | 4.3119 | - |
| Timer-3.5 | 0 | 0.1122 | 0.1087 | 0.094 | 0.4225 | 0.4424 | 0.4087 | 0.095 |
| Timer-3.0 | 0 | 0.1129 | 0.1274 | 0.096 | 0.4257 | 0.5182 | 0.4166 | 0.097 |
| Chronos-2 | 0 | 0.1125 | 0.1084 | 0.0902 | 0.4235 | 0.4434 | 0.3919 | 0.0912 |


## 初步观察

- `trend`：平均 MAE 最低的是 `Chronos-2`（0.3433）。
- `multi_seasonal`：平均 MAE 最低的是 `Chronos-2`（0.1062）。
- `timesfm2.5` 当前在服务模型列表中是 inactive，本轮未能比较 TimesFM；后续需要先在推理服务侧启用它。
- 这份结果用于观察真实模型响应，不替代后续更大样本、多随机种子和更多能力维度的论文主实验。

## 复现

```bash
cd backend && PYTHONPATH=.:../scripts uv run python ../scripts/run_synthetic_v2_real_model_experiment.py --models Timer-3.5 Timer-3.0 Chronos-2 timesfm2.5 --sample-count 12 --batch-size 6
```
