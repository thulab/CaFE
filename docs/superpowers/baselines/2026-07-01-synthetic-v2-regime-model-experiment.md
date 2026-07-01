# Synthetic v2 真实模型响应实验

日期：2026-07-01

## 目的

用本机 `timer-rest-service` 的真实模型验证 synthetic v2 `regime_switching` probe 是否能呈现模型能力差异。实验直接调用 `http://127.0.0.1:10810/ai/api/v1/forecast`，并保留 naive / seasonal naive 作为基线。

## 配置

- 服务：`http://127.0.0.1:10810`
- context / horizon / season：`168 / 24 / 24`
- 每个能力每个难度样本数：`12`
- batch size：`6`
- 能力维度：`regime_switching`
- requested 模型：Timer-3.5, Timer-3.0, Chronos-2, moirai2, toto2.0, timesfm2.5
- 参评 active 模型：Timer-3.5, Timer-3.0, Chronos-2, moirai2, toto2.0, timesfm2.5
- 跳过模型：none
- runtime 输出：`runtime/research/synthetic-v2-regime-model-experiment`

## 模型运行状态

- `Timer-3.5`: `failed`, failed=60, elapsed=0.311s, error=http://127.0.0.1:10810/ai/api/v1/forecast returned 500: Inference produced no result for task 0 on model 'Timer-3.5' (the model worker failed or returned an error).
- `Timer-3.0`: `failed`, failed=60, elapsed=0.29s, error=http://127.0.0.1:10810/ai/api/v1/forecast returned 500: Inference produced no result for task 0 on model 'Timer-3.0' (the model worker failed or returned an error).
- `Chronos-2`: `succeeded`, failed=0, elapsed=1.98s
- `moirai2`: `succeeded`, failed=0, elapsed=1.537s
- `toto2.0`: `succeeded`, failed=0, elapsed=5.336s
- `timesfm2.5`: `succeeded`, failed=0, elapsed=27.396s

## 结果汇总

### `regime_switching`

| Model | Fail | MAE d1 | MAE d3 | MAE d5 | MASE d1 | MASE d3 | MASE d5 | MAE d5 / SNaive d5 |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| naive | 0 | 1.4687 | 1.0929 | 1.5035 | 5.975 | 3.6103 | 4.2798 | - |
| seasonal_naive | 0 | 1.5278 | 1.1549 | 1.5785 | 6.1776 | 3.7663 | 4.4143 | - |
| Timer-3.5 | 60 | - | - | - | - | - | - | - |
| Timer-3.0 | 60 | - | - | - | - | - | - | - |
| Chronos-2 | 0 | 1.3614 | 0.939 | 1.3941 | 5.4436 | 3.0328 | 3.996 | 0.8832 |
| moirai2 | 0 | 1.3648 | 0.9288 | 1.2995 | 5.276 | 2.9745 | 3.7774 | 0.8233 |
| toto2.0 | 0 | 1.2891 | 0.9571 | 1.4143 | 5.0248 | 3.1512 | 4.1461 | 0.896 |
| timesfm2.5 | 0 | 1.3988 | 0.9891 | 1.3888 | 5.5999 | 3.2113 | 4.0448 | 0.8798 |


## 初步观察

- `regime_switching`：平均 MAE 最低的是 `moirai2`（1.2787）。
- `Timer-3.5`, `Timer-3.0` 本轮所有样本均返回失败，暂不纳入能力排序；需要先排查对应推理 worker。
- 这份结果用于观察真实模型响应，不替代后续更大样本、多随机种子和更多能力维度的论文主实验。

## 复现

```bash
cd backend && PYTHONPATH=.:../scripts uv run python ../scripts/run_synthetic_v2_real_model_experiment.py --models Timer-3.5 Timer-3.0 Chronos-2 moirai2 toto2.0 timesfm2.5 --capabilities regime_switching --sample-count 12 --batch-size 6
```
