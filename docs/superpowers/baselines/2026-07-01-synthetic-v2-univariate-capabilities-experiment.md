# Synthetic v2 真实模型响应实验

日期：2026-07-01

## 目的

用本机 `timer-rest-service` 的真实模型验证 synthetic v2 `trend` / `multi_seasonal` / `regime_switching` / `long_memory_nonlinear` / `intermittent_heteroskedastic` probe 是否能呈现模型能力差异。实验直接调用 `http://127.0.0.1:10810/ai/api/v1/forecast`，并保留 naive / seasonal naive 作为基线。

## 配置

- 服务：`http://127.0.0.1:10810`
- context / horizon / season：`168 / 24 / 24`
- 每个能力每个难度样本数：`12`
- batch size：`6`
- 能力维度：`trend` / `multi_seasonal` / `regime_switching` / `long_memory_nonlinear` / `intermittent_heteroskedastic`
- required target / covariate dim：`1 / 0`
- requested 模型：Timer-3.5, Timer-3.0, Chronos-2, moirai2, toto2.0, timesfm2.5
- 参评 active 模型：Timer-3.5, Timer-3.0, Chronos-2, moirai2, toto2.0, timesfm2.5
- 跳过模型：none
- runtime 输出：`runtime/research/synthetic-v2-univariate-capabilities-experiment`

## 模型运行状态

- `Timer-3.5`: `succeeded`, failed=0, elapsed=54.768s
- `Timer-3.0`: `succeeded`, failed=0, elapsed=21.639s
- `Chronos-2`: `succeeded`, failed=0, elapsed=9.801s
- `moirai2`: `succeeded`, failed=0, elapsed=8.692s
- `toto2.0`: `succeeded`, failed=0, elapsed=27.035s
- `timesfm2.5`: `succeeded`, failed=0, elapsed=133.275s

## 结果汇总

### `trend`

| Model | Fail | MAE d1 | MAE d3 | MAE d5 | MASE d1 | MASE d3 | MASE d5 | MAE d5 / SNaive d5 |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| naive | 0 | 0.9327 | 0.8316 | 0.8593 | 2.1433 | 2.0557 | 2.3645 | - |
| seasonal_naive | 0 | 0.4991 | 0.4779 | 0.4641 | 1.1457 | 1.1811 | 1.2787 | - |
| Timer-3.5 | 0 | 0.3287 | 0.3867 | 0.4682 | 0.7559 | 0.9516 | 1.2742 | 1.0087 |
| Timer-3.0 | 0 | 0.3231 | 0.3721 | 0.4437 | 0.7422 | 0.9165 | 1.2121 | 0.9559 |
| Chronos-2 | 0 | 0.3126 | 0.3248 | 0.3746 | 0.7199 | 0.8018 | 1.0277 | 0.8071 |
| moirai2 | 0 | 0.3754 | 0.3888 | 0.4071 | 0.8634 | 0.9586 | 1.1154 | 0.8771 |
| toto2.0 | 0 | 0.3138 | 0.3349 | 0.3874 | 0.7217 | 0.8295 | 1.0619 | 0.8347 |
| timesfm2.5 | 0 | 0.3162 | 0.3445 | 0.386 | 0.728 | 0.8501 | 1.0503 | 0.8317 |

### `multi_seasonal`

| Model | Fail | MAE d1 | MAE d3 | MAE d5 | MASE d1 | MASE d3 | MASE d5 | MAE d5 / SNaive d5 |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| naive | 0 | 1.1247 | 1.021 | 1.0905 | 4.2286 | 4.1555 | 4.7452 | - |
| seasonal_naive | 0 | 0.1549 | 0.6126 | 0.9894 | 0.5833 | 2.5044 | 4.3119 | - |
| Timer-3.5 | 0 | 0.1124 | 0.1085 | 0.0937 | 0.4232 | 0.4417 | 0.4076 | 0.0947 |
| Timer-3.0 | 0 | 0.1108 | 0.124 | 0.0948 | 0.4173 | 0.5048 | 0.4117 | 0.0958 |
| Chronos-2 | 0 | 0.1125 | 0.1084 | 0.0902 | 0.4235 | 0.4434 | 0.3919 | 0.0912 |
| moirai2 | 0 | 0.1341 | 0.2768 | 0.1093 | 0.5033 | 1.1263 | 0.4739 | 0.1104 |
| toto2.0 | 0 | 0.1048 | 0.1052 | 0.0834 | 0.3949 | 0.4299 | 0.3617 | 0.0843 |
| timesfm2.5 | 0 | 0.1103 | 0.1182 | 0.0877 | 0.4161 | 0.4823 | 0.3811 | 0.0886 |

### `regime_switching`

| Model | Fail | MAE d1 | MAE d3 | MAE d5 | MASE d1 | MASE d3 | MASE d5 | MAE d5 / SNaive d5 |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| naive | 0 | 1.4687 | 1.0929 | 1.5035 | 5.975 | 3.6103 | 4.2798 | - |
| seasonal_naive | 0 | 1.5278 | 1.1549 | 1.5785 | 6.1776 | 3.7663 | 4.4143 | - |
| Timer-3.5 | 0 | 1.367 | 0.9784 | 1.3905 | 5.3809 | 3.1812 | 3.9835 | 0.8809 |
| Timer-3.0 | 0 | 1.3245 | 0.8715 | 1.288 | 5.172 | 2.8017 | 3.6787 | 0.816 |
| Chronos-2 | 0 | 1.3614 | 0.939 | 1.3941 | 5.4436 | 3.0328 | 3.996 | 0.8832 |
| moirai2 | 0 | 1.3648 | 0.9288 | 1.2995 | 5.276 | 2.9745 | 3.7774 | 0.8233 |
| toto2.0 | 0 | 1.2891 | 0.9571 | 1.4143 | 5.0248 | 3.1512 | 4.1461 | 0.896 |
| timesfm2.5 | 0 | 1.3988 | 0.9891 | 1.3888 | 5.5999 | 3.2113 | 4.0448 | 0.8798 |

### `long_memory_nonlinear`

| Model | Fail | MAE d1 | MAE d3 | MAE d5 | MASE d1 | MASE d3 | MASE d5 | MAE d5 / SNaive d5 |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| naive | 0 | 1.5715 | 1.1172 | 0.7469 | 6.9754 | 8.3011 | 2.2576 | - |
| seasonal_naive | 0 | 0.9254 | 0.7586 | 0.5138 | 4.128 | 5.7181 | 1.5236 | - |
| Timer-3.5 | 0 | 0.2831 | 0.5159 | 0.3817 | 1.2514 | 3.8228 | 1.174 | 0.7428 |
| Timer-3.0 | 0 | 0.4713 | 0.5056 | 0.4079 | 2.0979 | 3.716 | 1.2628 | 0.7939 |
| Chronos-2 | 0 | 0.3035 | 0.5461 | 0.3524 | 1.3414 | 4.0233 | 1.076 | 0.6859 |
| moirai2 | 0 | 0.7185 | 0.5998 | 0.4036 | 3.1994 | 4.4025 | 1.2452 | 0.7854 |
| toto2.0 | 0 | 0.4343 | 0.5947 | 0.3651 | 1.921 | 4.4084 | 1.0951 | 0.7105 |
| timesfm2.5 | 0 | 0.2782 | 0.6268 | 0.4209 | 1.2307 | 4.683 | 1.3223 | 0.8192 |

### `intermittent_heteroskedastic`

| Model | Fail | MAE d1 | MAE d3 | MAE d5 | MASE d1 | MASE d3 | MASE d5 | MAE d5 / SNaive d5 |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| naive | 0 | 0.7274 | 0.8851 | 0.8848 | 1.4473 | 1.2878 | 1.1671 | - |
| seasonal_naive | 0 | 0.5796 | 0.7062 | 0.7316 | 1.1425 | 1.0301 | 0.9709 | - |
| Timer-3.5 | 0 | 0.5084 | 0.4248 | 0.4814 | 1.0073 | 0.6159 | 0.6403 | 0.6581 |
| Timer-3.0 | 0 | 0.5048 | 0.4507 | 0.5834 | 0.9997 | 0.6533 | 0.7719 | 0.7974 |
| Chronos-2 | 0 | 0.4485 | 0.4086 | 0.4749 | 0.8863 | 0.5919 | 0.6315 | 0.6491 |
| moirai2 | 0 | 0.5409 | 0.4383 | 0.4882 | 1.0697 | 0.6349 | 0.6497 | 0.6673 |
| toto2.0 | 0 | 0.4493 | 0.4091 | 0.4825 | 0.8875 | 0.5925 | 0.6422 | 0.6596 |
| timesfm2.5 | 0 | 0.5461 | 0.4282 | 0.4837 | 1.0816 | 0.6203 | 0.6433 | 0.6611 |


## 初步观察

- `trend`：平均 MAE 最低的是 `Chronos-2`（0.3433）。
- `multi_seasonal`：平均 MAE 最低的是 `toto2.0`（0.1008）。
- `regime_switching`：平均 MAE 最低的是 `Timer-3.0`（1.2165）。
- `long_memory_nonlinear`：平均 MAE 最低的是 `Timer-3.5`（0.4122）。
- `intermittent_heteroskedastic`：平均 MAE 最低的是 `Chronos-2`（0.4461）。
- 这份结果用于观察真实模型响应，不替代后续更大样本、多随机种子和更多能力维度的论文主实验。

## 复现

```bash
cd backend && PYTHONPATH=.:../scripts uv run python ../scripts/run_synthetic_v2_real_model_experiment.py --models Timer-3.5 Timer-3.0 Chronos-2 moirai2 toto2.0 timesfm2.5 --capabilities trend multi_seasonal regime_switching long_memory_nonlinear intermittent_heteroskedastic --sample-count 12 --batch-size 6
```
