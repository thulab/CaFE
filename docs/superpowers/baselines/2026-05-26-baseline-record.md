# TSBenchmark 基线记录（Track A · plan A0）

- **生成时间**：2026-05-26T02:07:06.123288+00:00
- **数据**：`/Users/zhanghongyin/code/python/TSBenchmark/test/flow_template.csv`，列 = ['time', 'target', 'extra']
- **后端**：`http://127.0.0.1:18900`（隔离 runtime + 进程内确定性桩适配器 `TSBENCHMARK_MODEL_ADAPTER=stub`）
- **切分**：{'context_length': 12, 'horizon': 6, 'stride': 6, 'target_columns': ['target']}
- **Shard**：value_columns=['target', 'extra']，target_columns=['target']，sample_count=3（序列真值存 SQLite SeriesPoint）

> 通路：CSV/TsFile 输入 → 全列摄入 → SQLite SeriesPoint(逐点行) → 指针切片 → ModelInput(无 target_future) → 桩推理 → MSE/MAE/MASE → 榜单。
> **桩确定性**：同一次 load 内，相同 (model, sample, seed) 必得相同 forecast；但 `sample_id` 为随机 UUID，故**绝对预测值不跨重载可比**（既有特性）。

## 榜单 · mase （默认主排名）

| rank | model | value |
| --- | --- | --- |
| 1 | Timer 3.0 | 3.536933 |
| 2 | Timer 3.5 | 3.538125 |

## 榜单 · mse

| rank | model | value |
| --- | --- | --- |
| 1 | Timer 3.0 | 24.137166 |
| 2 | Timer 3.5 | 24.183551 |

## 榜单 · mae

| rank | model | value |
| --- | --- | --- |
| 1 | Timer 3.0 | 4.412555 |
| 2 | Timer 3.5 | 4.413949 |

## 样本预测视图（首个样本）

- sample_id：`70817d47-1347-4a82-a5f7-de1fb4631e5d`
- 真值 target_future：['115.2000', '116.1000', '117.5000', '118.7000', '119.8000', '121.2000']

| model | forecast (h=1..n) | mase | mse | mae |
| --- | --- | --- | --- | --- |
| Timer 3.5 | ['113.8943', '113.8747', '113.8523', '113.8775', '113.9082', '113.8826'] | 3.325106 | 21.912683 | 4.201725 |
| Timer 3.0 | ['113.9029', '113.8585', '113.8633', '113.8928', '113.8458', '113.8822'] | 3.330941 | 22.007455 | 4.209099 |

---

> 由 `scripts/baseline-run.sh` 自动生成；重跑即覆盖。重构前后对照见 plan 的「实现状态」节。
