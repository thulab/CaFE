# TSBenchmark 基线记录（Track A · plan A0）

- **生成时间**：2026-05-25T16:03:11.553523+00:00
- **数据**：`/Users/zhanghongyin/code/python/TSBenchmark/test/flow_template.csv`，列 = ['time', 'target', 'extra']
- **后端**：`http://127.0.0.1:18900`（隔离 runtime + 进程内确定性桩适配器 `TSBENCHMARK_MODEL_ADAPTER=stub`）
- **切分**：{'context_length': 12, 'horizon': 6, 'stride': 6, 'target_columns': ['target']}
- **Shard**：value_columns=['target', 'extra']，target_columns=['target']，sample_count=3（序列真值存 SQLite SeriesPoint）

> 通路：CSV/TsFile 输入 → 全列摄入 → SQLite SeriesPoint(逐点行) → 指针切片 → ModelInput(无 target_future) → 桩推理 → MSE/MAE/MASE → 榜单。
> **桩确定性**：同一次 load 内，相同 (model, sample, seed) 必得相同 forecast；但 `sample_id` 为随机 UUID，故**绝对预测值不跨重载可比**（既有特性）。

## 榜单 · mase （默认主排名）

| rank | model | value |
| --- | --- | --- |
| 1 | Timer 3.5 | 3.478014 |
| 2 | Timer 3.0 | 3.507220 |

## 榜单 · mse

| rank | model | value |
| --- | --- | --- |
| 1 | Timer 3.5 | 23.486199 |
| 2 | Timer 3.0 | 23.859702 |

## 榜单 · mae

| rank | model | value |
| --- | --- | --- |
| 1 | Timer 3.5 | 4.338837 |
| 2 | Timer 3.0 | 4.375369 |

## 样本预测视图（首个样本）

- sample_id：`916105d0-592a-46d2-9e11-1f713f44cf4a`
- 真值 target_future：['115.2000', '116.1000', '117.5000', '118.7000', '119.8000', '121.2000']

| model | forecast (h=1..n) | mase | mse | mae |
| --- | --- | --- | --- | --- |
| Timer 3.5 | ['114.0052', '113.9408', '113.9248', '113.9656', '113.9602', '114.0059'] | 3.257466 | 21.190746 | 4.116253 |
| Timer 3.0 | ['113.9538', '113.9566', '113.9148', '113.8737', '113.8760', '113.9279'] | 3.297003 | 21.711983 | 4.166213 |

---

> 由 `scripts/baseline-run.sh` 自动生成；重跑即覆盖。重构前后对照见 plan 的「实现状态」节。
