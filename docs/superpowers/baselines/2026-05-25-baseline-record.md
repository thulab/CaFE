# TSBenchmark 基线记录（Track A · plan A0）

- **生成时间**：2026-05-25T13:28:15.829792+00:00
- **数据**：`/Users/zhanghongyin/code/python/TSBenchmark/test/flow_template.csv`，列 = ['time', 'target', 'extra']
- **后端**：`http://127.0.0.1:18900`（隔离 runtime + 进程内确定性桩适配器 `TSBENCHMARK_MODEL_ADAPTER=stub`）
- **切分**：{'context_length': 12, 'horizon': 6, 'stride': 6, 'target_columns': ['target']}
- **Shard**：value_columns=['target', 'extra']，target_columns=['target']，sample_count=3，tsfile=`/Users/zhanghongyin/code/python/TSBenchmark/runtime/baseline/tsfiles/80c8b534-61d2-4944-8cf1-c9238e6db1fe.tsfile`

> 通路：CSV → 全列摄入 → per-dataset TsFile → 指针切片 → ModelInput(无 target_future) → 桩推理 → MSE/MAE/MASE → 榜单。
> **桩确定性**：同一次 load 内，相同 (model, sample, seed) 必得相同 forecast；但 `sample_id` 为随机 UUID，故**绝对预测值不跨重载可比**（既有特性）。

## 榜单 · mase （默认主排名）

| rank | model | value |
| --- | --- | --- |
| 1 | Timer 3.0 | 3.581560 |
| 2 | Timer 3.5 | 3.593766 |

## 榜单 · mse

| rank | model | value |
| --- | --- | --- |
| 1 | Timer 3.0 | 24.609410 |
| 2 | Timer 3.5 | 24.729992 |

## 榜单 · mae

| rank | model | value |
| --- | --- | --- |
| 1 | Timer 3.0 | 4.468202 |
| 2 | Timer 3.5 | 4.483521 |

## 样本预测视图（首个样本）

- sample_id：`97e7dd0f-c2d4-45d1-8d08-eb9c9b947971`
- 真值 target_future：['115.2000', '116.1000', '117.5000', '118.7000', '119.8000', '121.2000']

| model | forecast (h=1..n) | mase | mse | mae |
| --- | --- | --- | --- | --- |
| Timer 3.5 | ['113.7849', '113.8144', '113.8014', '113.8194', '113.7817', '113.8339'] | 3.384986 | 22.534463 | 4.277391 |
| Timer 3.0 | ['113.8050', '113.7872', '113.7928', '113.8105', '113.8255', '113.8146'] | 3.384997 | 22.530779 | 4.277406 |

---

> 由 `scripts/baseline-run.sh` 自动生成；重跑即覆盖。重构前后对照见 plan 的「实现状态」节。
