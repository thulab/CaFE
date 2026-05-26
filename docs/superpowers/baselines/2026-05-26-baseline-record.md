# TSBenchmark 基线记录（Track A · plan A0）

- **生成时间**：2026-05-25T16:32:58.741263+00:00
- **数据**：`templates/hourly_trend.csv`，列 = ['time', 'target', 'temperature']
- **后端**：`http://127.0.0.1:18900`（隔离 runtime + 进程内确定性桩适配器 `TSBENCHMARK_MODEL_ADAPTER=stub`）
- **切分**：{'context_length': 12, 'horizon': 6, 'stride': 6, 'target_columns': ['target']}
- **Shard**：value_columns=['target', 'temperature']，target_columns=['target']，sample_count=38（序列真值存 SQLite SeriesPoint）

> 通路：CSV/TsFile 输入 → 全列摄入 → SQLite SeriesPoint(逐点行) → 指针切片 → ModelInput(无 target_future) → 桩推理 → MSE/MAE/MASE → 榜单。
> **桩确定性**：同一次 load 内，相同 (model, sample, seed) 必得相同 forecast；但 `sample_id` 为随机 UUID，故**绝对预测值不跨重载可比**（既有特性）。

## 榜单 · mase （默认主排名）

| rank | model | value |
| --- | --- | --- |
| 1 | Timer 3.5 | 1.826004 |
| 2 | Timer 3.0 | 1.832259 |

## 榜单 · mse

| rank | model | value |
| --- | --- | --- |
| 1 | Timer 3.5 | 5.710683 |
| 2 | Timer 3.0 | 5.753713 |

## 榜单 · mae

| rank | model | value |
| --- | --- | --- |
| 1 | Timer 3.5 | 2.016720 |
| 2 | Timer 3.0 | 2.023537 |

## 样本预测视图（首个样本）

- sample_id：`bf959ce2-2dc7-4962-ab6c-bdbdbe25757d`
- 真值 target_future：['105.0706', '105.0902', '106.8216', '107.5574', '107.9993', '108.3720']

| model | forecast (h=1..n) | mase | mse | mae |
| --- | --- | --- | --- | --- |
| Timer 3.5 | ['105.4894', '105.5475', '105.5611', '105.4883', '105.5021', '105.4927'] | 1.318091 | 3.463479 | 1.597023 |
| Timer 3.0 | ['105.4999', '105.5156', '105.5551', '105.4820', '105.5198', '105.5178'] | 1.310948 | 3.428446 | 1.588368 |

---

> 由 `scripts/baseline-run.sh` 自动生成；重跑即覆盖。重构前后对照见 plan 的「实现状态」节。
