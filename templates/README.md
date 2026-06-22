# 模板数据集

可直接上传到 TSBenchmark 试跑的示例 CSV（合成、确定性）。每份符合 CSV 输入契约：
第一列 `time` 为时间列（严格递增、等间隔），其余为数值列（至少包含一个可选目标列），
全部为有限 float。已用真实 `CsvDatasetReader` 校验通过。

> 加载时在 `target_columns` 里选**恰好一个**目标列。当前 MVP 只摄入这个目标列；
> CSV 中的其他数值列可保留作数据说明或未来协变量实验，但不会写入切片。

| 文件 | 频率 | 行数 | 列 | 形态 | 建议 target | 建议 context / horizon / stride |
| --- | --- | --- | --- | --- | --- | --- |
| `hourly_trend.csv` | 1h | 240 | `time,target,temperature` | 线性上升趋势 + 噪声 | `target` | 48 / 24 / 24 |
| `hourly_daily_seasonality.csv` | 1h | 336 | `time,target,temperature,load` | 日内 24h 季节性 + 缓趋势 | `target` | 72 / 24 / 24 |
| `daily_weekly_seasonality.csv` | 1d | 168 | `time,sales,is_weekend` | 周内 7d 季节性 + 趋势（零售口径） | `sales` | 28 / 7 / 7 |
| `multivariate_hourly.csv` | 1h | 240 | `time,target,cov_a,cov_b,cov_c` | 含额外数值列，当前只选单目标 | `target` | 48 / 24 / 24 |

约束提醒（见 `docs/developer/data-model.md §10`）：
- **单序列 / 单设备**：一个 CSV = 一条序列，时间轴严格递增不重复。
- **等间隔**：仅支持固定间隔（hourly/daily/weekly）。月/季/年（天数不齐）会被
  `csv_time_not_equidistant` 拒——模板因此不含月度数据。
- MASE 季节项按 `m=1`（last-value naive）。

## 重新生成

确定性脚本（stdlib、固定随机种子，复跑产出一致）：

```bash
python3 scripts/generate_template_data.py            # 输出到 ./templates
python3 scripts/generate_template_data.py <out_dir>  # 自定义目录
```

要新增形态/频率，在 `scripts/generate_template_data.py` 加一个 builder 并登记到 `main()`。
