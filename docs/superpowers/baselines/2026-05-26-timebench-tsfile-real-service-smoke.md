# TimeBench TsFile 真实服务 smoke 记录

- 日期：2026-05-26
- 输入：`/nfs/timeBench/TimeBench_TsFile/lotsa/BEIJING_SUBWAY_30MIN/part_0.tsfile`
- 表模型 series：`beijing_subway_30min.0_0.value`
- 读取方式：`TsFileDataFrame` 表模型，按完整 series path 从多 `timeseries_id` 分片中选择单设备序列
- 切分：`context_length=48`，`horizon=12`，`stride=240`，`max_samples=4`
- 模型服务：`http://127.0.0.1:10810`，`model_adapter=rest`
- 模型：`toto2.0`（真实服务当前已 loaded）

## 结果

| 项目 | 值 |
| --- | --- |
| load job | succeeded |
| row_count | 1,572 |
| sample_count | 4 |
| inferred frequency | `30m` |
| run status | succeeded |
| completed samples | 4 / 4 |
| failed samples | 0 |

## 榜单 · MASE

| rank | model | value |
| --- | --- | --- |
| 1 | `toto2.0` | 1497.331698 |

## 诊断指标

| model | MASE | MSE | MAE |
| --- | ---: | ---: | ---: |
| `toto2.0` | 1497.331698 | 0.330865 | 0.511559 |

备注：MASE 很大主要说明该样本的 naive scaling denominator 很小；本次记录重点是验证 TimeBench 多序列表模型 TsFile 可以直接选单条 series 进入 TSBenchmark 的 load → run → ranking 链路。
