# TSBenchmark MVP 实体结构设计

**日期：** 2026-05-16

**输入材料：**

- `docs/superpowers/specs/2026-05-15-tsbenchmark-platform-functional-definition-design.md`
- `README.md`

**设计范围：** 本文档设计合并后的 `MVP` 实体结构。该 `MVP` 必须支持完整最小评测闭环：

```text
Dataset Manifest
-> Dataset Load Job
-> Shard(real)
-> Sample Index / materialized sample
-> Capability Block(真实)
-> Track
-> Model
-> Benchmarking Run
-> Unit
-> Task
-> Forecast Artifact
-> Metric Result
-> Report
-> Ranking List
-> Sample Forecast
```

## 1. 已确认设计决策

### 1.1 榜单指标与策略

- 默认榜单指标不固定，由创建 `Track` 时选择 `primary_metric_id`。
- `RankingList` 每条 `Track` 一个。
- 榜单支持 `latest_valid_result` 和 `best_result` 两种策略。
- 默认策略是 `latest_valid_result`。
- `metric` 和 `policy` 作为榜单查询或视图参数，不为每个 metric/policy 组合创建独立 `RankingList`。

### 1.2 真实数据集与 Shard

- `DatasetManifest` 与 `Shard(real)` 在 `MVP` 中是一对一。
- 同一个 `DatasetManifest` 不允许重新加载。
- 如果列配置、切分参数、文件内容或校验规则需要变化，必须创建新的 `DatasetManifest`。
- `DatasetLoadJob` 对同一个 `DatasetManifest` 最多产生一个成功的 `Shard(real)`。

### 1.3 Capability Block 与 Shard

- `CapabilityBlock(真实)` 与 `Shard(real)` 是一对多。
- 一个 `CapabilityBlock(block_type = real)` 可以包含多个 `Shard(shard_type = real)`。
- 每个 `Shard(real)` 只属于一个 `CapabilityBlock(真实)`。
- 多个 `Shard(real)` 不简单合并后一次性算指标，而是每个 shard 单独评测，最后对 shard 指标取平均，形成 `CapabilityBlock` / `Task` 级指标。

### 1.4 Track 与 Capability Block

- `Track` 与 `CapabilityBlock` 是一对多。
- 一条 `Track` 可以包含多个 `CapabilityBlock`。
- 每个 `CapabilityBlock` 只属于一条 `Track`。
- `Track` 不能直接引用 `DatasetManifest` 或原始数据文件，只能通过 `CapabilityBlock` 间接使用 `Shard`。

### 1.5 MetricResult 粒度

- 使用统一 `MetricResult` 实体。
- 通过 `result_level` 区分 `sample`、`shard`、`task`、`unit`、`run`、`ranking`。
- `MVP` 保存 sample/task/unit 全部粒度的 MSE 和 MAE。
- shard 级指标用于支持多个 `Shard(real)` 的先独立评测、再平均聚合。

### 1.6 Forecast 与 Sample 存储

- forecast 数组存文件或对象存储。
- `ForecastArtifact` 只保存 `storage_uri`、schema、checksum 和关联 ID。
- `MVP` 先物化每条 sample 的 `target_history` / `target_future` 数组产物。
- `SampleIndex` 仍记录切片位置和 `storage_ref`，并增加 `materialized_sample_uri`。
- 后续数据量变大时，可以保留 `SampleIndex`，关闭 sample 物化，改为按需读取。

### 1.7 Report 与 Model Stub

- `Report` 只做基础报告：模型指标表、task 摘要、sample forecast 链接。
- `MVP` 的 5 个模型统一通过同一个 `Timer/model service` 接入。
- 当前阶段可以用 stub 模拟外部 API 服务。
- stub 输出必须可复现：同一个 `model_id + sample_id + seed` 必须生成同一个 forecast。

## 2. 设计原则

### 2.1 逻辑实体稳定，物理存储可替换

实体结构先定义稳定领域对象，不绑定具体数据库、ORM 或文件格式。后续可以用关系数据库、文档数据库、文件系统、IoTDB、TsFile 或对象存储实现，但不能改变核心领域关系：

```text
真实数据集 -> Shard(real) -> Capability Block(真实) -> Track
```

### 2.2 用类型字段兼容后续扩展

不要为 `Shard(real)` 或 `Capability Block(真实)` 创建只适用于真实数据的特殊实体类。代码中应使用统一实体和类型字段：

- `Shard.shard_type = real`
- `CapabilityBlock.block_type = real`
- `Track.track_type = real_dataset`

后续接入 synthetic 数据时扩展为：

- `Shard.shard_type = synthetic`
- `CapabilityBlock.block_type = generated`
- `Track.track_type = generated_benchmark` 或 `mixed`

### 2.3 MVP 使用一对多外键，后续再升级复用关系

`MVP` 已明确：

```text
Track 1 -> N CapabilityBlock
CapabilityBlock 1 -> N Shard
```

因此 `MVP` 不引入 `CapabilityBlockShard` 或 `TrackCapabilityBlock` 作为必需实体。后续如果需要跨 track 复用 capability block，或让一个 shard 被多个 capability block 复用，再引入关联实体迁移。

### 2.4 大数组和产物不进入元数据实体

`target_history`、`target_future`、forecast、评测明细和报告文件不直接塞进核心元数据实体。实体只保存 `storage_uri`、`storage_ref`、`materialized_sample_uri`、schema 和校验信息。

### 2.5 DTO 与持久化实体分离

`SamplePreviewDTO`、`SampleForecastDTO`、`RankingRowDTO` 是 API 读模型，不是核心持久化实体。它们由实体和产物引用组合生成。

## 3. MVP 实体总览

### 3.1 核心实体

| Entity | 类型 | MVP 职责 |
| --- | --- | --- |
| `DatasetManifest` | 持久化实体 | 描述真实数据源 |
| `DatasetLoadJob` | 持久化实体 | 记录一次真实数据加载、校验、切分和索引任务 |
| `Shard` | 持久化实体 | 统一数据单元；`MVP` 中使用 `shard_type = real` |
| `SampleIndex` | 持久化实体 | 记录 sample 切片位置、物化 sample 产物引用和读取引用 |
| `CapabilityBlock` | 持久化实体 | 统一能力块；`MVP` 中使用 `block_type = real` |
| `Track` | 持久化实体 | 评测赛道；`MVP` 中使用 `track_type = real_dataset` |
| `Model` | 持久化实体 | 描述可评测模型和 `Timer/model service` adapter |
| `BenchmarkingRun` | 持久化实体 | 一次评测执行 |
| `Unit` | 持久化实体 | 某模型在一次 run 中的完整结果 |
| `Task` | 持久化实体 | 某模型在某个 `CapabilityBlock` 上的结果 |
| `ForecastArtifact` | 持久化实体 | 记录预测结果产物位置和 schema |
| `MetricDefinition` | 持久化实体 | 指标定义；`MVP` 使用 MSE 和 MAE |
| `MetricResult` | 持久化实体 | 统一指标结果，支持 sample/shard/task/unit/run/ranking |
| `Report` | 持久化实体 | 基础评测报告产物 |
| `RankingList` | 持久化实体 | 某条 `Track` 的榜单定义 |
| `RankingEntry` | 持久化实体 | 榜单中的模型成绩行 |
| `RunEvent` | 持久化实体 | run/task 过程日志、错误和状态事件 |

### 3.2 派生 DTO

| DTO | 来源 | 用途 |
| --- | --- | --- |
| `SamplePreviewDTO` | `SampleIndex + materialized_sample_uri` | 展示 `target_history` 和 `target_future` |
| `SampleForecastDTO` | `SampleIndex + ForecastArtifact + MetricResult` | 展示 history、ground truth、forecast 和 sample metric |
| `RankingRowDTO` | `RankingEntry + Model + MetricResult` | 榜单页面行数据 |
| `RunProgressDTO` | `BenchmarkingRun + Unit + Task + RunEvent` | run 进度页面 |

## 4. 实体字段设计

### 4.1 DatasetManifest

`DatasetManifest` 描述真实数据源。它不是评测输入单元，不能直接挂到 `Track`。

| 字段 | 说明 |
| --- | --- |
| `dataset_manifest_id` | 主 ID |
| `name` | 数据集展示名称 |
| `domain` | 数据领域 |
| `source_type` | `local_file`、`managed_file`、`remote_uri` 等 |
| `source_uri` | 原始数据位置 |
| `file_format` | `csv`、`parquet`、`tsfile` 等 |
| `time_column` | 时间列 |
| `target_columns` | 目标列列表 |
| `frequency` | 时间频率 |
| `timezone` | 时区 |
| `schema_version` | manifest schema 版本 |
| `status` | `draft`、`ready_to_load`、`loaded`、`disabled` |
| `created_at` / `updated_at` | 创建和更新时间 |

约束：

- `status = loaded` 后不能重新加载。
- 如果要更换文件、列配置或切分参数，创建新的 `DatasetManifest`。

### 4.2 DatasetLoadJob

`DatasetLoadJob` 记录一次真实数据加载过程。

| 字段 | 说明 |
| --- | --- |
| `load_job_id` | 主 ID |
| `dataset_manifest_id` | 来源 manifest |
| `status` | 加载状态 |
| `current_step` | 当前步骤 |
| `validation_summary` | 时间列、target columns、缺失值、频率、长度等校验摘要 |
| `split_config` | `context_length`、`horizon`、`stride` |
| `seed` | sample 物化和 stub 可复现的默认 seed |
| `error_code` / `error_message` | 失败原因 |
| `output_shard_id` | 成功后产生的唯一 `Shard(real)` |
| `started_at` / `finished_at` | 起止时间 |

状态机：

```text
created -> validating -> loading -> materializing_samples -> succeeded
                         |-> failed
                         |-> cancelled
```

约束：

- 同一个 `DatasetManifest` 最多有一个成功的 `DatasetLoadJob`。
- 成功 job 产生且只产生一个 `Shard(real)`。

### 4.3 Shard

`Shard` 是统一数据单元。`MVP` 中真实数据集加载后必须形成 `Shard(shard_type = real)`。

| 字段 | 说明 |
| --- | --- |
| `shard_id` | 主 ID |
| `shard_type` | `MVP` 固定为 `real` |
| `dataset_manifest_id` | 真实数据来源，一对一 |
| `load_job_id` | 产生该 shard 的加载任务 |
| `capability_block_id` | 所属 `CapabilityBlock(真实)` |
| `source_uri` | 原始数据位置 |
| `storage_uri` | 规范化后数据或索引产物位置 |
| `checksum` | 数据校验值 |
| `time_range_start` / `time_range_end` | 时间范围 |
| `row_count` | 行数 |
| `target_columns` | 目标列 |
| `target_dim` | 目标维度 |
| `frequency` | 时间频率 |
| `context_length` | 样本历史长度 |
| `horizon` | 预测长度 |
| `stride` | 样本滑动步长 |
| `sample_count` | 样本数 |
| `status` | `created`、`ready`、`failed`、`disabled` |
| `created_at` / `updated_at` | 创建和更新时间 |

约束：

- 一个 `Shard(real)` 只属于一个 `CapabilityBlock(真实)`。
- 一个 `DatasetManifest` 只对应一个 `Shard(real)`。

### 4.4 SampleIndex

`SampleIndex` 记录样本切片位置和物化 sample 的读取引用。

| 字段 | 说明 |
| --- | --- |
| `sample_id` | 主 ID |
| `shard_id` | 所属 `Shard` |
| `sample_index` | shard 内序号 |
| `context_start` / `context_end` | history 时间窗口 |
| `horizon_start` / `horizon_end` | future 时间窗口 |
| `target_columns` | 当前样本目标列 |
| `context_length` | history 长度 |
| `horizon` | future 长度 |
| `storage_ref` | 从 shard 原始数据读取样本的引用 |
| `materialized` | `MVP` 固定为 `true` |
| `materialized_sample_uri` | 物化 sample 数组产物位置 |
| `checksum` | 物化 sample 校验值 |
| `metadata` | 可选样本摘要 |

约束：

- `MVP` 物化每条 sample 的 `target_history` 和 `target_future`。
- 后续可将 `materialized = false`，改为只用 `storage_ref` 按需读取。

### 4.5 CapabilityBlock

`CapabilityBlock` 是统一能力块。`MVP` 中 `CapabilityBlock(真实)` 由 `block_type = real` 表示。

| 字段 | 说明 |
| --- | --- |
| `capability_block_id` | 主 ID |
| `track_id` | 所属 `Track`，每个 block 只属于一条 track |
| `block_type` | `MVP` 固定为 `real` |
| `capability_type` | `MVP` 可为 `real_data` |
| `name` | 展示名称 |
| `task_type` | `MVP` 为 `univariate_forecast` |
| `target_dim` | 目标维度 |
| `shard_count` | shard 数 |
| `sample_count` | 汇总样本数 |
| `aggregation_policy` | `MVP` 为 `mean_over_shards` |
| `status` | `draft`、`ready`、`disabled` |
| `created_at` / `updated_at` | 创建和更新时间 |

约束：

- 一个 `CapabilityBlock(真实)` 可以包含多个 `Shard(real)`。
- 每个 `Shard(real)` 只属于一个 `CapabilityBlock(真实)`。
- shard 级指标先独立计算，再按 `mean_over_shards` 聚合到 task/unit。

### 4.6 Track

`Track` 是评测赛道。`MVP` 中 `track_type = real_dataset`。

| 字段 | 说明 |
| --- | --- |
| `track_id` | 主 ID |
| `name` | 赛道名称 |
| `track_type` | `MVP` 为 `real_dataset` |
| `description` | 描述 |
| `primary_metric_id` | 创建 track 时选择的默认榜单指标 |
| `default_ranking_policy` | 默认 `latest_valid_result` |
| `benchmark_version` | benchmark 版本 |
| `data_version` | 数据版本 |
| `status` | `draft`、`ready`、`disabled` |
| `created_at` / `updated_at` | 创建和更新时间 |

约束：

- 一条 `Track` 可以包含多个 `CapabilityBlock`。
- 每个 `CapabilityBlock` 只属于一条 `Track`。
- `Track.primary_metric_id` 可选择 MSE 或 MAE。

### 4.7 Model

`Model` 描述可评测模型。`MVP` 中 5 个模型统一通过 `Timer/model service` 接入，当前阶段可用 stub。

| 字段 | 说明 |
| --- | --- |
| `model_id` | 主 ID |
| `name` | 展示名称 |
| `model_family` | Timer、Chronos、toto、TimesFM 等 |
| `model_version` | 模型版本 |
| `adapter_type` | `MVP` 为 `timer_service` |
| `endpoint_uri` | 推理服务地址；stub 可用 `stub://timer-service/{model_name}` |
| `supported_task_types` | `MVP` 至少支持 `univariate_forecast` |
| `input_schema_version` | 输入协议版本 |
| `stub_seed` | stub 可复现输出的 seed |
| `status` | `registered`、`available`、`disabled` |
| `created_at` / `updated_at` | 创建和更新时间 |

约束：

- stub forecast 必须由 `model_id + sample_id + seed` 确定。
- 同一输入必须得到同一 forecast。

### 4.8 BenchmarkingRun

`BenchmarkingRun` 表示一次评测执行。

| 字段 | 说明 |
| --- | --- |
| `benchmarking_run_id` | 主 ID |
| `track_id` | 被评测 `Track` |
| `benchmark_version` | benchmark 版本 |
| `data_version` | 数据版本 |
| `status` | run 状态 |
| `model_count` | 模型数量 |
| `task_count` | task 数量 |
| `sample_count` | 样本数量 |
| `metric_set` | `MVP` 为 `mse, mae` |
| `started_at` / `finished_at` | 起止时间 |
| `created_at` / `updated_at` | 创建和更新时间 |

状态机：

```text
created -> queued -> running -> succeeded
                         |-> partial_succeeded
                         |-> failed
                         |-> cancelled
```

状态语义：

- 全部模型成功：`succeeded`。
- 至少一个模型成功且至少一个模型失败：`partial_succeeded`。
- 全部模型失败：`failed`。
- 用户主动取消：`cancelled`。

### 4.9 Unit

`Unit` 是某模型在某次 run 中的完整结果。

| 字段 | 说明 |
| --- | --- |
| `unit_id` | 主 ID |
| `benchmarking_run_id` | 所属 run |
| `model_id` | 模型 |
| `status` | `created`、`running`、`succeeded`、`failed`、`skipped` |
| `task_count` | task 数 |
| `sample_count` | 样本数 |
| `started_at` / `finished_at` | 起止时间 |

### 4.10 Task

`Task` 是某模型在某个 `CapabilityBlock` 上的结果集合。`MVP` 中通常是某模型在 `CapabilityBlock(真实)` 上的评测。

| 字段 | 说明 |
| --- | --- |
| `task_id` | 主 ID |
| `benchmarking_run_id` | 所属 run |
| `unit_id` | 所属 unit |
| `model_id` | 模型 |
| `capability_block_id` | 能力块 |
| `status` | `created`、`running`、`succeeded`、`failed`、`skipped` |
| `shard_count` | shard 数 |
| `sample_count` | 样本数 |
| `aggregation_policy` | `mean_over_shards` |
| `error_code` / `error_message` | 失败原因 |
| `started_at` / `finished_at` | 起止时间 |

约束：

- 一个 `Task` 可以产生多个 `ForecastArtifact`，通常每个 shard 一个。
- `Task` 聚合指标由 shard 级指标平均得到。

### 4.11 ForecastArtifact

`ForecastArtifact` 记录预测结果位置，不直接保存 forecast 数组。

| 字段 | 说明 |
| --- | --- |
| `forecast_artifact_id` | 主 ID |
| `benchmarking_run_id` | 所属 run |
| `unit_id` | 所属 unit |
| `task_id` | 所属 task |
| `model_id` | 模型 |
| `shard_id` | 数据 shard；每个 shard 可有一个 forecast artifact |
| `storage_uri` | 预测产物位置 |
| `schema_version` | 预测产物 schema 版本 |
| `sample_count` | 覆盖样本数 |
| `checksum` | 产物校验值 |
| `created_at` | 创建时间 |

### 4.12 MetricDefinition

`MetricDefinition` 是指标注册表。`MVP` 只要求 MSE 和 MAE。

| 字段 | 说明 |
| --- | --- |
| `metric_id` | 主 ID |
| `name` | `mse`、`mae` |
| `display_name` | 展示名称 |
| `direction` | `lower_is_better` |
| `supported_levels` | `sample`、`shard`、`task`、`unit`、`run`、`ranking` |
| `status` | `active`、`disabled` |

### 4.13 MetricResult

`MetricResult` 记录指标结果。它通过 `result_level` 表示结果粒度，避免为 sample/task/unit 分别建多套指标表。

| 字段 | 说明 |
| --- | --- |
| `metric_result_id` | 主 ID |
| `metric_id` | 指标定义 |
| `result_level` | `sample`、`shard`、`task`、`unit`、`run`、`ranking` |
| `benchmarking_run_id` | 所属 run |
| `unit_id` | 可选 |
| `task_id` | 可选 |
| `sample_id` | sample 粒度必填 |
| `shard_id` | shard 粒度或 shard 聚合相关结果必填 |
| `model_id` | 模型 |
| `capability_block_id` | 可选 |
| `value` | 指标值 |
| `aggregation` | `raw`、`mean_over_samples`、`mean_over_shards` 等 |
| `created_at` | 创建时间 |

约束：

- `MVP` 保存 sample、shard、task、unit 粒度的 MSE 和 MAE。
- ranking 可读取 unit 或 run 聚合结果生成，不要求额外复制所有指标。

### 4.14 Report

`Report` 是基础评测报告产物。

| 字段 | 说明 |
| --- | --- |
| `report_id` | 主 ID |
| `report_type` | `MVP` 为 `run_summary` |
| `benchmarking_run_id` | 所属 run |
| `track_id` | 所属 track |
| `status` | `created`、`generating`、`ready`、`failed` |
| `storage_uri` | 报告产物位置 |
| `summary` | 基础摘要 |
| `created_at` / `updated_at` | 创建和更新时间 |

`MVP` 报告内容：

- 模型指标表。
- task 摘要。
- sample forecast 链接。

### 4.15 RankingList

`RankingList` 是某条 `Track` 的榜单定义。`MVP` 中每条 `Track` 只有一个 `RankingList`。

| 字段 | 说明 |
| --- | --- |
| `ranking_list_id` | 主 ID |
| `track_id` | 所属 track |
| `default_metric_id` | 默认排序指标，来自 `Track.primary_metric_id` |
| `default_policy` | `latest_valid_result` |
| `supported_policies` | `latest_valid_result`、`best_result` |
| `status` | `active`、`disabled` |
| `updated_at` | 更新时间 |

### 4.16 RankingEntry

`RankingEntry` 是榜单中的模型成绩行。因为同一个 `RankingList` 支持不同 metric 和 policy 视图，所以 entry 必须记录 `metric_id` 与 `policy`。

| 字段 | 说明 |
| --- | --- |
| `ranking_entry_id` | 主 ID |
| `ranking_list_id` | 榜单 |
| `track_id` | 赛道 |
| `metric_id` | 当前视图指标 |
| `policy` | `latest_valid_result` 或 `best_result` |
| `model_id` | 模型 |
| `benchmarking_run_id` | 采用的 run |
| `unit_id` | 对应 unit |
| `metric_value` | 排序指标值 |
| `rank` | 排名 |
| `status` | `active`、`stale` |
| `updated_at` | 更新时间 |

### 4.17 RunEvent

`RunEvent` 用于支持进度、日志和错误追踪。

| 字段 | 说明 |
| --- | --- |
| `run_event_id` | 主 ID |
| `benchmarking_run_id` | 所属 run |
| `unit_id` | 可选 |
| `task_id` | 可选 |
| `level` | `info`、`warning`、`error` |
| `event_type` | `status_changed`、`metric_computed`、`adapter_error` 等 |
| `message` | 文本说明 |
| `payload` | 结构化补充信息 |
| `created_at` | 创建时间 |

## 5. 关系基数

```text
DatasetManifest 1 -> 0..1 DatasetLoadJob(succeeded)
DatasetManifest 1 -> 1 Shard(real)
DatasetLoadJob 1 -> 0..1 Shard(real)

Track 1 -> N CapabilityBlock
CapabilityBlock N -> 1 Track

CapabilityBlock 1 -> N Shard
Shard N -> 1 CapabilityBlock

Shard 1 -> N SampleIndex

Track 1 -> N BenchmarkingRun
BenchmarkingRun 1 -> N Unit
BenchmarkingRun 1 -> N Task
Unit 1 -> N Task
Task 1 -> N ForecastArtifact

MetricDefinition 1 -> N MetricResult
BenchmarkingRun 1 -> N MetricResult
Task 1 -> N MetricResult
Unit 1 -> N MetricResult
Shard 1 -> N MetricResult
SampleIndex 1 -> N MetricResult

BenchmarkingRun 1 -> 0..N Report
Track 1 -> 1 RankingList
RankingList 1 -> N RankingEntry
```

MVP 中的关键约束：

- 一个 `DatasetManifest` 只能成功加载一次。
- 一个 `DatasetManifest` 必须对应唯一 `Shard(real)`。
- 一个 `Shard(real)` 必须归属到唯一 `CapabilityBlock(block_type = real)`。
- 一个 `CapabilityBlock(真实)` 可以包含多个 `Shard(real)`。
- 一个 `CapabilityBlock` 只能属于一条 `Track`。
- `BenchmarkingRun` 只能基于 `Track` 创建。
- `RankingEntry` 必须引用 `BenchmarkingRun` 和 `Unit`，以保证榜单成绩可回溯。

## 6. 运行流程

### 6.1 数据加载流程

```text
Create DatasetManifest
-> Create DatasetLoadJob
-> Validate source data
-> Normalize or index source data
-> Create Shard(shard_type = real)
-> Materialize Sample files
-> Create SampleIndex records
-> Create or reuse CapabilityBlock(block_type = real)
-> Set Shard.capability_block_id
-> Create Track(track_type = real_dataset, primary_metric_id)
-> Set CapabilityBlock.track_id
-> Create RankingList(track_id, default_metric_id, default_policy = latest_valid_result)
```

### 6.2 评测运行流程

```text
Select Track
-> Select Models
-> Create BenchmarkingRun
-> Create Unit per Model
-> Create Task per Unit-CapabilityBlock
-> For each Shard in CapabilityBlock:
     Read materialized Sample files
     Call Timer/model service or deterministic stub
     Save ForecastArtifact
     Compute sample-level MetricResult(MSE, MAE)
     Compute shard-level MetricResult(MSE, MAE)
-> Aggregate shard metrics into task-level MetricResult
-> Aggregate task metrics into unit-level MetricResult
-> Mark Task and Unit status
-> Mark BenchmarkingRun as succeeded / partial_succeeded / failed
-> Generate basic Report
-> Refresh RankingEntry views for latest_valid_result and best_result
```

### 6.3 榜单刷新策略

`MVP` 中每条 `Track` 维护一个 `RankingList`：

- 默认指标来自 `Track.primary_metric_id`。
- 默认策略为 `latest_valid_result`。
- 查询视图支持 `metric=mse|mae`。
- 查询视图支持 `policy=latest_valid_result|best_result`。
- `latest_valid_result`：每个模型采用最新一次成功或部分成功 run 中该模型成功的 `Unit`。
- `best_result`：每个模型采用历史上指定 metric 最优的有效 `Unit`。
- 如果某模型最新 run 失败，不覆盖该模型上一条 latest valid entry。

## 7. API 资源口径

`MVP` 后端 API 可以按资源组织：

```text
/dataset-manifests
/dataset-load-jobs
/shards
/sample-indexes
/capability-blocks
/tracks
/models
/benchmarking-runs
/units
/tasks
/metrics
/reports
/ranking-lists
/samples/{sample_id}/preview
/samples/{sample_id}/forecast
```

读模型 DTO：

- `SamplePreviewDTO`：用于数据加载后预览。
- `SampleForecastDTO`：用于评测后解释模型预测。
- `RunProgressDTO`：用于 run 页面。
- `RankingRowDTO`：用于榜单页面。

## 8. 未来兼容点

该实体结构应兼容以下扩展：

- synthetic 数据：新增 `Shard.shard_type = synthetic`。
- 多变量任务：扩展 `target_dim`、`target_columns` 和模型 `supported_task_types`。
- 协变量任务：使用已有 `history_cov`、`future_cov` schema。
- 更多能力维度：新增 `CapabilityBlock.block_type = generated` 和 `capability_type`。
- 复用关系：后续如果需要跨 track 复用 block 或跨 block 复用 shard，再引入 `TrackCapabilityBlock` / `CapabilityBlockShard` 关联实体。
- Anchor profile：从 `Shard(real)` 派生 `AnchorProfile`，不改变真实数据实体链路。
- 大规模存储：替换 `storage_uri` 背后的实现，不改变实体关系。
- 高级指标：新增 `MetricDefinition`，复用 `MetricResult`。

## 9. 自查

- 真实数据集没有被建模为绕过 `Shard` 的顶层评测入口。
- `DatasetManifest` 与 `Shard(real)` 在 `MVP` 中保持一对一。
- `CapabilityBlock(真实)` 与 `Shard(real)` 是一对多。
- `Track` 与 `CapabilityBlock` 是一对多。
- sample 数组在 `MVP` 中物化，但仍保留 `SampleIndex` 以兼容后续按需读取。
- forecast 数组通过 `ForecastArtifact.storage_uri` 引用，不进入核心元数据实体。
- `MetricResult` 统一承载 sample/shard/task/unit/run/ranking 粒度。
- `BenchmarkingRun` 支持 `partial_succeeded`。
- 榜单每条 `Track` 一个，支持 latest valid 和 best result 两种视图。
- MVP 可以完成真实数据加载、模型评测、报告生成和榜单查看。
