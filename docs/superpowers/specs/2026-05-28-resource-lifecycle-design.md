# Resource Lifecycle Design

## Goal

为数据集、切片、赛道和评测运行提供一致的生命周期管理：普通用户默认执行可恢复的归档，管理员可以在看清影响范围后执行不可恢复的物理删除。归档后的历史详情、报告和榜单链接仍可访问；默认工作台和新建流程只展示可继续复用的资源。

## Current State

当前代码没有数据库级外键，关系由 service 层通过 ID 字段维护。核心链路是：

```text
DatasetManifest -> DatasetLoadJob -> Shard -> SeriesPoint / SampleIndex
Shard -> CapabilityBlockShard -> CapabilityBlock -> Track
Track -> BenchmarkingRun -> Unit / Task / Report / RankingEntry / ForecastArtifact / MetricResult
```

`DatasetManifest`、`Shard`、`Track` 已有 `status` 字段，但 `BenchmarkingRun.status` 是执行状态，不能直接改成 `archived`。为避免破坏运行状态，也避免给现有 SQLite runtime DB 追加列，本设计新增一张独立状态表，而不是修改既有实体字段。

## Recommended Model

新增 `ArchivedResource` 表：

| 字段 | 说明 |
| --- | --- |
| `resource_type` | `dataset_manifest`、`shard`、`track`、`benchmarking_run` |
| `resource_id` | 对应实体 ID |
| `archived_at` | 归档时间 |
| `archived_reason` | 可选原因，MVP 前端可不填 |

该表以 `(resource_type, resource_id)` 为复合主键。归档不删除任何业务行，也不修改 run 的执行状态。列表接口默认排除已归档资源；详情接口继续返回资源，并附带 `archived_at`。

## User Semantics

### Archive

归档是默认删除动作：

- 数据集归档：从数据集工作台默认隐藏；其切片不会再默认出现在新建赛道可选列表中。既有赛道、运行、报告仍可访问。
- 切片归档：从切片列表和赛道创建候选中默认隐藏。既有赛道仍保持可复现。
- 赛道归档：从赛道工作台默认隐藏；赛道详情和历史榜单仍可访问；不允许基于已归档赛道创建新的运行。
- 运行归档：从运行列表和赛道详情的运行列表默认隐藏；运行详情和报告链接仍可访问。运行中或排队中的 run 需要先取消，不能直接归档。

列表提供 `include_archived=true` 参数。前端提供“显示已归档”开关，并用状态徽标显示归档。

### Restore

归档资源可以恢复：

- 恢复只删除 `ArchivedResource` 行。
- 恢复赛道后可以再次启动新运行。
- 恢复数据集或切片后可再次被新赛道选择。

### Purge

物理删除是管理员操作：

- 每个资源提供影响预览接口，返回会被删除的实体计数和是否需要级联。
- `DELETE` 默认只允许无下游引用的安全删除。
- `DELETE ?cascade=true` 允许管理员级联删除，但必须由前端二次确认。
- 物理删除不可恢复；删除顺序由 service 层显式维护。

## Impact Rules

### DatasetManifest

影响范围包括该 manifest、其 load jobs、shards、series points、sample indices，以及引用这些 shards 的 tracks 和 runs。若任何 shard 被 track 引用，非级联 purge 必须拒绝。

级联 purge 一个 dataset 时：

1. 找出所有相关 shards。
2. 找出引用这些 shards 的 capability blocks 和 tracks。
3. 先 purge 这些 tracks 下的 runs 和 reports。
4. 删除 rankings、blocks、block-shard links。
5. 删除 shards 的 samples、series points、metric results、forecast artifacts。
6. 删除 load jobs、manifest 和 archive marks。

托管上传文件（`runtime/uploads/` 下的源文件）随 dataset manifest 的物理删除一起删除；非托管路径不自动删除。

### Shard

影响范围包括 shard、series points、sample indices、block links，以及引用它的 tracks/runs。若 shard 被任何 track 引用，非级联 purge 拒绝。

级联 purge shard 与 dataset 类似，但只删除依赖该 shard 的 tracks/runs；不删除同 manifest 下其它 shards。

### Track

影响范围包括 track、capability blocks、ranking list/entries、runs、reports、metric results、forecast artifacts、events。非级联 purge 仅在 track 没有 runs 时允许；有 runs 时需要 `cascade=true`。

归档 track 不删除榜单或运行，但新建运行接口必须拒绝归档 track。

### BenchmarkingRun

影响范围包括 run、units、tasks、forecast artifacts、run events、reports、metric results、ranking entries。运行中或排队中 run 不能 purge；需要先取消并进入终态。非级联 purge run 没有额外下游实体限制，因为这些子表都属于该 run。

## API Surface

每个资源暴露一致操作：

```text
GET    /dataset-manifests/{id}/deletion-impact
POST   /dataset-manifests/{id}/archive
POST   /dataset-manifests/{id}/restore
DELETE /dataset-manifests/{id}?cascade=false

GET    /shards/{id}/deletion-impact
POST   /shards/{id}/archive
POST   /shards/{id}/restore
DELETE /shards/{id}?cascade=false

GET    /tracks/{id}/deletion-impact
POST   /tracks/{id}/archive
POST   /tracks/{id}/restore
DELETE /tracks/{id}?cascade=false

GET    /benchmarking-runs/{id}/deletion-impact
POST   /benchmarking-runs/{id}/archive
POST   /benchmarking-runs/{id}/restore
DELETE /benchmarking-runs/{id}?cascade=false
```

列表接口增加 `include_archived=false`：

```text
GET /dataset-manifests?include_archived=true
GET /shards?include_archived=true
GET /tracks?include_archived=true
GET /benchmarking-runs?include_archived=true
```

建议权限：

- `dataset.delete`：归档/恢复 dataset manifest 与 shard。
- `track.delete`：归档/恢复 track。
- `run.delete`：归档/恢复 run。
- `admin.purge`：所有物理删除。

## Frontend UX

### Workspace Lists

数据集、赛道、运行工作台增加：

- “显示已归档”开关。
- 每行操作菜单或紧凑按钮：归档/恢复、查看影响、管理员永久删除。
- 已归档资源保留深链可访问，行内显示归档徽标。

### Confirmation Flow

点击归档：

1. 拉取 deletion impact。
2. 展示将被隐藏但仍可访问的对象摘要。
3. 确认后调用 archive。

点击永久删除：

1. 拉取 deletion impact。
2. 明确展示 cascade 范围。
3. 若需要级联，展示“会同时永久删除 N 个赛道 / N 个运行 / N 个报告”等摘要。
4. 二次确认后调用 `DELETE ?cascade=true`。

MVP 不需要输入资源名确认，但按钮文案必须清楚标注“永久删除，不可恢复”。

## Error Handling

后端统一返回 `ApiError`：

- `resource_not_found`：资源不存在。
- `resource_archived`：对已归档 track 启动 run。
- `run_not_terminal`：运行中/排队 run 不能归档或 purge。
- `purge_requires_cascade`：存在下游引用且未传 `cascade=true`。
- `purge_forbidden`：普通用户尝试物理删除。

前端沿用现有 i18n 错误映射；未知错误显示后端 message。

## Testing

后端：

- 归档 track 后列表默认隐藏，详情可访问，创建 run 被拒绝，恢复后可运行。
- 归档 run 后运行列表默认隐藏，报告仍可访问。
- 数据集/切片影响预览正确计数。
- 非级联 purge 有引用资源时返回 409。
- 级联 purge track 删除 run 子表、report、ranking entry。
- 级联 purge dataset 删除关联 track/run/data 子表。

前端：

- 工作台显示归档开关并向列表接口追加 `include_archived=true`。
- 赛道页/运行页/数据集页可以触发归档、恢复和永久删除。
- 永久删除前显示影响预览。
- 已归档赛道详情不能启动新运行，并展示可恢复状态。

## Rollout

第一步提交 spec 和 plan。第二步后端加入状态表、服务和 API。第三步前端加入归档视图和确认流。第四步更新用户手册和开发者数据模型文档。每步都单独提交。
