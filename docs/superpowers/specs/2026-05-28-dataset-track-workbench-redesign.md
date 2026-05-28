# Dataset / Track Workbench Redesign

**日期：** 2026-05-28

**状态：** 已确认，进入实现。

**目标：** 把前端从“一次性上传到运行”的线性向导，调整为可复用的数据资产工作台：数据文件独立上传、切片独立创建、切片可被多条赛道复用；赛道作为一等页面展示，并从赛道发起新的评测运行。

## 1. 已确认决策

1. **数据切片允许被赛道复用。** 一个 `Shard` 可以被多个 `CapabilityBlock` / `Track` 使用。后端不再把 `Shard.capability_block_id` 当作唯一归属来源，而是通过关联关系表达 `CapabilityBlock -> Shard`。
2. **数据集与切片是资产，不是向导临时产物。** 用户可以在数据集页上传 CSV/TsFile，创建一个或多个切片，然后在赛道页复用这些切片。
3. **赛道是前端一等概念。** 左侧导航新增 `Tracks`，展示已创建赛道、主指标、样本数、排行榜入口和运行入口。
4. **新建评测默认从赛道开始。** `New evaluation` 更像“快速运行模型”：先选择已有赛道，再选模型运行。若没有可用赛道，用户可跳转到数据集页上传/切片或在向导内走上传分支。
5. **TsFile 上传必须在前端可达。** 后端已经支持 `.tsfile` 上传嗅探和 load，前端文件选择器与 split 表单需要使用 `preview.file_format`，不能硬编码 `csv`。

## 2. 目标信息架构

```text
Dataset file (CSV / TsFile)
  -> DatasetManifest
  -> DatasetLoadJob
  -> Shard (reusable slice)
  -> Track (one or more reusable shards)
  -> BenchmarkingRun (track + selected models)
  -> Report / Ranking / Sample forecast
```

左侧导航：

- Overview
- New evaluation
- Datasets
- Tracks
- Runs
- Leaderboards

## 3. 页面行为

### 3.1 Datasets

- 展示 dataset manifest 与 shard 切片。
- 提供独立的 `Upload dataset` 入口，支持 `.csv` 与 `.tsfile`。
- 在数据集详情或数据集页内提供 `Create slice` 操作：
  - CSV：选择 time column、value columns、target column、context/horizon/stride/max_samples。
  - TsFile：用上传嗅探返回的 series / measurement 作为 value columns；时间轴来自文件本身，time column 不作为关键交互。
- 切片创建完成后，该 shard 可用于任意赛道。

### 3.2 Tracks

- 新增 `TracksPage` 列出所有赛道。
- 列表展示：track name、primary metric、policy、sample count、created_at、ranking link、run models action。
- 新建赛道时可以选择多条 ready shard。
- 赛道详情页展示元数据、排行榜、关联切片、历史 runs，并提供 `Run models` 操作。

### 3.3 New Evaluation

- 点击顶部或侧栏 `New evaluation` 时必须创建新的向导会话；如果用户已经在 `#/new`，也要 reset wizard，而不是保留上一次完成状态。
- 快速路径：
  1. 选择已有 Track。
  2. 选择模型。
  3. 创建 run 并轮询进度。
- 辅助路径：没有合适 track 时，跳转/引导到 Datasets 和 Tracks 创建资产。

## 4. 后端关系调整

当前实体里 `Shard.capability_block_id` 与 `CapabilityBlock.track_id` 是单归属关系。为支持切片复用，本轮新增：

```text
CapabilityBlockShard
  capability_block_id
  shard_id
  created_at
```

规则：

- 新创建的 real capability block 通过 `CapabilityBlockShard` 关联 shards。
- `Shard.capability_block_id` 保留作历史兼容字段，但新链路不再写它，也不再用它判断 shard 是否被占用。
- 运行执行时通过关联表取 block 下的 shards；如果旧数据没有关联表记录，则 fallback 到 `Shard.capability_block_id`。
- `CapabilityBlock.track_id` 仍保持单归属：一个 capability block 属于一条 track；复用通过“为另一条 track 创建新的 block，关联同一批 shard”实现。

## 5. 本轮验收

- 同一 shard 可以创建两条 track，两个 track 都能运行模型并各自生成榜单。
- 前端可以上传 `.tsfile`，并创建 manifest/load job。
- 数据集页面可以独立上传数据和创建切片。
- 新增 Tracks 页面，可以查看赛道并从赛道发起 run。
- `New evaluation` 按钮在 `#/new` 页面内再次点击会重置当前向导状态。
- 新增/修改 UI 文案同时提供 `en-US` 与 `zh-CN`。
