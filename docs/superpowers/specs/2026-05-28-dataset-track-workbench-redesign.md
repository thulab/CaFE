# Dataset / Track Workbench Redesign

**日期：** 2026-05-28

**状态：** 已确认并持续实现。2026-06-02 补充：新建评测改为“先创建赛道配置，再选择或生成测试用例集”的向导。

**目标：** 把前端从“一次性上传到运行”的线性向导，调整为可复用的数据资产工作台：数据文件独立上传，由数据集生成的测试用例集（底层仍为 `Shard`）可被多条赛道复用；赛道作为一等页面展示，并从赛道发起新的评测运行。

## 1. 已确认决策

1. **数据切片允许被赛道复用。** 一个 `Shard` 可以被多个 `CapabilityBlock` / `Track` 使用。后端不再把 `Shard.capability_block_id` 当作唯一归属来源，而是通过关联关系表达 `CapabilityBlock -> Shard`。
2. **数据集与切片是资产，不是向导临时产物。** 用户可以在数据集页上传 CSV/TsFile，创建一个或多个切片，然后在赛道页复用这些切片。
3. **赛道是前端一等概念。** 左侧导航新增 `Tracks`，展示已创建赛道、主指标、样本数、排行榜入口和运行入口。
4. **新建评测默认从赛道开始。** `New evaluation` 更像“快速运行模型”：先选择已有赛道，再选模型运行。若没有可用赛道，用户可跳转到数据集页上传/切片或在向导内走上传分支。
5. **TsFile 上传必须在前端可达。** 后端已经支持 `.tsfile` 上传嗅探和 load，前端文件选择器与 split 表单需要使用 `preview.file_format`，不能硬编码 `csv`。
6. **新建评测入口先分流。** 用户进入 `New evaluation` 后先在“创建新赛道”和“选择已有赛道”两个卡片中选择路径；创建新赛道路径先配置赛道名称/主指标，再上传或复用测试用例集；已有赛道路径直接选择模型运行。
7. **命名与主指标由用户显式确认。** 赛道名称与主指标在向导第一步确认。上传数据默认用文件名填充数据集名称与测试用例集名称，但不覆盖已经配置的赛道名称。
8. **未完成向导可恢复。** 当前向导草稿写入浏览器 `sessionStorage`；只要尚未生成报告，`New evaluation` 入口显示为继续当前评测。所有从向导跳出的产物详情页都提供返回当前向导的 CTA。
9. **产品展示术语面向非专业用户。** 前端 UI 使用“测试用例集 / Test case set”表达由数据集生成、可被赛道复用的评测样本集合；代码、API、数据库仍沿用 `Shard` 命名，避免无必要的底层迁移。

## 2. 目标信息架构

```text
Dataset file (CSV / TsFile)
  -> DatasetManifest
  -> DatasetLoadJob
  -> Shard (reusable test case set)
  -> Track (one or more reusable test case sets)
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

- 展示 dataset manifest 与测试用例集（底层 `Shard`）。
- 提供独立的 `Upload dataset` 入口，支持 `.csv` 与 `.tsfile`。
- 在数据集详情或数据集页内提供生成测试用例集操作：
  - CSV：选择 time column、value columns、target column、context/horizon/stride/max_samples。
  - TsFile：用上传嗅探返回的 series / measurement 作为 value columns；时间轴来自文件本身，time column 不作为关键交互。
- 测试用例集创建完成后，该 `Shard` 可用于任意赛道。
- 上传预览成功后，数据集名称默认取文件名（去掉扩展名），测试用例集名称默认取同一基础名加 `test cases` 后缀；两者都可在生成前修改。

### 3.2 Tracks

- 新增 `TracksPage` 列出所有赛道。
- 列表展示：track name、primary metric、policy、sample count、created_at、ranking link、run models action。
- 新建赛道时可以选择多条 ready 测试用例集，并输入赛道名称、选择主指标（`mase` / `mse` / `mae`）。
- 赛道详情页展示元数据、排行榜、关联测试用例集、历史 runs，并提供 `Run models` 操作。

### 3.3 New Evaluation

- 点击顶部或侧栏 `New evaluation` 时必须创建新的向导会话；如果用户已经在 `#/new`，也要 reset wizard，而不是保留上一次完成状态。
- 页面入口先显示两个卡片：
  1. 创建新赛道：进入赛道配置、可选上传、测试用例集选择、选择模型的完整向导。
  2. 选择已有赛道：展示可用赛道，选择模型并创建 run。
- 创建新赛道分支和已有赛道分支都写入同一个向导草稿。新赛道分支保存 preview、manifest/load job/shard/selected shard ids/track/run/report 等产物 ID；已有赛道分支至少保存所选 track 和创建出的 run。
- 从 Created artifacts 面板、`Open run`、`Open report` 或样本预测详情离开向导时，目标详情页需要识别当前草稿里的对应资源 ID，并显示 `Continue current evaluation`。
- 快速路径：
  1. 选择已有 Track。
  2. 选择模型。
  3. 创建 run 并轮询进度。
- 新赛道路径：
  1. 填写赛道名称，选择主指标（MASE / MSE / MAE）。
  2. 上传 CSV/TsFile 并预览，或跳过上传直接复用已有测试用例集。
  3. 如果上传了数据，配置时间列、目标列、context/horizon/stride/max_samples，生成一个新的测试用例集。
  4. 在可搜索、可分页、可多选的测试用例集列表中选择要绑定到赛道的数据集合。刚生成的测试用例集自动预选；用户可以取消或追加已有集合。
  5. 用所选测试用例集一次性创建真实数据赛道，然后选择模型启动 run。

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
- `New evaluation` 首屏提供“创建新赛道 / 选择已有赛道”两个明确入口，避免两种路径混在同一工作区。
- 引导创建赛道时第一步输入赛道名称并选择主指标，创建出的 `RankingList.default_metric_id` 跟随该主指标。
- 上传数据后数据集名称默认来自文件名，创建测试用例集时可以指定集合名称；测试用例集列表和详情优先展示该名称。
- 上传数据步骤可以跳过；跳过后进入已有测试用例集选择。
- 新建评测在创建赛道前展示可搜索、可分页、可多选的测试用例集列表；列表支持按名称、数据集、目标列或 ID 过滤。
