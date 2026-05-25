# TSBenchmark 数据摄入与样本生成设计

**日期：** 2026-05-25

**输入材料：**

- `README.md`
- `docs/superpowers/specs/2026-05-16-tsbenchmark-mvp-entity-structure-design.md`
- `docs/reference/tsfile-dataframe-manual.md`
- 现有实现：`backend/app/services/{csv_dataset_reader,dataset_reader,dataset_load_service,sample_store,run_executor}.py`、`backend/app/models/{dataset,sample}.py`

**设计范围：** 本文档固化一次设计讨论的结论，覆盖**从 CSV 文件读取到评测样本生成**这一整段数据操作（原 `STAGE 0 → 样本视图`）。它定义新的中间态数据格式：CSV 通用多列摄入、目标列在选择期指定、原始序列以 `TsFile` 落盘作为单一真值源、样本以窗口坐标指针化、模型输入与答案结构分离。评测/输出（forecast 存储、聚合、榜单）不在本文范围。

**与现状的关系：** 本文是对现有实现的**修订设计**，尚未落地。实现前需要 DB 重建、测试同步与 `tsfile` 依赖引入（见第 8 节）。

---

## 1. 背景与动机

当前中间态（`sample.v1` JSONL）是"单变量、纯真实数据、无协变量、无分析元数据"的最小可用版，存在三个结构性问题：

1. **入口锁死单变量**：`CsvDatasetReader` 强制 `csv_single_target_only`，非目标列被读入 `rows` 后即丢弃，协变量从入口就进不来。
2. **答案泄露**：`sample.v1` 把 `target_history`（输入）与 `target_future`（答案）写在同一条记录，评测时整条塞给模型适配器，任何模型可 `return sample["target_future"]` 作弊。
3. **物化冗余 + 无单一真值源**：每个窗口把数据拷一份进 JSONL，重叠窗口重复存储；`SampleIndex` 已存窗口坐标却又物化数据；原始序列从不干净落盘（每次加载重读 CSV）。

本设计在不引入协变量实现的前提下，重构数据通路以根除前两个问题、为协变量预留通路，并把存储切换到时序原生的 `TsFile`。

---

## 2. 数据流总览

```text
CSV (time,a,b,c,d,e)            ── 全列数值校验, 严格等距
  │
  ├─ DatasetManifest: value_columns=[a,b,c,d,e], 不记 target
  │
  ├─ 写 TsFile: per-dataset, 表模型 tsbench.<dataset_id>.<列>,
  │            全列对齐序列, ISO→ms epoch                     ← 单一真值源
  │
  ├─ DatasetLoadJob: 选 target=a(单列), ctx/hor/stride + 可选 max_samples
  │
  ├─ 窗口坐标: 滑动窗口; 超 max_samples 沿序列均匀采样; 全部零样本测试题
  │
  ├─ Shard: 引用 TsFile + 目标列 + 配置(不拷数据); checksum=hash(配置)
  │
  └─ SampleIndex: 只存窗口坐标; storage_ref → TsFile 序列路径 + 行范围
        │
   [读取] 切片器: ts[context 段] → 输入视图(target_history, 原始值)
                 ts[horizon 段] → 答案(系统侧, 不给模型)
        │
   评测: 模型只拿输入视图 → forecast; MASE(主) + MAE/MSE(诊断)
```

协变量字段（`history_cov` / `future_cov` / `*_cov_columns`）全程保留为**空占位**，预留不实现。

---

## 3. 已确认设计决策

### 3.1 CSV 摄入契约（通用多列）

- CSV = 1 个时间列 + **N 个数值值列**（`time,a,b,c,d,e`）。
- **所有非时间列在 reader 期校验为有限浮点**，非数值列直接报错（新增 `csv_value_not_numeric`）。语义即"CSV = 纯数值多变量序列"。
- 保留原有约束：表头必填、列名唯一、UTF-8、时间严格递增不重复、**严格等距**（频率推断）。
- 删除 `csv_single_target_only`：摄入期不再有"目标列"概念，全列一视同仁。

### 3.2 目标列在选择期指定（v1 单目标）

- **摄入与选择解耦**：摄入存全列；**目标列在 `DatasetLoadJob` 选**。
- v1 **只选 1 个目标列**，其余列读入矩阵后在物化时**直接丢弃**。
- 角色模型（`target` / `known_future_covariate` / `past_covariate` / `ignore`）作为未来方向**仅保留概念**；v1 只实现 `target` 与隐式 `ignore`。
- 协变量字段在 schema 中保留空占位，不填充。

### 3.3 评测范式：纯零样本

- 被测模型为 Chronos/Timer 类**零样本基础模型**，不在被测数据上训练。
- **不划分 train/test**：滑动窗口抽出的每个 `(context, horizon)` 都是独立测试样本，模型只看窗口内 `target_history` 做推理。
- 系统中**不存在"训练"步**，只有推理。文中"训练/training"一律指推理。

### 3.4 样本数量：stride 为主 + 可选上限

- 用户设 `context_length` / `horizon` / `stride`，样本数 = 滑动窗口能切出的数量（派生）。
- 新增可选 `max_samples` 上限，给样本数封顶（也给指针化下的窗口数封顶）。

### 3.5 超量采样：沿序列均匀采样

- 当可切窗口数 > `max_samples`，**按等间隔沿序列抽取** `max_samples` 条。
- 确定、可复现，且覆盖序列全跨度（不偏序列某一段）。

### 3.6 target 存原始值，不标准化

- `target_history` / `target_future` 存**原始值**，中间态保持模型无关。
- 不在数据链做标准化：Chronos/Timer 内部自带 scaling，链上再归一化会**双重归一化**。
- 跨序列可比性交由指标解决（见 3.7），而非改数据。

### 3.7 指标：MASE 主排名，MAE/MSE 诊断

- 排行榜主指标改用**尺度无关的 MASE**（`模型误差 ÷ naive 基线误差`），保证跨序列公平。
- MAE/MSE 保留为原始尺度的诊断指标。
- MASE 的 naive 基线可逐样本从 context 段算出（对应 README 的 `baseline_mase`）。

### 3.8 物化：指针方案

- 放弃"每窗口拷一份"，改为**原始序列落盘一份 + `SampleIndex` 只存窗口坐标**，读取时按坐标现切。
- `sample.v1` 不再是磁盘数据文件，退化为**按需算出的视图契约**（定义切出来的形状，但不存储）。
- 印证 `README.md:745`："切换存储不应改变模型看到的 sample schema，存储系统作为底层实现细节隐藏在相同逻辑实体后面"。

### 3.9 落盘粒度：per-dataset + 全列矩阵

- 每个数据集落**一份**完整 `values[N, num_cols]` 全列矩阵（含未来协变量列）。
- 同一数据集派生的多个 `Shard` **共享引用**这一份，零重复、单一真值源、协变量就绪。

### 3.10 落盘格式：TsFile

- 原始序列用 **Apache IoTDB `TsFile`** 列式时序格式落盘，经 `TsFileDataFrame` 读取。
- 选型契合点：
  - **行号切片** `ts[start:end]` → `np.ndarray`，正是窗口"按 `context_start..horizon_end` 现切"所需（手册「场景一：时序大模型预训练」即此用例）。
  - **懒加载**：初始化只扫元数据，按行号索引才触发 I/O。
  - **对齐序列**：同一设备多 field 共享时间轴 = 我们的全列矩阵。
- 时间戳为**毫秒 epoch 整数**，CSV 的 ISO 时间写入时转换。

### 3.11 序列命名：表模型

- 路径 `tsbench.<dataset_id>.<列名>`：逻辑表 `tsbench`，`dataset_id` 作标签，列名作 field。
- 所有数据集统一管理、可跨集查询。

### 3.12 输入/答案结构分离

- **切片器从结构上分离**输入与答案：给模型的输入视图只含 context 段，**绝不含 `target_future`**。
- 答案（horizon 段）与 `future_timestamps` 只留系统侧算指标/画图。
- 根除答案泄露（不依赖运行器"自律不传"）。

### 3.13 复现校验：只 hash 配置

- `Shard.checksum = hash(目标列 + 切分配置 + seed)`，**不含数据内容指纹**。
- v1 便宜优先；代价是检测不出源数据被替换的漂移（可后续升级为含数据指纹）。

---

## 4. 实体与字段变更

### 4.1 `DatasetManifest`

- `target_columns` → **`value_columns`**（= CSV 全部非时间列；创建 manifest 时 peek 表头自动得出）。
- 不再在 manifest 期记录目标列。

### 4.2 新增：原始序列存储（TsFile）

- 位置示意：`runtime/tsfiles/<dataset_id>.tsfile`（per-dataset）。
- 内容：表模型 `tsbench.<dataset_id>.<列>` 的全列对齐序列 + ms epoch 时间戳。
- 写入发生在加载阶段（CSV → 校验 → 写 TsFile）。

### 4.3 `DatasetLoadJob` / 选择配置

- 新增目标选择 `target_columns: list[str]`（v1 校验**恰好 1 个**且 ∈ `value_columns`）。
- 新增可选 `max_samples: int | None`。

### 4.4 `Shard`

- 新增 `value_columns`（全列）；`target_columns` 保留为"选中的单列"；`target_dim = 1`。
- 通过 `dataset_manifest_id` 引用数据集 TsFile（不拷数据）。
- `checksum = hash(target_columns + split_config + seed)`。
- 协变量列字段（如 `known_future_cov_columns` / `past_cov_columns`）保留空占位。

### 4.5 `SampleIndex`

- 保留窗口坐标 `context_start/end`、`horizon_start/end`。
- `storage_ref` 改为指向 **TsFile 序列路径 + 行范围**（替代原 `{"line": N}`）。
- `materialized = False`（指针化）；`materialized_sample_uri` 指 TsFile / 序列路径。

### 4.6 样本视图契约（`sample.v1`，按需计算）

由切片器读取时生成，不落盘：

```jsonc
{
  "schema_version": "sample.v1",
  "sample_id", "shard_id", "sample_index",
  "target_column_names": ["a"],
  "history_timestamps": [...], "future_timestamps": [...],
  "target_history": [[..]],   // [ctx, 1] 原始值
  "target_future":  [[..]],   // [hor, 1] 原始值, 仅系统侧
  "history_cov": [], "future_cov": [],            // 空占位
  "history_cov_columns": [], "future_cov_columns": []  // 空占位
}
```

### 4.7 模型输入契约（`ModelInput`）

切片器交给模型适配器的对象，**不含答案**：

```jsonc
{
  "sample_id",
  "target_history": [[..]],   // [ctx, 1]
  "target_dim": 1,
  "horizon": 3,
  "history_cov": [], "future_cov": []   // 空占位
}
```

- 无 `target_future`、无 `future_timestamps`。
- `ModelAdapter.forecast(model_input, model, timeout_seconds)` 返回 `[hor, target_dim]`。

### 4.8 指标

- 新增 `MASE` 指标定义与逐样本计算（naive 基线取自 context 段）。
- `Track.primary_metric_id` 默认改为 `mase`；`mae` / `mse` 作为诊断保留并继续逐级聚合。

---

## 5. v1 明确不做（预留不实现）

- 协变量（`history_cov` / `future_cov`）的实际填充——字段保留空占位。
- 多目标 / 多变量预测——`target_columns` 字段已可装多列，但 v1 校验恰好 1。
- 不规则 / 缺失值 / 多序列面板 / 静态协变量。
- 含数据内容指纹的 checksum、源数据漂移检测。
- 标准化烘入数据。

---

## 6. 设计决策日志

| # | 议题 | 结论 | 关键理由 |
|---|---|---|---|
| 摄入-1 | CSV 列 | 多列通用，全列数值校验 | 通用性；为协变量铺路 |
| 摄入-2 | 目标选择 | 选择期单目标，其余丢弃；协变量预留 | 摄入/选择解耦 |
| 1 | 评测范式 | 纯零样本，所有窗口=测试题，无训练步 | 基础模型零样本 |
| 2 | 样本数量 | stride 为主 + 可选 `max_samples` | 确定可复现 + 可封顶 |
| 2b | 超量采样 | 沿序列均匀采样 | 代表性 + 确定性 |
| 3 | target 存值 | 原始值，不标准化 | 避免双重归一化 |
| 4 | 指标 | MASE 主排名，MAE/MSE 诊断 | 跨序列可比 |
| 5 | 物化 | 指针方案 | 单一真值源，去冗余 |
| 6 | 落盘粒度 | per-dataset + 全列矩阵 | 零重复，协变量就绪 |
| 7 | 落盘格式 | TsFile | 时序原生，行号切片 |
| 8 | 序列命名 | 表模型 `tsbench.<dataset_id>.<列>` | 统一管理可跨集查 |
| 9 | 输入/答案 | 结构分离，模型只拿输入视图 | 根除泄露 |
| 10 | 复现 | `checksum=hash(配置)` | v1 便宜优先 |

---

## 7. 落地影响（实现前须知）

1. **DB 重建**：`DatasetManifest` 列改名、`Shard`/`SampleIndex` 字段变更，`SQLModel.create_all` 只增表不改列，现有 `runtime/tsbenchmark.db` 需删库重生成（dev 数据可重跑）。
2. **测试同步**：约 11 个测试文件引用 `target_columns` / `csv_single_target`（csv reader、manifest DTO、load flow、e2e），需随接口变更更新。
3. **新依赖**：引入 `tsfile`（`TsFileDataFrame`）到 `backend/pyproject.toml`；`stub_service` 已认 `input.tsfile` / `output_tsfile_path`。
4. **受影响源文件**：`csv_dataset_reader`、`dataset_reader`、`dataset_load_service`、`sample_store`（→ 切片器）、`run_executor`、`models/dataset`、`schemas/dataset`、`api/routes/{dataset_manifests,dataset_load_jobs}`、`services/metric_service`（MASE）。

---

## 8. 后续待讨论（不在本文）

- forecast 输出存储（是否也用 TsFile / `output_tsfile_path`）、逐级指标聚合、排行榜刷新。
- 协变量正式实现（角色模型落地、`*_cov_columns` 填充、known-future vs past 区分）。
- 多目标 / 多变量预测放开。
- 含数据指纹的复现校验升级。
