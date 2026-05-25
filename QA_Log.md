# 学习问答日志（SpriCoder）

## 2026-05-25 ｜ 讲讲 L9（run_executor 编排 + 队列 + 状态机）

- **答案核心**：
  - 派发：`POST /benchmarking-runs` → `create_benchmarking_run`（建 Run+Units+Tasks 骨架）→ `queue.submit` → **仅当 "running" 才起 daemon 线程** `_execute_in_background`（开独立 Session → `execute_run` → `queue.complete`）→ 立即返回。
  - `RunQueue`（内存单并发）：submit 占位/入队；complete 推进指针。execute_run：入口查取消 → running → 四层循环 → 终态 → 报告+刷榜。
  - **坑（从重到轻）**：
    - (G) ⚠️ **`adapter.forecast` 无 try/except**（`run_executor.py:212`）→ 真实 REST 失败会崩 run，且 `queue.complete` 不被调用 → **队列永久卡死、后续 run 不跑**；forecast.v1 的 failed 行/error_code 设计未落地（与 spec §8.5 不符）。接真实推理前必补，建议 P1。
    - (A) ⚠️ **排队 run 不自动执行**：线程只在 create_run 起，`complete` 只推指针不起线程 → 并发第二个 run 永远 queued 到重启。
    - (B) 终态只数 succeeded/failed unit、**忽略 partial_succeeded**：`[1成功+1partial]`→run 判 succeeded（掩盖）；`[全 partial]`→run 判 failed（夸大）。
    - (C) 取消入口-only（执行中不停）；排队中取消的 run 卡住到重启。(D) 重启把 queued/running/cancel_requested 全判 failed。(E) progress 的 sample 计数恒 0。
  - **存储无关**：pivot 不动 L9（仅 _execute_shard 的 read_by_ref 被 SampleStore 挡着）。
- **相关引用**：`backend/app/services/run_executor.py:94,116-123,212`；`backend/app/workers/run_queue.py`；`backend/app/api/routes/benchmarking_runs.py:22,40`。

## 2026-05-25 ｜ 讲讲 L8（回看层：report + sample forecast 视图）

- **答案核心**：
  - 职责：把分数还原成曲线，解释模型为什么赢/输。两产物：`report`（run 级汇总）+ `sample forecast`（样本级，真值+各模型预测画一张图）。
  - `generate_run_report`（`report_service.py:12`）：execute_run 收尾时捞 units/tasks/metrics/artifacts → 写 `reports/{run_id}.json` + Report 行；`model_metrics` 现自带 mase/mse/mae 三指标；`_sample_links` 全量扫每个 forecast 文件每行。
  - `build_sample_forecast`（`sample_forecast_service.py:11`）：`read_by_ref` 读真值（含 target_future，**供画图，非模型输入，合法**）+ 查 (run,shard) 的 ForecastArtifacts 过滤本 sample_id → 组 models[]。
  - 存在意义：真值在 sample.v1、预测在 forecast.v1，两边不存对方（forecast.v1 故意不存 ground truth）→ 回看时按 sample_id 重拼。
  - 观察：(a) report sample 链接全量扫；(b) sample view 每模型整文件读再过滤；(c) **回看链接 `links.report/ranking` 恒 None**（前端跳不回报告/榜单）；(d) report 不显式标注 mase 缺席（#14 影子）；(e) `ForecastStore(".")` 目录参无意义 smell。
  - **与存储 pivot 几乎无关**：sample view 经 read_by_ref 被 SampleStore 抽象挡着；report 只读 SQLite + forecast JSONL，不碰 TsFile。
- **相关引用**：`backend/app/services/report_service.py:12,82`；`backend/app/services/sample_forecast_service.py:11,18,52`。

## 2026-05-25 ｜ 讲讲 L7（榜单计算）

- **答案核心**：
  - 两半：`execute_run` 收尾对 METRIC_NAMES 每个调 `refresh_ranking`（写快照）；`GET /tracks/{id}/ranking → query_ranking`（读快照）。**榜单是持久化 RankingEntry 快照，run 完成时刷一次，查询只读不算。**
  - 算法（`ranking_service.py:8`）：取 RankingList → 收有效 unit 行 → 两 policy 各「删旧 entry → 选 → 按 value 升序 → 写 rank 1..N」。一次 run = 3 指标 × 2 策略 = 重建 6 套。
  - 进榜三道闸（`:53`）：`result_level=unit` + metric 匹配；run 属本 track 且 status∈{succeeded,partial_succeeded}；**unit.status==succeeded**。partial/failed unit 不进任何榜。
  - 两策略：latest=按 `run.created_at` 最新；best=value 最小。查询默认 metric=primary(mase)、policy=latest_valid_result（`ranking_lists.py:14`）。
  - 观察：(a) **排序硬编码升序，`MetricDefinition.direction` 没被用** → 未来加 higher-is-better 指标会排反；(b) L6 #14 榜单侧复现：flat-history 的 succeeded unit 无 mase MetricResult → 主榜静默缺席；(c) 每次全量 delete+重建（非增量）；(d) latest 用创建时刻非完成时刻。
  - **L7 本就全在 SQLite**（读 MetricResult、写 RankingEntry），从不碰 TsFile → 存储 pivot 不动它。
- **相关引用**：`backend/app/services/ranking_service.py:8,21,53`；`backend/app/api/routes/ranking_lists.py:14`；`backend/app/services/run_executor.py:134`。

## 2026-05-25 ｜ 聊一聊 L5（输入/答案分离）

- **答案核心**：
  - 职责：让模型只拿题面、拿不到答案；`target_future`（真值）只在服务端算分用，绝不进模型输入。
  - 心脏在 `run_executor._execute_shard`（`:213-216`）：一次样本读出后分两路——`build_model_input(sample)` 给 adapter（无答案），`sample["target_future"]+target_history` 给 `compute_sample_metrics`（服务端打分）。
  - `build_model_input`（`model_input.py:24`）：拷输入键 + 加 `horizon=len(target_future)`，**故意排除 target_future**。构造器读 target_future 只为取长度，产出 dict 不含它 → 泄露堵在"模型边界"。
  - 两 adapter：`horizon = sample.get("horizon") if not None else len(target_future)`，回退分支实为死代码（model_input 必有 horizon）；请求只用输入字段不碰答案值。
  - 价值：从"承诺不看"变成"看不到"——零样本基准可信度的地基。`test_model_input_no_leak.py` 做回归断言。
  - 与存储无关：操作 `read_by_ref` 拼出的 sample.v1，TsFile/SQLite 皆同 → SQLite pivot 不碰 L5。
  - 此前 ⚠️「run_executor 未接 build_model_input」**已解除**：`run_executor.py:214` 已接。小命名漂：adapter 形参仍叫 `sample`，实收 model_input。
- **相关引用**：`backend/app/services/run_executor.py:213-216`；`backend/app/services/model_input.py:24`；`backend/app/services/stub_timer_adapter.py:15`；`backend/app/services/timer_rest_adapter.py:31`。

## 2026-05-25 ｜ 给我讲讲 L4（TsFileStore + TsFileSlicer 存取引擎）

- **答案核心**：
  - 读写分离两类：`TsFileStore` 只写、`TsFileSlicer` 只读（`tsfile_store.py`）。
  - 写：`write` 拼 `{time(ms), 各 value 列, dataset_id}` 的 DataFrame → `dataframe_to_tsfile(table_name="tsbench", time_column="time", tag_column=["dataset_id"])`；表模型 series = `tsbench.<dataset_id>.<列>`，时间是 ms-epoch 整数键，走高层 API（非手搓 Tablet），tag 只填一个值→一文件一序列。
  - 读：`_open` 懒打开（`TsFileDataFrame` 本身懒视图，初始化只扫元数据）；`slice` 按列读 `df[key][a:b]` 再转置成 row-major；`slice_timestamps` 取 `.timestamps[a:b]`；上下文管理 close。
  - **关键不变量**：`slice` 用**位置索引** `[row_start:row_end]`，要求「TsFile 第 N 行 == CSV 第 N 行」；而 TsFile 内部按时间排序存 → 只有源数据**本就严格递增**才成立。**L1 的 `csv_time_not_monotonic` 是 L4 位置切片正确性的前提**。连带：L1 把「<1s 间隔」判非递增挡掉 → ms 键不撞，但亚秒级数据当前不支持。
  - L4 特有：① `read_by_ref` 每读一样本 new+close 一个 slicer → 一 shard N 次打开同一 TsFile（性能，可复用句柄）；② **非原子写**——`write` 直写最终路径、无 temp+rename，与主 plan §执行约束矛盾（硬崩留半截损坏文件）。
- **相关引用**：`backend/app/services/tsfile_store.py:32,51,80,99,119`；`backend/app/services/sample_store.py:75-80`；`backend/app/services/csv_dataset_reader.py:54,117-119`。

## 2026-05-25 ｜ 当前 L3 的存储方式与形式（指针化已落地版）

- **答案核心**：
  - 存储方式（三层分工）：**值 → per-shard TsFile（`runtime/tsfiles/{shard}.tsfile`，一份、列式、ms-epoch 索引）；定位+元数据 → SQLite `SampleIndex`（storage_ref 行号指针）；sample.v1 → 读时现切现拼、不落盘**。`samples/` 目录现已闲置（不再写 JSONL）。
  - 存储形式① TsFile：表模型 `tsbench.<dataset_id>.<列>`，`TsFileStore.write` 写**全列矩阵**（target + extra 都进，"全列落盘"已落地）；`dataset_id = shard_id.replace("-","")`（UUID 横线在 device 名不合法）。
  - 存储形式② SampleIndex：`storage_ref={dataset_id,shard_id,sample_id,sample_index,target_columns,context:[s,e],horizon:[s,e]}`，**纯元数据+行号区间，无数组值**；`materialized_sample_uri` 现指向 `.tsfile`。
  - 存储形式③ sample.v1：`_assemble` 用指针切 TsFile 拼出与旧版逐字段一致的 dict（`slice(…, end+1)` 修了差一；`_ms_to_iso` 处理 ms→ISO，注释"同机本地时区往返恒等"）。**存储换了、sample.v1 契约没换**——对上层透明。
  - 值得收的小账：① `materialized=True` 名不副实（实为现切，应 false 或重定义）；② 行号在顶层与 storage_ref 存两份（_assemble 只用后者）；③ checksum 含随机 sample_id/shard_id → 跨加载不稳定；④ `materialized_sample_uri` 命名已漂（指向 .tsfile）。
- **相关引用**：`backend/app/services/sample_store.py:26,56,59,82`；`backend/app/services/dataset_load_service.py:168,174,184`；`backend/app/services/tsfile_store.py`；`backend/app/core/config.py:43`；`backend/app/models/dataset.py:47`。

## 2026-05-25 ｜ 给我讲讲 L3（SampleStore 物化 / 指针化）

- **答案核心**：
  - 职责：L3（`sample_store.py`）对 L2 每个窗口建一条 `SampleIndex` + 写一行 `sample.v1` JSONL；**真值 `target_future` 在这里连同输入 `target_history` 落地**。`read_by_ref` 读样本是 O(N) 行扫描、每次重开文件。
  - **现状问题（已验证）**：上游 `DatasetReadResult` 已被改成 `value_columns + values + column_matrix()`、删了 `target_values`，但 `sample_store.py:83-84` 仍读 `read_result.target_values` → **当前加载会炸 AttributeError**。这是 plan B4 还没做的迁移点。
  - 指针化迁移两个坑：① **差一错位**——`TsFileSlicer.slice` 是半开 `[start,end)`，而 `SampleWindow` 行号是闭区间，必须 `slice(ctx_start, ctx_end+1)`；② **ms-epoch 时区往返**——`TsFileStore.write` 用 `dt.timestamp()`（naive 按本地时区），而 sample.v1 原存 isoformat 不转换，迁移后时间戳保真度变（换 TZ 会漂），与 L1 混合时区同源。
  - 物化 vs 指针化动机：物化把重叠窗口的值复制多份（W0 ctx 行0-5、W1 行3-8 → 行3-5 各存一份）；指针化后整列矩阵只写一次（单一真值源）、SampleIndex 只留指针。
  - checksum 真相：`SampleIndex.checksum=sha256(canonical(record))` 含随机 `sample_id/shard_id`(UUID4) → 同数据每次加载 checksum 不同，只验单次加载内完整性，验不了跨加载内容同一性；要跨加载可比须把随机 ID 排除出 checksum 输入。
- **相关引用**：`backend/app/services/sample_store.py:27,58,66,83-84`；`backend/app/services/tsfile_store.py:32,99`；`backend/app/services/dataset_reader.py:12-25`；plan Task B4。

## 2026-05-25 ｜ 给我讲讲 L2（build_windows 切窗）

- **答案核心**：
  - 职责：`build_windows(row_count, context_length, horizon, stride=None)`（`dataset_load_service.py:26`）把一条序列切成 N 个 `SampleWindow`，每个 = 一道考题（给 ctx 个历史点、考后 horizon 个点）。**只算行偏移区间，不碰数据值**（值到 L3 才取）。
  - 机制：`for start in range(0, row_count-required+1, stride)`，`required=ctx+horizon`；窗口字段是左闭右闭行号。
  - **关键更正**：stride 控的是「答案(horizon)段怎么铺」，不是「窗口重不重叠」。full window 因 context 复用历史**天然重叠**；`stride=horizon` 时**答案段严丝合缝、每个未来点被考恰好一次**（干净口径）；`stride<horizon` 答案段重叠→重复计权；`stride>horizon` 留缝→漏测。尾部不满一窗丢弃。
  - 范式：纯零样本，每窗独立考题、无训练步；窗口重叠在零样本下不构成训练泄露，只影响指标计权。
  - 设计点（plan B3）：`max_samples` + `numpy.linspace` 均匀采样防样本爆炸。提醒①均匀抽稀不保证答案段仍铺满（抽后有效间距<horizon 仍重叠）；②max_samples 给「固定条数与长度无关」，多序列时让每序列贡献均等，单序列阶段主要是封顶算量。
  - 与指针化的串联：`SampleWindow` 的行号区间 = decision#2/plan B4 里 `SampleIndex.storage_ref` 要存的切片引用；L2 切窗逻辑不改，改的是下游对行号的用法（物化值→指针）。
- **相关引用**：`backend/app/services/dataset_load_service.py:26-54`；plan Task B3 / B4。

## 2026-05-25 ｜ csv 的结构是如何的？

- **答案核心**：
  - 物理：行式文本，首行表头 + 数据行；分隔符 `, \t ;` 自动识别；编码 UTF-8 / UTF-8-BOM。
  - 逻辑：一行 = 一个时间点（宽表）；列分三类——时间列（1 个，严格递增 / 唯一 / 等间隔）、目标列（MVP 恰好 1 个，数值有限）、其余列（读入 `rows` 但不验不用，协变量预留位）。
  - 硬假设：**一个 CSV = 一条单变量序列**。长表 / 面板（`time,item_id,value`，时间戳重复）直接被拒（`csv_duplicate_timestamp` / `csv_time_not_monotonic`）；宽表多列只评 1 列。
  - 张力：TsFile spike 用 `tag_column=["dataset_id"]`（表模型 `tsbench.<dataset_id>.<列>` 天生支持多序列），但 CSV 摄入每文件只一个 `dataset_id` → **存储层支持多序列、摄入层只喂单序列**。真实 TS benchmark（M4 / electricity / traffic）几乎都是多序列面板，与当前单序列结构错配。
- **相关引用**：`backend/app/services/csv_dataset_reader.py:21,52-55`；`test/flow_template.csv`；living doc §2（`tag_column`）。

## 2026-05-25 ｜ 我和你聊天的时候，有时候文字会变成表情或者乱码，如何处理

- **答案核心**：
  - 先诊断：实测 `locale` 为 `zh_CN.UTF-8`、`LC_CTYPE=zh_CN.UTF-8`，终端是 iTerm.app → **系统编码正常**，可排除经典编码问题。
  - "变表情"属**输入侧误触**：①macOS 地球键 🌐/fn 弹 Emoji 选择器（系统设置→键盘→"按下🌐键时"改为"不执行任何操作"）；②输入法的 emoji 联想候选；③macOS 文本替换。
  - "显示乱码/方块/错位"属 **iTerm2 字体渲染**：用 CJK+emoji 覆盖好的字体（Sarasa Mono SC / Nerd Font）、勾 "Use a different font for non-ASCII text"、勾 "Treat ambiguous-width characters as double width"、确认 Character Encoding=UTF-8。
  - 临场：若只是本地显示异常但字发对了，我读的是底层文本通常没事；若真发出乱码，重发或补一句澄清即可，我以后说为准。
- **相关引用**：本机实测 `locale` 输出；iTerm2 Settings→Profiles→Text / Terminal。
