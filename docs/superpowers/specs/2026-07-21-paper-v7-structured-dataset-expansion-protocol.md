# Paper v7 结构化数据集扩展与预注册协议

- 状态：已批准，实施中
- 更新日期：2026-07-21
- 适用范围：CapTS-Bench dataset-local 合成生成、E1/E2/E3；不涉及平台前后端

## 1. 决策摘要

v7 不再以增加单变量数据集数量为目标。现有单变量覆盖已经足够支撑六项单变量能力，
真正缺少的是：

1. 同一结构化能力跨数据集复现；
2. 结构化真实窗口上的模型外部效度检验；
3. 比当前固定三目标、单层两子节点、两条合成协变量更丰富的变量结构；
4. 对 known-future covariate 的 issue-time 证据，而不是事后可见的天气、库存或促销字段。

因此冻结以下优先级：

| 优先级 | 数据/动作 | 正式用途 | 决策 |
| --- | --- | --- | --- |
| P0a | 现有 M5 增加 sibling panel 与 covariate view | common factor、hierarchy、covariate | 先做；无需新增原始资产 |
| P0a | 现有 ETT1/H 使用原生同步 7 通道，ETT2/H 作 sensitivity | common factor | 先做；替代“连续取三条独立序列”的弱语义 panel |
| P0a | 审计现有 GEFCom2014 Wind 资产 | common factor、NWP covariate | 先做小试；须恢复 forecast issue/release 语义 |
| P0b | Swiss Hierarchical Demand | common factor、严格层级、NWP covariate | 首个应新增的外部数据集 |
| P0c | GEFCom2012 Load | common factor、严格层级、日历 covariate | 技术上优先；许可证确认后才进入正式集 |
| P1 | FreshRetailNet-50K | common factor、多层聚合、日历 covariate | 强候选；先处理缺货删失和抽样层级 |
| P1 | UCI Hierarchical Sales | 严格层级、小型跨域复现 | 适合低成本 PoC；促销默认不算 known-future |
| P2 | SDWPF、NYC TLC、Low Carbon London | 风电、交通、需求响应扩域 | 条件候选；ETL、权利或 forecast-vintage 成本较高 |
| smoke | UCI Bike Sharing | 两子节点层级、日历 covariate | 只做管线测试，不进入主结论 |

单变量正式默认集从“所有可运行配置”改为 `core + stress`，其余放入 sensitivity；
但不根据 E2 对齐分数高低事后删数据集。当前硬失败的 KDD Cup 2018/H、
Bitbrains RND/H 和 BizITObs L2C/H 单变量配置，以及 BizITObs panel，不再进入 v7
默认生成。

## 2. v6 覆盖审计与研究缺口

依据
[`2026-07-20-paper-v6-e2-formal-rerun.md`](../baselines/2026-07-20-paper-v6-e2-formal-rerun.md)
及 runtime 产物，v6 有 20 个 dataset/task views、64 个 supported
`dataset × capability` cells：

| 任务族 | supported cells | master samples | v6 数据集覆盖 |
| --- | ---: | ---: | --- |
| 六项单变量能力 | 58 | 46,400 | 10 个可用单变量数据集 |
| `common_factor` | 3 | 2,400 | Electricity、Traffic、Jena |
| `hierarchical_coherence` | 1 | 800 | M5 |
| `covariate_response` | 2 | 1,600 | GEFCom2014 Load、Solar |
| 合计 | 64 | 51,200 | 结构化样本仅占 9.4% |

这批样本产生 1,561,600 个八模型 compatible synthetic inference views，最终失败为
0；但 417 个 real-source masters、1,668 个 real-source views 全部来自单变量任务。
因此 v6 能回答“合成结构能力上模型有什么差异”，还不能回答“这些差异是否转移到相同
结构的真实多变量任务”。

E3 也只为每项结构化能力使用一个代表数据集：

- common factor：Electricity；
- hierarchy：M5；
- covariate response：GEFCom2014 Load。

这使数据集与能力完全混杂，不能把某一个数据集上的结果解释为能力级普遍结论。v7 的
最低目标不是再增加数万个同分布样本，而是让每项结构化能力至少有三个不同数据集、
两个不同领域，并生成相匹配的 structured real-source evaluation windows。

runtime 已有完整的 51,200 条 v6 Bank B 合成母样本，但用户已决定不再为它补推理。
v7 不复用、不等待这批样本；新的样本量和 160-group 分块直接在 v7 内完成稳定性审计。

## 3. 生成器对数据集的准入契约

### 3.1 通用硬条件

每个 task view 必须独立满足：

- 固定 \(H=48\)，\(L\in\{96,168,336,504\}\)；
- 能形成完整 \(L+H\) 母窗口，四个 lookback 共享相同 future；
- 官方 test/evaluation tail 在 profile 拟合前排除；
- parameter、gate-reference、gate-calibration 三路拆分，时间边界至少使用
  \(L+H\) embargo；panel 优先按 series/group 隔离；
- 所有值有限，缺失处理规则冻结且不使用 future target；
- 实际实现至少需要 30 个 reference 和 20 个 calibration windows；考虑 parameter
  split、embargo 和失败余量，候选初筛以每个 view 至少 80 个有效母窗口为安全线；
- dataset-local 五档真实容忍区间与生成器可行区间有足够交集，所有 gate fail closed。

频率不是越高越好。固定 48 步 horizon 的物理意义必须在数据集登记表中明确；若重采样，
必须说明聚合量是 sum、mean 还是 last，并在 profile 前完成。不得仅为适配 \(H=48\)
对 target 做上采样或复制。

### 3.2 Common factor

正式 panel 必须：

- 至少三个同频、同步、具有共同业务或物理语义的 leaf targets；
- 各通道在同一个预测起点可同时获得；
- 缺失掩码与时间对齐规则可审计；
- 不把 total/parent 和其 children 同时放入 factor target matrix。

最后一条很重要：把精确加总列与 children 一起做 PCA 会机械抬高
`pca_top1_explained`，不是共同因子证据。当前 TSF loader 把若干独立 series 按顺序
拼成三通道的做法只可用于管线 smoke；没有时间同步和分组语义证明时不得进入 v7
正式 common-factor 集。

v7 分两个维度轨：

- `canonical-d3`：固定三个 leaf targets，保持与 v6 生成器和更多模型兼容；
- `native-d`：保留 7、8、16 或 24 个同步 targets，检验通道维度和 factor rank
  错配；只在共同兼容模型集合内比较。

`canonical-d3` 与 `native-d` 是不同实验轴，不把模型因输入维度不兼容记为预测失败。

### 3.3 Hierarchy

层级 view 必须提供 summing matrix \(S\)、节点路径和 bottom-level 身份，并对每个时间点
验证：

\[
y_{\mathrm{all}} = S y_{\mathrm{bottom}}.
\]

接受两类层级，但必须分层报告：

- `publisher-native`：发布者明确给出或定义层级与 aggregates；
- `metadata-derived-exact`：依据稳定的地理、商品或设备元数据，从 bottom series
  确定性聚合。

不接受根据相关性聚类后命名为“层级”，也不接受缺失节点后悄悄改变 parent 构成。
current v6 的 `parent + 2 children` 保留为 canonical topology；完整多层 \(S\) 另设
topology 实验，避免把“层级更深”和“heterogeneity 更强”混成同一干预。

### 3.4 Known-future covariates

每个 covariate 必须有 provenance：

```text
covariate_id
source
issue_time
valid_time
availability_lag
revision_policy
known_future_reason
```

准入顺序如下：

1. **安全**：确定性日历、星期、小时、法定节假日；
2. **有条件安全**：明确提前公布的活动、SNAP、价格计划、动态电价；
3. **仅 forecast vintage 安全**：在预测起点已经发布、且覆盖完整 48 步的 NWP；
4. **默认不安全**：实际观测天气、ERA5 reanalysis、事后库存状态、销量驱动的
   stockout 标记、没有发布时间证据的促销/折扣、后修订宏观值。

同一自然变量的“实际天气”和“当时可获得的天气预报”是不同字段。不得用完整数据表中
未来行存在这一事实代替 issue-time 可用性证明，也不得拼接预测起点之后才发布的新一版
天气预报来覆盖 horizon。

## 4. 候选数据集评估

符号：`✓` 可作为正式 view；`△` 有条件或仅 metadata-derived；`—` 不适合。

| 数据集 | 同步 common factor | 严格 hierarchy | known-future | H/L 与规模 | 结论 |
| --- | --- | --- | --- | --- | --- |
| Swiss Hierarchical Demand | ✓ 24 meters | ✓ 7 aggregates，publisher-native | ✓ NWP forecast vintage；仅覆盖 24h | 10min target；可聚合至 30min，使 H48=24h | **P0，综合最强** |
| GEFCom2012 Load | ✓ 20 zones | ✓ Zone 21 = 前 20 区之和 | ✓ 日历/假日；△ temperature | 约 4.5 年 hourly | **P0，许可证 gate** |
| M5 | ✓ 同店/品类 siblings | ✓ 42,840 series、12 levels | ✓ calendar/event/SNAP；△ price | daily，现有资产 | **P0，先补 view** |
| GEFCom2014 Wind | ✓ 10 farms | △ 可构造总功率，不是原生层级 | △ u/v NWP，须恢复 release vintage | hourly，现有 zip | **P0 小试，先审计 48 步可用性** |
| FreshRetailNet-50K | ✓ store-product groups | △ 从元数据精确聚合 | ✓ calendar/holiday；△ discount/activity | 50K series、hourly、约 97 天 | **P1，强候选** |
| UCI Hierarchical Sales | △ 118 SKU siblings | ✓ 3 levels | ✓ calendar；△ promotion | daily，2014--2018 | **P1，小型 PoC** |
| SDWPF | ✓ 134 turbines | △ farm total | — ERA5 不是 forecast vintage | 10min、两年、约 11.4M rows | P2，适合 factor/派生 hierarchy |
| NYC TLC Taxi | ✓ zones | △ zone→borough→city | ✓ calendar/holiday | hourly 可长时间覆盖；ETL 重 | P2，运输领域价值高 |
| Low Carbon London | ✓ household groups | △ group→all | ✓ day-ahead dynamic tariff | half-hourly；权利与版本需核 | P2，若保留 tariff 则很独特 |
| UCI Bike Sharing | — 只有 casual/registered 两个 leaves | ✓ `cnt=casual+registered` | ✓ calendar/holiday/workingday | 17,389 hourly rows | **仅 smoke** |

### 4.1 对原推荐表的修正

**Swiss Hierarchical Demand** 是最优先的外部新增项。公开数据有 24 个 meters 和
S1/S2/S11/S12/S21/S22/all 七个严格 aggregates；NWP 每 12 小时更新，每次只提供未来
24 小时。为保持 v7 的 \(H=48\)，使用 30 分钟 target 聚合，使 48 步恰好等于 24 小时；
小时级 NWP 可按 valid time 映射到两个半小时槽。不得把第二次发布的新 forecast
拼到第一次 forecast 后面。

**GEFCom2012 Load** 的 Zone 21 明确定义为 Zone 1--20 之和，因此同时适合 factor 和
native hierarchy。其未来温度是否以 forecast vintage 提供不能仅凭 competition 文件有
未来 temperature 列推定；第一版只允许日历和假日 covariates。数据使用权未像 UCI 或
Zenodo 数据那样给出清晰开放许可证，正式下载、再分发和 artifact 发布前必须完成人工
license gate。

**M5** 不应再只承担一个三节点 hierarchy view。它可以从同店、同州、同品类的
bottom series 建立 leaf-only factor panels，也能用 calendar、event 和 SNAP 建立
covariate view。`sell_price` 只有在相应预测起点能证明为预先制定且已知时才允许进入；
滚动回测不能默认把事后表中的未来价格视为 known-future。

**FreshRetailNet-50K** 的价值在规模、小时频率和 city/store/category/product 元数据。
它的 hierarchy 是从 bottom series 按稳定元数据精确聚合，属于
`metadata-derived-exact`，不标作 publisher-native。`discount`、`activity` 是否提前
公布需要审计；实际天气和 stock status 默认不能用作 known-future。由于生鲜销量被
缺货删失，必须预注册“过滤缺货窗口”或“保留并加 censoring 标记”之一，不能把缺货后的
零销量直接解释为需求间歇性。

**UCI Hierarchical Sales** 有 118 个 SKU 和自然三层结构，许可证清晰，适合作为快速
跨域复现。元数据记录 1,798 天，与 2014--2018 完整日历的天数不一致；pilot 必须先
重建 28 个日期差异，不能用行号直接生成日期。promotion 没有公告时点证据时只可作
observed/history covariate。

**UCI Bike Sharing** 的严格恒等式很干净，但只有两个 leaf targets；把 total 一并放进
factor matrix 会产生精确共线性。因此它只验证 hierarchy/covariate loader、\(S\) 矩阵
和日历编码，不承担 common-factor 或论文 headline。

### 4.2 额外候选与不推荐项

- **GEFCom2014 Wind** 已包含在仓库 runtime 的 GEFCom2014 zip 中，十个风场的功率与
  10m/100m 的 u/v weather forecasts 都已存在。它是最低下载成本的 factor/covariate
  扩展，但必须从 competition task release 恢复 `issue_time`，确认每个起点有完整
  48 步 forecast；否则只进入 common-factor view。
- **SDWPF** 能提供高维风机 panel 和总场聚合，但配套 ERA5 是 reanalysis，不是预测时
  已知的天气 forecast。其异常、限电和负功率处理也必须先冻结。
- **NYC TLC** 可建立 zone→borough→city 的跨域层级，但 DST、无效 location、
  vendor coverage 和疫情断点会显著增加 ETL 与解释成本，排在 P2。
- **Low Carbon London** 的 day-ahead dynamic-pricing group 是少见的真实干预
  covariate；但保留 tariff 的原始版本没有清晰许可证，CC BY 的重构版又可能不含该组。
  只有在权利与字段同时满足时才纳入。
- FRED-MD、Nixtla Tourism/Labour、Australian Tourism 当前不推荐。其月/季频率下
  \(H=48\) 代表 4--12 年，且常用 horizon 明显短于 48；强行纳入会改变问题定义。
  FRED-MD 还需要 ALFRED vintage 才能避免宏观修订泄漏。

## 5. 单变量集的收缩方案

### 5.1 默认正式集

| 分层 | 数据集 | 保留理由 |
| --- | --- | --- |
| core | M4 Hourly | 经济/金融域、成熟基准 |
| core | Electricity/H | 能源域，且与 structured Electricity 有连续性 |
| core | ETT1/H | 原生多通道，可同时支持单变量与 panel 分析 |
| core | Loop Seattle/H | 交通域，且具有可继续开发为同步路网 panel 的结构复用价值 |
| core | Bitbrains Fast Storage/H | Web/CloudOps 域，保留跨域覆盖 |
| stress | Jena Weather/H | 保留困难/低对齐案例，防止只报告有利数据 |
| stress | Solar/H | 保留当前不支持 nonlinear 的真实 support 边界 |

### 5.2 Sensitivity

ETT2/H、M_DENSE/H 和 SZ-Taxi/H 不删除 artifact 或结果，但从新一轮默认预算移到
sensitivity：

- ETT2 与 ETT1 同域、同 schema，新增信息有限；
- M_DENSE 与 Electricity 的现有模型 rank vector 高度相似；
- SZ-Taxi 与 Loop/M_DENSE 同属交通域，可作为 transport 内部敏感性。

这个收缩依据领域冗余、结构复用价值和运行预算，不依据“哪一个能提高平均
Spearman”。v6 已出现的低对齐/负结论必须保留；不得为改善 headline 在看到 v7
模型结果后调整 core、stress 或 sensitivity。

### 5.3 硬排除

- KDD Cup 2018/H：仅形成 4 个有效窗口；
- Bitbrains RND/H：仅形成 4 个有效窗口；
- BizITObs L2C/H univariate：`L=504` 时 reference 为 0、calibration 为 27；
- BizITObs L2C/H panel：总有效窗口约 38。

它们保留在完整 support matrix 中并记录 `unsupported`，但不再反复消耗校准预算。
若以后通过不同频率、较短 \(L\) 或新的缺失处理恢复，必须作为新 task view，不覆盖
v6 的 unsupported 事实。

## 6. v7 目标 suite

目标不是每个数据集都支持三种结构，而是每种结构有可解释的跨数据集覆盖：

| 能力 | 最低正式 views | 建议首批 |
| --- | ---: | --- |
| common factor | ≥ 6，≥ 3 领域 | Electricity、Traffic、ETT1、M5、Swiss、GEFCom2012 |
| hierarchy | ≥ 4，≥ 2 领域 | M5、Swiss、GEFCom2012、UCI Sales；NYC TLC 为跨到第三领域的 stretch goal |
| covariate response | ≥ 5，含 ≥ 2 个真实 forecast/intervention | GEF Load、GEF Solar、M5、Swiss、GEF Wind |

FreshRetailNet-50K 在缺货规则、权限和 ETL pilot 通过后替换或补充其中一个零售 view。
这里的计数单位是 task view，不是原始压缩包；统计推断仍以原始 dataset 为 cluster，
同一 M5 的三种 view 不能当作三个独立数据集。

每个正式结构化 dataset 还必须生成真实源任务：

- 使用同一 `task_view_id`、targets、covariates 和 \(S\)；
- 使用未参与 profile/gate 的 official tail 或 time/group holdout；
- 同样产生四 lookback、同一 future；
- 只在共同兼容模型集合中计算 structured synthetic-real alignment。

没有 structured real-source evaluation 的新增 dataset，只能算 generator calibration
扩展，不能算外部效度扩展。

### 6.1 v7 模型比较与降级推理

v7 不再因模型原生不支持多目标或 known-future covariates 而跳过整个结构化 cell。
所有模型都产生与原任务相同形状的最终 forecast，但必须记录实际 inference mode：

- `native`：原生接收全部 targets 和 covariates；
- `targetwise_univariate`：每个 target 独立请求，再按原通道顺序重组；
- `covariates_dropped`：保留 targets，但不向模型提供 covariates；
- `targetwise_univariate_covariates_dropped`：同时使用上述两种降级。

降级只改变模型可见输入，不改变 target、future、context、sample id 或评分器。层级任务
拆成单变量后 parent 与 children 独立预测，不做事后 reconciliation；其 coherence
误差是模型输出的一部分。协变量被丢弃时不使用任何替代未来信息。

主表使用全模型、同一任务的最终 forecasts 比较，并按 inference mode 分层披露；
`native-only` 结果作为敏感性分析。不能把降级模式写成模型“原生支持”结构化输入。
`covariate_response` 的 E3 中，丢弃 covariates 的预测构成无协变量模型结果；只有
原生协变量模型再执行 intact/ablated 配对请求。

## 7. 分阶段实施

### Phase 0：先修身份与审计契约

当前 suite 在内存中以 `source_views[dataset_id]` 保存 views；同一 `dataset_id`
注册多个 task views 时后者会覆盖前者。这与“一个 dataset 可有多个 task views”的方法
定义冲突。扩展前必须：

1. 引入稳定的 `task_view_id`；
2. 所有 lookup、profile、gate、sample manifest 和 real-source manifest 使用
   `(dataset_id, task_view_id)`；
3. 原始 asset identity 单独保存，避免把 task view 伪装成独立 dataset；
4. hierarchy manifest 保存完整 \(S\)、node path 和 hierarchy provenance；
5. covariate manifest 保存 issue/valid time 与 allowlist decision。

在这个修复落地前，不运行正式 v7 大规模生成。

### Phase 1：零新增原始资产 pilot

依次做：

1. ETT1/H 原生 7-target panel；
2. M5 sibling leaf panel；
3. M5 hierarchy 保留 canonical 三节点，同时验证完整 \(S\)；
4. M5 covariate view，只启用 calendar/event/SNAP；
5. GEFCom2014 Wind common-factor view，并审计 NWP release。

目标是验证多 view identity、structured real-source、native dimension、hierarchy
metadata 和 covariate provenance 全链路，不追求模型结论。

### Phase 2：首批外部扩展

1. Swiss Hierarchical Demand；
2. GEFCom2012 Load（通过 license gate 后）；
3. UCI Hierarchical Sales；
4. FreshRetailNet-50K。

每个数据集先只跑 `max_windows` 小试、五档校准和每 cell 少量 qualification；全部
admission checks 通过后再冻结 support matrix 和正式 sample budget。

### Phase 3：预算与复现

1. 冻结 v7 dataset registry、原始文件 checksum、ETL 版本和 support matrix；
2. 用 development seeds 完成 profile/gate 调试；
3. 每个 supported `dataset × task view × capability × intensity` 生成 320 个
   paired groups；
4. 以确定性顺序切成两个互斥的 160-group analysis blocks，两个 block 都完整保留
   I1--I5；
5. 先跑 structured real-source，再跑正式 synthetic inference；
6. E2/E3 分别报告全 320 主估计与两个 160-block 的稳定性；
7. dataset-level cluster bootstrap，不把同一原始数据集的多个 views 当独立样本。

## 8. Pilot 验收与停止规则

每个候选的 pilot 报告至少包含：

- 许可证/使用条款 URL、下载时间、原始文件 SHA-256；
- 原始频率、时区、DST、缺失、重复时间戳与异常值策略；
- target/covariate/node inventory；
- 四个 lookback 的 parameter/reference/calibration/holdout window counts；
- panel 时间同步与 group 语义证据；
- hierarchy 最大绝对/相对 coherence residual；
- covariate issue-time 覆盖率和 48-step horizon 覆盖率；
- 五档 target、feasible overlap、calibration error、feature/near-distance gate；
- structured real-source window count 与模型兼容矩阵；
- 明确的 `accepted` 或稳定 `unsupported_reason`。

以下任一条件触发 fail closed：

- 不能证明同步 panel 或 exact hierarchy；
- future covariate 无 issue-time 证据，或完整 48 步覆盖率不足；
- 某个正式 lookback 的安全有效窗口不足；
- hierarchy residual 超出预注册浮点容差；
- 需要使用 holdout/test tail 才能完成 profile 或 gate；
- dataset-local 五档没有足够可行区间；
- 权利不允许研究使用、派生 artifact 保存或必要的结果发布。

模型分数、synthetic-real alignment 或“是否得到漂亮结论”不是数据集 admission
criterion。候选一旦在看模型结果之前通过并冻结，就不能因结果不理想被静默删除。

## 9. 预期能回答的新问题

按上述扩展，v7 才能把以下因素拆开：

- rank-1 合成 factor 结论在 3、7、16、24 通道真实 panel 上是否稳定；
- hierarchy 模型表现来自加总恒等式、child heterogeneity，还是多层 topology；
- 日历型 covariate 与 NWP forecast、公告型干预的能力排序是否相同；
- 同一个 M5 原始资产的 factor、hierarchy、covariate 三种 view 是否给出一致的模型画像；
- 结构化 synthetic rank 是否能对齐同 dataset、同 view 的真实 holdout rank；
- 结构化结论在能源、零售、交通三个领域是否复现。

这比继续把单变量数据从 10 个扩到 15 个更可能形成可解释的论文结论。

## 10. 主要外部依据

- Swiss Hierarchical Demand：
  [Zenodo 数据与许可证](https://zenodo.org/records/3463137)；
  [数据论文](https://arxiv.org/abs/1910.03976)
- GEFCom2012 Load：
  [组织者数据说明与下载](https://blog.drhongtao.com/2016/07/gefcom2012-load-forecasting-data.html)；
  [竞赛数据描述](https://prac.im.pwr.edu.pl/~hugo/RePEc/wuu/wpaper/HSC_13_16.pdf)
- M5：
  [竞赛论文](https://www.sciencedirect.com/science/article/pii/S0169207021001874)；
  [Competitors' Guide](https://storage.googleapis.com/kaggle-forum-message-attachments/772349/15032/M5-Competitors-Guide-Final-10-March-2020.pdf)
- FreshRetailNet-50K：
  [Hugging Face data card](https://huggingface.co/datasets/Dingdong-Inc/FreshRetailNet-50K)；
  [数据论文](https://arxiv.org/abs/2505.16319)
- UCI Hierarchical Sales：
  [UCI dataset page](https://archive.ics.uci.edu/dataset/611/hierarchical%2Bsales%2Bdata)
- UCI Bike Sharing：
  [UCI dataset page](https://archive.ics.uci.edu/dataset/275/bike%2Bsharing%2Bdataset)
- GEFCom2014 Wind：
  [竞赛论文](https://www.sciencedirect.com/science/article/abs/pii/S0169207016000133)；
  [IEEE PES data catalog](https://ieee-pes-data-sharing.org/datasets/detail/b1680aa5-a4b8-4423-8760-e509094cacec)
- SDWPF：
  [Scientific Data 论文](https://www.nature.com/articles/s41597-024-03427-5)；
  [Figshare 数据](https://doi.org/10.6084/m9.figshare.24798654)
- NYC TLC：
  [官方 trip record data](https://www.nyc.gov/site/tlc/about/tlc-trip-record-data.page)
- Low Carbon London：
  [英国政府数据页](https://www.data.gov.uk/dataset/df8e55d4-c6d0-498e-abd8-f4c678766cbe/smartmeter-energy-consumption-data-in-london-households1)；
  [TU Delft 重构版](https://research.tudelft.nl/en/datasets/low-carbon-london-smart-meter-data-refactored/)
