# CaFE Benchmark Extension：Canonical 全流程决策

## 文档地位

本文档是 CaFE 校准、生成、回验、推理和分析的唯一规范性决策文档。
它覆盖真实准确率、真实路径锚定反事实和确定性合成机制三条彼此隔离的研究轨，
不覆盖 FastAPI/Vue 评测平台。

本文档描述的是当前已冻结协议。代码和产物必须与本文一致；发生协议变更时，
必须更新协议版本并创建新的不可变实验目录，不能静默复用旧产物。历史方案只在
文末“已废止设计”中保留极简索引，不再作为实现依据。

## 协议快照

| 项目 | 当前值 |
|---|---|
| 真实 calibration history | 168 |
| 真实 anchor held-out future | 48 |
| real-anchored decomposition history | 504（history-only） |
| real-anchored model master | trailing L336 + observed H48 |
| real-anchored 主表 | `fixed_l168`，不进入 synthetic rank |
| 合成 master history | 336 |
| 合成 forecast horizon | 48 |
| 推理 suffix views | 96 / 168 / 336 |
| 固定主表 | `fixed_l168` |
| Oracle context | 在 96 / 168 / 336 中选择 |
| 合成 MASE denominator | 每个 clean L336 master history 计算一次；seasonal lag 退化时逐通道回退 lag-1；三个 view 共用 |
| 真实 anchor MASE denominator | 每个真实 L168 history 独立计算 |
| 每数据集 anchor 上限 | 256 个通过质量检查的窗口 |
| 正式默认模型 | Timer 4.0、Chronos-2、TimesFM 2.5、TiRex2、Moirai 2、Timer 3.5、Toto 2.0 |
| 主生成 | clean、deterministic future、primary family |
| 预测 horizon 内的新随机创新 | 无 |

“无新随机创新”只适用于 deterministic synthetic。real-anchored 的 H48 是真实
观测到的 nuisance/innovation realization；给定 baseline 后，干预 delta 完全确定。

`oracle_context` 是同一 master 的乐观视野上界，不替代固定视野的受控比较。
反事实 pair 的两个 member 必须共用同一个 Oracle context；该 context 由 pair
平均 MASE 选择。

## 研究目标与解释边界

CaFE 的意图是扩展已有真实 benchmark 的能力分析，而不是只训练或检验一个合成
生成器。三条轨道回答不同问题：

- `real_accuracy`：模型在未经修改的真实 H48 上是否准确；
- `real_anchored_counterfactual`：在真实趋势、残差、尺度和 future nuisance 上只改变
  一个声明机制后，模型能否恢复真实 effect；
- `deterministic_synthetic`：机制构造本身是否可辨识，以及模型在纯净、完全确定题面
  上的能力上界和 stress behavior。

- 真实数据既用于校准经验形态和可用参数尺度，也直接提供 real-anchored baseline
  的数值路径。
- 合成结构是能力题面的主体；合成 future 完全由已知确定性机制产生。
- 主表随机化样本参数、相位、符号、事件位置、载荷和 lag，但不在 forecast
  区间注入不可预测的过程创新。
- 真实校准与合成回验只支持 construct alignment：
  “生成器是否落在合理经验尺度并正确实现目标机制”。
- 三轨分别出表、分别排名；不构造任意加权总分。真实轨结果绝不进入合成机制排名。
- 不要求每个真实数据集天然具备层级、共同因子、跨序列因果或 known-future
  covariate 语义；但缺失校准结构时，对应 `dataset × capability` 不进入实验生成。

## Canonical 全流程与实现入口

```text
注册的真实数据 adapter（GIFT-Eval Arrow 或 M5 CSV）
  → 排除官方 benchmark test tail
  → 构造 L168 calibration anchors 与 L504+H48 authentic backgrounds
  → 提取唯一的 history-only CaFE feature profile
  → 冻结真实分解/干预契约，同时标定 synthetic I1–I5
  → 生成 real-anchored pairs 与 L336+H48 deterministic masters
  → 分轨回验机制、配对关系、hash 与共享 normalization
  → 切出 L96/L168/L336 suffix views
  → 模型推理与真实 anchor 辅助预测
  → fixed-L168 / oracle-context 能力分析
```

正式入口和主要职责如下：

| 文件 | 职责 |
|---|---|
| `src/cafe/pipeline.py` | 完整流程编排和阶段状态 |
| `src/cafe/protocol.py` | 公共协议常量、registry、anchor、校准与 view 逻辑 |
| `src/cafe/data/real.py` | 低耦合真实数据 adapter、统一记录与结构语义 |
| `src/cafe/features/profile.py` | 唯一的 history-only 特征实现 |
| `src/cafe/calibration/runner.py` | 真实 anchor 和 capability calibration bundle |
| `src/cafe/generation/anchored.py` | history-only 真实分解与可解析外推契约 |
| `src/cafe/generation/real_counterfactuals.py` | 真实路径干预、availability 与 paired masters |
| `src/cafe/generation/runner.py` | real-anchored、clean、secondary、robustness 和 input ablation 样本 |
| `src/cafe/validation/runner.py` | 生成合法性、配对关系、强度和尺度回验 |
| `src/cafe/inference/runner.py` | view 准备、多服务推理、断点恢复和严格合并 |
| `src/cafe/analysis/runner.py` | fixed/oracle 指标、排名、matched audit 与 split-bank |
| `src/cafe/analysis/structured.py` | common/cross 结构正控 |
| `src/cafe/generation/families.py` | 十能力 primary/secondary 生成器 |
| `src/cafe/validation/mechanisms.py` | 精简 feature/structure gate |

拆仓前的 standalone pilot/report 不定义正式协议；其历史可通过
`monorepo-cutover-2026-07-28` tag 追溯。

## 真实数据接入与 anchor

### 数据集 registry

当前 canonical registry 包含 21 个逻辑数据集配置：

- 小时级或 M4 Hourly：Electricity、Solar、ETT1、ETT2、Jena Weather、
  KDD Cup 2018、Loop Seattle、SZ-Taxi、M_DENSE、Bitbrains Fast、
  Bitbrains RND、BizITObs L2C、M4 Hourly；
- 日频：Restaurant、Hierarchical Sales、US Births、Saugeen River Flow、
  Temperature Rain、M5；
- 10 秒级：BizITObs Application、BizITObs Service。

COVID Deaths 只有 212 点，Hospital 和 Car Parts 更短，无法稳定提供
`168 history + 48 future`，当前不接入。

registry 只声明 adapter id 和逻辑资产。读取、CSV/Arrow 解析及原生结构组织
由 `src/cafe/data/real.py` 的注册 adapter 完成，anchor/calibration 不包含
数据集格式分支。M5 adapter 使用 evaluation sales 和官方 calendar：

- 同 store/category 的五个不同 active leaf 组成 native multivariate panel；
- 只使用天然含两个 department child 的 HOBBIES/HOUSEHOLD category，显式构造
  `parent = child_1 + child_2`；三子节点的 FOODS 不做静默投影；
- known-future 只包含 day-of-week sin/cos、event count 和对应州 SNAP；
- sell price 及其派生变化不是发行时保证已知的输入，因此显式排除。

每条 anchor 必须携带 `dataset_id`、`config_id`、`task_view_id`、
`anchor_id`、`item_id`、`channel_id`、`window_start`、frequency、
period policy、observed fraction 和输入文件 hash，保证可回查来源。

### 窗口与采样

每个候选真实窗口包含 168 点 calibration history 和连续 48 点 held-out
future，共 216 点。特征只从前 168 点提取；future 只用于真实 anchor 辅助预测。

按 task-view 展开后的每条序列，以不重叠的 216 点容量划分覆盖全时间范围的
strata，每个 stratum 使用冻结 seed 做一次带 jitter 的候选抽样。候选顺序为
确定性的无放回顺序，质量失败后继续使用剩余 strata 回填，直到达到：

```text
N_anchor = min(sum_s floor(T_s / 216), 256)
```

窗口不能跨序列边界。256 是上限，不是最低配额；候选不足时保留所有合格
anchor，不复制窗口凑数。缺失值采用统一、可审计的轻量插补，并保存
`observed_fraction`；低于最低观测比例或无信息窗口明确拒绝。

### 标准化与 period 语义

所有标准化统计量只从 history 计算。

- 普通单变量按该变量 history 的均值和标准差标准化。
- additive hierarchy 使用保持加和恒等式的共享尺度策略。
- paired/counterfactual members 共用标准化统计量，避免尺度差异冒充机制响应。
- 连续 covariate 使用 history 统计量；离散事件保留语义编码。

四种时间尺度必须分开保存：

- `calendar_season_length`：由原生频率得到的日历 provenance；
- `feature_period`：在 L168 内可观察的日历周期；不足两周期时使用真实窗口的
  dominant period；
- `generator_period`：按各能力在 L336/H48 内的可识别性裁剪后的生成尺度；
- `mase_period`：日历周期在 L168 内可定义时使用，否则明确回退为 lag 1。

### Real-anchored authentic background 与干预契约

real-anchored 不复用或拉伸 L168 calibration anchor，而是从已经排除官方测试尾段的
source record 独立抽取：

```text
decomposition_fit = x[o-504:o]
model_history     = x[o-336:o]
real_future       = x[o:o+48]
```

分解只能读取 `decomposition_fit`；改变 `real_future` 不得改变 contract、period、
join point 或 intervention delta。L504 中较早的 168 点只用于分解，模型 master
仍为 L336+H48。future 必须 48/48 有真实观测；fit history 至少 90% 有观测，插补
比例、三段 origin 和 hash 全部写入 calibration artifact。

GIFT-Eval Arrow 本身包含官方 evaluation tail。所有 calibration/real-anchored
strata 都必须先调用 registry 对应的 tail policy；普通 GIFT short-term 使用官方
rolling holdout 长度，M4 Hourly 固定排除官方单个 H48。不能从 upstream benchmark
test tail 抽取 CaFE calibration 或反事实任务。
因此本轨是基于 GIFT source 的 CaFE capability extension，不等同于复现或替代
GIFT-Eval 官方 test-set leaderboard；如报告官方准确率，必须另走官方 split，且
不能把该 future 用于分解、availability 或剂量选择。

baseline 先用未修改的 trailing L336 history 计算 location、scale 和 MASE
denominator，随后同一 pair 的全部 α 共用这组 reference，禁止逐 member 重新
z-score。真实 baseline（含其 residual 与 held-out innovation）逐点保留；只添加
history-fitted component delta：

\[
x^{(c,\alpha)}_t=x_t+(\alpha-1)\widehat M^{(c)}_t,
\qquad \alpha\in\{1.2,1.4,1.6,1.8,2.0\}.
\]

当前核心单变量定义至少包括：

- multi-seasonal：joint harmonic regression 固定主 carrier，只令
  \(\widehat M=S_{secondary}\)。P24 的 P8/P6 等整数高次谐波不能冒充次季节；
  P168 等更长的独立频率可以保留；
- trend：joint regression 同时吸收固定的 carrier/secondary nuisance，只令
  \(\widehat M=T_{nonlinear,W96}\)。该二/三阶局部基在 join point 的值和一阶导均为
  0，水平与线性切线不随 α 改变。当前共享分解实现会先要求一个
  history-resolved carrier 通过 visibility gate 以作为 nuisance；若 carrier
  不可辨识，trend contract 也会显式 unavailable；
- time-varying seasonality：从逐 carrier-cycle 的 amplitude envelope 中，只在
  carrier 足够强、振幅变异和 envelope 主峰通过 history-only gate 时确定较慢的
  modulation period。contract 使用 phase-locked symmetric constrained AM；每个
  carrier harmonic 只有 cosine/sine 两个 envelope 自由度，不再独立拟合四个自由
  sideband 系数。carrier 本身固定，future 为有界周期外推；
- regime switching：先去掉 history-only polynomial 与 harmonic nuisance，只在
  fixed-L168 可见区内选择同时通过标准化 jump 和局部 SSE-reduction gate 的 join
  point \(\tau\)。contract 只缩放 \(\beta 1[t\ge\tau]\)，join 后常数延续，不假定
  future 会发生一个尚未观测的新 regime。

新增单变量能力：

- predictable intermittency：从去 nuisance 的 L504 residual 冻结可预测 sparse
  clock、event width 和 empirical pulse template，仅当 chronological timing 与
  amplitude gates 通过时，令
  \(x_t^{(\alpha)}=x_t+(\alpha-1)\widehat E_t\)；
- nonlinear persistence：不是静态 additive scaling。L504 history 上冻结有界
  nonlinear recurrence 与共享 innovation，正式 H48 从 baseline/treatment 终态做
  zero-future-innovation paired rollout，再把 rollout 差加到真实 future。
  history-residual replay 作为独立、明确排除主分数与排名的 sensitivity task；
  主 validator 不要求 effect 对 \(\alpha\) 严格线性。

结构能力使用原生同步 background，不把独立 item 或展开后的 channel 临时拼接：

- common factor 与 directed cross-series predictive transfer 的正式 panel 固定
  \(D\ge3\)；\(D=2\) 至少要有两个不同 donor background，才进入单独推理与汇总的
  sensitivity track 并产生 matched input ablation，绝不进入主排名；cross 的结论是 predictive transfer，不是
  observational causal edge；
- covariate response 只允许 adapter 明确声明的 known-future input，剂量缩放响应
  coefficient 而不是修改 covariate value；
- common/cross 必须配套 `real_anchored_input_ablation`。ablation 与 main 使用相同
  truth pair，单独报告 effect-NRMSE degradation，主分数权重严格为 0；
- hierarchical coherence 当前只生成 qualification contract 与 raw negativity
  audit，不创建正式 forecast task，也不进入 rank。

资格阈值的数值由协议预声明，不从 reference bank 统计反解，也不用
evaluation origins 调参。source-time-disjoint reference bank 只负责验证同一
capability 的 threshold payload 与 policy id 一致，然后冻结其 hash。一个 native
item 上时间重叠的窗口（含不同 channel）不能跨 reference/evaluation
bank；最终 evaluation origin 只能复用该 policy id 与 threshold hash，
reference rows 永远不进入推理。

α 是物理 component-amplitude dose，不冒充 synthetic 的 real-q10/q90 I1–I5。
real-anchored sample 的 `target_feature` 固定为
`real_anchored_intervention_rms`；原 synthetic feature coordinate 只作 provenance，
不能把 intervention RMS 重新标成 `multi_period_score`、`trend_strength` 等经验特征。
每个 dose 保存一对 `alpha=1` baseline 与 treatment；重复 baseline 是显式配对
设计，validation 不将其误判为 synthetic duplicate。anti-copy 对此轨必须记录
`not_applicable:intentional_real_anchor_counterfactual`，不能全局关闭 synthetic
near-distance gate。

每个 `dataset × capability × background` 在模型推理前冻结 availability；至少四个
eligible authentic backgrounds 才允许该 dataset-capability 进入正式生成和排名轨道。
hierarchical coherence 是明确例外：它只进入 qualification-only 审计，不使用
\(N\ge4\) 开启一个它本就不存在的 formal task 或 rank。未保存真实同步
panel、hierarchy 或 known-future covariate path 的结构能力必须显式 unavailable，
不能用真实标量或独立通道拼成“真实路径”。

decomposition 与 dynamic/event contract 分别检查 L504 history、fixed-L168 可见区与
history-only H48 外推的 component/effect RMS。structural contract 则在按 baseline
L336 标准化后，检查 trailing L336 history 与 H48 component RMS，不把其统称为
L504 history gate。上述非退化门限均为 baseline L336 scale 的 1%（结构标准化
坐标中即 0.01），H48 gate 不读取真实 future。这样既避免把数值上非零但实际
不可见的拟合项当作真实能力，也避免最大剂量的 truth effect 过小而令 effect NRMSE
失稳。

真实 background 是统计单位，不能用多个 synthetic seed 重复包装来扩大样本量。
每个 dataset-capability 先冻结 eligible background permutation，再以全实验
`seed_index` 作为该 permutation 的全局 ordinal 做无放回分配；当 seed ordinal
超过 eligible 数量时不再生成 real-anchored 样本。该映射不随 shard 边界改变，
generation availability 必须保存各能力的 assigned seed indexes 与 effective
background count。分析中的有效 N 因而等于不同真实 background 数，而不是请求的
synthetic seed 数。

real-anchored 的 absolute accuracy 与 mechanism effect 分开报告。对 pair：

\[
\Delta y=y^{treat}_{future}-y^{base}_{future},\qquad
\Delta\hat y=\hat y^{treat}-\hat y^{base}.
\]

保存 treatment MASE、effect NRMSE、effect correlation、amplitude ratio 和以共享
baseline MASE 归一化的 effect MAE。主机制 rank 只使用 fixed-L168 的最大可用
dose effect NRMSE；结果写入独立 `real_anchored_scores.json`，不追加到
`scores.json`，也不参与 synthetic experiment aggregate。
accuracy 以 authentic background 为统计单位：序列化用于配对的重复 baseline
只计一次，再与各 treatment dose 各计一次并先在 background 内平均，最后对
background 等权。mechanism 在最大 dose 上每个 background 恰好一条 effect；
dataset score 必须记录实际 background count 与 ID-set hash。experiment aggregate
仍对可用 dataset 等权，不按某个 dataset 的 background 数量池化加权。

## 特征、真实校准与参数映射

### 唯一 history-only 特征实现

所有真实校准、合成 realized feature 和对齐回验使用
`src/cafe/features/profile.py` 的 feature v6 定义。每个数据集保存唯一经验
feature matrix；p05、p10、p25、p50、p75、p90、p95 只是摘要，不再维护
相互冲突的参数范围、强度范围和 gate 范围。

重点特征采用以下去串扰定义：

- trend：最近 W96 中线性项以外二/三次多项式的能量占比；
- spectral complexity：去趋势后加 taper 的稳定频谱；
- amplitude nonstationarity：去局部趋势后的周期系数变化；
- transition sparsity：去趋势和稳定 carrier 后的 residual difference，
  并乘以未解释残差能量的平方根。

paired I1/I5 off-target selectivity matrix 作为非阻断诊断。部分 sideband
在数学上不可完全分离，因此不设置一个跨能力统一硬阈值。

### 真实特征来源与 fallback

参数映射按下列顺序尝试：

1. 可用的 real univariate feature；
2. 可用的 native multivariate、synchronized hierarchy children 或 known-future
   covariate feature；
3. 明确登记的 protocol constant；
4. 带原因的 protocol fallback。

一个真实特征至少有 12 个有限窗口才标记为可用。不可定义、退化或样本不足时，
不能用 `0.0` 伪装成真实观测；必须在 calibration bundle 的逐参数 provenance
中记录 fallback 和原因。

普通真实数据主要校准背景和 nuisance。每个能力的 intensity 还必须有语义一致的
真实主特征：trend、multi-seasonal、time-varying-seasonality、regime、
nonlinear 和 intermittency 从单变量真实窗口读取；common factor 和 cross-series
只从同步 native multivariate view 读取；hierarchy 从至少两个同步 children
读取，校准时直接构造 `parent = sum(children)`，不要求真实数据另带 parent 或
summing matrix；covariate response 只从声明的 known-future view 读取。不同时间
戳或无共同语义的多变量数据不能冒充 hierarchy 或 covariate 结构。目标主特征缺失、有限窗口少于
12 个或真实范围退化时，该 `dataset × capability` 标记为 unavailable，不得使用
生成器内部剂量替代。

Hierarchical Sales 使用专用 adapter：先对完整 Arrow fail-closed 校验 118 个
item-level leaves、B1–B4 brand 分组、共同日轴和长度，再在同 brand 内按自然顺序
形成不重叠 sibling pairs。adapter 只声明两个 children，校准器构造
`parent = child_1 + child_2`。原始 `hierarchical_sales_data.csv` 存在时，还要求其
QTY 在补齐日轴后与 Arrow 逐点一致，并把一一对应的二元 PROMO 指示器声明为
known-future covariate；原始 CSV 缺失或校验失败时不得从其他目标通道伪造
covariate。普通多变量 sibling 只属于 native multivariate 结构，不能因此获得
hierarchy 或 known-future-covariate provenance。

### I1–I5 标定

同一 seed-group 的 I1–I5 共享真实 anchor、path realization 和所有非目标
nuisance，只改变主机制剂量及其必然下游结果。

I1–I5 使用数据集×family 级标尺，不做逐正式 seed 的精确反解：

1. 从数据集真实 profile 读取可用的 `[q10, q90]`；
2. 用独立 qualification path pool 在 21 个 λ 点分别估计 primary 和 secondary
   family mean `lambda → realized feature` response curve；
3. 取真实区间、primary support 和 secondary support 的共同交集。共同交集必须
   覆盖真实 `q10–q90` span 的 10% 以上。raw λ span 仅作诊断；准入改由
   qualification paths 上的 paired observable gate 决定：每一对相邻档至少
   75% path 同向，且配对差分的标准化分离度至少为 3；
4. 在共同交集内等距放置五个真实来源目标，并分别反解两个 family 的 λ；
5. 正式 seed 按指定 seed index 生成不同 anchor、相位和 nuisance，但不重新
   计算 21 点曲线，也不因单样本偏离参考目标而拒绝。

对结构能力还要在独立 qualification path bank 上，用 primary family 精确的
selected-I5 λ 运行与真实 observable 对齐的结构 gate。common factor 和
cross-series 另外在 `λ=1` 运行盲正控：前者要求 joint-factor 反事实恢复，
后者要求 pair 中唯一发生 history 干预的通道与声明 driver 一致、声明边带来
增量预测收益，并且全 horizon 反事实恢复通过。单 history 的 driver/lag
恢复保留为诊断：高度自相关 panel 中因果方向未必由观察分布唯一识别，不能把
这种统计不可辨识误报成生成结构不可达。盲源诊断最大化所有 outgoing edge
中的最小增量 gain，避免单条 responder-to-responder 捷径掩盖通向真实 driver
的负边。四种结构能力的
qualification path 必须 100% 通过；缺失路径、畸形结果或任一路径不通过均把
当前 cell 标为 unavailable。
该检查不运行 near-distance，也不通过扩大 path 数量掩盖系统性结构不可达。

任一真实主特征或共同支持条件失败时，该 `dataset × capability` 记为 unavailable，
生成阶段跳过；不允许 generator-relative、generator-structural 或内部机制剂量
fallback。unavailable 只影响当前 cell，不阻断同一数据集的其他能力。

`lambda` 的数学坐标始终是 `[0,1]`，具体机制参数可以按物理含义超过 1。
不能使用累计最大包络伪造可逆性。先确定从 `lambda=0` 开始的稳定 support，
再选择原始均值曲线中的可逆分支；原始 curve 与选中分支都保存。

qualification pool 使用独立冻结 namespace 的 32 条随机但可复现路径，其
anchor 和机制 realization 均不与正式生成 seed 对齐，也没有论文样本序号语义。
不要求每条 qualification path 单独覆盖真实 `q10–q90`。两半 path 的 support
差异只作非阻断诊断；只有 family mean response 退化或不可逆才依次扩到
64、96，达到上限仍失败则显式终止。正式 seed 的编号、anchor、path、样本 ID
和 sensitivity 身份保持确定性。最多五个候选 path 的重试只服务于数值合法性、
启用时的 anti-copy gate，以及 selected-dose paired construction gate，
不服务于贴合真实特征目标。当前不自动把过窄的真实强度抬升到统一下限。

## 十种能力的当前生成与评分

| 能力 | Primary family | 主剂量 | I5 主机制指标 |
|---|---|---|---|
| trend | C1 joinpoint quadratic | `local_polynomial_energy_share_w96` | `trend_curvature_component_nrmse` |
| multi-seasonal | sample-specific Fourier basis | `multi_period_score` | `seasonal_spectral_amplitude_relative_error` |
| time-varying seasonality | modulated oscillator | `seasonal_amplitude_modulation` | `instantaneous_frequency_nmae` |
| regime switching | deterministic duration motif | `regime_sparse_transition_score` | `regime_jump_nmae` |
| nonlinear persistence | centered bounded quadratic recurrence | `nonlinear_conditional_effect_size` | `nonlinear_recurrence_residual_nrmse` |
| predictable intermittency | deterministic Gaussian event clock | `event_positive_residual_energy_share` | `event_window_nmae` |
| common factor | dense dynamic factor with joint-state relay | `pca_top1_explained` | protected-target paired `counterfactual_effect_nrmse` |
| hierarchical coherence | aggregate/contrast linear state space | `hierarchy_child_heterogeneity` | `hierarchy_structure_nmae = child_contrast_nmae + coherence_nmae` |
| cross-series dependence | persistent delayed linear state SCM | `cross_series_incremental_r2` | full-horizon paired `counterfactual_effect_nrmse` |
| covariate response | known-future linear response | `covariate_incremental_r2` | `counterfactual_effect_nrmse` |

关键限定如下：

- Hierarchy 的主机制分同时要求恢复 I5 子节点 contrast，并原生满足
  `parent = sum(children)`；`coherence_nmae` 以 1.0 权重作为加和违约惩罚，
  不与 contrast 取平均，因而完全一致时保留原 contrast 分，违反一致性只会
  使机制分变差。两项原始指标继续单独报告。
- Trend 最近 96 点和 H48 future 共用同一二次曲线，更早 history 使用连接点
  切线；secondary 是同样连接语义的受限 cubic。
- Regime 的零强度背景使用周期比为 `sqrt(2)` 的确定性双频平滑纹理；它仍然
  完全可预测且不含 future randomness，但不会因为 8 点纹理与 24 点评分周期
  整除而产生零 seasonal-MASE denominator。
- Nonlinear 的 history-only 条件增量 adjusted-R² 的平方根（相关系数量纲的
  `nonlinear_conditional_effect_size`）是与真实数据校准的强度坐标；硬 gate
  仍独立使用生成器已知系数、实际动态贡献和零状态裁剪，避免把可观测 proxy
  当成机制已正确实现的证明。动态贡献相对 recurrence residual 的比值允许
  饱和或局部 foldback；作为 actuator-health gate，它要求低高剂量端点均值、
  paired 中位数上升且严格多数 seed 为正，不再以 75% 的辅助比值方向门限
  覆盖已经通过的公开强度坐标。
- Intermittency 的强度坐标是在历史窗口内减去固定的居中 9 点移动平均后，
  正残差占正负残差总能量的比例；固定窗口避免数据依赖的频谱选模造成强度曲线
  跳变。spike rate 与 clock R² 只作诊断，
  生成器已知事件能量及 event-window 恢复仍用于结构 gate。
- Common main 是标准 dense dynamic factor；strict joint-state relay 是独立
  联立解码审计，不作为标准 DFM 的硬通过条件。
- Cross primary 是带混合符号 responders 的线性 lag SCM；正确边、方向和 lag
  必须由 history-only gate 恢复。
- Hierarchy 在真实校准与合成生成两侧都只生成/读取两个 children，再以
  `parent = child_1 + child_2` 构造 aggregate；加和残差仍是硬 gate。
- Covariate 以 history-only incremental R² 作为真实强度坐标，并用
  known-future counterfactual pair 证明响应。Primary/secondary 各自反解
  family λ，配对 gate 要求相同的真实参考目标、covariate path 和 baseline，
  不错误要求两个 family 的内部系数或单 seed realized proxy 完全相等。

## 生成组织与回验

### Seed 与样本身份

正式生成只接受扁平的 `seed_start` 和 `seed_count`：

```text
seed indexes = [seed_start, seed_start + seed_count)
```

扩展 seed 范围不会改变已有 seed 的 anchor、nuisance 或样本 ID。不存在
`round_index`、`round_seed` 或 `samples_per_round`。不同日期若需要一批独立
样本，使用不重叠的 seed index 范围；实验目录和 manifest 仍是协议身份。

### 样本表

- 主表：所有 seed 的 clean primary I1–I5。
- Secondary sensitivity：stable-hash 选中的部分 seed，只在 I3/I5。
- Observation-noise robustness：同一子集的 primary I3/I5，只给可见 history
  添加相对 history scale 为 0.15 的观测噪声，评分使用 clean latent future。
- Primary mechanism counterfactual：common/cross 的全部正式 seed、I5 配对表；
  其效应 NRMSE 进入正式机制排名。
- Multivariate input ablation：common/cross 使用 matched donor 替换辅助输入，
  保持受评 target history 与 future 不变。common 在替换段做均值/标准差
  affine matching；cross 的替换段可能只有一个点，因此用 pair-invariant
  driver prefix 估计 affine 变换，避免单点标准化把干预直接消掉。
- Covariate main：counterfactual members 属于主能力构造。

### 必要 gate

生成 acceptance 只保留能直接证明题目成立的检查：

1. shape、finite、history/future 长度和 hash 合法；
2. sample ID、target hash 和必要的近重复检查；
3. 主剂量在完整 seed batch 上对 I1–I5 聚合响应有序且非退化；
4. 每种能力一到两个核心机制约束；
5. robustness、counterfactual 和 ablation 与 clean parent 的不变量成立。

结构 gate 包括层级加和与 contrast、common joint observability、cross 正确
driver/edge/lag/符号和 holdout、covariate counterfactual recovery 等。
selected-I5 的这些结构约束已在校准阶段先做可达性资格检查；正式生成仍逐样本或
逐 counterfactual pair 复验，校准通过不替代生成 acceptance。

对具有真实主特征的能力，仍报告生成主特征是否落在真实 anchor 原始
`[min,max]` 以中点为中心扩成的 `1.2 × span` 范围内，并报告相对 family
参考目标的误差。这两项是 construct-alignment audit，不参与逐样本 acceptance，
也不触发重试。强度有效性由完整 seed batch 上的 I1–I5 聚合响应顺序与跨度回验。

near-distance gate 默认开启，但只承担 anti-copy 语义：先在所有真实 anchor
内部做 leave-one-out，得到 pooled-z RMS DCR 与 NNDR 的 p01；channel 同时满足
`DCR <= p01` 和 `NNDR <= p01` 时标记风险。p01 是面向整批数千次查询的保守
anti-copy 尾部，而不是把单次查询的 p05 假阳性率重复应用到每个样本。单变量样本
直接据此拒绝；多变量样本
使用多数 channel 表决，避免把多个单变量比较中的任一偶然命中误判成整条多变量
样本污染。若可用真实 anchor masters 少于 12，或 pooled scale 退化，该 gate 会带
原因记录为 `not_enforced`，不会伪造距离门限。它不使用 held-out，不参与参数标定，
也不把“像不像真实曲线”当作生成质量目标；可用 `--no-near-distance-gate` 显式旁路并
记录。

合成 MASE 默认使用真实 anchor 给出的 seasonal lag。对完全确定、精确周期的
通道，seasonal-naive history error 可能严格为零；该通道显式回退到标准 lag-1
MASE denominator，不加任意数值 floor，并在样本与 validation audit 中记录
effective period 和 fallback 计数。

cross-series 的真实与合成坐标统一为公开 lag 范围 1–24 上的 history-only
incremental gain，并用整段 panel 时间反向后的同规模搜索作为 paired null。
真实 panel 还提取 `cross_series_effect_memory` 作为 nuisance coordinate，
校准 responder state 的 persistence；它不替代主强度坐标。driver 是由真实
`acf1` 校准的一阶冻结创新过程，最后 `d` 个 history 点接受 pair-specific
innovation intervention，并按同一 driver 状态方程自然传播到 future。每个
responder 直接遵循
`y_t = rho * y_(t-1) + beta * x_(t-d) + epsilon_t`：真实 lag 决定起效时刻，
真实 effect memory 决定延续程度，冻结 innovation path 在强度和 pair 间共享。
因此 responder history 在 pair 内完全相同，而 history 初始化的 driver 与
responder 状态共同把 effect 延续到完整 H48。结构正控从 history 拟合一阶、
双 source-tap ARX recurrence，不读取生成系数。
selected-dose gate 要求声明边相对 responder 自身历史带来正的聚合 holdout
gain，并通过反事实恢复；弱真实剂量下 driver/lag 的唯一识别只作诊断。独立的
`λ=1` 强正控必须从 pair 观察到唯一 driver 干预、通过声明边增量预测和
反事实恢复；单 history 的 driver/lag 恢复在强剂量下也只作可辨识性诊断。旧的
source-only `R² >= 0.5` 与真实增量坐标不同量纲，已经移除。

## 推理协议

正式默认模型按下列顺序登记：

1. Timer 4.0
2. Chronos-2
3. TimesFM 2.5
4. TiRex2
5. Moirai 2
6. Timer 3.5
7. Toto 2.0

本地 `last_value` 和 `seasonal_naive` 只作参考，不参与 foundation-model
正式排名。未进入默认集合的服务模型只有在命令行显式指定并具有 execution
config 时才运行。

模型输入适配必须写入每条 prediction：

- 原生多变量模型使用 joint target；
- 不支持原生多变量的模型由适配器逐变量调用并重组；
- 不支持 known-future covariates 的模型明确省略 covariates；
- 这些输入语义必须在报告中与结构能力结果一起解释。

推理从同一 L336 master 切出 L96/L168/L336 suffix，不重新标准化。所有真实
anchor 同时构造一个 fixed-L168 辅助任务。正式推理 manifest 分开记录 synthetic
views 和 real-anchor views，真实预测另存子集文件。

多服务调度使用模型阶段、确定性 endpoint shards、shape buckets、持久连接和
append-only checkpoint。每个模型结束后严格校验 task hash、row count 和
sample/model 覆盖再合并。`--resume` 只跳过已验证成功任务；非 resume 重新构造
当前 inference shard，禁止静默复用同 ID 的旧预测。设备、endpoint 和并发属于
执行 provenance，不改变科学协议。

多数据集推理默认使用 `--dataset-ids` 的 model-major controller：一个模型只
加载一次并跑完全部数据集，再切换到下一个模型。任务预处理默认使用 16 个 CPU
worker；Chronos-2/Toto 2.0 同时跑 4 个数据集，Timer 4.0/TimesFM 2.5/TiRex2/
Moirai2/Timer 3.5 同时跑 2 个。并发数据集共享同一 endpoint 时，各子任务的 HTTP
concurrency 按活跃数据集数等分，维持既定服务总并发上限。上述数值属于执行
参数并写入 model-major status，不改变实验协议。

推理阶段开始时读取服务 `forecast_limits.input_mode`，把 `-1` 规范化为无上限，
并分别冻结 target 数量、history covariate 数量、future covariate 支持及其 horizon
上限。缺少 `input_mode` 时才回退到旧版顶层字段。规范化结果进入 inference stage
contract 和每条预测的 `input_adaptation`；运行中服务能力与已冻结合同不一致时
立即拒绝继续。原生支持多目标的模型使用一次 multivariate 请求，仅明确声明单
目标的模型才拆为 independent-univariate；协变量支持也独立判定，不再由旧版
`max_covariate_count` 单字段推断。

四张 RTX 5090 的本地服务执行参数以 2026-08-04 修复后的两类端到端 bulk
基准为准：一类覆盖正式主请求形状，一类覆盖原生多目标、层级和协变量形状。
Timer 4.0、TimesFM 2.5 为每卡 4 副本/总并发 32；Chronos-2、Toto 2.0 为
每卡 2 副本/总并发 16；TiRex2、Moirai2、Timer 3.5 为每卡 1 副本/总并发 8。
TimesFM 2.5 的任务批量为 64，其余任务批量沿用模型 execution config。全局配置
按 20 数据集正式主负载定稿；层级/协变量复核用于确认输入路径和显存边界，不让
唯一一个层级数据集覆盖其余 19 个数据集的主负载权重。

校准、生成和回验允许按数据集并发；推荐在 16 核机器上使用 4 个数据集 job，
每个 job 4 个 capability worker。推理在每个模型阶段内按声明顺序组成确定性
数据集 batch；某个数据集的七模型都完成后即可独立下载和校验。并发度只写入
execution provenance，不改变科学协议。

## 分析与报告

### 正式结果表

每个数据集独立报告，不默认跨数据集等权平均或生成单一总排名。主能力表只使用
clean primary family，同时报告：

- `fixed_l168`；
- `oracle_context`；
- `accuracy_score`：I1–I5 的 seed-group mean MASE；
- `mechanism_score`：I5 的能力专属主机制误差；common factor 使用 protected
  target 的全 horizon 配对 effect NRMSE，cross-series dependence 也使用
  responder 的全 horizon 配对 effect NRMSE；另外保留 direct-driver prefix
  与 persistent tail 的分段诊断；
- `accuracy_rank` 和 `mechanism_rank`，两者不合并成任意加权总分；
- history-std normalized MAE 与 MASE denominator 分布作为尺度审计。

Reference baselines 显示分数但不进入 foundation-model 排名。Secondary、
observation-noise robustness 和 multivariate input ablation 分表报告，并与
同 seed/intensity 的 clean primary 匹配比较。common/cross 的 paired
counterfactual 表同时保留 correlation、amplitude ratio；cross 另报告
direct-prefix/tail effect profile，但这些诊断量不与 NRMSE 加权合并。

只需要正式模型 MASE 和机制分时，分析可显式使用 `scores_only` profile。
它仍计算七个 foundation model 的 fixed/oracle MASE、十个能力的主机制分，
以及 common/cross/covariate 主分所需的配对反事实 effect；不运行 reference
baselines、结构正控、split-bank、matched comparison 或 multivariate
utilization audit。被省略的分析会记录在 v4 analysis manifest 中，不能与
`full` profile 静默复用。对既有不可变推理结果的重分析写入新的 experiment
目录，并在 manifest 中绑定源 inference manifest hash。

split-bank 以完整 `seed_index` group 为单位；I1–I5、pair members 和 context
views 不拆开。当前产物可按 32/64/128 等完整 batch 报告相对得分差、Kendall
tau-b、Top-1 一致率和必要的 Top-3 overlap。它是当前实验内部的描述性稳定性
结果，不替代 deferred 的外部校准稳定性审计。

### 结构正控

结构正控单独写入 `structured_positive_controls.json`，不参与 foundation-model
排名。

- Common main：rank-1/2 dynamic factor + factor VAR，与 matched diagonal AR
  比较；主要看 factor trajectory correlation 和 auxiliary-input ablation。
  相对 diagonal AR 的点误差改善只作诊断，不设统一 10% 硬阈值。
- Common strict relay：标准 DFM 标记为 not applicable；生成器感知 oracle
  负责证明联立解码构造数学上可解。
- Cross main：正文 effect reference 使用不知道生成机制的 history-only Full
  Ridge-VAR，并与单变量/diagonal AR 比较；附录另保留盲 source/lag 的 sparse
  ridge VAR 和 input ablation 诊断。
- Cross strict pair：使用一套在 pair-invariant history 上拟合的 ARDL/VARX
  参数应用于两个 member，各自从 observed driver state 递归 future；报告完整
  H48 的 NRMSE、correlation、amplitude，以及 direct-prefix/persistent-tail
  分段结果。

Oracle gate 失败表示生成构造无效。Oracle 通过但结构基线不能利用相应输入时，
优先诊断题面强度、context 和结构信息，而不是直接把失败归因于 foundation model。

## 已验证的关键 pilot

- ETT1 16-seed feature pilot：multi/time-varying 对 local curvature 的归一化
  串扰从 `1.684/2.186` 降至 `0.145/0.157`；对 transition sparsity 的串扰
  从 `1.084/0.810` 降至 `0.229/0.235`；trend 和 regime 的 I1–I5 均为
  16/16 正向。
- ETT1 16-seed structure pilot：Cross VAR 相对 diagonal AR 在
  L96/L168/L336 改善 `27.3%/23.9%/16.9%`；strict active NRMSE 为
  `0.0053/0.0080/0.0030`、相关约 1、zero-tail leakage 为 0。Common
  factor trajectory correlation 为 `0.738/0.779/0.786`，input ablation
  分别恶化 `47.0%/77.9%/76.6%`，三档 context 均无 fallback。
- Jena Weather 64-seed nonlinear 复测：observable primary 五档和 secondary
  I3/I5 递增，actual-lag primary/secondary gate、结构、robustness 与 ablation
  全部通过。
- Jena Weather cross-series v31 复测：真实 history-only incremental-gain
  q10–q90 为 `0–0.175`，两个 persistent delayed family 都得到五档真实校准
  λ；32/32 qualification paths 通过 full-H48 结构 gate，effect NRMSE
  范围 `0.015–0.144`、correlation `0.997–1.000`、amplitude ratio
  `0.900–1.107`。8-seed near-distance generation 无重试且独立验证通过。
  不读取生成机制的 L336 Full Ridge-VAR 在 8 个 I5 pairs 上 full-H48 effect
  NRMSE 中位数为 `0.214`、correlation 中位数为 `0.992`、amplitude ratio
  中位数为 `0.952`；最后 8 点真实 effect RMS 仍为最前 8 点的约 `50.5%`。
- KDD 全链路并行 pilot：16 核、8 workers 下校准约 4.65 倍、生成约 4.96 倍
  加速，64-seed gate 全部通过；worker 数不进入科学协议。
- ETT1 旧逐-seed conditional-inverse pilot 证明严格对齐在部分 generator
  realization 上可实现，但同时暴露出对真实尾部分位数和 family 随机实现过度
  敏感，因此已被当前 family-level 标尺取代，不再作为正式准入结论。
- ETT1 当前 family-level 复测：256 anchors、32 qualification paths 下十能力
  校准用时 107.8 秒；64 seeds 生成 3,858 个 clean masters 用时 32.5 秒，
  零重试，强度、结构、robustness 和 ablation 回验全部通过。该结果来自旧协议；
  当时 Cross 的真实参考只映射到 family λ support 的 17.6%，并曾回退到
  generator-relative 标尺。现协议不再允许此回退，同样情形会把该 cell 标记为
  unavailable。13 个 strict I5 seeds 的最小 history holdout R²
  为 0.996，driver/lag/方向和正控全部通过。Cross primary 约 53.2% 样本落在
  1.2× real-anchor support 内，该比例只作透明审计，不作为准入条件。

这些结果证明当前题面和实现链路在代表性 pilot 上可用，不构成所有数据集或所有
模型的外部性能保证。

## 产物与不可变性

正式根目录：

```text
runtime/experiments/
  <experiment_id>/
    experiment.json
    stage_contracts/
      calibration.json
      generation.json
      validation.json
      inference.json
      analysis.json
    pipeline_status.json
    <dataset_id>/
      01_calibration/
        anchors.jsonl
        real_anchor_masters.jsonl
        real_anchored_backgrounds.jsonl
        real_anchored_contracts.jsonl
        real_anchored_availability.json
        real_anchored_reference_backgrounds.jsonl
        structural_real_anchored_reference_backgrounds.jsonl
        real_anchored_reference_contracts.jsonl
        structural_real_anchored_backgrounds.jsonl
        structural_real_anchored_contracts.jsonl
        structural_real_anchored_availability.json
        structural_hierarchy_qualification.jsonl
        real_anchored_bank_split_audit.json
        real_anchored_qualification_policy.json
        capability_calibration.json
        calibration_bundle.json
      02_generation/
        sample_shards/
          seed_<start>_<end>.jsonl
          seed_<start>_<end>__robustness.jsonl
          seed_<start>_<end>__input_ablation.jsonl
          seed_<start>_<end>__real_anchored_counterfactual.jsonl
        real_anchored_availability__seed_<start>_<end>.json
        structural_real_anchored_availability__seed_<start>_<end>.json
        manifest__seed_<start>_<end>.json
        validation__seed_<start>_<end>.json
      03_inference/
        seed_<start>_<end>/
          synthetic_forecast_views.jsonl
          real_anchored_forecast_views.jsonl
          real_anchor_views.jsonl
          forecast_views.jsonl
          task_manifest.json
          model_task_shards/
          model_shards/
          real_anchor_predictions/
          inference_manifest.json
      04_analysis/
        seed_<start>_<end>/
          prediction_metrics.jsonl
          counterfactual_effects.jsonl
          scores.json
          real_anchored_prediction_metrics.jsonl
          real_anchored_counterfactual_effects.jsonl
          real_anchored_scores.json
          real_anchored_input_ablation_attribution.jsonl
          real_anchored_input_ablation_summary.json
          split_bank.json
          matched_comparisons.json
          structured_positive_controls.json
          REPORT_ZH.md
          MATCHED_AUDITS_ZH.md
          analysis_manifest.json
```

`experiment.json` 只冻结实验身份与存储布局，不提前冻结尚未执行阶段的代码和
配置。每个 stage 在首次启动时创建独立、不可覆盖的 stage contract，保存本阶段
配置、Git revision/dirty hash，并引用上游 stage contract 的文件 hash。因此可以
在校准生成完成后，用后续 Git revision 继续推理或分析，而不改变上游 provenance。
已经存在的 stage contract 不能用另一套代码或配置原地重定义；需要重跑该阶段时
必须创建新的 experiment id。

`pipeline_status.json` 是允许原地更新的运行状态文件。各阶段自己的产物 manifest
继续保存输入 hash、输出 hash、row count 和必要 provenance，拒绝跨生成器或跨
seed shard 静默合并。

多数据集 aggregate 也保持分轨：synthetic 继续写 fixed/oracle 两张能力表；
real-anchored 只从各 dataset manifest v2/v3 的独立 score record 读取，在该能力实际
available 的数据集与完整 foundation-model 交集上等权聚合，写
`capability_scores_real_anchored_fixed_l168.json`。两边的输入行、rank 和 manifest
record 不互相复用。

## 明确 deferred

以下工作不阻塞当前 CaFE：

- M5：后续作为一个逻辑数据集的 covariate、sibling panel、additive hierarchy
  三个 task views 接入；跨数据集汇总时只能计一次逻辑数据集权重。Hierarchy
  adapter 必须按同步时间窗从 bottom series 构造 `y_t = S b_t`，并保存
  summing matrix、node/level ID 和 bottom-node 索引。标准化使用保持加和的共享
  统计量。真实 M5 的 coherence 是恒等约束而不是 I1–I5 强度；真实小树只校准
  子节点贡献集中度、动态异质性、父子尺度和 level profile，合成器仍控制跨层
  信息的机制剂量。
- 外部真实校准稳定性：固定 train/reference/calibration 三路切分、时间 holdout、
  leave-one-group-out、block/group bootstrap、分位数置信区间、conformal
  coverage 和跨时期 feature stability。
- 多 seed-shard 的 suite 级组合分析；现阶段每个 shard 由独立 manifest 引用，
  不修改已存在实验身份。
- 统一的真实低强度自动放大下限、替换或顺延失败 seed。
- 额外 surrogate families 和更重的分布相似性审计。

这些内容若启用，必须作为新协议或明确的补充分析层，不能悄悄改变当前主表。

## 已废止设计

以下历史设计不再属于 CaFE canonical 协议：

- 真实校准和合成 master 使用 L504；
- 96/168/336/504 四视野、`fixed_l504` 和共享 L504 denominator；
- 仅接入 19 个数据集并排除 Restaurant；
- 把 tabpfn-ts3 或 TimePFN 放入正式默认模型集合；
- 真实窗口的三路永久切分、真实 feature range 或 conformal 的逐样本硬准入；
- nonlinear 使用专门的 12/64/128 path 特例或按结果挑选 seed；
- 将斜率误差作为 trend 的主机制分，或要求标准 DFM 通过 common strict relay。

如需追溯这些方案，只查看 Git 历史，不得从本清单恢复实现。
