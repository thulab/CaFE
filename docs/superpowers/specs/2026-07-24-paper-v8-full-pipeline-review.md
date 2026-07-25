# Paper v8 合成机制 Benchmark：Canonical 全流程决策

## 文档地位

本文档是 Paper v8 校准、生成、回验、推理和分析的唯一规范性决策文档。
它只覆盖论文的合成数据研究，不覆盖 FastAPI/Vue 评测平台。

本文档描述的是当前已冻结协议。代码和产物必须与本文一致；发生协议变更时，
必须更新协议版本并创建新的不可变实验目录，不能静默复用旧产物。历史方案只在
文末“已废止设计”中保留极简索引，不再作为实现依据。

## 协议快照

| 项目 | 当前值 |
|---|---|
| 真实 calibration history | 168 |
| 真实 anchor held-out future | 48 |
| 合成 master history | 336 |
| 合成 forecast horizon | 48 |
| 推理 suffix views | 96 / 168 / 336 |
| 固定主表 | `fixed_l168` |
| Oracle context | 在 96 / 168 / 336 中选择 |
| 合成 MASE denominator | 每个 clean L336 master history 计算一次；seasonal lag 退化时逐通道回退 lag-1；三个 view 共用 |
| 真实 anchor MASE denominator | 每个真实 L168 history 独立计算 |
| 每数据集 anchor 上限 | 256 个通过质量检查的窗口 |
| 正式默认模型 | Chronos-2、TimesFM 2.5、TiRex2、Moirai 2、Timer 3.5、Toto 2.0 |
| 主生成 | clean、deterministic future、primary family |
| 预测 horizon 内的新随机创新 | 无 |

`oracle_context` 是同一 master 的乐观视野上界，不替代固定视野的受控比较。
反事实 pair 的两个 member 必须共用同一个 Oracle context；该 context 由 pair
平均 MASE 选择。

## 研究目标与解释边界

Paper v8 测量模型对十种时间序列机制的响应，而不是训练生成模型去复刻真实曲线。

- 真实数据用于校准经验形态、背景 nuisance 和可用参数尺度。
- 合成结构是能力题面的主体；合成 future 完全由已知确定性机制产生。
- 主表随机化样本参数、相位、符号、事件位置、载荷和 lag，但不在 forecast
  区间注入不可预测的过程创新。
- 真实校准与合成回验只支持 construct alignment：
  “生成器是否落在合理经验尺度并正确实现目标机制”。
- 真实 anchor 预测只提供外部 sanity check，不进入合成机制得分或排名。
- 不要求每个真实数据集天然具备层级、共同因子、跨序列因果或 known-future
  covariate 语义；缺失结构由合成机制定义。

## Canonical 全流程与实现入口

```text
GIFT-Eval 原始数据
  → 构造 forecastable real anchor pool
  → 提取唯一的 history-only v8 feature profile
  → 映射背景参数并标定 I1–I5
  → 生成 L336 + H48 deterministic masters
  → 生成机制与配对关系回验
  → 切出 L96/L168/L336 suffix views
  → 模型推理与真实 anchor 辅助预测
  → fixed-L168 / oracle-context 能力分析
```

正式入口和主要职责如下：

| 文件 | 职责 |
|---|---|
| `scripts/run_paper_v8_pipeline.py` | 完整流程编排、不可变协议与步骤状态 |
| `scripts/paper_v8_pipeline_common.py` | 公共协议常量、数据 registry、anchor、校准与 view 逻辑 |
| `scripts/paper_v8_features.py` | 唯一的 history-only 特征实现 |
| `scripts/calibrate_paper_v8.py` | 真实 anchor 和 capability calibration bundle |
| `scripts/generate_paper_v8_samples.py` | clean、secondary、robustness 和 input ablation 样本 |
| `scripts/validate_paper_v8_samples.py` | 生成合法性、配对关系、强度和尺度回验 |
| `scripts/run_paper_v8_inference.py` | view 准备、多服务推理、断点恢复和严格合并 |
| `scripts/analyze_paper_v8.py` | fixed/oracle 指标、排名、matched audit 与 split-bank |
| `scripts/paper_v8_structured_baselines.py` | common/cross 结构正控 |
| `backend/app/services/synthetic_v8_generation.py` | 十能力 primary/secondary 生成器 |
| `backend/app/services/synthetic_v8_feature_gate.py` | 精简 feature/structure gate |

旧 v2–v7 脚本和 `run_paper_v8_model_response.py` 的 standalone pilot/report
不定义正式协议。正式分析仍复用的通用指标函数应保持与上述入口一致。

## 真实数据接入与 anchor

### 数据集 registry

当前 canonical registry 包含 20 个逻辑数据集配置：

- 小时级或 M4 Hourly：Electricity、Solar、ETT1、ETT2、Jena Weather、
  KDD Cup 2018、Loop Seattle、SZ-Taxi、M_DENSE、Bitbrains Fast、
  Bitbrains RND、BizITObs L2C、M4 Hourly；
- 日频：Restaurant、Hierarchical Sales、US Births、Saugeen River Flow、
  Temperature Rain；
- 10 秒级：BizITObs Application、BizITObs Service。

COVID Deaths 只有 212 点，Hospital 和 Car Parts 更短，无法稳定提供
`168 history + 48 future`，当前不接入。

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

## 特征、真实校准与参数映射

### 唯一 history-only 特征实现

所有真实校准、合成 realized feature 和对齐回验使用
`scripts/paper_v8_features.py` 的 feature v5 定义。每个数据集保存唯一经验
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
2. 可用的 native multivariate、explicit hierarchy 或 known-future
   covariate feature；
3. 明确登记的 protocol constant；
4. 带原因的 protocol fallback。

一个真实特征至少有 12 个有限窗口才标记为可用。不可定义、退化或样本不足时，
不能用 `0.0` 伪装成真实观测；必须在 calibration bundle 的逐参数 provenance
中记录 fallback 和原因。

普通真实数据主要校准背景和 nuisance。trend、multi-seasonal、
time-varying-seasonality 和 regime 使用单变量真实主特征范围；common factor
和 cross-series 只有存在同步 native multivariate feature 时才使用真实主特征
范围。nonlinear 和 intermittency 的现有 observable proxy 不能可靠反解，
继续使用生成器内部剂量。hierarchy 必须由声明了 summing matrix 的显式 adapter
提供，covariate 必须有声明的 known-future 输入；普通多变量数据不能冒充这两种
结构。语义不满足时使用并记录 protocol fallback。

### I1–I5 标定

同一 seed-group 的 I1–I5 共享真实 anchor、path realization 和所有非目标
nuisance，只改变主机制剂量及其必然下游结果。

I1–I5 使用数据集×family 级标尺，不做逐正式 seed 的精确反解：

1. 从数据集真实 profile 读取可用的 `[q10, q90]`，作为辅助参考；
2. 用独立 qualification path pool 在 21 个 λ 点估计 family mean
   `lambda → realized feature` response curve；
3. 若真实区间与 primary family mean support 的交集至少覆盖真实
   `q10–q90` span 的 10%，且反解后能覆盖 family 可用 λ support 的至少
   25%，就在交集内等距放置五个参考目标；否则明确回退到
   generator-relative 五级标尺，避免真实窄区间抹平能力难度；
4. 只反解一次 family mean curve，得到该数据集与 family 共享的五个 λ；
5. 正式 seed 按指定 seed index 生成不同 anchor、相位和 nuisance，但不重新
   计算 21 点曲线，也不因单样本偏离参考目标而拒绝。

`lambda` 的数学坐标始终是 `[0,1]`，具体机制参数可以按物理含义超过 1。
不能使用累计最大包络伪造可逆性。先确定从 `lambda=0` 开始的稳定 support，
再选择原始均值曲线中的可逆分支；原始 curve 与选中分支都保存。

qualification pool 使用独立冻结 namespace 的 32 条随机但可复现路径，其
anchor 和机制 realization 均不与正式生成 seed 对齐，也没有论文样本序号语义。
不要求每条 qualification path 单独覆盖真实 `q10–q90`。两半 path 的 support
差异只作非阻断诊断；只有 family mean response 退化或不可逆才依次扩到
64、96，达到上限仍失败则显式终止。正式 seed 的编号、anchor、path、样本 ID
和 sensitivity 身份保持确定性。最多五个候选 path 的重试只服务于数值合法性、
启用时的 anti-copy gate，以及 strict common-factor/cross-series 结构正控，
不服务于贴合真实特征目标。当前不自动把过窄的真实强度抬升到统一下限。

## 十种能力的当前生成与评分

| 能力 | Primary family | 主剂量 | I5 主机制指标 |
|---|---|---|---|
| trend | C1 joinpoint quadratic | `local_polynomial_energy_share_w96` | `trend_curvature_component_nrmse` |
| multi-seasonal | sample-specific Fourier basis | `multi_period_score` | `seasonal_spectral_amplitude_relative_error` |
| time-varying seasonality | modulated oscillator | `seasonal_amplitude_modulation` | `instantaneous_frequency_nmae` |
| regime switching | deterministic duration motif | `regime_sparse_transition_score` | `regime_jump_nmae` |
| nonlinear persistence | signed rational quadratic recurrence | `nonlinear_strength` | `nonlinear_recurrence_residual_nrmse` |
| predictable intermittency | deterministic Gaussian event clock | `event_effect_energy_share` | `event_window_nmae` |
| common factor | dense dynamic factor with joint-state relay | `pca_top1_explained` | `common_component_nmae` |
| hierarchical coherence | aggregate/contrast linear state space | `hierarchy_child_heterogeneity` | `child_contrast_nmae` |
| cross-series dependence | dense delayed linear SCM | `lead_lag_peak_abs` | `responder_normalized_mae` |
| covariate response | known-future linear response | `covariate_effect_variance_share` | `counterfactual_effect_nrmse` |

关键限定如下：

- Trend 最近 96 点和 H48 future 共用同一二次曲线，更早 history 使用连接点
  切线；secondary 是同样连接语义的受限 cubic。
- Regime 的零强度背景使用周期比为 `sqrt(2)` 的确定性双频平滑纹理；它仍然
  完全可预测且不含 future randomness，但不会因为 8 点纹理与 24 点评分周期
  整除而产生零 seasonal-MASE denominator。
- Nonlinear 的 observable adjusted-R² 只作诊断，不反向控制剂量；硬 gate
  使用生成器已知系数、实际动态贡献和零状态裁剪。
- Intermittency 的 spike rate 与 clock R² 只作诊断；事件能量占比是连续剂量。
- Common main 是标准 dense dynamic factor；strict joint-state relay 是独立
  联立解码审计，不作为标准 DFM 的硬通过条件。
- Cross primary 是带混合符号 responders 的线性 lag SCM；正确边、方向和 lag
  必须由 history-only gate 恢复。
- Covariate 主任务用 known-future counterfactual pair 证明响应；历史
  incremental R² 只作可解释诊断。

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
- Strict counterfactual：common/cross 的抽样 seed、I5 独立诊断表。
- Multivariate input ablation：common/cross 使用 matched donor 替换辅助输入，
  保持受评 target history 与 future 不变，并做均值/标准差 affine matching。
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
样本污染。它不使用 held-out，不参与参数标定，也不把“像不像真实曲线”当作生成
质量目标；可用 `--no-near-distance-gate` 显式旁路并记录。

合成 MASE 默认使用真实 anchor 给出的 seasonal lag。对完全确定、精确周期的
通道，seasonal-naive history error 可能严格为零；该通道显式回退到标准 lag-1
MASE denominator，不加任意数值 floor，并在样本与 validation audit 中记录
effective period 和 fallback 计数。

cross-series strict gate 要求盲搜索找对 driver 与 lag、反事实正控恢复通过，并且
声明边在 chronology holdout 上至少解释 50% 响应方差。`R² >= 0.5` 只承担
“依赖强度不是微弱信号”的下限；结构识别本身由正确 edge/lag 和正控指标判断，
不再用 `R² >= 0.8` 把“响应几乎纯线性”误当作“结构可识别”的必要条件。

## 推理协议

正式默认模型按下列顺序登记：

1. Chronos-2
2. TimesFM 2.5
3. TiRex2
4. Moirai 2
5. Timer 3.5
6. Toto 2.0

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
worker；Chronos-2/Toto 2.0 同时跑 4 个数据集，TimesFM 2.5/TiRex2/Moirai2/
Timer 3.5 同时跑 2 个。并发数据集共享同一 endpoint 时，各子任务的 HTTP
concurrency 按活跃数据集数等分，维持既定服务总并发上限。上述数值属于执行
参数并写入 model-major status，不改变实验协议。

校准、生成和回验允许按数据集并发；推荐在 16 核机器上使用 4 个数据集 job，
每个 job 4 个 capability worker。推理在每个模型阶段内按声明顺序组成确定性
数据集 batch；某个数据集的六模型都完成后即可独立下载和校验。并发度只写入
execution provenance，不改变科学协议。

## 分析与报告

### 正式结果表

每个数据集独立报告，不默认跨数据集等权平均或生成单一总排名。主能力表只使用
clean primary family，同时报告：

- `fixed_l168`；
- `oracle_context`；
- `accuracy_score`：I1–I5 的 seed-group mean MASE；
- `mechanism_score`：I5 的能力专属主机制误差；
- `accuracy_rank` 和 `mechanism_rank`，两者不合并成任意加权总分；
- history-std normalized MAE 与 MASE denominator 分布作为尺度审计。

Reference baselines 显示分数但不进入 foundation-model 排名。Secondary、
observation-noise robustness、multivariate input ablation 和 strict
counterfactual 分表报告，并与同 seed/intensity 的 clean primary 匹配比较。

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
- Cross main：history-only、盲 source/lag 的 ridge VAR 与 diagonal AR 比较，
  同时检查 input ablation。
- Cross strict pair：使用一套 history-only shared-fit ARDL/VARX 参数应用于
  两个 member，future driver 共享；分别报告 active prefix 的 NRMSE、
  correlation、amplitude，以及理论零 effect tail 的 leakage。

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
- KDD 全链路并行 pilot：16 核、8 workers 下校准约 4.65 倍、生成约 4.96 倍
  加速，64-seed gate 全部通过；worker 数不进入科学协议。
- ETT1 旧逐-seed conditional-inverse pilot 证明严格对齐在部分 generator
  realization 上可实现，但同时暴露出对真实尾部分位数和 family 随机实现过度
  敏感，因此已被当前 family-level 标尺取代，不再作为正式准入结论。
- ETT1 当前 family-level 复测：256 anchors、32 qualification paths 下十能力
  校准用时 107.8 秒；64 seeds 生成 3,858 个 clean masters 用时 32.5 秒，
  零重试，强度、结构、robustness 和 ablation 回验全部通过。Cross 的真实参考
  只映射到 family λ support 的 17.6%，因此按统一 25% 规则回退到
  generator-relative 标尺；13 个 strict I5 seeds 的最小 history holdout R²
  为 0.996，driver/lag/方向和正控全部通过。Cross primary 约 53.2% 样本落在
  1.2× real-anchor support 内，该比例只作透明审计，不作为准入条件。

这些结果证明当前题面和实现链路在代表性 pilot 上可用，不构成所有数据集或所有
模型的外部性能保证。

## 产物与不可变性

正式根目录：

```text
runtime/paper_exp/v8/
  <experiment_id>/
    experiment_manifest.json
    pipeline_status.json
    <dataset_id>/
      01_calibration/
        anchors.jsonl
        real_anchor_masters.jsonl
        capability_calibration.json
        calibration_bundle.json
      02_generation/
        sample_shards/
          seed_<start>_<end>.jsonl
          seed_<start>_<end>__robustness.jsonl
          seed_<start>_<end>__input_ablation.jsonl
        manifest__seed_<start>_<end>.json
        validation__seed_<start>_<end>.json
      03_inference/
        seed_<start>_<end>/
          synthetic_forecast_views.jsonl
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
          split_bank.json
          matched_comparisons.json
          structured_positive_controls.json
          REPORT_ZH.md
          MATCHED_AUDITS_ZH.md
          analysis_manifest.json
```

`experiment_manifest.json` 保存完整科学协议、协议 hash、代码版本和存储约定，
创建后不可覆盖。`pipeline_status.json` 是唯一允许原地更新的根状态文件。
每一级 manifest 保存 schema/version、输入 hash、输出 hash、row count 和必要
provenance，拒绝跨版本、跨生成器或跨 seed shard 静默合并。

## 明确 deferred

以下工作不阻塞当前 v8：

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

以下历史设计不再属于 Paper v8 canonical 协议：

- 真实校准和合成 master 使用 L504；
- 96/168/336/504 四视野、`fixed_l504` 和共享 L504 denominator；
- 仅接入 19 个数据集并排除 Restaurant；
- 把 tabpfn-ts3 或 TimePFN 放入正式默认模型集合；
- 真实窗口的三路永久切分、真实 feature range 或 conformal 的逐样本硬准入；
- nonlinear 使用专门的 12/64/128 path 特例或按结果挑选 seed；
- 将斜率误差作为 trend 的主机制分，或要求标准 DFM 通过 common strict relay。

如需追溯这些方案，只查看 Git 历史，不得从本清单恢复实现。
