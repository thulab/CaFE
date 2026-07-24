# Paper v8 全流程梳理与执行决策

## 文档定位

本文档记录大规模推理前对 Paper v8 全流程的逐阶段梳理结果和已确认决策。讨论完成后，以本文档作为校准、生成、对齐回验、推理和分析流程的实施依据。

本文档区分三类内容：

- **已决定**：后续实现必须遵循。
- **待讨论**：尚未形成执行结论。
- **暂缓**：当前 v8 不实施，不作为主流程阻塞条件。

当前范围只覆盖合成数据研究，不涉及时序预测模型评测平台的前后端。

## 总体目标

v8 的主测试采用以下原则：

- 真实数据只用于校准合成机制的经验特征范围和生成参数分布。
- 合成 future 完全由确定性机制产生。
- 主能力表使用确定性机制、样本参数随机化。
- robustness 表可以对观测 history 添加噪声，但评分使用 clean latent future。
- 正式预测 horizon 固定为 48。
- 每个样本生成 504 点 history 和 48 点 future，再从同一母本截取 96、168、336、504 四种 lookback。
- 生成后必须进行机制和真实经验范围的对齐回验。

## v8 核心流程

当前拟定的精简主链路如下：

```text
真实数据
  → 构造统一校准窗口池
  → 提取真实特征分布
  → 将特征分布映射到生成参数
  → 生成确定性合成 history + future
  → 特征对齐和机制有效性回验
  → 组织正式推理任务
  → 调用模型推理
  → 计算通用误差和机制指标
  → 汇总能力结果并分析
```

当前已按本文档完成单数据集、64-seed 的首轮全链路实现和 pilot。

## 决策 1：真实数据校准不做三路切分

状态：**已决定**

v8 不再将真实窗口永久切分为：

- generator parameter；
- gate reference；
- gate calibration。

给定一个真实数据集及其 task view，所有满足质量条件的候选窗口共同组成一个 calibration window pool。该窗口池的全集用于：

- 提取真实特征经验分布；
- 建立特征到生成参数的映射；
- 描述特征之间的联合关系；
- 标定能力强度；
- 作为生成后特征对齐回验的真实参照。

这里不存在预测模型训练意义上的数据泄漏，因为真实窗口不用于训练或评价下游预测模型，合成 future 也不是从真实窗口复制得到的。

使用同一真实窗口池进行校准和对齐回验时，回验结论应解释为：

> 合成生成器是否正确匹配了指定真实数据集的经验特征范围，即 construct alignment。

它不被解释为对未见时间段、未见序列或其他数据集的外部泛化保证。

## 决策 2：v8 必须保留的验证只有对齐回验

状态：**已决定**

v8 当前必须实施的是生成后的对齐回验，用于检查：

- 样本形状和数值合法，且没有重复或近重复样本；
- I1–I5 的主机制特征对强度参数有正确且非退化的响应；
- 每个能力最核心、不能由普通数值检查替代的机制约束成立。

真实特征范围只用于给生成机制提供合理尺度，不作为过强的样本准入条件。对齐回验的主语始终是合成生成器，而不是要求真实数据集先通过复杂的能力资格审计。

以下内容暂缓，不作为 v8 主流程的阻塞条件：

- 固定的 train/reference/calibration 三路切分；
- conformal coverage 保证；
- 独立 gate calibration；
- block bootstrap；
- group bootstrap；
- 时间后段 holdout；
- leave-one-group-out；
- 特征分位数置信区间和外部稳定性审计。

这些审计未来可以作为补充实验增加，但不应增加当前核心链路复杂度。

## 决策 3：真实校准视野固定为 504 点

状态：**已决定**

真实数据只用于 history-only 特征校准，因此每个真实 anchor 固定包含：

```text
504 点 history
```

不再为真实校准窗口附加 48 点真实 future，也不使用 336/168/96/48 的自适应校准视野。无法提供完整 504 点 history 的数据集或配置暂不进入 v8 正式实验。

“真实全集”指：

> 按确定的 task view、固定 504 点长度和质量条件能够构造出的全部合格 calibration anchor 候选。

真实 anchor 不再使用统一固定 stride。按每条序列可覆盖的非重叠 504 点单元计算数据集容量，再使用固定种子的分层随机抽样，最多保留 256 个通过质量检查的 anchor。抽样规则、seed、候选容量和实际数量必须写入产物。

## 决策 4：96/168/336/504 是同一母本的后缀视图

状态：**已决定**

合成样本仍严格生成 `504 history + 48 future` 母本。四种推理 lookback 必须共享同一个 48 点 future：

```text
L=96：  history [408, 504) + future [504, 552)
L=168： history [336, 504) + future [504, 552)
L=336： history [168, 504) + future [504, 552)
L=504： history [0,   504) + future [504, 552)
```

正式实验不为每种 lookback 单独生成样本。模型可以从同一 504 点母本中选择不同长度的可见历史，因此不同 lookback 的比较不会混入 future 或生成 seed 的差异。

## 决策 5：删除两类 embargo

状态：**已决定**

v8 核心流程不再需要以下两类设计：

### 母窗口尾部的 48 点 validation tail

旧流程加载：

```text
504 history + 48 benchmark future + 48 validation tail = 600 点
```

其中最后 48 点不会进入任务窗口，而且它仍可能被相邻滑动窗口使用，因此并未真正隔离相邻窗口。v8 将母窗口直接简化为 552 点，不再要求额外尾部。

### 不同数据角色之间的 552 点 temporal embargo

旧流程在单序列的 generator parameter、gate reference 和 gate calibration 之间保留 `context + horizon = 552` 点隔离。

v8 不再划分这些角色，因此删除相应 temporal embargo。

需要保留的概念只有：

- `real_calibration_context_length = 504`；
- `synthetic_context_length = 504`；
- `synthetic_horizon = 48`；
- `synthetic_master_length = 552`。

真实校准不再有固定 stride。合成任务的 `horizon` 不应被命名为 embargo。

## 决策 6：标准化只使用 history

状态：**已决定，后续需复核具体实现**

真实窗口的标准化统计量只从 history 计算，避免 future 反向影响模型可见 history 的尺度：

- 普通目标：每个变量使用自己的 history 均值和标准差；
- additive hierarchy：各变量分别中心化，所有变量共享能够保持加和恒等式的尺度；
- covariates：连续变量使用 history 统计量归一化，离散/事件变量保持其语义编码。

标准化后的特征分布表示真实数据的形态、动态和跨变量结构，不表示原始业务量纲。

对于 paired/counterfactual members，必须共享同一组标准化统计量，避免全局尺度差异混入机制比较。

## 决策 7：使用唯一的真实特征经验分布

状态：**已决定**

每个“数据集 × task view × capability”保存一份基于所有合格窗口的 history-only 特征矩阵。p05、p10、p25、p50、p75、p90、p95 只是该经验分布的摘要，不再分别构造互不一致的参数范围、强度范围和旧版 gate 范围。

主流程不再复用 v7 feature gate，也不再把 Mahalanobis 或 conformal coverage 作为 v8 的必要组成。

## 决策 8：样本参数直接使用真实 anchor，不压缩到 IQR

状态：**已决定**

每个合成样本从 calibration window pool 中选择一个真实 feature row 作为 anchor。除被实验主动干预的主能力参数外，其余可校准参数直接从该 anchor 的特征映射得到。

不再执行把真实经验 rank 从 `[0, 1]` 压缩到 `[0.25, 0.75]` 的逻辑，也不对各特征进行相互独立的边际采样。

同一 paired sample 的 I1–I5 必须共享：

- 真实 anchor；
- parameter seed 中与能力强度无关的部分；
- path seed；
- nuisance realization。

五档之间只允许主能力干预及其必然下游结果发生变化。这样既保留真实特征组合，又能把跨强度差异归因于被测试机制。

真实 anchor 只负责数据集校准的背景参数。生成器仍通过 path seed 随机化相位、符号、事件位置、变量载荷、lag、排列、局部形状等机制实例，因此 anchor 采样不等于复制真实曲线。

## 决策 9：由真实范围和生成器支持范围共同标定 I1–I5

状态：**已决定**

先在代表性 anchors 和 seeds 上扫描生成参数，得到：

```text
lambda → realized primary feature
```

这条 response curve 定义生成器对该主特征的可实现范围。真实 history 提供稳健参考区间，默认使用 `[q10, q90]`，并以 q50 为中心设置可配置的范围放大系数：

```text
range_expansion_factor = 1.0
```

系数为 1 时不放大真实区间；未来若需要增强压力，可围绕 q50 温和扩展，但最终目标必须限制在生成器可实现范围内。

将放大后的真实参考区间裁到生成器支持范围，得到有效目标区间，再在该区间内等距放置 I1–I5 的五个目标特征值。最后对 response curve 做反插值，得到五个实际生成参数或 lambda。

真实区间很窄时，默认保留这种温和校准结果，不额外要求数据集通过复杂的能力支持审计。是否需要将 `range_expansion_factor` 调大，由后续合成样本和模型响应 pilot 决定。

生成器的数学 `lambda` 坐标固定定义在 `[0, 1]`，但实际生成参数可以按各自含义超过 1。正式校准不能用累计最大包络掩盖原始 response curve 的回落：每个 `capability × family` 保存完整原始曲线，并自动识别从 `lambda=0` 开始的最大稳定单调分支，只在该有效支持域内反解 I1–I5。检测到的上界、foldback 容差和原始曲线都写入 calibration bundle，不为某个 family 硬编码一个跨数据集通用截断常数。

## 决策 10：真实范围和强度标定统一使用 history

状态：**已决定**

以下步骤统一使用 504 点 history 和同一套特征函数：

- 真实参数校准；
- 真实主特征分位数；
- 合成参数对齐；
- 合成强度对齐。

完整 552 点不再用于定义真实参数或强度范围，只用于各能力最核心的合成机制检查。

## 决策 11：生成有效性回验保持精练

状态：**已决定**

用于参数映射的 required feature 必须存在且为有限值；不可定义的特征不能以 `0.0` 冒充真实观测。除此之外，不为真实数据增加复杂的准入审计。

生成有效性只保留以下核心检查：

1. 基本合法性：shape、finite、history/future 长度正确。
2. 非重复性：样本哈希和必要的近重复检查。
3. 强度响应：主特征随 I1–I5 的总体方向正确，且不是完全不变。
4. 能力核心约束：每个能力只选择一到两个最能说明机制成立的检查，例如层级加和、cross-series 正确边/lag、covariate counterfactual response。

参数 clip 比例、唯一值数量和更完整的分布差异可以作为诊断输出，但不默认升级为硬门槛。

`predictable_intermittency` 的能力结构仍由确定性 event clock 保证，`intermittency_clock_incremental_r2` 和 realized `spike_rate` 保留为可观察诊断；但二者在有限 L504 窗口上均可能高度零膨胀，不适合作为五档反解坐标。正式主强度特征改为生成器可精确分解的 `event_effect_energy_share`：在 L504 history 内计算事件成分能量占“事件成分 + 确定性背景 texture”总能量的比例。该坐标连续、同一路径下严格单调；真实窗口仍校准事件时间尺度、宽度和背景 nuisance，事件剂量使用合成机制支持域等距标定。模型机制评分继续使用 event-window 预测误差，而不是拟合该生成器元数据。

`nonlinear_persistence` 继续以可观察的 `nonlinear_conditional_gain` 作为主强度特征，但 response support 不再只看少量路径的均值曲线。普通能力使用 12 条 response paths；nonlinear 专用 64 条路径，分别寻找稳定支持边界，并取逐路径支持边界的下 10% 分位与均值支持域的交集。若 secondary family 为匹配 primary 数值目标而把 sensitivity audit 的 I3–I5 压缩到不足其支持域的 30%，则 secondary 改用自身保守支持域内的五档相对网格；主表 primary 标定不受影响。

## 决策 12：替换退化的 covariate future correlation

状态：**已决定**

删除 `future_abs_covariate_target_corr`，不保留将不可定义值写为零的兼容路径。

真实校准继续使用仅在 504 点 history 上即可计算的 `covariate_incremental_r2` 作为主强度特征。它用于提供协变量作用的合理尺度，不承担严格证明真实协变量样本外预测价值的任务。

生成器扫描作用强度并测量 realized `covariate_incremental_r2`，再通过 response curve 反解 I1–I5，不要求从真实特征值解析求出协变量作用系数。

模型能力的主要诊断来自合成 counterfactual future：两个 member 共享 target history 和 past covariates，只改变 known-future covariates；正确的 future target 差异由协变量响应律唯一确定。真实校准与模型能力判定因此保持解耦。

## 决策 13：保留精简 feature gate，默认关闭 near-distance gate

状态：**已决定**

仓库已有 feature gate 实现，但现有版本基于旧三路切分、control-feature robust Mahalanobis 距离和 conformal threshold。clean-deterministic 模式只是从旧 gate 中投影掉噪声、异常值等随机控制特征，旧阈值不能直接作为正式 v8 gate。

v8 复用现有特征提取和 gate 接口，但正式语义精简为：

- required realized features 为有限值；
- primary feature 对 I1–I5 有正确且非退化的总体响应；
- 可选 control features 只做宽松的合理性诊断，不恢复旧版复杂的 conformal 准入流程。

能力专属的层级恒等式、跨序列边/lag、共同因子联合可识别性和协变量反事实响应继续作为 structural/identifiability gate，不混入通用 feature gate。

near-distance gate 在 v8 校准、生成、重试和 acceptance 中默认关闭：

```text
near_distance_gate_enabled = false
```

原因是合成机制不会复制真实窗口的具体曲线，原始序列和最近邻距离不能有效衡量机制 benchmark 的质量。v8 不要求 near-distance artifact，不要求 profile 同时具有 near-distance calibration，也不因 near-distance 结果重试或拒绝样本。

现有 near-distance 实现暂不删除，只保留为显式开启的可选诊断工具，不进入正式默认流程。

## 决策 14：v8 使用独立文件和扁平 seed pool

状态：**已决定**

v8 不继续扩展旧 `synthetic_generation_service.py` 的生成、校准和 gate 分支。全流程使用名称中明确带 `v8` 的新文件，旧 v2-v7 文件只作为实现参考。

建议的核心文件边界为：

```text
backend/app/services/synthetic_v8_generation.py
backend/app/services/synthetic_v8_feature_gate.py
scripts/calibrate_paper_v8.py
scripts/generate_paper_v8_samples.py
scripts/validate_paper_v8_samples.py
scripts/run_paper_v8_inference.py
scripts/analyze_paper_v8.py
```

最终文件数量可以在实现时合并，但不能让正式 v8 重新依赖旧 generation service 的 mandatory near-distance、三路切分或旧 acceptance/retry 语义。

正式样本生成阶段唯一的样本预算参数是 `seed_count`，同时提供一个不改变预算、只确定种子身份范围的 `seed_start`：

```text
seed_start = K    # 默认 0
seed_count = N
```

本次生成使用全局 seed indexes `[K, K+N)`。`seed_count` 表示每个 `capability × primary family` 的 master seed-group 数，而不是 JSONL 行数：

- 同一 seed-group 共享 anchor 和 path realization，并产生 I1–I5；
- covariate 主任务在同一 seed-group 下产生配对 members；
- common-factor 和 cross-series 主任务只生成事实样本；严格反事实只在抽样 seed 的 I5 诊断表生成配对 members；
- common-factor 和 cross-series 另派生保持边际均值/标准差的跨变量输入消融样本，future 保持不变；
- 每个 master sample 只存储 504+48 的母本；
- 96/168/336/504 views 在后续推理准备阶段切出；
- noisy-history robustness 样本从 clean primary 样本派生，不单独建立生成轮次。

response-curve 使用的内部 seeds 属于校准阶段，随 calibration bundle 冻结，不作为正式生成命令参数。secondary-family 在 I3/I5 和部分 seeds 上的敏感性策略也由冻结配置确定，不增加 generation round。

secondary/robustness seed 子集只由每个 seed 自身的 stable hash 决定。小 seed pool 若没有命中允许审计子集为空，不能为了凑样本临时选择第一个 seed；否则扩展 seed pool 时会破坏前缀稳定性。

v8 删除以下统计轮次身份：

- `round_index`；
- `round_seed`；
- `samples_per_round`；
- 在生成阶段预先写入的 `analysis_block_id`。

正式样本身份只需要稳定记录：

```text
capability_id
family_role
intensity
seed_index
counterfactual_member（如适用）
generator_version
```

相同 generator/calibration 版本和相同 seed index 必须生成相同内容。由此支持两种操作：

```text
# 重建或扩展同一个 pool：前 32 个保持不变
seed_start=0,  seed_count=32
seed_start=0,  seed_count=64

# 生成两个互不重叠的新批次
seed_start=0,  seed_count=32
seed_start=32, seed_count=32
```

不得用当前日期或运行时间隐式改变随机种子，否则同一命令无法复现。若 generator version 或 calibration bundle 改变，即使 seed index 相同也属于不同样本版本，manifest 必须阻止它们被误合并。

生成完成后，再由独立的推理准备步骤根据 sample manifest 切分：

- inference shards；
- analysis blocks；
- pilot/full subsets；
- lookback views。

这些切分只改变任务组织和索引，不重新生成样本。为了断点恢复允许物理文件分 chunk 写出，但 chunk 只是执行细节，不具有统计 round 语义。

## 决策 15：三机模型级并行推理

状态：**已决定**

v8 正式推理默认使用当前可用的两到三台服务：

```text
http://127.0.0.1:10810
http://192.168.99.17:10811
http://192.168.99.18:10810
```

运行前进行简单健康检查，只将可用服务加入本次调度。并行分为两层：

1. 服务之间按模型队列并行，同一服务一次只加载一个模型；
2. 单个模型内部使用冻结的 `replicas_per_device` 和 `http_concurrency`。

沿用已验证的模型执行配置：

| model | replicas/device | HTTP concurrency |
|---|---:|---:|
| Timer-3.5 | 1 | 64 |
| Timer-3.0 | 1 | 32 |
| Chronos-2 | 4 | 32 |
| moirai2 | 2 | 16 |
| toto2.0 | 2 | 16 |
| timesfm2.5 | 8 | 32 |
| tirex2 | 1 | 32 |
| tabpfn-ts3 | 8 | 24 |

三台服务都可用时，先以历史耗时做模型级 longest-processing-time 分配；服务数量减少时重新平衡模型队列。若某个模型排在一台服务的队尾、其他兼容服务会提前空闲，则默认启用尾部协作：

- 正式七模型实验先对 Chronos、TiRex、Moirai、Timer 等较快模型做 LPT 分配，再将 `toto2.0`、`timesfm2.5`、`tabpfn-ts3` 作为慢尾部任务分散到三台可用服务；
- 三台服务和全部模型均可用时，预期队列分别以 `timesfm2.5`、`toto2.0`、`tabpfn-ts3` 结尾；不足三台服务或某服务缺少模型时，只在兼容服务间重新平衡；
- 按 `model_id × sample_id` 的 stable hash 将该尾部模型任务确定性分片；
- 各服务完成自己的前序模型后处理一个 tail part；
- 每个 part 使用独立任务文件、预测文件和状态文件，禁止并发写同一文件；
- 汇总时验证 part hash、样本 ID 唯一性和完整覆盖，再生成模型 canonical prediction；
- resume 从所有 part 的已完成 ID 继续，服务可用性减少时允许一台服务顺序接多个 part。

已有 canonical prediction 的模型不在 resume 时重新分片，避免为已经完成或部分完成的旧式 shard 重复推理。可用 `--disable-tail-sharding` 显式关闭该优化。

每台服务写入独立 inference shard，任务使用确定性 ID：

```text
model_id × sample_id × context_length
```

`--resume` 模式下成功结果 append-only，重启时跳过已完成任务，只重试失败或缺失任务；已有 task manifest、generation config 和 tail-part hash 必须一致。不带 `--resume` 时，当前数据集与当前 seed shard 的 inference 目录必须精确重建，不能因 sample ID 相同而静默复用另一版生成器的预测。全部模型完成后验证输入 manifest/hash、模型覆盖、任务行数和状态，再合并 shard。

naive 和 seasonal-naive 在本地计算，不占用推理服务。模型输入适配继续沿用已登记策略，包括不支持原生多变量的模型按变量拆分后重组，以及模型不支持已知未来协变量时省略协变量。

## 决策 16：同时报告 Oracle context、固定 L504 和 split-bank 稳定性

状态：**已决定**

每个模型和能力同时报告两种 context policy：

- `fixed_l504`：所有样本固定使用 504 点 history；
- `oracle_context`：在 96/168/336/504 中按 MASE 选择最佳 view。

Oracle context 是允许模型选择最适视野的乐观能力上界，不替代固定 L504 的受控比较。对 counterfactual pairs，两个 members 必须共享同一个 oracle context，该 context 由 pair 两个 members 的平均 MASE 决定，避免用不同视野拼接反事实 effect。

稳定性分析以 master `seed_index` 为切分单位：

- 同一 seed 的 I1–I5 不拆开；
- counterfactual members 不拆开；
- 四种 context views 不拆开；
- 所有模型使用完全相同的 batch membership。

按可用 seed pool 依次考察 `N=32,64,128,...`，每个 N 将排序后的 seed groups 切成不重叠 batch，分别重算模型平均得分和排名。v8 不恢复旧 round 身份，也不默认增加随机重复切分、partial order、practical-tie 或 bootstrap 层。

split-bank 核心输出暂定为：

- 模型 batch 间相对得分差；
- 模型排名 Kendall tau-b；
- Top-1 一致率；
- 必要时附 Top-3 overlap。

所有结果分别对 `fixed_l504` 和 `oracle_context` 计算。最终表格中通用 MASE 与能力专属机制指标如何共同展示，继续讨论后冻结。

正式能力表不将 MASE 和机制指标合成为任意加权总分：

- `accuracy_score`：I1–I5 等权的 seed-group mean MASE，越低越好；
- `mechanism_score`：I5 的能力专属主机制误差，越低越好；
- 分别报告 `accuracy_rank` 和 `mechanism_rank`。

counterfactual 能力先在 pair 内聚合 MASE，再按 seed-group 和 intensity 聚合；机制指标直接按完整 pair 计算。层级能力以 child-contrast NMAE 为主机制指标，并把 coherence NMAE 作为必报辅助项。

十个能力的主机制指标为：

| capability | primary mechanism metric |
|---|---|
| trend | `trend_slope_relative_abs_error` |
| multi_seasonal | `seasonal_spectral_amplitude_relative_error` |
| time_varying_seasonality | `instantaneous_frequency_nmae` |
| regime_switching | `regime_jump_nmae` |
| nonlinear_persistence | `nonlinear_recurrence_residual_nrmse` |
| predictable_intermittency | `event_window_nmae` |
| common_factor | `common_component_nmae`（事实结构恢复） |
| hierarchical_coherence | `child_contrast_nmae` |
| cross_series_dependence | `responder_normalized_mae`（事实 responder 预测） |
| covariate_response | `counterfactual_effect_nrmse` |

每个能力分别列出 Oracle/L504 的 MASE、accuracy rank、主机制得分和 mechanism rank；另提供跨能力矩阵。naive/seasonal-naive 作为参考行显示，但不进入 foundation-model 正式排名。primary family 进入主表，secondary family 与 noisy-history robustness 分表报告。

`common_factor` 和 `cross_series_dependence` 还必须单列跨变量输入消融：

- common-factor 只评分 history 完全未改的 protected target，比较完整联合输入与辅助通道 donor 替换后的 `protected_target_nmae`；
- cross-series 只评分 history 完全未改的 responders，比较完整 driver 与 forecast-covering driver block donor 替换后的 `responder_normalized_mae`；
- donor 来自相同 capability、family 和 intensity 的另一 seed，替换段按 recipient 的均值和标准差做 affine matching；
- control 与 ablation 在 Oracle 分析中使用 control 选择的同一个 context；
- 正的误差增量表示模型使用了被消融的跨变量信息，但必须和事实预测误差一起解释，不能单独把“对错误输入敏感”当成能力。

common/cross 的 I5 strict counterfactual effect NRMSE 只作为高难度诊断审计，不并入主能力分。首轮模型 pilot 显示该指标对多数现有 zero-shot 模型接近 1；若把它作为主分，会把“没有恢复任意 in-context 反事实映射”误写成所有模型共同失去普通动态因子或 lead-lag 预测能力。

上述正式结果以数据集为独立报告单位：

```text
dataset → capability → model score/rank
```

不要求把不同数据集等权或按样本量加权压成一个正式总分。跨数据集平均、总体模型排名和 pooled split-bank 只作为开发期预实验诊断，必须明确标记为 preliminary，不作为论文对模型能力强弱的最终结论。

split-bank 优先在每个数据集内部对 accuracy score 和 mechanism score 分别计算，核心稳定性指标为 batch 间相对得分差、Kendall tau-b、Top-1 一致率，并将 Top-3 overlap 作为辅助。若某数据集的 seed pool 不足以支持某个 N，则不计算该档，不通过跨数据集合并来补足。

## 决策 17：区分真实背景校准和结构能力强度标定

状态：**已决定**

真实数据校准是合成机制的背景约束，不是要求每个真实数据集预先具备十种能力语义。正式 v8 将校准分成两层：

1. 所有能力都从当前数据集的 504 点 anchor pool 校准边际形态和 nuisance，包括尺度无关的趋势、周期、自相关、稀疏性以及生成器实际使用的其他背景特征。
2. 六个单变量能力继续使用当前数据集的真实主特征 `[q10, q90]` 与生成器 response-curve 支持范围的交集标定 I1–I5。
3. `common_factor`、`hierarchical_coherence`、`cross_series_dependence` 和 `covariate_response` 不要求普通 GIFT 数据集提供真实层级、边、共同因子或 known-future covariate 语义。它们使用冻结且跨数据集一致的生成器 response curve 和结构强度目标标定 I1–I5；真实 anchor 只负责背景与 nuisance。

因此正式实现必须把 feature contract 显式拆成：

```text
background_features
primary_strength_feature
structural_identifiability_features
```

结构能力的 `primary_strength_feature` 和 identifiability gate 从合成样本测量，不再列入普通真实 anchor 的 required features。该设计确保结构题在不同数据集之间具有一致难度，同时保留“该样本由哪个真实数据集校准背景”的来源语义。

## 决策 18：四种 context 共用 L504 MASE 分母

状态：**已决定**

Oracle context 比较中的 L96、L168、L336 和 L504 是同一个 552 点 master sample 的不同可见 history suffix，不是四个独立样本。若分别用各 suffix history 计算 MASE 分母，context 选择会同时受到预测误差和分母变化影响，不能解释为模型对视野长度的选择。

正式 v8 对每个 master sample 使用完整 L504 history 预计算一次 MASE denominator，并将其保存到 sample/view metadata。四个 context views、counterfactual members 的各自 target channel 都复用由对应 clean L504 master history 得到的 denominator；robustness 样本也沿用 clean latent history 的 denominator，不用加噪 history 重新定标。

日历季节由 canonical config 的原生频率确定，但不再重载为生成时间尺度和 MASE period。正式字段拆分为：

- `calendar_season_length`：频率推导的日历周期，只作来源语义和可观测时的 calendar-season 特征；
- `feature_period`：L504 内至少可观察的真实特征周期；若完整日历周期在 L504 内不足两次，则使用窗口频谱主时间尺度；
- 生成机制时间尺度：从直接 anchor 的 `profile_dominant_period` 出发，按能力的可识别范围裁剪；
- `mase_period`：日历周期在 L504 内可定义时沿用，否则明确使用 non-seasonal lag 1。

因此 10 秒 BizITObs 保留 `calendar_season_length=8640`，但使用窗口内可观察的 feature/generator period 和 `mase_period=1`。四种 context 仍共享 clean L504 MASE denominator，不能按 suffix 静默改变 lag。Oracle context 仍按 MASE 选取，但 counterfactual pair 的两个 members 共用由 pair 平均 MASE 最小化得到的 context。

## 决策 19：common-factor 与 cross-series 使用稠密事实机制，配对反事实降为审计

状态：**已决定并完成单数据集 64-seed 回验**

旧 common-factor 使用 8–12 点短 code block，旧 cross-series 使用少量历史事件加边界处第 4 个事件。虽然正控可以恢复机制，但 zero-shot 模型容易把最终局部结构当成异常，严格 paired effect 对所有模型接近不响应，不能作为主能力题。

正式 primary family 调整为：

1. `common_factor = dense_dynamic_factor_with_joint_state_relay`
   - 五个 target 全路径包含确定性动态共同因子和异质局部成分；
   - I1–I5 用 history `pca_top1_explained` 的生成器实际支持范围等距定标；
   - 24–32 点长观测块反复教授 rank-2 联合状态方程，只在抽样 seed 的 I5 pair 中沿 protected channel 的 null direction 改变状态；
   - 主事实表报告共同成分恢复，跨变量使用由 protected-target donor ablation 判断，严格 state-relay effect 只作审计。
2. `cross_series_dependence = dense_delayed_linear_scm`
   - driver 的完整 history 使用平滑随机 knot path 形成稠密连续激励，不再使用孤立脉冲；
   - primary responder 使用固定混合符号 `[+1,-1,...]`，并保留低强度确定性局部背景；
   - `lag = horizon = 48`，因此 responder future 的 48 点全部由最后 48 点可见 driver 覆盖；能力重点是跨序列使用，不额外混入长 lag 搜索难度；
   - I1–I5 以 `lead_lag_peak_abs` 的实际生成响应等距定标，正确 driver/lag、chronological holdout R² 和正控 effect recovery 作为独立结构 gate。

两种机制的严格 pair 和标准化统计量继续共享不变量前缀，保证 effect 比较不混入全局尺度差异。输入消融不是第二套 ground truth：它只破坏模型可见 history 的跨变量对齐，评分仍使用原 clean future。

## 当前审计：GIFT-Eval 接入和窗口上限

状态：**现状已核清，v8 核心方案与 canonical config 清单已冻结**

### GIFT-Eval 的“23 个数据集”与本地配置

GIFT-Eval 论文中的 23 个数据集是逻辑数据集口径。官方数据包将其中许多数据集按频率继续展开，本地标准配置共 55 个 `dataset × frequency` 配置；M4 Yearly、Quarterly、Monthly、Weekly、Daily、Hourly 在论文的数据集总数中合并计为一个 M4 逻辑数据集。

23 个逻辑数据集为：

1. Jena Weather；
2. BizITObs Application；
3. BizITObs Service；
4. BizITObs L2C；
5. Bitbrains Fast Storage；
6. Bitbrains RND；
7. Restaurant；
8. ETT1；
9. ETT2；
10. Loop Seattle；
11. SZ-Taxi；
12. M_DENSE；
13. Solar；
14. Hierarchical Sales；
15. M4；
16. Hospital；
17. COVID Deaths；
18. US Births；
19. Saugeen；
20. Temperature Rain；
21. KDD Cup 2018；
22. Car Parts；
23. Electricity。

### 当前仓库实际怎么接入

当前正式链路并未接入完整 23 个逻辑数据集：

- `build_paper_v4_profile_suite.py` 只注册了 12 个 GIFT-Eval 小时级配置和 M4 Hourly；
- v7 九能力构建器复用了这批单变量来源，再额外注册 Swiss、GEFCom、M5 等结构化来源；
- v8 pilot 进一步缩到每个能力一个开发期数据集，六个单变量能力都使用 `gift_electricity_h`。

当前 GIFT Arrow 读取器只读取：

- `item_id`；
- `target`；
- `freq`。

它忽略 `start`，也忽略部分 GIFT 配置中存在的 `past_feat_dynamic_real`。单变量 task view 会把原生多变量 target 拆成各通道；panel task view 只取预先指定的前若干通道。

当前 v7 正式构建路径还会：

- 在切窗前排除 GIFT-Eval 官方 test tail；
- 加载 `504 history + 48 benchmark future + 48 validation tail`；
- 使用 48 点步长；
- 在展平后的所有候选窗口上做确定性等距抽样；
- 先截到 `max_windows`，再做 finite/informative 检查，因此无效窗口不会回填；
- 对含任意非有限值的窗口整窗拒绝。

这不是严格的序列均衡抽样。长序列因为拥有更多候选 origin，会自然获得更高权重。`BucketSpec.max_series` 对 GIFT loader 实际没有生效，真正限制结果规模的是 task view 级别的 `max_windows`。

仓库中的上限目前不统一：

- v4 profile builder 默认每数据集 240 个母窗口；
- v7 九能力构建器代码默认每 task view 120 个；
- 当前冻结的 v7 产物实际使用 160 个；
- 因为先截候选、后做质量检查，当前 v7 GIFT task view 实际经常只有 127–160 个窗口。

另一个旧实现 `build_paper_v4_profile_suite.py` 已经有序列均衡和备用候选回填，但该逻辑没有进入 v7 九能力使用的通用 GIFT loader。两个路径对缺失值的处理也不一致：前者允许受控插补，后者整窗拒绝。

所有当前单变量 GIFT 来源还硬编码 `season_length=24`。这只适合当前选取的小时级子集，不能直接扩展到秒、分钟、日、周、月、季度和年度配置。

### 固定 504 点校准视野与数据集资格

v8 不再为了覆盖完整 23 个逻辑数据集而引入自适应真实校准长度。真实 anchor 必须具有完整 504 点 history；不满足的数据集或配置暂不测试。

按当前数据长度，Restaurant、Hospital、COVID Deaths 和 Car Parts 没有任何 504 点原生序列，因此暂不进入正式 v8。M4 Yearly 也不能形成 504 点 anchor，正式 canonical config 固定采用 M4 Hourly。正式报告必须写明实际纳入的数据集清单，不能把可用子集表述为完整 23 数据集结果。

正式纳入的19个 canonical config 为：

- 小时频率、`calendar_season_length=24`：Jena Weather H、BizITObs L2C H、Bitbrains Fast H、Bitbrains RND H、ETT1 H、ETT2 H、Loop Seattle H、SZ-Taxi H、M_DENSE H、Solar H、M4 Hourly、KDD Cup 2018 H、Electricity H；
- 日频、`calendar_season_length=7`：Hierarchical Sales D、US Births D、Saugeen D、Temperature Rain D；
- 10秒频率、`calendar_season_length=8640`：BizITObs Application、BizITObs Service。

### 已确认的 v8 接入方案

1. 新建 v8 专用 GIFT registry，主身份为 23 个逻辑数据集；每条 anchor 同时保留 `dataset_id`、`config_id`、`frequency`、`item_id`、`channel_id` 和 `window_start`。
2. 每个逻辑数据集预注册一个支持 504 点的 canonical config，避免同一原序列的多个下采样版本被重复计权。canonical config 优先保证 504 点内包含有意义的动态/季节尺度、具有足够 origin 且缺失可控；具体清单在实现前冻结。无法找到合格 canonical config 的逻辑数据集标记为未纳入。
3. 不再排除官方 test/validation tail，因为 v8 真实数据只做校准；使用完整可用序列构造 calibration pool。
4. 合成母本固定为 504+48；真实校准固定为 history-only 504 点。两者的校准和对齐特征统一在完整 504 点 history 上测量。
5. 每个逻辑数据集最多保留 256 个**通过质量检查后的** anchor；上限作用于接受窗口而不是候选窗口。真实校准不再使用固定 `stride=48`。对每条序列或连续有效片段 `s`，定义非重叠覆盖容量：

   ```text
   C_s = floor(T_s / 504)
   C_dataset = sum_s C_s
   N_anchor = min(C_dataset, 256)
   ```

   不能先把不同序列长度相加再除以 504，因为窗口不能跨越序列边界。原生 panel 的同一时间窗口只计一次；拆成单变量的 channel 则按独立序列计数。

   将每条序列按 `C_s` 划成覆盖全时间范围的 strata，每个 stratum 提供一个带固定种子 jitter 的候选 origin；再从所有 strata 中无放回抽取 `N_anchor` 个。抽样使用冻结的 `calibration_sample_seed` 或稳定 hash，保证同一数据、配置和 seed 可完全复现。质量失败后从尚未选中的 strata 回填。

   这种抽样等价于按可覆盖观测长度加权，但不会让大量高度重叠的窗口伪装成大量独立 anchor。大数据集的有效步长会自动变长，不需要人为设置统一 stride。
6. 256 是上限而不是最低配额。候选不足的数据集保留全部有效窗口并记录实际数量，不复制窗口凑数。
7. 缺失值配置使用一个统一、可审计的轻量插补策略，不再因单个缺失点整窗拒绝，也不把插补后的窗口伪装成完全观测；产物记录 `observed_fraction`。
8. 日历季节从原生频率解析并随 anchor 保存，不再全局硬编码为 24；同时显式保存 `feature_period` 和 `mase_period`，避免将日历周期、生成周期与评价周期混为一个字段。
9. 同一逻辑数据集的 calibration pool 由十个能力共享，不再为每个 capability 重复切一套真实窗口。
10. 对没有真实多变量、层级或 known-future covariate 语义的数据集，真实 anchor 只校准边际背景和 nuisance；共同因子、层级、cross-series 和 covariate 的识别结构由合成机制定义，不要求真实数据先具备相同结构。
11. 每个 clean、robustness 和 inference sample 都必须直接携带数据集来源字段，不能只在汇总文件中保存。最低字段为 `dataset_id`、`config_id`、`task_view_id` 和 `profile_id`；正式 v8 另保存 `anchor_id`，并能回查 `item_id/channel_id/window_start`。正式多数据集 `sample_id` 也必须包含 `dataset_id` 或其稳定短标识，避免不同数据集的同能力、同强度、同 seed 发生 ID 冲突。

因此真实 anchor 数会自然随数据集大小变化：

```text
N_anchor(dataset) = min(sum_s floor(T_s / 504), 256)
```

这种变化只属于真实校准阶段。anchor 较少的数据集可以在不同 path seeds 下复用同一真实背景，但不能复制 anchor 来伪造更大的真实经验样本量。

## 第一阶段实施时必须处理的问题

状态：**已处理**

### v8 不应继续读取 v7 gate-reference 子集

当前 pilot 直接读取 v7 `real_source_samples.jsonl`，该文件来源于旧 near-distance artifact 的 `reference_raw`，只覆盖旧三路切分中的 gate-reference 子集。

正式 v8 应从原始数据集重新构造统一 calibration window pool，或者冻结一份包含全部合格窗口的新 v8 校准产物。

当前实现已从 `/root/xmy/gift-eval/electricity/H` 原始 Arrow 数据重建 504 点统一 calibration pool，不读取 v7 gate-reference 子集。

## 正式实验存储协议

状态：**已决定**

正式产物统一写入 `runtime/paper_exp/v8`，每次完整协议运行创建一个不可变实验目录：

```text
runtime/paper_exp/v8/
  <experiment_id>/
    experiment_manifest.json
    pipeline_status.json
    <dataset_id>/
      01_calibration/
      02_generation/
      03_inference/
      04_analysis/
```

默认实验标识为：

```text
v8_<generator-version>_<protocol-hash-prefix>_<created-at-utc>
```

`experiment_manifest.json` 在运行任何数据集前写入，只保存不可变身份、完整科学协议、协议 SHA-256、代码版本和存储约定；同名目录已经存在时必须逐字段验证协议一致，禁止覆盖成另一套实验。执行端点仅作为运行环境 provenance，不进入科学协议哈希。

`pipeline_status.json` 是唯一允许原地更新的根状态文件，记录当前数据集、当前步骤、已完成数据集和失败原因。各数据集继续独立保存校准、生成、回验、推理和分析产物，不默认生成跨数据集平均或排名。

本阶段不实现跨 seed shard 的组合分析。生成 shard 仍按 `[seed_start, seed_end)` 命名；将来由独立的 suite/analysis manifest 引用多个已存在 shard，不修改本次实验身份和已有产物。

## 正式实现清单

- 新建 v8 GIFT registry，冻结每个可用逻辑数据集的 canonical config、频率、season length 和数据源 hash。
- 从原始数据重新建立 504 点 calibration pool；实现分层固定种子抽样、质量失败回填、轻量缺失值插补和完整 anchor provenance。
- 将真实 feature contract 拆成背景、单变量主强度和合成结构可识别性三层；删除 `future_abs_covariate_target_corr`。
- 实现直接 anchor 参数映射、response-curve 反解 I1–I5、扁平 `seed_start/seed_count` 生成和 stable sample ID。
- 实现独立 v8 feature/structural gate、非重复检查、强度响应回验和 manifest/hash 校验。
- 实现 clean primary、I3/I5 secondary-family sensitivity 和 noisy-history robustness 的冻结样本组织。
- 实现三服务健康检查、模型级 LPT 调度、模型内冻结并发、append-only shard、断点恢复和严格合并。
- 实现共享 L504 MASE denominator、fixed-L504/oracle-context、能力机制指标和每数据集 split-bank 分析。
- 所有 calibration、generation、validation、inference 和 analysis 产物保存 schema/version/config/input hashes，阻止跨版本误合并。

## 变更记录

- 2026-07-24：建立文档；确认真实全集校准、仅保留对齐回验、合成 552 点母本、共享 future 的四种 lookback，以及删除三路切分和两类 embargo。
- 2026-07-24：确认统一经验特征分布、直接 anchor 参数采样和 history-only 校准。
- 2026-07-24：将强度标定精简为“真实 q10-q90 参考区间 × 默认 1.0 放大系数 × 生成器可实现范围”，通过 response curve 反解 I1–I5；将生成回验缩减为合法性、非重复性、强度响应和少量能力核心约束；covariate 保留 history `covariate_incremental_r2` 校准并删除退化的 future correlation。
- 2026-07-24：确认 v8 保留精简 feature gate，near-distance gate 默认关闭并从校准前置条件、生成 acceptance 和重试链路中移除。
- 2026-07-24：确认 v8 全流程使用独立且名称带 v8 的文件；正式生成使用一个扁平 master seed pool，以 `seed_count` 表示样本预算、`seed_start` 表示可复现且可追加的种子范围，不再生成统计 rounds，所有 inference/analysis 切分在生成后完成。
- 2026-07-24：确认 v8 使用两到三台推理服务做模型级并行；每台服务一次加载一个模型，模型内部沿用已验证的 replica/HTTP concurrency，结果按服务独立落盘后校验合并。
- 2026-07-24：确认分析同时包含 Oracle context 和固定 L504，并按完整 seed-groups 做 32/64/128/... split-bank 得分与排名稳定性；能力表分别报告五档平均 MASE accuracy score/rank 与 I5 mechanism score/rank，不合并成单一加权总分。
- 2026-07-24：核清当前 GIFT-Eval 接入只覆盖开发期子集，论文的 23 个逻辑数据集对应 55 个官方频率配置；记录当前 loader 的 tail 排除、展平抽样、质量检查后不回填、缺失值整窗拒绝和 seasonality 硬编码问题；曾讨论自适应真实校准视野，随后由下一条决策废止。
- 2026-07-24：放弃自适应真实校准长度；真实 anchor 固定为 504 点 history，不足 504 点的数据集/配置暂不测试。真实 anchor 数按 `min(sum floor(T_s/504), 256)` 随有效数据规模变化；正式结果按数据集分别报告，跨数据集总分和排名仅作为预实验。确认每个生成、robustness 和 inference sample 必须直接保存数据集来源和可回查的 anchor provenance。
- 2026-07-24：确认真实校准分成背景/nuisance 与主能力强度两层；六个单变量能力使用数据集内真实范围标定，四个结构能力使用跨数据集冻结的合成 response curve 标定，普通真实数据不再被要求具备结构语义。
- 2026-07-24：确认同一 master sample 的 L96/L168/L336/L504 共用 clean L504 history 计算的 MASE denominator；Oracle context 只比较预测误差，不允许分母随 context 改变。
- 2026-07-24：确认生成器数学 `lambda` 保持 `[0,1]`，实际机制参数不受统一上界 1 限制；校准从完整原始 response curve 自动识别最大稳定单调支持域，不能用单调包络掩盖 foldback，也不能把单数据集观测到的截断点硬编码为普遍常数。
- 2026-07-24：曾将 `predictable_intermittency` 的强度坐标改为 realized `spike_rate`，但 ETT2 全量预检证明其与 event-clock R² 一样可能零膨胀；最终改为连续且可精确分解的 `event_effect_energy_share`，真实 spike/clock 特征只作诊断。
- 2026-07-24：首轮全链路 pilot 发现并修正三类执行/分析边界：secondary seed fallback 会破坏前缀稳定性；短 context view 的 regime/intermittent 索引 metadata 必须随裁窗平移；event window 覆盖完整 future 时不能对空 background 求均值。
- 2026-07-24：推理调度从纯模型级 LPT 扩展为“模型级 LPT + 队尾同模型确定性多机分片”，每个 endpoint part 独立落盘并在严格覆盖校验后合并。
- 2026-07-24：split-bank 只在至少两个 batch 时报告稳定性，并补充 batch 间相对得分差和 Top-3 overlap；secondary family 与 observation-noise robustness 必须和相同 seed/intensity 的 clean primary 做 matched-control 比较。
- 2026-07-24：common-factor 主机制改为五变量稠密动态因子和长块 joint-state relay；cross-series 改为稠密连续 driver、混合符号 responders 和 horizon-aligned lag。主事实样本不再全量成对，严格反事实只在抽样 seed 的 I5 审计。
- 2026-07-24：为 common/cross 新增 affine-matched donor input ablation；common 只评分 protected target，cross 只评分 responders，Oracle control/ablation 强制共用 context。
- 2026-07-24：修复 covariate secondary 的强度与背景混杂：移除额外 `1.7` 放大，保持 primary response 数值路径不变，将 secondary response 的 history 均值和标准差仿射匹配到同 seed 的 primary reference，并用生成器已知的 `covariate_effect_variance_share` 标定 I1–I5。matched seed 的两族严格共享 weather driver、事件、baseline 和符号，只将即时线性响应替换为半线性饱和、一期 lag 与 distributed-event 响应，避免原 spline driver 改变 seasonal MASE denominator。`covariate_incremental_r2` 保留为可解释的线性审计特征，不再作为跨 family 的剂量坐标。
- 2026-07-24：结构能力 I1–I5 改为在生成器实际 realized-strength 支持内等距，再反解 lambda；cross 的强度轴与正确边/lag gate 解耦。
- 2026-07-24：修正非 resume 推理会静默复用同 ID 旧预测的问题；非 resume 精确重建当前 inference seed shard，resume 才执行 hash 校验与增量续跑。
- 2026-07-24：冻结19个满足 L504 的 GIFT-Eval canonical 配置：13个小时频率、4个日频和2个10秒频率；Restaurant、Hospital、COVID Deaths、Car Parts 因无504点原生序列暂不纳入。拆分 calendar season、feature period、能力专属生成时间尺度和 MASE period；10秒配置明确使用 `calendar_season_length=8640`、窗口内可观察生成尺度与 `mase_period=1`。生成时间尺度按 L504/H48 的可识别次数裁剪，MASE 不加隐藏 floor，并并列保存 history-std-normalized MAE 与 denominator 分布审计。
- 2026-07-24：正式存储根目录固定为 `runtime/paper_exp/v8/<experiment_id>`；根 experiment manifest 以完整协议哈希为不可变身份，运行进度单独写入可更新的 pipeline status。本阶段暂不实现多个 seed shard 的组合分析。
- 2026-07-24：正式推理模型冻结为 Chronos-2、toto2.0、timesfm2.5、tabpfn-ts3、tirex2、moirai2、Timer-3.5；先对较快模型做 LPT，再将 Toto、TimesFM、TabPFN 分散为三条队列的慢尾部任务，使先结束前序队列的服务可参与队尾协作。
- 2026-07-24：ETT2 正式运行在生成回验阶段暴露两个剂量问题：intermittency 的 thresholded spike rate 退化为零，nonlinear 的 12-path 均值支持域不能外推到64个生成 seeds。修复后普通能力保留12条校准路径，nonlinear 使用64条路径的下10%保守支持边界；压缩的 nonlinear secondary 匹配改用其支持域相对网格。ETT2 64 seeds 的 primary/secondary dose、结构、robustness 和 ablation 回验全部通过。
