# CapTS-Bench 真实锚点与 Synthetic v2 方法

更新日期：2026-07-18

本文描述仓库当前的 capts-paper-v2 实现，而不是早期 synthetic pilot。当前冻结标尺为：

```text
canonical_scale_id          = synthetic-v2-paper-v2-shortcut-resistant-2026-07-18
canonical_scale_fingerprint = 715a6bb980f0e4aa
generator_version           = capts-paper-v2
```

方法的代码与机器可读事实分别以以下内容为准：

- 生成与在线验收：`backend/app/services/synthetic_generation_service.py`
- 绝对强度与 bucket conditioning：`backend/app/services/synthetic_generator_conditioning.py`
- 三个冻结 artifact：`synthetic_v2_generator_conditioning_artifact.json`、`synthetic_v2_feature_gate_artifact.json`、`synthetic_v2_near_distance_artifact.json`
- 标尺冻结协议：`docs/superpowers/specs/2026-07-16-synthetic-v2-canonical-reference-v1-freeze.md`
- v2 shortcut 审计：`docs/superpowers/baselines/2026-07-18-capts-paper-v2-shortcut-audit.md`
- 旧 E1 协议（须按 v2 重跑）：`docs/superpowers/specs/2026-07-16-paper-e1-method-validity-protocol.md`

旧文档中的 11 能力集合、bucket-local intensity、p95 乘启发式倍率、12 次失败后仍保存样本、DCR/NNDR 仅离线等说法均不再代表当前实现。

## 1. 方法目标与证据边界

CapTS-Bench 使用动态生成的测试样本评估时序模型在不同结构上的预测能力。它解决的不是“复刻一条看起来像真实数据的曲线”，而是同时满足：

```text
可解释且可预测的结构机制
    + 真实数据条件化的 nuisance 与控制特征支持
    + 与已提交真实 reference 保持足够近邻距离
    + seed 可复现、批次可重新生成
```

真实数据与合成数据承担不同角色：

- 真实数据窗口定义结构强度标尺、bucket nuisance、控制特征支持域和自然近邻距离基线。
- 合成生成器有意干预某个 primary realized feature，用于构造 capability-focused stress tests。
- 真实 Shard 仍用于直接评估实际数据表现，并与合成能力成绩做关联分析。

动态生成和 near-distance gate 能降低相对已提交 reference 的近复制风险，但不能证明未知预训练语料中不存在相似模式。当前九个 capability 也不是已经证明相互正交的九个潜在坐标；在新的完整 E1 重跑前，正确表述是“能力聚焦的结构压力测试”，不能无保留地宣称“完全 disentangled capabilities”。

## 2. 完整数据链路

```text
冻结真实数据资产
  ├─> 窗口化与特征提取
  │     ├─> canonical reference corpus ─> capability-global intensity targets
  │     ├─> R_param ─> bucket nuisance + inverse conditioning
  │     ├─> R_ref / R_cal ─> control-feature conformal support
  │     └─> R_train / R_holdout ─> DCR / NNDR thresholds
  │
生成请求
  └─> 精确匹配已校准 profile
        └─> 生成候选完整窗口
              └─> construction predictability gate
                    └─> context-only 标准化与 realized features
                          └─> control-feature support gate
                                └─> full/context near-distance gate
                                      ├─ accepted ─> Synthetic Shard
                                      └─ rejected ─> 最多 32 次；仍失败则整次报错

Real Shard / Synthetic Shard
  └─> SampleIndex ─> 模型输入隔离 ─> 推理、指标、报告、榜单
```

平台入口为 `GET /synthetic/capabilities` 和 `POST /synthetic/shards`。一次请求选择多个 capability 时，每个 capability 生成独立 Shard；系统不会把多种目标机制混进同一条样本后再做归因。

## 3. 三类 profile 不可混为一谈

真实 bucket 的身份为：

\[
b=(profile\_id,frequency,context,horizon,target\_dim,covariate\_dim,season\_length).
\]

当前实现使用三类 profile：

| 类型 | 当前规模 | 用途 |
| --- | ---: | --- |
| canonical reference profiles | 18 | 只定义每个 capability 的全局五档 target；其中一部分是 canonical-only，不直接在线生成 |
| generator conditioning profiles | 9 | 为精确任务/窗口拟合 nuisance、结构尺度和到全局 target 的逆映射 |
| capts-paper-v2 online profiles | 8 | 同时具备 conditioning、feature-support 和 near-distance artifact，可由平台正式生成 |

`electricity_hourly_daily_2048ctx_24h` 已有 generator conditioning，但没有冻结的 2048-context near-distance artifact，因此是 `research_only_pending_near_distance_gate`，不属于 capts-paper-v2 在线集合。它在相关 artifact 补齐前会 fail closed。

canonical-only profile 只扩充标尺覆盖面，不会自动成为在线 bucket。反过来，在线 M4、Electricity、Traffic profile 可以使用由更广泛 development corpus 定义的同一个绝对强度标尺。

## 4. 真实窗口、拆分与预处理

### 4.1 真实评测样本

真实 CSV 或 TsFile 经格式、有限数值、唯一递增时间戳和等间隔检查后写入：

```text
DatasetManifest
  └─ Shard
      ├─ SeriesPoint：每个时间点只存一次
      └─ SampleIndex：保存 context / horizon 的行号闭区间
```

若长序列长度为 \(R\)，context 为 \(C\)，horizon 为 \(H\)，stride 为 \(S\)，窗口数为：

\[
N=\left\lfloor\frac{R-C-H}{S}\right\rfloor+1.
\]

Real 与 Synthetic Shard 共用同一存储和下游读取协议。

### 4.2 生成器与 feature gate 的三路拆分

每个在线 bucket 的真实窗口拆成：

```text
R_param：只拟合 generator nuisance 和 conditioning
R_ref：只估计 control-feature 联合支持域
R_cal：只计算 split-conformal threshold
```

多序列或 panel 数据按 source series / panel group 拆分，同一 group 不跨分区；单序列数据按时间阻塞拆分，并在相邻分区间设置至少 `C + H` 的 embargo。`R_cal` 不参与生成器参数拟合，合成 acceptance rate 也不参与阈值选择。

near-distance artifact 使用独立的 real-train / real-holdout 拆分。holdout 到 train 的自然最近邻距离用于确定 p01/p05 阈值；在线 reference 固定为 artifact 中提交的 192 个 real-train 窗口。

### 4.3 标准化与特征

候选序列先生成完整的 \(C+H\) 点，再只使用 context 统计量标准化：

\[
\tilde y_{t,d}=\frac{y_{t,d}-\mu_d(Y_{1:C})}{\sigma_d(Y_{1:C})}.
\]

- 单变量和普通多目标任务逐目标使用 context 均值与标准差。
- 层级任务逐通道中心化但共享一个尺度，保持 `target_0 = sum(target_1:)`。
- 连续协变量使用 context 均值/标准差；0/1 event 保持原值。
- shape feature 的内部计算在需要时使用 median/IQR robust scaling。
- future target 从不参与归一化参数估计。

`realized_features` 是对标准化后实际曲线重新计算的结果；它不同于 `latent_params`。绝对强度标定和在线门控使用 realized features，而不是假设设置了某个 latent 参数就一定观察到相应结构。

## 5. 当前九个 capability

论文主协议包含 6 个单变量 capability 和 3 个 structured capabilities：

| Capability | 任务协议 | 生成机制 | Primary realized feature |
| --- | --- | --- | --- |
| `trend` | 单目标 | 跨 forecast boundary 连续的线性/二次趋势，叠加非周期稳定 AR(2) 背景 | `trend_strength` |
| `multi_seasonal` | 单目标 | 主周期与多个历史可见的附加周期叠加 | `multi_period_score` |
| `time_varying_seasonality` | 单目标 | 振幅和相位按历史可观察的平滑规律调制 | `seasonal_amplitude_modulation` |
| `regime_switching` | 单目标 | 历史中重复出现、预测期继续运行的固定状态时钟 | `regime_clock_history_incremental_r2` |
| `nonlinear_persistence` | 单目标 | 稳定的短滞后、profile-scale 长滞后和非线性多滞后递推；不叠加周期载波 | `nonlinear_conditional_gain` |
| `predictable_intermittency` | 单目标 | 历史可识别的非等间隔 motif 驱动稀疏脉冲 | `spike_rate` |
| `common_factor` | 多目标，当前在线为 3 targets | 共享非周期动态因子与目标局部成分 | `pca_top1_explained` |
| `hierarchical_coherence` | 多目标，当前在线为 parent + 2 children | 父节点始终严格等于子节点之和，强度改变 child heterogeneity | `hierarchy_child_heterogeneity` |
| `covariate_response` | 单目标 + known-future covariates | 历史已有作用样例，预测期提供 weather/event 等已知信号 | `covariate_incremental_r2` |

`common_factor` 和 `hierarchical_coherence` 是多目标预测；`covariate_response` 是单目标加 known-future covariates。论文中可统称 structured capabilities，但模型输入、基线和结果必须按三种协议分别解释。

`nonlinear_persistence` 的内部状态先运行 `max(256, 8P)` 步固定 burn-in，
再截取长度 \(C+H\) 的发布轨迹；随机初始化段不会作为历史前缀暴露给模型。

`lead_lag_coupling`、`coherent_regime_shift`、旧的伪 long-memory 和随机 burst 能力已不在注册表中。仓库的历史 baseline 或图片仍可能包含旧名称，它们不能用于描述 capts-paper-v2 当前能力集合。

## 6. Capability-global absolute intensity

### 6.1 全局 target 的定义

`intensity ∈ {1,2,3,4,5}` 在同一 capability 内表示跨 bucket 共享的绝对 realized-strength target。不同 capability 的主指标量纲不同，因此 `trend=3` 与 `common_factor=3` 不能解释为相同物理强度，也不表示相同模型难度。

对 capability \(c\)，每个冻结 reference profile 先计算自己的五点分位曲线；默认分位为 q20/q35/q50/q70/q90。随后对 profile 等权逐坐标取中位数，并做 endpoint-preserving 的相邻分辨率投影：中间相邻 target 至少相隔原始 target range 的 10%。

`regime_switching` 是预注册例外：I1 固定为 history-clock qualification
边界 0.10，I5 为 qualified-real 的等 profile 权重 q90，I2–I4 在该绝对
统计量尺度上等距。这样不会把一次性 change point 混入标尺，也不会让
合格真实窗口集中在狭窄强区间时五档失去分辨率。

`nonlinear_persistence` 同样使用可解释端点：I1 是有符号
`nonlinear_conditional_gain` 的 adjusted-\(R^2\) 零效应边界，I5 是等 profile
权重的 real q90，I2–I4 等距。这样低档不会挤在有限样本估计器的零点噪声内。

\[
T_{c,k}=\operatorname{Project}_{10\%}\left(
\operatorname{median}_{b\in\mathcal B_c}Q_b(p_k)
\right).
\]

同一 dataset family 在同一 capability 中最多贡献一条曲线，额外 context/horizon 不重复加权。GIFT-Eval 官方 test windows 和冻结 held-out families 不参与 target、conditioning 或 gate 拟合。

### 6.2 每个 bucket 只反解“怎样达到 target”

bucket 不再用自己的 q20--q90 重新定义 intensity。对每个在线 `bucket × capability`，系统固定该 bucket 的 nuisance，并反解 structure scale 与单调映射：

\[
\lambda_{b,c}(k)=\arg\min_{\lambda}
\left|\mathbb E_{seed} f_c(G_c(\theta_{b,c},\lambda))-T_{c,k}\right|.
\]

标定默认使用两个独立 fit seed banks、每个 grid cell 各 64 个样本。方差更高的 `nonlinear_persistence` 与 `covariate_response` 使用四个独立 fit banks、每 bank 至少 128 个样本。响应均值曲线单调化后连续反解。另一组独立样本验证五档 realized mean：普通能力样本数至少 256，高方差能力至少 1024；结果必须单调，且相对 capability target range 的最大归一化误差不超过 0.20，否则该 conditioning cell 不可发布。

artifact 同时记录：

- `canonical_scale_id` 与 content-derived fingerprint；
- 每档 `canonical_target_strength`；
- bucket 的 `profile_lambda` 和 `calibrated_profile_expected_strength`；
- target 在该 bucket 真实窗口中的 `local_real_percentile`；
- reference profile、held-out family、资产哈希和协议代码版本。

`local_real_percentile` 可以接近 0 或 1：这表示同一个绝对 target 对该 bucket 是温和样本或反事实压力测试，不会反过来改变 intensity 含义。新增/删除 reference 或改变 target 曲线必须发布新的 `canonical_scale_id`，不得根据模型成绩事后重拟合。

## 7. 候选生成与可预测性契约

### 7.1 Profile 和 seed 在生成前确定

请求必须在 `frequency/context/horizon/target_dim/capability` 上找到同时具有三类 artifact 的精确匹配 profile。研究调用可以用 `anchor_profile_ids` 固定 profile；普通请求对匹配 profile 做 seed-deterministic 的均衡轮转。系统不会生成后再挑一个最容易通过的 bucket。

主季节周期来自所选 profile；请求中的 `season_length` 仅保留为兼容/记录字段，不覆盖 profile 周期。单变量能力强制 `target_dim=1`，多目标能力至少 3 targets，协变量能力固定单目标并生成声明的 covariate columns。

第 \(i\) 个样本 seed 为：

\[
s_i=\operatorname{BLAKE2s}(seed\Vert capability\Vert i)\bmod(2^{32}-1).
\]

第 \(a\) 次尝试使用 `(s_i + 104729a) mod (2^32-1)`。相同配置和 seed 可复现；相同 seed 下扩展 horizon 不改变已有轨迹前缀。

### 7.2 Construction-level predictability gate

生成公式必须在构造时证明预测所需信息可用：

- 决定 future 条件均值的规律已在 target history 中重复出现；或
- future covariates 显式提供了预测期条件。

只在 forecast horizon 新采样随机 change point、burst 或 hierarchy shock 的配置不允许进入生成。每个样本记录 capability-specific contract、evidence 和 `construction_validated`。该条件是配置级必要条件；失败会立即返回 `synthetic_predictability_contract_failed`，不会通过重采样绕过。

construction gate 证明信息存在，不等于证明任意模型都能利用。模型层可预测性由
固定 seed bank 上的 capability-aware forecast 与 capability-blind matched baseline
对照，以及后续正式模型实验验证。这里的 aware predictor 是 construction oracle：
可读取已冻结的机制参数（如周期族、历史事件时钟、固定效应系数），并在协议允许时
读取 known-future covariates；blind predictor 只从 context 选择通用预测器。两者都
不能读取 future target 的任何实现值，future target 只用于事后计分。该 oracle
contrast 验证的是“构造是否留下可利用的预测增益”，不是正式模型之间的公平比赛。
其中 nonlinear oracle 在通用 blind forecast 与非线性递推 forecast 之间使用冻结的
50% correction shrinkage：
\(\hat y_{\mathrm{aware}}=\hat y_{\mathrm{blind}}+
0.5(\hat y_{\mathrm{nonlinear}}-\hat y_{\mathrm{blind}})\)。
该系数跨 profile 固定，用于抑制递归多步误差累积，不按被评分 future 调整。

该 contrast 不进入逐样本 acceptance。按未来误差拒绝单条样本会选择“恰好容易预测”
的 future 并产生选择偏差，因此在线元数据只记录诊断，发布资格由预注册 seed bank
在 I5 的配对聚合增益下界与胜率决定；I1–I4 仍完整报告，但 I1 可以是显式 null
边界，因此不要求每一档都产生正的 oracle headroom。为避免对 blind loss 很小的简单样本赋予畸高权重，
增益使用 ratio of means，而不是逐样本相对增益的均值：

\[
\widehat G=
\frac{\overline{L_{\mathrm{blind}}-L_{\mathrm{aware}}}}
{\overline{L_{\mathrm{blind}}}},\qquad
\operatorname{LCB}_{0.95}=
\frac{\overline{\Delta L}-1.645\,\operatorname{SE}(\Delta L)}
{\overline{L_{\mathrm{blind}}}}.
\]

## 8. 当前在线硬门控

候选样本必须同时满足：

\[
Accept(x)=PredictabilityGate(x)
\land FeatureSupportGate(x,b)
\land NearDistanceGate(x,\mathcal B_{compatible}).
\]

任何所需 artifact 缺失、schema 不兼容、必要 feature 非有限或任务窗口不精确匹配都按失败处理，不存在 fail-open。

### 8.1 Control-feature 联合支持域

feature-support gate 只使用每个 capability 预注册的非目标 control vector。`R_ref` 上先做 median/IQR robust 标准化，再对裁剪后的 reference covariance 做对角收缩；`R_cal` 的 Mahalanobis scores 用 finite-sample split-conformal 95% quantile 得到阈值：

\[
d_c^b(x)=\sqrt{(z-\mu)^T\Pi(z-\mu)/D},
\qquad
\tau_b=Q^{conformal}_{0.95}(d_c^b(R_{cal})).
\]

当前 control features 为：

| Capability | Online controls |
| --- | --- |
| `trend` | `outlier_rate`, `spike_rate` |
| `multi_seasonal` | `trend_strength`, `outlier_rate`, `spike_rate` |
| `time_varying_seasonality` | `trend_strength`, `outlier_rate`, `spike_rate` |
| `regime_switching` | 无独立 observable control（显式 no-control contract） |
| `nonlinear_persistence` | `trend_strength`, `outlier_rate`, `spike_rate` |
| `predictable_intermittency` | 无独立 observable control（显式 no-control contract） |
| `common_factor` | `trend_strength`, `outlier_rate`, `spike_rate` |
| `hierarchical_coherence` | `hierarchy_residual_mean_abs`, `outlier_rate`, `spike_rate` |
| `covariate_response` | `covariate_residual_acf_abs_mean`, `covariate_residual_outlier_rate`, `covariate_residual_spike_rate` |

primary target feature 不进入该在线硬门控，否则会把有意施加的结构干预压回 bucket-local 分位范围。target 相对真实分布的位置仅作诊断。代码中保留的旧单边 `PILOT_ACCEPTANCE_CAPS` 只供 archived experiments 使用，不属于正式在线 acceptance。

### 8.2 Full-window 与 model-visible context 近距离门控

near-distance gate 在所有任务兼容的真实 bucket 上检查：

- context-standardized full target 的 raw MAE 与 raw L2 DCR；
- 模型可见 context target 的 raw MAE 与 raw L2 DCR；
- robust-z feature L2；
- full/context 的 raw-MAE NNDR。

real holdout 到 real train 的自然距离分布给出 p01/p05 阈值。只要任一兼容 bucket 触发以下风险就拒绝：

```text
strict risk:
  full 或 context 的 raw MAE 与 L2 同时 <= p01

combined risk:
  full raw MAE/L2 <= p05，且 feature L2 或 raw NNDR <= p01
  或 context raw MAE/L2 <= p05，且 context NNDR <= p01
```

该 gate 约束“不要和已提交真实 reference 过近”，不要求合成曲线在 raw 空间长得像某一条真实曲线。当前 raw DCR 直接检查 target；known-future covariates 通过相关性等 realized features 进入 feature distance，尚未作为完整原始输入向量进入 raw DCR。

### 8.3 有限重试与事务结果

feature-support 或 near-distance 失败时最多生成 32 个候选。任一候选通过即返回；32 次均失败则抛出 `synthetic_acceptance_failed`，不会保存最后一个失败样本，也不会创建部分合格 Shard。

MMD/SWD 是批量方法报告，不是每个 Web 请求的在线 gate。当前 E1 使用 fixed-bandwidth RBF-MMD 和 128-projection SWD，将 synthetic-vs-real 与 3-IQR shifted negative control 比较。

## 9. 统一评测与结果解释

真实和合成样本最终都读取为 `sample.v1`。模型输入构造只提供：

\[
\mathcal I=(Y_{history},X_{history},X_{future}^{known},H),
\]

不会提供 `target_future`。服务端用隐藏 future truth 计算 MASE、MSE、MAE；层级任务还可报告 forecast coherence 指标。Shard 通过 CapabilityBlock 组成 Track 和榜单。

结果分析必须遵循：

- intensity 只表示 realized structure strength，不要求模型误差随档位单调。
- 同一 capability 的不同 bucket 可比较同一 canonical target；跨 capability 不比较 target 数值。
- common factor、hierarchy 和 covariate response 必须按各自任务协议报告，不能只给一个含糊的 multivariate 平均分。
- 真实—合成对应性应在冻结 held-out dataset families 上检验，不能把用于 target 拟合的数据同时当作完全独立验证。

## 10. 当前正式证据与已知限制

旧 `capts-paper-v1` E1 结果使用共享周期 carrier、通用 change-point 主指标和旧非线性
统计量，不能作为 `capts-paper-v2` 的有效证据。当前 v2 conditioning 已有 28 个
`profile × capability` cells 全部 supported，独立验证的最大 normalized dose error
为 0.165820。128-seed 的
[shortcut-resistance baseline](../superpowers/baselines/2026-07-18-capts-paper-v2-shortcut-audit.md)
中，9/9 capability-aware contrast 与 7/7 非季节 seasonal-naive shortcut 判据通过。

完整论文实验仍需重新报告：

- I=1/3/5 的完整 contrast 诊断与所有 profile 的 canonical dose-response；
- control-feature MMD/SWD、DCR/NNDR、跨 seed 重复率；
- held-out 真实数据上的 synthetic-to-real 能力缺陷对应性。

当前不能声称：

- 九种 realized features 已完全正交；
- 所有 profile 的绝对 target 都在任意有限样本实验中无误差命中；
- 相对当前 reference 的 DCR 结果证明对未知预训练语料“零污染”；
- synthetic 与 real 的整体分布必须不可区分；
- discriminative score 或 train-on-synthetic/test-on-real predictive score 已作为在线质量门槛。

旧 E1 已观察到 trend/change-point、intermittency/nonlinear 等 feature overlap，以及部分
amplitude-changing capability 对 `noise_ratio` 等 ratio feature 的机械影响。v2 已移除
会强迫共享周期 carrier 的 target-coupled controls，但仍应使用“capability-focused”
而不是“完全 disentangled”表述，除非新的正交性实验能够支持更强主张。

## 11. 变更与扩展规则

### 11.1 新增 canonical reference

1. 冻结源文件哈希、协议代码版本、许可和 test-tail/held-out 规则。
2. 按 capability 做资格审计；例如 regime 必须证明 history-selected recurring clock 在 untouched future 中继续有效。
3. 保证同一 dataset family 在同一 capability 中只贡献一条等权曲线。
4. 任何 target 变化都发布新的 scale ID/fingerprint，并在模型评测前冻结。

### 11.2 新增在线 bucket 或窗口组合

必须分别生成：

1. generator conditioning；
2. exact-task feature-support calibration；
3. exact-task near-distance calibration。

三者缺一即 fail closed。不能把 168-context 的 gate 阈值复用于 2048-context，也不能只因为 canonical-only profile 进入了全局 target 就直接在线生成。

Web 上传真实数据当前不会自动更新任何冻结 artifact。artifact 变更后必须运行对应单元/API 测试、完整在线 acceptance sweep，并使用新的不可变目录重跑方法实验。

## 12. 代码与实验索引

| 内容 | 位置 |
| --- | --- |
| 能力注册、生成器、标准化、在线编排 | `backend/app/services/synthetic_generation_service.py` |
| profile 选择、绝对强度 metadata、conditioning 解析 | `backend/app/services/synthetic_generator_conditioning.py` |
| generator conditioning artifact 构建 | `scripts/build_synthetic_v2_generator_conditioning_artifact.py` |
| control-feature support gate | `backend/app/services/synthetic_feature_gate.py` |
| feature gate artifact 构建与三路拆分 | `scripts/build_synthetic_v2_feature_gate_artifact.py` |
| full/context DCR/NNDR gate | `backend/app/services/synthetic_near_distance_gate.py` |
| near-distance 校准 | `scripts/run_synthetic_v2_near_distance_calibration.py` |
| 真实窗口特征与 canonical-only 数据加载 | `scripts/synthetic_feature_profile.py` |
| capability reference 资格审计 | `scripts/synthetic_capability_qualification.py` |
| E1 方法有效性 runner | `scripts/run_paper_e1_method_validity.py` |
| 能力公式与 predictability contracts | `docs/superpowers/specs/2026-07-01-synthetic-v2-capability-modeling-definitions.md` |
| 真实分布与门控定义 | `docs/superpowers/specs/2026-07-08-synthetic-v2-real-distribution-validation-chain.md` |
| canonical reference 冻结记录 | `docs/superpowers/specs/2026-07-16-synthetic-v2-canonical-reference-v1-freeze.md` |
| E1 正式结果 | `docs/superpowers/baselines/2026-07-16-paper-e1-method-validity.md` |
