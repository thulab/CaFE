# Synthetic v2 真实分布与生成校验最终链路

日期：2026-07-08

paper-v1 profile-conditioned generator / feature gate 修订：2026-07-15

## 目标

本文档固定 synthetic v2 论文阶段采用的三个定义：

1. 真实分布如何从真实数据基底中抽取。
2. 合成数据保留哪些核心特征维度。
3. 序列生成后如何检验结构强度、真实性和预训练污染风险。

这里的 `intensity` 表示目标结构强度。naive / seasonal naive 的单调响应可作为诊断证据，深度模型误差单调性不进入生成验收条件。

## 记号

一个 forecast sample window 记为：

```text
x = (Y_{1:C+H}, Z_{1:C+H})
```

其中 `Y` 是目标序列，`Z` 是可选的 known-future covariates，`C` 是 context length，`H` 是 horizon。单变量任务 `target_dim=1`，多目标任务 `target_dim>1`，协变量任务 `covariate_dim>0`。

预处理函数记为 `psi(x)`：

- 目标列按 context 段做 z-score 标准化。
- 层级一致性样本使用共享尺度标准化，避免破坏 `target_0=sum(target_1:)`。
- 连续协变量按 context 段标准化，二值 event 协变量保持 0/1。
- 丢弃含非有限值的窗口。

特征抽取函数记为 `phi(psi(x))`，返回本文后面列出的显式统计特征。

## 真实分布定义

真实分布定义为每个 anchor bucket 内的经验分布。bucket 定义为：

```text
b = (profile_id, domain, frequency, context_length, horizon, target_dim, covariate_dim, season_length)
```

对 bucket `b` 中抽到的真实窗口集合 `R_b={x_i}_{i=1..N}`，定义两类真实分布：

```text
P_real_raw^b  = (1/N) * sum_i delta_{psi(x_i)}
P_real_feat^b = (1/N) * sum_i delta_{phi(psi(x_i))}
```

论文中引用“真实分布”时，应明确指 `P_real_raw^b`、`P_real_feat^b` 或二者的组合。当前生成验收使用 `P_real_feat^b` 校准 control features 的联合支持域，并使用 `P_real_raw^b` 校准近邻距离。

主季节周期也是 bucket 条件变量，不再由生成请求直接决定。生成前先选定一个任务、窗口和频率完全匹配的 `anchor_profile_id=b`，再直接使用该 bucket 的周期；例如 hourly daily bucket 使用 24，M5 daily bucket 使用 7。

当前版本的显著周期集合定义为 profile 抽取阶段固定的 `significant_periods` 元数据，来自该 profile 的频率和季节设定：hourly daily 为 `{24}`，hourly weekly 为 `{168}`，daily weekly 为 `{7}`，daily annual diagnostic 为 `{365}`。后续可以把它升级为自动检测：在每个真实 bucket 内对 robust-scaled 真实窗口计算候选周期 `p` 的 phase-mean seasonal strength / periodogram energy / ACF peak，保留 median score 超过阈值且与已选周期不存在近似 harmonic duplicate 关系的 top-K 周期。

## Anchor Profiles

当前代码使用多 profile anchor。真实数据基底由多个公开数据集 bucket 组成：

| profile_id | 用途 |
| --- | --- |
| `m4_hourly_daily_168ctx` | M4 小时级单变量 profile |
| `electricity_hourly_daily_168ctx` | 电力负荷 hourly 单变量控制 profile |
| `electricity_hourly_daily_2048ctx_24h` | Electricity 长 context 研究 profile；当前由专用实验脚本执行近距离门控 |
| `electricity_hourly_panel_168ctx` | 电力负荷 3-target panel profile，用于 low-rank common factor / lead-lag 校准 |
| `traffic_hourly_daily_168ctx` | 交通占用率 hourly 单变量控制 profile |
| `traffic_hourly_panel_168ctx` | 交通占用率 3-target panel profile，用于跨序列相关和 lead-lag 校准 |
| `m5_daily_covariate_365ctx_28h` | 零售日频 known-future covariate profile，覆盖 calendar/event/SNAP/price 信号 |
| `m5_daily_hierarchy_365ctx_28h` | 零售 store-category additive hierarchy profile，覆盖 parent=sum(children) 结构 |
| `gefcom2014_load_hourly_covariate_168ctx_24h` | 小时级负荷-温度 covariate profile，覆盖强 known-future weather signal |

默认批次对所有精确匹配 bucket 做 seed-deterministic 的均衡分层；也可通过 API 的 `anchor_profile_ids[capability_id]` 固定一个 bucket。选定发生在生成前，而不是生成后挑一个最容易通过的 bucket。`generation_config.anchor_profiles` 只记录本 shard 实际候选，样本级 `anchor_profile_id` 记录本样本的预选 bucket。后续引入 Weather/ETT、Tourism 等数据集时，应新增独立 bucket 与独立生成器校准，而不是放大既有 bucket 的阈值。

## Intensity 定义

`intensity in {1,2,3,4,5}` 是跨 bucket 可比较的有序结构强度级别。基础坐标为：

```text
u(intensity) = (intensity - 1) / 4
```

实际结构参数使用 bucket-conditional 映射 `lambda_{b,c}(intensity)`。生成器只在真实参数拟合 split 上估计 profile nuisance 和结构尺度，并把五档连续目标特征对齐到该 bucket 的经验 q10/q30/q50/q70/q90；对稀疏脉冲这类离散/饱和指标，保留五档 effect-size grid，再用 bootstrap dose response 验收。切点时钟、噪声尾部、季节残差、factor rank 等 nuisance 可以随 `b,c` 改变，但在同一个 `b,c,seed` 的 intensity 扫描中固定。

因此，Electricity 与 Traffic 可以有不同噪声尾部、季节残差和局部通道成分，M5 与 GEFCom 可以有不同 covariate effect 尺度；这不是为了放松门控，而是定义不同的真实条件 DGP。完整校准写入 `synthetic_v2_generator_conditioning_artifact.json`。

重要约束：

- `intensity` 应让目标结构特征在聚合均值上增强。
- `intensity` 不要求所有模型误差单调增加。
- 如果强度更高但深度模型误差更低，可以解释为结构更规则、更容易学习。
- API 仍兼容旧字段 `difficulty`；新请求、文档和 UI 统一使用 `intensity`。

## 核心特征维度

### 单变量核心特征

论文正文保留 6 个单变量结构组：

| 结构组 | 主指标 | 辅助/护栏 | 用途 |
| --- | --- | --- | --- |
| Trend shape | `trend_strength` | `slope_abs`, `curvature_abs` | 趋势方向、斜率和曲率外推 |
| Multi-seasonal composition | `multi_period_score` | `seasonal_strength` | 多周期叠加 |
| Evolving seasonality | `seasonal_amplitude_modulation` | `seasonal_phase_variation` | 历史可观察的振幅/相位调制 |
| Predictable regime switching | `change_point_shift_energy` | `level_shift_strength` | 从重复切换时钟预测下一状态 |
| Nonlinear persistence | `nonlinear_multi_lag_gain` | stability bound | 稳定的季节滞后与非线性中程滞后 |
| Predictable intermittency | `spike_rate` | `burst_rate`, `outlier_rate` | 从重复事件时钟预测稀疏脉冲 |

这里的辅助指标不一定进入主表，但应进入 appendix 或生成日志，用来解释主指标异常和控制非目标特征。

### 多目标/协变量核心特征

论文正文建议保留 3 个多/协变量结构组：

| 结构组 | 主指标 | 辅助/护栏 | 用途 |
| --- | --- | --- | --- |
| Low-rank common factors | `pca_top1_explained` | `effective_factor_rank`, `avg_abs_target_corr` | 多目标共享因子结构 |
| Known-future covariate response | `covariate_incremental_r2` | `future_abs_covariate_target_corr`, `event_lift_abs` | 是否真正存在 future covariate signal |
| Hierarchical coherence | `hierarchy_child_heterogeneity` | input `hierarchy_residual_mean_abs`, forecast-side `coherence_mae` | 异质子节点下的父子加总一致性 |

`lead_lag_coupling` 已从 paper-v1 注册表移除；若未来重新加入，必须先移除 common-factor confound 并完成 leader-permutation ablation。

## 生成后校验方式

生成器先选定 `b` 并读取 `theta_{b,c}`，再执行三个在线硬门控和两个批量报告检查：

```text
theta ~ P_hat(theta | b,c),  x_syn = G_c(theta, lambda_{b,c}(intensity), seed)

Accept(x_syn | capability c, selected bucket b, compatible buckets B)
  = PredictabilityGate(x_syn, c)
    AND FeatureGate(x_syn, c, b)
    AND NearDistanceGate(x_syn, B)
```

`PredictabilityGate` 检查决定预测期条件均值的结构已在历史中重复出现，或已通过 known-future covariates 提供。`FeatureGate` 只对预选 `b` 检查控制特征支持，禁止生成后择优换桶。`NearDistanceGate` 仍对所有兼容 `B` 检查，避免样本虽远离 `b` 却近似复制另一个真实 bucket。

Predictability gate 的 contract、构造证据和结果写入 `sample_metadata.latent_params.predictability` 与 `acceptance.validation.predictability_gate`。这是一项 construction-level 必要条件，不替代 oracle/naive 的模型层可预测性验证。

### 1. 结构强度校验

对每个 capability 和 intensity，重新计算目标特征：

```text
f_target = phi_target(psi(x_syn))
```

批量验收使用聚合统计；单样本严格单调不进入验收条件：

```text
mean_b,k[f_target] should increase with intensity k
```

当前在线生成器把结果写入：

```text
sample_metadata.latent_params.acceptance.validation.target_features
```

生成时使用的主周期记录在：

```text
generation_config.season_length
generation_config.season_length_source
generation_config.season_length_candidates
sample_metadata.season_length
sample_metadata.requested_season_length
```

其中 `requested_season_length` 只是旧 API 兼容字段，不再驱动生成。

### 2. Control-feature 联合支持域在线门控

旧实现把多个 profile 的 p95 取最大值后再乘启发式倍率，并且大部分只有上界。这既不是文档声称的 p05--p95 双边约束，也会接受“每个边际都正常、联合组合却不真实”的样本。paper-v1 改为按精确 bucket 独立校准的联合门控。

对 capability `c` 只使用预注册的非目标 control vector `g_c(x)`。在真实 reference split 上用 median/IQR 标准化，并估计收缩协方差：

```text
z_c(x) = (g_c(x) - median(R_ref)) / IQR(R_ref)
d_c^b(x) = sqrt((z_c(x)-mu_b)^T Precision_b (z_c(x)-mu_b) / dim(g_c))
```

真实窗口做三路拆分：`R_param` 只用于拟合生成器 conditioning，`R_ref` 用于估计 feature support，`R_cal` 只用于 conformal 阈值。同一序列或 panel group 不跨 split；GEFCom 这类单序列 bucket 使用时间阻塞切分，并在相邻分区设置至少 `C+H` 的 embargo。阈值只由 real calibration score 决定：

```text
tau_b = split_conformal_quantile_0.95({d_c^b(x): x in R_cal})
FeatureGate(x,c,b) = [d_c^b(x) <= tau_b]
```

只有 `(frequency, context, horizon, target_dim, capability)` 完全匹配的预选 bucket 可参与；`FeatureGate` 不再接受 profile group，也不会在多个通过者中选择最低 `d/tau`。不存在所选 bucket 校准时 fail closed。阈值计算不读取 synthetic acceptance rate，也没有 1.5/2.5 multiplier。

目标特征不进入该在线支持域硬门控。原因是它们正是 intervention 对象，把目标特征强制压入逐样本真实分位带会改变 DGP，并与第 1 节的批量 dose-response 契约冲突。系统仍记录每个 target feature 在绑定真实 bucket 中的大致经验分位位置，用于 E1 批量验收和论文报告。

`noise_ratio`、`seasonal_strength` 等相对强度统计会随目标 effect/factor/regime strength 机械变化，因此不能冒充 nuisance control。`regime_switching`、`common_factor` 和 `hierarchical_coherence` 使用非目标异常率或层级 invariant；`covariate_response` 先回归掉季节基线和 covariates，再用 residual ACF/outlier/spike 特征检查剩余过程。正式 E1 仍需用完整 capability × feature effect matrix 检查 selectivity。

校准结果提交在：

```text
backend/app/data/synthetic_v2_feature_gate_artifact.json
```

当前 artifact 覆盖主协议的 8 个 bucket，并额外覆盖 Electricity `context=2048, horizon=24` 的单变量研究 bucket，供长 context 扫描脚本使用。新增任何 context/horizon/task 组合前必须先生成独立真实 bucket 校准，不能复用 168-window 的阈值。

32 次拒绝采样仅是工程故障保护，不是校准手段。正式验收要求每个 `profile × capability × intensity` 的未拒绝 first-pass rate 至少 95%，并同时报告拒绝前、拒绝后的控制特征分布；若不达标，应重拟合生成器 conditioning，不能放宽门限或依赖反复抽样。

该门控结果写入：

```text
sample_metadata.latent_params.acceptance.validation.feature_gate
```

### 3. 近邻距离在线门控

把真实 bucket 拆成 `R_train` 和 `R_holdout`，合成数据只能和 `R_train` 比较；`R_holdout` 用来给自然近邻距离定基线。同一 source series 或 panel group 不跨分区；只有一个 group 时使用连续时间块，并删除边界处所有与 holdout 重叠的窗口（间隔按 `C+H` 窗长检查）。正式在线 artifact 固定使用 split 0，阈值和落盘的 192 条 reference 来自完全相同的 `R_train` 子集，不再用全体真实行另抽 reference。

原始完整目标窗口与模型可见的目标 context 分别计算距离：

```text
D_raw(x, R_train) = min_r mean_abs(psi(x) - psi(r))
D_ctx(x, R_train) = min_r mean_abs(psi(Y_context) - psi(Y_context^r))
```

特征距离：

```text
D_feat(x, R_train) = min_r || z(phi(psi(x))) - z(phi(psi(r))) ||_2
```

最近邻距离比：

```text
NNDR(x) = D_1(x, R_train) / max(D_2(x, R_train), eps)
```

其中 `D_1`、`D_2` 是第一和第二近邻距离。`z(.)` 使用 `R_train` 的 median 和 robust scale。p01/p05 从严格正的 real-holdout 距离尾部估计；跨 group 的 exact duplicate 本身仍计为风险样本，但不能把阈值塌缩为 0。这样既能检出 affine/jitter copy，也避免“context 完全复制、只替换 future”绕过 full-window DCR。

当前实现把校准后的 reference artifact 提交在：

```text
backend/app/data/synthetic_v2_near_distance_artifact.json
```

生成器按 capability 对应的 profile group 逐 bucket 评估。单变量能力检查 M4/Electricity/Traffic 三个 hourly 单变量 bucket；多目标能力检查 Electricity/Traffic panel bucket；协变量能力检查窗口设定匹配的 M5 或 GEFCom bucket；层级能力检查 M5 hierarchy bucket。artifact 在 `context_length`、`horizon`、`target_dim` 与校准 bucket 完全一致时执行强制门控；缺少匹配校准 bucket 的请求直接返回 `synthetic_near_distance_not_calibrated`，不生成 shard。artifact 缺少 context reference、阈值字段或所需 realized feature 时按 schema mismatch fail closed，不再把缺失特征静默填成中心值。

当前已校准的在线组合为：

| profile group | capabilities | context | horizon | target_dim | frequency |
| --- | --- | ---: | ---: | ---: | --- |
| hourly univariate envelope | `trend`, `multi_seasonal`, `time_varying_seasonality`, `regime_switching`, `nonlinear_persistence`, `predictable_intermittency` | 168 | 24 | 1 | hourly |
| hourly panel envelope | `common_factor` | 168 | 24 | 3 | hourly |
| known-future covariate envelope | `covariate_response` | 168 | 24 | 1 | hourly / GEFCom2014 |
| known-future covariate envelope | `covariate_response` | 365 | 28 | 1 | daily / M5 |
| M5 hierarchy envelope | `hierarchical_coherence` | 365 | 28 | 3 | daily |

风险规则：

- Strict risk：full-window 或 context-only 任一视图的 raw z-L2 与 raw z-MAE DCR 同时低于对应 real-holdout p01。
- Combined risk：full-window 仍采用“raw z-L2/z-MAE 低于 p05，且 feature DCR 或 NNDR 低于 p01”；context-only 采用“raw z-L2/z-MAE 低于 p05，且 context NNDR 低于 p01”。两者任一触发即拒绝。
- 样本接受规则：`strict_risk=false` 且 `combined_risk=false`。
- 批量报告阈值：strict risk = 0%，combined risk <= 1%。

这个检验服务于“合成数据不容易导致预训练污染”的理论论据：合成窗口相对真实训练窗口的最近邻距离，不应比真实 holdout 自然产生的最近邻距离更近。

证据边界必须明示：当前 raw DCR 只对提交在 artifact 中的 192 条 `R_train` 目标轨迹直接检查。known-future covariates 通过相关性等显式特征进入 feature DCR，尚未作为原始模型输入向量参与 DCR。`R_holdout` 只用于校准自然距离，未进入在线 reference；未知预训练语料更不在覆盖范围内。因此论文应使用“相对已提交 reference 的低近复制风险”，而不是“证明无预训练污染”。如要升级为完整模型输入与全真实基底覆盖，需要增加 `target_context + history_cov + future_cov` 输入视图，并用互补的 cross-fit reference folds 覆盖 calibration 行。

该门控结果写入：

```text
sample_metadata.latent_params.acceptance.validation.near_distance_gate
```

### 4. 分布距离批量报告

对控制特征集合计算分布距离：

```text
MMD(P_real_feat^b, P_syn_control^b)
SWD(P_real_feat^b, P_syn_control^b)
```

并用 real-vs-real split 作为参考：

```text
MMD(P_real_a, P_real_b), SWD(P_real_a, P_real_b)
```

MMD/SWD 作为批量报告证据，用来检查控制特征分布是否失控。合成数据会有意增强目标结构，整体分布贴近真实分布不作为硬门控目标。

## 与 discriminative / predictive score 的关系

Discriminative score 和 predictive score 属于实验性论据，用来说明生成质量和任务可用性；本文定义的 DCR/NNDR、MMD/SWD 和显式结构特征属于理论性和可解释性论据。论文中建议分开报告：

- 显式特征和真实分布距离：说明合成曲线位于真实控制范围内。
- DCR/NNDR：说明低近邻污染风险。
- Discriminative / predictive score：说明生成数据保留可学习预测结构。

## 当前代码映射

| 定义 | 代码位置 |
| --- | --- |
| `intensity` API 字段 | `backend/app/api/routes/synthetic.py` |
| synthetic shard 生成 | `backend/app/services/synthetic_generation_service.py` |
| profile 选择与 conditioning 读取 | `backend/app/services/synthetic_generator_conditioning.py` |
| generator conditioning 校准脚本 | `scripts/build_synthetic_v2_generator_conditioning_artifact.py` |
| generator conditioning artifact | `backend/app/data/synthetic_v2_generator_conditioning_artifact.json` |
| anchor protocol 记录 | `MOCK_ANCHOR` 与样本级 `anchor_profile_id` |
| 在线 realized features | `_realized_features()` |
| 生成后 validation summary | `_validation_summary()` |
| 在线联合 feature-support 门控 | `backend/app/services/synthetic_feature_gate.py` |
| feature-support 校准脚本 | `scripts/build_synthetic_v2_feature_gate_artifact.py` |
| feature-support artifact | `backend/app/data/synthetic_v2_feature_gate_artifact.json` |
| 在线近距离门控 | `backend/app/services/synthetic_near_distance_gate.py` |
| 近距离 reference artifact | `backend/app/data/synthetic_v2_near_distance_artifact.json` |
| 平台导入脚本 | `scripts/import_synthetic_v2_experiment_shards.py` |
| 近距离阈值校准脚本 | `scripts/run_synthetic_v2_near_distance_calibration.py` |
| 前端生成 UI | `frontend/src/components/wizard/SyntheticConfigStep.vue` |

生成出的 shard 和 sample metadata 同时写入：

```text
intensity: <1..5>
difficulty: <same value, compatibility only>
intensity_definition: target temporal structure strength; not a required monotonic model-error difficulty
```

## 论文中建议采用的简短表述

本文将真实数据基底按 `profile_id/frequency/context/horizon/target_dim/covariate_dim` 分桶，并把真实窗口按 group 或带 `C+H` 非重叠区的时间块拆为 generator-parameter、gate-reference 与 gate-calibration 分区。每个样本在生成前选定真实 bucket，按该 bucket 拟合的 nuisance 参数和 intensity 映射生成；目标特征以配对批量 dose-response 验收。生成后，我们只在预选 bucket 内用 real-only split-conformal 的联合 control-feature 支持域限制曲线不过分远离真实数据，同时对所有兼容 bucket 在 full-window 与 model-visible context 两个视图上用 DCR/NNDR 限制曲线不过分靠近真实 reference；MMD/SWD 作为批量分布报告。该链路分别提供 capability validity、real support 与低近复制风险证据。
