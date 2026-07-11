# Synthetic v2 真实分布与生成校验最终链路

日期：2026-07-08

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

论文中引用“真实分布”时，应明确指 `P_real_raw^b`、`P_real_feat^b` 或二者的组合。当前生成验收主要使用 `P_real_feat^b` 的分位数和 `P_real_raw^b` 的近邻距离校准。

主季节周期也是 bucket 条件变量，不再由生成请求直接决定。若 capability 对应的真实 profile bucket 只有一个明确周期，生成器直接使用该周期；例如 hourly daily bucket 使用 24，M5 daily bucket 使用 7，hourly weekly diagnostic bucket 使用 168。若一个 capability 的 profile envelope 混合了多个频率或多个显著周期，生成器先按请求频率筛选匹配 bucket；仍不唯一时，从该 envelope 的显著周期集合中按真实窗口数加权、用 seed 确定性采样一个周期。

当前版本的显著周期集合定义为 profile 抽取阶段固定的 `significant_periods` 元数据，来自该 profile 的频率和季节设定：hourly daily 为 `{24}`，hourly weekly 为 `{168}`，daily weekly 为 `{7}`，daily annual diagnostic 为 `{365}`。后续可以把它升级为自动检测：在每个真实 bucket 内对 robust-scaled 真实窗口计算候选周期 `p` 的 phase-mean seasonal strength / periodogram energy / ACF peak，保留 median score 超过阈值且与已选周期不存在近似 harmonic duplicate 关系的 top-K 周期。

## Anchor Profiles

当前代码使用多 profile anchor。真实数据基底由多个公开数据集 bucket 组成：

| profile_id | 用途 |
| --- | --- |
| `m4_hourly_daily_96ctx` | 小 context 小时级单变量 sanity profile |
| `m4_hourly_daily_168ctx` | 主 profile，当前在线生成器的主要控制边界 |
| `m4_hourly_weekly` | 小时级长周期/周季节补充 |
| `electricity_hourly_daily_168ctx` | 电力负荷 hourly 单变量控制 profile |
| `electricity_hourly_panel_168ctx` | 电力负荷 3-target panel profile，用于 low-rank common factor / lead-lag 校准 |
| `traffic_hourly_daily_168ctx` | 交通占用率 hourly 单变量控制 profile |
| `traffic_hourly_panel_168ctx` | 交通占用率 3-target panel profile，用于跨序列相关和 lead-lag 校准 |
| `m5_daily_covariate_365ctx_28h` | 零售日频 known-future covariate profile，覆盖 calendar/event/SNAP/price 信号 |
| `m5_daily_hierarchy_365ctx_28h` | 零售 store-category additive hierarchy profile，覆盖 parent=sum(children) 结构 |
| `gefcom2014_load_hourly_covariate_168ctx_24h` | 小时级负荷-温度 covariate profile，覆盖强 known-future weather signal |
| `us_births_weekly` | 日频单变量外部 sanity profile |
| `us_births_annual_diagnostic` | 年周期诊断 profile，不作为短窗口硬边界 |

在线生成器在 `generation_config.anchor_profiles` 中记录上述 profile。当前 hard acceptance 使用四类真实 profile envelope：单变量能力使用 M4/Electricity/Traffic hourly 单变量 profile 的 envelope，多目标能力使用 Electricity/Traffic panel profile 的 envelope，协变量能力使用 M5/GEFCom2014 known-future covariate envelope，层级能力使用 M5 additive hierarchy envelope。US Births profile 作为跨频率 sanity reference。后续如果继续引入 Weather/ETT、Tourism、M5 validation/evaluation 变体等数据集，应新增 bucket 并独立统计。

## Intensity 定义

`intensity in {1,2,3,4,5}` 是生成器的结构强度控制量。映射函数为：

```text
lambda(intensity) = (intensity - 1) / 4
```

每个能力维度把 `lambda` 映射到不同结构参数，例如趋势斜率、次级周期振幅、切点数量、非线性强度、burst rate、factor rank、covariate effect size。

重要约束：

- `intensity` 应让目标结构特征在聚合均值上增强。
- `intensity` 不要求所有模型误差单调增加。
- 如果强度更高但深度模型误差更低，可以解释为结构更规则、更容易学习。
- API 仍兼容旧字段 `difficulty`；新请求、文档和 UI 统一使用 `intensity`。

## 核心特征维度

### 单变量核心特征

论文正文建议保留 5 个单变量结构组：

| 结构组 | 主指标 | 辅助/护栏 | 用途 |
| --- | --- | --- | --- |
| Trend shape | `trend_strength` | `slope_abs`, `curvature_abs` | 趋势方向、斜率和曲率外推 |
| Seasonal structure | `multi_period_score` | `seasonal_strength`, `seasonal_drift_score` | 多周期叠加和时变季节性 |
| Structural breaks | `change_point_shift_energy` | `level_shift_strength`, `volatility_shift_strength` | level/volatility regime change |
| Nonlinear persistence | `nonlinear_lag1_gain` | `acf_abs_mean` | 非线性自反馈和持久性 |
| Intermittency/volatility | `burst_rate` | `spike_rate`, `outlier_rate`, `noise_ratio` | 突发、重尾和异方差 |

这里的辅助指标不一定进入主表，但应进入 appendix 或生成日志，用来解释主指标异常和控制非目标特征。

### 多目标/协变量核心特征

论文正文建议保留 3 个多/协变量结构组：

| 结构组 | 主指标 | 辅助/护栏 | 用途 |
| --- | --- | --- | --- |
| Low-rank common factors | `pca_top1_explained` | `effective_factor_rank`, `avg_abs_target_corr` | 多目标共享因子结构 |
| Known-future covariate response | `future_abs_covariate_target_corr` | `avg_abs_covariate_target_corr`, `event_lift_abs` | 是否真正存在 future covariate signal |
| Hierarchical coherence | `hierarchy_residual_mean_abs` | forecast-side `coherence_mae` | 父子加总一致性 |

`lead_lag_peak_abs` 保留为 secondary diagnostic 或 ablation 特征。它有价值，但和 common factor 容易混淆，正文中不应把它放在和三大多/协变量组同等的主定义位置。

## 生成后校验方式

生成候选样本后执行两个在线硬门控和两个批量报告检查。在线硬门控决定样本是否进入最终 shard：

```text
Accept(x_syn | capability c, bucket group B)
  = FeatureGate(x_syn, c, B) AND NearDistanceGate(x_syn, B)
```

`FeatureGate` 控制合成曲线保持在真实 profile 的结构支持域内，避免过度远离真实数据分布。`NearDistanceGate` 控制合成曲线与真实 reference 窗口保持足够距离，避免近复制和预训练污染风险。

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

### 2. 特征阈值在线门控

对非目标特征使用主 anchor 的分位数范围作为护栏：

```text
q05_real^b(j) <= phi_j(psi(x_syn)) <= q95_real^b(j)
```

当前服务已经对全部 synthetic capability 启用 hard acceptance caps。`trend` / `multi_seasonal` / 其他单变量结构能力使用 hourly 单变量真实 profile envelope，多目标能力使用 hourly panel profile envelope，`covariate_response` 使用 M5+GEFCom2014 covariate profile envelope，`hierarchical_coherence` 使用 M5 hierarchy profile envelope。`hierarchy_residual_mean_abs` 的真实 p95 为 0，线上 hard cap 使用 `1e-6` 浮点容差；`event_lift_abs` 使用 M5 p95 的较宽倍数，原因是 synthetic 维度刻意测试 event response，稀疏真实事件的 1.5 倍上限会过早截断目标结构。

该门控结果写入：

```text
sample_metadata.latent_params.acceptance.validation.feature_gate
```

### 3. 近邻距离在线门控

把真实 bucket 拆成 `R_train` 和 `R_holdout`，合成数据只能和 `R_train` 比较；`R_holdout` 用来给自然近邻距离定基线。

原始窗口距离：

```text
D_raw(x, R_train) = min_r mean_abs(psi(x) - psi(r))
```

特征距离：

```text
D_feat(x, R_train) = min_r || z(phi(psi(x))) - z(phi(psi(r))) ||_2
```

最近邻距离比：

```text
NNDR(x) = D_1(x, R_train) / max(D_2(x, R_train), eps)
```

其中 `D_1`、`D_2` 是第一和第二近邻距离。`z(.)` 使用 `R_train` 的 median 和 robust scale。

当前实现把校准后的 reference artifact 提交在：

```text
backend/app/data/synthetic_v2_near_distance_artifact.json
```

生成器按 capability 对应的 profile group 逐 bucket 评估。单变量能力检查 M4/Electricity/Traffic 三个 hourly 单变量 bucket；多目标能力检查 Electricity/Traffic panel bucket；协变量能力检查窗口设定匹配的 M5 或 GEFCom bucket；层级能力检查 M5 hierarchy bucket。artifact 在 `context_length`、`horizon`、`target_dim` 与校准 bucket 完全一致时执行强制门控；缺少匹配校准 bucket 的请求直接返回 `synthetic_near_distance_not_calibrated`，不生成 shard。

当前已校准的在线组合为：

| profile group | capabilities | context | horizon | target_dim | frequency |
| --- | --- | ---: | ---: | ---: | --- |
| hourly univariate envelope | `trend`, `multi_seasonal`, `time_varying_seasonality`, `regime_switching`, `long_memory_nonlinear`, `intermittent_heteroskedastic` | 168 | 24 | 1 | hourly |
| hourly panel envelope | `common_factor`, `lead_lag_coupling`, `coherent_regime_shift` | 168 | 24 | 3 | hourly |
| known-future covariate envelope | `covariate_response` | 168 | 24 | 1 | hourly / GEFCom2014 |
| known-future covariate envelope | `covariate_response` | 365 | 28 | 1 | daily / M5 |
| M5 hierarchy envelope | `hierarchical_coherence` | 365 | 28 | 3 | daily |

风险规则：

- Strict risk：raw z-L2 和 raw z-MAE DCR 都低于 real-holdout p01。
- Combined risk：raw z-L2 和 raw z-MAE DCR 都低于 real-holdout p05，且 feature DCR 低于 p01 或 NNDR 低于 p01。
- 样本接受规则：`strict_risk=false` 且 `combined_risk=false`。
- 批量报告阈值：strict risk = 0%，combined risk <= 1%。

这个检验服务于“合成数据不容易导致预训练污染”的理论论据：合成窗口相对真实训练窗口的最近邻距离，不应比真实 holdout 自然产生的最近邻距离更近。

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
| anchor profiles 记录 | `MOCK_ANCHOR` |
| 在线 realized features | `_realized_features()` |
| 生成后 validation summary | `_validation_summary()` |
| 在线特征阈值门控 | `_accept_synthetic_features()` |
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

本文将真实数据基底按 `profile_id/frequency/context/horizon/target_dim/covariate_dim` 分桶，并在每个 bucket 中定义原始窗口经验分布 `P_real_raw` 与显式特征经验分布 `P_real_feat`。合成序列通过 `intensity` 控制目标结构强度。生成后，我们重新抽取特征，用特征阈值门控控制曲线不过分远离真实 profile，用 DCR/NNDR 近距离门控控制曲线不过分靠近真实窗口；MMD/SWD 作为控制特征分布批量报告。该链路与 discriminative/predictive score 互补，分别提供可解释的非污染论据和生成质量论据。
