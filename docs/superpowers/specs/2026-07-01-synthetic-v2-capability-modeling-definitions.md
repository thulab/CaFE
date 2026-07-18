# CapTS-Bench 论文能力维度定义（paper-v2）

日期：2026-07-18

## 1. 定位与版本

本文档定义 `capts-paper-v2` shortcut-resistant 生成器进入论文主实验的能力集合。论文暂定标题为：

> CapTS-Bench: A Real-Anchored, Capability-Focused Live Benchmark for Time Series Foundation Models

主实验固定为 6 个单变量能力和 3 个结构化能力。这里的“结构化”包括多目标结构和已知未来协变量协议，不把二者误写成同一种 multivariate forecasting task。

默认 hourly 实验使用 `context=168, horizon=24, period=24`；层级实验使用 `context=365, horizon=28, period=7`。所有样本先按显式数据生成过程生成，再用 context 统计量标准化。层级样本使用共同尺度标准化，因而不会破坏父子加总关系。

## 2. 最终能力集合

| 论文分组 | capability id | 论文名称 | intensity 的唯一主含义 |
| --- | --- | --- | --- |
| 单变量 | `trend` | Trend extrapolation | 趋势形状的整体尺度 |
| 单变量 | `multi_seasonal` | Multi-seasonal composition | 附加周期的能量 |
| 单变量 | `time_varying_seasonality` | Evolving seasonality | 振幅/相位调制强度 |
| 单变量 | `regime_switching` | Predictable regime switching | 状态水平差异 |
| 单变量 | `nonlinear_persistence` | Nonlinear multi-lag persistence | profile-scale 线性长滞后与非线性滞后的联合依赖强度 |
| 单变量 | `predictable_intermittency` | Predictable intermittency | 非等间隔、历史可识别的稀疏脉冲显著性 |
| 结构化 | `common_factor` | Shared latent factor | 共享因子相对通道局部成分的强度 |
| 结构化 | `hierarchical_coherence` | Hierarchical coherence | 子节点异质性；加总关系始终严格成立 |
| 结构化 | `covariate_response` | Known-future covariate response | 已知未来协变量的效应强度 |

`lead_lag_coupling` 和 `coherent_regime_shift` 已从注册表移除，不作为兼容能力保留。前者的旧实现与 common factor 混杂；后者在 forecast horizon 内注入无先兆冲击，不能解释为可预测能力。

## 3. 共同建模约束

### 3.1 可预测性不是低噪声，而是信息在预测时可用

每个生成器都必须满足 construction-level predictability contract：

1. 决定预测期条件均值的结构，必须已在 target history 中重复出现，或通过 future covariates 明确提供。
2. 禁止只在 forecast horizon 中采样新的 latent cut point、随机 burst 或层级 shock。
3. 允许小幅不可约观测噪声，但 intensity 不得通过增加噪声来伪装成结构强度。
4. 生成后元数据必须记录 contract、证据和 `construction_validated=true`；不满足时直接拒绝生成。

construction gate 只能证明生成过程提供了可用信息，不能代替模型层面的可预测性实验。正式实验仍需与 naive、seasonal naive 和 capability-specific oracle 比较。

paper-v2 额外禁止把同一个强固定周期载波叠加到所有能力。`multi_seasonal` 与 `time_varying_seasonality` 可以把周期作为目标机制；其余能力使用 profile-conditioned、稳定但无固定周期的 AR(2) 背景。对该背景，取两个实根：非季节单变量 nuisance 使用 `r_s ∈ [0.82,0.94]`，结构化动态因子使用 `r_s ∈ [0.94,0.985]`，并统一取 `r_f ∈ [0.15,0.32]`，令：

```text
z_t = (r_s+r_f) z_{t-1} - r_s r_f z_{t-2} + eps_t
```

该过程只用 context 统计量归一化。这样保留持续性、噪声族和尾部等真实 nuisance，同时不会让一个固定 seasonal-naive 同时破解趋势、脉冲、公共因子、层级与协变量任务。

### 3.2 intensity 是结构干预，不是预设难度

`intensity ∈ {1,2,3,4,5}` 是固定
`dataset × task view × capability × context × horizon` 内部的相对结构强度。
五档目标只由该 profile 的 generator-parameter split 定义：

\[
T_{b,c,k}=Q_{R^{param}_b}\left(f_c(x),p_k\right),\qquad
(p_1,\ldots,p_5)=(0.10,0.30,0.50,0.70,0.90).
\]

每个样本先选定 profile `b`，再使用同一 profile 拟合的单调逆映射
`λ_{b,c}(intensity)`，使独立 seed bank 上合成样本主目标特征的均值逼近
`T_{b,c,intensity}`。profile 同时决定 nuisance、“相对强度是多少”和“怎样生成到该
强度”。不同 dataset 的五档不做绝对强度比较，也不共享 target、nuisance 或逆映射。

五档目标必须有限、严格递增，最小相邻间距不低于
`max(1e-6, 0.02 * (T5-T1))`。不满足时，该
`dataset × task view × capability` 如实记录为 `unsupported`，不人为投影出五档，也
不借用其他 dataset 的 target。变量结构、窗口数或校准条件不满足时同理。

噪声尺度、尾部形态、非周期背景持续性和局部通道成分等 nuisance 可以随 `b,c`
改变，但在同一个 `b,c,seed` 的五档 intensity 中必须固定。生成器使用更密的连续
`λ`/结构尺度网格，并记录 `target_values` 与
`calibrated_realized_strengths`。论文不预设模型误差随 intensity 单调，只要求预注册
的 realized target feature 按预期方向响应。形式上，文中各节的 `α(λ)` 均应理解为
`α_{b,c}(λ_{b,c}(intensity))`。

### 3.3 时间参数不依赖总窗口长度

趋势坐标以距 forecast origin 的季节周期数表示；所有周期、调制周期、驻留时长和滞后均以 `period` 为单位。生成公式不能用 `linspace(..., total_length)` 定义结构，否则仅改变 horizon 就会改变历史段本身。

## 4. 单变量能力

### 4.1 `trend`

生成式：

```text
x_t = (t - forecast_origin) / P
g_t = s * x_t + c * x_t^2
y_t = alpha(λ) * g_t + nonperiodic_AR2_t + eps_t
```

- intensity 只提高 `alpha(λ)`；AR(2) 背景和噪声固定。
- slope 与 curvature 的符号按样本采样，但 context 和 horizon 共用同一组系数。
- contract：context 至少 32 点，且同一多项式与稳定背景规律跨 forecast boundary 连续。
- realized target features：`trend_strength` 与 `slope_abs` 预期增大；`curvature_abs` 仅作 shape diagnostic。生成器固定 curvature/slope 比例，context 标准化后不要求曲率系数独立单调。

### 4.2 `multi_seasonal`

生成式：

```text
y_t = A1 sin(2πt/P + φ1)
    + alpha(λ) sin(2πt/(2P) + φ2)
    + 0.35 alpha(λ) sin(2πt/(P/2) + φ3)
    + eps_t
```

- 三个周期在所有强度都存在，避免用“是否出现第三个周期”的离散跳变定义强度。
- intensity 只控制附加周期能量，主周期和噪声固定。
- contract：最长周期 `2P` 在 context 中至少完成两次。
- realized target feature：`multi_period_score`，预期增大。

### 4.3 `time_varying_seasonality`

生成式采用历史中可重复观察的平滑调制，而不是仅根据总窗口归一化的一次性漂移：

```text
m_t = sin(2πt/(4P) + φm)
A_t = 1 + depth_A(λ) m_t
ψ_t = 2π depth_ψ(λ) m_t
y_t = A_t sin(2πt/P + φ + ψ_t) + residue_t + eps_t
```

- intensity 同时放大同一个 modulation factor 对振幅和相位的作用。
- contract：context 至少覆盖一个完整调制周期，调制规律跨边界连续。
- realized target features：逐周期谐波拟合得到的 `seasonal_amplitude_modulation` 和 `seasonal_phase_variation`，预期均增大。

### 4.4 `regime_switching`

生成式：

```text
z_t ∈ {-1,+1}, z_t every D steps alternates deterministically
y_t = alpha(λ) z_t + nonperiodic_AR2_t + eps_t
```

- `D` 在一个样本内固定；预测区间的下一切换点通过同一时钟确定。
- intensity 只提高两个状态的水平差异，不改变切换次数、驻留时长或噪声。
- contract：context 中至少有两次历史切换，horizon 中至少有一次切换，全部相邻切点间隔相同，状态顺序交替。
- primary realized feature：`regime_clock_history_incremental_r2`。它只在 context 上比较“趋势 + 普通季节谐波 + clock-period 谐波”基线与额外加入离散 regime state 后的拟合；通用 `change_point_shift_energy` 和 `level_shift_strength` 只保留为辅助诊断。

这个维度测“从重复状态时钟预测下一次切换”，不再测无先兆 structural break robustness。后者如果需要，应作为单独 stress track，而不是 capability forecast track。

### 4.5 `nonlinear_persistence`

生成式：

```text
r_t = 0.10 r_{t-1}
    + 0.05 d(λ) r_{t-P}
    + 0.75 d(λ) [sin²(1.1 r_{t-P/2}) - 0.25]
    + eps_t
y_t = [1+2d(λ)] r_t + weak_nonperiodic_AR2_t
```

- intensity 共同提高 profile-scale 线性长滞后和非线性中程滞后的依赖强度，二者比例固定；`P` 在这里定义递推滞后尺度，不叠加固定周期载波。
- 递推先运行固定 `max(256, 8P)` 步 burn-in，再截取发布轨迹；初始化的前 `P` 个随机状态不得直接出现在样本中。
- 生成器检查保守稳定性界 `0.10 + 0.05d + 1.1×0.75d < 1`。
- contract：context 至少覆盖两个最大滞后，所有递推系数跨边界不变。
- primary realized feature：`nonlinear_conditional_gain`。它是完整线性滞后设计 \(D_L=(1,y_{t-1},y_{t-P},y_{t-P/2})\) 与额外加入 \(\sin^2(1.1y_{t-P/2})\) 后的**有符号 adjusted-\(R^2\) 差**。自由度修正和保留负值共同消除无结构时由嵌套回归与逐样本截断造成的正偏地板；`nonlinear_multi_lag_gain` 仅保留为辅助诊断。
- 离线 predictive-headroom audit 使用固定 50% shrinkage 的递归非线性 oracle：\(\hat y_{\rm aware}=\hat y_{\rm blind}+0.5(\hat y_{\rm nonlinear}-\hat y_{\rm blind})\)。它在所有 profile 共用同一系数，避免递归误差累积，且不能按被评分 future 调整。

该维度明确不声称 ARFIMA 或 fractional differencing 意义上的 long memory。

### 4.6 `predictable_intermittency`

生成式：

```text
pulse_t = Σ_k exp(-(t-c_k)^2 / (2w^2))
c_{k+1}-c_k cycles through [round(.75P), P, round(1.25P)]
y_t = alpha(λ) pulse_t + nonperiodic_AR2_t + eps_t
```

- pulse centers 由一个重复的三段间隔 motif 产生；horizon 中的 pulse 与历史 pulse 同源，但 lag-P seasonal naive 不再是 oracle。
- intensity 只提高 pulse prominence，事件频率、宽度和噪声固定。
- contract：context 暴露完整间隔 motif，horizon 中至少一个 pulse。
- realized target features：`burst_rate`、`spike_rate`、`outlier_rate`，预期增大。

该维度不再把不可预测 Bernoulli burst 与异方差噪声混成一个“能力”。异方差更适合未来的 probabilistic forecasting track。

## 5. 结构化能力

### 5.1 `common_factor`

生成式：

```text
f_t = shared_nonperiodic_AR2_t
u_{t,j} = channel_specific_nonperiodic_AR2_t
y_{t,j} = alpha(λ) b_j f_t + u_{t,j} + eps_{t,j}
```

- factor rank、loading 和噪声固定；intensity 只提高 shared factor strength。
- 各通道保留固定幅度、独立创新和 fast root 的局部动态成分，避免所有通道成为简单缩放复制。
- contract：至少三个 target channels、context 至少 32 点；共享/局部动态公式和 loading 跨 forecast boundary 不变。
- realized target features：`pca_top1_explained` 与 `avg_abs_target_corr` 增大，`effective_factor_rank` 减小。
- 正式实验需增加 channel-independent 对照或 channel permutation，证明收益来自跨通道结构。

### 5.2 `hierarchical_coherence`

生成式：

```text
child_{t,j} = shared_t / J + alpha(λ) local_{t,j} + eps_{t,j}
Σ_j local_{t,j} = 0,  Σ_j eps_{t,j}=0
parent_t = Σ_j child_{t,j}
```

- intensity 控制 bottom-level heterogeneity，不控制“是否一致”；输入数据在所有强度都严格 coherent。
- 局部成分在通道方向上中心化，使 parent 的尺度不随 heterogeneity 一起膨胀，减少 intensity 混杂。
- contract：至少一个 parent 和两个 children；没有 horizon-only shock；子节点规律跨边界连续。
- invariant：`hierarchy_residual_mean_abs ≈ 0`。
- realized target feature：`hierarchy_child_heterogeneity`，预期增大。
- 模型评价必须同时报告 forecast error 与 prediction coherence error。

### 5.3 `covariate_response`

生成式：

```text
y_t = base_t
    + beta_weather(λ) weather_t
    + beta_event(λ) event_t
    + eps_t
```

- weather 是带创新的非周期稳定 AR(2) 外生过程；未来实现值不能只由 target history 精确外推，但会作为 known-future covariate 提供。
- 历史段固定安排至少三个 event effect 样例，预测段保证至少一个 event。
- intensity 只提高统一的 covariate effect scale，事件数、位置规则和噪声固定。
- contract：历史中至少两个效应样例；horizon 中存在已提供的 future event；系数跨边界不变。
- realized target features：相对季节基线的 `covariate_incremental_r2`、`future_abs_covariate_target_corr` 和 `event_lift_abs`，预期增大。
- 正式实验必须做 intact、drop-future-covariate、shuffle-future-covariate 和 event-flip 配对消融。

## 6. 进入主实验前的验收

每个 supported `dataset × capability × intensity × lookback` cell 至少生成预注册
数量的无需模型推理样本，依次检查：

1. **Construction gate**：全部样本满足本文件的 predictability contract。
2. **Dose response**：预注册 realized target feature 的 bootstrap 置信区间和强度方向符合定义。
3. **Selectivity**：构建 capability × feature effect matrix；目标 feature 的效应应明显大于非目标 feature 的漂移。
4. **Real support**：非目标 control features 落在对应真实 bucket 的联合支持域内。
5. **Novelty**：通过独立校准的近距离门限和 copy/jitter/shift/warp 攻击测试。
6. **Shortcut resistance**：对非季节能力，固定 `seasonal-naive(P)` 不得相对 last-value 形成统一捷径。
7. **Predictive headroom**：在预注册 seed bank 的 I5 上，capability-aware forecast 相对 capability-blind matched baseline 的配对 loss difference 之 ratio-of-means 单侧下界和胜率均达标；I1–I4 作为诊断报告，不要求显式 null 边界也产生正 headroom。不使用会过度加权低 blind-loss 样本的逐样本 relative-gain 均值。该比较只做聚合资格审计，不按单样本未来误差筛选生成结果。
8. **Support accounting**：缺 task view、变量结构不支持、窗口不足、档间距不足或校准失败均写入完整九能力 support matrix；不补值，也不把 unsupported 当作最差结果。

旧 `capts-paper-v1`、跨 dataset pooled profile、canonical intensity 与 paper-v4
prototype runtime 结果不作为当前实验的有效结果；generator conditioning、
feature-support、near-distance 与 shortcut audit 均须按 dataset-local v4 schema
重新生成。
