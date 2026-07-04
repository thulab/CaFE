# Synthetic v2 能力维度建模定义

日期：2026-07-01

## 定位

这份文档整理当前 synthetic v2 生成器中可用于论文实验的能力维度。目标不是把 difficulty 解释成“模型一定更难”，而是把每个维度定义成一个可控的数据结构扰动，再用真实模型实验观察模型响应。

当前实验表见：

- `docs/superpowers/baselines/2026-07-01-synthetic-v2-all-capabilities-experiment-table.md`
- `runtime/research/synthetic-v2-*/summary.json`

默认窗口：`context=168`，`horizon=24`，`season_length=24`。单目标维度使用 `target_dim=1`；多目标维度使用 `target_dim=3`；协变量维度当前使用 `target_dim=1, covariate_dim=2`。

## 共同原则

1. 合成序列先用显式公式生成，再按 context 归一化，保证不同样本尺度可比。
2. `trend` / `multi_seasonal` 已接入 M4 Hourly profile 的 acceptance caps；其余维度当前是 v2 pilot，已有真实模型响应实验，但还需要补真实分布 cap。
3. difficulty 控制的是目标结构强度，例如切换次数、非线性强度、burst rate、因子结构、协变量效应；它不等同于观测到的模型误差强度。
4. 对 horizon 内无先兆的 shift，不应在论文里简单称为“可预测能力”，更准确是结构突变鲁棒性或快速适应能力。

## 维度定义

### `trend`

建模定义：单目标序列由二次趋势、弱季节残差和小噪声组成。

公式概念：

```text
y_t = a * x_t + b * (x_t^2 - c) + s * sin(2*pi*t/P) + slow_t + eps_t
```

难度控制：提高 `slope_abs`、`curvature_abs` 和 `trend_strength`，同时降低噪声，避免把难度伪装成随机扰动。

测什么：模型是否能外推趋势方向和曲率，而不是复制 last value 或单周期季节位点。

实验观察：真实模型响应稳定，`Chronos-2` 平均 MAE 最低。高难度下 Timer 系列接近或略差于 seasonal naive，说明强趋势/曲率外推仍有区分度。

论文表述风险：当前序列仍保留明显季节残差，应称为 “trend with seasonal residue”。

### `multi_seasonal`

建模定义：单目标序列由一个主周期和 1-2 个次级周期叠加。

公式概念：

```text
y_t = A1*sin(2*pi*t/P1 + phi1)
    + A2(lambda)*sin(2*pi*t/P2 + phi2)
    + A3(lambda)*sin(2*pi*t/P3 + phi3)
    + eps_t
```

难度控制：提高次级/三级周期振幅，让单周期 seasonal naive 越来越不够用。

测什么：模型是否能识别多周期叠加和相位，而不是只复用一个季节周期。

实验观察：seasonal naive MAE 随 difficulty 明显增大，而深度模型 MAE 反而保持很低；这说明高难度数据更规则、更可学习，并不矛盾。`toto2.0` 平均 MAE 最低。

论文表述风险：`seasonal_strength` 本身不适合作为难度唯一解释，应报告 “single-period seasonal naive degradation” 作为行为证据。

### `time_varying_seasonality`

建模定义：单目标序列的季节振幅和相位随时间漂移。

公式概念：

```text
A_t = A0 + delta(lambda) * t
phi_t = phi0 + drift(lambda) * t^1.35
y_t = A_t * sin(2*pi*t/P + phi_t) + slow_t + eps_t
```

难度控制：提高振幅变化幅度 `amplitude_delta_mean` 和相位漂移 `phase_drift_cycles`。

测什么：模型是否能处理非平稳季节性。它不同于 `multi_seasonal`：这里周期数量不变，难点是同一周期的形态在变。

实验观察：6 个单目标模型全部成功，`toto2.0` 平均 MAE 最低，`Chronos-2` 次之。该维度适合保留为论文里的“非平稳季节性”测试。

论文表述风险：当前只做平滑漂移，没有突发 phase jump；若要测节假日错位或制度性日历漂移，需要单独加 event/covariate 版本。

### `regime_switching`

建模定义：单目标序列由多个 level/volatility regime 分段组成，并确保至少可能有一个 cut point 落在 forecast horizon。

公式概念：

```text
y_t = level_k + seasonal_t + slow_t + eps_t(sigma_k),  t in segment k
```

难度控制：提高切换数量、level 方差和 segment volatility。

测什么：模型面对结构突变时是否过度平滑，能否在新水平/新波动率下快速适应。

实验观察：Timer 修复后本轮全部模型成功；`Timer-3.0` 平均 MAE 最低。误差随 difficulty 不单调，因为切点位置、level 方向和 horizon 内是否可观察共同影响可预测性。

论文表述风险：如果 shift 没有先兆，它更像 robustness stress test；后续若要测“识别将发生的 regime change”，应加入 leading covariate 或 pre-shift warning pattern。

### `long_memory_nonlinear`

建模定义：单目标序列由高持久性的自回归项、非线性 carry-over、季节项和噪声组成。

公式概念：

```text
y_t = phi(lambda) * y_{t-1}
    + gamma(lambda) * sin(y_{t-1})
    + seasonal_t + slow_t + eps_t
```

难度控制：提高 `phi` 和 nonlinear strength。

测什么：模型是否能保留较长上下文状态，并处理非线性自反馈。

实验观察：`Timer-3.5` 平均 MAE 最低。difficulty 与 MAE 不单调，尤其 d5 反而更容易，说明当前增强更像“非线性持久性强度”，不等同于严格预测难度。

论文表述风险：当前不是 ARFIMA / fractional differencing 意义上的严格 long memory。论文里建议命名为 “nonlinear persistence”，除非后续补 Hurst 指数或 ACF hyperbolic decay 验收。

### `intermittent_heteroskedastic`

建模定义：单目标序列由稀疏 Bernoulli burst、Gamma burst size 和时间变动 volatility 组成。

公式概念：

```text
event_t ~ Bernoulli(p(lambda))
burst_t ~ Gamma(shape(lambda), scale(lambda)) * event_t
sigma_t = base + amp(lambda) * seasonal_volatility_t
y_t = trend_t + seasonal_t + burst_t + eps_t(sigma_t)
```

难度控制：提高 burst 频率/幅度和 volatility 变化。

测什么：模型在 intermittent demand、稀疏突发、重尾误差和异方差下的稳健性。

实验观察：`Chronos-2` 和 `toto2.0` 几乎并列最好。`noise_ratio`、`outlier_rate`、`spike_rate` 随 difficulty 增强明显，但 `target_max_abs` 偏高。

论文表述风险：标准化后它不再是严格非负需求序列。后续最好用 M5 / 零售类 intermittent demand 的真实分布重新定 cap，并增加 burst recall 或 event-window error 指标。

### `common_factor`

建模定义：多目标序列由低秩 latent factors 和 channel loadings 生成。

公式概念：

```text
F_t = [seasonal_t, slow_t, trend_t, ar_t]
Y_t = F_t * B^T + E_t
```

难度控制：提高 factor rank 和噪声水平。

测什么：模型是否能利用跨通道共享结构，而不是把每个 channel 当成独立单变量序列。

实验观察：当前服务里只有 `toto2.0` 支持 `target_dim=3`，它明显优于 naive baselines。该维度当前是 multi-target sanity check，尚不能做横向模型排名。

论文表述风险：随机 loading 会让 channel correlation 有波动，应补 PCA explained variance / factor strength 验收。

### `lead_lag_coupling`

建模定义：在 common factor 基底上，给后续 channel 加入来自 leader channel 的滞后影响。

公式概念：

```text
y_{t,j} = base_{t,j} + w_j(lambda) * y_{t-lag_j, leader(j)}
```

难度控制：主要提高 coupling strength；`max_lag` 由 `season_length` 上限约束，默认 `season_length=24` 时会被 `season_length // 3 = 8` 卡住，因此不随 difficulty 增长。更长周期配置下，`8 + floor(10 * lambda)` 才可能让可选滞后范围随 difficulty 扩大。

测什么：模型是否能识别跨通道滞后依赖，利用先行 channel 提前预测滞后 channel。

实验观察：`toto2.0` 明显优于 naive baselines。MAE 随 difficulty 有增长，说明 lag/coupling 强化带来一定挑战。

论文表述风险：common factor 可能混淆 lag signal。后续应增加 lagged cross-correlation peak 验收，并做 leader permutation ablation。

### `coherent_regime_shift`

建模定义：多目标序列共享一个系统级 shift time，各 channel 同时发生 level shift。

公式概念：

```text
Y_t = common_t + E_t + 1[t >= tau] * delta
```

难度控制：提高 shift vector norm 和噪声。

测什么：模型面对系统级多通道冲击时，是否能统一调整多个目标，而不是逐通道孤立处理。

实验观察：`toto2.0` 优于 naive baselines，但整体误差高于 common factor / lead-lag。难度增强后 MAE 上升明显。

论文表述风险：这里的 “coherent” 指多目标共同 regime shift，不是层级加总一致性。后者由 `hierarchical_coherence` 单独覆盖。

### `hierarchical_coherence`

建模定义：多目标序列包含父子加总结构，`target_0 = sum(target_1:)`。生成后使用层级保真标准化，避免逐列 z-score 破坏加总关系。

公式概念：

```text
child_{j,t} = seasonal_{j,t} + slow_t + trend_t + shock_{j,t} + eps_{j,t}
parent_t = sum_j child_{j,t}
Y_t = [parent_t, child_1_t, child_2_t, ...]
```

难度控制：提高 child shock count 和 shock strength。

测什么：模型是否既能准确预测各层级目标，又能输出满足 parent-child 加总关系的预测。

实验观察：输入样本的 `hierarchy_residual_mean_abs` 接近 0；`toto2.0` 的预测 `coherence_mae` 随 difficulty 上升，从 d1 的 0.0892 到 d5 的 0.2141。该维度有明确创新价值，因为它把 forecast accuracy 和 structural validity 分开看。

论文表述风险：当前平台主指标仍是 MAE/MASE，`coherence_mae` 只在实验脚本中生成；若要进入正式评测平台，应把 coherence metric 注册进 MetricDefinition。

### `covariate_response`

建模定义：单目标序列由目标历史、连续 weather covariate 和二值 event covariate 共同驱动，并把 covariates 的未来段作为 known-future 输入给模型。

公式概念：

```text
y_t = seasonal_t + slow_t + trend_t
    + beta_weather(lambda) * weather_t
    + beta_event(lambda) * event_t
    + eps_t
```

难度控制：提高 weather/event effect size 和 event count。

测什么：模型是否真正使用 known-future covariates，而不是只从 target history 外推。

实验观察：当前只有 `Chronos-2` 支持该协议，且明显优于 naive / seasonal naive。生成后 `avg_abs_covariate_target_corr` 随 difficulty 从 0.4097 增至 0.5709。

论文表述风险：仅看完整窗口 correlation 会高估能力。后续应增加 no-covariate、future-covariate permutation、event-only ablation。

## 暂不纳入的维度

| 维度 | 暂缓原因 | 后续条件 |
| --- | --- | --- |
| Irregular / missing sampling | 当前 timer service 和数据校验要求规则时间轴和有限 float。 | 模型输入协议支持 missing mask 或 irregular timestamp 后再加。 |
| Probabilistic calibration | 当前评测链路是点预测，缺少 quantile / interval forecast。 | 模型服务返回分位数或区间后，引入 pinball loss / coverage。 |
| Full covariate intervention | 当前 `covariate_response` 已覆盖相关性响应，但还没有反事实对照。 | 在实验脚本中加入 future covariate permutation / counterfactual event flip。 |

## 参考依据

- Trend / seasonal strength：Hyndman 的 feature-based time series work、`tsfeatures` 和 Forecasting: Principles and Practice 对 trend/seasonal strength 的定义。参考：[Large-scale unusual time series detection](https://robjhyndman.com/papers/icdm2015.pdf)、[`tsfeatures` reference](https://pkg.robjhyndman.com/tsfeatures/reference/stl_features.html)、[FPP time series features](https://otexts.com/fpppy/04-features.html)。
- 多季节性：STL / MSTL 分解思路，适合把多周期和时变季节性分开建模。参考：[MSTL paper](https://arxiv.org/abs/2107.13462)。
- Anchor 数据：M4 Competition、Monash Time Series Forecasting Archive、M5 Competition 可作为真实分布来源。参考：[M4 IJF 2020](https://ideas.repec.org/a/eee/intfor/v36y2020i1p54-74.html)、[Monash Archive paper](https://arxiv.org/abs/2105.06643)、[Monash repository](https://forecastingdata.org/)、[M5 IJF 2022](https://ideas.repec.org/a/eee/intfor/v38y2022i4p1346-1364.html)。
- Regime shift：Hamilton Markov switching 和 structural break 文献。参考：[Hamilton 1989](https://www.jstor.org/stable/1912559)、[Hamilton PDF](https://www.ssc.wisc.edu/~bhansen/718/Hamilton1989.pdf)。
- Intermittent / heteroskedastic：Croston intermittent demand、Engle ARCH。参考：[Croston 1972](https://link.springer.com/article/10.1057/jors.1972.50)、[Hyndman Croston note](https://robjhyndman.com/papers/croston.pdf)、[Engle 1982 ARCH](https://www.jstor.org/stable/1912773)。
- Multivariate dependencies：dynamic factor model、VAR / Granger causality。参考：[Stock & Watson dynamic factors](https://stock.scholars.harvard.edu/publications/macroeconomic-forecasting-using-diffusion-indexes)、[Granger causality 1969](https://ideas.repec.org/a/ecm/emetrp/v37y1969i3p424-38.html)。
- Known-future covariates：Temporal Fusion Transformer 等模型把 observed inputs 与 known future inputs 明确分开。参考：[TFT paper](https://arxiv.org/abs/1912.09363)。
