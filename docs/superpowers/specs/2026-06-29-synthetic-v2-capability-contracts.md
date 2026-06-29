# Synthetic v2 能力契约草案

日期：2026-06-29

## 背景

本契约基于 `scripts/synthetic_feature_profile.py` 和真实数据 profile 烟测结果制定：

- 烟测记录：`docs/superpowers/baselines/2026-06-29-synthetic-v2-profile-smoke.md`
- 主小时级 anchor：`m4_hourly_daily_168ctx`
- 日频 sanity anchor：`us_births_weekly`
- 默认目标特征上限：真实 profile `p95 * 1.5`；`trend_strength` / `seasonal_strength` 等天然 `[0, 1]` 特征额外截断到 `1.0`

第一版只覆盖 `trend` 和 `multi_seasonal`，因为这两个能力能用显式统计特征解释，也最适合作为后续 acceptance check 的模板。

## 共同验收原则

每个 synthetic sample 生成后都应重新抽取特征，并按下列规则验收：

1. 目标特征随 `difficulty=1..5` 单调增强，允许小幅随机波动，但聚合均值必须单调。
2. 目标特征不得超过真实分布上限倍数；例如主 anchor 的 `trend_strength` cap 为 `1.0`，`slope_abs` cap 为 `0.5314`，`curvature_abs` cap 为 `1.0135`。
3. 非目标特征不得无意义爆炸。第一版先以 `noise_ratio <= anchor p95` 和 outlier/spike 不显著高于真实 p95 作为硬约束。
4. 生成样本必须让 naive / seasonal naive 的 MASE 或 MAE 有有限值；若基线误差为 0，应退回 MAE 或跳过该样本。
5. acceptance 失败时优先 resample；连续失败后降级调弱目标增强，不直接返回失控样本。

## Anchor Quantiles

主 anchor `m4_hourly_daily_168ctx` 的关键分位数：

| Feature | p05 | p25 | p50 | p75 | p95 | cap |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| `trend_strength` | 0.0000 | 0.0398 | 0.1659 | 0.4801 | 0.7714 | 1.0000 |
| `seasonal_strength` | 0.5768 | 0.7746 | 0.9129 | 0.9943 | 0.9961 | 1.0000 |
| `acf_abs_mean` | 0.2515 | 0.4407 | 0.5201 | 0.5446 | 0.5574 | 0.8361 |
| `slope_abs` | 0.0136 | 0.0628 | 0.1264 | 0.1731 | 0.3543 | 0.5314 |
| `curvature_abs` | 0.0042 | 0.0212 | 0.0711 | 0.2464 | 0.6756 | 1.0135 |
| `noise_ratio` | 0.0038 | 0.0056 | 0.0821 | 0.2141 | 0.3871 | 0.5807 |
| `spike_rate` | 0.0000 | 0.0000 | 0.0000 | 0.0157 | 0.1257 | 0.1886 |

日频 sanity anchor `us_births_weekly` 的关键分位数：

| Feature | p50 | p75 | p95 | cap |
| --- | ---: | ---: | ---: | ---: |
| `trend_strength` | 0.2459 | 0.2660 | 0.2938 | 0.4406 |
| `seasonal_strength` | 0.7040 | 0.7693 | 0.8309 | 1.0000 |
| `slope_abs` | 0.2102 | 0.2992 | 0.3606 | 0.5410 |
| `curvature_abs` | 0.3674 | 0.4536 | 0.5142 | 0.7713 |
| `noise_ratio` | 0.2693 | 0.2984 | 0.3567 | 0.5351 |

## Capability: `trend`

### 意图

考察模型能否在上下文窗口中识别趋势方向、趋势强度和一定程度的曲率，并把趋势外推到预测窗口，而不是只复制季节形状或最后一个值。

### 目标特征

- `trend_strength`：趋势解释方差占比，采用 Hyndman trend strength 形式。
- `slope_abs`：robust scale 后二次多项式的一阶系数绝对值。
- `curvature_abs`：robust scale 后二次多项式的二阶系数绝对值。

### 控制特征

- `seasonal_strength`：允许存在弱季节残差，但不能成为主导目标。
- `noise_ratio`：不超过 anchor p95 附近，避免把难度伪装成噪声。
- `outlier_rate` / `spike_rate`：不超过真实 p95 的宽松上限。

### 难度映射

第一版用主 anchor 的分位数做目标区间：

| Difficulty | `trend_strength` target | `slope_abs` target | `curvature_abs` target |
| ---: | --- | --- | --- |
| 1 | p25 附近：0.04 | p25 附近：0.06 | p25 附近：0.02 |
| 2 | p50 附近：0.17 | p50 附近：0.13 | p50 附近：0.07 |
| 3 | p75 附近：0.48 | p75 附近：0.17 | p75 附近：0.25 |
| 4 | p85-p90 插值 | p85-p90 插值 | p85-p90 插值 |
| 5 | p95 附近，但不超过 cap | p95 附近，但不超过 cap | p95 附近，但不超过 cap |

工程上不要求每个单样本精确命中所有三个特征，但样本集合均值应随难度单调增强，且难度 5 的目标特征不超过 cap。

### 预期基线响应

- naive 对强趋势和曲率外推应明显变差。
- seasonal naive 在强趋势样本上也应变差，因为它只复用历史季节位置。
- 如果 seasonal naive 与 naive 都没有随 difficulty 变差，说明趋势增强没有进入 forecast horizon 或被标准化抵消。

## Capability: `multi_seasonal`

### 意图

考察模型能否识别叠加周期及其相位，而不是只拟合单一周期或短期自相关。

### 目标特征

- `seasonal_strength`：基础周期性约束。主 anchor 上该特征 p50 已很高，因此高难度不要求它继续升高，只要求保持在真实 p05 以上，避免退化成纯噪声。
- `acf_abs_mean`：辅助目标，用于反映周期性自相关结构。
- `seasonal amplitude ratio`：工程 latent 参数中的多周期振幅占比；第一版先记录在 `latent_params`，后续再加入 profiler。
- `single-period seasonal naive degradation`：单周期 seasonal naive 的 MAE 应随难度上升，这是第一版 multi-seasonal 的主要行为验收。

### 控制特征

- `trend_strength`：保持在 anchor p50-p75 左右，避免 multi-seasonal 退化成 trend。
- `noise_ratio`：不超过 anchor p95 附近。
- `spike_rate`：不超过真实 p95 的宽松上限。

### 难度映射

由于 M4 Hourly 的 `seasonal_strength` 已高度集中，第一版难度不只看该值，而是用“周期数量 + 次级周期振幅 + 噪声控制”组合：

| Difficulty | Periods | Secondary amplitude | `seasonal_strength` guardrail | seasonal naive MAE |
| ---: | --- | --- | --- | --- |
| 1 | 1 个主周期 | 0 | 高于 p50：约 0.91 | 最低 |
| 2 | 1 个主周期 + 弱次周期 | 低 | 高于 p25 | 低 |
| 3 | 2 个周期 | 中 | 高于 p25 | 中 |
| 4 | 2-3 个周期 | 中高 | 高于 p05 | 中高 |
| 5 | 3 个周期 | 高，但 noise 不超过 cap | 高于 p05 | 最高 |

### 预期基线响应

- naive 在所有季节性样本上应弱于 seasonal naive。
- 单周期 seasonal naive 在难度 1-2 应表现较好，但在难度 4-5 的多周期叠加样本上 MAE 应上升。
- 如果 seasonal naive 不随 difficulty 变差，说明次级周期振幅或相位扰动不够。

## 论文和方法依据

- Trend / seasonal strength 采用 Hyndman 在 forecasting 文献和 `tsfeatures` 中常用的方差解释形式。
- M4 Hourly 作为 anchor 的理由是它有足够多小时级真实序列，适合统计 trend、seasonality、acf 和噪声分布。
- Monash Archive 的 TSF 数据适合作为公开、可复现的真实分布来源；第一版只使用显式数学特征，不引入深度生成模型。

## Pilot 输出要求

实现 trend / multi-seasonal v2 pilot 后，至少输出：

1. 每个 difficulty 的生成后 feature summary。
2. 目标特征均值是否单调。
3. 是否超过真实分布 cap。
4. naive / seasonal naive 的误差曲线。
5. 与旧生成器的对比结论：哪些特征更稳定、哪些仍不符合预期。
