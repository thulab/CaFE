# Synthetic Data v2 Research Plan

## 背景

当前合成数据生成器已经能按能力维度生成 trend、seasonality、regime shift、common factor、covariate response 等测试组，但生成方式主要是手写公式叠加。下一阶段的目标不是简单增加公式复杂度，而是让合成数据同时满足两件事：

1. **不过度偏离真实数据分布**：合成序列的非目标特征应该落在真实数据集的合理范围内。
2. **能力增强有依据且可验证**：每个能力维度要明确增强了哪些可测特征，并能用论文、基线实验或消融实验说明其合理性。

## 核心判断

### 1. 特征分布是否都能通过数学方式提取？

大多数第一阶段需要的时间序列特征都可以用显式数学/统计方法提取，不需要深度模型：

| 类别 | 可提取特征 | 建议方法 |
| --- | --- | --- |
| 趋势 | trend strength、线性斜率、二次曲率、trend/residual ratio | STL/MSTL 分解、窗口内回归 |
| 季节性 | seasonal strength、主周期、谱峰强度、周期稳定性 | STL/MSTL、periodogram/FFT、ACF |
| 自相关/长记忆 | ACF/PACF、lag decay、Hurst-like scaling | ACF/PACF、DFA/Hurst 估计 |
| 非线性 | nonlinear forecastability、局部复杂度 | tsfeatures/catch22/hctsa 子集 |
| 异方差 | rolling variance、残差方差漂移、ARCH-like strength | rolling statistics、残差分解 |
| 间歇性 | zero ratio、burst rate、spike/outlier rate、平均非零间隔 | 阈值统计、inter-arrival statistics |
| regime shift | change point count、level shift size、volatility shift size | change-point detection、分段统计 |
| 多变量 | 平均相关、主成分解释率、common factor strength | correlation、PCA/factor analysis |
| lead-lag | lagged cross-correlation peak、lead-lag direction | cross-correlation、Granger-like diagnostics |
| 协变量响应 | covariate-target correlation、lagged effect、future-known event lift | 回归/互相关/事件窗口差异 |

这些特征不是“完全无争议”的真值，而是可解释的估计量。需要配套记录：

- 提取窗口：整条序列、训练段、还是 forecast sample window。
- 频率与周期假设：小时、天、周；season length 如何确定。
- 缺失值和异常值处理。
- 置信度或失败标记：例如无法稳定分解、序列过短、全常数。

第一版应优先使用显式特征，因为它们能直接支撑“能力维度”的定义、难度分层和生成后验收。

### 2. 什么时候需要引入深度模型？

深度模型不应该作为第一阶段的默认方案。它适合在以下情况引入：

1. **显式特征无法覆盖真实数据形态**
   - 例如复杂多变量依赖、非平稳高阶交互、跨变量条件关系无法用相关/PCA/lead-lag 解释。

2. **需要条件生成而不是规则增强**
   - 例如给定一个真实行业/频率/上下文分布，生成整体形态高度相似、但目标能力增强的样本。

3. **需要学习真实数据的联合分布**
   - 显式特征通常是边际或低阶统计量，深度生成模型能学习更复杂的联合结构。

4. **显式生成器通过不了分布验收**
   - 如果规则生成的 synthetic 在非目标特征上长期偏离真实分布，即使目标能力增强成功，也应该考虑引入生成模型或混合模型。

5. **多变量/协变量任务成为主评测对象**
   - 当目标是评估 foundation model 对复杂协变量、层级结构、跨序列共享模式的能力时，纯手写公式可能不足。

但即使引入深度模型，也建议保留显式 feature profiler 作为验收层：

```text
深度/规则生成器 -> 重新抽取显式特征 -> 分布约束验收 -> 能力增强验收
```

也就是说，深度模型可以负责“生成”，但不要让它成为唯一的“合理性解释”。

### 3. 真实数据基底怎么选？

建议采用分层基底，而不是只选一个数据集：

#### Tier 1：通用校准基底

优先使用 **Monash Time Series Forecasting Repository / Archive** 作为第一基底。

理由：

- 它是专门为 forecasting benchmark 整理的公开时间序列仓库。
- 覆盖多领域、多频率、多长度、有缺失值等差异。
- 论文中本身就做了 feature analysis，方向和我们要做的 profile 很一致。
- 仓库维护了统一格式，工程接入成本相对低。

#### Tier 2：经典竞赛基底

使用 M4 作为大规模单变量校准补充。

理由：

- M4 有 100,000 条时间序列，覆盖 yearly、quarterly、monthly、weekly、daily、hourly 等频率。
- 适合统计 trend/seasonality/noise/intermittency 的大样本分布。
- 对单变量 forecasting 的论文认可度高。

使用 M5 作为零售/间歇性/层级销售数据补充。

理由：

- M5 来自 Walmart 销售数据，适合测试 intermittent demand、promotion/event response、hierarchical retail patterns。
- 它不应该作为所有能力维度的通用基底，但适合作为“间歇性、事件响应、零售销售”类能力的专门基底。

#### Tier 3：多变量与协变量基底

后续如果多变量/协变量评测成为重点，可以补：

- Electricity / Traffic / Solar / Weather 等多变量数据集。
- ETT/Weather 类长序列数据，用于 long-horizon、covariate-like、multivariate setting。
- 项目内部真实上传数据，用于形成“本地业务 profile”。

#### 推荐策略

第一版不要把所有数据混成一个大池，而是按以下 key 分桶：

```text
domain / frequency / horizon / context_length / target_dim / covariate_dim
```

每个桶独立统计特征分布。生成 synthetic 时先选择一个 anchor bucket，再在该 bucket 的 feature quantile 范围内增强目标能力。

## Proposed v2 Pipeline

```text
Real datasets
  -> Window sampler
  -> Feature profiler
  -> Anchor bucket distributions
  -> Capability-specific target ranges
  -> Synthetic generator
  -> Feature re-extraction
  -> Distribution acceptance / rejection
  -> Benchmark shard
```

### Feature Profiler

第一版建议实现一个离线 profiler，输入真实 shard 或公开数据集，输出 JSON profile：

```json
{
  "profile_id": "...",
  "dataset": "...",
  "bucket": {
    "domain": "energy",
    "frequency": "h",
    "context_length": 96,
    "horizon": 24,
    "target_dim": 1,
    "covariate_dim": 0
  },
  "features": {
    "trend_strength": {"p05": 0.1, "p50": 0.35, "p95": 0.8},
    "seasonal_strength": {"p05": 0.0, "p50": 0.5, "p95": 0.9},
    "acf1": {"p05": 0.2, "p50": 0.7, "p95": 0.95}
  }
}
```

### Capability Definition Contract

每个能力维度必须写成可验证契约：

```text
capability: trend
target_features:
  - trend_strength
  - local_slope_abs
  - curvature_abs
controlled_features:
  - seasonal_strength
  - noise_ratio
  - outlier_rate
difficulty:
  1: target feature near bucket p50
  3: target feature near bucket p75
  5: target feature near bucket p90/p95
acceptance:
  - target features hit range
  - controlled features remain inside bucket p05-p95
  - generated MASE baseline remains finite
```

### Generator Acceptance

生成器不应该只根据输入参数返回结果，而应该做生成后验收：

1. 生成候选 sample。
2. 重新抽取特征。
3. 判断目标特征是否命中难度区间。
4. 判断目标特征是否没有超过真实分布的上限倍数，例如默认不超过 `p95 * multiplier`；`trend_strength` / `seasonal_strength` 等天然落在 `[0, 1]` 的特征还必须夹到 `1.0`。
5. 判断非目标特征是否仍在真实分布范围内。
6. 不通过则 resample / adjust / reject。

## Prototype Status

2026-06-29 已落地第一版离线 profiler 原型：

```text
scripts/synthetic_feature_profile.py
```

当前能力：

- 读取 CSV。
- 读取 Monash `.tsf` 文件。
- 读取包含 `.tsf` 的 Monash/Zenodo `.zip`。
- 按 `context_length + horizon` 生成 forecast windows。
- 输出窗口级显式特征的 `p05/p25/p50/p75/p95/mean/std/min/max`。
- 输出 `target_feature_caps`，默认按 `p95 * target_max_multiplier` 给出目标特征增强上限。
- 对 bounded features 自动把 cap 限制到 `1.0`。

示例：

```bash
python3 scripts/synthetic_feature_profile.py \
  runtime/research/us_births_dataset.zip \
  --context-length 365 \
  --horizon 30 \
  --stride 30 \
  --season-length 7 \
  --max-windows 20 \
  --domain demographics \
  --dataset-name us_births_weekly \
  --target-max-multiplier 1.5 \
  --out runtime/research/us_births_weekly_profile.json
```

US Births smoke 观察：

- 数据源：Monash / Zenodo `us_births_dataset.zip`，约 16 KB，daily，单变量，单序列。
- `season_length=365` 且窗口长度只有 `365+30` 时，无法覆盖两个完整年周期，季节强度会被保守置为 0。
- `season_length=7` 能捕获 weekly seasonal strength，适合作为最小外部数据 smoke。

子代理外部数据调研结论：

- 最小 smoke 首选 **US Births Dataset**：daily、单变量、1 条序列、zip 约 16 KB。
- 第一轮多序列/hourly profile 建议 **M4 Hourly Dataset**：414 条 hourly 序列、zip 约 485 KB。
- Monash 数据入口：https://forecastingdata.org/
- US Births Zenodo：https://zenodo.org/records/4656049
- M4 Hourly Zenodo：https://zenodo.org/records/4656589

## Research Tasks

### R1. Feature Literature Matrix

为每个能力维度建立“特征 - 论文依据 - 工程实现 - 风险”的矩阵。

优先来源：

- STL/MSTL trend and seasonality strength。
- tsfeatures。
- catch22 / hctsa。
- tsfresh。
- time-series augmentation survey。

### R2. Real Dataset Inventory

梳理可使用真实基底：

- Monash archive：通用 forecasting anchor。
- M4：单变量大规模 anchor。
- M5：零售/间歇性/event anchor。
- Electricity/Traffic/Weather/ETT：多变量与协变量候选。
- TSBenchmark 内部真实上传样本：业务本地 anchor。

产出：

- 数据集来源。
- license / research use 限制。
- 频率、长度、数量、缺失值。
- 是否支持多变量/协变量。
- 接入成本。

### R3. Prototype Feature Profiler

先用 2-3 个数据源跑：

- Monash 中一个 hourly/daily 数据集。
- M4 hourly/monthly 子集。
- M5 或一个 intermittent demand 数据集。

输出 quantile profile 和质量报告。

### R4. Trend + Seasonality v2 Pilot

先重构 trend 和 multi-seasonal 两个能力：

- 它们最容易用显式特征解释。
- STL/tsfeatures 依据相对成熟。
- 当前 UI 能力雷达已经主要依赖这些维度，收益直接。

### R5. Model Response Experiment

对旧生成器、新生成器、真实样本子集分别跑：

- naive / seasonal naive。
- ARIMA/ETS 或轻量统计基线。
- 当前接入的 forecasting models。

检查：

- difficulty 是否与 baseline error 单调相关。
- 不同模型短板是否符合能力维度预期。
- synthetic 排名是否和真实 anchor 子集完全脱节。

## Initial Decisions

1. 第一阶段使用显式数学/统计特征，不引入深度生成模型。
2. Monash archive 作为通用真实分布基底，M4/M5 作为补充基底。
3. 生成器必须加入 feature re-extraction 和 acceptance check。
4. 能力维度先从 trend、multi-seasonal 做 v2 pilot。
5. 深度模型只在显式特征无法稳定描述或规则生成无法通过验收时引入。

## References

- Monash Time Series Forecasting Repository: https://forecastingdata.org/
- Godahewa et al., “Monash Time Series Forecasting Archive”: https://arxiv.org/abs/2105.06643
- Hyndman, “Measuring strength of trend and seasonality”: https://otexts.com/fpp2/seasonal-strength.html
- tsfeatures `stl_features`: https://pkg.robjhyndman.com/tsfeatures/reference/stl_features.html
- Makridakis et al., “The M4 Competition: 100,000 time series and 61 forecasting methods”: https://ideas.repec.org/a/eee/intfor/v36y2020i1p54-74.html
- M5 Forecasting Accuracy competition: https://www.kaggle.com/competitions/m5-forecasting-accuracy
- Christ et al., “tsfresh - A Python package”: https://tsfresh.readthedocs.io/en/latest/text/introduction.html
- Fulcher et al., “Highly comparative time-series analysis”: https://arxiv.org/abs/1304.1209
- Wen et al., “Time Series Data Augmentation for Deep Learning: A Survey”: https://www.ijcai.org/proceedings/2021/0631.pdf
