# capts-paper-v3 九能力公式验证

日期：2026-07-19

## 目的

本轮验证检查三项公式级条件：

1. 对同一 candidate seed，`intensity` 只改变目标机制强度，不改变周期、
   motif、lag、载荷、协变量路径、背景或噪声；
2. 在 dataset-local profile、真实容忍区间和四 lookback 双门控下，多数可用
   dataset 能完成五档校准；
3. supported cell 能稳定生成，且 intensity 5 的主特征相对 intensity 1
   有清楚的增强。

这是一轮公式接受性审计，不是正式论文模型实验。

## 数据与预算

- 单变量：M4 Hourly、Electricity/H、Solar/H、ETT1/H、ETT2/H、Jena
  Weather/H、Loop Seattle/H、SZ Taxi/H、M_DENSE/H、Bitbrains Fast/H；
- panel：3 个具有足够有效拆分的数据集；
- hierarchy：M5 Daily；
- covariate：GEFCom2014 Load 与 Solar；
- 固定 `H=48`，联合验收 `L=96/168/336/504` 四个 suffix views；
- 每个 dataset 最多 120 个真实窗口；
- 八个能力使用每网格 4 个样本的保守低预算筛查；时变季节性因统计量方差较高，
  按正式默认预算 16 个样本复验，代码内部将其拟合 bank 加倍到 32；
- 每个 supported cell、每个 intensity 验收 2 个样本，最多重试 64 次。

无法形成独立三路拆分的数据集不进入分母；它们保留
`dataset_split_failed`，不伪装成公式不支持。

## Dataset-local 支持结果

| capability | supported / usable dataset | 支持率 |
| --- | ---: | ---: |
| `trend` | 6 / 10 | 60% |
| `multi_seasonal` | 9 / 10 | 90% |
| `time_varying_seasonality` | 9 / 10 | 90% |
| `regime_switching` | 9 / 10 | 90% |
| `nonlinear_persistence` | 8 / 10 | 80% |
| `predictable_intermittency` | 9 / 10 | 90% |
| `common_factor` | 3 / 3 | 100% |
| `hierarchical_coherence` | 1 / 1 | 100% |
| `covariate_response` | 2 / 2 | 100% |

所有能力均达到“多数可用 dataset 支持”。时变季节性唯一仍不支持的可用数据集为
Bitbrains Fast/H：其真实 `seasonal_amplitude_modulation` 下界仍高于生成器响应
上界，记录为 `no_real_generator_tolerance_overlap`，没有继续为单一数据集放大机制。

## 生成稳定性与强度可见性

替换时变季节性的正式预算复验后，共有 56 个 supported cells。每个 cell
验收 `5 intensities × 2 samples`，合计：

```text
accepted = 560
failed   = 0
四个 lookback 的 construction + feature-support + near-distance 均通过
```

下表汇总验收样本在 `L=504` 上的主特征中位数。状态切换使用补齐
latent-aligned history clock 诊断后的代表性复验 shard；其余能力覆盖对应 supported
cells。

| capability | intensity 1 | intensity 5 | 变化 |
| --- | ---: | ---: | ---: |
| `trend` | 0.021253 | 0.713090 | +0.691837 |
| `multi_seasonal` | 0.068016 | 0.511797 | +0.443781 |
| `time_varying_seasonality` | 0.266399 | 0.443888 | +0.177489 |
| `regime_switching` | 0.013938 | 0.173079 | +0.159141 |
| `nonlinear_persistence` | 0.000027 | 0.001475 | +0.001448 |
| `predictable_intermittency` | 0.006352 | 0.161525 | +0.155173 |
| `common_factor` | 0.711215 | 0.993438 | +0.282223 |
| `hierarchical_coherence` | 0.218500 | 0.495508 | +0.277008 |
| `covariate_response` | 0.334148 | 0.709358 | +0.375210 |

九个主特征的 I5 中位数均高于 I1。公式级配对测试进一步验证同 seed 下非目标
结构不随 intensity 改变。状态切换的高强度状态间距在独立 seed 审计中为
`1.25–1.81` 个 context 标准差；层级加总残差保持为 0。

## 本轮公式决定

- trend：样本特有的有界二次曲率比；
- multi-seasonal：profile 主周期加两个样本特有、谱上可分辨的整数周期；
- time-varying seasonality：样本特有的二谐波调制律，AM/FM 深度上限有限放宽；
- regime switching：四段 explicit-duration motif，历史和未来确定延续；
- nonlinear persistence：样本特有 lag 与有界非线性族，递归稳定性受约束；
- predictable intermittency：三段非等间隔 motif；只额外记录
  `interval_reconstruction_mae` 和 `event_window_radius`，不引入复杂事件模型；
- common factor：固定 rank 1 的稳定动态因子与 RMS-normalized loadings；
- hierarchical coherence：零和正交 contrast paths，父节点严格等于子节点和；
- covariate response：历史多事件和已知未来事件共享固定系数结构。

生成器版本更新为 `capts-paper-v3`。旧 conditioning calibration 不作为本轮公式的
资格证据，正式 Paper v4 实验应从当前代码重新构建 dataset-local artifacts。

## 回归

```text
九公式及相关 focused tests: 110 passed
backend full suite:             499 passed
```
