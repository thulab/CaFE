# Paper v2 E3：六个单变量能力画像

日期：2026-07-17

## 结论

E3-v2 已完成。它只读复用 sealed E2-v2 的 21,600 条样本与 172,800 条基础模型 +
seasonal-naive 预测，形成 42 个 `model × capability` 连续画像；没有重新生成数据或调用
模型。

点估计上，`Chronos-2` 的六能力 macro MASE 为 0.5562，`toto2.0` 为 0.5581，两者边际
95% CI 重叠，不能据此宣称稳定冠亚军。更重要的是模型存在局部能力差异：

- `Chronos-2` 在 trend、multi-seasonal、regime switching、nonlinear persistence 的
  MASE 点估计最低。
- `toto2.0` 在 time-varying seasonality 与 predictable intermittency 最低。
- 最大的实质配对缺陷候选集中在 time-varying seasonality、predictable intermittency
  与 trend，而不是所有模型按一个固定顺序整体变差。

E2 的严格 cell-wise 排名判据未通过，所以 E3 以连续分数、relative skill、intensity 曲线
和配对 CI 为主要证据；hard rank 只作导航。

## 六能力 macro

| Model | Macro MASE [95% CI] | Skill vs SNaive | Mean cap. rank* |
| --- | ---: | ---: | ---: |
| `Chronos-2` | 0.5562 [0.5527, 0.5599] | 43.6% | 1.33 |
| `toto2.0` | 0.5581 [0.5546, 0.5617] | 43.4% | 1.67 |
| `tirex2` | 0.5770 [0.5736, 0.5808] | 41.5% | 4.33 |
| `timesfm2.5` | 0.5772 [0.5739, 0.5810] | 41.6% | 4.17 |
| `Timer-3.5` | 0.5802 [0.5767, 0.5841] | 41.3% | 4.00 |
| `moirai2` | 0.5892 [0.5858, 0.5929] | 40.4% | 6.17 |
| `Timer-3.0` | 0.5910 [0.5876, 0.5946] | 40.2% | 6.33 |

\* hard rank 对近似并列敏感，不作为主要证据。七个模型在全部六能力上均优于
seasonal naive。

## 分能力点估计

| Capability | Leader | MASE [95% CI] | Worst point estimate | Leader skill |
| --- | --- | ---: | --- | ---: |
| `trend` | `Chronos-2` | 0.7105 [0.6983, 0.7225] | `Timer-3.0` | 30.8% |
| `multi_seasonal` | `Chronos-2` | 0.3630 [0.3559, 0.3703] | `moirai2` | 54.3% |
| `time_varying_seasonality` | `toto2.0` | 0.4699 [0.4631, 0.4775] | `moirai2` | 56.0% |
| `regime_switching` | `Chronos-2` | 0.4240 [0.4186, 0.4298] | `moirai2` | 63.7% |
| `nonlinear_persistence` | `Chronos-2` | 0.6445 [0.6370, 0.6529] | `Timer-3.0` | 28.5% |
| `predictable_intermittency` | `toto2.0` | 0.7140 [0.7036, 0.7256] | `Timer-3.0` | 29.3% |

## 配对能力缺陷候选

`model_capability_contrasts.csv` 使用相同 bootstrap draws，将每个模型与该 capability 的
observed leader 配对比较。CSV 保留全部结果；下表列 relative MASE gap 最大的前 10 项，
且 gap 的 95% CI 完全高于 0。

| Model | Capability | Reference | MASE gap [95% CI] | Relative gap |
| --- | --- | --- | ---: | ---: |
| `moirai2` | `time_varying_seasonality` | `toto2.0` | 0.0618 [0.0573, 0.0663] | 13.1% |
| `Timer-3.0` | `time_varying_seasonality` | `toto2.0` | 0.0533 [0.0494, 0.0570] | 11.3% |
| `Timer-3.0` | `predictable_intermittency` | `toto2.0` | 0.0684 [0.0639, 0.0731] | 9.6% |
| `Timer-3.5` | `predictable_intermittency` | `toto2.0` | 0.0665 [0.0620, 0.0710] | 9.3% |
| `Timer-3.0` | `trend` | `Chronos-2` | 0.0583 [0.0534, 0.0637] | 8.2% |
| `timesfm2.5` | `time_varying_seasonality` | `toto2.0` | 0.0377 [0.0348, 0.0405] | 8.0% |
| `Timer-3.5` | `time_varying_seasonality` | `toto2.0` | 0.0354 [0.0317, 0.0395] | 7.5% |
| `moirai2` | `trend` | `Chronos-2` | 0.0528 [0.0484, 0.0571] | 7.4% |
| `tirex2` | `predictable_intermittency` | `toto2.0` | 0.0444 [0.0409, 0.0479] | 6.2% |
| `moirai2` | `predictable_intermittency` | `toto2.0` | 0.0443 [0.0405, 0.0479] | 6.2% |

这些是 E4 hypothesis-generation 结果，不是 multiplicity-adjusted 显著性声明：reference
leader 由同一 E3 数据选出。E4 必须先按真实 train-only loading 预注册数据集与模型对照，
再读取真实 test 结果。

## Intensity-response

- `trend`：除 `Chronos-2` 随 intensity 增强而略改善外，多数模型变差；
  `Timer-3.0` 的上升最明显。
- `multi_seasonal`、`time_varying_seasonality`、`regime_switching`：结构越显著，所有模型
  通常越容易预测，证明 intensity 不是统一“难度”。
- `nonlinear_persistence`：五档曲线接近平坦，模型差距也很小；真实 train audit 中 9 个
  profiles 的 nonlinear coordinate 均为 1，E4 不应把它作为首批强外部效度案例。
- `predictable_intermittency`：`toto2.0` 与 `Chronos-2` 相对稳定；Timer-3.0/3.5 在强脉冲
  档出现明显劣化。
- 最大端点响应为 `moirai2 / regime_switching`：MASE 0.5940 → 0.2537（-57.3%）。

## 与 E4 的预注册接口

真实 train-only audit 给出的 high-loading 候选为：

- trend：`SZ_TAXI/H`、`ett1/H`、`ett2/H`
- multi-seasonal：`SZ_TAXI/H`、`kdd_cup_2018_with_missing/H`、`ett2/H`
- time-varying seasonality：KDD、LOOP Seattle、SZ Taxi、ETT1/2、Bitbrains 均接近 5
- regime switching：KDD、SZ Taxi、ETT1/2
- predictable intermittency：Solar、LOOP Seattle、Bitbrains Fast/Rnd

首批最有力的 synthetic→real 假设是：

1. `moirai2` 与 `Timer-3.0` 相对 `toto2.0` 的时变季节性缺陷，应在 high-loading 的
   KDD/LOOP/ETT/Bitbrains 真实 test windows 上表现为同向 paired gap。
2. `Timer-3.0` 与 `Timer-3.5` 相对 `toto2.0` 的可预测间歇性缺陷，应在
   Solar/LOOP/Bitbrains 上同向出现。
3. `Timer-3.0` 与 `moirai2` 相对 `Chronos-2` 的趋势缺陷，应在 SZ Taxi/ETT1/ETT2
   上同向出现。

E4 不应基于真实模型结果再挑数据集；dataset/capability mapping、窗口、模型对照和相关性
统计必须先写入 selection manifest。

## 封存信息

- runner/protocol commit：`47c2d61466a1d17a41afb238f4bf1c5bb745857b`
- 唯一输入：E2 manifest
  `91b61c7d4b3d4cd81da28f011d6d6e0810db423d1c16b0bb336a6f17a2e1f34d`
- 输出：`runtime/paper_exp/v2/E3_model_capability_profiles/`
- 大小：3,102,988 bytes
- manifest SHA-256：
  `147e494f89017a3cb7938f92e2b75f7dffdf3290286408d51f1485d9aca1f1a1`
- manifest 文件数：25

正式输出包含 5 个 CSV 主表、`model_capability_contrasts.csv`、`summary.json`、
`report.md`、`paper_tables.md` 与 5 张图；每张图均保留 PNG、SVG、PDF。

