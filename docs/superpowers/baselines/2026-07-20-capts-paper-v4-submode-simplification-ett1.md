# capts-paper-v4 capability 子模式简化：ETT1 小规模验证

日期：2026-07-20

## 结论

本轮不回退九能力生成公式，而是把同一 capability cell 内会改变预测规律的随机
子模式收缩为 dataset-relative 固定规律。ETT1 的逐数据集校准、四 lookback 验收、
两组独立 `N=160` seed bank 和六个快速模型均已完成。

结果支持保留本次简化：

- ETT1 单变量 6 个能力全部 supported，结构不适用的 3 个多变量能力如实
  unsupported；
- 校准资格样本 `240/240` 通过，正式生成每个 bank `4,800` 条 master samples、
  `960` 个 paired groups，均在第一次尝试通过；
- 相比 `capts-paper-v3`，Oracle-context 正式排名的平均 pairwise agreement
  从 `0.8178` 提高到 `0.8911`，通过 `≥0.80` 的 cells 从 `20/30`
  提高到 `27/30`；
- 固定 `L=504` 的平均 agreement 从 `0.9022` 提高到 `0.9111`，通过 cells
  从 `27/30` 提高到 `29/30`；Top-1 agreement 从 `0.9333` 降为
  `0.8667`，因此排名改善并非
  所有统计量都单调改善；
- nonlinear 的两个早期候选分别因“消除强度效应”和“产生 24 步周期捷径”被淘汰。
  最终固定非线性机制令六模型平均 MASE 从 I1 到 I5 增加约 `0.064–0.087`，
  同时保持两组 seed bank 的连续分数和排名可复现。

本结果仅验证 ETT1，不外推为所有 dataset 均已通过。正式扩展前仍需对其他 dataset
重复校准、生成和小规模模型检查。

## 1. 问题诊断

`capts-paper-v3` 的一个 `dataset × capability × intensity` cell 内仍混合了多个
预测规律：

- `trend` 随机抽取 shape scale、curvature ratio 和背景 AR 快根；
- `regime_switching` 随机排列 4 段 duration motif，并随机缩放通道；
- `nonlinear_persistence` 在 3 个 lag、2 个 transform、4 个 frequency、
  4 个 offset、3 个 AR 系数和 3 个 seasonal-memory 系数间组合，共
  `3×2×4×4×3×3=864` 种离散机制组合；
- `predictable_intermittency` 随机排列 3 段 interval motif、连续抽取 spread，
  并随机缩放通道。

这些变化都能从 history 学习，但它们同时改变了模型面对的预测问题。seed bank
改变后，有限样本中的子模式配比也会变化，给 capability 主效应叠加不必要的
mixture variance。

`multi_seasonal` 和 `time_varying_seasonality` 的 v3 排名可靠性已经较高，且内部
频率/调制变化属于能力定义本身，本轮不改。

## 2. capts-paper-v4 冻结规则

强度仍只控制机制剂量；dataset-local profile、五档相对强度、随机 innovations 和
必要的对称性随机量不变。

| 能力 | 固定的 cell 内规律 | 保留的样本差异 |
|---|---|---|
| `trend` | shape scale=`1`，curvature ratio=`0.06`，背景 AR 快根=`0.235` | 正/负方向、噪声 |
| `regime_switching` | dataset-relative 非均匀 duration motif `[b,b+2s,b-s,b+s]`，通道 scale=`1` | 初始状态正负号、噪声 |
| `nonlinear_persistence` | lag=`season/3`，`tanh(1.4x+0.6)`，AR=`0.08`，seasonal memory=`0.05` | innovations |
| `predictable_intermittency` | context-feasible motif order=`[-1,+2,+1]`，relative spread=`0.18`，通道 scale=`1`；ETT1 motif=`[17,29,25]` | phase、噪声 |

固定的是 nuisance/submode law，不是样本值。不同 sample seed 仍会产生不同曲线；
同一 paired group 的五档 intensity 仍共享结构和 nuisance realization。

生成器版本更新为 `capts-paper-v4`，旧 calibration artifacts 不得与新版本混用。

## 3. nonlinear / intermittency 候选淘汰与修正

第一个候选使用固定 `sin²(1.1x)`、零 offset。它虽然通过 feature gate，但六模型
平均 MASE 几乎不随强度变化：

| Context policy | Bank | I1 | I5 | I5-I1 |
|---|---|---:|---:|---:|
| Oracle | A | 0.7080 | 0.7082 | 0.0002 |
| Oracle | B | 0.6913 | 0.6878 | -0.0035 |
| L=504 | A | 0.7384 | 0.7389 | 0.0004 |
| L=504 | B | 0.7297 | 0.7283 | -0.0013 |

这说明“资格验收通过”不能替代模型层面的能力辨识检查。该候选已淘汰。

第二个候选使用 lag=`season/2` 和 `sin²(1.4x+0.6)`。它在模型层面产生很强的
I1→I5 差异，但完整测试揭示：

- seasonal-naive / last-value MAE ratio 只有 `0.6799`，形成明显的 24 步捷径；
- 通用 `nonlinear_conditional_gain` 与五档 intensity 的相关仅 `0.1992`；
- 高强度 capability-aware contrast 未通过。

因此也予以淘汰，不能因为 seed-bank 排名好看而保留。

最终 nonlinear 使用 lag=`season/3` 的 shifted tanh 固定递推。intermittency 同时
从包含 nominal 24 步间隔的 `[20,28,24]` 改为在 phase 抽取前确定、上下文可行且
不含 24 步间隔的 `[17,29,25]`。两个能力的捷径和能力对照硬门槛为：

| Capability | Seasonal / last MAE ratio | Aware win rate | Mean relative loss gain | Gate |
|---|---:|---:|---:|---|
| `nonlinear_persistence` | 0.9689 | 0.8516 | 0.1003 | passed |
| `predictable_intermittency` | 1.1605 | 0.9375 | 0.4178 | passed |

最终 nonlinear 的六模型平均 MASE 为：

| Context policy | Bank | I1 | I2 | I3 | I4 | I5 | I5-I1 |
|---|---|---:|---:|---:|---:|---:|---:|
| Oracle | A | 0.7074 | 0.7137 | 0.7286 | 0.7472 | 0.7715 | 0.0641 |
| Oracle | B | 0.6917 | 0.7059 | 0.7233 | 0.7439 | 0.7706 | 0.0789 |
| L=504 | A | 0.7374 | 0.7470 | 0.7660 | 0.7872 | 0.8142 | 0.0767 |
| L=504 | B | 0.7298 | 0.7447 | 0.7629 | 0.7858 | 0.8163 | 0.0865 |

最终 nonlinear 的独立 seed-bank 结果：

| 指标 | Oracle | L=504 |
|---|---:|---:|
| Raw MASE Lin CCC / Spearman | 0.9435 / 0.9858 | 0.9866 / 0.9884 |
| Relative log-MASE Lin CCC / Spearman | 0.7708 / 0.6912 | 0.8732 / 0.6641 |
| Relative difference median / p90 | 0.0094 / 0.0232 | 0.0045 / 0.0109 |
| Rank agreement mean | 0.7467 | 0.7867 |
| Rank cells ≥0.80 | 3/5 | 4/5 |
| Top-1 agreement | 0.4 | 0.8 |
| Tie-aware 显著方向冲突 | 0 | 0 |

Oracle rank agreement 略高于 v3 的 `0.7333`，固定 L=504 则低于 v3 的
`0.8267`；但连续 raw MASE 高度一致，且两种 policy 都不存在显著方向冲突。按 E2
修正协议，排名是正式共同结果，但不能用完整名次波动替代连续分数和 tie-aware 结论。

## 4. 六能力正式排名结果

最终 nonlinear 和 intermittency 之外的 4 个能力，在修正候选前后每个 bank 各
`3,200/3,200` master targets 的 SHA256 全部一致，因此可以将已完成的四能力推理
与最终两能力推理无损拼接。下表是拼接后的正式 rank 结果。

### Oracle context

| Capability | Agreement mean | Cells ≥0.80 | Top-1 agreement |
|---|---:|---:|---:|
| `multi_seasonal` | 0.9733 | 5/5 | 1.0 |
| `nonlinear_persistence` | 0.7467 | 3/5 | 0.4 |
| `predictable_intermittency` | 0.9733 | 5/5 | 1.0 |
| `regime_switching` | 0.8133 | 4/5 | 0.8 |
| `time_varying_seasonality` | 0.9467 | 5/5 | 1.0 |
| `trend` | 0.8933 | 5/5 | 0.8 |
| **Overall** | **0.8911** | **27/30** | **0.8333** |

### 固定 L=504

| Capability | Agreement mean | Cells ≥0.80 | Top-1 agreement |
|---|---:|---:|---:|
| `multi_seasonal` | 0.9733 | 5/5 | 1.0 |
| `nonlinear_persistence` | 0.7867 | 4/5 | 0.8 |
| `predictable_intermittency` | 1.0000 | 5/5 | 1.0 |
| `regime_switching` | 0.8267 | 5/5 | 0.8 |
| `time_varying_seasonality` | 0.9733 | 5/5 | 1.0 |
| `trend` | 0.9067 | 5/5 | 0.6 |
| **Overall** | **0.9111** | **29/30** | **0.8667** |

## 5. 运行范围与产物

- Dataset：`gift_ett1_h`；
- Models：`Timer-3.5`、`Timer-3.0`、`Chronos-2`、`moirai2`、
  `toto2.0`、`tirex2`；
- Context：`96/168/336/504`，horizon=`48`；
- Bank A/B：每个 capability/intensity 各 `160` 条 master samples；
- 最终 nonlinear + intermittency 推理：每个 bank `38,400` 条
  model-view predictions，两个 bank 共 `76,800`，失败 `0`；
- Runtime 根目录：
  `runtime/paper_exp/v5_submode_simplified_ett1_r3/`；
- 最终两能力分析：
  `runtime/paper_exp/v5_submode_simplified_ett1_r3/E2_simplified_seed_bank_reliability/`。

为支持这种低成本回归，正式推理脚本新增 `--capabilities`，从完整 master collection
构造确定性的 capability 子集，并在 inference config 中记录过滤条件。单 capability
profile 的逐模型 correlation 没有定义，分析器现在显式记录为 `null`，不再报错或
产生伪相关。

## 6. 判定边界

本轮接受 `capts-paper-v4`，因为：

1. ETT1 的逐数据集提取和四 lookback 生成全部通过；
2. 高 intensity 的机制剂量明显高于低 intensity，模型响应也随档位发生清晰变化；
3. 两组独立 seed bank 的连续分数高度一致；
4. Oracle-context 和 L=504 的平均 rank agreement 均不低于 v3，且显著模型对
   没有方向冲突；
5. 子模式数量显著减少，同时保留了方向、phase、innovations 等不会改变能力定义的
   合理样本差异。

下一步不能直接运行全量正式 E2。应先把相同检查扩展到多个不同 domain 的 dataset，
确认不是 ETT1 特例，再冻结 `capts-paper-v4` calibration artifacts。
