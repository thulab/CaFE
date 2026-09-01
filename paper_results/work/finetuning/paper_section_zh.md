## 跨种子微调：能力拟合具有显著的目标依赖性

### 实验设置

我们进一步研究模型能否通过直接接触 CaFE 生成样本来拟合评测指标。以
GIFT-Eval short-term、预测长度 48 为基础，我们分别用增强种子 A
（`2026082701`）和 B（`2026082702`）确定性抽取约 10% 的官方实例。种子
A 用于微调和同种子评估，种子 B 仅用于跨种子评估。两者分别包含 50,535
和 48,365 个 treatment，且没有重复的 treatment sample ID。需要说明的是，
两组仍共享 240 个官方实例，约占各自实例数的 10%，因此该设置主要检验
新增强实例上的迁移，而不是严格的底层实例隔离。

我们比较两种 LoRA 微调方式。第一种使用 Chronos-2 官方多分位数损失，在
完整 treatment 序列内部随机采样预测原点；该损失基于历史均值和标准差进行
归一化，并不直接等价于 MASE。第二种在固定官方预测原点同时输入
official–treatment 对，直接最小化受影响维度上的成对 effect 平方 NRMSE。
两种方式均训练 40k 步并每 4k 步评估一次。除目标外，两者在训练输入形式、
预测原点、学习率（`1e-4` 对 `1e-5`）和等效数据遍历次数上也不同，因此本
实验应解释为两种完整适配协议的比较，而非仅改变损失函数的单因素消融。

### 主要结果

下表同时报告 step 0、各指标在观测 checkpoint 中的最优值和 40k 最终值。
括号内为相对 step 0 的变化，所有指标均为越低越好。“最优”是在报告语料上
事后选择的描述性结果；主要结论同时以预先固定的 40k 终点为依据。

| 训练协议 | 评估语料 | MASE@0 | 最优 MASE（step） | MASE@40k | effect NRMSE@0 | 最优 effect NRMSE（step） | effect NRMSE@40k |
|---|---|---:|---:|---:|---:|---:|---:|
| Chronos 默认分位数损失 | seed A | 1.0199 | **0.9605**（40k，−5.82%） | **0.9605**（−5.82%） | 0.4637 | **0.4306**（16k，−7.14%） | 0.4770（+2.88%） |
| Chronos 默认分位数损失 | seed B | 0.9489 | **0.8546**（40k，−9.94%） | **0.8546**（−9.94%） | 0.4594 | **0.4395**（8k，−4.33%） | 0.4865（+5.90%） |
| 成对 effect NRMSE 损失 | seed A | 1.0199 | 1.0199（0，0.00%） | 1.2357（+21.15%） | 0.4637 | **0.3441**（36k，−25.80%） | **0.3447**（−25.66%） |
| 成对 effect NRMSE 损失 | seed B | 0.9489 | 0.9489（0，0.00%） | 1.1379（+19.91%） | 0.4594 | **0.3852**（28k，−16.15%） | **0.3887**（−15.38%） |

默认分位数微调持续降低 MASE，却没有同步改善 effect NRMSE。在 seed A
上，MASE 于 40k 降低 5.82%，而 effect NRMSE 仅在 16k 前短暂改善，最终
反而上升 2.88%；seed B 呈现相同模式，MASE 于 40k 降低 9.94%，effect
NRMSE 则在 8k 达到短暂最优后上升 5.90%。相反，显式优化成对 effect 的
协议显著降低 effect NRMSE，却牺牲绝对预测精度：seed A/B 的最终 effect
NRMSE 分别降低 25.66%/15.38%，但 MASE 分别上升 21.15%/19.91%。因此，
MASE 和 effect NRMSE 在适配过程中提供了互不冗余的信号；仅优化绝对水平
预测或仅优化 treatment 响应，均不足以保证另一维能力随之提升。

该结论在新增强种子上仍然成立。对于默认协议，seed A 与 B 的 MASE 相对
变化曲线高度相关（Pearson `r=0.903`，Spearman `rho=0.952`），且 seed B
的最终改善反而比 seed A 大 4.12 个百分点。对于 effect 协议，两种子的
MASE 退化曲线同样高度一致（`r=0.995`，`rho=0.976`）。effect NRMSE 在
seed B 上仍获得 15.38% 的最终改善，约保留 seed A 改善幅度的 60%，但其
逐 checkpoint 走势相关性较低（`r=0.521`），并表现出更晚且非单调的改善；
seed A 与 B 的最终增益相差 10.27 个百分点。这说明精确 treatment 样本的
变化并未阻止指标定向适配，但 effect 拟合中存在可观的种子特异成分。

![微调相对变化曲线](/Users/xiangmy21/Documents/CaFE/paper_results/work/finetuning/finetuning_relative_change.png)

**图 X：不同微调协议下的 MASE 与 effect NRMSE 变化。** 实线表示跨种子
seed B，虚线表示训练种子 A。默认分位数协议最终沿 MASE 方向改善而在
effect NRMSE 方向退化；成对 effect 协议呈现相反运动。所有变化均相对于
各语料自己的 step-0 baseline。

### 两指标之间的 Pareto 轨迹

checkpoint 轨迹进一步表明，这一现象不是由单个异常终点造成的。默认协议
在早期存在同时改善两个指标的局部区间，但继续训练后沿着更低 MASE、较高
effect NRMSE 的方向移动；effect 协议则从第一个 checkpoint 起大幅牺牲
MASE，并沿着更低 effect NRMSE 的方向移动。该轨迹为联合目标

`L_joint = L_level + lambda * L_effect`

提供了直接动机：后续训练应在严格控制数据加载、固定预测原点和学习率的
条件下扫描 `lambda`，检验是否能够将 Pareto 前沿推向左下方。当前两条曲线
尚不能证明联合目标一定奏效，也不能证明两个指标存在不可避免的内在冲突。

![微调 Pareto 轨迹](/Users/xiangmy21/Documents/CaFE/paper_results/work/finetuning/finetuning_pareto_trajectory.png)

**图 Y：MASE–effect-NRMSE checkpoint 轨迹。** 横纵轴分别表示相对 step 0
的 MASE 和 effect NRMSE 变化，左下方向代表两个指标同时改善。标注数字为
训练步数（千步）。

### 能力维度上的异质性

总体趋势并不意味着所有能力维度均匀受益。在 seed B 的 40k checkpoint，
默认协议在八种能力中的七种降低 MASE，但只在 `multi_seasonal` 上降低
effect NRMSE；`trend` 的 effect NRMSE 上升 63.2%。成对 effect 协议使八种
能力的 MASE 全部变差，并在八种能力中的五种降低 effect NRMSE；`trend`、
`common_factor` 和 `covariate_impulse_response` 的 effect NRMSE 分别上升
31.7%、9.6% 和 33.2%。这提示联合训练不仅需要调节两个全局损失的权重，
还可能需要考虑能力或 stratum 的均衡采样与加权。

![能力维度热力图](/Users/xiangmy21/Documents/CaFE/paper_results/work/finetuning/finetuning_capability_heatmap.png)

**图 Z：seed B 上各能力在 40k 的相对变化。** 蓝色表示改善，红色表示
退化。数值为相对各自 step-0 的百分比变化。

### 关于训练污染与“记忆”的解释边界

这些实验支持“指标定向拟合具有目标依赖性”，但不支持将目标冲突直接解释
为记忆。首先，默认协议的 MASE 在 seed B 上改善幅度大于 seed A，不符合
“只记住训练增强样本、在新增强样本上失效”的简单预期。其次，两种指标的
误差几何本就不同：MASE 衡量所有目标维度上的绝对水平误差，而 effect
NRMSE 只衡量受影响维度中 treatment 相对 official 的变化；official 与
treatment 预测中共享的误差会在 effect 差分中抵消，却仍会损害 MASE。
因此，当前数据更直接地说明了 level accuracy 与 treatment-response fidelity
的分离，而不是“记忆”和“规律识别”的分离。

从在线评测设计看，只公开 treatment 后的序列而不公开其 official 配对，确实
会移除本实验中成对 effect 损失所需的直接监督，因此可能提高针对 effect
NRMSE 拟合的工程成本。然而，这只是由损失可访问信息推导出的设计假设，
尚未经过攻击实验验证；模型方仍可能近似 official baseline、构造代理目标，
或利用一般化训练取得改善。因而我们将当前实验定位为**跨种子的训练污染压力
测试**：它证明了精确样本随机化不能阻止分布层面的指标适配，同时揭示了
双指标和配对访问控制对降低单一指标过拟合风险的潜在价值。若要声称“抗训练
污染”，还需增加 strictly instance-disjoint 的第三种子、treated-only 与
paired-access 攻击对照，以及多个优化随机种子。

### 可放入摘要或结论的保守表述

> Fine-tuning on CaFE instances reveals a pronounced objective dependence:
> the standard Chronos objective improves absolute forecast accuracy but can
> eventually degrade treatment-effect fidelity, whereas a paired effect loss
> improves effect NRMSE at a substantial cost in MASE. Both patterns transfer
> to a new augmentation seed with no repeated treatment samples, showing that
> seed randomization prevents literal sample replay but does not by itself
> prevent metric-directed adaptation. These results motivate evaluating and
> jointly optimizing level accuracy and treatment-response fidelity, while
> treating access-controlled contamination resistance as a separate question.
