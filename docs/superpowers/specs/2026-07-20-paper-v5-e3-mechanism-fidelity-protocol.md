# Paper v5 E3：机制保真模型能力画像协议

日期：2026-07-20

## 1. 目的与边界

旧版 E3 主要用 MASE 和相对 seasonal-naive skill 描述模型能力。它可以回答预测值离
真实 future 多远，但不能区分：

1. 模型正确延续了目标机制；
2. 模型通过平滑、均值回归或缩小预测幅度获得较小点误差。

Paper v5 E3 因而将正式结果扩展为三个并列层次：

- `point accuracy`：保留逐样本 MASE；
- `mechanism fidelity`：判断输出轨迹是否保留目标机制；
- `ability score`：以点预测误差作为机制分的安全门。

机制评价可以读取合成 future、构造元数据和 latent schedule，但这些信息只存在于
evaluator，绝不能进入模型请求。机制分表示
`mechanism-aligned forecast behavior`，不表示模型内部识别了因果生成机制。

E2 继续负责证明完整能力估计量在独立 seed bank 上可复现；E3 负责描述不同模型在
九个机制上的能力差异。E3 不再把任意 32 样本 batch 的完整排名一致作为前提。

## 2. 冻结输入与小试验范围

E3 只读复用：

- `runtime/paper_exp/v5/E2_dynamic_stability/sample_shards/`；
- `runtime/paper_exp/v5/E2_dynamic_stability/predictions/`；
- E2 已保存的 oracle-context 选择与逐样本 MASE。

普通机制评价不重新生成样本，也不重新调用模型。模型仍按 E2 约定逐样本选择 MASE
最优 context；机制指标在完全相同的 forecast 上计算，不能为 MASE 和机制分分别选择
不同 context。

首轮小试验使用：

| 结构 | 数据集 | 能力 |
|---|---|---|
| 单变量 | `gift_ett1_h` | 前六个单变量能力 |
| 普通多变量面板 | `electricity_hourly_panel` | `common_factor` |
| 显式加和层级 | `m5_daily_hierarchy` | `hierarchical_coherence` |
| 已知未来协变量 | `gefcom2014_load` | `covariate_response` |

不存在一个现有数据集可以同时合法承载三项多变量能力：common factor 需要普通面板，
hierarchy 需要父子加和约束，covariate response 需要 known-future covariates。因此
小试验为每项结构化能力各选一个兼容数据集，不将 unsupported 记为零分。

默认 pilot 每个 `dataset × capability` 选择 8 个 paired groups，并保留其 I1–I5
五档，共 40 条样本；正式扩展恢复全部 160 个 paired groups。

## 3. 逐样本机制分

每种能力根据可用机制拆出四项 `[0,1]` 子分：

- detection：是否出现目标机制；
- timing：事件时间、相位或动态路径是否正确；
- magnitude：机制强度是否恢复；
- selectivity：是否把背景误当成目标机制，或制造多余结构。

逐样本机制保真分使用几何平均：

\[
MFS_{sample}
=
\left(
D^{w_D}T^{w_T}A^{w_A}Q^{w_Q}
\right)^{1/\sum w}.
\]

首版四项等权。使用几何平均意味着任一关键成分完全缺失时，其余成分不能补偿。
幅度比统一使用：

\[
S_{amp}
=
\exp\left(
-\left|\log\frac{\hat A+\epsilon}{A+\epsilon}\right|
\right).
\]

事件时间误差使用预先由生成分辨率确定的容差映射，不根据模型成绩调参。缺少 future
事件、通道数或必要 metadata 的样本记为 `formal_score_eligible=false`，不记为模型
失败。

## 4. 九个能力的冻结评价对象

### 4.1 Trend

在 future 上联合拟合常数、线性项和二次项：

- detection：future 趋势位移方向是否一致；
- timing：预测与真实趋势分量的路径相关；
- magnitude：趋势分量 RMS 比；
- selectivity：线性/二次系数向量夹角。

常数预测没有趋势位移，不能仅凭 MASE 获得趋势机制分。

### 4.2 Multi-seasonal

只在生成器记录的真实 period 上联合做正余弦投影：

- 每个周期的检出与幅度；
- 复系数相位一致性；
- 真实周期集合的增量解释度；
- intensity 剂量只使用 additional periods，不由固定 primary period 主导。

若 H=48 上的联合 harmonic design 数值不可识别，则该样本不进入正式机制分。

### 4.3 Time-varying seasonality

evaluator 根据冻结 metadata 重建 carrier 和 amplitude/phase modulation basis。普通
固定季节载波作为 nuisance，机制对象为 modulated carrier 与固定载波之差：

- 调制方向和路径；
- 调制分量 RMS；
- 相对固定载波的增量解释度。

因此只延续固定季节项的模型不会被误判为掌握时变季节性。

### 4.4 Regime switching

- future switch 的跳变方向正确率；
- 预测最大局部变化与真实 switch 的时间距离；
- alternating state 分量幅度；
- state schedule 的增量解释度。

平滑或均值回归预测没有明确跳变时，timing/magnitude 至少一项接近零。

### 4.5 Nonlinear persistence

不可预测 innovation 不进入机制目标。evaluator 使用 history 构造同一 blind forecast
与 nonlinear-aware conditional forecast，令二者之差为可预测非线性 contrast：

- 模型相对 blind forecast 的变化是否沿正确 nonlinear contrast；
- contrast 的动态路径、幅度与能量占比；
- 预测自身是否保留非平凡的条件动态。

该分数评价输出对非线性条件均值的响应，不要求命中未来随机 innovation。

### 4.6 Predictable intermittency

根据 future pulse centers 和 width 构造 pulse basis：

- 脉冲方向/检出；
- 峰值时间距离；
- pulse 分量幅度；
- pulse 相对背景的增量解释度。

评价采用事件均衡结构，避免大量背景点掩盖完全漏报的 pulse。

### 4.7 Common factor

对真实 future 与 forecast 分别中心化并做 SVD：

- leading loading subspace 的 sign-invariant cosine；
- leading factor path 的 sign-invariant correlation；
- rank-1 singular strength 的幅度恢复；
- top-1 variance share。

所有通道预测为同一个常数时，中心化能量为零，不能获得共同动态因子分。

### 4.8 Hierarchical coherence

- `parent - sum(children)` 的归一化违反度；
- child zero-sum contrast 的动态路径；
- child heterogeneity RMS；
- 对应幅度恢复。

coherence 与 child heterogeneity 共同进入几何平均；全零预测虽然完全 coherent，
但 heterogeneity 为零，因此不能获得高机制分。

### 4.9 Covariate response

正式评价必须在同一 history 上保存配对请求：

1. intact known-future covariates；
2. 消融 known-future covariates。

两次 forecast 之差与真实 future covariate effect 比较方向、路径、幅度和增量解释度。
现有单次 intact forecast 可以输出 future-covariate projection 诊断，但必须标记为
`formal_score_eligible=false`，不能参与正式机制排名。后续小规模补推只针对
`gefcom2014_load`，不重跑其余能力。

Runner 通过 `--covariate-ablation-predictions-dir` 接收配对结果。目录按模型保存
JSONL；每行至少包含与 intact 请求相同的 `master_sample_id`、oracle
`context_length`、`forecast`，并声明
`ablation="future_covariates_zero"`。缺行、context 不一致或混用其他消融定义时直接
失败，不静默退回观察性分数。

## 5. I1–I5 剂量响应

同一 paired group 的五档共享结构与 nuisance realization。逐组比较真实和预测恢复的
机制强度向量，报告：

- Spearman 相关；
- min-max 标准化后的 Lin CCC；
- 四个相邻强度变化的方向一致率。

三项映射到 `[0,1]` 后做几何平均得到 `dose_response_score`。Capability 级机制分为：

\[
MFS_{capability}
=
0.7\,\overline{MFS}_{level}
+0.3\,\overline{MFS}_{dose}.
\]

权重在正式模型比较前冻结。Intensity 表示机制剂量，不要求 MASE 随档位单调变化；
剂量响应检查的是预测恢复的机制强度，而不是点误差。

## 6. 点误差安全门与排名

MASE、机制分与能力分均作为正式结果保留。能力分不把 MASE 与机制分任意线性相加，
而使用 capability-blind `naive` reference 作安全门：

\[
AbilityScore
=
MFS
\cdot
\min\left(1,\frac{MASE_{naive}}{MASE_{model}}\right).
\]

模型优于 naive 时不因点误差获得额外机制奖励；模型比 naive 更差时，机制分按比例
受罚。每个 capability 同时报告：

- MASE rank；
- mechanism rank；
- ability rank；
- tie-aware 配对差异与 bootstrap CI（正式全样本阶段）。

Covariate response 在配对消融完成前不产生 mechanism/ability 正式名次。

## 7. 指标验收与正式扩展条件

机制指标先通过受控预测验收，不根据 foundation model 排名反向调节：

1. oracle future 的所有可识别子分应接近 1；
2. constant、mechanism-blind、fixed-seasonal 等针对性退化预测应显著降低对应分数；
3. controlled timing shift、amplitude shrink 和 false events 应分别只影响预期子分；
4. I1–I5 正确顺序的 dose score 应高于逆序或随机顺序；
5. 分数在独立 N=160 seed bank 上按 E2 协议验证可靠性。

Pilot 只验证实现、方向与数据完整性，不据此宣称正式模型能力结论。完成 covariate
paired ablation、受控退化验收和独立 bank 可靠性后，才扩到正式九能力模型画像。

## 8. 产物

Runner：

`scripts/analyze_paper_v5_e3_mechanism_fidelity.py`

输出：

- `sample_mechanism_scores.csv`；
- `intensity_cell_scores.csv`；
- `paired_dose_response_scores.csv`；
- `capability_profiles.csv`；
- `summary.json`、`report.md` 与 `manifest.json`。
