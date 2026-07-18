# CapTS-Bench Dataset-local Synthetic v2 方法

更新日期：2026-07-18

本文描述当前 `capts-paper-v2` 生成器和 Paper v4 实验协议。当前原则只有一个：

> 每个真实 dataset 独立定义 profile、五档相对强度、feature-support gate 和
> near-distance gate；不同 dataset 的真实窗口和统计量不合并。

旧的跨数据集 pooled profile、capability-global canonical intensity、dataset
family 权重和 held-out-family 标尺已废止。旧 v3 conditioning artifact 会被当前
loader 拒绝，不能静默回退。

代码与机器可读事实以以下内容为准：

- 生成公式和在线验收：`backend/app/services/synthetic_generation_service.py`
- v4 conditioning loader：`backend/app/services/synthetic_generator_conditioning.py`
- dataset profile：`scripts/build_paper_v4_profile_suite.py`
- 九能力校准与验收：`scripts/build_paper_v4_nine_capability_suite.py`
- runtime conditioning 构建：
  `scripts/build_synthetic_v2_generator_conditioning_artifact.py`

## 1. 真实数据的实际作用

真实数据不是合成曲线的素材池，也不提供一个宽泛的全局容忍范围。对 dataset \(d\)
和任务视图 \(v\)，它只校准该 dataset 自己的生成与验收条件：

```text
dataset d 的 development windows
  ├─ parameter split
  │    ├─ nuisance / background dynamics
  │    └─ capability c 的 q10/q30/q50/q70/q90 target
  ├─ gate-reference split
  │    ├─ control-feature support
  │    └─ near-distance reference bank
  └─ gate-calibration split
       ├─ feature-support conformal threshold
       └─ real-to-real DCR / NNDR threshold
```

因此，真实数据回答的是：

1. 在这个 dataset 上，某能力的“相对弱到相对强”分别是什么；
2. 哪些 nuisance 和非目标特征组合仍像这个 dataset；
3. 合成样本离真实 reference 多近才异常；
4. 这个 dataset 的变量结构和有效窗口是否足以支持该能力。

这些约束相对于纯合成的意义是：生成器不能只在自己定义的参数空间里自洽，而必须在
一个具体真实 dataset 的局部统计结构中完成干预、校准和防复制验收。

## 2. Dataset、task view 与 profile

dataset 是最上层独立单位。一个 dataset 可以提供多个 task view，例如：

- 单变量 target；
- 多目标 panel；
- parent/children 层级；
- target + known-future covariates。

不同 task view 可共享原始资产身份，但不共享 profile 统计量。每个 profile 至少绑定：

\[
(dataset\_id,task\_id,L,H,target\_dim,covariate\_dim,frequency).
\]

Paper v4 固定：

- \(H=48\)；
- \(L\in\{96,168,336,504\}\)；
- 同一母窗口的四个 lookback 使用相同 future；
- 每个 lookback 按自己的 context 重新标准化和验收。

不存在跨 dataset 汇总后的生成 profile。后续生成必须显式命中自己的
`dataset_id/profile_id`；某 dataset 的 q90、nuisance 或 gate 阈值不能借给另一个
dataset。

## 3. 九个能力与结构要求

| capability | 所需 task view | primary realized feature |
| --- | --- | --- |
| `trend` | 单变量 | `trend_strength` |
| `multi_seasonal` | 单变量 | `multi_period_score` |
| `time_varying_seasonality` | 单变量 | `seasonal_amplitude_modulation` |
| `regime_switching` | 单变量，历史中有重复状态时钟 | `regime_clock_history_incremental_r2` |
| `nonlinear_persistence` | 单变量 | `nonlinear_conditional_gain` |
| `predictable_intermittency` | 单变量 | `spike_rate` |
| `common_factor` | 多目标 panel | `pca_top1_explained` |
| `hierarchical_coherence` | parent + children | `hierarchy_child_heterogeneity` |
| `covariate_response` | target + known-future covariates | `covariate_incremental_r2` |

support matrix 对每个 dataset 固定列出九个能力，并审计四个 lookback。不是所有
dataset 都必须支持九个能力：

- 缂少所需 task view；
- 变量结构不满足要求；
- 有效、隔离后的窗口数量不足；
- dataset-local 五个目标没有足够间距；
- 逆校准或 gate 校准失败；

以上情况均写成 `status=unsupported`、稳定的 reason code 和审计细节。它不是失败样本，
不参与该能力的模型实验，也不阻断同一 dataset 的其他 supported cells。

## 4. Dataset-local 五档相对强度

对固定的 \(d,v,c,L,H\)，只在该 profile 的 parameter split 上计算主特征
\(f_c(x)\)：

\[
T_{d,v,c,k}=Q_{R^{param}_{d,v}}\left(f_c(x),p_k\right),\qquad
(p_1,\ldots,p_5)=(0.10,0.30,0.50,0.70,0.90).
\]

`intensity=1..5` 仅表示这个 dataset/profile/capability 内部从相对弱到相对强。
不能比较 Electricity 的 intensity 3 与 Traffic 的 intensity 3 的绝对强度；也不能
把不同能力的档号解释为相同物理量或相同难度。

生成器固定该 profile 的 nuisance，并反解结构参数：

\[
\lambda_{d,v,c}(k)=
\arg\min_\lambda
\left|
\mathbb E_{seed} f_c(G_c(\theta_{d,v,c},\lambda))-T_{d,v,c,k}
\right|.
\]

五档 target 必须有限、严格递增，并具有预注册的最小可分辨间距。当前实现要求：

\[
\min_k(T_{k+1}-T_k)
\ge
\max(10^{-6},0.02(T_5-T_1)).
\]

若不满足，记录 `insufficient_local_target_range` 或
`insufficient_local_intensity_spacing`，不人为拉开分位点，也不借用其他 dataset 的
目标。

独立 seed bank 验证 calibrated realized mean 的单调性和误差。最大误差按该
dataset-local target range 归一化，容差为 0.20。校准失败的 cell 同样记为
`unsupported`。

## 5. 生成公式与可预测性边界

最新生成公式保留 shortcut-resistant 修订：

- 除季节能力外，不给所有任务叠加同一个强固定季节载波；
- `regime_switching` 的状态规律在 history 中重复出现并延续至 future；
- `predictable_intermittency` 使用历史可识别的非等间隔 motif；
- `nonlinear_persistence` 经过稳定 burn-in 后才发布轨迹；
- `covariate_response` 的预测期信号由 known-future covariates 提供；
- 层级任务始终保持 parent 等于 children 之和；
- 禁止只在 future 中注入无先兆 cut point、burst 或 shock。

生成完整 \(L+H\) 曲线后，只用 context 统计量标准化：

\[
\tilde y_{t,j}=
\frac{y_{t,j}-\mu_j(y_{1:L,j})}
{\max(\sigma_j(y_{1:L,j}),\epsilon)}.
\]

层级任务逐通道中心化但共享尺度，避免破坏加总恒等式。future target 从不参与
标准化参数估计。

construction predictability contract 是配置级必要条件。它证明预测所需信息在
预测时可用，但不证明任意模型都能利用该信息；模型有效性仍由 E1 的 matched
baseline/oracle 和正式模型实验检验。

## 6. 三路拆分与防泄漏

每个 dataset/task view 内部独立拆成：

- `parameter`：只拟合 profile nuisance 和强度逆映射；
- `gate_reference`：只拟合 control-feature support 和近邻 reference；
- `gate_calibration`：只确定 real-only conformal / distance threshold。

多序列数据按 series/group 隔离。单长序列按时间阻塞切分，并在分区边界设置至少
\(L+H\) 的 embargo。官方 test tail 或预留 evaluation tail 在窗口化前排除。

同一 dataset 的不同 task view、不同 lookback 也不得互相充当独立 calibration 样本。

## 7. Feature-support 与 Near-distance gate

### 7.1 Feature-support

目标特征是实验干预对象，不进入真实支持域硬门控。对预注册的非目标 control vector
\(g_c(x)\)，使用 reference split 的 median/IQR 和收缩协方差计算联合距离，再用
calibration split 的 split-conformal 分位数得到阈值：

\[
d_c(x)=
\sqrt{\frac{(z(x)-\mu)^\top\Omega(z(x)-\mu)}{\dim g_c}},
\qquad
Accept_{feat}(x)=[d_c(x)\le\tau_c].
\]

阈值只对同一个 dataset/profile 生效。没有独立 control feature 的能力不伪造控制量，
但仍需通过 construction、强度校准和 near-distance 验收。

### 7.2 Near-distance

合成候选只与同一个 dataset/profile 的 real-train reference 比较。real-holdout 到
real-train 的自然最近邻距离用于标定阈值：

\[
DCR(x,R)=\min_{r\in R}d(x,r),\qquad
NNDR(x,R)=\frac{D_1(x,R)}{\max(D_2(x,R),\epsilon)}.
\]

分别检查完整窗口、模型可见 context 和 robust feature space。这里计算的是：

```text
synthetic candidate  <-> same-dataset real-train reference
real holdout         <-> same-dataset real-train reference（只用于定阈值）
```

它防止合成曲线贴近或复制已提交 reference，但不证明未知预训练语料中不存在相似模式。

## 8. 实验结论的统计单位

E1 在每个 dataset 的 `L=504,H=48` master task 上按
`dataset × capability × intensity` 检查：

- realized target dose-response；
- 非目标特征 selectivity；
- construction/oracle 对 matched baseline 的增益；
- feature-support 与 near-distance 通过情况。

四个 `L=96/168/336/504` suffix views 的共同 future 和逐 view gate 合格性由上游
nine-capability suite qualification 冻结；E1 不把四个相关 view 当成四份独立样本。

E2 同时在合成与真实测试窗口上运行相同模型。合理的主结论不是比较跨 dataset 的
绝对 intensity，而是：

1. 在每个 dataset 内得到 supported capabilities 的模型相对排名；
2. 将合成排名与该 dataset 的真实预测排名做 Spearman/Kendall、top-k overlap 和
   pairwise ordering agreement；
3. 再以 dataset 为统计单位报告这些一致性的分布或均值；
4. support matrix 中的 unsupported cells 原样披露，不补值、不算作最差名次。

这让真实数据承担外部效度检验：验证合成压力测试揭示的模型相对优劣，是否能对应到
同一个真实 dataset 上的预测表现。

## 9. Artifact 契约

conditioning artifact 当前 schema 为：

```text
synthetic_v2_generator_conditioning_artifact.v4
```

顶层必须声明：

```text
policy_id = dataset-local-relative-quantiles-v1
percentile_levels = [0.10, 0.30, 0.50, 0.70, 0.90]
```

每个 profile 必须记录 `dataset_id`。每个 supported capability 记录
`target_feature`、`target_values`、`intensity_lambdas`、
`calibrated_realized_strengths` 和 calibration 审计；unsupported capability 记录
reason 与 detail，且不能被在线选择。

扩展一个新 dataset 时，必须为它单独完成 profile、五档、feature-support 和
near-distance 校准。不能通过放宽已有 dataset 的阈值或重新引入跨数据集合并来获得
支持。
