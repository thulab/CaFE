# Synthetic v2 Dataset-local 真实分布与生成校验链路

日期：2026-07-18

## 目标

本协议规定真实数据如何约束 Synthetic v2。核心约束是：

```text
一个 dataset 的真实窗口
  只校准这个 dataset 的 profile、五档 target、feature gate 和 distance gate
```

不得把多个 dataset 合成一个真实分布，也不得在一个 dataset 缺少支持时借用另一个
dataset 的分位数、变量结构或阈值。

## 记号

- \(d\)：dataset；
- \(v\)：task view；
- \(c\)：capability；
- \(L\)：lookback；
- \(H\)：prediction length；
- \(b=(d,v,L,H)\)：dataset-local profile；
- \(\psi_b(x)\)：只用 context 统计量完成的标准化；
- \(\phi_c(x)\)：capability 相关 realized features；
- \(f_c(x)\)：预注册 primary realized feature；
- \(g_c(x)\)：预注册非目标 control-feature vector。

Paper v4 使用 \(H=48\) 和
\(L\in\{96,168,336,504\}\)。四个 lookback view 来自同一个 \(L=504,H=48\)
母窗口并共享 future。

## 1. Dataset-local 真实分布

对固定 profile \(b\) 的真实窗口 \(R_b=\{x_i\}_{i=1}^N\)，定义：

\[
P^{b}_{raw}=\frac1N\sum_i\delta_{\psi_b(x_i)},\qquad
P^{b}_{feat}=\frac1N\sum_i\delta_{\phi(\psi_b(x_i))}.
\]

`P_raw` 用于近邻距离和批量分布报告，`P_feat` 用于 profile、强度与
control-feature support。所有统计量都保留 `dataset_id/profile_id` 身份。

## 2. 三路拆分

每个 \(b\) 独立拆成：

```text
R_param  -> nuisance + dataset-local intensity inverse calibration
R_ref    -> control-feature support + near-distance reference
R_cal    -> conformal threshold + real-to-real distance threshold
```

多序列数据按 series/group 隔离；单序列按连续时间块切分，并在相邻分区间设置至少
\(L+H\) embargo。官方 test tail 或预留 evaluation tail 在窗口化前排除。同一
dataset 的不同 task view 与不同 lookback 不互相充当独立样本。

## 3. Profile 与五档相对强度

profile 记录真实特征分位数、nuisance、任务结构、频率、维数和窗口数量。对
supported `b × c`：

\[
T_{b,c,k}=Q_{R^{param}_b}(f_c(x),p_k),\qquad
p=(0.10,0.30,0.50,0.70,0.90).
\]

五档仅在固定 \(b,c\) 内排序。不同 dataset 的 target 值和档号不具有绝对可比性。

固定 profile nuisance \(\theta_{b,c}\) 后，使用密集结构尺度和 \(\lambda\) 网格反解：

\[
\lambda_{b,c}(k)=
\arg\min_\lambda
\left|
\mathbb E_{seed}f_c(G_c(\theta_{b,c},\lambda))-T_{b,c,k}
\right|.
\]

独立 seed bank 验证 calibrated realized mean。发布条件包括：

- 五个 target 有限且严格递增；
- 最小相邻间距至少为 `max(1e-6, 0.02 * target_range)`；
- 验证 realized mean 单调；
- 最大 dataset-local normalized error 不超过 0.20。

条件不满足时记录 `unsupported`，不投影、不补值、不回退到其他 profile。

## 4. 九能力 Support Matrix

每个 dataset 固定输出九行能力审计，每行包含四个 lookback 的 `view_support`。
unsupported reason 至少覆盖：

| reason | 含义 |
| --- | --- |
| `missing_required_task_view` | dataset 没有该能力需要的 task view |
| `variable_structure_not_supported` | target/covariate/hierarchy 结构不满足 |
| `insufficient_windows` | 拆分和 embargo 后有效窗口不足 |
| `insufficient_local_target_range` | 主特征整体跨度不足 |
| `insufficient_local_intensity_spacing` | 五个分位档不能可靠区分 |
| `inverse_calibration_failed` | 合成响应不能稳定逼近本地 target |
| `feature_gate_calibration_failed` | control support 无法独立标定 |
| `near_distance_calibration_failed` | real-to-real 距离基线无法标定 |

只有 supported cells 进入生成 qualification 和模型实验。unsupported 不是失败结果，
也不影响同一 dataset 的其他能力。

## 5. 生成后校验

生成顺序固定为：

```text
select dataset/profile before generation
  -> construction predictability contract
  -> context-only normalization
  -> realized target feature audit
  -> same-profile control-feature gate
  -> same-profile full/context/feature near-distance gate
  -> accepted synthetic sample
```

不得生成后再选择最容易通过的 profile。

### 5.1 Construction predictability

决定 future 条件均值的规律必须已在 history 中重复出现，或由 known-future
covariates 显式提供。只在 future 采样的新 cut point、burst 或 shock 不能发布。
construction gate 是必要条件；它不替代 E1 中 capability-aware oracle 与 matched
baseline 的模型层验证。

### 5.2 Realized target dose-response

重新计算：

\[
\widehat f_{b,c,k}
=
\frac1M\sum_{m=1}^M f_c(\psi_b(x^{syn}_{m,k})).
\]

验收聚合均值的方向和对 `T_{b,c,k}` 的偏差，不要求每条随机样本逐点单调。intensity
不是预设难度，模型误差可以非单调。

### 5.3 Control-feature 联合支持域

目标特征不进入 gate。对 \(g_c(x)\)，使用 \(R_{ref}\) 的 median/IQR 和收缩协方差：

\[
z(x)=\frac{g_c(x)-median(R_{ref})}{IQR(R_{ref})},
\]

\[
d_c(x)=
\sqrt{\frac{(z(x)-\mu)^\top\Omega(z(x)-\mu)}{\dim(g_c)}}.
\]

阈值 \(\tau_c\) 仅由 \(R_{cal}\) 的 split-conformal quantile 得到：

\[
FeatureGate(x,c,b)=[d_c(x)\le\tau_c].
\]

不存在同 profile 校准时 fail closed，不用多个 dataset 的边界求并集，也不根据
synthetic acceptance rate 放宽阈值。

### 5.4 Near-distance gate

同一 profile 的 real reference 分成 real-train 与 real-holdout。real-holdout 到
real-train 的自然最近邻距离用于定阈值；候选合成样本只与 real-train 比较。

\[
DCR(x,R)=\min_{r\in R}d(x,r),
\qquad
NNDR(x,R)=\frac{D_1(x,R)}{\max(D_2(x,R),\epsilon)}.
\]

分别检查：

- 标准化后的完整 target window；
- 模型可见的 target context；
- robust realized-feature vector。

明确的距离对为：

```text
threshold: real-holdout <-> same-dataset real-train
online:    synthetic    <-> same-dataset real-train
```

copy、轻微 jitter、affine shift 和只替换 future 的攻击样本都应被定向测试捕获。

### 5.5 批量分布报告

MMD、sliced Wasserstein、特征分位覆盖率和 first-pass acceptance rate 作为批量诊断，
按 dataset/profile 单独报告。它们不用于把不同 dataset 混成一个“总体真实分布”。

## 6. 合成与真实预测的比较

E2 对每个 dataset 同时执行：

1. supported synthetic capability cells 上的模型预测；
2. 该 dataset 独立保留的真实 test windows 上的模型预测；
3. 模型排名分别在合成与真实侧计算；
4. 报告 Spearman/Kendall、top-k overlap、pairwise ordering agreement；
5. 以 dataset 为统计单位汇总一致性。

support matrix 中的 unsupported 能力不进入该 dataset 的平均能力排名。不同 dataset
覆盖能力不同，因此必须同时报告有效能力数和有效模型对数，不能把缺失能力补成最差
名次。

## 7. Artifact 与元数据

conditioning artifact 使用
`synthetic_v2_generator_conditioning_artifact.v4`，并声明：

```text
intensity_policy.policy_id = dataset-local-real-bounded-generator-feasible-v1
intensity_policy.relative_dose_levels = [0.00, 0.25, 0.50, 0.75, 1.00]
intensity_policy.real_tolerance = [q05, 1.2*q95]
```

每个 supported capability 至少记录：

```text
dataset_id
profile_id
capability_id
target_feature
target_relative_levels
target_values
intensity_lambdas
calibrated_realized_strengths
calibration
```

v4 reader 兼容字段 `percentile_levels/target_percentile_levels` 仍镜像同一组相对坐标，
但在该 policy 下不表示经验分位数。

每个 unsupported cell 至少记录 dataset/profile/capability、reason 和 detail。旧的全局
强度字段不会被读取；旧 schema fail closed。

## 8. 论文简短表述

本文对每个真实 dataset 独立构建 task-specific profile，并在其 parameter split 上以
主 realized feature 的 `[q05,1.2×q95]` 定义真实容忍区间。合成生成器在固定
dataset-local nuisance 下估计自身响应区间，并在二者交集内取五个等距相对档后反解
结构参数；变量结构、有效窗口、可行交集、档间距或校准不足的能力如实标记
unsupported。候选样本随后使用同一 dataset 的独立真实拆分完成
control-feature support 与 full/context/feature DCR-NNDR 防复制校验。实验只比较
dataset 内的相对强度和模型表现变化，并用同一 dataset 的真实预测排名检验合成能力
排名的外部一致性。
