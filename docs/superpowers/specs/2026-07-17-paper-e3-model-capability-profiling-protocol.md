# E3：Model capability profiling 分析协议

日期：2026-07-17

## 目的与边界

E3 在不重新生成样本、不重新调用模型的前提下，复用已经封存并通过动态稳定性检验的 E2 预测，形成各基础模型在九个 capability 上的能力画像。E3 回答的是：模型在不同结构、不同 intensity 和不同真实 conditioning profile 下的相对强弱是什么，以及这些结论对 bucket 是否稳健。

E3 是对既有 E2 结果的预先定义式二次分析，不声称在看到模型输出前预注册。E3 也不单独证明合成能力对真实数据的外部效度；合成—真实对应关系需要后续实验验证。`common_factor` 的跨通道利用和 `covariate_response` 的未来协变量利用仍分别需要 channel-independent/permutation 对照和 future-covariate 配对消融，不能只由本实验的端到端预测误差推出。

正式输出固定保存在 `runtime/paper_exp/v1/E3_model_capability_profiles/`。完成分析并写入 `manifest.json` 后目录封存，不允许原地覆盖。

## 冻结输入与模型集合

- 唯一数据源：`runtime/paper_exp/v1/E2_dynamic_stability/`。
- E2 manifest SHA-256：`5e91a4a4dadba842939754c8ad3e2efa22c8af3e247bf169c94ef0afbf27cfe0`。
- canonical scale：`synthetic-v2-paper-v1-frozen-2026-07-16`，fingerprint `a76b66924562be4f`。
- 23 个 `profile × capability` cells、5 档 intensity、5 个独立生成轮次、每轮每格 32 条样本，共 18,400 条样本。
- 基础模型：`Timer-3.5`、`Timer-3.0`、`Chronos-2`、`moirai2`、`toto2.0`、`timesfm2.5`、`tirex2`。
- `seasonal_naive` 只作为 skill reference；`naive` 不进入基础模型画像或排名。
- 不兼容任务记为 `N/A`，绝不记为失败或最差分数。六个单变量 capability 由全部七个模型比较；结构化 capability 继续按 E2 的模型兼容集合分开报告。

runner 必须先验证 E2 manifest 中所有文件的大小与 SHA-256，随后只读 E2 的 `samples.jsonl`、七个基础模型预测和 `seasonal_naive` 预测。E3 manifest 记录 E2 manifest、自身协议、runner 和全部输出的哈希。

## 分析单位与聚合权重

最小汇总单元为 `model × profile × capability × intensity`，包含五轮共 160 条预测。该单元的 MASE 是 160 个逐样本 seasonal MASE 的算术均值。

从单元向 capability 聚合时遵守以下固定顺序：

1. 五轮与每轮 32 条样本在单元内等权；
2. 同一 capability 的真实 conditioning profiles 等权，避免增加某个数据集的 bucket 数就改变能力权重；
3. 五档 intensity 等权，避免把 intensity 误解释为样本频率或主观难度分布。

所有宏平均均为 macro average，不按 horizon、target channel 数或底层时间点数加权。`hierarchical_coherence` 只有一个在线 profile，因此其跨 bucket 统计记为未定义。

## 主要与辅助指标

### 1. Five-level mean performance

主指标为 capability 内五档 MASE 的算术均值，越低越好：

```text
M_bar(m,c) = (1/5) Σ_i [(1/|B_c|) Σ_b MASE(m,b,c,i)]
```

模型在每个 capability 上的主排名由 `M_bar` 给出。六个单变量 capability 另做等权 macro average，作为单变量总览；结构化任务因接口和兼容模型集合不同，不合成一个全局总分。

### 2. Intensity-response AUC

令 `x_i=(i-1)/4`，对五档 profile-macro MASE 曲线做梯形积分：

```text
AUC(m,c) = trapz(MASE_i, x_i), x in [0,1]
```

AUC 越低越好。它描述整个强度路径上的误差暴露，不预设误差应随 intensity 单调上升。同步保留 intensity 1、intensity 5、线性斜率、端点相对变化以及五档 Spearman 方向，供解释模型更擅长强结构还是弱结构。

### 3. Worst-level performance

“最差档”定义为五档 profile-macro MASE 中实际最大的档，并同时记录该 intensity 与误差。intensity 是结构强度而非难度，因此不预先把 intensity 5 当成最差档。若完全并列，取较小 intensity 作为确定性 tie-break。

### 4. Relative seasonal-naive skill

在每个 `profile × intensity` 单元先计算：

```text
skill = 1 - mean(MASE_model) / mean(MASE_seasonal_naive)
```

随后对 profiles 和 intensities 等权平均。`skill > 0` 表示优于 seasonal naive，`skill = 0` 表示持平，`skill < 0` 表示更差。先做单元比值再宏平均，可避免 baseline 难度不同的 bucket 由绝对误差尺度支配；不对逐样本比值取平均，以免小分母产生病态值。

### 5. Cross-bucket variance

先为每个 bucket 计算五档平均 MASE，再在同一 `model × capability` 内报告样本方差（`ddof=1`）、标准差、CV 和极差。该量同时包含真实 conditioning profile 的基础难度差异和模型敏感性，只解释为跨基底稳健性，不解释为纯模型随机方差。两个 bucket 的结果只作描述，单 bucket 不计算。

### 6. Normalized MAE 与层级一致性

辅助指标明确采用 absolute-target normalized MAE：

```text
NMAE_abs = Σ |forecast - target| / Σ |target|
```

先在最小单元内对全部未来时间点和 target channels 池化分子、分母，再按 profiles 和 intensities 做 macro average。分母若不为正则直接报错，不做静默 epsilon 修补。

`hierarchical_coherence` 除 MASE/NMAE 外，必须报告：

```text
coherence_MAE = mean_t |forecast_parent - Σ forecast_children|
coherence_NMAE = Σ_t |forecast_parent - Σ forecast_children| / Σ_t |target_parent|
```

预测准确度与 prediction coherence 分开解释。

## 不确定性

默认做 2,000 次配对分层 bootstrap。每次先有放回抽取五个 round，再在每个 profile 和被抽到的 round 内有放回抽取 32 个 `sample_index`；同一次抽样在所有 intensities、所有兼容模型和 seasonal-naive 间共享，从而保留配对 seed 结构。

对 intensity 曲线、five-level mean MASE、AUC、worst-level MASE、relative skill、NMAE 和 hierarchy coherence 输出 percentile 95% CI。profiles 是固定的 paper-v1 conditioning 基底，不在 bootstrap 中重抽；跨 bucket 方差只作固定基底上的描述统计。

## 输出与图像

正式目录至少保留：

- `profile_intensity_scores.csv`：最小汇总单元及 seasonal-naive 对照；
- `intensity_curves.csv`：逐 capability、模型、intensity 的 profile-macro 曲线和 CI；
- `bucket_scores.csv`：逐 bucket 的五档汇总；
- `capability_profiles.csv`：论文主画像表、CI、排名和跨 bucket 统计；
- `model_summary.csv`：六个单变量 capability 的模型 macro 总览；
- `summary.json` 与 `report.md`：结果摘要、主要发现和边界；
- `figures/`：seasonal-naive skill heatmap、MASE intensity-response small multiples、单变量能力轮廓、跨 bucket 变异图和单变量总览；每张图保存 PNG、SVG、PDF。

图中缺失兼容项显示为 `N/A`。跨 capability 的颜色比较优先使用 relative skill；原始 MASE 只在 capability 内或独立纵轴中比较，避免把不同生成机制的天然误差尺度误读为能力强弱。

## 解释规则

- 论文主结论以 capability 级 MASE、95% CI 和 relative skill 联合判断，不只看单个 rank。
- intensity 曲线可以说明性能如何随结构显著性变化，但 intensity 高不等于任务更难，曲线下降也不构成异常。
- 跨 bucket CV 高表示结论依赖 conditioning profile；只有 2--3 个 bucket，不能把它当作总体方差的精确估计。
- 一个模型在不兼容结构化任务上的空缺是接口覆盖差异，不是能力为零。
- E3 可提出待真实数据验证的能力缺陷假设，但“该缺陷反映真实数据表现”的主张必须由后续合成—真实 concordance/case-study 实验支持。
