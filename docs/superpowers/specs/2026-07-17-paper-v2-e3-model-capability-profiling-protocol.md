# Paper v2 E3：六个单变量能力画像协议

## 目的

E3 只读复用封存的 E2-v2 样本与预测，形成七个基础模型在六个单变量能力上的连续能力画像。
本实验不重新生成样本、不重新调用模型，也不根据 E3 结果改变 E2 的门限或样本集合。

正式输入为：

- `runtime/paper_exp/v2/E2_dynamic_stability`
- E2 manifest SHA-256：
  `91b61c7d4b3d4cd81da28f011d6d6e0810db423d1c16b0bb336a6f17a2e1f34d`
- 9 个 held-out hourly profile；
- `context=504`、`horizon=48`、`season_length=24`、单目标；
- 6 个能力、5 个绝对 intensity、5 轮、每轮 16 条；
- Timer-3.5、Timer-3.0、Chronos-2、moirai2、toto2.0、timesfm2.5、tirex2；
- seasonal naive 作为相对 skill 基线。

## 主指标与聚合

MASE 是主指标，absolute-normalized MAE 是辅指标。先在
`model × profile × capability × intensity` 内对 5 轮 × 16 条样本等权平均，再对 9 个
profile 等权宏平均。不得让序列数较多或误差尺度较大的 profile 获得更高权重。

每个 `model × capability` 报告：

- 五档平均 MASE；
- intensity-response AUC，横轴固定为 `(intensity - 1) / 4`；
- 五档中实际最差的 MASE 及其 intensity，不假设 intensity 5 必然最难；
- 相对 seasonal-naive skill：先在每个 profile/intensity 内计算
  `1 - model MASE / seasonal-naive MASE`，再等权宏平均；
- absolute-normalized MAE；
- profile-level 五档 MASE 的方差、标准差、CV 与 range。

另外报告每个模型相对该 capability observed leader 的配对 MASE/skill gap 与 95% CI。
该对照共享同一 bootstrap draws，因此保留 seed 配对。由于 leader 由同一 E3 数据选出且
未做 multiplicity adjustment，这张表只用于生成 E4 假设，不作为独立显著性声明。
机器可读 CSV 保留全部配对结果；人工报告按 relative MASE gap 排序，仅展示 CI 完全高于
0 的前 10 项，避免把大样本下极小但非零的差异当作实质缺陷。

六能力全局 macro 对 capability 再做等权平均，仅用于同一预注册能力集合内的摘要。

## 不确定性

使用 2,000 次配对分层 bootstrap：

1. 以 round 为 cluster 重采样；
2. 在每个被抽中的 round 内重采样 sample index；
3. 同一次抽样在所有模型、intensity 与 seasonal-naive 间共享；
4. 9 个 profile 固定，不对 profile 重采样；
5. 报告 percentile 95% CI。

这样保留动态生成轮次的不确定性，并利用同 seed 配对降低模型比较噪声。

## 排名解释

E2-v2 的分数稳定性与模型 profile ICC 通过，但严格逐-cell 全排序 Kendall τ 未通过：
大量模型的相邻分差小于 1%，轻微分数波动会交换名次。因此：

- 连续 MASE、relative skill、intensity 曲线和配对 CI 是 E3 主证据；
- hard rank、top-1 次数与“最强/最弱能力”只作描述性导航；
- 不得通过事后放宽 E2 Kendall 门限把排名宣称为已验证稳定；
- E4 的真实数据验证应使用预注册的连续合成分数或能力 contrast，而非事后挑选单个名次翻转。

## 图表

正式输出五组图，每组保留 PNG、SVG、PDF：

1. capability × model 的 seasonal-naive-relative skill 热图；
2. 六能力的五档 intensity-response MASE 曲线与 95% CI；
3. 七模型的六维 relative-skill fingerprint；
4. capability × model 的跨 profile CV 热图；
5. 六能力 macro MASE 与 relative skill 的模型摘要。

## 解释边界

intensity 表示目标结构强度，不表示预测难度，曲线不要求随 intensity 单调变差。E3 只能提出
模型能力缺陷假设；合成能力画像能否预测真实数据缺陷，必须由 E4 在不读取真实模型结果的
前提下预注册数据集/窗口选择后验证。
