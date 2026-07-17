# Paper v2：六个单变量能力的合成—真实迁移协议

日期：2026-07-17

## 目的与版本边界

本协议验证动态合成能力画像能否预测 held-out 真实数据上的模型相对缺陷。实验仅覆盖
`trend`、`multi_seasonal`、`time_varying_seasonality`、`regime_switching`、
`nonlinear_persistence` 和 `predictable_intermittency` 六个单变量能力。

`paper_exp/v2` 表示论文实验协议第二版，不改变 synthetic-v2 的六个能力生成公式。
paper-v1 的 `168/24` raw feature 标尺具有窗口长度依赖性，不可在延长 context 后逐字复用。
因此 v2 在最终 `context=504, horizon=48, period=24` shape 上只重冻结一次 canonical
intensity；paper-v1 scale 仅作为方法开发来源记录，不与 v2 结果混合。

实验依赖顺序固定为：

```text
00_transfer_protocol_freeze
  -> E2_dynamic_stability
  -> E3_model_capability_profiles
  -> E4_synthetic_real_transfer
```

E2-v2/E3-v2 不得读取真实 test targets。E4 只能读取已经由 manifest 封存的 E3-v2
输出，不能重新生成 synthetic probes 或修改选择规则。

## 数据全集与固定窗口

确认性候选全集是在 paper-v1 canonical freeze 中已声明 held-out、且具有 GIFT-Eval
hourly short-term 配置的数据。不得根据 E2/E3/E4 模型成绩增删 profile。

### Hourly：context 504 / horizon 48 / period 24

- `solar/H`
- `kdd_cup_2018_with_missing/H`
- `LOOP_SEATTLE/H`
- `SZ_TAXI/H`
- `M_DENSE/H`
- `ett1/H`
- `ett2/H`
- `bitbrains_fast_storage/H`
- `bitbrains_rnd/H`

ETT 与 Bitbrains 的 native multivariate rows 在本实验中按 GIFT-Eval 官方
`to_univariate=True` 语义拆成单变量 channels。汇总推断按 base family 聚类；
ETT1/2 合并为 `ETT` cluster，两套 Bitbrains 合并为
`bitbrains` cluster。

Jena Weather、BizITObs-L2C、Hospital、M4、Electricity 和 Traffic 曾参与
canonical calibration 或 E2-v1 development，不进入 v2 held-out headline。

Daily 配置不进入本轮确认性主实验。多个 GIFT daily 配置的全部 training prefix
只有 152–306 个点，无法支持与 hourly 同量级的长窗口；同时 `period=7` 会让
`spike_rate`、change-point energy 和单-bin谱分数产生采样分辨率漂移。在冻结
frequency-normalized estimator 前混合 H/D，会把频率效应误归因为模型能力。Daily
只能作为预先独立定标的后续扩展，不能在看到本轮模型结果后补入 headline。

## 严格的 train/test 隔离

对每个 GIFT 配置，先复现冻结代码中的 short-term 协议：

```text
prediction_length = 48
windows = clip(ceil(0.1 * min_series_length / prediction_length), 1, 20)
```

Transfer conditioning 的时间截止点为：

```text
series_length - prediction_length * (windows + 1)
```

即同时删除全部 official test tail 和紧邻 test 的一个 validation horizon。随后只在
该 training prefix 内构造 pseudo-forecast windows，并做 parameter、gate-reference、
gate-calibration 三路 leakage-safe split。任何 test observation、test feature 或模型
test score 都不得进入 nuisance、intensity inverse map、feature gate、near-distance
gate 或 capability audit。

缺失值窗口必须至少有 50% observed points 且至少两个有限值。仅在已观察的 training
window 内线性插值，并用最近 observed value 填充两端；不得从 validation/test 区间取值。
E4 真实输入采用同一 history-only 规则，future labels 不插值而只在指标中 mask。

## 00：最终 shape、canonical scale 与 artifacts 冻结

`00_transfer_protocol_freeze` 产生：

- `capability_audit.json`
- `generator_conditioning_artifact.json`
- `feature_gate_artifact.json`
- `near_distance_artifact.json`
- `preflight.json`
- `manifest.json`

Canonical scale 由五个不进入 E4 的 development families 在相同 `504/48/24` shape
上冻结：M4 Hourly、Electricity、Traffic、Jena Weather/H 与 BizITObs-L2C/H。每个能力
先取五个 family 的等权真实分位曲线，再做最小档间距投影。`regime_switching` 保留真实
分位的相对坐标，但映射到预注册的 recurring-clock 可观察区间 `[0.56, 0.94]`；普通
一次性 change point 低于该构造下限，不满足“历史重复且未来继续”的可预测性契约。

`nonlinear_persistence` 使用 `nonlinear_conditional_gain`：线性基线先包含 lag-1、
seasonal lag 和同一个 raw nonlinear lag，增强模型只额外加入
`sin²(1.1 × lag)`。这样强季节性不能再被误记为非线性增益。v2 递推采用同一有界变换，
最大 Lipschitz 稳定界为 0.975；真实分位坐标映射到 `[0.002, 0.025]` 的构造支持区间，
使五档高于有限样本估计器底噪且仍只解释至多 2.5% 的增量方差。该能力每个 calibration
grid cell 使用 32 条 fit 样本，其余能力使用 16 条；所有能力均使用至少 256 条独立
validation 样本，误差阈值仍为 0.20。

Generator conditioning 只改变 profile nuisance 与到冻结 canonical target 的单调逆映射。
Feature-support 和 near-distance gates 必须按 profile、context、horizon、target dimension
精确匹配；缺少任一 artifact 时 fail closed。

Feature-support 只能使用不与目标机制机械耦合的 controls。nonlinear recurrence 会改变
seasonal/residual 方差占比，因此控制 trend、outlier 与 spike，而不控制
seasonal-strength/noise-ratio。可预测稀疏脉冲本身会被季节分解器恢复为季节信号，其
相位与周期也会被 trend smoother 投影为趋势，因此当前 realized feature family 中同样
没有独立 observable control。背景趋势 latent nuisance 仍由真实 training median 做
capability-specific inverse mapping；Intermittency 仍必须同时通过固定脉冲时钟的
construction contract、绝对 spike-dose 校准和 near-distance gate。

`regime_switching` 的 level switch 也会机械改变 spike、diff-spike、outlier 以及分解后的
trend/noise 摘要；这两个能力的 feature-support artifact 都显式记录
`not_applicable_no_independent_observable_controls`，不拟合伪独立的联合门限。该能力仍
必须通过 train-only 真实参数支持、历史重复且预测期继续的 construction contract、冻结
canonical dose、目标特征 dose-response 和 near-distance gate；这不是取消真实性约束。

Canonical strength 的估计窗口固定为 `context + 一个主周期 = 504 + 24 = 528`；
它同时是 24 与 48 的整数倍，避免谱泄漏随窗口端点改变 intensity。完整预测轨迹仍为
48 步，feature-support、near-distance 和模型指标也都检查完整轨迹。

Capability audit 将真实 primary feature 映射到连续 canonical intensity coordinate。
由于真实序列可以同时包含多个模式，audit 输出六维 multi-label loading，不要求能力纯度。
`regime_switching` 额外执行 history-selected recurring-clock qualification，普通一次性
change point 不得被解释为 predictable regime。

E2-v2 不按 audit 结果删 cell，因此 audit 本身不可能因 synthetic 模型表现而选择数据。
E4 若使用 high-loading case，具体资格门限和 case identity 必须在读取真实模型结果前写入
独立 selection manifest。

## E2-v2：新 shape/profile 的动态稳定性

每个 `profile × capability × intensity` 使用：

- 5 个独立 generation rounds；
- 每轮 16 条样本；
- 每格共 80 条；
- 五档 intensity；
- 模型与 intensity 使用 paired seeds。

五轮予以保留以估计 round CV、ICC 和排名稳定性；相较 E2-v1，仅将每轮 32 条降为 16 条。
9 个 profiles 与六能力的最大规模为 21,600 条 synthetic samples。

E2-v2 报告与 v1 相同的 score CV、model-profile ICC、round-wise Kendall tau、
bootstrap CI 和 cross-round nearest-neighbor distance。稳定性阈值沿用 v1，不因看到
v2 结果而放宽。

## E3-v2：冻结 synthetic predictor

E3-v2 只读取 E2-v2 sealed manifest，输出：

1. `profile × capability × intensity × model` 局部画像；
2. 每个 held-out profile 的 dataset-local capability profile；
3. 对 profiles/families 等权的 global transfer profile。

主要指标为 seasonal MASE，辅助指标为 absolute-target NMAE 和相对 seasonal-naive
skill。Dataset-local 结果保留五档曲线、five-level mean、AUC、最差档和 bootstrap CI。
E3-v1 的 development profile 作为不使用 held-out conditioning 的 global baseline，
不与 E3-v2 样本合并拟合。

## E4：真实缺陷迁移

真实评测使用 GIFT 官方 rolling origins 与 prediction horizon，但对所有模型固定共同
context。因此结果称为 controlled GIFT-Eval slice，不声称复现使用全历史的 leaderboard
分数。

跨数据集的模型效应优先表示为：

```text
log(MASE_model / MASE_seasonal_naive)
```

确认性主终点是：在预先 qualified 的 `dataset × capability` 单元上，
dataset-conditioned synthetic 模型排名与真实 high-loading windows 模型排名的
family-macro Kendall tau-b。同步报告 Spearman、模型 pairwise direction concordance、
family-cluster bootstrap 95% CI、leave-one-family-out 和 capability-label permutation
负对照。

E4 还比较：

- E3-v2 dataset-local capability predictor；
- E3-v2 global profile；
- E3-v1 development global profile；
- 忽略 capability、只使用单变量 synthetic macro rank 的标量基线。

只有 dataset-local 或 capability-aware predictor 相对标量/错标签基线获得稳定增益，
才支持“多维能力画像提供额外真实缺陷预测信息”。

## Structured capabilities

`common_factor`、`hierarchical_coherence` 和 `covariate_response` 不进入本轮 v2。
它们需分别完成 channel-independent/misalignment、joint/independent/reconciliation、
future-covariate mask/shuffle 等配对控制后，另设 E4-S，不与七模型单变量主统计合并。
