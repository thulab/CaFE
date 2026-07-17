# Paper v2 E4：合成能力画像能否预测真实缺陷

日期：2026-07-17

## 1. 研究问题与冻结顺序

E4 检验一个确认性问题：在没有读取真实 test 模型结果的前提下，由动态合成数据得到的
模型能力画像，能否预测 held-out 真实数据上的相对表现与能力缺陷。

本实验只覆盖 paper-v2 已冻结的六个单变量能力与七个基础模型。执行顺序不可交换：

1. 读取 sealed E3-v2、E3-v1 和 train-only capability audit；
2. 冻结数据资格、真实窗口抽样、合成 predictor、配对假设和统计量；
3. 生成真实 task manifest，并把其 SHA-256 写入版本控制中的 selection receipt；
4. 提交 selection receipt 后，才允许请求推理服务；
5. 推理完成后只运行预先冻结的主分析；任何探索性案例必须另行标记。

E4 不读取 `/root/xmy/gift-eval-code/results/` 中已有的模型成绩，也不声称复现使用全历史的
GIFT-Eval leaderboard。它是固定 `context=504, horizon=48` 的 controlled
GIFT-Eval slice。

## 2. 数据全集与真实窗口

数据全集沿用
[`2026-07-17-paper-v2-synthetic-real-transfer-protocol.md`](./2026-07-17-paper-v2-synthetic-real-transfer-protocol.md)
中预先声明的九个 hourly held-out profiles：

- `solar/H`
- `kdd_cup_2018_with_missing/H`
- `LOOP_SEATTLE/H`
- `SZ_TAXI/H`
- `M_DENSE/H`
- `ett1/H`
- `ett2/H`
- `bitbrains_fast_storage/H`
- `bitbrains_rnd/H`

family cluster 固定为 `solar`、`kdd_cup_2018_with_missing`、`LOOP_SEATTLE`、
`SZ_TAXI`、`M_DENSE`、`ETT` 与 `bitbrains`；ETT1/2 属于同一 family，两个
Bitbrains 配置属于同一 family。

对每个 profile 严格复现冻结 GIFT short-term tail：

```text
prediction_length = 48
windows = clip(ceil(0.1 * min_series_length / 48), 1, 20)
origin_j = series_length - 48 * windows + 48 * j
```

每个 origin 只给模型最后 504 个 history points。native multivariate row 按 GIFT
`to_univariate=True` 语义拆成 channel；时间戳使用 Arrow 中真实 `start + index × freq`，
不替换为统一的合成日期。

### 2.1 缺失与可计分资格

- context 至少 50% observed 且至少两个有限点；
- context 仅在自身 504 点内部线性插值，并用最近 observed value 填两端；
- future 不插值，至少 24/48 个点 observed；
- seasonal MASE scale 在插值后的 context 上按 lag 24 计算；
- 为避免近常数历史产生数值爆炸，要求
  `scale > max(1e-8, 1e-6 × mean(abs(context)))`。

未来缺失点在 MAE、MSE、MASE 和 NMAE 中统一 mask。模型永远不接收 future target。

### 2.2 profile 等权的确定性抽样

每个 profile 最多选 600 个 task。先按 official rolling-origin index 分层并尽可能均分
600 个名额，再在每个 origin 内按 Arrow 原始 row/channel 顺序做确定性等距抽样。某层
容量不足时按 origin 顺序循环回填剩余名额。候选不超过 600 时全部保留。

该方案使 rolling origins 都被覆盖，同时避免 Loop、KDD 或 Bitbrains 因 channel 数多而
支配 E4。所有 headline 汇总仍按 profile/family 等权，而不是按 task 数加权。

## 3. Train-only capability 资格

每个 profile 的 capability loading 固定为
`00_transfer_protocol_freeze/capability_audit.json` 中仅用 training prefix 计算的
`median_canonical_intensity_coordinate`。确认性 high-loading 门限预先固定为五档中点：

```text
qualified(profile, capability) := loading >= 3.0
```

资格不读取真实 test feature 或模型结果。所有 qualified
`profile × capability` 单元进入主终点；不根据 E4 成绩删单元。

`nonlinear_persistence` 在九个 profile 上的 train-only coordinate 均为 1，因此本轮没有
真实 support，必须报告为“本数据切片不可检验”，不能把它解释成迁移失败。`M_DENSE`
没有任何 coordinate 达到 3，保留为真实低 loading 对照，但不进入 high-loading 主终点。

## 4. 冻结的 synthetic predictors

令 E3 中一个模型在 profile `p`、capability `c`、intensity `i` 的 MASE 为
`M(m,p,c,i)`，同 cell 的 seasonal-naive MASE 为 `B(p,c,i)`。为与真实尺度对齐，先定义：

```text
S_local(m,p,c) = mean_i log(M(m,p,c,i) / B(p,c,i))
```

四个预先冻结的 predictor 为：

1. `v2_dataset_local_capability`：使用 `S_local(m,p,c)`；
2. `v2_global_capability`：先对九个 v2 profiles 等权平均 `S_local`；
3. `v1_development_global_capability`：从 sealed E3-v1 development profiles 以同一公式
   得到 capability-global predictor；
4. `v2_scalar_macro`：忽略 capability，对六个 v2 global capability scores 等权平均。

第五个负对照 `v2_dataset_local_wrong_label` 不单独选一组结果：固定 qualified cells，
枚举六个 capability 标签的全部非 identity 全局置换，将 cell 的 predictor 换成被置换
标签下的 dataset-local score，构成 exact label-permutation null。

## 5. 真实分数与确认性主终点

每个真实 task 先计算 masked seasonal MASE。profile-level model MASE 是该 profile 所有
selected tasks 的算术均值。真实相对效应为：

```text
R(m,p) = log(real_MASE(m,p) / real_MASE(seasonal_naive,p))
```

越低越好。NMAE 定义为 profile 内 `sum(abs(error)) / sum(abs(observed target))`，仅作辅助。

对每个 qualified `profile × capability` cell，在七个模型上比较 synthetic predictor
与 `R(m,p)`：

- Kendall tau-b（预注册主统计量）；
- Spearman rho；
- 21 个模型 pair 的方向一致率；
- cell 内 z-score 后的 Pearson correlation 与 RMSE（连续分数诊断）。

宏平均顺序固定为：

1. 一个 profile 内对其 qualified capabilities 等权；
2. 一个 family 内对 profiles 等权；
3. 对进入 high-loading 主终点的 families 等权。

主终点为 `v2_dataset_local_capability` 的 family-macro Kendall tau-b。支持多维能力迁移需
同时报告：

- 主终点是否为正及 family-cluster bootstrap 95% CI；
- 相对 `v2_scalar_macro` 的配对 family-macro tau 增量及 95% CI；
- exact wrong-label permutation p-value；
- 相对 v2 global 和 v1 development global predictors 的差异。

不因观察值调整显著性阈值。`alpha=0.05`，CI 为 percentile 95% CI。

## 6. 不确定性与稳健性

- 主 CI：以 family 为 cluster 有放回抽样 5000 次，predictors 与 cell 资格固定；
- leave-one-family-out：逐一移除 high-loading family，报告主统计量范围；
- capability-level：按同样的 profile→family 宏平均报告 tau/rho/方向一致率，但因 family
  数较少作为分解结果；
- task 数和 observed-future coverage 必须逐 profile 报告；
- hard rank 仅是主协议要求的跨模型对应性统计；结合 E2 的 near-tie 发现，同时报告连续
  score association，不把单个名次交换解释成能力断言。

## 7. E3 生成的配对缺陷假设

配对假设只由 sealed E3-v2 产生：对每个具有至少三个 high-loading family clusters 的
capability，选择相对该 capability observed leader 的 `relative_mase_gap` 最大、且 E3
paired 95% CI 下界大于 0 的前两个非 leader 模型。该规则在真实推理前执行并把确切列表
写入 selection manifest。

对每个假设 `(weaker, reference, capability)`：

- 主要方向为 high-loading families 上
  `mean_family log(real_MASE_weaker / real_MASE_reference) > 0`；
- 报告 family-cluster bootstrap 95% CI；
- 报告所有七个 family 上 train-only loading 与真实 pair gap 的 Spearman rho，作为
  dose association 诊断；
- 汇总假设方向命中率；个别 pair 不作未经校正的“发现”宣称。

E3 排名选择本身不是 multiplicity-adjusted，因此配对结果是主 Kendall 终点的可解释性
分解，而不是替代主终点的独立确证。

## 8. 预注册图表与产物

E4 目录固定为 `runtime/paper_exp/v2/E4_synthetic_real_transfer/`，保留：

- `tasks.jsonl`、`task_manifest.json`、`selection_manifest.json`；
- `synthetic_predictors.csv`、`qualified_cells.csv`、`pair_hypotheses.csv`；
- 每个模型的逐 task prediction JSONL 与失败日志；
- `real_profile_scores.csv`、`cell_concordance.csv`、`predictor_summary.csv`；
- `bootstrap_summary.csv`、`leave_one_family_out.csv`、`pair_hypothesis_results.csv`；
- `summary.json`、`report.md`、`paper_tables.md`、最终 manifest；
- PNG、SVG、PDF 三种格式的真实性能热图、predictor 对比、合成—真实对应图、配对缺陷
  forest plot 与 leave-one-family-out 图。

任何看过真实模型结果后新增的案例图或 subgroup 必须放入 `exploratory/` 并明确标注，
不得回写上述确认性规则。
