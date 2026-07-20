# Paper v6 E3：正式机制保真能力画像协议

日期：2026-07-20

## 1. 研究问题

E3 在 Paper v5 pilot 的九能力机制评分基础上，正式回答：

1. 模型的点预测误差如何；
2. 模型输出是否恢复了目标能力机制，而非只靠平滑或均值回归降低误差；
3. 机制保真结论在点误差安全门后是否仍成立；
4. 使用两个互斥的 80-group 子样本时，上述分数、模型对关系与排名是否可靠。

机制评价可以读取 synthetic future、构造 metadata 和 latent schedule，但这些信息只供
evaluator 使用，绝不能进入模型请求。机制分只表示
`mechanism-aligned forecast behavior`，不声称模型内部识别了因果机制。

## 2. 冻结输入

普通 E3 只读复用：

- `runtime/paper_exp/v6/E2_dynamic_stability/sample_shards/`；
- `runtime/paper_exp/v6/E2_dynamic_stability/predictions/`；
- `runtime/paper_exp/v6/E2_dynamic_stability/oracle_sample_scores/`。

同一个模型和 master sample 继续使用 E2 按最小 MASE 选出的 context。MASE、机制分与
能力分必须使用完全相同的 forecast，不能分别挑选 context。

正式 E3 每个 `dataset × capability` 使用全部 160 个 paired groups，每组保留
I1–I5 五档。前 80 与后 80 只用于可靠性审计；全部 160 组的估计量才是正式主结果。

## 3. 能力与模型兼容范围

| 结构 | 数据集 | 能力 | 兼容模型 |
|---|---|---|---|
| 单变量 | `gift_ett1_h` | 六项单变量能力 | 全部八模型 |
| 普通多变量面板 | `electricity_hourly_panel` | `common_factor` | Chronos-2、toto2.0、tirex2、tabpfn-ts3 |
| 显式加和层级 | `m5_daily_hierarchy` | `hierarchical_coherence` | Chronos-2、toto2.0、tirex2、tabpfn-ts3 |
| 已知未来协变量 | `gefcom2014_load` | `covariate_response` | Chronos-2、timesfm2.5、tirex2、tabpfn-ts3 |

Unsupported 组合必须记录为 N/A，不生成伪预测、不记零分，也不补成最差名次。每个
能力只在共同兼容该能力的模型集合中排名。

## 4. 机制与能力分

沿用 Paper v5 冻结的九项 evaluator。逐样本机制分仍为 detection、timing、
magnitude、selectivity 四项的等权几何平均。

同一 paired group 的五档 dose response 由以下三项的几何平均构成：

- 真实与预测机制强度的 Spearman；
- min-max 标准化后的 Lin CCC；
- 相邻档位变化方向准确率。

Capability 级机制分：

\[
MFS = 0.7\,\overline{MFS}_{level}
    + 0.3\,\overline{MFS}_{dose}.
\]

能力分：

\[
AbilityScore
=
MFS
\cdot
\min\left(1,\frac{MASE_{naive}}{MASE_{model}}\right).
\]

正式结果并列保留 MASE、MFS、Ability；Ability 是带点误差安全门的综合能力结果，
但不能替代前两项诊断。

## 5. Covariate paired ablation

`covariate_response` 必须补充同 history、同 oracle context 的配对请求：

1. intact known-future covariates：复用 E2；
2. zero known-future covariates：只把标准化后的 future covariates 置零。

History targets、history covariates、context、horizon 和模型配置保持不变。标准化连续
协变量的零值对应 history mean；二元 event 的零值对应无事件。

补推范围固定为 4 个兼容模型 × 160 paired groups × 5 intensities，共 3,200 个
请求。消融预测保存到：

`runtime/paper_exp/v6/E3_mechanism_fidelity/covariate_ablation_predictions/`

每行必须保存 `master_sample_id`、oracle `context_length`、forecast 和
`ablation=future_covariates_zero`。缺行或 context 不一致时正式评分直接失败。

## 6. Bootstrap、等价与排名

默认做 2,000 次 paired bootstrap，95% percentile CI。抽样单位为
`paired_group_id`，每次抽样：

- I1–I5 五档一起进入；
- 所有兼容模型共享同一 draws；
- 重新计算 level MFS、dose response、MFS、MASE gate 与 Ability。

每个模型对分为四种状态：

- `left_better`；
- `right_better`；
- `equivalent`；
- `unresolved`。

只有 CI 完全落在等价区间内才算 equivalent；CI 重叠不自动代表并列。

MASE 使用“正值表示左模型更好”的对称相对差：

\[
\Delta_{MASE}
=
\frac{2(MASE_R-MASE_L)}
{|MASE_L|+|MASE_R|}.
\]

MFS 与 Ability 使用绝对差 `score_L-score_R`。主等价阈值为 0.02；同时报告
0.01/0.05 敏感性。因此 MASE 的 0.02 表示 2% 相对差，MFS/Ability 的 0.02 表示
2 个百分点。

每个能力的 point rank 仍作为直观正式结果保留；paired-bootstrap 四状态模型对是对
“很接近的名次是否可区分”的正式推断。硬 tie tier 与保守 rank interval 不作为主
结果。

## 7. 80/80 可靠性审计

按 `round_index → sample_index → paired_group_id` 确定性排序：

- first bank：前 80 组；
- second bank：后 80 组。

两个 bank 分别完整重算三套分数、point rank、bootstrap CI 与模型对状态。报告：

- rank Spearman；
- point pair 方向一致率；
- top-model 一致率；
- tie-aware pair state 完全一致率；
- 无明确方向冲突率及冲突数。

可靠性审计不替代全部 160 组的正式估计，也不据此挑选有利阈值。

## 8. 指标验收

正式运行前，机制评分器必须通过：

- 九项能力 oracle 与 constant 对照；
- multi-seasonal primary-only；
- time-varying fixed carrier；
- regime smoothing 与 timing shift；
- amplitude shrink；
- intermittency missing/false event；
- nonlinear blind forecast；
- common-factor channel desynchronization；
- hierarchy parent violation；
- covariate paired no-response；
- I1–I5 正序/逆序 dose response。

验收测试只使用生成器 oracle 与人为退化预测，不根据 foundation model 成绩调指标。

## 9. 产物

正式 runner：

`scripts/analyze_paper_v5_e3_mechanism_fidelity.py`

文件名为兼容既有 pilot 保留，输出 schema 已升级为
`paper_v6_e3_mechanism_fidelity.v2`。

协变量补推 runner：

`scripts/run_paper_v6_e3_covariate_ablation.py`

正式输出除逐样本、intensity cell、dose 与 capability profiles 外，还包括：

- `model_capability_coverage.csv`；
- `profile_group_components.csv`；
- `capability_bootstrap_intervals.csv`；
- `capability_pair_states.csv`；
- `split_half_capability_profiles.csv`；
- `split_half_bootstrap_intervals.csv`；
- `split_half_pair_states.csv`；
- `split_half_pair_state_comparison.csv`；
- `split_half_reliability.csv` 与 `split_half_summary.json`；
- `summary.json`、`report.md`、`manifest.json`。
