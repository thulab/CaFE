# CaFE paper-results bundle

本目录是 2026-09-02 对四组冻结实验的只读汇总。论文主稿入口为 [`实验结果与分析.md`](实验结果与分析.md)。远端实验、预测与训练目录均未被修改。

## 目录

- `实验结果与分析.md`：可直接改写进论文的中文结果章节，含表格、图、图注、证据分级与局限。
- `figures/`：论文图的 PNG 预览与矢量 PDF。
- `tables/`：主文和附录所需的机器可读 CSV。
- `scripts/`：四部分二次分析和复核脚本。
- `work/`：各分析分支的完整中间产物、notes、JSON 审计与更细粒度表格。
- `data/`：新版 target-only 消融的原始二次汇总与 provenance。

## 冻结数据源

### 主实验

- `/data/xmy/CaFE/runtime/experiments/gift-v15-short-qualified-feasible-seed2026082701`
- `/data/xmy/CaFE/runtime/experiments/gift-v15-medium-qualified-feasible-seed2026082701`
- `/data/xmy/CaFE/runtime/experiments/gift-v15-long-qualified-feasible-seed2026082701`
- `/data/xmy/CaFE/runtime/experiments/fev-mini20-full-v6`

### 稳定性

- `/data/xmy/CaFE/runtime/orchestration/short_stability10_inference_3node_78ef32f_20260831/stability`

### 微调

- `/data/xmy/chronos-forecasting/chronos-2-finetuned/cafe-v15-qf-moirai16k-window10-replacement-40k`
- `/data/xmy/chronos-forecasting/chronos-2-finetuned/cafe-v15-qf-moirai16k-window10-nrmse-replacement-40k`

### 新版多变量消融

- `/data/xmy/CaFE/runtime/ablation_trials/gift-v15-seed2026082701-target-only-v1`
- `/data/xmy/CaFE/runtime/ablation_trials/fev-mini20-full-v6-target-only-v1`

## 推荐主文图

| 建议图号 | 文件 | 内容 |
|---|---|---|
| Figure 1 | `fig_capability_heatmaps.pdf` | 四个 suite 的模型 × capability heatmap |
| Figure 2 | `fig_rank_divergence.pdf` | Official MASE 与描述性能力排名的分化 |
| Figure 3 | `fig_level_curves_gift_short.pdf` | Short 的 capability × level 曲线 |
| Figure 4 | `fig_stability_overall.pdf` | 十个 augmentation seeds 的宏观分数与排名 |
| Figure 5 | `fig_capability_level_winner_consistency.pdf` | 细粒度 winner frequency |
| Figure 6 | `finetuning_curves_absolute.pdf` | 两套微调协议的 MASE/NRMSE checkpoint 曲线 |
| Figure 7 | `fig_target_only_ablation.pdf` | 新版删除式多变量消融 |

Medium、Long、FEV level curves，official MASE 区间、seed/task uncertainty 和 structure diversity 更适合作为附录图。

## 关键表

- `experiment_inventory.csv`：任务、实例、treatment、预测数量与失败数。
- `official_mase_suite.csv`：Official MASE、task-bootstrap 区间与排名。
- `effect_nrmse_derived_macro.csv`：展示性 8-capability × 5-level 宏 NRMSE。
- `effect_nrmse_by_suite_model_capability_level.csv`：协议主体的完整 capability × level 结果。
- `effect_nrmse_level_averaged_capability.csv`：逐能力 5-level 等权概览。
- `capability_availability.csv`：任务与实例覆盖率。
- `model_overall_stability_extended.csv`：十-seed 模型稳定性。
- `capability_level_winner_consistency.csv`：40 个细粒度单元的 winner frequency。
- `key_checkpoints.csv`：微调 macro-stratum treatment MASE/effect NRMSE 的 baseline/best/final 数值。
- `seed_transfer_summary.csv`：A/B 轨迹相关与迁移差。
- `ablation_collapsed_summary.csv`：新版消融的 task/level-equal 汇总、区间与 leave-one-task-out 结果。
- `ablation_task_level.csv`：新版消融逐任务、逐 level 数据。

## 口径提醒

1. `effect_nrmse_derived_macro.csv` 是便于展示的派生宏平均，不是协议定义的单一排行榜分数。
2. NRMSE = 1 是零响应参考；小于 1 表示比忽略 treatment 更接近真实效应。
3. 主实验按 task-equal 聚合；宏指标 bootstrap 在 capability 内重采样合格任务，再对 capability 等权。
4. 稳定性结论只适用于固定的 10-task GIFT-Short panel。
5. 微调指标先在 `(dataset, capability, level)` strata 内聚合后再等权平均，不是主实验的 task-equal Official MASE；两者绝对值不可直接横比。
6. 微调的两套协议同时改变了 objective、训练输入、forecast origin、学习率与数据遍历次数，不能解释为 loss-only 因果消融。
7. 新版消融定义 `ΔMASE = MASE_removed − MASE_full`；正值表示辅助输入帮助绝对预测。
8. 旧 temporal-shift ablation 已被 target-only 设计取代，不应用于最终结构归因。
9. 所有 cell-level 置信区间与星号均未做多重比较校正。

## 复核状态

- 主实验：62 个任务、1,040 个 suite/model/capability/level 单元与跨层聚合恒等式均通过验证；25,279,591 条预测，0 inference failures。
- 稳定性：10 个 seeds 的 392 个 suite keys 完全一致；100/100 validations accepted，700/700 model-task inference complete；`work/stability/verification.json` 全部通过。
- 微调：44 个分布式指标组重新聚合后与服务器曲线一致；A/B treatment sample ID 交集为 0。
- 新版消融：GIFT 与 FEV 的 missing pair 数均为 0；远端来源与文件统计记录于 `data/ablation_provenance.json`。

## 复现提示

分析脚本中的默认输入可能保留本机缓存或远端路径约定，执行前请先阅读对应 `work/*/README.md` 与 provenance。微调远端目录存在未被训练 manifest 记录的脚本代码状态，因此当前 bundle 可复核数值汇总，但若要声称完全可重复训练，应先冻结训练代码 commit/hash、数据 manifest 与运行环境。
