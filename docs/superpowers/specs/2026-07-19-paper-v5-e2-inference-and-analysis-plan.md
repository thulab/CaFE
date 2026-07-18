# Paper v5 E2：正式推理与分析规划

日期：2026-07-19

## 1. 研究问题

E2 固定回答两个问题。

### E2-A：逐 cell 动态生成稳定性

对每个 supported
`dataset × task × capability × intensity` cell，改变五轮生成 seed 后：

1. 同一模型的平均预测误差是否稳定；
2. 兼容模型的相对排名是否近乎完全一致。

不得先对 capability、intensity 或 dataset 汇总再检验稳定性，以免掩盖局部不稳定
cell。

### E2-B：Synthetic–real source-window alignment

对每个至少支持两项 synthetic capability 的单变量 dataset，比较：

- 对 synthetic capability、intensity 和 round 等权平均得到的模型排名；
- 模型在同一 dataset 校准 source/reference windows 上的真实预测排名。

该部分检验 dataset-local 多能力合成测试是否能够描述源序列上的模型相对预测能力，
不声称对未见 test windows 的外部泛化。

## 2. 冻结输入

Synthetic master collection：

```text
runtime/paper_exp/v5/E2_dynamic_stability/
  generation_config.json
  sample_manifest.json
  samples.jsonl
```

固定为：

- `H=48`；
- `L={96,168,336,504}`；
- 63 个 supported dataset/task/capability cells；
- 5 个 intensity；
- 5 个 round；
- 每轮每 cell 32 个 paired master groups；
- 50,400 条 intensity-specific master samples。

Real source windows 从正式校准 artifact 的 `L=504,H=48`
`near_distance_artifact.buckets[*].reference_raw` 冻结，不再构造独立
test/holdout。主分析只纳入至少支持两项能力的十个单变量 dataset。当前 artifact
包含 417 条 real master windows，每个 dataset 35–44 条。

## 3. 模型与执行配置

基础模型固定为：

| Model | Replicas/GPU | Global HTTP concurrency |
| --- | ---: | ---: |
| Timer-3.5 | 1 | 64 |
| Timer-3.0 | 1 | 32 |
| Chronos-2 | 4 | 32 |
| moirai2 | 2 | 16 |
| toto2.0 | 2 | 16 |
| timesfm2.5 | 8 | 32 |
| tirex2 | 1 | 32 |
| tabpfn-ts3 | 8 | 24 |

使用 GPU `0,1`，每次只加载一个基础模型。`naive` 与 `seasonal_naive` 在本地计算，
仅作为诊断基线，不进入 foundation-model headline 排名相关性。

模型兼容性以运行开始前封存的 service catalog 为准。不兼容 sample/view 记录为
`N/A`，不记为失败或最差分数。TabPFN-TS-3 按当前正式约定支持单变量、多变量和
未来协变量任务。

## 4. 四 Context 推理单位

一条 `L=504,H=48` master 派生四个 view：

```text
view_history = master_history[504-L:504]
view_future  = master_future[504:552]
```

四个 view 共享完全相同的 future。每个 view 只用自己的 history 重新标准化 target
和 covariates。每个模型、master、context 均保存：

- deterministic `view_id`；
- forecast；
- MASE、MAE、MSE；
- request attempts、耗时与 shape bucket；
- 成功或失败状态。

逐样本主分数为：

\[
S_{m,s}=\min_{L\in\{96,168,336,504\}} MASE_{m,s,L}.
\]

完全相同的 MASE 按 `L=96,168,336,504` 顺序确定性选择。原始四 view 结果不得删除；
固定 `L=504` 作为敏感性分析。

## 5. E2-A 分析

对每个 `model × dataset × task × capability × intensity × round`，先对 32 条
master 的 oracle-context MASE 求均值，再在同一
`dataset × task × capability × intensity × round` 内按 MASE 升序排列兼容模型。

五轮产生十对排名比较。每个 cell 报告：

- Kendall τ-b：mean、minimum 和十对明细；
- pairwise ordering agreement：mean、minimum；
- exact rank-vector agreement；
- top-1 和 top-3 agreement；
- 每个模型五轮 MASE mean、standard deviation、CV、minimum 和 maximum；
- normalized round difficulty multiplier。

Foundation-model 主判据：

- 任意两轮 pairwise ordering agreement 不低于 `0.95`；
- 七模型 cell 中，这等价于任意两轮至多一个 model pair 反转；
- 模型数更少的结构化 cell 因排名分辨率较粗，通常要求完全一致。

排名稳定只说明相对难度结构稳定。只有排名稳定且 round MASE 水平波动小，才能解释为
随机 seed 没有明显改变绝对预测难度。所有不通过 cell 必须逐项列出，不能由整体
median 掩盖。

## 6. E2-B 分析

### Synthetic rank

在每个 `dataset × capability × intensity × round` 内对 foundation models 排名，
随后依次对：

1. 五轮；
2. 五档 intensity；
3. dataset 支持的 capabilities；

等权平均，得到 `dataset × model` synthetic mean rank。不同 capability 不按样本数
加权；unsupported cell 不补值。

### Real source-window rank

每个 real master 同样先按模型选择 oracle context。随后在
`dataset × model` 内对 master-level MASE 求均值并排序。

### 对齐统计

只使用 synthetic 与 real 两侧均完整的 foundation models，逐 dataset 报告：

- Spearman ρ；
- Kendall τ-b；
- top-3 overlap；
- pairwise ordering agreement；
- synthetic rank gap、real MASE gap；
- real master count、supported capability count 和共同模型数。

最终以 dataset 为独立统计单位报告 mean、median 和 dataset-level bootstrap CI。
结论限定为 source-window construct alignment，不表述为 held-out external validity。

## 7. 失败、恢复与封存

- 正式推理前对每个模型和每种兼容 shape 做 preflight；
- prediction JSONL append-only，以 `view_id` 恢复，只补未成功 view；
- 请求默认最多三次尝试，失败逐 view 落盘；
- 每500条成功结果 flush 并更新进度；
- 单个模型覆盖不完整时不得进入正式分析；
- manifest 封存 sample collection、real source suite、runner、protocol、catalog、
  predictions、analysis outputs 的大小和 SHA-256。

## 8. 正式产物

```text
runtime/paper_exp/v5/
  02_real_source_window_suite/
  E2_dynamic_stability/
    inference_config.json
    inference_model_catalog.json
    predictions/
    real_source_predictions/
    failures/
    real_failures/
    model_status.json
    oracle_sample_scores/
    cell_round_scores.csv
    cell_rank_stability.csv
    cell_score_stability.csv
    synthetic_model_ranks.csv
    real_source_model_ranks.csv
    synthetic_real_source_alignment.csv
    inference_summary.json
    inference_report.md
    inference_manifest.json
```
