# Paper v8 adaptive-v6j Chronos-2 跨 seed 微调迁移实验

## 结论

在最新全量实验 `v8_gift20_adaptive_v6j_full_20260727` 的 primary/main
合成样本上，将 64 个 formal seed 固定拆为 A（0–31）和 B（32–63）。
Chronos-2 只在 A 上进行 20,000 step full fine-tuning，并每 1,000 step
在完整 A/B 上评测。

结果显示：

- step 0→1,000 存在共享适配收益：A/B MAE 分别改善 29.56% 和 24.24%；
- step 1,000→20,000 时，A 的 MAE 又下降 35.94%，B 只下降 7.53%；
- 相对各自 step 0，最终 A 的 MAE 改善 54.88%，B 改善 29.94%；
- B 的 RMSE 在 step 9,000 最优，normalized WQL 在 step 7,000 最优，
  此后继续拟合 A 不再改善这些 B 指标，最终略有恶化。

因此准确表述不是“合成微调完全不迁移”，而是：早期训练带来明显的共享
适配，但继续拟合一批 synthetic seed 的边际收益主要留在训练批次，对另一批
disjoint seed 的迁移很弱，并在部分指标上出现轻微负迁移。

## 数据与任务口径

源实验：

```text
runtime/paper_exp/v8/v8_gift20_adaptive_v6j_full_20260727
protocol_sha256 =
fabcd39afb72fcc80fe7faf0c9e864defb0cd46a6b3e3dc34fd26a365099316e
```

选择规则：

```text
evaluation_table = main
generator_family_role = primary
```

排除 secondary generator family、strict counterfactual audit、robustness 和
input ablation。最终覆盖 20 个数据集、10 类 capability、116 个
dataset-capability cell 和 580 个 dataset-capability-intensity group。
其中 575 个 group 每 seed 一条 master，covariate-response 的 5 个 group
每 seed 保留两个 counterfactual member。

| Split | formal seeds | master samples | independent target tasks | forecast points |
|---|---:|---:|---:|---:|
| A | 0–31 | 18,720 | 23,520 | 1,128,960 |
| B | 32–63 | 18,720 | 23,520 | 1,128,960 |

每个 seed 的所有 intensity、counterfactual member、target 和 nuisance path
都留在同一 split，A/B 没有 seed 泄漏。

固定使用 Paper v8 main-table 任务：

```text
context = master_target[168:336]
labels  = master_target[336:384]
```

导出后每条序列长 216，训练使用 `min_past=168`、
`prediction_length=48`，因此只有一个合法训练切点 168。输入沿用 L336
master 的标准化，只做 slice、不重新标准化。多目标 master 拆成独立的
单目标 Chronos 任务；known-future covariates 与对应 target 保持成组。

A 的最后 48 点参与监督微调，因此 A 评测是有意设计的 in-sample
memorization probe；B 是 disjoint-seed transfer。

## MAE 曲线

| step | A MAE | B MAE | A 相对 step-0 改善 | B 相对 step-0 改善 |
|---:|---:|---:|---:|---:|
| 0 | 0.359789 | 0.354432 | 0.00% | 0.00% |
| 1,000 | 0.253423 | 0.268532 | 29.56% | 24.24% |
| 2,000 | 0.231266 | 0.259493 | 35.72% | 26.79% |
| 3,000 | 0.217276 | 0.255677 | 39.61% | 27.86% |
| 5,000 | 0.199758 | 0.252359 | 44.48% | 28.80% |
| 7,000 | 0.187757 | 0.250293 | 47.81% | 29.38% |
| 9,000 | 0.179640 | 0.249407 | 50.07% | 29.63% |
| 10,000 | 0.176727 | 0.249905 | 50.88% | 29.49% |
| 15,000 | 0.165708 | 0.248520 | 53.94% | 29.88% |
| 17,000 | 0.163522 | 0.248341 | 54.55% | 29.93% |
| 20,000 | **0.162349** | **0.248318** | **54.88%** | **29.94%** |

从 step 1,000 到 20,000，A 获得的额外相对改善是 B 的约 4.8 倍
（35.94% 对 7.53%）。最终 A 只剩 step-0 MAE 的 45.12%，B 仍为
70.06%。

其他指标的最佳 checkpoint：

| metric | A best | B best | B final |
|---|---:|---:|---:|
| RMSE | 0.525571 @ 20k | 0.587299 @ 9k | 0.588269 |
| MASE | 0.225161 @ 20k | 0.354134 @ 18k | 0.354378 |
| normalized WQL | 0.143215 @ 17k | 0.221659 @ 7k | 0.223278 |

## 微调配置

| 配置项 | 值 |
|---|---|
| 模型 | `amazon/chronos-2` |
| 微调 | full fine-tuning |
| context / horizon | 168 / 48 |
| steps / eval interval | 20,000 / 1,000 |
| batch size | 64 variates |
| learning rate | 1e-6 |
| optimizer | fused AdamW |
| schedule | linear，0 warmup |
| weight decay | 0 |
| precision | BF16 + TF32 |
| inference batch size | 256 |
| inference cross-learning | false |
| training seed | 20260727 |
| GPU | RTX 5090 |
| Chronos repo commit | `7dc4435706a4454feb79df44ca9f33631f3027bf` |
| wall time | 1,463.42 s |

## 产物与复现

本地归档：

```text
runtime/paper_exp/v8/
  v8_gift20_adaptive_v6j_chronos2_seed_transfer_20260727/
  ├── data/
  │   ├── A.jsonl
  │   ├── B.jsonl
  │   └── split_manifest.json
  └── results/
      ├── mae_transfer_curve.{png,svg}
      ├── all_metric_transfer_curves.{png,svg}
      ├── step_metrics.csv
      ├── config.json
      ├── result.json
      ├── status.json
      ├── experiment_script.py
      └── full_curve_20000.log
```

远程 checkpoint 和完整实验目录：

```text
timecho17:/home/xmy/chronos-forecasting/experiments/
  v8_gift20_adaptive_v6j_seed_transfer_20260727/full_curve_20000
```

数据准备：

```bash
python scripts/prepare_paper_v8_chronos_finetune.py \
  --source-experiment \
  runtime/paper_exp/v8/v8_gift20_adaptive_v6j_full_20260727 \
  --output-dir \
  runtime/paper_exp/v8/\
v8_gift20_adaptive_v6j_chronos2_seed_transfer_20260727/data
```

服务器执行：

```bash
cd ~/chronos-forecasting
HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 CUDA_VISIBLE_DEVICES=0 \
  .venv/bin/python \
  experiments/v8_gift20_adaptive_v6j_seed_transfer_20260727/\
run_paper_v8_chronos_finetune_curve.py \
  --data-a \
  experiments/v8_gift20_adaptive_v6j_seed_transfer_20260727/input/A.jsonl \
  --data-b \
  experiments/v8_gift20_adaptive_v6j_seed_transfer_20260727/input/B.jsonl \
  --output-dir \
  experiments/v8_gift20_adaptive_v6j_seed_transfer_20260727/full_curve_20000 \
  --max-steps 20000 \
  --eval-interval 1000 \
  --batch-size 64 \
  --inference-batch-size 256 \
  --learning-rate 1e-6 \
  --training-seed 20260727
```

本实验固定一个 A/B partition 和一个 training seed。若论文需要把效应大小
写成总体估计，应补充 alternate seed partitions 或 training-seed
replicates；当前结果可作为一次清晰的机制性曲线。
