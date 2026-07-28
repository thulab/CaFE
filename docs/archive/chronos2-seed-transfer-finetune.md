# Paper v8 Chronos-2 跨 seed 微调迁移实验

## 结论

在 `v8_full20_struct100_neardist_20260727` 的 primary/main 合成样本上，
将 64 个 formal seed 固定拆为 A（0–31）和 B（32–63），只在 A 上连续
full fine-tuning 20,000 step，并每 1,000 step 在完整 A/B 上评测。

结果显示：

- step 0→1,000 存在明显的共享适配收益；
- step 1,000→20,000 时，A 的 MAE 又下降 50.72%，B 只下降 5.70%；
- 相对各自 step-0，最终 A 的 MAE 改善 70.59%，B 改善 31.09%；
- B 的 normalized WQL 在 step 3,000 最优，之后继续拟合 A 反而从
  `0.146995` 恶化到 `0.151016`。

因此准确表述不是“合成微调完全不迁移”，而是：早期存在共享收益，但继续
拟合一批 synthetic seed 的边际收益主要停留在训练批次，对另一批 disjoint
seed 的迁移很弱，并会在部分概率预测指标上产生负迁移。

## 数据与任务口径

源实验：

```text
runtime/paper_exp/v8/v8_full20_struct100_neardist_20260727
protocol_sha256 =
6ce257b7255a50be163eb2d61fa73775e2265525b5733c427321694a3cf44656
```

选择规则：

```text
evaluation_table = main
generator_family_role = primary
```

排除 secondary generator family、strict counterfactual audit、robustness 和
input ablation。这样共有 73 个 dataset-capability cell、365 个
dataset-capability-intensity group。360 个 group 每 seed 一条 master，
covariate-response 的 5 个 group 每 seed 保留两个 counterfactual member。

| Split | formal seeds | master samples | independent target tasks | forecast points |
|---|---:|---:|---:|---:|
| A | 0–31 | 11,840 | 13,440 | 645,120 |
| B | 32–63 | 11,840 | 13,440 | 645,120 |

每个 seed 的所有 intensity、counterfactual member、target 和 nuisance path
都留在同一 split，A/B 没有 seed 泄漏。

固定使用 Paper v8 main-table 任务：

```text
context = master_target[168:336]
labels  = master_target[336:384]
```

导出后每条序列长 216，训练使用 `min_past=168`、`prediction_length=48`，
因此 Chronos-2 只有一个合法训练切点 168。输入沿用 L336 master 的标准化，
只做 slice、不重新标准化。多目标 master 按 v8 Chronos 输入适配拆成独立
单目标任务；known-future covariates 与对应 target 一起送入模型。

A 的最后 48 点参与监督微调，所以 A 评测是有意设计的 in-sample
memorization probe；B 是 disjoint-seed transfer。

## MAE 曲线

| step | A MAE | B MAE | A 相对 step-0 改善 | B 相对 step-0 改善 |
|---:|---:|---:|---:|---:|
| 0 | 0.264572 | 0.247622 | 0.00% | 0.00% |
| 1,000 | 0.157908 | 0.180933 | 40.32% | 26.93% |
| 2,000 | 0.134945 | 0.176395 | 48.99% | 28.76% |
| 3,000 | 0.122010 | 0.173873 | 53.88% | 29.78% |
| 5,000 | 0.105817 | 0.173183 | 60.00% | 30.06% |
| 10,000 | 0.087581 | 0.171669 | 66.90% | 30.67% |
| 15,000 | 0.080110 | 0.170850 | 69.72% | 31.00% |
| 17,000 | 0.078476 | **0.170459** | 70.34% | **31.16%** |
| 20,000 | **0.077814** | 0.170625 | **70.59%** | 31.09% |

B 的最佳 MAE 在 step 17,000，但与 step 3,000 的差异只有 `0.003414`；
同期 A 从 `0.122010` 进一步降至 `0.078476`。最终 A 只剩 step-0 MAE 的
29.41%，B 仍为 68.91%。

其他指标的最佳 checkpoint：

| metric | A best | B best | B final |
|---|---:|---:|---:|
| RMSE | 0.278602 @ 20k | 0.364402 @ 3k | 0.366158 |
| MASE | 0.120909 @ 18k | 0.245762 @ 18k | 0.247313 |
| normalized WQL | 0.069693 @ 13k | 0.146995 @ 3k | 0.151016 |

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
| wall time | 1,379.33 s |

## 产物与复现

本地归档：

```text
runtime/paper_exp/v8/v8_chronos2_seed_transfer_20260727/
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
  v8_seed_transfer_20260727/full_curve_20000
```

数据准备：

```bash
python scripts/prepare_paper_v8_chronos_finetune.py \
  --source-experiment \
  runtime/paper_exp/v8/v8_full20_struct100_neardist_20260727 \
  --output-dir \
  runtime/paper_exp/v8/v8_chronos2_seed_transfer_20260727/data
```

服务器执行：

```bash
cd ~/chronos-forecasting
HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 CUDA_VISIBLE_DEVICES=0 \
  .venv/bin/python \
  experiments/v8_seed_transfer_20260727/run_paper_v8_chronos_finetune_curve.py \
  --data-a experiments/v8_seed_transfer_20260727/input/A.jsonl \
  --data-b experiments/v8_seed_transfer_20260727/input/B.jsonl \
  --output-dir experiments/v8_seed_transfer_20260727/full_curve_20000 \
  --max-steps 20000 \
  --eval-interval 1000 \
  --batch-size 64 \
  --inference-batch-size 256 \
  --learning-rate 1e-6 \
  --training-seed 20260727
```

本实验固定了一个 A/B partition 和一个 training seed。论文若要把效应大小
写成总体估计，后续应补充 alternate seed partitions 或 training-seed
replicates；当前结果已经足以展示清晰的单次机制性曲线。
