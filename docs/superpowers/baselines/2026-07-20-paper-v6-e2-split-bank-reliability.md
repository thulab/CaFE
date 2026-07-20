# Paper v6 E2 split-bank 样本量可靠性

日期：2026-07-20

## 1. 问题与口径

v6 每个 `dataset × task × capability × intensity` cell 生成了 160 条
master samples。160 并非理论预设的必要样本数，而是此前发现 N=32 排名波动较大后
采用的正式规模。本分析直接用现有 160 个 paired groups 估计不同样本量下的测量
可靠性。

分析单位是 paired group：

- 每个 dataset/task/capability 的 160 个 paired groups 构成一个平坦样本池；
- round 字段被完全忽略，不作为生成模式或统计层级；
- 同一个 paired group 的五档 intensity 和所有兼容模型共享 bank assignment；
- ordered split 按 `paired_group_id` 确定性排序，比较前 N 与后 N；
- 另做 10 次固定 seed 的随机不相交二分，检查结果是否依赖一次切分；
- 比较 N=`32/48/64/80`，每个 bank 各含 N 条，因此 N=80 使用完整 160 条池。

这是同一冻结生成池内的 split-half reliability，适合选择样本量，但弱于独立重新
生成 Bank B 的 external seed-bank replication。

## 2. Oracle-context 样本量趋势

Ordered split：

| N per bank | Rank agreement | Cells ≥0.80 | Top-1 | Profile CCC | Tie-state match |
|---:|---:|---:|---:|---:|---:|
| 32 | 0.7973 | 58.1% | 56.9% | 0.9809 | 0.8057 |
| 48 | 0.8289 | 66.6% | 64.1% | 0.9871 | 0.8146 |
| 64 | 0.8443 | 71.9% | 68.1% | 0.9903 | 0.8128 |
| 80 | 0.8525 | 72.5% | 73.1% | 0.9908 | 0.8135 |

10 次 random split 的 rank agreement：

| N per bank | Mean | P10 | P90 | Mean cells ≥0.80 | Mean profile CCC |
|---:|---:|---:|---:|---:|---:|
| 32 | 0.7846 | 0.7755 | 0.7948 | 55.4% | 0.9774 |
| 48 | 0.8129 | 0.8087 | 0.8154 | 62.2% | 0.9839 |
| 64 | 0.8350 | 0.8261 | 0.8421 | 67.9% | 0.9882 |
| 80 | 0.8471 | 0.8420 | 0.8528 | 70.3% | 0.9905 |

结果显示样本量增加会稳定提升完整排名 agreement 和 capability profile
reliability。N=64 到 N=80 仍有收益，但边际收益已经下降。能力 profile 在 N=64–80
时已非常稳定；完整八模型名次仍明显更敏感。

## 3. N=80 的能力差异

Ordered oracle-context split：

| Capability | Rank agreement | Cells ≥0.80 | Top-1 |
|---|---:|---:|---:|
| trend | 0.7121 | 48.0% | 42.0% |
| regime_switching | 0.7743 | 46.0% | 64.0% |
| nonlinear_persistence | 0.7857 | 48.9% | 64.4% |
| predictable_intermittency | 0.8979 | 82.0% | 96.0% |
| hierarchical_coherence | 0.9000 | 100.0% | 100.0% |
| time_varying_seasonality | 0.9286 | 93.3% | 100.0% |
| common_factor | 0.9556 | 100.0% | 80.0% |
| multi_seasonal | 0.9586 | 100.0% | 66.0% |
| covariate_response | 0.9667 | 100.0% | 90.0% |

因此全局排名可靠性未达到近乎完全一致，主要不是所有能力都需要更多样本，而是
`trend`、`regime_switching` 和 `nonlinear_persistence` 三维的模型间差距较小或
样本误差方差较大。后续应同时检查这些维度的机制评价与模型 pair margin，不宜只靠
继续增加所有能力的统一样本数解决。

## 4. 解释与建议

当前证据支持：

1. N=32 对完整模型排名偏小；
2. N=48 已超过平均 0.80 ordering agreement；
3. N=64 的 capability profile CCC 约 0.99，作为能力画像已经足够稳定；
4. N=80 比 N=64 继续改善完整排名，但无法让所有 cell 的完整名次稳定；
5. 正式实验继续用每 cell 160 条总体均值是保守且合理的，因为 split-half 的每个
   bank 只有 80 条；但“160 对另一套 160 是否稳定”仍需独立 Bank B 才能直接回答。

论文中建议把连续能力分数和 capability profile 作为主要可靠性对象，把完整排名、
top-1、top-3 和 tie-aware pair states 作为直观但更敏感的正式结果共同报告。

## 5. 产物

分析脚本：

`scripts/analyze_paper_e2_split_bank_reliability.py`

运行命令：

```bash
uv run --project backend python \
  scripts/analyze_paper_e2_split_bank_reliability.py \
  --e2-dir runtime/paper_exp/v6/E2_dynamic_stability \
  --bank-sizes 32 48 64 80 \
  --random-repeats 10
```

运行目录：

`runtime/paper_exp/v6/E2_dynamic_stability/split_bank_reliability/`

主要文件：

- `summary.json`；
- `report.md`；
- `split_comparison_summary.csv`；
- 两种 context policy 的 ordered cell-model、capability-profile、
  tie-aware contrast 和 rank reliability 明细；
- 按 capability、dataset、intensity 汇总的 ordered rank tables；
- `manifest.json`。
