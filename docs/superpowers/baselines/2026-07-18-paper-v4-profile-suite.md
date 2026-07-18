# Paper v4：逐数据集 profile suite 重建记录

日期：2026-07-18
状态：协议与 builder 已更新；正式 runtime freeze 待重建

## 决策

原 profile freeze 将 13 个 source config 汇总成 family-balanced pooled distribution。
这会让异质 dataset 共同放宽 nuisance/support 范围，并削弱 synthetic sample 与具体真实
dataset 的对应关系。该 freeze 现已废止。

新版本把每个 dataset config 视为独立实验单元：

```text
dataset
  -> paired 504+48 master windows
  -> L={96,168,336,504} nested views
  -> 4 个 dataset-local profiles
  -> dataset-local relative intensity / conditioning / gates
```

不再存在 family、pooled profile 或 global profile。ETT1/H、ETT2/H、Bitbrains
Fast/RND 等分别校准、分别生成、分别报告。

## 保留的设计

- `H=48`、`L={96,168,336,504}`、`season_length=24`；
- 四个 L 共享同一母窗口、future、series/channel 和 forecast origin；
- GIFT-Eval official test tail 与相邻 48 点 validation embargo 不进入 profile；
- 每个 dataset 最多 240 条母窗口；
- series/channel 优先覆盖，再增加时间 origin；
- 任一 nested view 不合格时整条母窗口退出；
- 资产、协议、builder 和输出继续由 manifest 封存。

## 删除的设计与产物

以下内容不再属于正式实验：

- `family_id` 和 11-family 计数；
- family → source config → window 三级权重；
- `family_macro__L96_H48`、`family_macro__L168_H48`、
  `family_macro__L336_H48`、`family_macro__L504_H48`；
- `global_profiles`；
- `source_inventory.csv`；
- “dataset 只作 provenance、不作为 benchmark axis”的结论；
- 使用 pooled real distribution 驱动所有单变量生成的产物。

旧目录中的 v1 JSON、CSV、report、manifest 及其哈希只代表 superseded artifact，不得用于
新论文实验。此前记录的 3,120 条母窗口、12,480 条 view 和 4 个 family-macro profile
也不能作为新版本的正式 freeze 统计。

## 新 builder 契约

脚本：

```text
scripts/build_paper_v4_profile_suite.py
```

新 schema：

```text
paper_v4_dataset_local_multi_lookback_profile_suite.v2
```

默认输出：

```text
runtime/paper_exp/v4/00_profile_suite/
  profile_suite.json
  profile_rows.csv
  dataset_inventory.csv
  report.md
  manifest.json
```

`profile_suite.json` 应满足：

- `selection.dataset_count = 13`；
- `profiles` 数量为 `13 × 4 = 52`；
- 每个 profile_id 为 `{dataset_id}__L{L}_H48`；
- 每个 profile 只统计其自身 dataset 的窗口；
- 不含 `family_count`、`family_id` 或 `global_profiles`。

逐窗口 CSV 和 inventory 使用 `dataset_id`，不再使用 `source_id`。

## 能力支持规则

一个 dataset 不要求支持九项能力，也不保证每项能力能分出五档。下游按
`dataset × task view × capability` 审计：

- 真实 feature 分位数没有足够间距：`unsupported`；
- 缺少多变量、层级或 known-future covariate 结构：`unsupported`；
- inverse calibration 不单调或不能命中局部 target：`unsupported`；
- dataset-local feature/near-distance gate 无法通过：`unsupported`。

只有受支持的能力进入该 dataset 的相对 intensity 响应实验，如实保留完整支持矩阵。

## 重建要求

正式重跑前应删除或移出旧的封存目录，不能在旧 manifest 上覆盖。profile suite 重建后，
所有依赖 pooled profile 或全局 intensity 的 nine-capability conditioning、feature gate、
near-distance、qualification 和模型推理产物必须一并重建。

完整协议见
`docs/superpowers/specs/2026-07-18-paper-v4-multi-lookback-profile-protocol.md`。
