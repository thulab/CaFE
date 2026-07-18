# Paper v4：多 lookback 真实 profile freeze

日期：2026-07-18

## 结论

面向能力维度 benchmark 的真实 profile suite 已完成。正式 shape 为：

```text
H = 48
L = {96, 168, 336, 504}
season_length = 24
```

本版不为四个 L 独立抽样。每条 `504+48` 母窗口生成四个 suffix context，四种输入长度共享
完全相同的 raw future、series/channel 和 forecast origin。该设计保证后续 best-of-four
lookback 比较只改变模型可见历史，不改变测试题。

## 数据覆盖

正式来源为 13 个公开小时级配置、11 个独立 family、5 个领域：

| domain | source configs |
|---|---|
| Econ/Fin | M4 Hourly |
| Energy | Electricity/H、Solar/H、ETT1/H、ETT2/H |
| Nature | Jena Weather/H、KDD Cup 2018/H |
| Transport | Loop Seattle/H、SZ-Taxi/H、M_DENSE/H |
| Web/CloudOps | Bitbrains Fast Storage/H、Bitbrains RND/H、BizITObs L2C/H |

ETT1/2 合并为一个 family，Bitbrains Fast/RND 合并为一个 family。全局统计采用
`family -> source config -> paired window` 三级等权，不让同族变体或大数据集支配 profile。

数据集选择只依据公开 benchmark 来源、小时频率、最大窗口可行性和领域覆盖，不读取模型
成绩。

## 产物规模与来源审计

- paired master windows：3,120；
- nested view rows：12,480；
- provenance profiles：52（13 source configs × 4 L）；
- family-macro profiles：4；
- 每个 source config 最多 240 条母窗口；
- 四档 future SHA-256 逐母窗口一致，3,120/3,120 通过；
- 全部 13 个来源均达到 240 条有效母窗口。

主要 series/channel 覆盖：

| source | used series | 可覆盖上限 | coverage |
|---|---:|---:|---:|
| M4 Hourly | 240 | 240 | 100% |
| Electricity/H | 150 | 240 | 62.5% |
| Solar/H | 137 | 137 | 100% |
| ETT1/H | 7 | 7 | 100% |
| ETT2/H | 7 | 7 | 100% |
| Jena Weather/H | 21 | 21 | 100% |
| KDD Cup 2018/H | 224 | 240 | 93.3% |
| Loop Seattle/H | 240 | 240 | 100% |
| SZ-Taxi/H | 156 | 156 | 100% |
| M_DENSE/H | 30 | 30 | 100% |
| Bitbrains Fast Storage/H | 240 | 240 | 100% |
| Bitbrains RND/H | 240 | 240 | 100% |
| BizITObs L2C/H | 7 | 7 | 100% |

Electricity 的较低 series 覆盖来自原始缺失值 complete-case 拒绝，而不是扁平候选截断。
所有 nested view 至少 50% observed，任一 L 失败会让该母窗口的四档视图同时退出。

## 窗口长度依赖性

family-macro canonical-measurement 中位数如下，进一步证明不能把 504 profile 原样复用于
168/336：

| feature | L=96 | L=168 | L=336 | L=504 |
|---|---:|---:|---:|---:|
| trend strength | 0.2056 | 0.1173 | 0.0666 | 0.0755 |
| seasonal strength | 0.6019 | 0.5138 | 0.4400 | 0.3969 |
| multi-period score | 0.1611 | 0.1511 | 0.1576 | 0.1612 |
| amplitude modulation | 0.4159 | 0.4626 | 0.5003 | 0.5216 |
| change-point shift energy | 0.8286 | 0.6040 | 0.6739 | 0.8690 |
| nonlinear conditional gain | 0.000921 | 0.000529 | 0.000370 | 0.000275 |
| spike rate | 0.02521 | 0.02094 | 0.02507 | 0.02657 |

Canonical intensity 后续必须只冻结一次；上表的长度差异用于拟合各 L 的 nuisance/control
support，不能被解释成三套不同 intensity 标尺。

## 实现修正

第一次完整构建暴露出旧式扁平候选截断会减少 channel 覆盖，例如 ETT 只覆盖 2/7、
Jena 只覆盖 4/21。该产物已移至
`runtime/paper_exp/v4/00_profile_suite_superseded_flat_sampling/`，不进入正式实验。

正式构建改为 series/channel 优先的均衡抽样，再在同一 series 内均匀增加时间 origin。
修正后 ETT、Jena、M_DENSE 等小 channel panel 均实现完整 channel 覆盖。

## 封存信息

正式目录：

```text
runtime/paper_exp/v4/00_profile_suite/
```

关键哈希：

```text
profile_suite.json  a28116d13d972661bceb1f236326ec9442e4c6050a9bb3bc38f4706196c80e06
profile_rows.csv    3234af5b52af76308440b29834b79c9c2874da4645cf802661356cbafdcb2753
source_inventory.csv
                    4466c8819c8ac2412d3f7cb6e9c8c7e1c0ecf46ab7188debf8f05791f9bc1aee
manifest.json       99284863d77bb4900675c47992d5776c477e725ce6a3edf6dec138c8fbd76ec8
```

完整协议见
`docs/superpowers/specs/2026-07-18-paper-v4-multi-lookback-profile-protocol.md`。

## 下一阶段边界

本 freeze 完成的是真实 profile 选择、提取和长度配对。九能力的 generator
conditioning、feature-support 与 near-distance artifacts 已在后续
`01_nine_capability_suite` 发布并完成四档联合验收：

1. 在同一 canonical intensity scale 下生成 `504+48` synthetic master；
2. 对四个 suffix view 分别使用 length-conditional support；
3. 四档任一必要 gate 失败时，整条 paired master sample 失败；
4. 独立 seed 验收为 360/360；
5. 模型实验可执行 best-of-four lookback 选择。
