# Paper v6 E2 正式重跑计划

日期：2026-07-20

## 1. 目的与版本边界

本轮在九能力生成机制更新并简化 capability 内部子模式后，重新执行逐数据集校准、
正式 master sample 生成、八模型推理和真实 source-window 对照。所有新产物隔离到：

`runtime/paper_exp/v6/`

生成器冻结为 `capts-paper-v4`。v5 的 calibration 和 synthetic samples 不复用。
文件中的 `paper_v5_*` schema 名称表示沿用的文件格式版本，不表示复用 v5 实验数据。

本轮按用户指定数量生成一个 N=160 seed bank（Bank A）。它可用于能力分数、正式排名、
真实对齐和单 bank 不确定性分析；按
`2026-07-20-paper-v5-e2-seed-bank-reliability-protocol.md`，跨完整测试套件的可靠性结论
仍需同生成器、同校准、不同 seed 的独立 N=160 Bank B。

## 2. 冻结统计单位

- Horizon：`48`。
- Context views：`96/168/336/504`。
- 每条 master：`L=504,H=48`，四个 context 使用同一个 future。
- Intensity：dataset-local `I1–I5`，同一 paired group 五档共享 seed。
- 每个 supported `dataset × task × capability × intensity`：`160` 条 master。
- 原 `5 rounds × 32` 仅用于可恢复批次和小样本敏感性；主分数汇总全部 160 条。
- 每个模型、每条 master 保留四个 context 的预测与指标；主分析逐模型、逐样本选择
  MASE 最低 context，固定 `L=504` 作为敏感性分析。
- unsupported capability 如实保留在 support matrix，不生成伪样本。

## 3. v6 校准与样本规模

20 个 dataset/task views 分四个 dataset-disjoint build shards 并行校准，合并过程只做
artifact union，不汇总跨数据集统计。合并后的 suite 使用 `max_attempts=512` 对每个
supported cell、每档 8 条样本做四 context qualification。

正式冻结结果：

- supported cells：`64`；
- qualification：`2,560/2,560`，失败 `0`；
- paired groups：`64 × 160 = 10,240`；
- master samples：`64 × 160 × 5 = 51,200`；
- 每模型最大 synthetic views：`51,200 × 4 = 204,800`；
- real source masters/views：`417/1,668`。

能力 cell 数：

| capability | supported cells |
|---|---:|
| trend | 10 |
| multi_seasonal | 10 |
| time_varying_seasonality | 9 |
| regime_switching | 10 |
| nonlinear_persistence | 9 |
| predictable_intermittency | 10 |
| common_factor | 3 |
| hierarchical_coherence | 1 |
| covariate_response | 2 |

master 生成采用 `prepare → 4 个 dataset-disjoint shards → finalize`。finalize
重新读取全部 64 个 shard，验证每个 cell/intensity 恰有 160 条、每个 group 具有
完整 I1–I5、所有四个 context gate 均通过，再按确定顺序拼接总文件。

## 4. 三机推理分配

三台服务均使用 GPU `0,1` 和冻结的模型 replica/HTTP concurrency。按 v5 正式实验
的实测耗时做模型级 longest-processing-time 平衡，同一服务一次只加载一个模型：

| service | 顺序 |
|---|---|
| `127.0.0.1:10810` | `timesfm2.5 → moirai2` |
| `192.168.99.18:10810` | `tabpfn-ts3 → Timer-3.0` |
| `192.168.99.17:10811` | `toto2.0 → Chronos-2 → tirex2 → Timer-3.5` |

远端请求显式设置 `NO_PROXY/no_proxy`。三条队列写入独立 shard 目录，避免 config、
status、catalog 和 append-only prediction 文件并发覆盖。完成后使用
`scripts/merge_paper_v6_e2_inference_shards.py`：

1. 验证三条 shard 的 synthetic/real 输入身份完全相同；
2. 验证八模型分配互斥且覆盖完整；
3. 验证每个 status 为 complete，预测 JSONL 行数等于 compatible count；
4. 核对 SHA-256 后优先硬链接到主结果目录，跨文件系统才复制；
5. 合并 status，并保留服务、执行配置、失败记录和来源路径。

两个诊断基线 `naive/seasonal_naive` 在主目录直接计算，不进入 foundation-model
headline 排名。

## 5. 完整性判定

推理完成至少满足：

- 八个 foundation models 均为 complete；
- 每个模型 synthetic/real 的 succeeded count 等于 compatible count；
- 所有请求失败 JSONL 的总行数被报告；若存在重试失败，不删除；
- synthetic prediction 总数与模型能力兼容矩阵一致；
- real source 总数为 `8 × 1,668 = 13,344`；
- 合并后重新运行正式 E2 分析，旧 round 指标只解释为 N=32 batch sensitivity；
- 运行相关测试并保存生成、推理和分析的 manifest/hash。
