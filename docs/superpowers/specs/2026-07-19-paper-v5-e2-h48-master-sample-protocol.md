# Paper v5 E2：H=48 四 Context 母样本协议

日期：2026-07-19

## 1. 本轮冻结范围

- 仅运行 `H=48`；`H=24` 暂不进入本轮正式实验；
- context 为 `L={96,168,336,504}`；
- intensity 为 dataset-local 五档；
- 生成轮次固定为 5；
- 每轮、每个 `dataset/task/capability/intensity` 固定 32 条样本；
- 每个模型必须接收同一份冻结样本集合；
- 每个模型、每条母样本、每个 context 的 forecast 和全部逐样本指标必须保留。

## 2. 母样本与模型 view

统计上的生成单位是一条 `L=504,H=48` 母样本：

```text
target[0:504]   = master history
target[504:552] = shared future
```

模型输入由母样本派生：

```text
history_L = target[504-L:504]
future    = target[504:552]
L         = 96/168/336/504
```

四个 context 使用完全相同的 48 步 future。它们是同一母样本的四次重复测量，不是
四条独立样本。每个 view 在送入模型和评分前，必须只用自己的 history suffix
重新标准化；不得直接沿用 `L=504` 的均值和尺度。

## 3. 样本数量

eligible cell 定义为正式支持矩阵中 `status=supported` 的
`dataset × task-view × capability`。母样本总数动态计算：

\[
N_{\mathrm{master}}
=N_{\mathrm{eligible}}
\times5_{\mathrm{intensity}}
\times5_{\mathrm{round}}
\times32.
\]

每个母样本产生四个模型请求，因此兼容模型的请求量为：

\[
N_{\mathrm{request}}
=\sum_{\mathrm{compatible\ model-cell}}
5\times5\times32\times4.
\]

unsupported cell 和模型不兼容项不补值、不视为最差表现。

## 4. 五档配对与联合验收

固定 `dataset/profile/capability/round/sample_index` 后，五档使用同一
`sample_seed`。正式生成按 paired group 联合重试：

1. 从共同 `attempt_seed` 生成 I1–I5；
2. 五档共享周期、motif、lag、载荷、协变量路径、背景和噪声；
3. intensity 只改变目标机制强度；
4. 每档分别派生四个 context view；
5. 只有 `5 intensities × 4 contexts` 全部通过 construction、
   dataset-local feature-support 和 near-distance gate，整个 paired group 才接受；
6. 任一档或任一 view 失败时，五档共同进入下一个 attempt。

这样避免 intensity-specific rejection sampling 改变配对样本的 nuisance realization。

## 5. 数据产物

正式目录：

```text
runtime/paper_exp/v5/
  01_nine_capability_suite/
  E2_dynamic_stability/
    generation_config.json
    sample_shards/
    samples.jsonl
    sample_manifest.json
```

每条 `samples.jsonl` 记录至少保存：

- `sample_id/master_sample_id/paired_group_id`；
- dataset、task view、profile、capability、intensity、round、sample index；
- base seed、共同 attempt 和 attempt seed；
- `L=504,H=48` 完整 target 与协变量；
- construction metadata；
- 四个 context 的 realized features 和 gate audit；
- target、future、covariates 的 float64 SHA-256。

分 cell shard 是中断恢复单位；`samples.jsonl` 是所有完整 shard 的确定性拼接。

## 6. 模型结果与 Context 汇总

每个模型对每条母样本分别运行四个 context。原始 prediction row 必须保存：

- master sample id 和 view id；
- context、horizon；
- forecast 与 future truth；
- MASE、MAE 及运行时产生的其他逐样本指标；
- 模型版本、状态、耗时和失败信息。

本轮约定允许在逐母样本层选择该模型表现最好的 context：

\[
Score_{m,s}=\min_{L\in\{96,168,336,504\}}MASE_{m,s,L,H=48}.
\]

该汇总解释为模型在候选 context 中“尽己所能”的 oracle-context 表现。四个原始
context 指标不得删除，因此后续可以在不重新推理的情况下改用固定 `L=504`、context
均值或其他预注册汇总规则。

动态稳定性、模型排名和 bootstrap 均先把四个 view 汇总回母样本，再以母样本为最小
统计单位；不能把四个共享 future 的 view 当作四份独立证据。

## 7. 正式命令

正式校准由 dataset-disjoint shard 构建、纯 union 合并后统一 qualification。分片不
混合任何 dataset 的真实窗口或统计量。

正式母样本生成：

```bash
cd backend
uv run python ../scripts/generate_paper_e2_master_samples.py
```

生成前必须满足：

- calibration suite manifest 完整；
- 所有 supported cells 的统一 qualification 通过；
- generator version 与冻结 calibration suite/manifest 完全一致；下一轮正式扩展
  使用经 ETT1 小试验验证的 `capts-paper-v4`；
- 轮次恰为 5、每轮样本数恰为 32。
