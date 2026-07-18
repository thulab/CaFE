# Paper v4：多 lookback、真实数据锚定的 profile 协议

日期：2026-07-18

## 1. 研究目标与固定 shape

本协议服务于能力维度 benchmark，而不是长上下文 scaling 研究。预测长度固定为
`H=48`，模型获得四个等权候选 lookback：

```text
L ∈ {96, 168, 336, 504}
```

最终允许每个模型按预注册主指标选择一个 best-of-four lookback。数据集不作为主结果轴；
它只负责约束合成样本的真实 nuisance/support 分布，并作为来源域稳健性分层。

## 2. 为什么四档 L 必须共享母样本

若分别在 96、168、336、504 上独立生成样本或重新定义五档 intensity，lookback 比较会同时
改变样本、future 与难度标尺。正式 profile 因此先选取一条 `504+48` 的真实母窗口，再取
其末尾历史形成四个 nested view：

```text
master: [---------------- context 504 ----------------][future 48]
L=504:  [---------------- context 504 ----------------][future 48]
L=336:                    [--------- context 336 -------][future 48]
L=168:                                      [context 168][future 48]
L=96:                                             [ctx 96][future 48]
```

同一 `master_window_id` 的四个 view 必须拥有完全相同的 raw future、来源 series 和
forecast origin。后续合成数据也应生成 `504+48` 母样本，再暴露四个 suffix context；
不得为四个 L 独立抽 seed。

Canonical intensity 只冻结一次，不能按 L 重定义。各长度 profile 只描述
length-conditional nuisance、control-feature support 与 near-distance reference。

## 3. 数据集选择

### 3.1 纳入标准

1. 来自公开的 GIFT-Eval 或其收录的 Monash M4 Hourly；
2. 小时频率，主季节周期统一为 24，避免把频率差异误记为能力差异；
3. 在排除官方 test tail 和相邻 validation horizon 后，仍支持 `504+48`；
4. 可按单变量或 GIFT-Eval 官方 `to_univariate` 语义拆 channel；
5. 选择只依据来源、频率、长度和领域覆盖，不读取任何模型成绩。

### 3.2 正式来源集合

| source config | family | domain |
|---|---|---|
| M4 Hourly | M4 Hourly | Econ/Fin |
| Electricity/H | Electricity | Energy |
| Solar/H | Solar | Energy |
| ETT1/H、ETT2/H | ETT | Energy |
| Jena Weather/H | Jena Weather | Nature |
| KDD Cup 2018/H | KDD Cup 2018 | Nature |
| Loop Seattle/H | Loop Seattle | Transport |
| SZ-Taxi/H | SZ-Taxi | Transport |
| M_DENSE/H | M_DENSE | Transport |
| Bitbrains Fast Storage/H、Bitbrains RND/H | Bitbrains | Web/CloudOps |
| BizITObs L2C/H | BizITObs L2C | Web/CloudOps |

总计 13 个 source config、11 个独立 family、5 个领域。ETT1/2 与两套 Bitbrains 保留为
同 family 内部的来源变体，不获得双倍 family 权重。

Daily、weekly、monthly 版本不进入本轮。它们需要分别冻结 season length、H 和特征估计器，
不能在同一主表中与 hourly profile 直接混合。

## 4. 时间隔离与窗口选择

GIFT-Eval 来源按官方 short-term 规则计算 test tail，profile 只能读取：

```text
series[: series_length - official_test_tail - 48]
```

最后额外删除的 48 点作为 validation embargo。M4 Hourly 的 Monash TSF 只包含训练历史，
因此删除末尾 48 点作为内部 validation embargo。

每个 source config 最多等距选择 240 个母窗口。缺失值处理在每个 nested view 内独立执行：

- 至少 50% observed 且至少两个有限值；
- 只在该 view 内线性插值，并最近值填充两端；
- 任一 L view 不合格，则整条母窗口从四个 L 同时删除；
- 任一 context 近常数，则四个 L 同时删除。

该 complete-case pairing 防止不同 L 获得不同难度的样本集合。

## 5. Profile 内容

每个 `source config × L` 形成一个 provenance profile，同时形成四个
family-macro reference profile。profile 至少保存：

- source、family、domain、series/channel、forecast origin；
- `master_window_id` 与 raw future SHA-256；
- context、horizon、period、observed fraction；
- full `L+48` realized-feature 分位数；
- canonical measurement `L+24` realized-feature 分位数；
- source asset hash、时间截止规则、候选数、拒绝数与有效 series 数。

数据集名称不会提供给模型，也不决定合成样本标签。

## 6. 家族平衡聚合与合成采样

全局 profile 不对所有窗口做 micro average。权重按三级分配：

```text
family 等权
  -> family 内 source config 等权
    -> source config 内 paired master window 等权
```

因此 ETT1/2 合计只占一个 family，Bitbrains 两套配置合计也只占一个 family。正式合成采样
沿用同一层级：先均匀选 family，再选 source config，最后选 profile/window nuisance。
最终论文能力分数也应先在来源内汇总，再做 family macro。

## 7. 与旧 paper-v2/v3 的关系

旧实验固定 `504/48/24`，且 dataset-local synthetic predictor 的方差较大。v4 不修改或
覆盖已封存 artifact，而是新增多 lookback profile suite：

- 旧结果只作为方法开发证据；
- v4 的数据集集合和 weighting 在新模型推理前冻结；
- v4 不复用 paper-v2 的 504 profile 冒充 96/168/336 profile；
- 未来真实迁移确认集必须使用新增、未查看模型成绩的真实 family，不能再把本 profile
  corpus 声称为完全独立的 confirmatory set。

## 8. 产物

构建命令：

```bash
cd backend
uv run python ../scripts/build_paper_v4_profile_suite.py
```

默认输出：

```text
runtime/paper_exp/v4/00_profile_suite/
  profile_suite.json
  profile_rows.csv
  source_inventory.csv
  report.md
  manifest.json
```

`manifest.json` 封存协议、脚本、来源资产和输出哈希。默认目录一旦存在 manifest，构建器
拒绝覆盖。
