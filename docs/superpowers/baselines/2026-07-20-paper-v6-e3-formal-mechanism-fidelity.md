# Paper v6 E3：正式机制保真能力画像结果

日期：2026-07-20

## 结论

Paper v6 E3 已按正式协议完成。普通九能力预测复用 sealed v6 E2 的逐样本
oracle-context forecast；另在三台服务上补完 4 个兼容模型的 future-covariate 配对
消融。最终形成：

- 48,000 条逐样本机制评分；
- 60 个正式 `model × capability` 画像；
- 每个画像 160 个 paired groups × I1–I5；
- MASE、MFS、Ability 三套 point rank、95% bootstrap CI 与 tie-aware 模型对；
- 前 80 / 后 80 paired groups 的独立可靠性审计。

本次最重要的统计结论不是“所有精确名次都不变”，而是：

1. 80/80 间 tie-aware 模型对没有一例明确方向冲突；
2. 模型对状态完全一致率为 0.8262，point pair 方向一致率为 0.8373；
3. 9 个能力中有 6 个能力的 MASE 第一名与 MFS 第一名不同，机制分提供了点误差之外
   的信息；
4. 对差异很小的模型，point rank 可以变化，但正式模型对往往是 equivalent 或
   unresolved，不能强行解释为严格胜负。

## 输入与执行

正式协议：

`docs/superpowers/specs/2026-07-20-paper-v6-e3-formal-mechanism-fidelity-protocol.md`

普通预测来源：

`runtime/paper_exp/v6/E2_dynamic_stability/`

协变量补推：

| 模型 | 服务 | 完成 | 失败 |
|---|---|---:|---:|
| Chronos-2 | `192.168.99.17:10811` | 800/800 | 0 |
| tirex2 | `192.168.99.17:10811` | 800/800 | 0 |
| timesfm2.5 | `192.168.99.18:10810` | 800/800 | 0 |
| tabpfn-ts3 | `127.0.0.1:10810` | 800/800 | 0 |

补推只把标准化后的 known-future covariates 置零，history targets、history
covariates、模型、horizon 与各模型 E2 oracle context 均保持不变。Manifest 已验证
4 个文件各 800 行、master sample 覆盖完整且 context 一致。

模型兼容范围：

- 六项单变量能力：全部八模型；
- common factor 与 hierarchy：Chronos-2、toto2.0、tirex2、tabpfn-ts3；
- covariate response：Chronos-2、timesfm2.5、tirex2、tabpfn-ts3；
- 其余 12 个 `model × capability` 组合保持 N/A，不进入排名。

## 正式能力画像

下表列出每个能力的 point-estimate 第一名。Ability 的点误差安全门在本次 60 个画像
上全部为 1，因为所有模型的 capability-level MASE 都优于 naive。因此本次
Ability 与 MFS 数值和排名相同；这是预注册安全门的实际结果，不能在看到成绩后修改
公式制造额外区分。

| Capability | MASE 第一 | MASE | MFS 第一 | MFS | 第一名是否相同 |
|---|---|---:|---|---:|---|
| trend | timesfm2.5 | 0.7630 | Timer-3.5 | 0.4053 | 否 |
| multi-seasonal | toto2.0 | 0.3846 | tabpfn-ts3 | 0.8815 | 否 |
| time-varying seasonality | toto2.0 | 0.4089 | tirex2 | 0.8873 | 否 |
| regime switching | timesfm2.5 | 0.6552 | timesfm2.5 | 0.4478 | 是 |
| nonlinear persistence | Timer-3.5 | 0.7289 | Timer-3.0 | 0.5268 | 否 |
| predictable intermittency | toto2.0 | 0.5768 | toto2.0 | 0.8324 | 是 |
| common factor | Chronos-2 | 0.6681 | toto2.0 | 0.5047 | 否 |
| hierarchical coherence | Chronos-2 | 1.1230 | toto2.0 | 0.4630 | 否 |
| covariate response | timesfm2.5 | 0.3180 | timesfm2.5 | 0.8227 | 是 |

这里的“第一”仍是直观 point rank，是否足以形成胜负结论必须查看 paired-bootstrap
模型对。

### 机制难度差异

各能力跨兼容模型平均 MFS：

| Capability | 平均 level MFS | 平均 dose | 平均 MFS |
|---|---:|---:|---:|
| time-varying seasonality | 0.8364 | 0.9856 | 0.8812 |
| multi-seasonal | 0.8326 | 0.9501 | 0.8679 |
| covariate response | 0.8070 | 0.7823 | 0.7996 |
| predictable intermittency | 0.4263 | 0.8294 | 0.5472 |
| nonlinear persistence | 0.4106 | 0.7279 | 0.5058 |
| common factor | 0.4745 | 0.5432 | 0.4951 |
| hierarchical coherence | 0.2420 | 0.9011 | 0.4397 |
| regime switching | 0.2721 | 0.7064 | 0.4024 |
| trend | 0.2964 | 0.5417 | 0.3700 |

Hierarchy 的 dose 较高但 level fidelity 低，说明模型较能保持强度相对变化，却普遍
不能充分恢复 child contrast/heterogeneity。Regime 与 trend 也表现为 dose 关系优于
逐样本机制路径恢复。

### Covariate response

配对消融完成后，四模型均获得正式分：

| 模型 | MASE | Detection | Timing | Magnitude | Selectivity | MFS |
|---|---:|---:|---:|---:|---:|---:|
| timesfm2.5 | 0.3180 | 0.9744 | 0.9611 | 0.8066 | 0.6984 | 0.8227 |
| Chronos-2 | 0.3392 | 0.9300 | 0.9285 | 0.7716 | 0.7387 | 0.8100 |
| tabpfn-ts3 | 0.3637 | 0.9494 | 0.9352 | 0.7423 | 0.7101 | 0.7979 |
| tirex2 | 0.3590 | 0.7838 | 0.8716 | 0.7850 | 0.7113 | 0.7676 |

主 0.02 等价阈值下，timesfm2.5 与 Chronos-2、tabpfn-ts3 的 MFS 差异仍为
unresolved；timesfm2.5 和 Chronos-2 分别明确优于 tirex2。不能把 point rank
1–4 解释成四个都已严格区分。

## Tie-aware 正式比较

2,000 次 paired bootstrap、95% CI、主等价阈值 0.02 的模型对状态：

| Metric | Left better | Right better | Equivalent | Unresolved | 总数 |
|---|---:|---:|---:|---:|---:|
| MASE | 26 | 38 | 50 | 72 | 186 |
| MFS | 18 | 35 | 49 | 84 | 186 |
| Ability | 18 | 35 | 49 | 84 | 186 |

两个特别能说明“并列”必要性的例子：

- time-varying seasonality 的 28 个 MFS 模型对全部为 equivalent。虽然 point rank
  可以列出 1–8，但现有精度下不应宣称这些模型在该机制上严格有序；
- nonlinear persistence 的 28 个 MASE 模型对中 25 个 equivalent、3 个
  unresolved，没有一个达到明确胜负。Timer-3.5 的 point MASE rank 1 只是描述性
  次序。

Common factor 的 6 个 MFS 模型对全部 unresolved，说明当前 160 组虽能估计连续
分数，但还不能据此选出确定胜者。

## 80/80 可靠性

总体结果：

| 指标 | 数值 |
|---|---:|
| 平均 rank Spearman | 0.7721 |
| point pair 方向一致率 | 0.8373 |
| top-model 完全一致率 | 0.6296 |
| tie-aware pair state 完全一致率 | 0.8262 |
| tie-aware 无方向冲突率 | 1.0000 |
| 明确方向冲突数 | 0 |

分 metric：

| Metric | 平均 rank Spearman | point pair 方向一致率 | top-model 一致率 |
|---|---:|---:|---:|
| MASE | 0.8434 | 0.8611 | 0.5556 |
| MFS | 0.7365 | 0.8254 | 0.6667 |
| Ability | 0.7365 | 0.8254 | 0.6667 |

精确 rank vector 只有 3/27 完全一致，不能继续把“完整名次完全相同”当可靠性标准。
但两个 half-bank 没有任何已经明确判为 A 胜 B、在另一半又明确反向的模型对。结合
0.8262 的状态一致率，这支持把正式结果表述为连续分数 + point rank + tie-aware
关系，而不是强制唯一全序。

较低的 point-rank 相关主要集中在：

- time-varying seasonality MFS/Ability：rho 0.2619，但全部模型对 equivalent；
- common factor MFS/Ability：rho 0.4000，但全部模型对 unresolved；
- regime switching MFS/Ability：rho 0.5714，top model 在两个 half 间变化。

前两项恰好说明，精确 rank 不稳定不等于科学结论矛盾；更合理的结论是模型之间尚未
形成可分辨差异。

## 完整性与验收

正式产物审计：

- 48,000/48,000 sample rows；
- 每个正式画像 800 条样本、160 个 paired groups、每组五档；
- 60/60 profiles 均为 `formal_score_eligible=true`；
- 180/180 bootstrap intervals 均包含对应 point estimate；
- 4 个协变量消融文件共 3,200 行，失败为 0；
- source E2 sample manifest、inference manifest、ablation manifest 与四个消融
  JSONL 的 SHA-256 已写入正式 manifest。

机制指标的受控验收已覆盖 exact future、constant、timing shift、amplitude shrink、
false event、nonlinear blind、channel desynchronization、hierarchy violation、
covariate no-response 与 dose 顺逆序。

## 产物

正式结果：

`runtime/paper_exp/v6/E3_mechanism_fidelity/formal_analysis/`

关键文件：

- `report.md`：完整 60-profile 表；
- `capability_profiles.csv`：正式连续分数、CI 与 point ranks；
- `capability_pair_states.csv`：全部模型对、三个阈值和四状态；
- `split_half_reliability.csv`：逐能力、逐 metric 的 80/80 可靠性；
- `split_half_pair_state_comparison.csv`：两个 half-bank 的模型对状态；
- `summary.json` 与 `manifest.json`：冻结配置、输入身份和输出哈希。

协变量消融：

`runtime/paper_exp/v6/E3_mechanism_fidelity/covariate_ablation_predictions/`

复现命令：

```bash
uv run --project backend python \
  scripts/run_paper_v6_e3_covariate_ablation.py \
  --finalize-only

uv run --project backend python \
  scripts/analyze_paper_v5_e3_mechanism_fidelity.py \
  --covariate-ablation-predictions-dir \
  runtime/paper_exp/v6/E3_mechanism_fidelity/covariate_ablation_predictions \
  --overwrite
```
