# Paper v5 E3：机制保真能力画像小试验

日期：2026-07-20

## 结论

Paper v5 E3 已从“只按 MASE 形成能力画像”扩展为并列报告：

- 点预测 MASE；
- 逐能力机制保真分；
- I1--I5 paired dose-response；
- 以 naive MASE 作安全门的 ability score；
- MASE、mechanism 和 ability 三套正式排名。

本次只做实现和方向小试验，不形成正式模型能力结论。两模型、每格 8 个 paired
groups 的结果已经显示机制分不是 MASE 的换皮：18 个 `model × capability` profiles
中，covariate 两行因尚无配对消融而只作诊断；其余 16 行均能产生正式机制分。

## 输入与范围

- Synthetic source：sealed paper-v5 E2 samples 与逐 view predictions；
- Context policy：沿用 E2 的逐样本最优 MASE context，同一 forecast 同时用于 MASE
  与机制评价；
- 模型：`Chronos-2`、`tabpfn-ts3`；
- 每个 `dataset × capability`：8 个 paired groups × 5 intensities；
- 总计：720 条逐模型机制评分；
- 单变量：`gift_ett1_h` 的六项能力；
- 结构化能力：
  - `electricity_hourly_panel/common_factor`；
  - `m5_daily_hierarchy/hierarchical_coherence`；
  - `gefcom2014_load/covariate_response`。

三项多变量能力不能由同一现有数据集合法承载：普通面板、显式父子层级和已知未来
协变量是不同任务接口。因此每项各用一个兼容数据集。

## 指标受控验收

新增测试首先只使用生成器 oracle 和人为退化预测，不查看 foundation-model 排名来
调指标：

1. 九项能力的 exact future 均获得约 1.0 机制分；
2. constant forecast 在九项能力上的平均机制分均低于 0.5；
3. 只保留 primary period 会丢失 multi-seasonal coverage；
4. fixed carrier 会丢失 time-varying modulation；
5. 平滑 regime 会损失 switch timing/amplitude；
6. 线性背景会漏掉 intermittent pulse；
7. 只破坏 hierarchy parent 会降低 coherence/selectivity；
8. 正序 I1--I5 dose-response 得分为 1，逆序为 0；
9. covariate intact-only 只能产生诊断分，提供 zero-future-covariate 配对 forecast
   后才具备正式资格。

Focused test：`29 passed`。

## Pilot 结果

下表的 MFS 已按 `0.7 × level fidelity + 0.3 × dose response` 汇总。Covariate
response 没有配对消融，虽然保留观察性 projection 分数，但不产生 mechanism/ability
正式名次。

| Dataset | Capability | Model | MASE | Level MFS | Dose | MFS | MASE rank | Mechanism rank |
|---|---|---|---:|---:|---:|---:|---:|---:|
| ETT1 | multi-seasonal | Chronos-2 | 0.4014 | 0.8124 | 0.7889 | 0.8053 | 2 | 1 |
| ETT1 | multi-seasonal | tabpfn-ts3 | 0.3983 | 0.8138 | 0.7714 | 0.8011 | 1 | 2 |
| ETT1 | nonlinear persistence | Chronos-2 | 0.7027 | 0.3409 | 0.6183 | 0.4241 | 1 | 1 |
| ETT1 | nonlinear persistence | tabpfn-ts3 | 0.7155 | 0.3206 | 0.5543 | 0.3907 | 2 | 2 |
| ETT1 | intermittency | Chronos-2 | 0.6252 | 0.6475 | 0.9283 | 0.7318 | 2 | 1 |
| ETT1 | intermittency | tabpfn-ts3 | 0.6024 | 0.5797 | 0.8749 | 0.6683 | 1 | 2 |
| ETT1 | regime switching | Chronos-2 | 0.6259 | 0.3108 | 0.7105 | 0.4307 | 2 | 1 |
| ETT1 | regime switching | tabpfn-ts3 | 0.6247 | 0.2257 | 0.6682 | 0.3584 | 1 | 2 |
| ETT1 | time-varying seasonality | Chronos-2 | 0.4196 | 0.7967 | 0.9616 | 0.8462 | 1 | 2 |
| ETT1 | time-varying seasonality | tabpfn-ts3 | 0.4230 | 0.8200 | 0.9723 | 0.8657 | 2 | 1 |
| ETT1 | trend | Chronos-2 | 0.7462 | 0.3195 | 0.6753 | 0.4262 | 1 | 1 |
| ETT1 | trend | tabpfn-ts3 | 0.7682 | 0.3438 | 0.6059 | 0.4225 | 2 | 2 |
| Electricity | common factor | Chronos-2 | 0.6047 | 0.5777 | 0.7474 | 0.6286 | 1 | 1 |
| Electricity | common factor | tabpfn-ts3 | 0.7047 | 0.4636 | 0.6132 | 0.5085 | 2 | 2 |
| M5 | hierarchy | Chronos-2 | 1.2102 | 0.2100 | 0.8909 | 0.4143 | 1 | 1 |
| M5 | hierarchy | tabpfn-ts3 | 1.2823 | 0.2088 | 0.8205 | 0.3923 | 2 | 2 |

### MASE 与机制分提供了不同信息

在 ETT1 的 multi-seasonal、intermittency、regime switching 和
time-varying seasonality 上，两种排名方向不同。例如：

- regime：tabpfn MASE 略低，但 Chronos 的 switch/dose 机制分更高；
- intermittency：tabpfn 点误差更低，但 Chronos 对 pulse 与强度路径的恢复更好；
- time-varying seasonality：Chronos MASE 更低，tabpfn 的 modulation fidelity
  略高。

这支持继续并列保留三套结果，不能用 MASE rank 替代 mechanism rank，也不能只选择
更有利的一套排名。

### 层级结果揭示了 coherence 之外的问题

两模型的 hierarchy level MFS 只有约 0.21。逐样本诊断显示 forecast coherence
通常较高，但 child zero-sum contrast 的动态路径恢复较弱。全零或同形预测不会因为
满足加总关系就获得高分；child heterogeneity 是必要的 anti-shortcut guard。

### Covariate 仍未完成正式评价

观察性 projection 对 Chronos/tabpfn 分别得到约 0.814/0.831 的综合诊断分，但这不能
证明模型使用了 future covariates。Runner 已支持读取
`future_covariates_zero` 配对预测；完成同 history、同 context 的小规模补推后才进入
正式机制排名。

## 产物与后续

- 协议：
  `docs/superpowers/specs/2026-07-20-paper-v5-e3-mechanism-fidelity-protocol.md`
- 机制评分器：
  `backend/app/services/synthetic_mechanism_fidelity.py`
- Runner：
  `scripts/analyze_paper_v5_e3_mechanism_fidelity.py`
- Pilot runtime：
  `runtime/paper_exp/v5/E3_mechanism_fidelity_pilot/`

下一步不是扩成全模型大实验，而是：

1. 补 GEFCom future-covariate paired ablation；
2. 增加 timing shift、amplitude shrink、false-event 等分项退化测试；
3. 用两个独立 N=160 seed banks 检查 MFS 与 dose score 的可靠性；
4. 冻结 CI/tie-aware 规则后，再运行正式 E3。
