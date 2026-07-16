# E2：Dynamic evaluation stability 实验协议

日期：2026-07-16；推理执行配置冻结于 2026-07-17

## 目的与边界

E2 检验动态生成一批新 probe 时，模型的能力分数和相对排名是否对生成轮次稳定。这里的稳定指同一 `profile × capability × intensity` 下更换独立生成 seed 后，基于一批样本得到的统计结论可复现；不要求不同随机样本上的逐条预测相同。

E2 不用于证明合成能力对真实数据的外部效度，也不把一次实验中的绝对排名当作论文最终排名。前者由后续合成—真实对应实验验证，后者由完整 benchmark 实验给出。E2 只回答动态评测是否具有足够低的 Monte Carlo 波动。

正式输出固定保存在 `runtime/paper_exp/v1/E2_dynamic_stability/`。runner 支持按模型追加成功预测和 `--resume` 断点续跑，但完成分析并写入 `manifest.json` 后目录封存，不再覆盖。

## 冻结生成设计

- generator conditioning：`synthetic-v2-paper-v1-frozen-2026-07-16`，fingerprint `a76b66924562be4f`。
- 在线集合：8 个 profile、23 个 `profile × capability` cells；仅运行 profile 正式支持的 capability。
- intensity：1--5 全部运行。
- 五轮独立根 seed：`2026071621`、`2026071622`、`2026071623`、`2026071624`、`2026071625`。
- 每轮每个 `profile × capability × intensity` 32 条样本，共 `23 × 5 × 5 × 32 = 18400` 条。
- 同一 `profile × capability × round × sample_index` 的五档 intensity 使用同一个 base sample seed；所有模型接收完全相同的生成样本。因而模型比较和 intensity 比较均为配对设计。
- 不同 round 的根 seed 不重用；所有样本仍经过 paper-v1 construction predictability、feature support 和 near-distance 门控。

## 冻结模型集合与兼容矩阵

基础模型固定为：`Timer-3.5`、`Timer-3.0`、`Chronos-2`、`moirai2`、`toto2.0`、`timesfm2.5`、`tirex2`。`tabpfn-ts3` 因当前服务路径无法形成可扩展并发、正式运行成本与其余模型不在同一量级，不进入 paper-v1 E2。另运行 `naive` 与 `seasonal_naive` 作为诊断基线。正式运行保存推理服务的 model catalog 快照，并按其中的输入长度、预测长度、目标数和 known-future covariate 限制决定兼容性；不把不兼容任务记为失败或最差分数。

- 单变量 capabilities：7 个基础模型共同排名。
- `common_factor`、`hierarchical_coherence`：`Chronos-2`、`toto2.0`、`tirex2` 共同排名。
- `covariate_response`：`Chronos-2`、`timesfm2.5`、`tirex2` 共同排名。

按当前 catalog，预计基础模型预测共 112800 条，两个基线共 36800 条。Kendall τ、CV、ICC 和置信区间的主要判据只作用于基础模型；包含两个基线的排名另行保留作诊断，避免基线改变大模型排名稳定性的判定。

## 冻结推理执行配置

执行配置来自 timer-rest-service 双卡多 replica 实测报告 `20260716T182346Z-replicas/REPLICA_OPTIMIZATION_ZH.md`（SHA-256 `cd16830aa17985e1c45701aa6a56454b7a42bec85e1a724ca51563e319cfec46`）。两张 GPU 固定为 `devices="0,1"`。每个独立样本构造一个 HTTP 请求，`targets` 中恰有一条 task；多变量样本仍是该 target 的多列，不拆成多个请求。

| 模型 | replica/卡 | 双卡总 worker | 全局 HTTP 并发 |
| --- | ---: | ---: | ---: |
| `Timer-3.5` | 1 | 2 | 64 |
| `Timer-3.0` | 1 | 2 | 32 |
| `Chronos-2` | 4 | 8 | 32 |
| `moirai2` | 2 | 4 | 16 |
| `toto2.0` | 2 | 4 | 16 |
| `timesfm2.5` | 8 | 16 | 32 |
| `tirex2` | 1 | 2 | 32 |

runner 每次只加载一个基础模型，并在推理前通过 `models/list_loaded` 核验每张卡的 endpoint 数、总 worker 数和 PID 唯一性。兼容样本按完整 signature 分成五个桶：`168×1→24`、`168×3→24`、`365×3→28`、`168×1+2cov→24`、`365×1+2cov→28`；一个桶全部完成后才进入下一个桶，不在 shape 间轮转。并发额度是双卡全局值。单请求最多尝试三次；失败不会产生预测分数，恢复运行只补没有成功落盘的 sample id。

## 指标与统计单位

主要误差指标为逐样本 seasonal MASE，分母由该 profile 冻结的 `season_length` 在历史窗口内计算；MAE 为辅助指标。最小分析单元为一个 `model × profile × capability × intensity × round`，其 round score 是 32 条样本 MASE 的算术均值。

### E2.1 分数轮次变异

对每个 `model × profile × capability × intensity` 的五个 round score 计算样本标准差和 `CV = std / |mean|`。汇总基础模型全部 cells 的中位数、p90、p95 和最大值，并分模型报告。

操作性稳定标准：CV 中位数不超过 0.10，p95 不超过 0.25。该阈值衡量 32 条/轮是否足以支持后续实验，而不是对生成分布相同的统计检验。

### E2.2 ICC 绝对一致性

使用 two-way absolute-agreement single-measure `ICC(A,1)`：

- 主要统计以某个模型兼容的 `profile × capability × intensity` cells 为 subjects、五轮为 raters，衡量该模型跨能力表现轮廓的可复现性；逐模型报告，所有基础模型的最低值须不低于 0.90。
- 另以单个 cell 中的兼容模型为 subjects、五轮为 raters，输出模型排名分值的 absolute-agreement ICC，仅作诊断。

### E2.3 模型排名稳定性

在每个 `profile × capability × intensity` 内，以 MASE 越低排名越高，计算五轮两两组合的 10 个 Kendall τ-b；τ-b 显式处理并列。逐 cell 报告均值、最小值和 p10，再跨 cell 汇总。

基础模型排名的 cell-level 平均 τ 中位数须不低于 0.80，p10 须不低于 0.50。两个基线加入后的 `all_predictors` 结果同时输出但不参与判定。

### E2.4 分层 bootstrap 置信区间

对每个 `model × profile × capability × intensity` 做 1000 次分层 bootstrap：先有放回抽取五个 round，再在被抽到的每轮内有放回抽取 32 条样本，得到 pooled mean MASE 的 percentile 95% CI。这样同时纳入轮次间和轮次内波动。

报告相对区间宽度 `(upper - lower) / |mean|`。基础模型 cells 的中位数须不超过 0.20，p95 须不超过 0.50。固定由 cell identity 派生的 bootstrap seed，保证分析重算确定性。

### E2.5 跨轮新颖性

在相同 `profile × capability × intensity` 内，对全部 10 对轮次双向计算完整 target 轨迹的最近邻 MAE（DCR）和最近/次近距离比（NNDR）。同时检查 float64 精确哈希、六位小数哈希和 DCR ≤ `1e-6` 的近重复率。

三种重复率均须为 0。DCR 与 NNDR 分位数用于描述不同生成轮次是否只是复刻同一小组曲线，不设置事后调节的距离阈值。

## 失败、恢复与保留文件

- 单请求推理失败会逐样本写入 `failures/`，不会伪造分数；成功结果持续 append，并在进度点及每个 shape 桶结束时 flush。重跑 `--resume` 只补缺失的 sample id。
- 单模型加载或执行异常会记录到 `model_status.json` 并继续后续模型；只有所有兼容预测完整时才允许统计分析。
- 正式分析保留 `samples.jsonl`、逐模型完整 forecast、model catalog、coverage、round score、CV、bootstrap CI、排名、ICC、跨轮距离、`summary.json`、`report.md` 与包含代码和输入/输出 SHA-256 的 `manifest.json`。
- 上述样本数、模型集合、seed、统计定义和阈值在正式运行前冻结；smoke test 只验证执行正确性，不据结果调整阈值。
