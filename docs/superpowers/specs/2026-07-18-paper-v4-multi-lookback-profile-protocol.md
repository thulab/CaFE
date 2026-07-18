# Paper v4：逐数据集、多 lookback profile 协议

日期：2026-07-18
状态：替代同日发布的 pooled/family-balanced profile 协议

## 1. 研究单位

真实数据与合成测试的对应单位固定为单个 dataset config。ETT1/H、ETT2/H、
Bitbrains Fast/RND 等配置分别视为独立 dataset，不再组成 family，也不共享 profile。

对任意 dataset \(d\)，正式 shape 为：

```text
H = 48
L ∈ {96, 168, 336, 504}
season_length = 24
```

每个 `dataset × L` 独立形成 profile：

```text
{dataset_id}__L{L}_H48
```

系统不生成 pooled、global 或 family-macro profile。某个 dataset 的 nuisance、相对
intensity、feature-support 与 near-distance 校准均不得读取其他 dataset 的窗口。

## 2. 为什么四档 L 共享母样本

先从 dataset 内选择一条 `504+48` 母窗口，再从历史末尾裁出四个 nested view：

```text
master: [---------------- context 504 ----------------][future 48]
L=504:  [---------------- context 504 ----------------][future 48]
L=336:                    [--------- context 336 -------][future 48]
L=168:                                      [context 168][future 48]
L=96:                                             [ctx 96][future 48]
```

同一 `master_window_id` 的四个 view 必须共享 dataset、series/channel、forecast
origin、raw future 和 `future_sha256`。任一 view 因缺失或近常数不合格，整条母窗口
退出。后续合成也必须先生成一条母样本，再暴露四个 suffix context；不得为四个 L
分别抽 seed。

## 3. Dataset 集合与选择规则

### 3.1 纳入标准

1. 来自公开 GIFT-Eval 或 Monash M4 Hourly；
2. 小时频率，主季节周期为 24；
3. 排除真实 test tail 和 validation embargo 后仍支持 `504+48`；
4. 可按单变量或官方 channel-wise univariate 语义提取目标；
5. 选择不读取模型成绩。

当前配置共 13 个 dataset、5 个业务领域：

| dataset | domain |
|---|---|
| M4 Hourly | Econ/Fin |
| Electricity/H | Energy |
| Solar/H | Energy |
| ETT1/H | Energy |
| ETT2/H | Energy |
| Jena Weather/H | Nature |
| KDD Cup 2018/H | Nature |
| Loop Seattle/H | Transport |
| SZ-Taxi/H | Transport |
| M_DENSE/H | Transport |
| Bitbrains Fast Storage/H | Web/CloudOps |
| Bitbrains RND/H | Web/CloudOps |
| BizITObs L2C/H | Web/CloudOps |

这些 dataset 相互独立。数量更多不会改变已有 dataset 的 profile，只会增加新的独立
实验单元。

## 4. 时间隔离与窗口选择

GIFT-Eval dataset 按官方 short-term 规则隔离 test tail，profile 只能读取：

```text
series[: series_length - official_test_tail - 48]
```

紧邻 test tail 的 48 点作为 validation embargo。M4 Hourly 的 Monash TSF 只有训练
历史，因此保留末尾 48 点作为内部 validation embargo。

每个 dataset 最多选择 240 条母窗口。抽样先覆盖不同 series/channel，再在已有
series/channel 内增加时间 origin，避免少数长序列垄断 profile。每个 nested view：

- 至少 50% observed 且至少两个有限值；
- 只在 view 内线性插值，并用最近值填充两端；
- context 必须满足 informative-target 检查；
- 任一 L 失败时四个 L 同时删除。

## 5. Profile 内容

每个 `dataset × L` profile 只汇总该 dataset 的有效窗口，保存：

- `dataset_id`、dataset name、domain；
- context、horizon、season length、window count；
- full `L+48` realized-feature 的 p05/p25/p50/p75/p95；
- fixed measurement `L+24` realized-feature 的
  p05/p25/p50/p75/p95。

逐窗口审计记录额外保存：

- series/channel、forecast origin、dataset cutoff；
- `master_window_id` 与 raw future SHA-256；
- observed fraction 和全部有效 realized features。

dataset inventory 保存资产哈希、频率、时间隔离规则、候选/接受/拒绝数量与有效
series 数。profile 的描述性分位数不是跨 dataset 可比的绝对能力标尺，也不直接等同
于后续五档 intensity target。

## 6. 下游校准边界

后续能力 suite 必须在每个 dataset 内先做独立且可复现的
parameter/reference/calibration 划分，再分别构造：

```text
parameter split
  -> dataset-local nuisance
  -> dataset-local relative intensity targets
  -> generator inverse calibration

reference split
  -> dataset-local feature-support reference
  -> dataset-local near-distance reference

calibration split
  -> dataset-local gate / DCR / NNDR thresholds
```

禁止先混合多个 dataset 再拆分，也禁止一个 dataset 因校准失败而改用另一个 dataset
的 gate 或 nearest-neighbour reference。

五档 intensity 由该 dataset/task 的 L=504 parameter split primary feature 的
`{q10,q30,q50,q70,q90}` 定义，表示 dataset 内的相对弱到相对强；它们不是本 profile
文件中的描述性 `{p05,p25,p50,p75,p95}`。某项能力若真实特征没有足够档间距、变量结构
不满足要求，或生成后无法同时通过单调性和局部 gate，应明确记录为 `unsupported`，
无需强行凑齐九能力或五档。

## 7. 结论边界

该设计允许的主张是：

> 在某个具体真实 dataset 所定义的环境中，模型对某项相对能力强度的表现如何变化。

不同 dataset 的同一 intensity 编号只表示各自分布中的相对位置，不能解释为相同的绝对
feature strength。跨 dataset 汇总时比较的是模型响应方向、标准化斜率、相对性能变化
或排名稳定性，而不是直接平均 realized feature 数值。

## 8. 产物

构建命令：

```bash
cd backend
uv run python ../scripts/build_paper_v4_profile_suite.py
```

可用 `--datasets <dataset_id ...>` 只构建指定 dataset，使用
`--max-windows-per-dataset` 调整每个 dataset 的上限。默认输出：

```text
runtime/paper_exp/v4/00_profile_suite/
  profile_suite.json
  profile_rows.csv
  dataset_inventory.csv
  report.md
  manifest.json
```

`profile_suite.json` 只含 `datasets` 与 dataset-local `profiles`，不含
`sources`、`family_id` 或 `global_profiles`。`manifest.json` 封存协议、builder 与
输出哈希；目录存在 manifest 时构建器拒绝覆盖。

## 9. 与旧产物的关系

旧 `paper_v4_multi_lookback_profile_suite.v1`、`source_inventory.csv` 和
`family_macro__L*_H48` 全部作废，不得被新实验读取。重建必须使用
`paper_v4_dataset_local_multi_lookback_profile_suite.v2`，并同步重建所有依赖旧 pooled
profile 或全局 canonical intensity 的 conditioning、gate、qualification 和推理产物。
