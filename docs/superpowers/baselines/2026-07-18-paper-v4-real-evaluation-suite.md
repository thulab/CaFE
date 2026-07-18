# Paper v4：Dataset-local Real Evaluation Suite Smoke

日期：2026-07-18
状态：真实 ETT1 smoke 通过；正式多 dataset freeze 待运行

## 目的

该 suite 为 E2 的 synthetic–real ranking alignment 提供与 profile/gate development
窗口完全隔离的真实预测样本。

## Smoke 配置

```text
dataset = gift_ett1_h
task = univariate
H = 48
L = 96/168/336/504
max master samples = 2
```

ETT1/H 的 official short-term test tail 为 960 点。profile/gate 读取在该 tail 之前
停止；real evaluation origins 全部位于 test tail 内。

## 结果

- dataset support：`supported`
- 母样本：2
- 输出 view：8
- lookback：4 档全部存在
- 每个母样本的四个 view：只有一个 `future_sha256`
- future：48 点全观测
- development/evaluation 边界检查：通过
- 结构化 task views：该 smoke 未构造，也未冒充支持

输出：

```text
runtime/paper_exp/v4/02_real_evaluation_suite/
```

哈希：

```text
config.json           4d6b33902e6f4ea1e69644daa1d8442e6bd34d00b5aab276094bd559acc7b510
dataset_support.json  df9746f48a1af87bd16a3e930ee9b4bdede34a6c5b474276b88f00c75fbe26bb
real_samples.jsonl    3b1a65787bd6b3b0d22081563a9f078f8a7a52f02d0f2da2e35e4f115a109acc
manifest.json         8f990df902cdcd600ac5916ce2470717711bd8ae2513d3fbd1931b65ab4af2a9
```

该 smoke 只验证数据隔离、四 lookback pairing 和产物契约，不用于报告模型排名一致性。
