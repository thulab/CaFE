# Paper v4 E1：逐 Dataset 方法有效性 Smoke

日期：2026-07-18

状态：ETT1 端到端 smoke 通过；正式多 dataset、每轮 64 样本的 freeze 待运行。

## 输入与配置

E1 直接读取：

```text
runtime/paper_exp/v4/01_nine_capability_suite/
  generator_conditioning_artifact.json
  feature_gate_artifact.json
  near_distance_artifact.json
  dataset_capability_support_matrix.json
```

本次 smoke 只包含 `gift_ett1_h`。固定 `L=504,H=48` master task，两个独立 round，
每个 supported `dataset/profile/capability` 每轮每档 8 个样本。ETT1 的完整九能力矩阵
为 3 个 supported、6 个 unsupported，因此总样本数为：

\[
3\times 5\times 8\times 2=240.
\]

unsupported cells 仍完整写入
`dataset_capability_support_matrix.csv`，没有填零或计作实验失败。

## 结果

- 预注册判据：`8/8` 通过；
- dataset-local dose-response：`3/3`；
- control selectivity：`5/5`；
- construction predictability：`3/3`；
- feature-support 首轮通过：`15/15`，总体首轮通过率 `1.0`，最大 attempts `1`；
- dataset-local DCR/NNDR：`15/15`；
- 两轮重复检查：`15/15`；
- capability oracle：`3/3`；
- MMD/SWD：2 个有合法 nuisance controls 的能力均比 shifted negative 更接近自身
  real reference；1 个无合法 control features 的能力明确记录为
  `not_applicable_no_control_features`。

该 smoke 证明 E1 已经不再读取后端通用 168/24 artifact，也不从原始数据重建真实 split；
生成、feature gate、near-distance 和分布对照均来自同一个 dataset-local Paper v4
suite。

## Smoke 哈希

```text
config.json
  3e343c79b47deddbbd3fda2cf04b9cbc88730ae41df680f6a98930975b70c46a
dataset_capability_support_matrix.csv
  733098162060308b441e4e1438c56d6307e6a0dd45d7f041e967c3dcd3e74551
summary.json
  d22382803867c790e76b7edbf14dff7414034f73eee7e82abf00cdfe9ff68f70
report.md
  71597711636af9a898160854bab4257df146f1f88e7dc56105e7a28b889aada3
manifest.json
  0d470fbb86f528484829071832d3a1e80e55c55a9805e2d241a3af476d53ddd7
```

协议：

`docs/superpowers/specs/2026-07-16-paper-e1-method-validity-protocol.md`
