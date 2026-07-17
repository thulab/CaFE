# Paper v2 E4 看结果前选择冻结

日期：2026-07-17

本记录在任何 E4 真实模型请求之前生成。机器可读 receipt 为
[`2026-07-17-paper-v2-e4-selection-freeze.json`](./2026-07-17-paper-v2-e4-selection-freeze.json)。

## 固定范围

- 9 个预先声明的 GIFT-Eval hourly held-out profiles；
- 7 个基础模型，以及 naive / seasonal-naive；
- `context=504`、`horizon=48`、`season_length=24`；
- 每个 profile 最多 600 条，按 official rolling origin 分层等额抽样；
- 4,192 条真实任务；
- train-only canonical coordinate `>=3.0` 的 29 个
  `profile × capability` cells，覆盖 8 个 profiles、6 个 family clusters；
- 10 个由 sealed E3-v2 确定的配对缺陷假设；
- family-macro Kendall tau-b 为确认性主终点，另报连续分数对应性、family bootstrap、
  leave-one-family-out 与 719 个非 identity capability-label 精确置换。

## 精确身份

- Selection manifest SHA-256：
  `ffb4e8d63daa2538e4e06ab9f400d5ad449f479bd8baac0b62d45a9e57b1cd8a`
- Task manifest SHA-256：
  `26a633f8efac79b457286a0830861a7888400b87381761bdd796065fb8f1cd04`
- `tasks.jsonl` SHA-256：
  `d684156e49f7166b792c4fe1c5d472633a0ede35c9848a2c5a9e9ba7e1da4841`
- Synthetic predictors SHA-256：
  `c7d16771a5a5207272b21bf0a8014175d624b86855252c573e85c40280162bc2`
- Qualified cells SHA-256：
  `be542947966028206686d3d070fd2b5d292e929b93676c8a78e0cb3ad84006d0`
- Pair hypotheses SHA-256：
  `926c28e073b5c819b9df9abd2face3c2c2917eb644fad0087803c587861f85e7`

`nonlinear_persistence` 在九个 held-out profiles 的 train-only coordinate 均为 1，
因此本轮没有真实 high-loading support，预先标记为不可检验，而不是在看到模型结果后
删去。`M_DENSE` 没有能力达到 coordinate 3，保留为低 loading 真实对照，但不进入
high-loading 主终点。

推理脚本会校验 receipt 已被 Git 跟踪、receipt/protocol/runner 相对 `HEAD` 无改动，且
上述 runtime artifacts 的哈希完全一致；任一条件不满足都会在模型加载前 fail closed。
