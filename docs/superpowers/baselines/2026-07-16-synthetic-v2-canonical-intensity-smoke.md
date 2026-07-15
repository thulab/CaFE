# Synthetic v2 capability-global canonical intensity smoke

日期：2026-07-16

## 目的

验证 `intensity=1..5` 不再表示各 bucket 自己的局部分位，而是同一 capability 内跨 bucket 共享的 canonical realized-strength target；每个 bucket 只拟合 nuisance、结构尺度和到统一目标的单调逆映射。

## 标尺与标定协议

- scale id：`synthetic-v2-paper-v1-development-2026-07`
- scale fingerprint：`66c4bf1341a9370c`
- canonical reference profiles：8；conditioning profiles：9
- capabilities：9；`profile × capability` conditioning cells：29
- 同一数据集的 Electricity 2048-context 研究 profile 不重复参与 canonical target 聚合
- 每个单元使用 64 个配对 seed 拟合单调响应曲线并连续求逆，再用另一组 64 个 seed 验证
- 验收条件：五档 realized median 单调，且相对该 capability 全目标跨度的最大归一化误差不超过 0.20

## Artifact 结果

- supported cells：29 / 29
- 最大归一化误差：0.16544487
- 最差单元：`m4_hourly_daily_168ctx/regime_switching`
- canonical target 在所选 bucket 局部真实分布中的经验分位范围：0.01149425--0.99534884

最后一项说明统一强度不会退化为 bucket-local percentile：同一 canonical target 在不同真实基底上可以是常规强度，也可以是反事实压力测试。

## 在线生成链路 smoke

主在线协议排除由专用长 context 实验脚本处理的 `electricity_hourly_daily_2048ctx_24h`，其余组合为：

- eligible `profile × capability` cells：23
- intensities：5
- independent seeds per cell/intensity：10
- requests：1150
- accepted：1150
- first-pass accepted：1147（99.7391%）
- maximum attempts：2
- failures：0

3 次重试均来自 `gefcom2014_load_hourly_covariate_168ctx_24h/covariate_response`，分别位于 intensity 2、3、5，均在第二次尝试通过。first-pass rate 高于预注册的 95% 要求。

2048-context profile 对通用在线 near-distance artifact 按设计 fail closed；它保留 generator conditioning，但不计入上述主在线协议分母。

## 验证命令

```bash
cd backend
uv run python ../scripts/build_synthetic_v2_generator_conditioning_artifact.py
uv run pytest tests/unit/test_synthetic_generator_conditioning.py \
  tests/unit/test_synthetic_generator_conditioning_script.py \
  tests/unit/test_synthetic_paper_capabilities.py \
  tests/api/test_synthetic_generation.py -q
```

本记录只证明当前开发参考语料上的绝对标尺和生成链路成立。正式论文实验前仍需扩充并冻结更均衡的 development corpus；如果用 GIFT-Eval 做外部对应性验证，应保留 dataset-level held-out 子集，不参与 canonical target 定义。
