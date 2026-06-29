# Synthetic v2 调研实验完成记录

日期：2026-06-29

## 目标

围绕合成数据重构完成一轮从真实数据分布提取、能力契约设计、生成器 v2 pilot、离线基线响应实验到验证的闭环。

## 已完成内容

1. 建立 synthetic v2 研究计划，明确第一阶段使用显式数学/统计特征，不引入深度生成模型。
2. 实现离线真实数据 feature profiler，支持 CSV、Monash TSF、TSF zip，并输出 feature quantile 和 target caps。
3. 修复 profiler 对 Monash 非 UTF-8 TSF zip 的兼容性，并把 TSF `max_windows` 改为全数据集统一限流。
4. 下载并 profile 两个公开真实数据基底：
   - US Births：日频、小型 sanity anchor。
   - M4 Hourly：小时级、多序列主 anchor。
5. 形成 trend / multi-seasonal 能力契约草案。
6. 将后端 `trend` 和 `multi_seasonal` 生成器调整为 v2 pilot：
   - trend 的 `trend_strength` / `slope_abs` / `curvature_abs` 随 difficulty 单调增强。
   - multi-seasonal 通过 48 点次级周期制造单周期 seasonal naive 难度。
   - 生成后抽取显式 realized features 写入 sample metadata。
   - 对 v2 pilot 能力加入轻量 acceptance check，防止目标/控制特征超过真实 profile cap。
7. 建立 generator experiment，对比旧公式、新公式和 M4 真实窗口的特征与 naive / seasonal naive 响应。

## 关键结果

真实 profile 烟测：

- `m4_hourly_daily_168ctx` 作为主小时级 anchor。
- 主 anchor 关键 cap：
  - `trend_strength <= 1.0`
  - `slope_abs <= 0.5314`
  - `curvature_abs <= 1.0135`
  - `noise_ratio <= 0.5807`
- US Births 的周季节性可用于日频 sanity；年季节性在 `365+30` 窗口下不足两个完整周期，因此只保留为诊断项。

生成器实验：

- v2 trend strength 单调：`True`
- v2 trend slope 均值不超过 cap：`True`
- legacy trend slope 均值不超过 cap：`False`
- v2 multi-seasonal seasonal naive MAE 单调：`True`
- v2 multi-seasonal seasonal naive MAE 增长倍数：`6.1774`

## 产物

- 计划：`docs/superpowers/plans/2026-06-29-synthetic-v2-research-plan.md`
- 能力契约：`docs/superpowers/specs/2026-06-29-synthetic-v2-capability-contracts.md`
- 真实 profile 烟测脚本：`scripts/run_synthetic_v2_profile_smoke.py`
- 真实 profile 烟测记录：`docs/superpowers/baselines/2026-06-29-synthetic-v2-profile-smoke.md`
- 生成器实验脚本：`scripts/run_synthetic_v2_generator_experiment.py`
- 生成器实验记录：`docs/superpowers/baselines/2026-06-29-synthetic-v2-generator-experiment.md`
- 后端实现：`backend/app/services/synthetic_generation_service.py`
- 覆盖测试：
  - `backend/tests/unit/test_synthetic_feature_profile_script.py`
  - `backend/tests/unit/test_synthetic_v2_profile_smoke_script.py`
  - `backend/tests/unit/test_synthetic_v2_generation_pilot.py`
  - `backend/tests/unit/test_synthetic_v2_generator_experiment_script.py`
  - `backend/tests/api/test_synthetic_generation.py`

## 验证

已执行：

```bash
cd backend && uv run pytest
python3 scripts/run_synthetic_v2_profile_smoke.py --skip-download
cd backend && PYTHONPATH=.:../scripts uv run python ../scripts/run_synthetic_v2_generator_experiment.py
```

结果：

- Backend test suite：`272 passed`
- Profile smoke：成功复现报告与 runtime JSON profile
- Generator experiment：成功复现报告与 runtime JSON summary

## 提交

- `3d42da8 add synthetic v2 research plan`
- `c53fde7 add synthetic feature profiler prototype`
- `26a8e04 improve synthetic feature profiler compatibility`
- `a9daaf1 add synthetic v2 profile smoke`
- `53c9621 add synthetic v2 capability contracts`
- `8c0fdfc add synthetic v2 generator pilot`

## 后续边界

- 本轮只把 `trend` 和 `multi_seasonal` 做到 v2 pilot；其他能力维度仍沿用旧公式。
- multi-seasonal 第一版以 seasonal naive MAE 退化作为主要行为验收，后续需要把 multi-period strength / amplitude ratio 加入 profiler 的显式特征。
- 当前 acceptance cap 使用内置 M4 Hourly profile 常量；后续应接入可配置 anchor profile。
- 深度生成模型暂不引入。只有当显式特征无法描述目标能力，或规则生成长期无法通过真实分布验收时，再进入深度/混合生成路线。
