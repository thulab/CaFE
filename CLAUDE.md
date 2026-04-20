# TSBenchmark

## 项目概况

TSBenchmark 是一个基于 FastAPI + Flask 的时序预测动态 Benchmark 系统。

- **后端**：FastAPI，负责动态数据集生成、模型注册、任务执行、报告和排行榜
- **前端**：Flask，分为用户页面与管理页面
- **配置**：统一在 `conf/system.toml`
- **测试**：`test/unit/` 单元测试，`test/integration/` 集成测试

详细说明见 [README.md](README.md)。

## 规则与约定

项目规则、变更策略、通信规范、验证要求等请参考 [AGENTS.md](AGENTS.md)。

核心要点：

- **默认直接实现**，低风险局部改动直接执行
- **改动后优先跑单元测试**：`python -m pytest test/unit/`
- **禁止自行执行 git 操作**
- **后端入口**：`python -m backend.app.main`
- **前端入口**：`python -m frontend.app`
- **启停脚本**：`bash scripts/start_system.sh` / `bash scripts/stop_system.sh`

## 目录结构

```
backend/app/          FastAPI 后端
  datasets/           数据集管理（domain, synthetic, data_loader, processors, validators）
  models/             模型管理（domain, huggingface）
  tasks/              任务管理（domain, executor）
  leaderboards/       排行榜管理（domain, manager）
frontend/             Flask 前端
conf/                 系统配置
scripts/              启停脚本
test/                 单元测试与集成测试
runtime/              运行时生成内容
docs/                 架构说明
```

## 模块职责

| 模块 | 路径 | 职责 |
|------|------|------|
| Datasets | `backend/app/datasets/` | 合成数据生成、加载、处理、验证 |
| Models | `backend/app/models/` | 模型元信息、HuggingFace 适配器 |
| Tasks | `backend/app/tasks/` | 任务执行、指标计算 |
| Leaderboards | `backend/app/leaderboards/` | 排行榜聚合、排名策略 |
