# TSBenchmark

基于 `detail.pdf` 和 `paper.png` 实现的时序预测动态 Benchmark 系统原型。当前版本包含两个独立系统：

- 后端：`FastAPI`，负责动态数据集、模型注册、任务执行、报告和排行榜。
- 前端：`Flask`，负责作为在线评测客户端展示并触发最小闭环。

## 架构落地

整体设计与讨论材料一一对应：

- `Dataset Manager`：动态生成纯合成多周期数据，按赛道执行变换并校验。
- `Model Manager`：管理模型元信息和 adapter；当前用桩模型完成集成。
- `Mission Manager / Executor`：把“模型 + 批次”组合为可回溯任务并执行。
- `Reporter`：聚合 MSE、MAE、sMAPE、延迟、Token 成本，输出 bad case 与总结。
- `Online System`：独立前端系统，仅通过 API 访问后端。

详细设计见 [docs/architecture.md](/Users/zhanghongyin/code/python/TSBenchmark/docs/architecture.md)。

## 目录

```text
backend/app/     FastAPI 后端
frontend/        Flask 前端
scripts/         冒烟脚本
runtime/         运行时生成内容
docs/            架构说明
```

## 启动

后端：

```bash
python -m backend.app.main
```

前端：

```bash
python -m frontend.app
```

默认情况下，前端会请求 `http://127.0.0.1:8000` 的后端。可通过环境变量 `TSBENCHMARK_BACKEND_URL` 覆盖。

## 冒烟

```bash
python scripts/smoke_test.py
```

该脚本使用临时目录存储运行数据，验证前后端最小闭环，不污染正式 `runtime/` 目录。
