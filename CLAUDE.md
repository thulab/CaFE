# TSBenchmark CLAUDE.md

## 项目概况

TSBenchmark 是一个**时序预测动态 Benchmark 系统**，包含两套评测体系：

1. **动态数据集系统** - 动态生成/加载数据集，支持模型评测与排行榜
2. **V1 Benchmark** - 离线合成零样本评测，验证模型能力维度

```
backend/app/          FastAPI 后端（动态数据集 + V1 算法）
frontend/             Flask 前端（管理页面 + 榜单页面）
conf/system.toml     系统配置
test/unit/           单元测试
runtime/              运行时生成内容
```

## 核心入口

| 组件 | 入口命令 |
|------|---------|
| 后端 | `python -m backend.app.main` |
| 前端 | `python -m frontend.app` |
| 启停 | `bash scripts/start_system.sh` / `bash scripts/stop_system.sh` |
| 单元测试 | `python -m pytest test/unit/ -q` |

## 模块架构

### 动态数据集模块 (`backend/app/datasets/`)

| 子模块 | 职责 |
|--------|------|
| `domain.py` | 领域模型：`DatasetRecord`、`DatasetBatch`、`SeriesSample`、`TrackSpec` |
| `manager.py` | 数据集 CRUD、批次管理 |
| `synthetic.py` | 合成数据生成器 |
| `data_loader/` | CSV/TSFile 加载器 |
| `processors/` | 数据预处理管道 |
| `validators/` | 数据校验器 |
| `benchmark_v1/` | **V1 合成评测算法**（独立子模块）|

### V1 Benchmark (`backend/app/datasets/benchmark_v1/`)

| 文件 | 职责 |
|------|------|
| `anchor.py` | 真实数据特征锚定、k-medoids 原型簇 |
| `features.py` | 8 维结构特征提取 |
| `families.py` | 5 类诊断数据生成器 |
| `calibration.py` | 代理难度校准（isotonic regression）|
| `generate.py` | 序列生成与接受准则 |
| `runner.py` | V1 评测运行器 |
| `adapters.py` | 模型适配器（基线 + Foundation Models）|
| `model_backends.py` | 外部模型后端（TimesFM/Chronos/Sundial/Moirai/LagLlama）|
| `validation.py` | 统计校验（难度单调性、特征漂移）|
| `aggregate.py` | 结果聚合 |

### 任务与榜单

| 模块 | 路径 | 职责 |
|------|------|------|
| Tasks | `backend/app/tasks/` | 任务执行、指标计算（MASE/sMAPE）|
| Models | `backend/app/models/` | 模型元信息、HuggingFace 适配 |
| Leaderboards | `backend/app/leaderboards/` | 榜单聚合、排名 |
| Reports | 前端模板 | 报告详情页 |

## 关键数据模型

### SeriesSample（单条时序样本）
```python
sample_id: str
history: list[float]           # 输入序列
target: list[float]            # 预测目标
covariates: dict[str, list[float]]
input_channel_values: dict[str, list[float]]
target_channel_values: dict[str, list[float]]
channel_layout: ChannelLayout
track_tags: list[str]
truth: SeriesTruth             # 趋势/周期/噪声等真值
```

### DatasetBatch（批次）
```python
batch_id: str
track: TrackKind
track_variant_id: str
sample_count: int
context_length: int
horizon: int
samples: list[SeriesSample]
validation: ValidationReport
feature_profile: DatasetFeatureProfile
```

### V1 Benchmark 数据配额
- Anchor Track: 2000 条
- Diagnostic Track: 5 family × 5 difficulty × 3 horizon_ratio × 100 = 7500 条
- 总计: 9500 条

## API 路由概览

### 数据集
- `GET /api/v1/datasets/batches` - 列出批次
- `GET /api/v1/datasets/batches/{batch_id}` - 批次详情
- `POST /api/v1/datasets/generate` - 生成合成批次
- `POST /api/v1/datasets/load/csv` - CSV 导入

### V1 Benchmark
- `POST /api/v1/benchmarks/v1/anchor-stats` - 构建锚定统计
- `POST /api/v1/benchmarks/v1/datasets` - 生成 V1 数据集
- `POST /api/v1/benchmarks/v1/evaluations/run` - 运行 V1 评测
- `POST /api/v1/benchmarks/v1/reports` - 生成 V1 报告
- `GET /api/v1/benchmarks/v1/artifacts` - 列出 V1 产物

### 任务与模型
- `GET /api/v1/tasks` / `GET /api/v1/tasks/{task_id}`
- `POST /api/v1/tasks/run`
- `GET /api/v1/models` / `POST /api/v1/models/register`

### 榜单
- `GET /api/v1/leaderboard?track=&metric_id=mse`
- `GET /api/v1/overview/admin`

## 前端页面路由

| 路由 | 页面 |
|------|------|
| `/admin/datasets` | 数据集管理（含 V1 Benchmark Tab）|
| `/admin/models` | 模型管理 |
| `/admin/tasks` | 任务管理（含 V1 Eval/Report Tab）|
| `/admin/leaderboard` | 榜单 |
| `/admin/datasets/<batch_id>` | 批次详情（新增）|
| `/` | 用户榜单页 |
