# TSBenchmark Agents Guide

## 范围与权限

- 本项目工作目录：`/root/tsbenchmark/TSBenchmark`
- **禁止**对目录外的任何文件进行读写操作
- **禁止**自行执行 git 操作（commit、push、pull、reset 等）

## 默认行为

**低风险局部改动**：直接实现，无需确认
- 修复 bug、添加日志、重构内部实现
- 不涉及 API 契约变更

**中高风险改动**：先分析影响，再执行
- 涉及 API 端点增减或参数变化
- 涉及数据模型结构变更
- 涉及前端路由或页面逻辑变更

## 验证要求

### 必须执行的验证
```bash
# 单元测试（每次代码改动后）
python -m pytest test/unit/ -q

# 如改动涉及 V1 / benchmark_v1 模块
python -m pytest test/unit/backend/app/datasets/test_benchmark_v1.py -q
```

### 可选验证（按需）
```bash
# 集成测试
python -m pytest test/integration/ -q

# 前端语法校验
python -c "from frontend.app import app"
python -c "from backend.app.api import create_api"
```

### 危险行为警告
- **禁止**跳过测试直接提交
- **禁止**在未验证的情况下声称"测试通过"
- **禁止**修改 `runtime/` 目录下的运行时生成物（它们是验证副产品）
- **禁止**修改 `conf/system.toml` 中的生产配置默认值

## 变更策略

1. 优先保持模块边界清晰，不跨模块堆砌无关逻辑
2. 涉及领域模型时，放入对应模块的 `domain.py`
3. 涉及 V1 算法时，优先放入 `backend/app/datasets/benchmark_v1/`
4. 文档/配置/脚本中的路径若因重构失效，必须一并更新

## 通信规范

### 必须说明的内容
- 做了什么修改
- 为什么这样做
- 运行了什么验证
- 未覆盖的风险或已知缺口

### 输出格式
```
## 变更摘要
[一句话描述]

## 变更详情
[具体修改内容]

## 验证结果
[测试输出摘要]

## 风险/缺口
[如有]
```

## 项目约定

| 约定 | 说明 |
|------|------|
| 后端入口 | `python -m backend.app.main` |
| 前端入口 | `python -m frontend.app` |
| 配置 | `conf/system.toml` |
| 启停 | `bash scripts/start_system.sh` / `stop_system.sh` |
| 测试发现 | `python -m pytest test/unit/ -q` |

## 决策原则

1. **先看现有实现，再动手改** - 不基于猜测重写
2. **能局部修复，不做大规模改造** - 保持改动收敛
3. **性能优先场景** - 避免数据处理、批量推理、序列转换中的明显低效实现
4. **跨模块依赖** - 避免循环导入，保持依赖单向
