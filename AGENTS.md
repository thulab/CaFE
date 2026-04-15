# TSBenchmark Agents Guide

## Scope

- 本文件只约束 `/Users/zhanghongyin/code/python/TSBenchmark` 目录内的工作。
- 禁止读取、修改或创建本项目目录之外的任何文件，除非用户明确要求且已获得相应授权。

## Default Role

- 默认直接实现用户需求，而不是停留在方案讨论。
- 是否先征求确认由风险决定：
  - 低风险、局部改动：直接实现并验证。
  - 中高风险、涉及接口变更或较大重构：先分析影响范围，再给出结论和实施理由，必要时再执行。

## Change Policy

- 允许中等规模重构。
- 在完成评审并说明收益、风险和迁移影响后，可以进行较大改动。
- 优先保持目录结构清晰、模块职责清楚、命名可读。
- 性能有限是本项目的重要约束；涉及数据处理、批量推理、序列转换、排行榜聚合时，优先避免明显低效实现。
- 修改时优先复用现有模块边界：
  - `backend/app/datasets/`
  - `backend/app/models/`
  - `backend/app/tasks/`
  - `backend/app/leaderboards/`
  - `frontend/`

## Communication

- 给出较为详细的推理与解释，尤其是在以下场景：
  - 选择实现方案时
  - 做结构调整或重构时
  - 判断风险、兼容性或性能影响时
  - 无法完成验证或发现仓库缺口时
- 先给结论，再给关键依据和后续动作。
- 不要空泛表态；解释要落到文件、模块、接口或运行行为上。

## Validation

- 完成代码改动后，默认跑完与改动相关的验证，优先跑单元测试。
- 当前仓库已经有标准测试目录 `test/`，新增或修改功能后应优先补充并执行对应的 `unit` / `integration` 测试，而不是继续把测试逻辑放回 `scripts/`。
- 默认的测试发现入口：
  - `python -m unittest discover -s test -t . -p 'test_*.py'`
- 当前仓库可见的主要验证入口：
  - `test/unit/`：模块级单元测试，优先覆盖 `datasets`、`models`、`tasks`、`leaderboards`、`frontend` 的局部行为。
  - `test/integration/test_smoke_flow.py`
  - `test/integration/test_verify_chronos2_e2e.py`（真实 Chronos2 端到端验证，依赖较重，默认受 `TSBENCHMARK_RUN_CHRONOS2_E2E=1` 控制，仅在改动涉及 Hugging Face/Chronos2 链路且环境允许时使用）
- `scripts/smoke_test.py` 和 `scripts/verify_chronos2_e2e.py` 现在是兼容入口；如无特殊需要，优先直接运行 `test/` 下对应测试或统一的 `unittest discover` 命令。
- 如果只做静态结构调整，至少补充最小导入或编译校验，并在结果中说明。
- 不要声称运行了未实际执行的测试。

## Git Rules

- 禁止自行执行任何 git 操作。
- 包括但不限于：`git status`、`git diff`、`git add`、`git commit`、`git checkout`、`git reset`、`git restore`、`git clean`、`git rebase`。
- 如需了解变更状态，应通过直接读取文件和运行项目级验证来判断，不依赖 git。

## Project Conventions

- 后端入口：`python -m backend.app.main`
- 前端入口：`python -m frontend.app`
- 配置文件：`conf/system.toml`
- 启停脚本：
  - `bash scripts/start_system.sh`
  - `bash scripts/stop_system.sh`
- 如果修改配置读取、服务启动、端口、超时、运行目录等行为，必须同步检查 `conf/system.toml`、启动脚本和 README 是否仍一致。

## Structure Expectations

- 新增代码时，优先放入职责明确的模块中，不要把无关类型和逻辑重新堆回单一大文件。
- 涉及领域模型时，优先放到对应模块下的 `domain` 中，而不是创建新的全局杂项文件。
- 控制跨模块依赖方向，避免循环导入；若需要聚合导出，应采用清晰且低耦合的方式。
- 文档、配置、脚本中的路径或命令若因重构失效，必须一并更新。

## Decision Heuristics

- 先看现有实现，再动手改，不基于猜测重写。
- 能局部修复时，不做无必要的大范围改造。
- 需要重构时，先判断是否能显著改善以下至少一项：
  - 可读性
  - 模块边界
  - 维护成本
  - 性能
  - 错误率
- 若收益不明显，保持改动收敛。

## Output Requirements

- 结果说明中应包含：
  - 做了什么
  - 为什么这样做
  - 跑了什么验证
  - 未覆盖的风险或缺口
