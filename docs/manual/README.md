# TSBenchmark 用户手册

本文档面向本地 MVP 使用者，说明如何启动 TSBenchmark、加载真实 CSV 数据集、运行模型评测、查看榜单/报告/样本预测结果，以及如何排查常见问题。所有命令默认在项目根目录执行。

## 1. 适用范围与能力边界

当前手册覆盖仓库内的 MVP Web 平台，请先了解它能做什么、不能做什么：

- 后端：FastAPI + SQLModel + SQLite。
- 前端：Vue 3 + Vite 7。
- 数据集：支持 CSV 和单设备表模型 TsFile 输入，也支持生成合成测试用例集；内部统一存成 SQLite `SeriesPoint`。
- 目标列：真实数据加载支持选择一个或多个 target column；多个 target 会作为同一个目标向量一起切窗、评分和预测。
- 协变量：支持选择 known-future covariate columns。系统会用同一列名把协变量切成 history 与 future 两段，并在推理请求中分别发送 `history_covs` / `future_covs`。
- 合成数据：在新建评测向导中选择“生成合成数据”后，可选择一个或多个能力维度并指定共享参数。当前真实数据锚定由后端固定 mock 数据源提供，前端不暴露 anchor 选择。
- 模型：REST 模式下以 timer-rest-service 的 `/models/list` 为准，隐藏 `state=inactive` 的不可用模型，并读取 `forecast_limits.max_target_count` 判断是否支持多目标、读取 `forecast_limits.max_covariate_count` 判断是否支持协变量；`max_target_count=null` 表示不限制目标数，`max_covariate_count>0` 表示可接收协变量。进程内桩模式保留本地可复现模型。
- 指标：MASE、MSE、MAE，均为 lower is better；榜单主指标为 **MASE**（赛道 `primary_metric_id`）。
- 推理方式：实际推理通过外部 **timer-rest-service** 的 REST API 完成；本地无真实服务时可用桩程序顶上（见第 4 节）。
- 访问控制：公开榜单可匿名浏览；工作台页面需要登录；写操作、运行评测和用户/角色管理受 RBAC 权限控制。

## 2. 环境准备

建议先确认以下工具可用：

```bash
uv --version
node --version
npm --version
```

版本要求：

- 后端：Python `>=3.14`（由 `uv` 管理虚拟环境；当前后端依赖栈在 `backend/pyproject.toml` 中声明该下限）。
- 前端：Node.js 必须为 `20.19+` 或 `22.12+`（前端依赖 Vite 7）。如果 `node --version` 显示更低版本，启动脚本会直接拒绝并提示，请先切换到受支持版本。

首次运行时，启动脚本会自动准备后端虚拟环境和前端依赖（即检测不到 `.venv/bin/uvicorn` 或 `node_modules/.bin/vite` 时分别执行 `uv sync` 与 `npm install`）。如需手动预装：

```bash
cd backend && uv sync
cd ../frontend && npm install
```

## 3. 启动、停止和查看状态

项目提供统一脚本管理前后端：

```bash
./scripts/start-system.sh
./scripts/status-system.sh
./scripts/stop-system.sh
```

默认地址：

- 前端页面：`http://127.0.0.1:5173`
- 后端 API：`http://127.0.0.1:8000`

脚本会在运行目录（默认 `.tsbenchmark-system/`）下写入：

- `backend.pid` / `frontend.pid`：进程 ID。
- `backend.log` / `frontend.log`：启动日志和运行日志。
- （若启动了桩服务）`stub-service.pid` / `stub-service.log`。

### 3.1 端口与主机覆盖

```bash
TSBENCHMARK_BACKEND_PORT=8010 TSBENCHMARK_FRONTEND_PORT=5174 ./scripts/start-system.sh
```

支持的相关环境变量：

| 环境变量 | 默认值 | 说明 |
| --- | --- | --- |
| `TSBENCHMARK_BACKEND_HOST` | `127.0.0.1` | 后端监听地址 |
| `TSBENCHMARK_BACKEND_PORT` | `8000` | 后端端口 |
| `TSBENCHMARK_FRONTEND_HOST` | `127.0.0.1` | 前端监听地址 |
| `TSBENCHMARK_FRONTEND_PORT` | `5173` | 前端端口 |
| `TSBENCHMARK_SYSTEM_DIR` | `.tsbenchmark-system` | pid / 日志的存放目录 |
| `TSBENCHMARK_AUTH_SECRET` | 启动脚本默认注入开发密钥 | JWT 签名密钥；直接运行后端时必须显式设置 |
| `TSBENCHMARK_ADMIN_PASSWORD` | 启动脚本默认 `admin` | 首次初始化 admin 用户的密码；User 表已存在时不覆盖 |

> 注意：前端默认通过 `/api` 代理到 `http://127.0.0.1:8000`（见 `frontend/vite.config.ts`）。如果用 `TSBENCHMARK_BACKEND_PORT` 改了后端端口，需要同步调整该代理目标，否则前端调不通后端。

### 3.2 运行目录覆盖

```bash
TSBENCHMARK_SYSTEM_DIR=/tmp/tsbenchmark-system ./scripts/status-system.sh
```

`status-system.sh` 会把过期（进程已不存在）的 pid 文件自动清理掉，再报告 `running` / `stopped`。

### 3.3 Docker 部署入口

如果要部署成 Docker，并让后端连接另一个已经运行的 timer-rest-service 容器或服务地址，
请看开发者手册里的 [Docker 部署与环境变量](../developer/deployment.md)。那里包含
`docker compose` 用法、外部推理服务地址配置、runtime volume 说明和完整环境变量表。

## 4. 模型推理服务与本地桩程序

评测的实际推理通过外部 **timer-rest-service** 的 REST API 完成（契约见 [`docs/reference/rest-api.md`](../reference/rest-api.md)）。服务地址被抽象成配置项，本地无真实服务时可启动桩程序顶上。

### 4.1 配置项

所有配置都用 `TSBENCHMARK_` 前缀的环境变量覆盖：

| 环境变量 | 默认值 | 说明 |
| --- | --- | --- |
| `TSBENCHMARK_TIMER_SERVICE_BASE_URL` | `http://127.0.0.1:10810` | 服务前缀 `http://<host>:<port>` |
| `TSBENCHMARK_TIMER_SERVICE_API_PREFIX` | `/ai/api/v1` | 文档约定的统一路径前缀 |
| `TSBENCHMARK_MODEL_ADAPTER` | `rest` | `rest`=走 HTTP；`stub`=进程内确定性桩，无需网络 |
| `TSBENCHMARK_MODEL_LIFECYCLE_MODE` | `sequential_unload` | `sequential_unload`=每次运行逐模型加载/评测/卸载；`keep_loaded`=保留模型常驻 |
| `TSBENCHMARK_SAMPLE_FORECAST_TIMEOUT_SECONDS` | `300` | 单个样本调用推理服务的超时秒数 |
| `TSBENCHMARK_RUN_SAMPLE_PARALLELISM` | `4` | 单个 task 内样本 forecast 的最大并发数，便于让 timer-rest-service 聚合并发请求成批 |
| `TSBENCHMARK_RUN_PROGRESS_UPDATE_INTERVAL_SAMPLES` | `10` | 运行中每处理多少个样本刷新一次 task 样本进度 |
| `TSBENCHMARK_RUNTIME_DIR` | `runtime` | 运行产物根目录（见第 8 节） |
| `TSBENCHMARK_DATABASE_URL` | `sqlite:///runtime/tsbenchmark.db` | SQLite 数据库地址 |

`base_url` 与 `api_prefix` 会拼成业务端点根地址，例如默认值拼出 `http://127.0.0.1:10810/ai/api/v1`。

默认模型生命周期为 `sequential_unload`：创建运行后，后端会先尽量卸载 timer-rest-service 中已加载的模型，然后按模型逐个加载、评测、卸载，以降低多模型评测时的显存峰值。前端不会在启动运行前批量预加载模型。小模型调试场景如果希望保留常驻加载，可显式设置 `TSBENCHMARK_MODEL_LIFECYCLE_MODE=keep_loaded`。

### 4.2 三种接入模式

**模式 A：本地/真实 REST 服务（默认开箱即用）**

后端默认以 REST 方式调用 `127.0.0.1:10810`。如果这台机器上已经部署真实 timer-rest-service，直接使用即可；没有真实服务时，可用仓库内桩程序占用同一地址：

```bash
./scripts/stub-service.sh start     # 启动（日志写入 <运行目录>/stub-service.log）
./scripts/stub-service.sh status
./scripts/stub-service.sh stop
```

桩的预测是确定性的（last-value + 按 model_id 计算的偏置 + 固定种子噪声），便于可复现演示。桩监听地址可用 `TSBENCHMARK_STUB_HOST` / `TSBENCHMARK_STUB_PORT` 覆盖（覆盖后需同步设置后端的 `TSBENCHMARK_TIMER_SERVICE_BASE_URL`）。

**模式 B：进程内桩（不起任何 HTTP 服务）**

完全离线、不想额外起进程时，让后端使用进程内桩：

```bash
TSBENCHMARK_MODEL_ADAPTER=stub ./scripts/start-system.sh
```

此模式下后端不会去查询 timer-rest-service，模型列表里的「实时加载状态」会显示为未知。

**模式 C：指向真实服务**

```bash
TSBENCHMARK_TIMER_SERVICE_BASE_URL=http://<gpu-host>:<port> ./scripts/start-system.sh
```

模型列表（`GET /models`）直接读取真实服务的 `/models/list` 并同步到本地模型镜像；其中 `state=inactive` 的模型视为不可用，不显示在列表中，也不能通过后端手动加载或创建运行。服务不可达时模型列表会返回错误；前端点击 Run 后会先对未加载的已选模型调用后端 `POST /models/{model_id}/load`，后端执行期也会再次兜底确认 loaded。加载失败会写入 run/task 错误而不是卡住后台执行线程。

## 5. 数据文件要求

上传的 CSV 必须满足以下规则（每条都对应后端真实校验）：

- 编码为 UTF-8 或 UTF-8 BOM，其它编码会被拒绝。
- 必须有 header row，且首行不能看起来像数据行。
- 列名必须唯一（去掉首列可能的 BOM 后判断）。
- 分隔符支持逗号 `,`、制表符 `\t`、分号 `;`，系统按首行出现次数自动识别。
- 必须显式选择时间列和至少一个 target 列；所选列都要真实存在于列名中且不能重复。
- 可选协变量列必须是数值列，不能与 target 列重叠。当前只支持 known-future 协变量，也就是同一个协变量列在 history 与 horizon 区间都必须有真实值；系统会按窗口自动切成历史段和未来已知段。
- 时间格式必须能被 ISO 8601 解析（如 `YYYY-MM-DD`、`YYYY-MM-DD HH:mm:ss`，末尾 `Z` 会按 UTC 处理）。
- 时间列必须严格单调递增，不允许重复时间戳。
- 时间间隔必须等距，系统据此推断 frequency（如 `1h`、`1d`）；推断需要至少 2 个时间戳。
- 可以选择 **1 个或多个** target column；多目标只表示同一时间轴上的多个预测目标，不表示多序列/面板数据。
- target 和协变量值都必须能转换为有限浮点数，不允许缺失、非数字、NaN 或 Inf。
- 切分约束：`context_length + horizon` 不能超过数据行数；`context_length`、`horizon`、`stride` 都必须为正数；`stride` 缺省时等于 `horizon`。
- TsFile 输入只支持表模型。单设备文件可直接选择一个或多个物理量列；多 `timeseries_id` 文件需要选择同一设备下的完整 series path（形如 `table.device.value`），每次 load 仍只允许一个设备。

仓库中有一个可用于试跑的示例文件：

```text
backend/tests/fixtures/valid_hourly_20.csv
```

它有 20 行、列名为 `time,target,extra`，时间为整点（1 小时间隔），target 为 `10.0`～`29.0`。

## 6. 界面导航与完成一次评测

打开前端页面：

```text
http://127.0.0.1:5173
```

### 6.1 界面布局

页面是一个带左侧边栏的工作台：

- **匿名状态**：默认进入 `#/leaderboards`，只能看到公开榜单与登录入口。
- **登录后侧边栏**：**概览（Overview）**、**新建评测（New evaluation）**、**数据集（Datasets）**、**运行（Runs）**、**榜单（Leaderboards）**；具备管理权限时额外显示 Users / Roles / My profile。
- **顶栏**：左侧是面包屑（当前位置），右侧有「New evaluation」快捷按钮（需 `run.execute`）和**主题切换**按钮（亮色 → 暗色 → 跟随系统，三态循环；偏好记在浏览器本地）。
- **概览（首页 `#/`）**：登录后的工作台首页，展示统计卡与上手引导。

> ⚠️ 关于登录：`./scripts/start-system.sh` 会用开发默认值启动，首次初始化的 admin 用户密码默认为 `admin`。直接运行后端时必须设置 `TSBENCHMARK_AUTH_SECRET`；生产或共享环境应同时设置强随机 `TSBENCHMARK_ADMIN_PASSWORD`。

详情页（数据集清单 / 加载任务 / 测试用例集 / 赛道 / 排行 / 报告 / 样本预测）通过列表、面包屑或向导右侧的「Created artifacts」面板进入，也可直接用 URL 哈希深链访问（如 `#/reports/<id>`）。前端把底层 `Shard` 展示为“测试用例集”，用于表达由数据集生成、可被赛道复用的预测评测样本集合。

### 6.2 走查向导

点侧边栏「新建评测」（`#/new`）。页面会先让你选择入口：

- **创建新赛道**：先填写赛道名称和主指标，再上传或复用测试用例集，最后选择模型启动评测。
- **选择已有赛道**：复用已创建且未归档的赛道，直接选择模型启动一次新的评测运行。

向导草稿会保存在浏览器 `sessionStorage` 中。只要当前评测还没生成报告，顶部和侧边栏的「New evaluation」会切换为「Continue evaluation」，点击后回到当前向导而不是重新开始。向导中的产物链接（Dataset manifest / Load job / Test case set / Track / Ranking / Run / Report / Sample forecast）打开详情页后，也会在详情页顶部显示「Continue current evaluation」用于回到刚才的流程。若需要显式开始新的流程，在向导页点右上角「Reset」清空当前草稿。

选择创建新赛道后，向导左侧是**分步进度条**（每完成一步解锁下一步，已完成的步骤可点击回看），右侧是当前步骤卡片，底部有「Back」按钮；右侧常驻的「Created artifacts」面板会随流程累积各产物的快捷链接。完整流程如下：

1. **Create track**：填写赛道名称并选择主指标（MASE / MSE / MAE，均为 lower is better），点「Continue」进入数据步骤。此时还不会创建后端赛道，避免产生没有绑定数据的半成品赛道。
2. **Choose data source**：选择本次评测的数据来源：
   - **Upload real data**：把 CSV / TsFile 拖入虚线框，或点「Choose file」选择。上传成功后显示列数 / 预览行徽章和预览表，并默认把文件名（去掉扩展名）作为数据集名称和测试用例集名称的基础。
   - **Generate synthetic data**：跳过上传，进入合成数据配置。
   - **Reuse existing sets**：不生成新数据，直接选择已有测试用例集。
3. **Generate test cases / Generate synthetic test cases**：
   真实数据路径：
   - `Dataset name` 默认为上传文件名，可改成业务可读名称。
   - `Test case set name` 默认为“文件名 test cases”，用于在数据集页、测试用例集详情和赛道选择中识别该集合。
   - `Time column` 下拉选时间列（默认 `time`）。
   - `Target columns` 勾选一个或多个目标列；不选会提示 “Select at least one target”。列很多时可用搜索框和分页按钮定位。
   - `Known future covariates` 可选一个或多个协变量列；目标列与协变量列互斥，同一列不能两边都选。列很多时同样使用可搜索、可分页列表。
   - 填切分参数 `Context`（历史窗口长度，默认 `6`）、`Horizon`（预测长度，默认 `3`）、`Stride`（滑动步长，默认 `3`）、`Max samples`（可选，留空不限）。
   - 点「Generate test case set」：依次创建 dataset manifest、提交 load job 并生成样本，成功后自动进入下一步。

   合成数据路径：
   - `Test case set name` 是生成集合的名称前缀；多选能力时，每个能力维度生成一个测试用例集。
   - `Capabilities` 可多选。当前内置能力包括趋势、多季节性、状态切换、长记忆非线性、间歇异方差、公共因子、lead-lag、协同状态切换和协变量响应。
   - 共享参数包括 `Sample count`、`Context`、`Horizon`、`Difficulty`、`Season length`、`Target dimension`、`Seed`、`Frequency`。单变量能力固定目标维度为 1；多变量和协变量能力使用目标维度参数。
   - `Covariate response` 会生成 known-future 协变量 `weather` 和 `event`，结果页会在目标预测图下方单独显示协变量曲线。
   - 点「Generate synthetic test cases」后，后端生成 synthetic shard，并自动预选到下一步。
   - 研究实验样本可用脚本导入到平台库，便于在前端查看样本曲线。例如：
     ```bash
     cd backend
     uv run python ../scripts/import_synthetic_v2_experiment_shards.py \
       --summary ../runtime/research/synthetic-v2-univariate-capabilities-experiment/summary.json
     ```
     脚本默认写入 `backend/runtime/tsbenchmark.db` 和 `backend/runtime/synthetic/imports/`，按 `capability × difficulty` 生成测试用例集；同一 summary 和参数重复执行会跳过已导入 shard，可加 `--allow-duplicates` 强制新建。

   如果第 2 步选择复用已有集合，本步只显示提示并继续到已有测试用例集选择。
4. **Select test cases**：在可搜索、可分页的列表中勾选一个或多个测试用例集。刚生成的集合会自动预选；也可以搜索名称、数据集、目标列、能力维度或 ID，并追加已有集合。列表详情会显示真实/合成类型、样本数、窗口、目标列；只有测试用例集实际带协变量时才显示协变量列，合成集合还会显示能力、难度和 seed。点「Create track from selected sets」后，系统基于所选集合创建评测赛道与默认榜单；合成集合会按能力维度自动拆成多个 capability block。
5. **Run models**：在模型列表里勾选一个或多个适配器（可一键「Select all」），点「Run」。多目标测试用例集会自动禁用不支持该目标维度的模型：后端和前端都按模型目录中的 `forecast_limits.max_target_count` 判断，`null` 视为原生多目标无限制。带协变量的测试用例集会自动禁用 `forecast_limits.max_covariate_count` 小于所选协变量数量的模型。系统创建 benchmarking run 并**每 5 秒轮询**一次进度——卡片上实时显示状态徽章、进度条与 模型/任务/样本 计数；样本进度按 `processed_samples`（成功 + 失败）推进，`completed_samples` 仅表示成功样本。REST 模式下未加载模型会在执行前自动加载；加载或推理失败会反映到 run 详情和报告里。若「失败样本」大于 0，可点击数字查看错误原因统计；再点某类原因的「查看样本」才加载分页明细，明细包含样本、模型、能力块和错误信息。终态后有权限的用户可点「重跑失败样本」，后端会创建后台重跑任务，运行详情页显示已处理/总数、成功、仍失败、待处理和当前阶段；刷新页面后仍能恢复进行中的重跑进度。重跑完成后会覆盖原失败行，并重新计算运行状态、报告和榜单。运行期间可点「Cancel」请求取消，页面会显示「正在取消」并继续轮询直到 run 变为 `cancelled`。
6. **Open report**：run 到达非取消终态（`succeeded` / `partial_succeeded` / `failed`）并生成 report 后，向导自动跳到本步，给出「Open report」「View ranking」「Run detail」入口。`cancelled` run 不生成报告、不进入榜单，可从「Run detail」查看取消事件和已处理进度。

使用示例 CSV 和默认参数 `Context=6 / Horizon=3 / Stride=3` 时，应生成 **4 个 sample**（窗口长度 `6+3=9`，从第 0 行起按步长 3 滑动，起点为 0/3/6/9）。

### 6.3 资源归档、恢复与永久删除

工作台中的数据集、测试用例集、赛道和评测运行支持两类删除语义：

- **归档（Archive）**：默认删除动作，可恢复。归档不会删除业务行、历史报告或榜单结果；资源默认从列表和新建流程中隐藏，详情深链仍可打开。
- **永久删除（Permanent delete）**：管理员操作，不可恢复。前端会先展示影响范围，再二次确认；后端按依赖关系删除 DB 行，并清理报告/预测产物。数据集 purge 还会删除位于 `runtime/uploads/` 下的托管上传文件。

具体行为：

- 数据集 / 测试用例集：在「数据集」页面上传数据、生成测试用例集、查看和归档。归档后默认不再出现在新建评测或赛道创建的候选列表；打开「Show archived」可恢复或永久删除。
- 赛道：在「赛道」页面或赛道详情中归档。归档赛道会保留历史榜单、运行列表和报告，但不能再基于该赛道启动新的评测运行。
- 运行：只能在终态（`succeeded` / `partial_succeeded` / `failed` / `cancelled`）后归档或永久删除；排队中、运行中或取消请求中的 run 需要先等待终态或取消完成。
- 永久删除数据集或测试用例集可能级联删除依赖它的赛道、运行、报告、指标、榜单条目和预测产物；永久删除赛道会级联删除其运行与榜单；永久删除运行会删除该 run 的 unit/task、报告、预测、指标、失败样本重跑记录和事件。

常见错误：

- `resource_archived`：试图在已归档赛道上启动新运行。
- `run_not_terminal`：运行未到终态，不能归档或永久删除。
- `purge_requires_cascade`：资源存在下游引用，管理员需在影响范围确认后用级联删除。

## 7. 查看结果

下列接口为后端真实路径，挂载在后端根路径下（不带 `/api` 前缀）。前端访问时通过 Vite 的 `/api` 代理转发，代理会自动去掉 `/api` 前缀。直接用 `curl` 调试请使用后端地址，例如 `http://127.0.0.1:8000/...`。

### 7.1 榜单（ranking）

榜单按 Track 维度展示模型成绩：

```text
GET /tracks/{track_id}/ranking?metric=mse&policy=latest_valid_result
```

可选查询参数：

- `metric`：`mase` / `mse` / `mae`（缺省跟随赛道主指标，当前为 `mase`）。
- `policy`：`latest_valid_result`（默认）或 `best_result`。

管理员可在排行榜总览或赛道排名页临时切换某个榜单是否对匿名访客公开。隐藏后，未登录用户不会在排行榜总览中看到该榜，直接访问对应赛道排名也会返回不可见；已登录用户仍可查看。该开关是临时管理入口，后续会随整体权限管理重新整理。

> 前端在「赛道详情」内嵌榜单与独立「排行（Ranking）」页都提供 `metric` / `policy` 下拉，并用**条形图 + 榜单表**（冠军高亮、奖牌序号、数值格式化）展示，模型 ID 会解析为模型名。

返回结构：

```json
{
  "track_id": "...",
  "metric": "mse",
  "policy": "latest_valid_result",
  "items": [{ "model_id": "...", "rank": 1, "metric_value": 0.12 }]
}
```

排序方向为 lower is better。

### 7.2 报告（report）

报告按 run 生成，保存为 JSON 产物，默认位置：

```text
runtime/reports/{run_id}.json
```

接口：

```text
GET /reports/{report_id}
GET /reports/{report_id}?sample_link_limit=10&sample_link_offset=0
```

返回字段包括：

- `model_metrics`：模型（unit）级指标表。
- `task_summaries`：task 摘要（含 task 级指标、`error_code`、`error_message`）。
- `capability_blocks`：本次 run 涉及的能力测试块元数据，含 `block_type`（`synthetic` / `real`）、`capability_type`、展示名、样本数、目标维度和协变量维度。
- `capability_metrics`：每个模型在每个能力测试块上的 task 级指标，用于报告页展示能力画像。报告页默认用合成能力维度绘制雷达图；同一能力维度有多个测试块时按样本数加权聚合。真实数据不进入默认雷达轴，但会在“全部测试组”分解表中展示。
- `sample_forecast_links`：按样本去重后的预测链接列表，每项含 `sample_id`、`run_id`、`forecast_artifact_id(s)`，并尽量附带 `sample_index`、行号窗口和预测时间戳范围，方便前端显示可读样本名称。大型 run 可用 `sample_link_limit` / `sample_link_offset` 分页读取；响应同时给出 `sample_forecast_links_total`、`sample_forecast_links_limit`、`sample_forecast_links_offset`。
- `status`：run 终态。
- `cancellation_reason`：兼容字段；正常执行报告中为 `null`。取消运行不生成报告。
- `benchmarking_run_id`、`track_id`、`report_id`。报告页右上角提供返回赛道入口，便于回到该 track 的榜单视图。

### 7.3 样本预测（sample forecast）

样本预测视图用于检查单个 sample 上多个模型的预测曲线和指标：

```text
GET /samples/{sample_id}/forecast?run_id={benchmarking_run_id}
```

`run_id` 为必填查询参数。返回内容包括：

- `sample_id`。
- `sample_index`、行号窗口和预测时间戳范围（如可从样本索引恢复）。
- `target_history`：历史 target 值。
- `target_future`：未来真值。
- `covariate_column_names`、`history_cov`、`future_cov`：如果样本来自带协变量的测试用例集，会返回协变量列名、历史段和未来已知段。
- `models[]`：每个模型的 `status`、`forecast`、sample-level 指标（MASE/MSE/MAE；MASE 在平稳历史等场景下可能无定义），失败时附 `error_message`。

前端「样本预测对比」页把历史、真值与各模型预测画在同一张**交互式折线图**上（按真实数据缩放、带坐标轴与网格，各模型用不同颜色、预测用虚线，悬浮显示对应步数值，图例可点选开关各序列，多维目标可切换维度）。如果样本带协变量，页面会在下方另画一张协变量折线图，标出 history 与 future 已知段，便于对照预测表现。测试用例样本曲线页和样本预测页都提供上一条 / 下一条样本导航，便于连续检查同一测试用例集内的窗口。页面还会列出每模型指标表（最优值高亮）；非 `succeeded` 的模型以告警条单独列出。

### 7.4 生命周期接口

列表接口默认隐藏已归档资源，可用 `include_archived=true` 显式包含：

```text
GET /dataset-manifests?include_archived=true
GET /shards?include_archived=true
GET /tracks?include_archived=true
GET /benchmarking-runs?include_archived=true
```

四类资源都提供一致的生命周期接口：

```text
GET    /<resource-path>/{id}/deletion-impact
POST   /<resource-path>/{id}/archive
POST   /<resource-path>/{id}/restore
DELETE /<resource-path>/{id}?cascade=true
```

其中 `<resource-path>` 为 `dataset-manifests`、`shards`、`tracks` 或 `benchmarking-runs`。归档/恢复分别需要 `dataset.delete`、`track.delete` 或 `run.delete` 权限；永久删除需要 `admin.purge` 权限。

## 8. 运行产物

默认运行目录是 `runtime/`（可用 `TSBENCHMARK_RUNTIME_DIR` 覆盖），主要结构如下：

```text
runtime/
  uploads/
  samples/
  forecasts/
  reports/
  tsbenchmark.db
```

说明：

- `uploads/`：保存上传后的 CSV / TsFile 输入文件。
- `samples/`：遗留兼容目录；当前样本真值不再写成 JSONL，而是写入 SQLite `SeriesPoint`，`SampleIndex` 只保存行号区间指针。
- `forecasts/`：保存模型预测 JSONL。
- `reports/`：保存 run summary JSON（即 `{run_id}.json`）。
- `tsbenchmark.db`：SQLite 数据库，包含元数据、`SeriesPoint` 序列真值和 `SampleIndex` 样本指针。

如需使用隔离的运行目录（注意运行目录与数据库地址要一起改）：

```bash
TSBENCHMARK_RUNTIME_DIR=/tmp/tsbenchmark-runtime \
TSBENCHMARK_DATABASE_URL=sqlite:////tmp/tsbenchmark-runtime/tsbenchmark.db \
./scripts/start-system.sh
```

## 9. 常见问题排查

### 9.1 启动失败

先查看状态，再看日志：

```bash
./scripts/status-system.sh
sed -n '1,160p' .tsbenchmark-system/backend.log
sed -n '1,160p' .tsbenchmark-system/frontend.log
```

常见原因：

- 端口已被占用（用第 3.1 节的端口覆盖换一个端口）。
- `uv`、`node` 或 `npm` 未安装，或 Node.js 版本低于 `20.19` / `22.12`。
- 依赖安装被网络或权限限制阻止。
- 上一次异常退出留下了 stale pid；再执行一次 `stop-system.sh` 后重试。

### 9.2 页面能打开但 API 请求失败

确认后端是否运行（注意后端接口在根路径，不带 `/api`）：

```bash
curl http://127.0.0.1:8000/models
```

确认前端代理仍指向后端：

```text
frontend/vite.config.ts
```

默认前端通过 `/api` 代理到 `http://127.0.0.1:8000`，并在转发时去掉 `/api` 前缀。若改过后端端口，需同步改这里。

### 9.3 CSV 加载失败

后端错误响应统一为 JSON：`{ "error_code": ..., "message": ..., "details": {...} }`。前端会把 `message` 显示在告警条上。常见 `error_code`：

CSV 结构 / 编码类：

- `csv_encoding_unsupported`：编码不是 UTF-8。
- `csv_missing_header`：缺少 header row，或首行被识别为数据行。
- `csv_duplicate_columns`：列名重复。
- `csv_time_column_missing`：选择的时间列不存在。
- `csv_target_columns_invalid`：未选择目标列，或目标列选择重复。
- `csv_target_column_missing`：选择的目标列不存在。

时间列类：

- `csv_time_parse_failed`：时间值无法解析。
- `csv_duplicate_timestamp`：时间戳重复。
- `csv_time_not_monotonic`：时间列不是严格递增。
- `csv_time_not_equidistant`：时间间隔不等距。
- `csv_frequency_not_inferable`：时间戳不足 2 个，无法推断 frequency。
- `csv_frequency_mismatch`：显式提供的 frequency 与推断结果不一致。

目标值类：

- `csv_value_missing`：目标值缺失。
- `csv_value_not_float`：目标值无法转换为浮点数。
- `csv_value_not_finite`：目标值是 NaN 或 Inf。

切分 / 样本类：

- `load_target_columns_invalid`：load job 未选择 target，或 target 选择重复。
- `load_covariate_columns_invalid`：协变量列选择重复，或与 target 列重叠。
- `split_config_invalid`：`context_length`、`horizon` 或 `stride` 非正数。
- `split_length_exceeds_rows`：`context_length + horizon` 超过数据行数。
- `sample_count_empty`：切分配置没有产生任何样本。
- `dataset_manifest_not_found`：load job 指向的 dataset manifest 不存在。
- `model_target_dim_unsupported`：选择了多目标测试用例集，但参评模型的 `forecast_limits.max_target_count` 不支持该目标维度；请换用支持多目标的模型或生成单目标测试用例集。
- `model_covariate_dim_unsupported`：选择了带协变量的测试用例集，但参评模型的 `forecast_limits.max_covariate_count` 不支持该协变量数量；请换用支持协变量的模型或重新生成不带协变量的测试用例集。

### 9.4 想重新开始一次干净测试

先停止服务，再清理运行产物，然后重启：

```bash
./scripts/stop-system.sh
rm -rf runtime .tsbenchmark-system
./scripts/start-system.sh
```

如果之前用桩服务（模式 A），别忘了一并 `./scripts/stub-service.sh stop`。

## 10. 开发者验证命令

后端测试：

```bash
cd backend && uv run pytest
```

前端单元测试：

```bash
cd frontend && npm test
```

前端 e2e smoke 测试：

```bash
cd frontend && npm run test:e2e
```

启停脚本测试：

```bash
bash scripts/tests/test_system_scripts.sh
```

这些命令通过后，说明当前 MVP 的主要加载、评测、报告、榜单、样本预测和脚本启停行为处于可用状态。
