# TSBenchmark Docker 部署手册

本文档用于指导试用方在一台已经安装 Docker 的服务器上部署 TSBenchmark。

TSBenchmark 本身包含前端页面和后端服务；模型推理服务（timer-rest-service）需要提前单独部署好，TSBenchmark 后端通过 HTTP 地址调用它。

---

## 1. 部署形态

| 模块 | Docker 容器 | 端口 | 说明 |
| --- | --- | --- | --- |
| 前端 | `tsbenchmark-frontend` | `5173 -> 80` | 浏览器访问入口，内置 nginx，并把 `/api` 转发到后端 |
| 后端 | `tsbenchmark-backend` | `8000 -> 8000` | FastAPI 服务，负责数据集上传、切片、赛道、评测和报告 |
| 推理服务 | 外部服务 | 通常 `10810` | 需要另行部署；TSBenchmark 通过 `TSBENCHMARK_TIMER_SERVICE_BASE_URL` 访问 |
| 数据存储 | Docker volume | 无端口 | SQLite、上传文件、预测结果和报告存放在 `tsbenchmark-runtime` 卷中 |

默认访问地址：

```text
http://<服务器IP>:5173
```

后端 OpenAPI 文档地址：

```text
http://<服务器IP>:8000/docs
```

---

## 2. 镜像版本

本次交付包含两个 TSBenchmark 镜像：

```text
tsbenchmark-backend:latest    约 418 MB
tsbenchmark-frontend:latest   约 49 MB
```

推理服务镜像不在本部署包内。请确认推理服务已经部署，并能提供 timer-rest-service REST API。

---

## 3. 服务器前置条件

目标服务器需要满足以下条件：

| 项目 | 要求 | 检查命令 |
| --- | --- | --- |
| 操作系统 | Linux x86_64 / amd64 推荐 | `uname -m` |
| Docker | 已安装并启动 | `docker --version` |
| Docker Compose | 支持 `docker compose` 子命令 | `docker compose version` |
| 磁盘空间 | 建议至少 5 GB 可用空间；如上传大量 TsFile，按数据量额外预留 | `df -h` |
| 端口 | 默认需要开放 `5173` 和 `8000` | `ss -tlnp | grep -E ':(5173|8000) '` |
| 推理服务 | timer-rest-service 已运行，TSBenchmark 后端可访问 | 见第 7 节 |

说明：

- TSBenchmark 后端镜像已内置 Python 运行环境，不需要在宿主机安装 Python。
- TSBenchmark 前端镜像已内置构建后的静态页面，不需要在宿主机安装 Node.js。
- 如果只从浏览器访问系统，通常只需要开放前端端口 `5173`；后端端口 `8000` 可按现场安全策略决定是否对外开放。

---

## 4. 部署包清单

交付包中包含以下文件：

```text
tsbenchmark-docker-deploy.tar.gz
```

解压后目录如下：

```text
tsbenchmark-docker/
  docker-compose.yml
  .env
  tsbenchmark-backend.tar
  tsbenchmark-frontend.tar
  TSBenchmark Docker 部署手册.pdf
```

其中：

| 文件 | 说明 |
| --- | --- |
| `docker-compose.yml` | Docker Compose 编排文件 |
| `.env` | 部署配置文件，主要修改端口、管理员密码、推理服务地址 |
| `tsbenchmark-backend.tar` | 后端 Docker 镜像 |
| `tsbenchmark-frontend.tar` | 前端 Docker 镜像 |
| `TSBenchmark Docker 部署手册.pdf` | 本部署说明 |

可选校验：

```bash
md5sum tsbenchmark-docker-deploy.tar.gz
```

---

## 5. 准备部署目录

以下以 `/opt/tsbenchmark-deploy` 作为部署目录。也可以换成现场约定目录。

```bash
mkdir -p /opt/tsbenchmark-deploy
```

把交付包上传到该目录：

```bash
ls /opt/tsbenchmark-deploy
# 期望看到 tsbenchmark-docker-deploy.tar.gz
```

解压：

```bash
cd /opt/tsbenchmark-deploy
tar -xzf tsbenchmark-docker-deploy.tar.gz
cd tsbenchmark-docker
```

---

## 6. 加载镜像

在部署目录执行：

```bash
docker load -i tsbenchmark-backend.tar
docker load -i tsbenchmark-frontend.tar
```

确认镜像已经加载：

```bash
docker images | grep tsbenchmark
```

期望看到类似输出：

```text
tsbenchmark-backend    latest
tsbenchmark-frontend   latest
```

---

## 7. 修改配置文件 `.env`

部署前必须检查 `.env` 文件。示例如下：

```bash
# Required for backend startup. Use strong random values outside local smoke tests.
TSBENCHMARK_AUTH_SECRET=replace-with-at-least-32-random-bytes
TSBENCHMARK_ADMIN_PASSWORD=replace-with-strong-admin-password

# External timer-rest-service address.
# Same Docker network example: http://timer-service:10810
# Host-published service example: http://host.docker.internal:10810
TSBENCHMARK_TIMER_SERVICE_BASE_URL=http://host.docker.internal:10810
TSBENCHMARK_TIMER_SERVICE_API_PREFIX=/ai/api/v1

# Model execution mode.
TSBENCHMARK_MODEL_ADAPTER=rest
TSBENCHMARK_MODEL_LIFECYCLE_MODE=sequential_unload
TSBENCHMARK_TIMER_SERVICE_MODEL_LOAD_TIMEOUT_SECONDS=600
TSBENCHMARK_SAMPLE_FORECAST_TIMEOUT_SECONDS=300

# Host ports published by docker-compose.yml.
TSBENCHMARK_FRONTEND_PUBLISHED_PORT=5173
TSBENCHMARK_BACKEND_PUBLISHED_PORT=8000
```

### 7.1 必改项：JWT 密钥

`TSBENCHMARK_AUTH_SECRET` 用于签发登录 token。生产或共享环境必须使用强随机值。

生成方式示例：

```bash
openssl rand -hex 32
```

把输出填入 `.env`：

```text
TSBENCHMARK_AUTH_SECRET=<openssl 输出的 64 位 hex>
```

### 7.2 必改项：管理员密码

`TSBENCHMARK_ADMIN_PASSWORD` 是首次初始化管理员账号 `admin` 时使用的密码。

```text
TSBENCHMARK_ADMIN_PASSWORD=<现场强密码>
```

注意：

- 这个密码只在第一次初始化数据库时生效。
- 如果系统已经启动过并创建了用户表，后续修改 `.env` 不会覆盖已有 admin 密码。
- 忘记密码时，通常需要在页面中由管理员修改，或清空 runtime volume 后重新初始化。

### 7.3 必改项：推理服务地址

`TSBENCHMARK_TIMER_SERVICE_BASE_URL` 指向已经运行的 timer-rest-service。

常见场景：

| 场景 | 配置值 |
| --- | --- |
| 推理服务和 TSBenchmark 在同一个 Docker 网络内 | `http://timer-service:10810` |
| 推理服务发布在宿主机 `10810` 端口 | `http://host.docker.internal:10810` |
| 推理服务在另一台 GPU 服务器 | `http://<推理服务器IP>:10810` |

示例：

```text
TSBENCHMARK_TIMER_SERVICE_BASE_URL=http://192.168.1.20:10810
```

API 前缀通常不需要改：

```text
TSBENCHMARK_TIMER_SERVICE_API_PREFIX=/ai/api/v1
```

### 7.4 可改项：访问端口

默认端口：

```text
TSBENCHMARK_FRONTEND_PUBLISHED_PORT=5173
TSBENCHMARK_BACKEND_PUBLISHED_PORT=8000
```

如果端口冲突，只改左侧宿主机端口即可。例如把前端改成 `8081`：

```text
TSBENCHMARK_FRONTEND_PUBLISHED_PORT=8081
```

访问地址就变成：

```text
http://<服务器IP>:8081
```

### 7.5 可改项：模型生命周期

默认：

```text
TSBENCHMARK_MODEL_LIFECYCLE_MODE=sequential_unload
```

含义：

- 每次评测按模型逐个加载、推理、卸载。
- 好处是降低推理服务显存占用峰值。
- 如果推理服务希望模型常驻，可改成：

```text
TSBENCHMARK_MODEL_LIFECYCLE_MODE=keep_loaded
```

---

## 8. 启动服务

启动前确认端口未被占用：

```bash
ss -tlnp | grep -E ':(5173|8000) ' || echo "ports free"
```

启动：

```bash
cd /opt/tsbenchmark-deploy/tsbenchmark-docker
docker compose up -d
```

查看容器状态：

```bash
docker compose ps
```

期望看到：

```text
tsbenchmark-backend    Up
tsbenchmark-frontend   Up
```

查看日志：

```bash
docker compose logs -f backend
docker compose logs -f frontend
```

---

## 9. 验证部署

### 9.1 验证前端页面

浏览器访问：

```text
http://<服务器IP>:5173
```

如果改过 `TSBENCHMARK_FRONTEND_PUBLISHED_PORT`，请使用修改后的端口。

首次登录：

```text
用户名：admin
密码：.env 中的 TSBENCHMARK_ADMIN_PASSWORD
```

### 9.2 验证后端 API

打开：

```text
http://<服务器IP>:8000/docs
```

或在服务器上执行：

```bash
curl -s http://127.0.0.1:8000/models
```

如果未登录，部分接口返回认证错误是正常的。

### 9.3 验证推理服务连通性

先在宿主机上验证推理服务：

```bash
curl -s http://<推理服务地址>:10810/ai/api/v1/models/list
```

再验证后端容器内能访问推理服务。假设 `.env` 中配置的是 `http://host.docker.internal:10810`：

```bash
docker compose exec backend python - <<'PY'
import os
import httpx
base = os.environ["TSBENCHMARK_TIMER_SERVICE_BASE_URL"].rstrip("/")
prefix = os.environ.get("TSBENCHMARK_TIMER_SERVICE_API_PREFIX", "/ai/api/v1").strip("/")
url = f"{base}/{prefix}/models/list"
resp = httpx.get(url, timeout=10)
print(resp.status_code)
print(resp.text[:500])
PY
```

能返回 `200` 或包含模型列表，说明后端到推理服务网络是通的。

---

## 10. 使用方式速览

1. 浏览器打开 `http://<服务器IP>:5173`。
2. 使用 `admin` 和 `.env` 里的管理员密码登录。
3. 在左侧进入“新建评测”或“数据集”。
4. 创建评测赛道。
5. 上传 CSV 或 TsFile。
6. 选择一个目标列（Target column），设置切片参数：
   - Context：历史窗口长度
   - Horizon：预测长度
   - Stride：滑窗步长
   - Max samples：最多生成多少个样本，可留空
7. 生成并选择测试用例集。
8. 选择模型并启动评测。
9. 评测完成后查看报告、榜单和样本预测曲线。

说明：

- 当前 MVP 只支持单变量目标列，一次评测只能选择一个 target。
- TsFile 大文件可以直接从页面上传；前端 nginx 上传上限已配置为 `2g`。
- 上传后的文件、SQLite 数据库、报告和预测产物都存放在 Docker volume 中，不在镜像里。

---

## 11. 数据持久化位置

Compose 文件中有一个命名卷：

```yaml
volumes:
  tsbenchmark-runtime:
```

后端容器内路径：

```text
/var/lib/tsbenchmark
```

里面主要包含：

```text
tsbenchmark.db
uploads/
samples/
forecasts/
reports/
```

查看 volume：

```bash
docker volume ls | grep tsbenchmark
docker volume inspect tsbenchmark_tsbenchmark-runtime
```

如果现场的 Compose project name 不是默认值，实际 volume 名称可能不是
`tsbenchmark_tsbenchmark-runtime`。以 `docker volume ls | grep tsbenchmark`
看到的名字为准。

注意：

- `docker compose down` 不会删除这个 volume，数据会保留。
- `docker compose down -v` 会删除这个 volume，SQLite、上传文件和报告都会被清空。
- 不要直接把 TsFile/CSV 复制进 volume 后期待它出现在前端列表里。前端数据集列表来自 SQLite 元数据，文件需要通过页面上传或 API 上传，系统才会同时写入文件和数据库记录。

---

## 12. 停止、重启、卸载

### 12.1 停止服务但保留数据

```bash
cd /opt/tsbenchmark-deploy/tsbenchmark-docker
docker compose down
```

### 12.2 重新启动

```bash
docker compose up -d
```

### 12.3 清空数据后卸载

如果确认不再保留测试数据：

```bash
docker compose down -v
```

### 12.4 删除镜像

```bash
docker rmi tsbenchmark-backend:latest tsbenchmark-frontend:latest
```

如果镜像正在被容器使用，需要先 `docker compose down`。

---

## 13. 升级流程

升级前建议先备份 runtime volume。最简单方式是导出 volume 内容：

```bash
TS=$(date +%Y%m%d%H%M%S)
mkdir -p /backup
docker run --rm \
  -v tsbenchmark_tsbenchmark-runtime:/data:ro \
  -v /backup:/backup \
  busybox tar -czf /backup/tsbenchmark-runtime-${TS}.tar.gz -C /data .
```

升级步骤：

```bash
cd /opt/tsbenchmark-deploy
tar -xzf 新版本-tsbenchmark-docker-deploy.tar.gz
cd tsbenchmark-docker

docker compose down
docker load -i tsbenchmark-backend.tar
docker load -i tsbenchmark-frontend.tar
docker compose up -d
docker compose ps
```

如果新版本提供了新的 `.env` 示例，请对比旧 `.env`，只合并新增配置，不要直接覆盖现场密码和推理服务地址。

---

## 14. 常见问题

### 14.1 忘记 admin 密码

`TSBENCHMARK_ADMIN_PASSWORD` 只在第一次初始化数据库时生效。系统启动过以后，再改 `.env` 不会修改已有密码。

如果只是试用环境，可以清空数据后重新初始化：

```bash
docker compose down -v
docker compose up -d
```

注意：这会删除所有已上传数据集、切片、评测结果和报告。

### 14.2 模型列表加载失败

大概率是后端连不上推理服务。

检查 `.env`：

```bash
grep TSBENCHMARK_TIMER_SERVICE_BASE_URL .env
```

检查容器内访问：

```bash
docker compose exec backend python - <<'PY'
import os
import httpx
base = os.environ["TSBENCHMARK_TIMER_SERVICE_BASE_URL"].rstrip("/")
prefix = os.environ.get("TSBENCHMARK_TIMER_SERVICE_API_PREFIX", "/ai/api/v1").strip("/")
url = f"{base}/{prefix}/models/list"
print(url)
print(httpx.get(url, timeout=10).status_code)
PY
```

如果不通，请确认：

- 推理服务容器是否运行；
- 推理服务端口是否发布；
- `TSBENCHMARK_TIMER_SERVICE_BASE_URL` 是否填成后端容器可访问的地址；
- 如果推理服务在宿主机，Linux 下建议使用 `http://host.docker.internal:10810`。

### 14.3 上传大 TsFile 失败

TSBenchmark 前端 nginx 已允许最大 `2g` 上传。如果仍失败：

```bash
docker compose logs frontend --tail=100
docker compose logs backend --tail=100
```

同时确认：

- 浏览器或反向代理没有额外限制上传大小；
- 服务器磁盘空间足够；
- `tsbenchmark-runtime` volume 所在磁盘没有写满。

### 14.4 端口冲突

如果 `5173` 或 `8000` 被占用，修改 `.env`：

```text
TSBENCHMARK_FRONTEND_PUBLISHED_PORT=8081
TSBENCHMARK_BACKEND_PUBLISHED_PORT=18000
```

重启：

```bash
docker compose down
docker compose up -d
```

访问地址改为：

```text
http://<服务器IP>:8081
```

---

## 15. 备份建议

试用环境至少备份 Docker volume `tsbenchmark-runtime`。它包含：

- SQLite 数据库；
- 上传的数据文件；
- 评测预测文件；
- 报告文件。

备份命令示例：

```bash
TS=$(date +%Y%m%d%H%M%S)
mkdir -p /backup
docker run --rm \
  -v tsbenchmark_tsbenchmark-runtime:/data:ro \
  -v /backup:/backup \
  busybox tar -czf /backup/tsbenchmark-runtime-${TS}.tar.gz -C /data .
```

恢复时需要先停服务，再把备份内容解回 volume。恢复操作建议由运维或交付方协助执行。
