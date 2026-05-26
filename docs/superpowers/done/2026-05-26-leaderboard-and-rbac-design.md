# Leaderboard 展示页 + RBAC 重构 · 设计稿

- 日期：2026-05-26
- 状态：设计已敲钉，待进入实施
- 范围：一次性大重构（不分期），同时落地「榜单总览页面」+「完整可用 RBAC 含登录」
- 关联文档：
  - `docs/developer/data-model.md`（现有数据模型）
  - `docs/developer/key-flows.md`（现有关键流程）
  - `AGENTS.md`（工程约定）

---

## 0 · 重构原则

> **默认公开**的是「**展示**」（榜单 / 排名 / 公开的 track 元数据），**默认登录**的是「**工作台**」（数据集、运行、报告），**默认权限**的是「**会写后端 / 触发处理 / 改人改权**」的所有动作。

不引入资源 owner 字段，所有工作台数据在登录用户之间全局共享；权限只控 action，不控 ownership。

---

## 1 · 三层访问模型 + AOP 切面实现（方案 C）

```
Tier 0 · Public        匿名可访问，无需 token
Tier 1 · Authenticated 任意有效 token
Tier 2 · Permissioned  token + 特定权限码
```

### 实现：自定义 APIRoute 子类（TieredRoute）

所有请求统一过一个切面，鉴权逻辑只在一处。路由元数据贴在路由旁，不集中到 path→perm 映射表里。

`backend/app/core/route_class.py`（新文件）：

```python
from typing import Literal
from fastapi.routing import APIRoute

Tier = Literal["public", "authed", "perm"]

class TieredRoute(APIRoute):
    tier: Tier = "authed"      # 默认 Tier 1：漏标=要登录，不是漏标=公开
    perm: str | None = None    # tier == "perm" 时必填

    def get_route_handler(self):
        original = super().get_route_handler()
        async def handler(request):
            user = resolve_token(request)         # 解 token → User | None
            if self.tier == "public":
                pass
            elif user is None:
                raise ApiError("auth_required", "login required", status=401)
            elif self.tier == "perm":
                if not (user.is_superuser or self.perm in user_permission_codes(user)):
                    raise ApiError("forbidden", "missing permission",
                                   {"required_permission": self.perm}, status=403)
            request.state.user = user
            return await original(request)
        return handler
```

`backend/app/api/router_factory.py`：

```python
def make_router(**kwargs) -> APIRouter:
    return APIRouter(route_class=TieredRoute, **kwargs)
```

路由文件改写示例：

```python
from app.api.router_factory import make_router
router = make_router(prefix="/benchmarking-runs", tags=["runs"])

@router.get("", tier="authed")
def list_runs(...): ...

@router.post("", tier="perm", perm="run.execute")
def create_run(...): ...

@router.get("/ranking-lists", tier="public")        # ranking_lists 路由
def list_ranking_lists(...): ...
```

错误码沿用 `ApiError`：
- `auth_required` 401 — `{"hint": "login or refresh token"}`
- `forbidden` 403 — `{"required_permission": "run.execute"}`

### Dep 辅助（路由 handler 内取当前用户用）

```python
def current_user(request: Request) -> User:
    user = getattr(request.state, "user", None)
    if user is None:
        raise ApiError("auth_required", ..., 401)
    return user
```

不需要 `get_current_user_optional` / `require_permission` 等 dep——切面已经处理。handler 里若要拿当前用户对象，注入 `current_user` dep。

---

## 2 · 后端路由分层

### Tier 0 · Public

| 路由 | 备注 |
|---|---|
| `POST /auth/login` | 登录入口 |
| `GET /ranking-lists` | **新增**，榜单总览 |
| `GET /tracks/{id}/ranking` | 已存在 |
| `GET /tracks/{id}` | **新增**，榜单页头需要 track 元数据 |
| `GET /tracks` | **新增**，榜单总览可能用作过滤源 |
| `GET /models` | 已存在，榜单要显示模型名 |
| `GET /__test__/error-contract` | 测试探针 |

### Tier 1 · Authenticated

| 路由 |
|---|
| `GET /auth/me` |
| `POST /auth/password` |
| `POST /auth/logout`（可选；JWT 无状态可省） |
| `GET /dataset-manifests`, `/dataset-manifests/{id}` |
| `GET /dataset-load-jobs/{id}` |
| `GET /shards/{id}` |
| `GET /samples/{id}` |
| `GET /benchmarking-runs`, `/benchmarking-runs/{id}` |
| `GET /reports/{id}` |
| `GET /capability-blocks` |

### Tier 2 · Permissioned

| 路由 | 权限码 |
|---|---|
| `POST /dataset-manifests` | `dataset.write` |
| `DELETE /dataset-manifests/{id}` | `dataset.delete` |
| `POST /dataset-load-jobs` | `dataset.write` |
| `POST /shards` | `shard.write` |
| `POST /capability-blocks` | `track.manage` |
| `POST /tracks` | `track.manage` |
| `POST /models` | `model.register` |
| `POST /wizard/*` | `run.execute` |
| `POST /benchmarking-runs` | `run.execute` |
| `POST /benchmarking-runs/{id}/cancel` | `run.cancel` |
| `POST /reports`, `PATCH /reports/{id}` | `report.write` |
| `GET/POST/PATCH/DELETE /users`, `PUT /users/{id}/roles` | `user.manage` |
| `GET/POST/PATCH/DELETE /roles`, `PUT /roles/{id}/permissions` | `role.manage` |
| `GET /permissions` | `role.manage` |

### 路由标记护栏

每条路由通过 `TieredRoute.tier` / `TieredRoute.perm` 属性声明 Tier。

新增 `backend/tests/unit/test_route_tags.py`：

1. 遍历 `app.routes`，断言所有业务路由均为 `TieredRoute` 实例（非 `TieredRoute` 直接 fail）。
2. 断言 `tier ∈ {"public", "authed", "perm"}`；当 `tier == "perm"` 时 `perm` 必须是已知权限码。
3. **默认 tier="authed"** 意味着漏标的新路由不会变成 public——漏标的安全语义反转了。

这是大重构里最关键的护栏，比"漏标 fail"更进一步——漏标默认要登录，符合"工作台默认登录"的总纲。

---

## 3 · 数据模型

新文件 `backend/app/models/auth.py`：

```
User
  user_id (PK, default_factory=new_id)
  username      unique index
  email         unique nullable
  password_hash bcrypt
  is_active     bool default True
  is_superuser  bool default False     绕过所有权限检查
  created_at / updated_at

Role
  role_id (PK)
  name          unique
  description   nullable
  is_system     bool                   system 角色不可删
  created_at

Permission                              启动 seed，不让用户自造
  permission_id (PK)
  code          unique                 'dataset.read' 等
  description

UserRole (复合 PK)
  user_id, role_id

RolePermission (复合 PK)
  role_id, permission_id
```

不引入 `owner_user_id` / `created_by_user_id` 字段。已确认。

---

## 4 · 权限码枚举（初版）

```
dataset.read, dataset.write, dataset.delete
shard.read,   shard.write
track.read,   track.manage
run.read,     run.execute,  run.cancel
ranking.read    ← 注：Tier 0 接口不查权限，此码留作未来精细化用
report.read,  report.write
model.read,   model.register
user.manage,  role.manage
```

---

## 5 · 系统角色 seed（MVP 仅 2 角色）

```
admin   ── is_superuser=True，自动绕过所有 perm 检查
            RolePermission 仍 seed 全部权限码（结构完整，未来取消 is_superuser 也能用）
viewer  ── RolePermission 仅 seed *.read 权限码
```

权限矩阵：

| code | admin | viewer |
|---|---|---|
| `*.read`（dataset/shard/track/run/ranking/report/model） | ✓ | ✓ |
| `dataset.write` / `dataset.delete` | ✓ | — |
| `shard.write` | ✓ | — |
| `track.manage` / `model.register` | ✓ | — |
| `run.execute` / `run.cancel` | ✓ | — |
| `report.write` | ✓ | — |
| `user.manage` / `role.manage` | ✓ | — |

匿名用户不绑定任何角色，只能命中 Tier 0。

两个角色都是 `is_system = True`：**不能删除、不能改 perm 集合**（API 返回 409）。未来若需第三个角色，由 admin 通过 `POST /roles` 创建自定义角色（schema 已就绪，无需 migration）。

---

## 6 · 启动初始化

```
init_db
  → seed_permissions          确保 Permission 表覆盖代码里枚举的所有 code
  → seed_system_roles         admin / viewer（is_system=True）不存在则建
                              admin 绑全部权限码（虽然 is_superuser 已绕过，结构对齐）
                              viewer 仅绑 *.read
  → seed_admin_if_empty       User 表空 → 自动建 admin（is_superuser=True，bind admin role）
                              密码：env TSBENCHMARK_ADMIN_PASSWORD 优先
                                    否则随机生成并打印到 stdout 一次
```

JWT：
- 算法 HS256
- 密钥读 env `TSBENCHMARK_AUTH_SECRET`，缺省时启动报错（不静默走默认密钥）
- 过期默认 8h，可配 `TSBENCHMARK_AUTH_TTL_SECONDS`

**不保留 `TSBENCHMARK_AUTH_DISABLED` 开关。** 测试/脚本一律走 login 拿 token。

---

## 7 · 鉴权接口

```
POST   /auth/login         { username, password }
                            → { access_token, token_type:"bearer", expires_in }
GET    /auth/me             → { user_id, username, roles:[...], permissions:[...] }
POST   /auth/password       { old, new }   改自己密码

# user.manage
GET    /users
POST   /users
GET    /users/{id}
PATCH  /users/{id}                          改基本信息 / 启停
DELETE /users/{id}
PUT    /users/{id}/roles    { role_ids }    赋角色
POST   /users/{id}/reset-password           管理员重置

# role.manage
GET    /roles                               admin/viewer 都返回
POST   /roles                               MVP 不在 UI 暴露，但接口保留（创建自定义角色）
PATCH  /roles/{id}                          system 角色返回 409
DELETE /roles/{id}                          system 角色返回 409
PUT    /roles/{id}/permissions { codes }    system 角色返回 409
GET    /permissions                         列权限码字典（前端 RolesPage 展示用）
```

---

## 8 · 前端路由分层

### 公开

| 路由 | 页面 |
|---|---|
| `#/leaderboards` | LeaderboardsPage（新增） |
| `#/tracks/{id}/ranking` | RankingPage（已存在） |
| `#/rankings/{id}` | RankingPage（别名） |
| `#/login` | LoginPage（新增） |

### 需登录（任意角色）

| 路由 | 页面 |
|---|---|
| `#/` | HomePage（工作台首页） |
| `#/datasets`, `#/datasets/{id}` | 数据集 |
| `#/runs`, `#/runs/{id}` | 运行 |
| `#/load-jobs/{id}`, `#/shards/{id}`, `#/samples/{id}` | 中间产物 |
| `#/tracks/{id}` | track 详情 |
| `#/reports/{id}` | 报告 |
| `#/profile` | 个人 |

### 需特定权限

| 路由 | 权限 |
|---|---|
| `#/new` | `run.execute` |
| `#/admin/users` | `user.manage` |
| `#/admin/roles` | `role.manage` |

### 匿名首屏

匿名访问 `#/` **重定向到 `#/leaderboards`**。HomePage 不做双态，语义保持「工作台首页」。

### 应用壳两态

**匿名**：侧边栏只显示 brand + 一个「Leaderboards」链接 + 底部 Sign in；顶栏右侧只有「Sign in」+ 主题切换；不显示 New evaluation 按钮。

**已登录**：完整 Overview / New evaluation / Datasets / Runs / Leaderboards；admin 角色多一组 Administration（Users / Roles）；footer 显示用户名 + role chips + 登出。

`+ New evaluation` 按钮：未登录隐藏；登录但无 `run.execute` 时 disabled + tooltip "Requires run.execute"。

### 路由守卫

```
匹配路由 → 决定 Tier
  Tier 0 (public): 直接渲染
  Tier 1 (authed): user==null → 跳 #/login?next=<hash>
  Tier 2 (perm):   user==null → 跳 #/login?next=<hash>
                   has(code)==false → 跳 #/forbidden（新增极简页）
登录成功后回跳 next 参数对应的 hash。
```

---

## 9 · 榜单总览页面（LeaderboardsPage）

### 后端 `GET /ranking-lists`

响应（每条）：
```json
{
  "ranking_list_id": "...",
  "track_id": "...",
  "track_name": "ETT-h1 forecast",
  "track_type": "real_dataset",
  "primary_metric_id": "mase",
  "default_policy": "latest_valid_result",
  "updated_at": "2026-05-26T...",
  "model_count": 8,
  "run_count": 3,
  "top": [
    { "rank": 1, "model_id": "...", "metric_value": 0.412 },
    { "rank": 2, "model_id": "...", "metric_value": 0.487 },
    { "rank": 3, "model_id": "...", "metric_value": 0.612 }
  ]
}
```

实现要点：
- 复用 `RankingList`（status=active）+ `JOIN Track` 取名字与主指标。
- Top 3 走现有 `query_ranking(session, track_id, primary_metric, default_policy)[:3]`——避免引入新查询路径，保持和细节页同一份真相。
- `model_count` / `run_count` 用 `RankingEntry` 聚合（`COUNT(DISTINCT model_id)` / `COUNT(DISTINCT benchmarking_run_id)`）。

### 前端布局（low-fi）

```
┌──────────────────────────────────────────────────────────────┐
│ Results                                                       │
│ Leaderboards                                                  │
│ Compare model performance across all benchmark tracks.        │
├──────────────────────────────────────────────────────────────┤
│ [search]  Track type ▾  Primary metric ▾    12 boards         │
├──────────────────────────────────────────────────────────────┤
│ ┌──────── card ────────┐  ┌──────── card ────────┐           │
│ │ ETT-h1 forecast      │  │ Weather synthetic     │           │
│ │ real_dataset · MASE  │  │ synthetic · MSE       │           │
│ │ updated 2h ago       │  │ updated yesterday     │           │
│ │ 🥇 TimerXL  0.412 ▓▓ │  │ 🥇 Chronos  0.183 ▓▓ │           │
│ │ 🥈 Moirai   0.487 ▓▓ │  │ 🥈 TimerXL  0.201 ▓▓ │           │
│ │ 🥉 NaiveSe  0.612 ▓░ │  │ 🥉 Naive    0.290 ▓░ │           │
│ │ 8 models · 3 runs    │  │ 5 models · 1 run      │           │
│ │ [ View full board → ]│  │ [ View full board → ] │           │
│ └──────────────────────┘  └───────────────────────┘           │
└──────────────────────────────────────────────────────────────┘
```

复用样式 token：`.card` / `.rank-badge`（金/银/铜）/ `.bar-fill` / `.grid-auto`。

空态：`StateBlock` empty + CTA「Sign in to start an evaluation」（匿名）或「Start a new evaluation」（登录）。

---

## 10 · 管理页面布局

### Users 页（可编辑）

```
Administration                            [+ New user]
Users  |  Roles  |  My profile
─────────────────────────────────────────────────────
Username   Email           Role     Status   ⋯
alice      alice@x.com    admin     Active   ⋯
bob        bob@x.com      viewer    Active   ⋯
carol      —              viewer    Disabled ⋯
```

点行 → 右侧抽屉：基本信息 / 角色单选（admin / viewer）/ 重置密码 / 启停。

> MVP 每个用户绑一个角色（数据模型保留 m:n 关联以备扩展，UI 暂只暴露单选）。

### Roles 页（**只读**）

MVP 阶段两个 system 角色不可编辑，本页面为只读展示，让 admin 快速知道每个角色含哪些权限码。

```
Administration
Users  |  Roles  |  My profile
─────────────────────────────────────────────────────
┌─ Roles ──────┐  ┌─ Permissions for "admin" ───────┐
│ ▸ admin (sys)│  │ Dataset                          │
│   viewer(sys)│  │   • dataset.read                 │
└──────────────┘  │   • dataset.write                │
                  │   • dataset.delete               │
                  │ Run                              │
                  │   • run.read                     │
                  │   • run.execute                  │
                  │   • run.cancel                   │
                  │ ... （admin 全权限码）           │
                  └──────────────────────────────────┘
```

system 标识 chip 让用户明白"为何不可改"。未来一旦后端 `POST /roles` 在 UI 暴露，本页面再升级为"可编辑"——当前结构（左列角色 / 右列分组权限）已是可编辑布局的雏形。

---

## 11 · 前端工程改造清单

| 文件 | 改动 |
|---|---|
| `frontend/src/stores/auth.ts` | **新增**：token / user / perms / `has(code)` / `login()` / `logout()`；token 存 localStorage |
| `frontend/src/api/client.ts` | 改：请求拦截器附 `Authorization`；响应 401 → `logout()` + 跳 `#/login?next=<current>` |
| `frontend/src/pages/LoginPage.vue` | **新增** |
| `frontend/src/pages/ForbiddenPage.vue` | **新增**（极简） |
| `frontend/src/pages/LeaderboardsPage.vue` | **新增** |
| `frontend/src/components/results/LeaderboardCard.vue` | **新增** |
| `frontend/src/pages/admin/UsersPage.vue` | **新增** |
| `frontend/src/pages/admin/RolesPage.vue` | **新增** |
| `frontend/src/pages/admin/ProfilePage.vue` | **新增** |
| `frontend/src/api/auth.ts` | **新增**：login/me/password/users/roles/permissions |
| `frontend/src/api/results.ts` | 改：加 `listRankingLists()` |
| `frontend/src/composables/useAuthGuard.ts` | **新增**：在 `route computed` 内调用，按 Tier 重定向 |
| `frontend/src/App.vue` | 改：route computed 串入 guard；侧边栏区分匿名 / 已登录 / admin |

---

## 12 · 测试与脚本影响面

| 项 | 改动 |
|---|---|
| `backend/tests/conftest.py` | 加 `admin_token` / `viewer_token` / `anon_client` 三个 fixture |
| `backend/tests/api/*` | 现有 176 测试 90% 切到 `admin_token`（最小改动）；新增 Tier 0/1/2 边界用例（未授权 → 401，viewer 写 → 403） |
| `backend/tests/unit/test_route_tags.py` | **新增**护栏：遍历 `app.routes`，断言所有业务路由是 `TieredRoute` 实例且 tier 合法 |
| `scripts/baseline-run.sh` | 加 login 步骤，token 注入到后续 curl；env `TSBENCHMARK_BASELINE_PASSWORD` |
| `scripts/start-system.sh` | 不变 |
| `frontend/src/tests` | LoginPage / 守卫 / Leaderboards 总览单测 + E2E「未登录访问 /datasets 应弹回登录」 |

---

## 13 · 不做的事（避免 over-engineering）

- 不做资源 owner 隔离
- 不做用户邀请邮件 / 密码重置邮件 / 2FA / SSO
- 不做审计日志（如要做单独立项）
- 不做对象级权限（只有 action 级全局权限）
- 不保留 `TSBENCHMARK_AUTH_DISABLED` 开发开关
- 不在 RankingPage 上加多指标并列 / 跨 track 对比（v1 选项 b/c 未选）
- 不在前端 UI 暴露「创建自定义角色」入口（后端接口 `POST /roles` 保留，admin 可用 curl 调；UI 上 RolesPage 只读）
- 不做用户与角色的 m:n 编辑 UI（数据模型保留 m:n，UI 暂只暴露每用户单角色单选）

---

## 14 · 实施顺序（一次性合并，但内部仍有依赖序）

1. 后端：5 张表（User/Role/Permission/UserRole/RolePermission） + seed_permissions + seed_system_roles（admin/viewer）+ seed_admin_if_empty
2. 后端：`app/core/route_class.py`（TieredRoute）+ `app/api/router_factory.py`（make_router）
3. 后端：`/auth/login` + `/auth/me` + `current_user` dep
4. 后端：13 个现有路由文件改用 `make_router`，每条路由打 `tier=`/`perm=`；新增 `test_route_tags.py` 护栏
5. 后端：用户/角色/权限 CRUD（`/users` `/roles` `/permissions`）
6. 后端：`GET /ranking-lists` + `GET /tracks` + `GET /tracks/{id}`（Tier 0）
7. 后端：176 现有测试切 `admin_token` + Tier 边界测试
8. 脚本：`baseline-run.sh` 改 login + token
9. 前端：`stores/auth` + `api/client` 拦截器 + `LoginPage` + 守卫 + `ForbiddenPage`
10. 前端：`LeaderboardsPage` + `LeaderboardCard` + 侧边栏匿名/登录两态
11. 前端：admin/UsersPage（可编辑）+ admin/RolesPage（只读）+ admin/ProfilePage
12. 前端：测试与 E2E

---

## 15 · 敲钉判断（已确认）

| 项 | 决定 |
|---|---|
| 匿名访问 `#/` | 重定向到 `#/leaderboards` |
| 资源 owner 字段 | 不引入，全局共享 |
| `AUTH_DISABLED` 开关 | 不保留 |
| AOP 切面实现 | 方案 C · `TieredRoute` 自定义 APIRoute 子类 |
| 系统角色 | 2 个：admin（is_superuser）+ viewer（仅 `*.read`），均 `is_system=True` |
| RBAC 结构 | 5 张表完整保留；前端 RolesPage 只读，UsersPage 可编辑 |
| 设计落档 | 本文件即为准 |
