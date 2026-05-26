"""权限码枚举（platform-wide）。

Tier 0 路由不消费此表。Tier 1 仅要求登录。Tier 2 路由在请求时检查
user.is_superuser 或 code ∈ 用户权限集。
"""

ALL_PERMISSION_CODES: list[tuple[str, str]] = [
    ("dataset.read", "read dataset manifests / shards / samples"),
    ("dataset.write", "create dataset manifests / load jobs"),
    ("dataset.delete", "delete dataset manifests"),
    ("shard.read", "read shards"),
    ("shard.write", "create / modify shards"),
    ("track.read", "read tracks"),
    ("track.manage", "create / modify tracks and capability blocks"),
    ("run.read", "read benchmarking runs"),
    ("run.execute", "start benchmarking runs (wizard / direct)"),
    ("run.cancel", "cancel running benchmarking runs"),
    ("ranking.read", "read rankings (reserved; Tier 0 routes don't check)"),
    ("report.read", "read reports"),
    ("report.write", "create / edit reports"),
    ("model.read", "read model registry"),
    ("model.register", "register new model adapters"),
    ("user.manage", "manage users (CRUD, assign roles, reset passwords)"),
    ("role.manage", "manage roles and their permission sets"),
]

ADMIN_ROLE = "admin"
VIEWER_ROLE = "viewer"

# viewer 拿所有 *.read 权限码
VIEWER_CODES: set[str] = {code for code, _ in ALL_PERMISSION_CODES if code.endswith(".read")}
ALL_CODES: set[str] = {code for code, _ in ALL_PERMISSION_CODES}
