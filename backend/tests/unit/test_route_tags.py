"""路由标签护栏：所有业务路由必须是 TieredRoute，且 tier/perm 标签合法。

设计稿 §1 / §2.3：默认 tier="authed"，漏标的路由不会变成 public。
本测试遍历 app.routes，断言：
  - 是 TieredRoute（即用了 make_router 工厂）
  - tier ∈ {"public", "authed", "perm"}
  - 当 tier == "perm" 时 perm 必须是已知权限码

例外白名单（plain APIRoute）：
  - /openapi.json, /docs, /redoc, /docs/oauth2-redirect 等 FastAPI 内置路由
  - /__test__/error-contract 测试探针（不通过 make_router 挂载）
"""

import pytest
from fastapi.routing import APIRoute

from app.core.permissions import ALL_CODES
from app.core.route_class import TieredRoute, VALID_TIERS
from app.main import create_app


# 这些路径不受 TieredRoute 保护，需在此列出。新增 public 路由请走 tier="public"。
WHITELIST_PATHS: frozenset[str] = frozenset(
    {
        "/openapi.json",
        "/docs",
        "/docs/oauth2-redirect",
        "/redoc",
        "/__test__/error-contract",
    }
)


@pytest.fixture
def routes(app):
    return [r for r in app.routes if isinstance(r, APIRoute)]


def test_every_business_route_is_a_tiered_route(routes):
    not_tiered = [
        r.path
        for r in routes
        if not isinstance(r, TieredRoute) and r.path not in WHITELIST_PATHS
    ]
    assert not_tiered == [], (
        f"these routes bypass the auth gate (must use make_router or be added to "
        f"WHITELIST_PATHS): {not_tiered}"
    )


def test_every_tiered_route_has_valid_tier(routes):
    bad = [(r.path, r.tier) for r in routes if isinstance(r, TieredRoute) and r.tier not in VALID_TIERS]
    assert bad == [], f"routes with invalid tier: {bad}"


def test_perm_tier_routes_reference_known_permission_codes(routes):
    bad = []
    for r in routes:
        if not isinstance(r, TieredRoute) or r.tier != "perm":
            continue
        if r.perm is None:
            bad.append((r.path, "missing perm= when tier='perm'"))
        elif r.perm not in ALL_CODES:
            bad.append((r.path, f"unknown perm code: {r.perm}"))
    assert bad == [], f"perm routes with bad permission code: {bad}"


def test_non_perm_tier_routes_have_no_perm(routes):
    """tier != 'perm' 的路由不应该带 perm=（防止误配置）。"""
    bad = [
        (r.path, r.tier, r.perm)
        for r in routes
        if isinstance(r, TieredRoute) and r.tier != "perm" and r.perm is not None
    ]
    assert bad == [], f"non-perm routes should not declare perm=: {bad}"
