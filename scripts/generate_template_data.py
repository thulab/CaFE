"""生成 TSBenchmark 模板数据集（确定性、可复跑）。

每份 CSV 符合平台 CSV 输入契约：第一列为时间列（严格递增、等间隔），其余为
数值列（目标 + 可选协变量），全部有限 float。覆盖不同频率与时序形态，便于
上手试跑 upload → load → track → run → ranking。

用法：python3 scripts/generate_template_data.py [out_dir]   默认 ./templates
"""
from __future__ import annotations

import csv
import math
import random
import sys
from datetime import datetime, timedelta
from pathlib import Path

START = datetime(2026, 1, 1, 0, 0, 0)


def _ts(start: datetime, step: timedelta, count: int) -> list[str]:
    return [(start + step * i).strftime("%Y-%m-%d %H:%M:%S") for i in range(count)]


def _write(out_dir: Path, name: str, header: list[str], rows: list[list]) -> Path:
    path = out_dir / name
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.writer(handle, lineterminator="\n")
        writer.writerow(header)
        for row in rows:
            writer.writerow([row[0]] + [f"{value:.4f}" for value in row[1:]])
    return path


def hourly_trend(out_dir: Path) -> Path:
    """小时频 · 线性上升趋势 + 噪声。target 与温度协变量。"""
    rng = random.Random(11)
    n = 240  # 10 天
    times = _ts(START, timedelta(hours=1), n)
    rows = []
    for i, t in enumerate(times):
        target = 100.0 + 0.5 * i + rng.uniform(-1.5, 1.5)
        temperature = 15.0 + 5.0 * math.sin(2 * math.pi * (i % 24) / 24) + rng.uniform(-0.6, 0.6)
        rows.append([t, target, temperature])
    return _write(out_dir, "hourly_trend.csv", ["time", "target", "temperature"], rows)


def hourly_daily_seasonality(out_dir: Path) -> Path:
    """小时频 · 日内（24h）季节性 + 缓趋势。target + 温度 + 负荷协变量。"""
    rng = random.Random(23)
    n = 336  # 14 天
    times = _ts(START, timedelta(hours=1), n)
    rows = []
    for i, t in enumerate(times):
        hour = i % 24
        season = 30.0 * math.sin(2 * math.pi * hour / 24)
        target = 200.0 + season + 0.15 * i + rng.uniform(-4.0, 4.0)
        temperature = 12.0 + 8.0 * math.sin(2 * math.pi * (hour - 3) / 24) + rng.uniform(-0.8, 0.8)
        load = 0.6 * target + 10.0 * math.cos(2 * math.pi * hour / 24) + rng.uniform(-2.0, 2.0)
        rows.append([t, target, temperature, load])
    return _write(out_dir, "hourly_daily_seasonality.csv", ["time", "target", "temperature", "load"], rows)


def daily_weekly_seasonality(out_dir: Path) -> Path:
    """日频 · 周内（7d）季节性 + 趋势。零售口径：sales + 周末指示协变量。"""
    rng = random.Random(37)
    n = 168  # 24 周
    times = _ts(START, timedelta(days=1), n)
    rows = []
    for i, t in enumerate(times):
        dow = i % 7
        weekly = 120.0 * math.sin(2 * math.pi * dow / 7)
        sales = 500.0 + weekly + 1.2 * i + rng.uniform(-25.0, 25.0)
        is_weekend = 1.0 if dow in (5, 6) else 0.0
        rows.append([t, sales, is_weekend])
    return _write(out_dir, "daily_weekly_seasonality.csv", ["time", "sales", "is_weekend"], rows)


def multivariate_hourly(out_dir: Path) -> Path:
    """小时频 · 多协变量（target + 3 协变量），演示全列摄入。"""
    rng = random.Random(53)
    n = 240
    times = _ts(START, timedelta(hours=1), n)
    rows = []
    for i, t in enumerate(times):
        hour = i % 24
        target = 50.0 + 10.0 * math.sin(2 * math.pi * hour / 24) + 0.1 * i + rng.uniform(-2.0, 2.0)
        cov_a = 5.0 + 2.0 * math.cos(2 * math.pi * hour / 24) + rng.uniform(-0.5, 0.5)
        cov_b = 0.3 * target + rng.uniform(-1.0, 1.0)
        cov_c = float(hour)  # 确定性外生特征（当天小时）
        rows.append([t, target, cov_a, cov_b, cov_c])
    return _write(out_dir, "multivariate_hourly.csv", ["time", "target", "cov_a", "cov_b", "cov_c"], rows)


def main() -> int:
    out_dir = Path(sys.argv[1]) if len(sys.argv) > 1 else Path(__file__).resolve().parent.parent / "templates"
    out_dir.mkdir(parents=True, exist_ok=True)
    for builder in (hourly_trend, hourly_daily_seasonality, daily_weekly_seasonality, multivariate_hourly):
        path = builder(out_dir)
        print(f"wrote {path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
