from __future__ import annotations

import json
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
LEADERBOARD = ROOT / "paper_results" / "leaderboard"


def load_data() -> dict:
    text = (LEADERBOARD / "leaderboard-data.js").read_text(encoding="utf-8")
    prefix = "window.CAFE_LEADERBOARD_DATA = "
    assert text.startswith(prefix)
    assert text.endswith(";\n")
    return json.loads(text[len(prefix) : -2])


def test_public_leaderboard_shape_and_coverage() -> None:
    data = load_data()
    assert data["schemaVersion"] == "cafe.public_leaderboard.v1"
    assert len(data["models"]) == 6
    assert len(data["capabilities"]) == 8
    assert len(data["suites"]) == 5
    assert set(data["metrics"]) == {
        "reference_mase",
        "probe_mase",
        "paired_nrmse",
    }
    assert len(data["overall"]["all"]) == 6
    tirex = next(row for row in data["overall"]["all"] if row["model"] == "tirex2")
    assert tirex["coverage"]["paired_nrmse"] == 2
    assert tirex["suiteCount"] == 4


def test_leaderboard_and_reviewer_home_are_public_only() -> None:
    paths = [
        ROOT / "README.md",
        *(path for path in LEADERBOARD.iterdir() if path.is_file()),
    ]
    rendered = "\n".join(path.read_text(encoding="utf-8") for path in paths)
    assert "Timer-4.0" not in rendered
    assert "/data/xmy" not in rendered
    assert "timecho" not in rendered
    assert "## History" not in rendered
    assert "paper_results/leaderboard/index.html" in rendered
    assert "reproducibility/README.md" in rendered


def test_leaderboard_assets_are_self_contained() -> None:
    html = (LEADERBOARD / "index.html").read_text(encoding="utf-8")
    assert 'src="leaderboard-data.js"' in html
    assert 'src="app.js"' in html
    assert 'href="styles.css"' in html
    assert "https://" not in html
