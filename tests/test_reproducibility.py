from __future__ import annotations

import importlib.util
import json
import subprocess
import sys
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
REPRODUCIBILITY = ROOT / "reproducibility"


def load_module(name: str, path: Path):
    spec = importlib.util.spec_from_file_location(name, path)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def test_reviewer_commands_use_only_public_models() -> None:
    module = load_module("cafe_reproduce", REPRODUCIBILITY / "reproduce.py")
    for name in module.CONFIG["main_experiments"]:
        command = module.command_for(name)
        assert "Timer-4.0" not in command
        assert not any("/data/xmy" in value for value in command)
        for model in module.CONFIG["public_models"]:
            assert model in command


def test_finetuning_config_pins_model_and_data() -> None:
    path = REPRODUCIBILITY / "chronos2_finetuning" / "config.json"
    config = json.loads(path.read_text(encoding="utf-8"))
    assert len(config["upstream"]["commit"]) == 40
    assert len(config["base_model"]["revision"]) == 40
    assert len(config["base_model"]["weights_sha256"]) == 64
    assert config["data"]["fit"]["treatment_count"] == 50_535
    assert config["data"]["evaluation"]["treatment_count"] == 48_365
    assert config["shared_training"]["steps"] == 40_000


def test_frozen_results_verify() -> None:
    completed = subprocess.run(
        [sys.executable, str(REPRODUCIBILITY / "reproduce.py"), "verify"],
        cwd=ROOT,
        text=True,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        check=False,
    )
    assert completed.returncode == 0, completed.stderr
    assert "verified" in completed.stdout


def test_finetuning_dry_run_is_self_contained(tmp_path: Path) -> None:
    script = REPRODUCIBILITY / "chronos2_finetuning" / "run_finetuning.py"
    completed = subprocess.run(
        [
            sys.executable,
            str(script),
            "all",
            "--cafe-root",
            str(ROOT),
            "--chronos-root",
            str(tmp_path / "chronos"),
            "--work-root",
            str(tmp_path / "work"),
            "--dry-run",
        ],
        cwd=ROOT,
        text=True,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        check=False,
    )
    assert completed.returncode == 0, completed.stderr
    assert "gift-v15-short-qualified-feasible-moirai16k-seed2026082701-r1" in completed.stdout
    assert "gift-v15-short-qualified-feasible-moirai16k-seed2026082702-r1" in completed.stdout
    assert "--num-steps 40000" in completed.stdout
    assert "/data/xmy" not in completed.stdout
