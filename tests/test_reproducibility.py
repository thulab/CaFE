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
    sys.modules[name] = module
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

    stability = module.CONFIG["stability"]
    assert stability["experiment_id_template"] == (
        "gift-v15-short-stability10-head78ef32f-seed{seed}"
    )


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


def test_smoke_check_accepts_complete_bounded_run(tmp_path: Path) -> None:
    module = load_module("cafe_reproduce_smoke", REPRODUCIBILITY / "reproduce.py")
    root = tmp_path / "reviewer-smoke"
    dataset = root / "gift_ett1_h"
    (root / "stage_contracts").mkdir(parents=True)
    (root / "experiment.json").write_text("{}\n", encoding="utf-8")
    for stage in ("generation", "validation", "inference", "analysis"):
        (root / "stage_contracts" / f"{stage}.json").write_text(
            "{}\n", encoding="utf-8"
        )
    records = {
        "01_generation/manifest.json": {
            "official_instance_count": 1,
            "treatment_count": 5,
            "config": {"capability_ids": ["trend"], "max_instances": 1},
        },
        "02_validation/report.json": {"accepted": True},
        "03_inference/manifest.json": {
            "complete": True,
            "model_statuses": [
                {"model_id": "Chronos-2", "status": "complete", "failure_count": 0}
            ],
        },
        "04_analysis/manifest.json": {"schema_version": "test"},
    }
    for relative, record in records.items():
        path = dataset / relative
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps(record), encoding="utf-8")
    suite = root / "04_analysis_suite" / "task_equal_summary.json"
    suite.parent.mkdir(parents=True)
    suite.write_text("{}\n", encoding="utf-8")

    assert module.smoke_check(root) == 0


def test_smoke_check_rejects_incomplete_run(tmp_path: Path) -> None:
    module = load_module("cafe_reproduce_bad_smoke", REPRODUCIBILITY / "reproduce.py")
    assert module.smoke_check(tmp_path) == 1


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


def test_reviewer_projection_removes_private_deployment_details() -> None:
    module = load_module(
        "cafe_package_reviewer",
        REPRODUCIBILITY / "package_reviewer_artifact.py",
    )
    payload = {
        "models": ["Chronos-2", "Timer-4.0"],
        "rows": [
            {"model_id": "Timer-4.0", "score": 0.1},
            {"model_id": "Chronos-2", "score": 0.2},
        ],
        "root": "/data/xmy/CaFE/runtime/experiments/example",
        "endpoint": "http://192.168.99.92:10810",
        "host": "timecho-vm",
    }
    projected = module.sanitize_json_value(payload)
    rendered = json.dumps(projected)
    assert "Timer-4.0" not in rendered
    assert "/data/xmy" not in rendered
    assert "192.168.99.92" not in rendered
    assert "timecho-vm" not in rendered
    assert projected["models"] == ["Chronos-2"]
    assert projected["rows"] == [{"model_id": "Chronos-2", "score": 0.2}]

    embedded, retain = module.sanitize_embedded_json(
        json.dumps(
            {
                "context_groups": [
                    {"model_ids": ["Chronos-2", "Timer-4.0"]},
                ]
            }
        )
    )
    assert retain
    assert json.loads(embedded) == {
        "context_groups": [{"model_ids": ["Chronos-2"]}]
    }


def test_reviewer_parquet_projection_filters_rows_and_embedded_json(
    tmp_path: Path,
) -> None:
    import pyarrow as pa
    import pyarrow.parquet as pq

    module = load_module(
        "cafe_package_reviewer_parquet",
        REPRODUCIBILITY / "package_reviewer_artifact.py",
    )
    source = tmp_path / "source.parquet"
    destination = tmp_path / "projected.parquet"
    pq.write_table(
        pa.table(
            {
                "model_id": ["Timer-4.0", "Chronos-2"],
                "payload_json": [
                    json.dumps({"model_ids": ["Timer-4.0"]}),
                    json.dumps(
                        {
                            "model_ids": ["Chronos-2", "Timer-4.0"],
                            "root": "/data/xmy/CaFE/runtime",
                        }
                    ),
                ],
            }
        ),
        source,
    )
    filter_columns, text_columns = module.parquet_projection_columns(source)
    module.project_parquet(
        source,
        destination,
        filter_columns,
        text_columns,
    )
    rows = pq.read_table(destination).to_pylist()
    assert len(rows) == 1
    assert rows[0]["model_id"] == "Chronos-2"
    assert "Timer-4.0" not in rows[0]["payload_json"]
    assert "/data/xmy" not in rows[0]["payload_json"]
