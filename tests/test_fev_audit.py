from __future__ import annotations

from cafe.data.fev_audit import build_fev_metadata_audit
from cafe.data.fev_audit import config_inventory_csv
from cafe.data.fev_audit import parse_fev_readme
from cafe.data.fev_audit import parse_fev_tasks
from cafe.data.fev_audit import task_matrix_csv


README = """---
license: other
dataset_info:
- config_name: multi
  features:
  - {name: id, dtype: string}
  - {name: timestamp, sequence: "timestamp[ms]"}
  - {name: target_a, sequence: float32}
  - {name: target_b, sequence: float32}
  - {name: holiday, sequence: string}
  splits:
  - {name: train, num_examples: 2}
  download_size: 10
  dataset_size: 20
- config_name: short
  features:
  - {name: id, dtype: string}
  - {name: timestamp, sequence: "timestamp[ms]"}
  - {name: target, sequence: float32}
  splits:
  - {name: train, num_examples: 1}
  download_size: 5
  dataset_size: 8
configs:
- config_name: multi
  data_files:
  - {split: train, path: multi/train-*}
- config_name: short
  data_files:
  - {split: train, path: short/train-*}
---

## Dataset statistics

| config | freq | # items | median length | # obs | # dynamic cols | # static cols | source | citation |
|---|---|---|---|---|---|---|---|---|
| `multi` | h | 2 | 1,000 | 4,000 | 3 | 0 | https://example.com/multi | paper |
| `short` | YE-DEC | 1 | 100 | 100 | 1 | 0 | https://example.com/short | paper |
"""


TASKS = """tasks:
- dataset_path: autogluon/fev_datasets
  dataset_config: multi
  horizon: 24
  num_windows: 2
  target: [target_a, target_b]
  known_dynamic_columns: [holiday]
- dataset_path: autogluon/fev_datasets
  dataset_config: short
  horizon: 5
"""


TREE = [
    {
        "type": "file",
        "path": "multi/train-00000-of-00001.parquet",
        "size": 10,
        "lfs": {"oid": "a" * 64},
    },
    {
        "type": "file",
        "path": "short/train-00000-of-00001.parquet",
        "size": 5,
        "lfs": {"oid": "b" * 64},
    },
]


def test_fev_metadata_audit_classifies_capabilities_and_length_scan():
    metadata, statistics = parse_fev_readme(README)
    tasks = parse_fev_tasks(TASKS)

    audit = build_fev_metadata_audit(
        tasks=tasks,
        readme_metadata=metadata,
        statistics=statistics,
        tree_entries=TREE,
    )

    assert audit["summary"]["task_count"] == 2
    assert audit["summary"]["config_count"] == 2
    assert audit["summary"]["download_bytes"] == 15
    multi, short = audit["task_rows"]
    assert multi["target_count"] == 2
    assert multi["categorical_known_columns"] == ["holiday"]
    assert multi["capability_status"]["common_factor"] == "candidate"
    assert multi["capability_status"]["covariate_response"] == (
        "candidate_requires_category_scan"
    )
    assert short["frequency_class"] == "calendar_offset"
    assert short["capability_status"]["trend"] == "requires_length_scan"
    assert short["capability_status"]["common_factor"] == (
        "not_applicable_single_target"
    )
    assert short["capability_status"]["covariate_response"] == (
        "not_applicable_no_known_covariates"
    )


def test_fev_audit_csv_outputs_include_all_capability_columns():
    metadata, statistics = parse_fev_readme(README)
    audit = build_fev_metadata_audit(
        tasks=parse_fev_tasks(TASKS),
        readme_metadata=metadata,
        statistics=statistics,
        tree_entries=TREE,
    )

    task_csv = task_matrix_csv(audit["task_rows"])
    config_csv = config_inventory_csv(audit["config_rows"])

    assert "covariate_response" in task_csv.splitlines()[0]
    assert "candidate_requires_category_scan" in task_csv
    assert "calendar_offset" in config_csv
