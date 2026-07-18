from __future__ import annotations

import importlib.util
import sys
from pathlib import Path


SCRIPT_PATH = (
    Path(__file__).parents[3] / "scripts" / "audit_synthetic_capability_shortcuts.py"
)


def load_module():
    spec = importlib.util.spec_from_file_location(
        "audit_synthetic_capability_shortcuts",
        SCRIPT_PATH,
    )
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


def test_audit_skips_and_counts_dataset_local_unsupported_cells(monkeypatch):
    module = load_module()
    profiles: dict[str, dict] = {}
    for capability_id, profile_id in module.AUDIT_PROFILE_BY_CAPABILITY.items():
        profile = profiles.setdefault(
            profile_id,
            {
                "profile_id": profile_id,
                "dataset_id": f"dataset_{profile_id}",
                "capabilities": {},
            },
        )
        profile["capabilities"][capability_id] = {
            "status": "unsupported",
            "unsupported_reason": "insufficient_target_spacing",
        }
    artifact = {
        "schema_version": "synthetic_v2_generator_conditioning_artifact.v4",
        "intensity_policy": {
            "policy_id": "dataset-local-relative-quantiles-v1",
            "percentile_levels": [0.1, 0.3, 0.5, 0.7, 0.9],
            "definition": "dataset local",
        },
        "profiles": profiles,
    }
    monkeypatch.setattr(
        module,
        "load_generator_conditioning_artifact",
        lambda: artifact,
    )

    summary = module.run_audit(
        seed_count=24,
        intensities=(1, 3, 5),
        seed=7,
    )

    assert summary["supported_cell_count"] == 0
    assert summary["unsupported_cell_count"] == len(
        module.AUDIT_PROFILE_BY_CAPABILITY
    )
    assert summary["rows"] == []
    assert summary["overall_passed"] is False
    assert "canonical_scale_id" not in summary
