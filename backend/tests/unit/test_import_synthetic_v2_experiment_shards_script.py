from __future__ import annotations

import importlib.util
import sys
from pathlib import Path
from types import SimpleNamespace

import numpy as np
from sqlmodel import Session, select

from app.db.session import create_db_engine
from app.models.dataset import Shard
from app.models.sample import SampleIndex
from app.services.sample_store import SampleStore


SCRIPT_PATH = Path(__file__).parents[3] / "scripts" / "import_synthetic_v2_experiment_shards.py"


def load_import_module():
    repo_root = SCRIPT_PATH.parents[1]
    if str(repo_root / "backend") not in sys.path:
        sys.path.insert(0, str(repo_root / "backend"))
    spec = importlib.util.spec_from_file_location("import_synthetic_v2_experiment_shards", SCRIPT_PATH)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


def test_import_creates_platform_shards_and_is_idempotent(tmp_path):
    module = load_import_module()
    conditioning = SimpleNamespace(
        dataset_id="ett1_h",
        profile_id="ett1_h__univariate__L168_H24",
        intensity_policy_id="dataset-local-relative-quantiles-v1",
        target_percentile_levels=(0.10, 0.30, 0.50, 0.70, 0.90),
        target_feature="trend_strength",
        target_values=(0.01, 0.03, 0.05, 0.07, 0.09),
        season_length=24,
    )
    module.resolve_generator_conditioning = lambda **_kwargs: conditioning

    def fake_generate(
        _capability_id,
        length,
        _context_length,
        target_dim,
        _season_length,
        intensity,
        _sample_seed,
        **kwargs,
    ):
        assert kwargs["anchor_profile_id"] == conditioning.profile_id
        assert kwargs["generator_conditioning"] is conditioning
        target = np.full((length, target_dim), float(intensity))
        return target, {"predictability": {"construction_validated": True}}, None, {
            "trend_strength": float(intensity)
        }

    module._generate_accepted_sample_values = fake_generate
    database_url = f"sqlite:///{tmp_path / 'tsbenchmark.db'}"
    config = module.ImportConfig(
        name="import smoke",
        dataset_id=conditioning.dataset_id,
        profile_id=conditioning.profile_id,
        capabilities=("trend",),
        intensities=(1,),
        sample_count=2,
        context_length=168,
        horizon=24,
        season_length=24,
        target_dim=3,
        seed=20260701,
        frequency="h",
        database_url=database_url,
        runtime_dir=tmp_path / "runtime",
        source_summary=None,
        allow_duplicates=False,
    )

    first = module.import_experiment_shards(config)
    second = module.import_experiment_shards(config)

    assert first["created_count"] == 1
    assert first["skipped_count"] == 0
    assert second["created_count"] == 0
    assert second["skipped_count"] == 1

    engine = create_db_engine(database_url)
    with Session(engine) as session:
        shards = session.exec(select(Shard)).all()
        samples = session.exec(select(SampleIndex).order_by(SampleIndex.sample_index)).all()
        assert len(shards) == 1
        assert len(samples) == 2
        assert shards[0].generation_config["schema_version"] == (
            "synthetic_v2_dataset_local_platform_import.v2"
        )
        assert shards[0].generation_config["dataset_id"] == "ett1_h"
        assert shards[0].generation_config["profile_id"] == (
            "ett1_h__univariate__L168_H24"
        )
        assert shards[0].generation_config["intensity"] == 1
        assert shards[0].generation_config["difficulty"] == 1
        assert shards[0].generation_config["target_percentile_level"] == 0.10
        assert "dataset/profile-local" in shards[0].generation_config[
            "intensity_definition"
        ]
        assert samples[0].sample_metadata["experiment_sample_id"] == "trend-i1-000"
        assert samples[0].sample_metadata["dataset_id"] == "ett1_h"
        assert samples[0].sample_metadata["intensity"] == 1
        assert samples[0].sample_metadata["difficulty"] == 1

        preview = SampleStore().read_by_ref(session, samples[0].storage_ref)
        assert preview["target_column_names"] == ["target_0"]
        assert len(preview["target_history"]) == 168
        assert len(preview["target_future"]) == 24


def test_config_can_be_built_from_experiment_summary(tmp_path):
    module = load_import_module()
    summary = tmp_path / "summary.json"
    summary.write_text(
        """
{
  "dataset_id": "electricity_h",
  "profile_id": "electricity_h__multivariate__L24_H6",
  "requested_capabilities": ["common_factor"],
  "sample_count_per_capability_intensity": 3,
  "context_length": 24,
  "horizon": 6,
  "season_length": 12
}
""".strip(),
        encoding="utf-8",
    )
    args = SimpleNamespace(
        summary=summary,
        name=None,
        dataset_id=None,
        profile_id=None,
        capabilities=None,
        intensities=[2],
        difficulties=[2],
        sample_count=None,
        context_length=None,
        horizon=None,
        season_length=None,
        target_dim=4,
        seed=20260701,
        frequency="h",
        database_url=f"sqlite:///{tmp_path / 'tsbenchmark.db'}",
        runtime_dir=tmp_path / "runtime",
        allow_duplicates=False,
    )

    config = module.config_from_args(args)

    assert config.name == "Imported " + summary.parent.name
    assert config.dataset_id == "electricity_h"
    assert config.profile_id == "electricity_h__multivariate__L24_H6"
    assert config.capabilities == ("common_factor",)
    assert config.intensities == (2,)
    assert config.sample_count == 3
    assert config.context_length == 24
    assert config.horizon == 6
    assert config.season_length == 12


def test_unsupported_dataset_capability_is_excluded(tmp_path):
    module = load_import_module()
    module.resolve_generator_conditioning = lambda **_kwargs: None
    config = module.ImportConfig(
        name="unsupported",
        dataset_id="m4_hourly",
        profile_id="m4_hourly__hierarchy__L168_H24",
        capabilities=("hierarchical_coherence",),
        intensities=(1, 2, 3, 4, 5),
        sample_count=2,
        context_length=168,
        horizon=24,
        season_length=24,
        target_dim=3,
        seed=20260701,
        frequency="h",
        database_url=f"sqlite:///{tmp_path / 'tsbenchmark.db'}",
        runtime_dir=tmp_path / "runtime",
        source_summary=None,
        allow_duplicates=False,
    )

    result = module.import_experiment_shards(config)

    assert result["created_count"] == 0
    assert result["unsupported_count"] == 1
    assert result["unsupported"] == [
        {
            "dataset_id": "m4_hourly",
            "profile_id": "m4_hourly__hierarchy__L168_H24",
            "capability_id": "hierarchical_coherence",
            "status": "unsupported",
            "reason": "dataset_profile_has_no_supported_conditioning",
        }
    ]
