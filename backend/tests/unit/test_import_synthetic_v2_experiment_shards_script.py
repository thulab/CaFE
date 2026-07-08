from __future__ import annotations

import importlib.util
import sys
from pathlib import Path
from types import SimpleNamespace

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
    database_url = f"sqlite:///{tmp_path / 'tsbenchmark.db'}"
    config = module.ImportConfig(
        name="import smoke",
        capabilities=("trend",),
        intensities=(1,),
        sample_count=2,
        context_length=16,
        horizon=4,
        season_length=4,
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
        assert shards[0].generation_config["schema_version"] == "synthetic_v2_platform_import.v1"
        assert shards[0].generation_config["intensity"] == 1
        assert shards[0].generation_config["difficulty"] == 1
        assert samples[0].sample_metadata["experiment_sample_id"] == "trend-i1-000"
        assert samples[0].sample_metadata["intensity"] == 1
        assert samples[0].sample_metadata["difficulty"] == 1

        preview = SampleStore().read_by_ref(session, samples[0].storage_ref)
        assert preview["target_column_names"] == ["target_0"]
        assert len(preview["target_history"]) == 16
        assert len(preview["target_future"]) == 4


def test_config_can_be_built_from_experiment_summary(tmp_path):
    module = load_import_module()
    summary = tmp_path / "summary.json"
    summary.write_text(
        """
{
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
    assert config.capabilities == ("common_factor",)
    assert config.intensities == (2,)
    assert config.sample_count == 3
    assert config.context_length == 24
    assert config.horizon == 6
    assert config.season_length == 12
