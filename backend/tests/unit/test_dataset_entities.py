from datetime import datetime
from uuid import UUID

from app.models.benchmark import BenchmarkingRun, CapabilityBlock, Task, Track, Unit, RunEvent
from app.models.dataset import DatasetLoadJob, DatasetManifest, Shard
from app.models.metric import MetricDefinition, MetricResult
from app.models.model_registry import Model
from app.models.ranking import RankingEntry, RankingList
from app.models.report import Report
from app.models.sample import SampleIndex


def assert_uuid4(value: str) -> None:
    assert UUID(value).version == 4


def test_core_entities_have_uuid_defaults_statuses_and_json_datetimes():
    entities = [
        DatasetManifest(name="demo", domain="energy", source_uri="runtime/uploads/demo.csv", time_column="time"),
        DatasetLoadJob(dataset_manifest_id="manifest-1", split_config={"context_length": 6, "horizon": 3}),
        Shard(dataset_manifest_id="manifest-1", load_job_id="load-1", source_uri="file.csv"),
        SampleIndex(shard_id="shard-1", sample_index=0, context_start=0, context_end=5, horizon_start=6, horizon_end=8),
        Track(name="real track", primary_metric_id="mse"),
        CapabilityBlock(name="real data", track_id="track-1"),
        Model(name="Timer 3.5", model_family="Timer", model_version="3.5"),
        BenchmarkingRun(track_id="track-1", model_ids=["model-1"]),
        Unit(benchmarking_run_id="run-1", model_id="model-1"),
        Task(benchmarking_run_id="run-1", unit_id="unit-1", model_id="model-1", capability_block_id="block-1"),
        MetricDefinition(name="mse", display_name="MSE"),
        MetricResult(metric_id="mse", result_level="sample", benchmarking_run_id="run-1", model_id="model-1", value=1.0),
        Report(benchmarking_run_id="run-1", track_id="track-1"),
        RankingList(track_id="track-1", default_metric_id="mse"),
        RankingEntry(ranking_list_id="ranking-1", track_id="track-1", metric_id="mse", model_id="model-1", benchmarking_run_id="run-1", unit_id="unit-1", metric_value=0.1, rank=1),
        RunEvent(benchmarking_run_id="run-1", message="queued"),
    ]

    id_fields = [
        "dataset_manifest_id",
        "load_job_id",
        "shard_id",
        "sample_id",
        "track_id",
        "capability_block_id",
        "model_id",
        "benchmarking_run_id",
        "unit_id",
        "task_id",
        "metric_id",
        "metric_result_id",
        "report_id",
        "ranking_list_id",
        "ranking_entry_id",
        "run_event_id",
    ]

    for entity, id_field in zip(entities, id_fields, strict=True):
        assert_uuid4(getattr(entity, id_field))
        assert isinstance(entity.created_at, datetime)
        dumped = entity.model_dump(mode="json")
        if "created_at" in dumped:
            assert "T" in dumped["created_at"]

    assert entities[0].status == "ready_to_load"
    assert entities[1].status == "created"
    assert entities[2].status == "created"
    assert entities[4].status == "ready"
    assert entities[7].status == "created"
    assert entities[14].policy == "latest_valid_result"
