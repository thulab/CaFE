from app.services.metric_service import ranking_metric_for_unit


def test_partial_unit_does_not_produce_ranking_metric():
    assert ranking_metric_for_unit(unit_status="partial_succeeded", unit_metrics={"mse": 0.5}, metric_name="mse") is None
    assert ranking_metric_for_unit(unit_status="succeeded", unit_metrics={"mse": 0.5}, metric_name="mse") == 0.5
