from app.services.metric_service import compute_sample_metrics


def test_sample_mse_and_mae_flatten_all_target_elements():
    metrics = compute_sample_metrics(target_future=[[1.0], [3.0], [5.0]], forecast=[[2.0], [1.0], [5.0]])

    assert metrics["mse"] == (1.0 + 4.0 + 0.0) / 3
    assert metrics["mae"] == 1.0
