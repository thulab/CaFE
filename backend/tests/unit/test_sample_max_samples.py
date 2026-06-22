"""max_samples 均匀采样（plan Task B3.3/B3.4）。"""
from app.services.dataset_load_service import build_windows, subsample_windows


def test_max_samples_uniformly_subsamples_with_endpoints():
    windows = build_windows(30, context_length=6, horizon=3, stride=1)
    assert len(windows) == 22

    picked = subsample_windows(windows, 5)

    assert len(picked) == 5
    assert picked[0] == windows[0]
    assert picked[-1] == windows[-1]
    # 可复现：同输入同结果。
    assert subsample_windows(windows, 5) == picked


def test_max_samples_none_or_oversize_returns_all():
    windows = build_windows(30, context_length=6, horizon=3, stride=1)

    assert subsample_windows(windows, None) == windows
    assert subsample_windows(windows, 100) == windows
    assert subsample_windows(windows, 0) == windows
