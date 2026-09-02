from __future__ import annotations

import sys
from pathlib import Path

import torch

SCRIPTS = Path(__file__).parents[1] / "scripts"
sys.path.insert(0, str(SCRIPTS))

from cafe_effect_finetune import effect_loss  # noqa: E402


def test_effect_loss_matches_nrmse_boundaries() -> None:
    truth = torch.tensor(
        [
            [1.0, 2.0],
            [3.0, 4.0],
            [2.0, 4.0],
            [7.0, 10.0],
        ]
    )
    ranges = torch.tensor([[0, 2, 2, 4]])
    mask = torch.tensor(
        [
            [True, True],
            [False, False],
            [True, True],
            [False, False],
        ]
    )
    scale = torch.tensor([2.0, 1.0, 2.0, 1.0])

    perfect = effect_loss(truth, truth, ranges, mask, scale)
    no_response = truth.clone()
    no_response[2:4] = no_response[0:2]
    absent = effect_loss(no_response, truth, ranges, mask, scale)

    assert perfect == 0.0
    assert torch.isclose(absent, torch.tensor(1.0))


def test_effect_loss_ignores_unaffected_targets() -> None:
    truth = torch.tensor([[1.0], [5.0], [3.0], [9.0]])
    ranges = torch.tensor([[0, 2, 2, 4]])
    mask = torch.tensor([[True], [False], [True], [False]])
    scale = torch.ones(4)
    predictions = truth.clone()
    predictions[3] = -1000.0

    assert effect_loss(predictions, truth, ranges, mask, scale) == 0.0
