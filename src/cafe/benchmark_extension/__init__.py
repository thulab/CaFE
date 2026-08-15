"""Native benchmark-extension pipeline.

The package contains the sole active CaFE data path: load official benchmark
instances, perturb their authentic paths, validate the resulting paired tasks,
run forecasts, and analyse capability effects.
"""

from cafe.benchmark_extension.gift_eval import (
    GiftEvalInstance,
    iter_gift_eval_instances,
)

__all__ = ["GiftEvalInstance", "iter_gift_eval_instances"]
