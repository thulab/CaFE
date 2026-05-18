import hashlib
import random


class StubTimerAdapter:
    def __init__(self, seed: int = 0):
        self.seed = seed

    def forecast(self, sample: dict, model: dict, timeout_seconds: int) -> list[list[float]]:
        del timeout_seconds
        sample_id = str(sample["sample_id"])
        model_id = str(model["model_id"])
        target_history = sample["target_history"]
        horizon = len(sample["target_future"])
        target_dim = len(target_history[-1])
        digest = hashlib.sha256(f"{model_id}:{sample_id}:{self.seed}".encode("utf-8")).hexdigest()
        rng = random.Random(int(digest[:16], 16))
        model_bias = ((int(hashlib.sha256(model_id.encode("utf-8")).hexdigest()[:4], 16) % 21) - 10) / 100.0
        forecast: list[list[float]] = []
        for _ in range(horizon):
            row = []
            for dim in range(target_dim):
                noise = rng.uniform(-0.05, 0.05)
                row.append(float(target_history[-1][dim]) + model_bias + noise)
            forecast.append(row)
        return forecast
