import io

import numpy as np


class MLP:
    def __init__(
        self,
        n_features: int,
        hidden: list[int] | None = None,
        seed: int = 42,
    ):
        self.n_features = n_features
        self.hidden = hidden or [64, 32]
        self.rng = np.random.default_rng(seed)
        self.weights: dict[str, np.ndarray] = {}
        self._init_weights()

    def _init_weights(self) -> None:
        layers = [self.n_features] + self.hidden + [1]
        for i in range(len(layers) - 1):
            self.weights[f"W{i}"] = self.rng.normal(0, 0.1, (layers[i], layers[i + 1]))
            self.weights[f"b{i}"] = np.zeros((1, layers[i + 1]))

    def forward(self, X: np.ndarray) -> np.ndarray:
        out = X
        n_layers = len(self.hidden) + 1
        for i in range(n_layers - 1):
            out = out @ self.weights[f"W{i}"] + self.weights[f"b{i}"]
            out = np.maximum(0, out)
        out = out @ self.weights[f"W{n_layers - 1}"] + self.weights[f"b{n_layers - 1}"]
        return 1 / (1 + np.exp(-out))

    def serialize(self) -> bytes:
        buf = io.BytesIO()
        np.savez_compressed(buf, **self.weights)
        return buf.getvalue()

    def deserialize(self, data: bytes) -> "MLP":
        buf = io.BytesIO(data)
        self.weights = dict(np.load(buf))
        return self
