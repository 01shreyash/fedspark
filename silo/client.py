import time
from pathlib import Path

import numpy as np
import requests

from common.model import MLP
from common.serialize import weights_from_b64, weights_to_b64
from silo.features import build_features, compute_per_silo_stats, standardize
from silo.trainer import add_gaussian_noise, l2_clip, train_mode_b


class SiloClient:
    def __init__(
        self,
        silo_id: str,
        coordinator_url: str = "http://coordinator:8000",
        config: dict | None = None,
        data_path: str | None = None,
    ):
        self.silo_id = silo_id
        self.coordinator_url = coordinator_url
        self.config = config or {}
        self.data_path = data_path
        self.token = ""
        self.n_rows = 0
        self.per_silo_mean: np.ndarray | None = None
        self.per_silo_std: np.ndarray | None = None
        n_features = self.config.get("model", {}).get("features", 30)
        if n_features == "auto":
            n_features = 30
        self.model = MLP(
            n_features=n_features,
            hidden=self.config.get("model", {}).get("hidden", [64, 32]),
            seed=self.config.get("run", {}).get("seed", 42),
        )

    def register(self):
        resp = requests.post(
            f"{self.coordinator_url}/register",
            json={"silo_id": self.silo_id, "n_rows": self.n_rows},
        )
        data = resp.json()
        self.token = data["token"]

    def fetch_model(self, version: int) -> dict[str, np.ndarray]:
        resp = requests.get(f"{self.coordinator_url}/model/{version}")
        data = resp.json()
        return weights_from_b64(data["weights_b64"])

    def poll_round(self) -> dict | None:
        resp = requests.get(f"{self.coordinator_url}/round/current")
        data = resp.json()
        if data.get("idle"):
            return None
        return data

    def train_and_submit(self, X: np.ndarray, y: np.ndarray, round_data: dict):
        w_global = self.fetch_model(round_data["version"])
        train_config = self.config.get("train", {})
        privacy_config = self.config.get("privacy", {})
        delta, n, loss = train_mode_b(X, y, w_global, self.model, train_config)
        if privacy_config.get("enabled", False):
            delta = l2_clip(delta, privacy_config.get("clip", 1.0))
            delta = add_gaussian_noise(
                delta,
                privacy_config.get("sigma", 0.8),
                privacy_config.get("clip", 1.0),
                self.config.get("run", {}).get("seed", 42),
            )
        delta_b64 = weights_to_b64(delta)
        resp = requests.post(
            f"{self.coordinator_url}/update",
            json={
                "silo_id": self.silo_id,
                "round": round_data["round"],
                "delta_b64": delta_b64,
                "n_samples": n,
                "train_loss": loss,
                "wall_time_s": 0.0,
                "token": self.token,
            },
        )
        return resp.json()

    def run_loop(self, X: np.ndarray, y: np.ndarray, max_rounds: int = 30):
        self.register()
        for _ in range(max_rounds):
            round_data = self.poll_round()
            if round_data is None:
                time.sleep(5)
                continue
            self.train_and_submit(X, y, round_data)
            time.sleep(1)
