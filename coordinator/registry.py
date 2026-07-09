import json
from pathlib import Path

import numpy as np


class ModelRegistry:
    def __init__(self, registry_path: str | Path = "registry"):
        self.path = Path(registry_path)
        self.path.mkdir(parents=True, exist_ok=True)
        self._version_file = self.path / "version.json"
        self._current_version = self._load_version()
        self._models: dict[int, bytes] = {}

    def _load_version(self) -> int:
        if self._version_file.exists():
            with open(self._version_file) as f:
                return json.load(f).get("version", 0)
        return 0

    def _save_version(self):
        with open(self._version_file, "w") as f:
            json.dump({"version": self._current_version}, f)

    @property
    def latest_version(self) -> int:
        return self._current_version

    def save(self, weights_bytes: bytes, metadata: dict | None = None) -> int:
        self._current_version += 1
        v = self._current_version
        model_path = self.path / f"model_v{v}.npz"
        with open(model_path, "wb") as f:
            f.write(weights_bytes)
        if metadata:
            meta_path = self.path / f"model_v{v}.json"
            with open(meta_path, "w") as f:
                json.dump(metadata, f)
        self._models[v] = weights_bytes
        self._save_version()
        return v

    def load(self, version: int) -> bytes | None:
        if version in self._models:
            return self._models[version]
        model_path = self.path / f"model_v{version}.npz"
        if model_path.exists():
            data = model_path.read_bytes()
            self._models[version] = data
            return data
        return None

    def get_latest(self) -> tuple[int, bytes] | None:
        v = self._current_version
        if v == 0:
            return None
        data = self.load(v)
        if data is None:
            return None
        return v, data
