import argparse
import time

import numpy as np
import yaml
from pathlib import Path

from silo.client import SiloClient


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--silo-id", required=True)
    parser.add_argument("--coordinator", default="http://coordinator:8000")
    parser.add_argument("--config", default="configs/base.yaml")
    parser.add_argument("--rounds", type=int, default=30)
    args = parser.parse_args()
    config_path = Path(args.config)
    if config_path.exists():
        with open(config_path) as f:
            config = yaml.safe_load(f)
    else:
        config = {}
    n_features = config.get("model", {}).get("features", 30)
    if n_features == "auto":
        n_features = 30
    rng = np.random.default_rng(config.get("run", {}).get("seed", 42))
    n_rows = config.get("data", {}).get("rows", 10000)
    X = rng.normal(size=(n_rows, n_features)).astype(np.float32)
    y = rng.binomial(1, 0.1, n_rows).astype(np.float32)
    client = SiloClient(
        silo_id=args.silo_id,
        coordinator_url=args.coordinator,
        config=config,
    )
    client.n_rows = n_rows
    client.run_loop(X, y, max_rounds=args.rounds)


if __name__ == "__main__":
    main()
