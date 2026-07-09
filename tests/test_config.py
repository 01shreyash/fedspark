import tempfile
from pathlib import Path

import yaml


def test_config_round_trip():
    config_dir = Path(__file__).resolve().parent.parent / "configs"
    with open(config_dir / "base.yaml") as f:
        config = yaml.safe_load(f)

    assert config["run"]["seed"] == 42
    assert config["run"]["mode"] == "dev"
    assert config["data"]["silos"] == 3
    assert config["data"]["size_shares"] == [0.5, 0.3, 0.2]
    assert config["model"]["hidden"] == [64, 32]
    assert config["train"]["mode"] == "B"
    assert config["protocol"]["rounds"] == 30
    assert config["aggregator"]["kind"] == "qwra"
    assert config["privacy"]["enabled"] is False
    assert config["streaming"]["enabled"] is False
    assert config["attacks"]["label_flip"]["frac"] == 0.0

    with tempfile.NamedTemporaryFile(mode="w", suffix=".yaml", delete=False) as f:
        yaml.dump(config, f)
        tmp_path = f.name

    try:
        with open(tmp_path) as f:
            reloaded = yaml.safe_load(f)
        assert reloaded == config, "Config round-trip: values differ after save/load"
    finally:
        Path(tmp_path).unlink()
