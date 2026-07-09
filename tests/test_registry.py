import tempfile
from pathlib import Path

from coordinator.registry import ModelRegistry


def test_registry_save_and_load():
    with tempfile.TemporaryDirectory() as td:
        reg = ModelRegistry(registry_path=td)
        v = reg.save(b"hello world", {"round": 1})
        assert v == 1
        data = reg.load(1)
        assert data == b"hello world"


def test_registry_latest_version():
    with tempfile.TemporaryDirectory() as td:
        reg = ModelRegistry(registry_path=td)
        assert reg.latest_version == 0
        reg.save(b"v1")
        assert reg.latest_version == 1
        reg.save(b"v2")
        assert reg.latest_version == 2
        latest = reg.get_latest()
        assert latest is not None
        assert latest[0] == 2
        assert latest[1] == b"v2"
