import os
import tempfile

os.environ["FEDSPARK_REGISTRY_PATH"] = tempfile.mkdtemp()
os.environ["FEDSPARK_METRICS_PATH"] = os.path.join(tempfile.mkdtemp(), "metrics.sqlite")

from fastapi.testclient import TestClient
from coordinator.app import app

client = TestClient(app)


def test_healthz():
    resp = client.get("/healthz")
    assert resp.status_code == 200
    assert resp.json() == {"status": "ok"}


def test_register():
    resp = client.post("/register", json={"silo_id": "test-silo", "n_rows": 1000})
    assert resp.status_code == 200
    data = resp.json()
    assert "token" in data
    assert data["token"].startswith("tok_test-silo_")


def test_register_duplicate():
    resp = client.post("/register", json={"silo_id": "test-silo", "n_rows": 1000})
    assert resp.status_code == 400


def test_round_current_idle():
    resp = client.get("/round/current")
    assert resp.status_code == 200
    data = resp.json()
    assert data.get("idle") is True
