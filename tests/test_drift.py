import numpy as np

from silo.drift import compute_psi, DriftMonitor


def test_compute_psi_identical():
    data = np.random.default_rng(42).normal(size=1000)
    psi = compute_psi(data, data)
    assert abs(psi) < 0.01


def test_compute_psi_different():
    ref = np.zeros(1000)
    cur = np.ones(1000) * 10
    psi = compute_psi(ref, cur)
    assert psi > 0.1


def test_drift_monitor():
    monitor = DriftMonitor(window_size=100, psi_threshold=0.2)
    ref = np.random.default_rng(0).normal(0, 1, 200)
    monitor.set_reference(ref)
    for _ in range(10):
        alert = monitor.update(np.random.default_rng(1).normal(0, 1, 50))
    assert alert is None or alert < 10
