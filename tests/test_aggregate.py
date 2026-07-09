import numpy as np

from coordinator.aggregate import fedavg, trimmed_mean
from common.model import MLP


def test_fedavg_basic():
    d1 = {"W0": np.array([1.0, 2.0])}
    d2 = {"W0": np.array([3.0, 4.0])}
    result = fedavg([d1, d2], sizes=[10, 10], server_lr=1.0, size_exp=0.5)
    expected = (d1["W0"] + d2["W0"]) / 2
    np.testing.assert_array_almost_equal(result["W0"], expected)


def test_trimmed_mean():
    d1 = {"W0": np.array([1.0])}
    d2 = {"W0": np.array([2.0])}
    d3 = {"W0": np.array([100.0])}
    result = trimmed_mean([d1, d2, d3], trim_frac=0.3, server_lr=1.0)
    assert abs(result["W0"][0] - 2.0) < 1e-6


def test_trimmed_mean_five_values():
    d1 = {"W0": np.array([1.0])}
    d2 = {"W0": np.array([2.0])}
    d3 = {"W0": np.array([3.0])}
    d4 = {"W0": np.array([4.0])}
    d5 = {"W0": np.array([100.0])}
    result = trimmed_mean([d1, d2, d3, d4, d5], trim_frac=0.2, server_lr=1.0)
    assert abs(result["W0"][0] - 3.0) < 1e-6
