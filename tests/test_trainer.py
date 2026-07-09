import numpy as np

from common.model import MLP
from silo.trainer import train_mode_b, l2_clip, add_gaussian_noise


def test_train_mode_b_returns_delta():
    model = MLP(n_features=5, hidden=[4], seed=42)
    X = np.random.default_rng(0).normal(size=(100, 5))
    y = np.random.default_rng(1).binomial(1, 0.3, 100)
    w_global = {k: v.copy() for k, v in model.weights.items()}
    config = {"local_epochs": 1, "batch_size": 32, "lr": 0.01, "momentum": 0.9, "prox_mu": 0.0, "pos_weight": 1.0}
    delta, n, loss = train_mode_b(X, y, w_global, model, config)
    assert n == 100
    assert loss >= 0
    for k in w_global:
        assert k in delta
        assert delta[k].shape == w_global[k].shape


def test_l2_clip():
    delta = {"W0": np.array([[10.0, 0.0], [0.0, 0.0]])}
    clipped = l2_clip(delta, clip_norm=1.0)
    norm = np.sqrt(np.sum(clipped["W0"] ** 2))
    assert abs(norm - 1.0) < 1e-6


def test_add_gaussian_noise():
    delta = {"W0": np.zeros((2, 2))}
    noisy = add_gaussian_noise(delta, sigma=0.5, clip_norm=1.0, seed=42)
    assert not np.allclose(noisy["W0"], 0)
