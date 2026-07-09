import numpy as np

from common.model import MLP


def test_mlp_forward_shape():
    model = MLP(n_features=30, seed=42)
    X = np.random.default_rng(0).normal(size=(10, 30))
    y = model.forward(X)
    assert y.shape == (10, 1), f"Expected (10, 1), got {y.shape}"
    assert np.all((y > 0) & (y < 1)), "Sigmoid output not in (0, 1)"


def test_mlp_serialize_roundtrip():
    model = MLP(n_features=30, seed=42)
    data = model.serialize()
    assert isinstance(data, bytes)
    assert len(data) > 0

    model2 = MLP(n_features=30, seed=99)
    model2.deserialize(data)
    for k in model.weights:
        np.testing.assert_array_equal(model.weights[k], model2.weights[k])


def test_mlp_forward_deterministic():
    model = MLP(n_features=5, seed=42)
    X = np.array([[1.0, 2.0, 3.0, 4.0, 5.0]])
    out1 = model.forward(X)
    out2 = model.forward(X)
    np.testing.assert_array_equal(out1, out2)
