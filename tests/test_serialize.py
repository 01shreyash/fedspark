import numpy as np

from common.serialize import weights_to_b64, weights_from_b64


def test_weights_roundtrip():
    weights = {"W0": np.array([[1.0, 2.0], [3.0, 4.0]]), "b0": np.array([[0.1, 0.2]])}
    b64 = weights_to_b64(weights)
    assert isinstance(b64, str)
    reloaded = weights_from_b64(b64)
    for k in weights:
        np.testing.assert_array_almost_equal(weights[k], reloaded[k])
