import numpy as np
import pandas as pd

from silo.features import build_features, compute_per_silo_stats, standardize, FEATURE_COLS


def test_build_features_output_shape():
    df = pd.DataFrame({
        "step": [1, 2],
        "type": ["PAYMENT", "TRANSFER"],
        "amount": [100.0, 200.0],
        "oldbalanceOrg": [500.0, 600.0],
        "newbalanceOrig": [400.0, 500.0],
        "oldbalanceDest": [0.0, 100.0],
        "newbalanceDest": [0.0, 50.0],
        "isFraud": [0, 1],
        "isFlaggedFraud": [0, 0],
    })
    X, y = build_features(df)
    assert X.shape[0] == 2
    assert X.shape[1] == len(FEATURE_COLS)
    assert y.shape == (2,)
    assert not np.any(np.isnan(X))


def test_compute_per_silo_stats():
    X = np.array([[1.0, 2.0], [3.0, 4.0], [5.0, 6.0]])
    mean, std = compute_per_silo_stats(X)
    assert mean.shape == (2,)
    assert std.shape == (2,)
    assert np.all(std > 0)


def test_standardize():
    X = np.array([[0.0, 10.0], [10.0, 20.0]])
    mean = np.array([5.0, 15.0])
    std = np.array([5.0, 5.0])
    Xs = standardize(X, mean, std)
    np.testing.assert_array_almost_equal(Xs, np.array([[-1.0, -1.0], [1.0, 1.0]]))
