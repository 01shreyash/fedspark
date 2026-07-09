import numpy as np
import pandas as pd

from generator.amplify import amplify_paysim
from generator.partition import dirichlet_partition, partition_by_size_shares
from generator.inject import inject_label_flip, inject_drift


def test_amplify_paysim_creates_rows():
    df = pd.DataFrame({"amount": [100, 200], "step": [1, 2], "nameOrig": ["C1", "C2"], "nameDest": ["M1", "M2"],
                        "oldbalanceOrg": [500, 600], "newbalanceOrig": [400, 400],
                        "oldbalanceDest": [0, 0], "newbalanceDest": [0, 0], "isFraud": [0, 0], "isFlaggedFraud": [0, 0], "type": ["PAYMENT", "TRANSFER"]})
    result = amplify_paysim(df, factor=3)
    assert len(result) == len(df) * 3
    assert "nameOrig" in result.columns


def test_dirichlet_partition_preserves_rows():
    df = pd.DataFrame({"type": ["A", "A", "B", "B", "C"], "val": range(5)})
    silos = dirichlet_partition(df, n_silos=2, alpha=0.5, size_shares=[0.5, 0.5], seed=42)
    total = sum(len(v) for v in silos.values())
    assert total == len(df)


def test_partition_by_size_shares():
    df = pd.DataFrame({"x": range(100)})
    silos = partition_by_size_shares(df, size_shares=[0.5, 0.3, 0.2], seed=42)
    assert len(silos) == 3
    assert sum(len(v) for v in silos.values()) == 100


def test_inject_label_flip():
    df = pd.DataFrame({"isFraud": [0, 0, 1, 1, 0]})
    flipped = inject_label_flip(df, frac=1.0, seed=42)
    assert flipped["isFraud"].tolist() == [1, 1, 0, 0, 1]


def test_inject_drift():
    df = pd.DataFrame({"step": [1, 2, 3, 4, 5], "amount": [100, 200, 300, 400, 500]})
    drifted = inject_drift(df, step_threshold=3, amount_scale=2.0)
    assert drifted.loc[drifted["step"] < 3, "amount"].tolist() == [100, 200]
    assert drifted.loc[drifted["step"] >= 3, "amount"].tolist() == [600, 800, 1000]
