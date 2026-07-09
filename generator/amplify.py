import numpy as np
import pandas as pd


def amplify_paysim(
    df: pd.DataFrame,
    factor: int = 1,
    seed: int = 42,
    noise_scale: float = 0.1,
) -> pd.DataFrame:
    rng = np.random.default_rng(seed)
    chunks = [df]
    for i in range(1, factor):
        chunk = df.copy()
        for col in ["amount", "oldbalanceOrg", "newbalanceOrig", "oldbalanceDest", "newbalanceDest"]:
            chunk[col] = (chunk[col] * rng.lognormal(0, noise_scale, len(chunk))).astype(chunk[col].dtype)
        step_offset = rng.integers(1, 100)
        chunk["step"] = chunk["step"] + step_offset
        chunk["nameOrig"] = chunk["nameOrig"] + f"_amp{i}"
        chunk["nameDest"] = chunk["nameDest"] + f"_amp{i}"
        chunks.append(chunk)
    return pd.concat(chunks, ignore_index=True)
