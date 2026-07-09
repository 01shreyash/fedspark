import numpy as np
import pandas as pd


def inject_label_flip(
    df: pd.DataFrame,
    frac: float,
    seed: int = 42,
    label_col: str = "isFraud",
) -> pd.DataFrame:
    rng = np.random.default_rng(seed)
    mask = rng.random(len(df)) < frac
    df = df.copy()
    df.loc[mask, label_col] = 1 - df.loc[mask, label_col]
    return df


def inject_model_poison(
    delta: np.ndarray,
    gamma: float = 5.0,
) -> np.ndarray:
    return delta * (-gamma)


def inject_drift(
    df: pd.DataFrame,
    step_threshold: int,
    amount_scale: float = 3.0,
    amount_col: str = "amount",
) -> pd.DataFrame:
    df = df.copy()
    mask = df["step"] >= step_threshold
    df.loc[mask, amount_col] = df.loc[mask, amount_col] * amount_scale
    return df
