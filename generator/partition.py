import numpy as np
import pandas as pd


def dirichlet_partition(
    df: pd.DataFrame,
    n_silos: int,
    alpha: float,
    size_shares: list[float] | None = None,
    seed: int = 42,
    label_col: str = "type",
) -> dict[int, pd.DataFrame]:
    rng = np.random.default_rng(seed)
    types = df[label_col].unique()
    proportions = rng.dirichlet([alpha] * n_silos, size=len(types))
    silo_dfs: dict[int, list[pd.DataFrame]] = {i: [] for i in range(n_silos)}
    for t_idx, t in enumerate(types):
        mask = df[label_col] == t
        type_df = df[mask].reset_index(drop=True)
        indices = np.arange(len(type_df))
        rng.shuffle(indices)
        if size_shares is not None:
            fracs = np.array(size_shares) / np.sum(size_shares)
            cumsum = np.concatenate(([0], np.cumsum(fracs)))
            split_idxs = (cumsum[1:] * len(type_df)).astype(int)
        else:
            prop = proportions[t_idx]
            cumsum = np.concatenate(([0], np.cumsum(prop)))
            cumsum[-1] = 1.0
            split_idxs = (cumsum[1:] * len(type_df)).astype(int)
        start = 0
        for i, end in enumerate(split_idxs):
            silo_indices = indices[start:end]
            silo_dfs[i].append(type_df.iloc[silo_indices])
            start = end
    return {i: pd.concat(silo_dfs[i], ignore_index=True) for i in range(n_silos)}


def partition_by_size_shares(
    df: pd.DataFrame,
    size_shares: list[float],
    seed: int = 42,
) -> dict[int, pd.DataFrame]:
    rng = np.random.default_rng(seed)
    indices = np.arange(len(df))
    rng.shuffle(indices)
    fracs = np.array(size_shares) / np.sum(size_shares)
    cumsum = np.concatenate(([0], np.cumsum(fracs)))
    split_idxs = (cumsum[1:] * len(df)).astype(int)
    silos: dict[int, pd.DataFrame] = {}
    start = 0
    for i, end in enumerate(split_idxs):
        silo_indices = indices[start:end]
        silos[i] = df.iloc[silo_indices].reset_index(drop=True)
        start = end
    return silos
