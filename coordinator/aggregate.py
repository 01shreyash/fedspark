import numpy as np

from common.model import MLP


def fedavg(
    deltas: list[dict[str, np.ndarray]],
    sizes: list[int],
    server_lr: float = 1.0,
    size_exp: float = 0.5,
) -> dict[str, np.ndarray]:
    if not deltas:
        return {}
    weights = np.array([s ** size_exp for s in sizes])
    weights = weights / weights.sum()
    avg = {k: np.zeros_like(deltas[0][k]) for k in deltas[0]}
    for i, d in enumerate(deltas):
        for k in avg:
            avg[k] = avg[k] + weights[i] * d[k]
    return {k: v * server_lr for k, v in avg.items()}


def trimmed_mean(
    deltas: list[dict[str, np.ndarray]],
    trim_frac: float = 0.2,
    server_lr: float = 1.0,
) -> dict[str, np.ndarray]:
    if not deltas:
        return {}
    n = len(deltas)
    trim = max(1, int(n * trim_frac))
    result = {}
    keys = list(deltas[0].keys())
    for k in keys:
        stacked = np.stack([d[k] for d in deltas], axis=-1)
        sorted_arr = np.sort(stacked, axis=-1)
        result[k] = np.mean(sorted_arr[..., trim : n - trim], axis=-1)
    return {k: v * server_lr for k, v in result.items()}


def qwra(
    deltas: list[dict[str, np.ndarray]],
    sizes: list[int],
    w_global: dict[str, np.ndarray],
    valset: tuple[np.ndarray, np.ndarray] | None,
    model_template: MLP,
    cos_tau: float = 0.0,
    size_exp: float = 0.5,
    server_lr: float = 1.0,
    epsilon_q: float = 1e-4,
    fallback_ratio: float = 5.0,
) -> tuple[dict[str, np.ndarray], list[bool]]:
    if not deltas:
        return {}, []
    n_dims = len(deltas[0])
    keys = list(deltas[0].keys())
    stacked = np.stack([np.concatenate([d[k].ravel() for k in keys]) for d in deltas])
    median = np.median(stacked, axis=0)
    norms = np.linalg.norm(stacked, axis=1)
    median_norm = np.linalg.norm(median)
    if median_norm < 1e-10:
        cos_sims = np.ones(len(deltas))
    else:
        cos_sims = (stacked @ median) / (norms * median_norm + 1e-10)
    rejected = [bool(c < cos_tau) for c in cos_sims]
    surviving = [i for i in range(len(deltas)) if not rejected[i]]
    if not surviving:
        surviving = list(range(len(deltas)))
        rejected = [False] * len(deltas)
    q_scores = []
    if valset is not None:
        X_val, y_val = valset
        base_model = model_template
        base_model.weights = {k: v.copy() for k, v in w_global.items()}
        base_preds = base_model.forward(X_val)
        from sklearn.metrics import average_precision_score as aps
        base_aps = aps(y_val, base_preds.ravel())
        for i in surviving:
            candidate_model = model_template
            candidate_weights = {k: w_global[k] + deltas[i][k] for k in w_global}
            candidate_model.weights = candidate_weights
            cand_preds = candidate_model.forward(X_val)
            cand_aps = aps(y_val, cand_preds.ravel())
            q = max(cand_aps - base_aps, 0) + epsilon_q
            q_scores.append(q)
    else:
        q_scores = [1.0] * len(surviving)
    surv_sizes = [sizes[i] for i in surviving]
    surv_deltas = [deltas[i] for i in surviving]
    u_weights = np.array([(s ** size_exp) * q for s, q in zip(surv_sizes, q_scores)])
    u_weights = u_weights / (u_weights.sum() + 1e-10)
    agg = {k: np.zeros_like(deltas[0][k]) for k in keys}
    for i, idx in enumerate(surviving):
        for k in keys:
            agg[k] = agg[k] + u_weights[i] * deltas[idx][k]
    d_qwra = {k: v * server_lr for k, v in agg.items()}
    aggregated_d = np.concatenate([d_qwra[k].ravel() for k in keys])
    d_tm_stacked = np.stack([np.concatenate([d[k].ravel() for k in keys]) for d in deltas])
    n = len(deltas)
    trim = max(1, int(n * 0.2))
    d_tm_median = np.median(d_tm_stacked, axis=0)
    ratio = np.linalg.norm(aggregated_d - d_tm_median) / (np.linalg.norm(d_tm_median) + 1e-10)
    if ratio > fallback_ratio:
        agg = trimmed_mean(deltas, trim_frac=0.2, server_lr=server_lr)
    return agg, rejected
