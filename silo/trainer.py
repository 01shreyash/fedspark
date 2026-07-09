import numpy as np

from common.model import MLP


def _weighted_bce_loss(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    pos_weight: float,
) -> float:
    eps = 1e-7
    y_pred = np.clip(y_pred, eps, 1 - eps)
    loss = -pos_weight * y_true * np.log(y_pred) - (1 - y_true) * np.log(1 - y_pred)
    return float(loss.mean())


def _compute_gradient(
    X: np.ndarray,
    y: np.ndarray,
    weights: dict[str, np.ndarray],
    hidden: list[int],
    pos_weight: float,
    prox_mu: float,
    w_global: dict[str, np.ndarray],
) -> dict[str, np.ndarray]:
    n_layers = len(hidden) + 1
    activations = [X]
    pre_activations = []
    out = X
    for i in range(n_layers - 1):
        z = out @ weights[f"W{i}"] + weights[f"b{i}"]
        pre_activations.append(z)
        out = np.maximum(0, z)
        activations.append(out)
    z = out @ weights[f"W{n_layers - 1}"] + weights[f"b{n_layers - 1}"]
    y_pred = 1 / (1 + np.exp(-z))
    grad = y_pred - y.reshape(-1, 1)
    grad = grad * pos_weight if pos_weight != 1.0 else grad
    dW = {}
    db = {}
    dout = grad
    for i in range(n_layers - 1, -1, -1):
        dW[f"W{i}"] = activations[i].T @ dout / len(X)
        db[f"b{i}"] = dout.mean(axis=0, keepdims=True)
        if i > 0:
            dout = dout @ weights[f"W{i}"].T
            dout[pre_activations[i - 1] <= 0] = 0
    for k in dW:
        prox_term = prox_mu * (weights[k] - w_global[k])
        dW[k] = dW[k] + prox_term / len(X)
    return dW, db, y_pred


def _apply_gradient(
    weights: dict[str, np.ndarray],
    dW: dict[str, np.ndarray],
    db: dict[str, np.ndarray],
    lr: float,
    momentum: float,
    velocity: dict[str, np.ndarray],
    n_layers: int,
) -> tuple[dict[str, np.ndarray], dict[str, np.ndarray]]:
    new_weights = {}
    new_velocity = {}
    for i in range(n_layers):
        w_key = f"W{i}"
        b_key = f"b{i}"
        vW = momentum * velocity.get(w_key, 0) - lr * dW[w_key]
        vb = momentum * velocity.get(b_key, 0) - lr * db[b_key]
        new_velocity[w_key] = vW
        new_velocity[b_key] = vb
        new_weights[w_key] = weights[w_key] + vW
        new_weights[b_key] = weights[b_key] + vb
    return new_weights, new_velocity


def train_mode_b(
    X: np.ndarray,
    y: np.ndarray,
    w_global: dict[str, np.ndarray],
    model: MLP,
    config: dict,
) -> tuple[dict[str, np.ndarray], int, float]:
    local_epochs = config.get("local_epochs", 2)
    batch_size = config.get("batch_size", 4096)
    lr = config.get("lr", 0.05)
    momentum = config.get("momentum", 0.9)
    prox_mu = config.get("prox_mu", 0.01)
    pos_weight = config.get("pos_weight", 1.0)
    if pos_weight == "auto":
        n_pos = int(y.sum())
        n_neg = n - n_pos
        pos_weight = n_neg / max(n_pos, 1)
    n = len(X)
    weights = {k: v.copy() for k, v in w_global.items()}
    velocity: dict[str, np.ndarray] = {}
    n_layers = len(model.hidden) + 1
    total_loss = 0.0
    n_batches = 0
    for _ in range(local_epochs):
        indices = np.arange(n)
        np.random.shuffle(indices)
        for start in range(0, n, batch_size):
            batch_idx = indices[start : start + batch_size]
            Xb = X[batch_idx]
            yb = y[batch_idx]
            dW, db, _ = _compute_gradient(Xb, yb, weights, model.hidden, pos_weight, prox_mu, w_global)
            weights, velocity = _apply_gradient(weights, dW, db, lr, momentum, velocity, n_layers)
            total_loss += _weighted_bce_loss(yb, model.forward(Xb), pos_weight)
            n_batches += 1
    delta = {k: weights[k] - w_global[k] for k in weights}
    avg_loss = total_loss / max(n_batches, 1)
    return delta, n, avg_loss


def l2_clip(delta: dict[str, np.ndarray], clip_norm: float) -> dict[str, np.ndarray]:
    total_norm = np.sqrt(sum(np.sum(v**2) for v in delta.values()))
    scale = min(1.0, clip_norm / max(total_norm, 1e-10))
    return {k: v * scale for k, v in delta.items()}


def add_gaussian_noise(delta: dict[str, np.ndarray], sigma: float, clip_norm: float, seed: int) -> dict[str, np.ndarray]:
    rng = np.random.default_rng(seed)
    noisy = {}
    for k, v in delta.items():
        noise = rng.normal(0, sigma * clip_norm, size=v.shape)
        noisy[k] = v + noise
    return noisy


def train_partition(
    partition_iter,
    w_global_bc,
    model_template: MLP,
    config: dict,
    per_silo_mean: np.ndarray | None = None,
    per_silo_std: np.ndarray | None = None,
) -> tuple[dict[str, np.ndarray], int, float]:
    from silo.features import build_features, standardize
    rows = list(partition_iter)
    if not rows:
        return {}, 0, 0.0
    import pandas as pd
    df = pd.DataFrame(rows)
    X, y = build_features(df)
    if per_silo_mean is not None and per_silo_std is not None:
        X = standardize(X, per_silo_mean, per_silo_sstd)
    w_global = dict(w_global_bc.value)
    model = model_template
    model.weights = {k: v.copy() for k, v in w_global.items()}
    delta, n, loss = train_mode_b(X, y, w_global, model, config)
    return delta, n, loss
