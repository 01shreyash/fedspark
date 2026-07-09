import csv
import time
from pathlib import Path

import numpy as np
import pandas as pd
import yaml

from common.metrics import init_db
from common.model import MLP
from generator.amplify import amplify_paysim
from generator.partition import dirichlet_partition, partition_by_size_shares
from silo.features import build_features, compute_per_silo_stats, standardize
from silo.trainer import train_mode_b


def load_config(config_path: str) -> dict:
    with open(config_path) as f:
        return yaml.safe_load(f)


def run_experiment(
    config: dict,
    output_dir: str = "results",
    seed: int | None = None,
) -> dict:
    out_path = Path(output_dir)
    out_path.mkdir(parents=True, exist_ok=True)
    exp_seed = seed if seed is not None else config["run"]["seed"]
    rng = np.random.default_rng(exp_seed)
    n_rows = config["data"]["rows"]
    n_silos = config["data"]["silos"]
    alpha = config["data"]["dirichlet_alpha"]
    size_shares = config["data"]["size_shares"]
    n_rounds = config["protocol"]["rounds"]
    agg_kind = config["aggregator"]["kind"]
    hidden = config["model"]["hidden"]
    n_features = config["model"]["features"]
    train_cfg = config["train"]
    results = {"rounds": [], "silo_metrics": []}
    fake_df = pd.DataFrame({
        "step": rng.integers(0, 744, n_rows),
        "type": rng.choice(["PAYMENT", "TRANSFER", "CASH_OUT", "CASH_IN", "DEBIT"], n_rows),
        "amount": rng.exponential(1000, n_rows),
        "nameOrig": [f"C{i:010d}" for i in range(n_rows)],
        "oldbalanceOrg": rng.exponential(5000, n_rows),
        "newbalanceOrig": rng.exponential(4000, n_rows),
        "nameDest": [f"M{i:010d}" if rng.random() > 0.5 else f"C{i:010d}" for i in range(n_rows)],
        "oldbalanceDest": rng.exponential(5000, n_rows),
        "newbalanceDest": rng.exponential(4000, n_rows),
        "isFraud": rng.binomial(1, 0.0013, n_rows),
        "isFlaggedFraud": np.zeros(n_rows),
    })
    public_val = fake_df.iloc[: config["data"]["public_val_rows"]]
    train_df = fake_df.iloc[config["data"]["public_val_rows"] :]
    silos = dirichlet_partition(train_df, n_silos, alpha, size_shares, exp_seed)
    silo_features = {}
    silo_labels = {}
    silo_stats = {}
    for sid, sdf in silos.items():
        X, y = build_features(sdf)
        mean, std = compute_per_silo_stats(X)
        X = standardize(X, mean, std)
        silo_features[sid] = X
        silo_labels[sid] = y
        silo_stats[sid] = (mean, std)
    X_val, y_val = build_features(public_val)
    val_mean, val_std = compute_per_silo_stats(X_val)
    X_val = standardize(X_val, val_mean, val_std)
    global_model = MLP(n_features=n_features, hidden=hidden, seed=exp_seed)
    w_global = {k: v.copy() for k, v in global_model.weights.items()}
    csv_rows = []
    for rnd in range(1, n_rounds + 1):
        round_deltas = []
        round_sizes = []
        round_losses = []
        for sid in range(n_silos):
            X_s = silo_features[sid]
            y_s = silo_labels[sid]
            model = MLP(n_features=n_features, hidden=hidden, seed=exp_seed)
            model.weights = {k: v.copy() for k, v in w_global.items()}
            delta, n, loss = train_mode_b(X_s, y_s, w_global, model, train_cfg)
            round_deltas.append(delta)
            round_sizes.append(n)
            round_losses.append(loss)
        from coordinator.aggregate import fedavg, qwra, trimmed_mean
        if agg_kind == "fedavg":
            agg = fedavg(round_deltas, round_sizes, config["aggregator"]["server_lr"], config["aggregator"]["size_exp"])
        elif agg_kind == "trimmed_mean":
            agg = trimmed_mean(round_deltas, trim_frac=0.2, server_lr=config["aggregator"]["server_lr"])
        else:
            agg, rejected = qwra(
                round_deltas, round_sizes, w_global, (X_val, y_val), global_model,
                cos_tau=config["aggregator"]["cos_tau"],
                size_exp=config["aggregator"]["size_exp"],
                server_lr=config["aggregator"]["server_lr"],
            )
        for k in w_global:
            w_global[k] = w_global[k] + agg[k]
        global_model.weights = {k: v.copy() for k, v in w_global.items()}
        preds = global_model.forward(X_val)
        from sklearn.metrics import roc_auc_score, average_precision_score
        auc_roc = roc_auc_score(y_val, preds.ravel())
        auc_pr = average_precision_score(y_val, preds.ravel())
        csv_rows.append({
            "round": rnd,
            "auc_roc": auc_roc,
            "auc_pr": auc_pr,
            "n_updates": len(round_deltas),
            "avg_loss": np.mean(round_losses),
            "agg_kind": agg_kind,
            "seed": exp_seed,
            "alpha": alpha,
        })
    csv_path = out_path / f"exp_{agg_kind}_alpha{alpha}_seed{exp_seed}.csv"
    with open(csv_path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=csv_rows[0].keys())
        writer.writeheader()
        writer.writerows(csv_rows)
    return {"csv_path": str(csv_path), "final_auc_pr": csv_rows[-1]["auc_pr"] if csv_rows else 0}


if __name__ == "__main__":
    import sys
    config_path = sys.argv[1] if len(sys.argv) > 1 else "configs/base.yaml"
    config = load_config(config_path)
    result = run_experiment(config)
    print(f"Results written to {result['csv_path']}")
    print(f"Final AUC-PR: {result['final_auc_pr']:.4f}")
