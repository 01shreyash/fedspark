import numpy as np
import pandas as pd


FEATURE_COLS = [
    "log1p_amount",
    "balance_delta_orig",
    "balance_delta_dest",
    "zero_balance_orig",
    "zero_balance_dest",
    "type_CASH_IN",
    "type_CASH_OUT",
    "type_DEBIT",
    "type_PAYMENT",
    "type_TRANSFER",
    "velocity_1h_count",
    "velocity_24h_count",
    "velocity_168h_count",
    "velocity_1h_sum",
    "velocity_24h_sum",
    "velocity_168h_sum",
    "in_degree_24h",
    "log1p_amount_orig_balance_ratio",
    "log1p_amount_dest_balance_ratio",
    "is_flagged_fraud",
    "hour_sin",
    "hour_cos",
]


def build_features(df: pd.DataFrame) -> tuple[np.ndarray, np.ndarray]:
    df = df.copy()
    if "amount" in df.columns and "log1p_amount" not in df.columns:
        df["log1p_amount"] = np.log1p(df["amount"].clip(lower=0))
    if all(c in df.columns for c in ["oldbalanceOrg", "newbalanceOrig", "amount"]):
        df["balance_delta_orig"] = df["oldbalanceOrg"] - df["newbalanceOrig"] - df["amount"]
    if all(c in df.columns for c in ["oldbalanceDest", "newbalanceDest", "amount"]):
        df["balance_delta_dest"] = df["oldbalanceDest"] - df["newbalanceDest"] - df["amount"]
    if "oldbalanceOrg" in df.columns:
        df["zero_balance_orig"] = (df["oldbalanceOrg"] == 0).astype(float)
    if "oldbalanceDest" in df.columns:
        df["zero_balance_dest"] = (df["oldbalanceDest"] == 0).astype(float)
    for t in ["CASH_IN", "CASH_OUT", "DEBIT", "PAYMENT", "TRANSFER"]:
        col = f"type_{t}"
        if col not in df.columns:
            df[col] = (df.get("type", "") == t).astype(float)
    if "step" in df.columns:
        if "velocity_1h_count" not in df.columns:
            df["velocity_1h_count"] = 0.0
            df["velocity_24h_count"] = 0.0
            df["velocity_168h_count"] = 0.0
            df["velocity_1h_sum"] = 0.0
            df["velocity_24h_sum"] = 0.0
            df["velocity_168h_sum"] = 0.0
        if "in_degree_24h" not in df.columns:
            df["in_degree_24h"] = 0.0
    if "log1p_amount" in df.columns and "amount" in df.columns:
        df["log1p_amount_orig_balance_ratio"] = np.where(
            df["oldbalanceOrg"] > 0,
            df["log1p_amount"] / np.log1p(df["oldbalanceOrg"]),
            0,
        )
        df["log1p_amount_dest_balance_ratio"] = np.where(
            df["oldbalanceDest"] > 0,
            df["log1p_amount"] / np.log1p(df["oldbalanceDest"]),
            0,
        )
    if "isFlaggedFraud" in df.columns:
        df["is_flagged_fraud"] = df["isFlaggedFraud"].astype(float)
    else:
        df["is_flagged_fraud"] = 0.0
    if "step" in df.columns:
        df["hour_sin"] = np.sin(2 * np.pi * (df["step"] % 24) / 24)
        df["hour_cos"] = np.cos(2 * np.pi * (df["step"] % 24) / 24)
    else:
        df["hour_sin"] = 0.0
        df["hour_cos"] = 0.0
    missing = [c for c in FEATURE_COLS if c not in df.columns]
    for c in missing:
        df[c] = 0.0
    features = df[FEATURE_COLS].values.astype(np.float32)
    label = df["isFraud"].values.astype(np.float32) if "isFraud" in df.columns else np.zeros(len(df))
    return features, label


def compute_per_silo_stats(features: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    mean = features.mean(axis=0)
    std = features.std(axis=0)
    std[std < 1e-8] = 1.0
    return mean, std


def standardize(features: np.ndarray, mean: np.ndarray, std: np.ndarray) -> np.ndarray:
    return (features - mean) / std
