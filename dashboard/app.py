import sqlite3
from pathlib import Path

import plotly.express as px
import plotly.graph_objects as go
import streamlit as st
import pandas as pd

st.set_page_config(page_title="FedSpark Dashboard", layout="wide")
st.title("FedSpark — Federated Learning Dashboard")

METRICS_DB = "/app/metrics/metrics.sqlite"


@st.cache_data(ttl=5)
def load_rounds():
    if not Path(METRICS_DB).exists():
        return pd.DataFrame()
    conn = sqlite3.connect(METRICS_DB)
    df = pd.read_sql("SELECT * FROM rounds ORDER BY round", conn)
    conn.close()
    return df


@st.cache_data(ttl=5)
def load_silo_metrics():
    if not Path(METRICS_DB).exists():
        return pd.DataFrame()
    conn = sqlite3.connect(METRICS_DB)
    df = pd.read_sql("SELECT * FROM silo_metrics ORDER BY round, silo_id", conn)
    conn.close()
    return df


@st.cache_data(ttl=5)
def load_drift_alerts():
    if not Path(METRICS_DB).exists():
        return pd.DataFrame()
    conn = sqlite3.connect(METRICS_DB)
    df = pd.read_sql("SELECT * FROM drift_alerts ORDER BY timestamp DESC", conn)
    conn.close()
    return df


col1, col2, col3 = st.columns(3)
rounds_df = load_rounds()
silo_df = load_silo_metrics()
drift_df = load_drift_alerts()

with col1:
    st.metric("Rounds Completed", len(rounds_df))
with col2:
    st.metric("Silos Active", silo_df["silo_id"].nunique() if not silo_df.empty else 0)
with col3:
    st.metric("Drift Alerts", len(drift_df))

st.subheader("Round Progress")
if not rounds_df.empty:
    col_a, col_b = st.columns(2)
    with col_a:
        if "auc_roc" in rounds_df.columns:
            fig = px.line(rounds_df, x="round", y="auc_roc", title="AUC-ROC per Round")
            st.plotly_chart(fig, use_container_width=True)
    with col_b:
        if "auc_pr" in rounds_df.columns:
            fig = px.line(rounds_df, x="round", y="auc_pr", title="AUC-PR per Round")
            st.plotly_chart(fig, use_container_width=True)

st.subheader("Per-Silo Metrics")
if not silo_df.empty:
    fig = px.line(
        silo_df,
        x="round",
        y="train_loss",
        color="silo_id",
        title="Training Loss per Silo",
    )
    st.plotly_chart(fig, use_container_width=True)
    flagged = silo_df[silo_df["flagged"] == 1]
    if not flagged.empty:
        st.warning(f"Flagged silos in {flagged['round'].nunique()} rounds")

st.subheader("Drift Alerts")
if not drift_df.empty:
    st.dataframe(drift_df)
else:
    st.info("No drift alerts recorded")

st.subheader("Privacy Budget")
if not silo_df.empty and "epsilon" in silo_df.columns:
    eps_by_silo = silo_df.groupby("silo_id")["epsilon"].sum().reset_index()
    fig = px.bar(eps_by_silo, x="silo_id", y="epsilon", title="Cumulative ε per Silo")
    st.plotly_chart(fig, use_container_width=True)

st.caption("FedSpark — Big Data Systems Project")
