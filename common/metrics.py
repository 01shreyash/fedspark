import sqlite3
from pathlib import Path


def init_db(db_path: str | Path) -> sqlite3.Connection:
    conn = sqlite3.connect(str(db_path))
    conn.execute("PRAGMA journal_mode=WAL")
    conn.execute(
        """
        CREATE TABLE IF NOT EXISTS rounds (
            round INTEGER PRIMARY KEY,
            version INTEGER,
            global_auc_roc REAL,
            global_auc_pr REAL,
            n_updates INTEGER,
            wall_time_s REAL,
            timestamp TEXT DEFAULT CURRENT_TIMESTAMP
        )
    """
    )
    conn.execute(
        """
        CREATE TABLE IF NOT EXISTS silo_metrics (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            round INTEGER,
            silo_id TEXT,
            n_samples INTEGER,
            train_loss REAL,
            auc_roc REAL,
            auc_pr REAL,
            staleness INTEGER DEFAULT 0,
            flagged INTEGER DEFAULT 0,
            epsilon REAL DEFAULT 0.0,
            psi REAL DEFAULT 0.0
        )
    """
    )
    conn.execute(
        """
        CREATE TABLE IF NOT EXISTS drift_alerts (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            round INTEGER,
            silo_id TEXT,
            psi REAL,
            feature TEXT,
            timestamp TEXT DEFAULT CURRENT_TIMESTAMP
        )
    """
    )
    conn.commit()
    return conn
