import tempfile
from pathlib import Path

from common.metrics import init_db


def test_init_db_creates_tables():
    with tempfile.TemporaryDirectory() as td:
        db_path = Path(td) / "test.db"
        conn = init_db(db_path)
        tables = conn.execute("SELECT name FROM sqlite_master WHERE type='table'").fetchall()
        table_names = [t[0] for t in tables]
        assert "rounds" in table_names
        assert "silo_metrics" in table_names
        assert "drift_alerts" in table_names
        conn.close()
