import json
import time
from typing import Iterator

import pandas as pd


def rows_to_json_lines(df: pd.DataFrame) -> Iterator[bytes]:
    for _, row in df.iterrows():
        yield (json.dumps(row.to_dict(), default=str) + "\n").encode("utf-8")


def replay_to_kafka(
    df: pd.DataFrame,
    topic: str,
    producer,
    events_per_s: int = 2000,
    key_col: str | None = None,
):
    interval = 1.0 / events_per_s if events_per_s > 0 else 0
    for _, row in df.iterrows():
        msg = json.dumps(row.to_dict(), default=str)
        key = str(row[key_col]) if key_col and key_col in row else None
        producer.produce(topic=topic, key=key, value=msg)
        if interval > 0:
            time.sleep(interval)
    producer.flush()
