import base64
import io

import numpy as np


def weights_to_b64(weights: dict[str, np.ndarray]) -> str:
    buf = io.BytesIO()
    np.savez_compressed(buf, **weights)
    return base64.b64encode(buf.getvalue()).decode("utf-8")


def weights_from_b64(data: str) -> dict[str, np.ndarray]:
    buf = io.BytesIO(base64.b64decode(data))
    return dict(np.load(buf))
