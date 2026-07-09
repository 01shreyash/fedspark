import numpy as np


def compute_psi(
    reference: np.ndarray,
    current: np.ndarray,
    n_bins: int = 20,
    epsilon: float = 1e-6,
) -> float:
    if len(reference) == 0 or len(current) == 0:
        return 0.0
    combined = np.concatenate([reference, current])
    bin_edges = np.percentile(reference, np.linspace(0, 100, n_bins + 1))
    unique_edges = np.unique(bin_edges)
    if len(unique_edges) < 2:
        bin_edges = np.linspace(
            combined.min() - epsilon,
            combined.max() + epsilon,
            n_bins + 1,
        )
    else:
        bin_edges[-1] = max(bin_edges[-1], combined.max() + epsilon)
        bin_edges[0] = min(bin_edges[0], combined.min() - epsilon)
    ref_counts, _ = np.histogram(reference, bins=bin_edges)
    cur_counts, _ = np.histogram(current, bins=bin_edges)
    ref_pct = ref_counts / max(len(reference), 1) + epsilon
    cur_pct = cur_counts / max(len(current), 1) + epsilon
    ref_pct = ref_pct / ref_pct.sum()
    cur_pct = cur_pct / cur_pct.sum()
    psi = np.sum((ref_pct - cur_pct) * np.log(ref_pct / cur_pct))
    return float(psi)


class DriftMonitor:
    def __init__(
        self,
        reference: np.ndarray | None = None,
        window_size: int = 50000,
        n_bins: int = 20,
        psi_threshold: float = 0.2,
    ):
        self.reference = reference
        self.window_size = window_size
        self.n_bins = n_bins
        self.psi_threshold = psi_threshold
        self.buffer: list[float] = []

    def set_reference(self, reference: np.ndarray):
        self.reference = reference

    def update(self, values: np.ndarray) -> float | None:
        self.buffer.extend(values.tolist())
        if len(self.buffer) > self.window_size:
            self.buffer = self.buffer[-self.window_size:]
        if self.reference is None or len(self.buffer) < self.window_size // 2:
            return None
        current = np.array(self.buffer[-self.window_size:])
        psi = compute_psi(self.reference, current, self.n_bins)
        if psi > self.psi_threshold:
            return psi
        return None
