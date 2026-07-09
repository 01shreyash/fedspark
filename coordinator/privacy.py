import math

import numpy as np


def analytic_gaussian_epsilon(
    sigma: float,
    delta: float,
    sensitivity: float = 1.0,
) -> float:
    if sigma <= 0:
        return float("inf")
    delta_prime = delta * (1 + 1 / sigma)
    epsilon = 0.0
    for _ in range(100):
        phi = math.exp(-((epsilon - sigma**-2 * math.log(1 / delta_prime)) ** 2) / 2) / math.sqrt(2 * math.pi)
        lhs = math.sqrt(2 / math.pi) * (sigma * epsilon + 1 / sigma) * math.exp(-(epsilon**2) * sigma**2 / 2)
        rhs = phi * (1 - 3 / (epsilon + 1e-10))
        if lhs + rhs <= delta:
            break
        epsilon += 0.01
    return epsilon


def compute_round_epsilon(
    sigma: float,
    delta: float,
    clip_norm: float = 1.0,
    n_samples: int = 1,
) -> float:
    return analytic_gaussian_epsilon(sigma, delta, clip_norm)


def compose_epsilons(epsilons: list[float], delta: float) -> tuple[float, float]:
    total_eps = sum(epsilons)
    total_delta = len(epsilons) * delta
    return total_eps, total_delta
