from coordinator.privacy import analytic_gaussian_epsilon, compute_round_epsilon


def test_analytic_gaussian_epsilon_positive():
    eps = analytic_gaussian_epsilon(sigma=0.8, delta=1e-5)
    assert eps > 0


def test_compute_round_epsilon():
    eps = compute_round_epsilon(sigma=0.8, delta=1e-5, clip_norm=1.0)
    assert eps > 0
