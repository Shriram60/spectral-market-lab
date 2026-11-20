import numpy as np
import pytest

from spectral_market import covariance as cv
from spectral_market.portfolio import backtest, min_variance_weights
from spectral_market.statarb import StatArbParams, fit_ou, performance, run_statarb

# --------------------------------------------------------------------------- covariance


def factor_world(rng, T, N=30, k=3):
    B = rng.standard_normal((N, k))
    F = rng.standard_normal((T, k))
    return 0.01 * (F @ B.T + rng.standard_normal((T, N)))


def test_sample_covariance_matches_numpy(rng):
    R = rng.standard_normal((100, 6))
    np.testing.assert_allclose(cv.sample_covariance(R), np.cov(R, rowvar=False), atol=1e-14)


def test_ledoit_wolf_matches_sklearn(rng):
    covariance = pytest.importorskip("sklearn.covariance")
    for T in (40, 100, 500):
        R = factor_world(rng, T)
        S, d = cv.ledoit_wolf(R)
        ref = covariance.LedoitWolf().fit(R)
        np.testing.assert_allclose(S, ref.covariance_, rtol=1e-10, atol=1e-16)
        assert d == pytest.approx(ref.shrinkage_, rel=1e-10)


def test_ledoit_wolf_shrinks_more_with_less_data(rng):
    R = factor_world(rng, 2000)
    assert cv.ledoit_wolf(R[:60])[1] > cv.ledoit_wolf(R)[1]


def test_rmt_clipping_preserves_trace_and_vols(rng):
    R = factor_world(rng, 300)
    S = cv.sample_covariance(R)
    C, kept = cv.rmt_clipped(R)
    np.testing.assert_allclose(np.diag(C), np.diag(S), rtol=1e-12)
    np.testing.assert_allclose(C, C.T, atol=1e-18)
    assert np.linalg.eigvalsh(C).min() > 0
    assert 1 <= kept <= 5  # three factors (market-adjusted edge may admit one or two more)


def test_cleaning_beats_sample_on_held_out_data(rng):
    """The point of cleaning: lower out-of-sample variance of the min-variance portfolio."""
    N, k = 40, 3
    B = rng.standard_normal((N, k))
    true = 1e-4 * (B @ B.T + np.eye(N))
    L = np.linalg.cholesky(true)
    R = rng.standard_normal((60, N)) @ L.T

    def oos(S):
        w = min_variance_weights(S)
        return w @ true @ w

    assert oos(cv.ledoit_wolf(R)[0]) < oos(cv.sample_covariance(R))
    assert oos(cv.rmt_clipped(R)[0]) < oos(cv.sample_covariance(R))
    assert oos(cv.factor_model(R, 3)) < oos(cv.sample_covariance(R))


def test_factor_model_structure(rng):
    R = factor_world(rng, 400)
    S = cv.factor_model(R, 3)
    np.testing.assert_allclose(np.diag(S), np.diag(cv.sample_covariance(R)), rtol=1e-12)
    assert np.linalg.matrix_rank(S - np.diag(np.diag(S))) >= 3


# --------------------------------------------------------------------------- portfolios


def test_min_variance_matches_closed_form(rng):
    R = factor_world(rng, 200, N=10)
    S = cv.sample_covariance(R)
    w = min_variance_weights(S)
    ref = np.linalg.solve(S, np.ones(10))
    np.testing.assert_allclose(w, ref / ref.sum(), rtol=1e-10)
    # KKT: S w is proportional to the ones vector.
    g = S @ w
    np.testing.assert_allclose(g, g.mean(), rtol=1e-10)


def test_long_only_min_variance_is_feasible_and_no_worse_than_equal_weight(rng):
    S = cv.sample_covariance(factor_world(rng, 200, N=15))
    w = min_variance_weights(S, long_only=True)
    assert w.min() >= 0 and w.sum() == pytest.approx(1.0)
    eq = np.full(15, 1 / 15)
    assert w @ S @ w <= eq @ S @ eq
    assert w @ S @ w >= min_variance_weights(S) @ S @ min_variance_weights(S) - 1e-18


def test_backtest_equal_weight_drifts_with_prices():
    R = np.log1p(np.array([[0.1, 0.0], [0.0, 0.0], [0.1, 0.0], [0.0, 0.1]]))
    bt = backtest(R, None, window=1, rebalance=10)
    held = np.array([0.5, 0.5])
    expected = []
    for day in np.expm1(R[1:]):
        r = held @ day
        expected.append(r)
        held = held * (1 + day) / (1 + r)
    np.testing.assert_allclose(bt.returns, expected)


def test_backtest_common_start(rng):
    R = factor_world(rng, 300, N=8)
    a = backtest(R, cv.sample_covariance, window=60, start=120)
    b = backtest(R, cv.sample_covariance, window=120, start=120)
    assert len(a.returns) == len(b.returns) == 180
    with pytest.raises(ValueError):
        backtest(R, cv.sample_covariance, window=150, start=120)


# --------------------------------------------------------------------------- stat arb


def simulate_ou(rng, n, kappa, sigma, T, dt=1 / 252):
    b = np.exp(-kappa * dt)
    sd = sigma * np.sqrt((1 - b**2) / (2 * kappa))
    X = np.zeros((T, n))
    for t in range(1, T):
        X[t] = b * X[t - 1] + sd * rng.standard_normal(n)
    return X


def test_fit_ou_recovers_parameters(rng):
    X = simulate_ou(rng, 200, kappa=25.0, sigma=0.3, T=2000)
    fit = fit_ou(X)
    assert fit.valid.all()
    assert np.median(fit.kappa) == pytest.approx(25.0, rel=0.15)
    sigma_eq = 0.3 / np.sqrt(2 * 25.0)
    assert np.median(fit.sigma_eq) == pytest.approx(sigma_eq, rel=0.1)


def test_fit_ou_flags_random_walk_and_trend():
    X = np.column_stack([np.arange(50.0), np.arange(50.0) ** 2])
    assert not fit_ou(X).valid.any()


def test_statarb_profits_from_mean_reverting_residuals(rng):
    """Market factor plus OU idiosyncratic prices: the engine should earn the reversion."""
    T, N = 900, 20
    market = 0.01 * rng.standard_normal((T, 1))
    beta = rng.uniform(0.5, 1.5, (1, N))
    X = simulate_ou(rng, N, kappa=40.0, sigma=0.25, T=T + 1)
    R = market * beta + np.diff(X, axis=0)
    res = run_statarb(R, StatArbParams(pca_window=120, n_factors=1))
    perf = performance(res.pnl)
    assert perf["sharpe"] > 2
    assert performance(res.net(50))["sharpe"] < perf["sharpe"]
    assert set(np.unique(res.positions)) <= {-1, 0, 1}


def test_statarb_finds_nothing_in_pure_noise(rng):
    R = 0.01 * rng.standard_normal((700, 15))
    perf = performance(run_statarb(R, StatArbParams(pca_window=120, n_factors=1)).pnl)
    assert abs(perf["sharpe"]) < 1.5


def test_performance_metrics():
    r = np.array([0.01, -0.02, 0.01, 0.01])
    p = performance(r)
    assert p["max_drawdown"] == pytest.approx(0.02)
    assert p["sharpe"] == pytest.approx(r.mean() / r.std(ddof=1) * np.sqrt(252))
