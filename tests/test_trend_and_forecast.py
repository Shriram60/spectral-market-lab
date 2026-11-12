import numpy as np
import pytest

from spectral_market.forecast import diebold_mariano, trend_forecast_weights, walk_forward
from spectral_market.trend import design_matrix, fit_trend


def test_eigenvalues_of_normal_matrix_ignore_prices():
    """The original project's 'variance explained' numbers came from A^T A, which only
    depends on the day indices: the same for every stock."""
    A = design_matrix(np.arange(1, 9), degree=1)
    eig = np.sort(np.linalg.eigvalsh(A.T @ A))[::-1]
    np.testing.assert_allclose(eig, [210.40, 1.60], atol=5e-3)


def test_standardising_time_fixes_conditioning():
    t = np.arange(1, 9)
    raw = np.linalg.cond(design_matrix(t, 1))
    fit = fit_trend(np.linspace(100, 120, 8) + np.sin(t))
    assert raw > 10 and fit.cond < 2


def test_linear_trend_recovers_known_slope(rng):
    t = np.arange(1, 201, dtype=float)
    y = 50 + 0.3 * t + rng.standard_normal(200)
    fit = fit_trend(y, degree=1)
    slope, lo, hi = fit.slope_ci()
    assert lo < 0.3 < hi
    assert slope == pytest.approx(0.3, abs=0.01)
    assert 0.9 < fit.r2 <= 1.0
    assert fit.orthogonality < 1e-12
    assert fit.leverage.sum() == pytest.approx(2.0)  # trace of hat matrix = #params


def test_r2_equals_squared_correlation_for_lines(rng):
    y = rng.standard_normal(50).cumsum() + 100
    fit = fit_trend(y, degree=1)
    r = np.corrcoef(np.arange(50), y)[0, 1]
    assert fit.r2 == pytest.approx(r**2)


@pytest.mark.parametrize("solver", ["normal", "qr", "svd"])
def test_solvers_agree_on_well_conditioned_trend(rng, solver):
    y = rng.standard_normal(100).cumsum() + 500
    ref = fit_trend(y, degree=3, solver="qr")
    np.testing.assert_allclose(fit_trend(y, degree=3, solver=solver).coef, ref.coef, rtol=1e-9)


def test_log_trend_predicts_in_price_units():
    t = np.arange(1, 101, dtype=float)
    prices = 100 * np.exp(0.01 * t)
    fit = fit_trend(prices, degree=1, log=True)
    assert fit.slope_at(50) == pytest.approx(0.01)
    assert fit.predict([101])[0] == pytest.approx(100 * np.exp(1.01))


def test_durbin_watson_flags_random_walk_residuals(rng):
    walk = rng.standard_normal(500).cumsum() + 1000
    noise = 1000 + rng.standard_normal(500)
    assert fit_trend(walk).durbin_watson < 0.3
    assert 1.6 < fit_trend(noise).durbin_watson < 2.4


# ---------------------------------------------------------------- forecasting


@pytest.mark.parametrize("degree", [0, 1, 2, 3])
@pytest.mark.parametrize("horizon", [1, 5])
def test_forecast_weights_reproduce_polynomials(degree, horizon):
    window = 20
    w = trend_forecast_weights(window, horizon, degree)
    t = np.arange(window, dtype=float)
    t_h = window - 1 + horizon
    for k in range(degree + 1):
        assert w @ t**k == pytest.approx(t_h**k, rel=1e-9)


def test_forecast_weights_equal_explicit_refit(rng):
    y = rng.standard_normal(30).cumsum()
    window, horizon, degree = 30, 3, 2
    w = trend_forecast_weights(window, horizon, degree)
    coef = np.polyfit(np.arange(window), y, degree)
    assert w @ y == pytest.approx(np.polyval(coef, window - 1 + horizon))


def test_walk_forward_cannot_beat_random_walk_on_a_random_walk(rng):
    y = 0.01 * rng.standard_normal(2000).cumsum()
    ev = walk_forward(y, window=20, horizon=1, degree=1)
    assert ev.n_forecasts == 2000 - 20
    assert ev.rmse_ratio > 1.0


def test_walk_forward_wins_on_a_real_trend(rng):
    t = np.arange(1500)
    y = 0.002 * t + 0.0005 * rng.standard_normal(1500)
    ev = walk_forward(y, window=60, horizon=5, degree=1)
    assert ev.rmse_ratio < 0.5 and ev.directional_accuracy > 0.95 and ev.dm_stat < 0


def test_diebold_mariano_detects_worse_forecast(rng):
    e_bench = rng.standard_normal(500)
    stat, p = diebold_mariano(2.0 * e_bench, e_bench, horizon=1)
    assert stat > 0 and p < 1e-6


def test_prediction_interval_widens_when_extrapolating(rng):
    y = 100 + 0.5 * np.arange(1, 31) + rng.standard_normal(30)
    fit = fit_trend(y)
    _, lo, hi = fit.prediction_interval([15, 31, 60])
    width = hi - lo
    assert width[0] < width[1] < width[2]
    A = fit.design(fit.t)
    np.testing.assert_allclose(fit.xtx_inv, np.linalg.inv(A.T @ A), atol=1e-15)


def test_drift_from_returns_is_honest_where_ols_is_not(rng):
    """On a random walk with drift, the OLS slope CI is far narrower than the truth."""
    from spectral_market.trend import drift_from_returns

    logp = np.log(100) + (0.0005 + 0.015 * rng.standard_normal(1200)).cumsum()
    prices = np.exp(logp)
    _, lo_ols, hi_ols = fit_trend(prices, log=True).slope_ci()
    _, lo_ret, hi_ret = drift_from_returns(prices)
    assert lo_ret < 0.0005 < hi_ret
    assert (hi_ret - lo_ret) > 5 * (hi_ols - lo_ols)
