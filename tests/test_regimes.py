import itertools

import numpy as np
import pytest

from spectral_market.hmm import (
    HMM,
    baum_welch,
    forward,
    forward_backward,
    predicted_variance,
    viterbi,
)
from spectral_market.kalman import (
    _loglik,
    fit_local_trend,
    kalman_filter,
    rts_smoother,
    trend_forecast,
)

# --------------------------------------------------------------------------- Kalman


def test_zero_state_noise_kalman_is_ols(rng):
    """With no state noise the filter is recursive least squares: final state = OLS line."""
    t = np.arange(40.0)
    y = 3.0 + 0.2 * t + rng.standard_normal(40)
    res = kalman_filter(y, 0.0, 0.0, 1.0, diffuse=1e12)
    # State is (level at t, slope); OLS on [1, t] gives level at the last t.
    A = np.column_stack([np.ones(40), t])
    b0, b1 = np.linalg.lstsq(A, y, rcond=None)[0]
    np.testing.assert_allclose(res.filtered_mean[-1], [b0 + b1 * t[-1], b1], rtol=1e-6)
    # ...and the filtered covariance matches the OLS covariance of (level_T, slope).
    XtX_inv = np.linalg.inv(A.T @ A)
    J = np.array([[1.0, t[-1]], [0.0, 1.0]])
    np.testing.assert_allclose(res.filtered_cov[-1], J @ XtX_inv @ J.T, rtol=1e-5)


def test_scalar_likelihood_matches_matrix_filter(rng):
    y = np.cumsum(0.01 * rng.standard_normal(300))
    for q in [(1e-4, 1e-8, 1e-5), (1e-5, 0.0, 1e-4), (1e-3, 1e-6, 1e-8)]:
        assert _loglik(y, *q, 1e7, 2) == pytest.approx(kalman_filter(y, *q).loglik, rel=1e-8)


def test_smoother_is_at_least_as_certain_as_filter(rng):
    y = np.cumsum(0.01 * rng.standard_normal(200)) + 0.002 * rng.standard_normal(200)
    res = kalman_filter(y, 1e-4, 1e-7, 4e-6)
    sm, sP = rts_smoother(res)
    np.testing.assert_allclose(sm[-1], res.filtered_mean[-1])
    assert np.all(sP[:-1, 0, 0] <= res.filtered_cov[:-1, 0, 0] + 1e-15)


def test_mle_recovers_local_level_noise(rng):
    """Random walk observed with noise: the fit separates level and observation variance."""
    level = np.cumsum(0.01 * rng.standard_normal(3000))
    y = level + 0.005 * rng.standard_normal(3000)
    fit = fit_local_trend(y)
    assert fit.q_level == pytest.approx(1e-4, rel=0.3)
    assert fit.r == pytest.approx(2.5e-5, rel=0.3)
    assert fit.signal_to_noise < 1e-3


def test_mle_detects_a_moving_slope(rng):
    slope = np.cumsum(1e-3 * rng.standard_normal(2000))
    level = np.cumsum(slope + 0.002 * rng.standard_normal(2000))
    fit = fit_local_trend(level)
    assert fit.q_slope == pytest.approx(1e-6, rel=0.5)
    profiled = fit_local_trend(level, snr=1e-10)
    assert 2 * (fit.loglik - profiled.loglik) > 50


def test_trend_forecast_extrapolates_slope():
    y = 1.0 + 0.1 * np.arange(30.0)
    res = kalman_filter(y, 1e-10, 1e-12, 1e-6)
    assert trend_forecast(res, 5)[-1] == pytest.approx(y[-1] + 0.5, abs=1e-4)


# --------------------------------------------------------------------------- HMM


def two_regime_model():
    return HMM(
        pi=np.array([0.6, 0.4]),
        A=np.array([[0.97, 0.03], [0.08, 0.92]]),
        mu=np.array([0.001, -0.002]),
        sigma=np.array([0.007, 0.02]),
    )


def sample(model, T, rng):
    s = np.empty(T, dtype=int)
    s[0] = rng.choice(model.k, p=model.pi)
    for t in range(1, T):
        s[t] = rng.choice(model.k, p=model.A[s[t - 1]])
    return model.mu[s] + model.sigma[s] * rng.standard_normal(T), s


def brute_force(model, x):
    """Sum and argmax over all K^T paths."""
    from scipy.stats import norm

    best, total = None, 0.0
    for path in itertools.product(range(model.k), repeat=len(x)):
        p = model.pi[path[0]] * norm.pdf(x[0], model.mu[path[0]], model.sigma[path[0]])
        for t in range(1, len(x)):
            p *= model.A[path[t - 1], path[t]]
            p *= norm.pdf(x[t], model.mu[path[t]], model.sigma[path[t]])
        total += p
        if best is None or p > best[0]:
            best = (p, path)
    return np.log(total), np.array(best[1])


def test_forward_and_viterbi_match_brute_force(rng):
    model = two_regime_model()
    x, _ = sample(model, 8, rng)
    loglik, path = brute_force(model, x)
    assert forward(model, x)[2] == pytest.approx(loglik, rel=1e-12)
    np.testing.assert_array_equal(viterbi(model, x), path)


def test_posteriors_are_distributions(rng):
    model = two_regime_model()
    x, _ = sample(model, 300, rng)
    gamma, xi, _ = forward_backward(model, x)
    np.testing.assert_allclose(gamma.sum(axis=1), 1.0)
    assert xi.sum() == pytest.approx(len(x) - 1)
    np.testing.assert_allclose(xi.sum(axis=1), gamma[:-1].sum(axis=0), rtol=1e-10)


def test_baum_welch_recovers_regimes(rng):
    model = two_regime_model()
    x, s = sample(model, 4000, rng)
    fit = baum_welch(x, 2, n_init=4)
    np.testing.assert_allclose(fit.sigma, model.sigma, rtol=0.08)
    np.testing.assert_allclose(np.diag(fit.A), np.diag(model.A), atol=0.03)
    assert (viterbi(fit, x) == s).mean() > 0.9
    assert fit.loglik > forward(model, x)[2] - 5  # at least as good as the truth, up to noise
    np.testing.assert_allclose(fit.stationary @ fit.A, fit.stationary, atol=1e-12)


def test_bic_prefers_true_number_of_states(rng):
    x, _ = sample(two_regime_model(), 3000, rng)
    bics = [baum_welch(x, k, n_init=4).bic(len(x)) for k in (1, 2, 3)]
    assert int(np.argmin(bics)) == 1


def test_predicted_variance_bounds(rng):
    model = two_regime_model()
    x, _ = sample(model, 500, rng)
    alpha, _, _ = forward(model, x)
    v = predicted_variance(model, alpha)
    assert v.min() >= model.sigma.min() ** 2 - 1e-12
    assert v.max() <= (model.sigma.max() ** 2 + np.ptp(model.mu) ** 2) + 1e-12
