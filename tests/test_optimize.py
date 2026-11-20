import numpy as np
import pytest

from spectral_market.linalg import lasso_cd, lasso_kkt_violation, lasso_lambda_max, lasso_path
from spectral_market.optimize import check_convex, simplex_kkt_violation, simplex_qp
from spectral_market.tracking import (
    lasso_tracking_path,
    random_basket_tracking,
    tracking_error,
    tracking_weights,
)

# --------------------------------------------------------------------------- simplex QP


def random_qp(rng, n, rank=None):
    X = rng.standard_normal((rank or 2 * n, n))
    return X.T @ X / len(X), rng.standard_normal(n)


@pytest.mark.parametrize("n", [1, 2, 5, 20, 50])
def test_simplex_qp_satisfies_kkt(rng, n):
    for _ in range(10):
        H, g = random_qp(rng, n)
        res = simplex_qp(H, g)
        assert res.x.min() >= 0 and res.x.sum() == pytest.approx(1.0, abs=1e-14)
        assert res.kkt_violation < 1e-10


def test_simplex_qp_matches_scipy(rng):
    from scipy.optimize import minimize

    for _ in range(20):
        H, g = random_qp(rng, 8)
        res = simplex_qp(H, g)
        ref = minimize(
            lambda x, H=H, g=g: 0.5 * x @ H @ x + g @ x,
            np.full(8, 1 / 8),
            jac=lambda x, H=H, g=g: H @ x + g,
            bounds=[(0, None)] * 8,
            constraints=[{"type": "eq", "fun": lambda x: x.sum() - 1}],
            method="SLSQP",
            options={"ftol": 1e-15, "maxiter": 500},
        )
        f = 0.5 * res.x @ H @ res.x + g @ res.x
        assert f <= ref.fun + 1e-10


def test_simplex_qp_singular_hessian(rng):
    """More stocks than days: the covariance is singular but the QP is still convex."""
    H, g = random_qp(rng, 30, rank=10)
    res = simplex_qp(H, g)
    assert res.kkt_violation < 1e-9


def test_unconstrained_minimum_inside_simplex_is_found():
    H = np.eye(3)
    target = np.array([0.2, 0.3, 0.5])
    res = simplex_qp(H, -target)
    np.testing.assert_allclose(res.x, target, atol=1e-14)


def test_min_variance_two_assets_closed_form():
    s1, s2, rho = 0.2, 0.3, 0.1
    H = np.array([[s1**2, rho * s1 * s2], [rho * s1 * s2, s2**2]])
    w1 = (s2**2 - rho * s1 * s2) / (s1**2 + s2**2 - 2 * rho * s1 * s2)
    np.testing.assert_allclose(simplex_qp(H).x, [w1, 1 - w1], atol=1e-14)


def test_convexity_check():
    check_convex(np.zeros((3, 3)))  # PSD, singular
    with pytest.raises(ValueError, match="semidefinite"):
        simplex_qp(np.diag([1.0, -1.0]))
    with pytest.raises(ValueError, match="symmetric"):
        simplex_qp(np.array([[1.0, 1.0], [0.0, 1.0]]))


def test_kkt_violation_flags_suboptimal_point(rng):
    H, g = random_qp(rng, 6)
    assert simplex_kkt_violation(H, g, np.full(6, 1 / 6)) > 1e-3


# --------------------------------------------------------------------------- long-only LASSO


def test_positive_lasso_matches_sklearn(rng):
    linear_model = pytest.importorskip("sklearn.linear_model")
    A = rng.standard_normal((120, 10))
    b = A @ np.r_[1.0, -2.0, 0.5, np.zeros(7)] + 0.3 * rng.standard_normal(120)
    lam = 0.05
    fit = lasso_cd(A, b, lam, positive=True)
    ref = linear_model.Lasso(alpha=lam, positive=True, tol=1e-12, max_iter=100_000).fit(A, b)
    np.testing.assert_allclose(fit.coef, ref.coef_, atol=1e-7)
    assert fit.coef.min() >= 0
    assert lasso_kkt_violation(A, b, fit.coef, lam, positive=True) < 1e-8


def test_positive_lambda_max_zeroes_everything(rng):
    A = rng.standard_normal((80, 6))
    b = -A[:, 0] + 0.5 * A[:, 1]
    lam = lasso_lambda_max(A, b, positive=True)
    assert np.all(lasso_cd(A, b, lam * 1.0001, positive=True).coef == 0)
    assert lasso_cd(A, b, lam * 0.9, positive=True).coef[1] > 0
    assert lasso_lambda_max(A, b, positive=True) < lasso_lambda_max(A, b)


def test_positive_lasso_path_is_nonnegative(rng):
    A, b = rng.standard_normal((60, 8)), rng.standard_normal(60)
    _, coefs, _ = lasso_path(A, b, n_lambdas=20, positive=True)
    assert coefs.min() >= 0


# --------------------------------------------------------------------------- index tracking


def index_world(rng, T=600, N=20):
    R = 0.01 * (0.7 * rng.standard_normal((T, 1)) + rng.standard_normal((T, N)))
    w = rng.dirichlet(np.ones(N))
    return R, R @ w, w


def test_tracking_weights_recover_exact_index(rng):
    R, r, w = index_world(rng)
    np.testing.assert_allclose(tracking_weights(R, r), w, atol=1e-9)
    assert tracking_error(r, R @ w) == pytest.approx(0.0, abs=1e-12)


def test_tracking_error_is_annualised():
    d = np.array([0.01, -0.01] * 50)
    assert tracking_error(d, np.zeros_like(d)) == pytest.approx(d.std(ddof=1) * np.sqrt(252))


def test_lasso_tracking_path_trades_size_for_error(rng):
    R, r, _ = index_world(rng)
    path = lasso_tracking_path(R[:400], r[:400], R[400:], r[400:], n_lambdas=40)
    assert path.te_train[-1] < path.te_train[0]
    assert np.allclose(path.weights.sum(axis=1), 1.0)
    assert path.weights.min() >= 0
    k = int(path.n_stocks[len(path.n_stocks) // 2])
    random = random_basket_tracking(R[:400], r[:400], R[400:], r[400:], k, n_draws=30)
    assert np.median(random) > path.te_test[len(path.n_stocks) // 2]


def test_greedy_tracking_finds_the_true_basket(rng):
    """Index built from 4 of 20 stocks: greedy selection picks exactly those first."""
    from spectral_market.tracking import greedy_tracking_path

    R = 0.01 * (0.7 * rng.standard_normal((500, 1)) + rng.standard_normal((500, 20)))
    w = np.zeros(20)
    w[[2, 7, 11, 16]] = [0.4, 0.3, 0.2, 0.1]
    r = R @ w
    path = greedy_tracking_path(R[:300], r[:300], R[300:], r[300:], max_stocks=6)
    np.testing.assert_array_equal(path.n_stocks[:4], [1, 2, 3, 4])
    assert path.n_stocks.max() == 4  # extra stocks receive exactly zero weight
    assert set(np.flatnonzero(path.weights[3])) == {2, 7, 11, 16}
    np.testing.assert_allclose(path.weights[3], w, atol=1e-8)
    assert np.all(np.diff(path.te_train) <= 1e-12)
