import numpy as np
import pytest
from conftest import matrix_with_condition

from spectral_market.linalg import (
    lasso_cd,
    lasso_kkt_violation,
    lasso_lambda_max,
    lasso_path,
    lstsq,
    lstsq_svd,
    ridge,
    ridge_path,
)

METHODS = ["normal", "qr", "svd"]


@pytest.mark.parametrize("method", METHODS)
def test_lstsq_matches_numpy(rng, method):
    A, b = rng.standard_normal((50, 7)), rng.standard_normal(50)
    expected = np.linalg.lstsq(A, b, rcond=None)[0]
    np.testing.assert_allclose(lstsq(A, b, method), expected, rtol=1e-10, atol=1e-12)


@pytest.mark.parametrize("method", METHODS)
def test_residual_is_orthogonal_to_column_space(rng, method):
    """The defining property of least squares: A^T (b - A x) = 0."""
    A, b = rng.standard_normal((40, 5)), rng.standard_normal(40)
    e = b - A @ lstsq(A, b, method)
    assert np.abs(A.T @ e).max() < 1e-11


@pytest.mark.parametrize("method", METHODS)
def test_lstsq_multiple_right_hand_sides(rng, method):
    A, B = rng.standard_normal((30, 4)), rng.standard_normal((30, 3))
    np.testing.assert_allclose(
        lstsq(A, B, method), np.linalg.lstsq(A, B, rcond=None)[0], rtol=1e-10, atol=1e-12
    )


def test_svd_gives_minimum_norm_solution_when_rank_deficient(rng):
    A = rng.standard_normal((20, 4))
    A = np.column_stack([A, A[:, :2] @ [1.0, -2.0]])  # 5th column is dependent
    b = rng.standard_normal(20)
    np.testing.assert_allclose(lstsq_svd(A, b), np.linalg.pinv(A) @ b, atol=1e-10)


def test_normal_equations_fail_where_qr_survives(rng):
    """cond(A) = 1e9 squares to 1e18 > 1/eps: Cholesky breaks or loses all accuracy."""
    A = matrix_with_condition(rng, 60, 8, 1e9)
    x_true = rng.standard_normal(8)
    b = A @ x_true
    err_qr = np.linalg.norm(lstsq(A, b, "qr") - x_true) / np.linalg.norm(x_true)
    assert err_qr < 1e-4
    try:
        x_ne = lstsq(A, b, "normal")
    except np.linalg.LinAlgError:
        return
    assert np.linalg.norm(x_ne - x_true) / np.linalg.norm(x_true) > 1e-2


def test_unknown_method():
    with pytest.raises(ValueError):
        lstsq(np.eye(2), np.ones(2), "magic")


# ---------------------------------------------------------------- ridge


def test_ridge_matches_closed_form(rng):
    A, b = rng.standard_normal((40, 6)), rng.standard_normal(40)
    lam = 3.0
    coef, intercept = ridge(A, b, lam)
    np.testing.assert_allclose(coef, np.linalg.solve(A.T @ A + lam * np.eye(6), A.T @ b))
    assert intercept == 0.0


def test_ridge_with_intercept_matches_sklearn(rng):
    sklearn = pytest.importorskip("sklearn.linear_model")
    A, b = rng.standard_normal((40, 6)), rng.standard_normal(40) + 5.0
    coef, intercept = ridge(A, b, 2.0, fit_intercept=True)
    ref = sklearn.Ridge(alpha=2.0).fit(A, b)
    np.testing.assert_allclose(coef, ref.coef_, atol=1e-12)
    assert intercept == pytest.approx(ref.intercept_)


def test_ridge_path_dof_rss_and_gcv(rng):
    A, b = rng.standard_normal((30, 5)), rng.standard_normal(30)
    lams = np.array([0.0, 0.5, 5.0, 500.0])
    path = ridge_path(A, b, lams)
    for i, lam in enumerate(lams):
        H = A @ np.linalg.solve(A.T @ A + lam * np.eye(5), A.T)
        resid = b - H @ b
        assert path.rss[i] == pytest.approx(resid @ resid)
        assert path.dof[i] == pytest.approx(np.trace(H))
        assert path.gcv[i] == pytest.approx((resid @ resid / 30) / (1 - np.trace(H) / 30) ** 2)
    assert np.all(np.diff(path.dof) < 0)  # more shrinkage, fewer effective parameters
    np.testing.assert_allclose(path.coefs[0], np.linalg.lstsq(A, b, rcond=None)[0], atol=1e-12)


# ---------------------------------------------------------------- lasso


def test_lasso_satisfies_kkt_and_matches_sklearn(rng):
    sklearn = pytest.importorskip("sklearn.linear_model")
    A = rng.standard_normal((80, 10))
    b = A[:, :3] @ [2.0, -1.0, 0.5] + 0.1 * rng.standard_normal(80)
    lam = 0.05
    fit = lasso_cd(A, b, lam)
    assert fit.converged
    assert lasso_kkt_violation(A, b, fit.coef, lam) < 1e-9
    ref = sklearn.Lasso(alpha=lam, tol=1e-12, max_iter=100_000).fit(A, b)
    np.testing.assert_allclose(fit.coef, ref.coef_, atol=1e-8)


def test_lasso_lambda_max_zeroes_everything(rng):
    A, b = rng.standard_normal((50, 6)), rng.standard_normal(50)
    top = lasso_lambda_max(A, b)
    np.testing.assert_array_equal(lasso_cd(A, b, top * 1.0001).coef, 0.0)
    assert np.any(lasso_cd(A, b, top * 0.9).coef != 0)


def test_lasso_path_is_sparse_then_dense(rng):
    A = rng.standard_normal((60, 8))
    b = A[:, 0] - 2 * A[:, 1] + 0.1 * rng.standard_normal(60)
    lams, coefs, _ = lasso_path(A, b, n_lambdas=30)
    nnz = (coefs != 0).sum(axis=1)
    assert nnz[0] == 0 and nnz[-1] >= 2
    assert np.all(np.diff(lams) < 0)
