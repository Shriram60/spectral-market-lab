import numpy as np
import pytest
from conftest import matrix_with_condition

from spectral_market.linalg import (
    back_substitution,
    cholesky,
    classical_gram_schmidt,
    forward_substitution,
    householder_qr,
    modified_gram_schmidt,
    qr,
)

QR_METHODS = ["householder", "mgs", "cgs"]


# ---------------------------------------------------------------- triangular solves


def test_back_substitution_matches_numpy(rng):
    R = np.triu(rng.standard_normal((6, 6))) + 6 * np.eye(6)
    b = rng.standard_normal(6)
    np.testing.assert_allclose(back_substitution(R, b), np.linalg.solve(R, b), rtol=1e-12)


def test_forward_substitution_multiple_rhs(rng):
    L = np.tril(rng.standard_normal((5, 5))) + 5 * np.eye(5)
    B = rng.standard_normal((5, 3))
    np.testing.assert_allclose(forward_substitution(L, B), np.linalg.solve(L, B), rtol=1e-12)


def test_triangular_solves_reject_singular():
    with pytest.raises(np.linalg.LinAlgError):
        back_substitution(np.array([[1.0, 2.0], [0.0, 0.0]]), [1.0, 1.0])


# ---------------------------------------------------------------- Cholesky


def test_cholesky_reconstructs(rng):
    X = rng.standard_normal((20, 6))
    M = X.T @ X
    L = cholesky(M)
    np.testing.assert_allclose(L @ L.T, M, rtol=1e-12)
    np.testing.assert_allclose(L, np.linalg.cholesky(M), rtol=1e-10)
    assert np.allclose(L, np.tril(L))


def test_cholesky_rejects_indefinite():
    with pytest.raises(np.linalg.LinAlgError):
        cholesky(np.array([[1.0, 2.0], [2.0, 1.0]]))


# ---------------------------------------------------------------- QR


@pytest.mark.parametrize("method", QR_METHODS)
@pytest.mark.parametrize("shape", [(30, 5), (7, 7), (100, 1)])
def test_qr_is_a_valid_factorisation(rng, method, shape):
    A = rng.standard_normal(shape)
    Q, R = qr(A, method)
    n = shape[1]
    assert Q.shape == shape and R.shape == (n, n)
    np.testing.assert_allclose(Q @ R, A, atol=1e-12)
    np.testing.assert_allclose(Q.T @ Q, np.eye(n), atol=1e-12)
    assert np.allclose(R, np.triu(R))
    # R is unique up to the sign of each row.
    np.testing.assert_allclose(np.abs(R), np.abs(np.linalg.qr(A)[1]), atol=1e-10)


def test_householder_implicit_q_matches_explicit(rng):
    A = rng.standard_normal((12, 4))
    f = householder_qr(A)
    Q = f.q("complete")
    np.testing.assert_allclose(Q.T @ Q, np.eye(12), atol=1e-13)
    b = rng.standard_normal(12)
    np.testing.assert_allclose(f.apply_qt(b), Q.T @ b, atol=1e-13)
    np.testing.assert_allclose(f.apply_q(f.apply_qt(b)), b, atol=1e-13)


def test_orthogonality_loss_ordering(rng):
    """At cond = 1e7: Householder ~ eps, MGS ~ cond * eps, CGS ~ cond^2 * eps."""
    A = matrix_with_condition(rng, 80, 12, 1e7)
    eye = np.eye(12)
    loss = {
        "householder": np.linalg.norm(eye - (Q := householder_qr(A).q()).T @ Q, 2),
        "mgs": np.linalg.norm(eye - (Q := modified_gram_schmidt(A)[0]).T @ Q, 2),
        "cgs": np.linalg.norm(eye - (Q := classical_gram_schmidt(A)[0]).T @ Q, 2),
    }
    assert loss["householder"] < 1e-13
    assert 1e-13 < loss["mgs"] < 1e-6
    assert loss["cgs"] > 1e-3


def test_gram_schmidt_detects_dependent_columns():
    A = np.array([[1.0, 2.0], [1.0, 2.0], [1.0, 2.0]])
    for f in (classical_gram_schmidt, modified_gram_schmidt):
        with pytest.raises(np.linalg.LinAlgError):
            f(A)


def test_qr_rejects_wide_matrix():
    with pytest.raises(ValueError):
        householder_qr(np.ones((2, 3)))
