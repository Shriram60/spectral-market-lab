import numpy as np
import pytest
from conftest import matrix_with_condition

from spectral_market.linalg import condition_number, jacobi_svd, numerical_rank
from spectral_market.linalg.svd import _round_robin


@pytest.mark.parametrize("shape", [(40, 6), (6, 6), (5, 13), (300, 41), (3, 1), (1, 1), (1, 5)])
def test_svd_matches_numpy(rng, shape):
    A = rng.standard_normal(shape)
    r = jacobi_svd(A)
    k = min(shape)
    assert r.U.shape == (shape[0], k) and r.s.shape == (k,) and r.Vt.shape == (k, shape[1])
    np.testing.assert_allclose(r.s, np.linalg.svd(A, compute_uv=False), rtol=1e-12, atol=1e-13)
    np.testing.assert_allclose(r.reconstruct(), A, atol=1e-12)
    np.testing.assert_allclose(r.U.T @ r.U, np.eye(k), atol=1e-12)
    np.testing.assert_allclose(r.Vt @ r.Vt.T, np.eye(k), atol=1e-12)
    assert np.all(np.diff(r.s) <= 0)


def test_svd_rank_deficient_has_orthonormal_completion(rng):
    A = rng.standard_normal((25, 3)) @ rng.standard_normal((3, 8))
    r = jacobi_svd(A)
    assert numerical_rank(A) == 3
    np.testing.assert_allclose(r.s[3:], 0.0, atol=1e-12)
    np.testing.assert_allclose(r.U.T @ r.U, np.eye(8), atol=1e-12)
    np.testing.assert_allclose(r.reconstruct(), A, atol=1e-12)


def test_svd_zero_matrix():
    r = jacobi_svd(np.zeros((4, 3)))
    np.testing.assert_array_equal(r.s, 0.0)
    np.testing.assert_allclose(r.U.T @ r.U, np.eye(3))


def test_small_singular_values_have_high_relative_accuracy(rng):
    """Graded matrix: tiny singular values are recovered to high relative accuracy."""
    A = matrix_with_condition(rng, 60, 10, 1e10)
    s_true = np.geomspace(1.0, 1e-10, 10)
    np.testing.assert_allclose(jacobi_svd(A).s, s_true, rtol=1e-5)


def test_condition_number(rng):
    A = matrix_with_condition(rng, 30, 5, 1e4)
    assert condition_number(A) == pytest.approx(1e4, rel=1e-8)
    assert condition_number(np.ones((3, 2))) == np.inf


@pytest.mark.parametrize("n", [2, 5, 8, 11])
def test_round_robin_covers_every_pair_once(n):
    seen = []
    for P, Q in _round_robin(n):
        idx = np.concatenate([P, Q])
        assert len(set(idx.tolist())) == idx.size  # pairs within a round are disjoint
        seen += list(zip(P.tolist(), Q.tolist(), strict=True))
    assert sorted(seen) == [(p, q) for p in range(n) for q in range(p + 1, n)]
