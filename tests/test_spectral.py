import numpy as np
import pytest

from spectral_market.linalg import jacobi_eigh
from spectral_market.spectral import (
    adjusted_rand_index,
    correlation,
    correlation_affinity,
    eigengap_k,
    eigenportfolio_weights,
    inverse_participation_ratio,
    kmeans,
    marchenko_pastur_bounds,
    marchenko_pastur_pdf,
    market_adjusted_variance,
    n_signal,
    normalized_laplacian,
    pca_correlation,
    remove_market_mode,
    shuffled,
    spectral_clustering,
)

# --------------------------------------------------------------------------- eigensolver


@pytest.mark.parametrize("n", [1, 2, 5, 30])
def test_jacobi_eigh_matches_numpy(rng, n):
    X = rng.standard_normal((n, n))
    S = X + X.T
    w, V = jacobi_eigh(S)
    np.testing.assert_allclose(w, np.linalg.eigvalsh(S), atol=1e-12 * max(1, np.abs(w).max()))
    np.testing.assert_allclose(V.T @ V, np.eye(n), atol=1e-13)
    np.testing.assert_allclose(S @ V, V * w, atol=1e-12 * np.linalg.norm(S))


def test_jacobi_eigh_repeated_eigenvalues(rng):
    Q, _ = np.linalg.qr(rng.standard_normal((6, 6)))
    S = (Q * np.array([3.0, 3.0, 3.0, 1.0, 1.0, -2.0])) @ Q.T
    w, V = jacobi_eigh(S)
    np.testing.assert_allclose(w, [-2, 1, 1, 3, 3, 3], atol=1e-13)
    np.testing.assert_allclose(S @ V, V * w, atol=1e-13)


def test_jacobi_eigh_small_eigenvalues_have_relative_accuracy():
    """A graded positive definite matrix: Jacobi keeps the tiny eigenvalue's digits."""
    d = np.array([1.0, 1e-6, 1e-12])
    H = np.array([[1.0, 0.5, 0.25], [0.5, 1.0, 0.5], [0.25, 0.5, 1.0]])
    S = np.sqrt(d)[:, None] * H * np.sqrt(d)[None, :]
    w, _ = jacobi_eigh(S)
    assert np.all(w > 0)
    np.testing.assert_allclose(np.prod(w), np.linalg.det(H) * np.prod(d), rtol=1e-8)


def test_jacobi_eigh_rejects_asymmetric():
    with pytest.raises(ValueError, match="symmetric"):
        jacobi_eigh(np.array([[1.0, 2.0], [0.0, 1.0]]))


# --------------------------------------------------------------------------- PCA


def factor_returns(rng, T=1500, groups=(8, 8, 8), market=0.6, sector=0.5):
    """Returns from a one-market, one-factor-per-group model with known structure."""
    N = sum(groups)
    labels = np.repeat(np.arange(len(groups)), groups)
    m = rng.standard_normal((T, 1))
    s = rng.standard_normal((T, len(groups)))[:, labels]
    e = rng.standard_normal((T, N))
    vols = rng.uniform(0.01, 0.03, N)
    return (market * m + sector * s + e) * vols, labels


def test_pca_matches_numpy_eigh_of_correlation(rng):
    R, _ = factor_returns(rng, T=400)
    pca = pca_correlation(R)
    C = np.corrcoef(R, rowvar=False)
    np.testing.assert_allclose(pca.eigenvalues, np.linalg.eigvalsh(C)[::-1], atol=1e-10)
    np.testing.assert_allclose(pca.eigenvalues.sum(), R.shape[1], rtol=1e-12)
    V = pca.eigenvectors
    np.testing.assert_allclose(C @ V, V * pca.eigenvalues, atol=1e-10)
    assert np.all(V.sum(axis=0) >= 0)  # sign convention


def test_correlation_matches_numpy(rng):
    R = rng.standard_normal((200, 7))
    np.testing.assert_allclose(correlation(R), np.corrcoef(R, rowvar=False), atol=1e-12)


def test_market_mode_is_uniform_and_delocalised(rng):
    R, _ = factor_returns(rng)
    pca = pca_correlation(R)
    v1 = pca.eigenvectors[:, 0]
    assert np.all(v1 > 0)
    ipr = inverse_participation_ratio(pca.eigenvectors)
    assert ipr[0] < 2.0 / R.shape[1]  # spread over every stock, close to the 1/N floor


def test_eigenportfolio_weights_are_normalised():
    w = eigenportfolio_weights([0.5, -0.5, 0.5], [0.01, 0.02, 0.04])
    np.testing.assert_allclose(np.abs(w).sum(), 1.0)
    np.testing.assert_allclose(w / w[0], [1.0, -0.5, 0.25])


def test_remove_market_mode_is_orthogonal_to_it(rng):
    R, _ = factor_returns(rng, T=500)
    v1 = pca_correlation(R).eigenvectors[:, 0]
    assert np.abs(remove_market_mode(R) @ v1).max() < 1e-10


# --------------------------------------------------------------------------- RMT


def test_marchenko_pastur_pdf_integrates_to_one():
    q = 0.25
    lo, hi = marchenko_pastur_bounds(q)
    x = np.linspace(lo, hi, 200_001)
    assert abs(np.trapezoid(marchenko_pastur_pdf(x, q), x) - 1.0) < 1e-3
    assert (lo, hi) == pytest.approx((0.25, 2.25))


def test_pure_noise_spectrum_stays_inside_marchenko_pastur(rng):
    T, N = 2000, 100
    lam = pca_correlation(rng.standard_normal((T, N))).eigenvalues
    lo, hi = marchenko_pastur_bounds(N / T)
    assert lam.max() < hi * 1.05 and lam.min() > lo * 0.9
    assert n_signal(lam, N / T) <= 1


def test_factor_model_signal_count(rng):
    R, labels = factor_returns(rng)
    lam = pca_correlation(R).eigenvalues
    q = R.shape[1] / R.shape[0]
    # market + 2 sector contrasts: the sum of the 3 sector directions lies in the market mode
    assert n_signal(lam, q) == 3
    sigma2 = market_adjusted_variance(lam)
    assert 0 < sigma2 < 1
    assert n_signal(lam, q, sigma2) >= n_signal(lam, q)


def test_shuffle_keeps_marginals_and_kills_correlation(rng):
    R, _ = factor_returns(rng)
    S = shuffled(R, rng)
    np.testing.assert_allclose(np.sort(S, axis=0), np.sort(R, axis=0))
    off = correlation(S)[~np.eye(R.shape[1], dtype=bool)]
    assert np.abs(off).max() < 0.15


def test_mp_bounds_reject_bad_ratio():
    with pytest.raises(ValueError):
        marchenko_pastur_bounds(1.5)


# --------------------------------------------------------------------------- clustering


def test_adjusted_rand_index_matches_sklearn(rng):
    sklearn_metrics = pytest.importorskip("sklearn.metrics")
    for _ in range(20):
        a, b = rng.integers(0, 4, 60), rng.integers(0, 5, 60)
        assert adjusted_rand_index(a, b) == pytest.approx(
            sklearn_metrics.adjusted_rand_score(a, b), abs=1e-12
        )
    assert adjusted_rand_index([0, 0, 1, 1], [5, 5, 9, 9]) == 1.0


def test_kmeans_recovers_separated_blobs(rng):
    centers = np.array([[0, 0], [10, 0], [0, 10]])
    X = np.vstack([c + rng.standard_normal((30, 2)) for c in centers])
    truth = np.repeat([0, 1, 2], 30)
    res = kmeans(X, 3, seed=1)
    assert adjusted_rand_index(res.labels, truth) == 1.0
    with pytest.raises(ValueError):
        kmeans(X, 0)


def test_normalized_laplacian_spectrum(rng):
    R, _ = factor_returns(rng, T=500)
    L = normalized_laplacian(correlation_affinity(correlation(R)))
    w = np.linalg.eigvalsh(L)
    assert w.min() > -1e-12 and w.max() < 2 + 1e-12
    assert abs(w.min()) < 1e-12  # connected graph: exactly one zero eigenvalue


def test_spectral_clustering_recovers_sectors(rng):
    R, labels = factor_returns(rng)
    Z = remove_market_mode(R)
    W = correlation_affinity(Z.T @ Z / len(Z))
    w = np.linalg.eigvalsh(normalized_laplacian(W))
    assert eigengap_k(w, 8) == 3
    assert adjusted_rand_index(spectral_clustering(W, 3), labels) == 1.0
