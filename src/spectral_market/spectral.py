"""Spectral analysis of a returns matrix.

* **PCA through the SVD of the data**, not the eigendecomposition of the correlation
  matrix: ``Z / sqrt(T) = U S V^T`` gives eigenvalues ``S^2`` and eigenvectors ``V``
  without ever forming ``Z^T Z`` (the same conditioning argument as QR vs the normal
  equations).
* **Random matrix theory.** For T observations of N *independent* unit-variance series,
  the correlation eigenvalues fill the Marchenko-Pastur interval
  ``[(1 - sqrt(q))^2, (1 + sqrt(q))^2]``, ``q = N / T``. Eigenvalues outside it carry
  structure; those inside are indistinguishable from noise.
* **Spectral clustering** on a correlation graph: the eigenvectors of the normalised
  Laplacian ``L = I - D^{-1/2} W D^{-1/2}`` with the smallest eigenvalues embed the
  stocks so that tightly connected groups become separable by k-means.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from numpy.typing import ArrayLike, NDArray

from .linalg import jacobi_eigh, jacobi_svd

# --------------------------------------------------------------------------- PCA


def standardize(R: ArrayLike) -> NDArray[np.float64]:
    """Columns rescaled to zero mean and unit (population) variance."""
    R = np.asarray(R, dtype=float)
    sd = R.std(axis=0)
    if np.any(sd == 0):
        raise ValueError("a column has zero variance")
    return (R - R.mean(axis=0)) / sd


def correlation(R: ArrayLike) -> NDArray[np.float64]:
    Z = standardize(R)
    return Z.T @ Z / Z.shape[0]


@dataclass(frozen=True)
class PCA:
    eigenvalues: NDArray[np.float64]  # descending; sum = N for a correlation matrix
    eigenvectors: NDArray[np.float64]  # columns, sign fixed so loadings sum >= 0

    @property
    def explained(self) -> NDArray[np.float64]:
        return self.eigenvalues / self.eigenvalues.sum()


def pca_correlation(R: ArrayLike) -> PCA:
    """PCA of the correlation matrix of ``R`` (T x N) via the SVD of the standardised data."""
    Z = standardize(R)
    svd = jacobi_svd(Z / np.sqrt(Z.shape[0]))
    V = svd.Vt.T
    V = V * np.where(V.sum(axis=0) < 0, -1.0, 1.0)
    return PCA(eigenvalues=svd.s**2, eigenvectors=V)


def eigenportfolio_weights(vector: ArrayLike, sigma: ArrayLike) -> NDArray[np.float64]:
    """Tradeable weights for an eigenvector of the correlation matrix.

    Correlation eigenvectors live in standardised units; dividing by each stock's
    volatility converts them to capital weights (Avellaneda & Lee, 2010). Normalised so
    the absolute weights sum to one.
    """
    w = np.asarray(vector, float) / np.asarray(sigma, float)
    return w / np.abs(w).sum()


def inverse_participation_ratio(V: ArrayLike) -> NDArray[np.float64]:
    """``sum_i v_i^4`` per unit eigenvector: 1/N if spread evenly, 1 if on one stock."""
    V = np.asarray(V, dtype=float)
    return (V**4).sum(axis=0)


# --------------------------------------------------------------------------- RMT


def marchenko_pastur_bounds(q: float, sigma2: float = 1.0) -> tuple[float, float]:
    if not 0 < q <= 1:
        raise ValueError("q = N / T must lie in (0, 1]")
    return sigma2 * (1 - np.sqrt(q)) ** 2, sigma2 * (1 + np.sqrt(q)) ** 2


def marchenko_pastur_pdf(x: ArrayLike, q: float, sigma2: float = 1.0) -> NDArray[np.float64]:
    """Limiting eigenvalue density of a pure-noise correlation matrix."""
    x = np.asarray(x, dtype=float)
    lo, hi = marchenko_pastur_bounds(q, sigma2)
    inside = (x > lo) & (x < hi)
    out = np.zeros_like(x)
    xi = x[inside]
    out[inside] = np.sqrt((hi - xi) * (xi - lo)) / (2 * np.pi * q * sigma2 * xi)
    return out


def market_adjusted_variance(eigenvalues: ArrayLike) -> float:
    """Variance left for the noise bulk after the market mode (Laloux et al., 1999).

    The trace of an N x N correlation matrix is N. The largest eigenvalue absorbs
    ``lambda_1`` of it, so the remaining eigenvalues behave like noise of variance
    ``1 - lambda_1 / N`` and the bulk edge shrinks accordingly.
    """
    lam = np.asarray(eigenvalues, dtype=float)
    return float(1.0 - lam.max() / lam.size)


def n_signal(eigenvalues: ArrayLike, q: float, sigma2: float = 1.0) -> int:
    """Number of eigenvalues above the Marchenko-Pastur upper edge."""
    return int((np.asarray(eigenvalues) > marchenko_pastur_bounds(q, sigma2)[1]).sum())


def shuffled(R: ArrayLike, rng: np.random.Generator) -> NDArray[np.float64]:
    """Permute each column independently in time: keeps every marginal distribution
    (fat tails included) and destroys all cross-correlation. An empirical null for RMT."""
    R = np.asarray(R, dtype=float)
    return np.column_stack([rng.permutation(R[:, j]) for j in range(R.shape[1])])


def remove_market_mode(R: ArrayLike, n_modes: int = 1) -> NDArray[np.float64]:
    """Standardised returns with the top ``n_modes`` principal components projected out."""
    Z = standardize(R)
    V = pca_correlation(R).eigenvectors[:, :n_modes]
    return Z - (Z @ V) @ V.T


# --------------------------------------------------------------------------- clustering


@dataclass(frozen=True)
class KMeansResult:
    labels: NDArray[np.int_]
    centers: NDArray[np.float64]
    inertia: float


def _kmeans_plus_plus(X: NDArray, k: int, rng: np.random.Generator) -> NDArray:
    centers = [X[rng.integers(len(X))]]
    for _ in range(1, k):
        d2 = ((X[:, None, :] - np.array(centers)[None]) ** 2).sum(-1).min(axis=1)
        p = d2 / d2.sum() if d2.sum() > 0 else np.full(len(X), 1 / len(X))
        centers.append(X[rng.choice(len(X), p=p)])
    return np.array(centers)


def kmeans(
    X: ArrayLike, k: int, n_init: int = 25, max_iter: int = 300, seed: int = 0
) -> KMeansResult:
    """Lloyd's algorithm with k-means++ seeding; best of ``n_init`` restarts."""
    X = np.asarray(X, dtype=float)
    if not 1 <= k <= len(X):
        raise ValueError("k must be between 1 and the number of points")
    rng = np.random.default_rng(seed)
    best: KMeansResult | None = None
    for _ in range(n_init):
        centers = _kmeans_plus_plus(X, k, rng)
        for _ in range(max_iter):
            labels = ((X[:, None, :] - centers[None]) ** 2).sum(-1).argmin(axis=1)
            new = np.array(
                [
                    X[labels == j].mean(axis=0) if np.any(labels == j) else centers[j]
                    for j in range(k)
                ]
            )
            if np.allclose(new, centers, atol=1e-12, rtol=0):
                break
            centers = new
        labels = ((X[:, None, :] - centers[None]) ** 2).sum(-1).argmin(axis=1)
        inertia = float(((X - centers[labels]) ** 2).sum())
        if best is None or inertia < best.inertia:
            best = KMeansResult(labels=labels, centers=centers, inertia=inertia)
    return best


def adjusted_rand_index(a: ArrayLike, b: ArrayLike) -> float:
    """Agreement between two partitions, corrected for chance: 1 = identical, ~0 = random."""
    _, ai = np.unique(np.asarray(a), return_inverse=True)
    _, bi = np.unique(np.asarray(b), return_inverse=True)
    table = np.zeros((ai.max() + 1, bi.max() + 1))
    np.add.at(table, (ai, bi), 1)

    def pairs(x):
        return (x * (x - 1) / 2).sum()

    n = len(ai)
    index = pairs(table)
    rows, cols = pairs(table.sum(axis=1)), pairs(table.sum(axis=0))
    expected = rows * cols / (n * (n - 1) / 2)
    maximum = (rows + cols) / 2
    if maximum == expected:
        return 1.0
    return float((index - expected) / (maximum - expected))


def correlation_affinity(C: ArrayLike) -> NDArray[np.float64]:
    """Graph weights from a correlation matrix: positive co-movement only, no self-loops."""
    W = np.clip(np.asarray(C, dtype=float), 0.0, None)
    np.fill_diagonal(W, 0.0)
    return W


def normalized_laplacian(W: ArrayLike) -> NDArray[np.float64]:
    W = np.asarray(W, dtype=float)
    d = W.sum(axis=1)
    inv_sqrt = np.where(d > 0, 1.0 / np.sqrt(np.where(d > 0, d, 1.0)), 0.0)
    return np.eye(len(W)) - inv_sqrt[:, None] * W * inv_sqrt[None, :]


def spectral_embedding(W: ArrayLike, k: int) -> tuple[NDArray, NDArray]:
    """Laplacian eigenvalues (ascending) and the row-normalised k-dimensional embedding."""
    eigenvalues, vectors = jacobi_eigh(normalized_laplacian(W))
    U = vectors[:, :k]
    norms = np.linalg.norm(U, axis=1, keepdims=True)
    return eigenvalues, U / np.where(norms > 0, norms, 1.0)


def spectral_clustering(W: ArrayLike, k: int, seed: int = 0) -> NDArray[np.int_]:
    """Normalised spectral clustering (Ng, Jordan & Weiss, 2002)."""
    _, U = spectral_embedding(W, k)
    return kmeans(U, k, seed=seed).labels


def eigengap_k(laplacian_eigenvalues: ArrayLike, k_max: int) -> int:
    """Number of clusters suggested by the largest gap among the smallest eigenvalues."""
    lam = np.sort(np.asarray(laplacian_eigenvalues, dtype=float))[: k_max + 1]
    return int(np.argmax(np.diff(lam)) + 1)
