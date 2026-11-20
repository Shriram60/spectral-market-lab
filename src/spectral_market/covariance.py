"""Covariance estimators for portfolio construction.

With N stocks and T days the sample covariance has N(N+1)/2 parameters estimated from
NT numbers. When q = N/T is not small its eigenvalues are spread out (Marchenko-Pastur):
the largest overstated, the smallest understated. A minimum-variance optimiser loads up
exactly on the smallest-eigenvalue directions, so it buys noise. Each estimator below
fixes the spectrum differently.

* **Ledoit-Wolf (2004)**: ``(1 - d) S + d mu I``, with ``mu = tr(S)/N`` and the shrinkage
  intensity ``d`` that minimises expected Frobenius loss, estimated in closed form.
* **RMT clipping (Laloux, Cizeau, Bouchaud & Potters, 2000)**: in correlation space,
  eigenvalues below the Marchenko-Pastur edge are indistinguishable from noise and are
  replaced by their average, which preserves the trace. Eigenvectors are kept.
* **PCA factor model**: the top k eigen-directions plus a diagonal idiosyncratic part.
"""

from __future__ import annotations

import numpy as np
from numpy.typing import ArrayLike, NDArray

from .linalg import jacobi_eigh
from .spectral import marchenko_pastur_bounds, market_adjusted_variance


def sample_covariance(R: ArrayLike) -> NDArray[np.float64]:
    """Unbiased sample covariance of the columns of ``R`` (T x N)."""
    R = np.asarray(R, dtype=float)
    Xc = R - R.mean(axis=0)
    return Xc.T @ Xc / (len(R) - 1)


def ledoit_wolf(R: ArrayLike) -> tuple[NDArray[np.float64], float]:
    """Ledoit-Wolf shrinkage towards ``mu I``; returns the estimate and the intensity.

    Uses the maximum-likelihood sample covariance (divide by T), matching the paper and
    scikit-learn's ``LedoitWolf``.
    """
    R = np.asarray(R, dtype=float)
    T, N = R.shape
    X = R - R.mean(axis=0)
    S = X.T @ X / T
    mu = np.trace(S) / N
    # d2: dispersion of S around the target; b2: estimation noise in S (both Frobenius^2/N).
    d2 = ((S - mu * np.eye(N)) ** 2).sum() / N
    X2 = X**2
    b2_bar = ((X2.T @ X2) / T - S**2).sum() / (T * N)
    b2 = min(b2_bar, d2)
    shrinkage = 0.0 if d2 == 0 else b2 / d2
    return (1 - shrinkage) * S + shrinkage * mu * np.eye(N), float(shrinkage)


def _to_cov(C: NDArray, sd: NDArray) -> NDArray:
    return sd[:, None] * C * sd[None, :]


def rmt_clipped(R: ArrayLike, market_adjusted: bool = True) -> tuple[NDArray[np.float64], int]:
    """Eigenvalue-clipped covariance; returns the estimate and the number of eigenvalues kept.

    The noise edge is ``(1 + sqrt(N/T))^2``, optionally scaled by the market-adjusted
    bulk variance ``1 - lambda_1/N``. The cleaned correlation matrix is renormalised to a
    unit diagonal before the sample volatilities are put back.
    """
    R = np.asarray(R, dtype=float)
    T, N = R.shape
    S = sample_covariance(R)
    sd = np.sqrt(np.diag(S))
    C = S / np.outer(sd, sd)
    lam, V = jacobi_eigh(C)  # ascending
    sigma2 = market_adjusted_variance(lam) if market_adjusted else 1.0
    edge = marchenko_pastur_bounds(N / T, sigma2)[1] if N < T else np.inf
    noise = lam < edge
    kept = int((~noise).sum())
    if noise.any():
        lam = lam.copy()
        lam[noise] = lam[noise].mean()
    Cc = (V * lam) @ V.T
    d = np.sqrt(np.diag(Cc))
    Cc = Cc / np.outer(d, d)
    return _to_cov(Cc, sd), kept


def factor_model(R: ArrayLike, k: int | None = None) -> NDArray[np.float64]:
    """Top-``k`` principal components of the correlation matrix plus diagonal residual.

    ``k=None`` keeps the eigenvalues above the market-adjusted Marchenko-Pastur edge,
    decided inside the estimation window (no look-ahead).
    """
    R = np.asarray(R, dtype=float)
    S = sample_covariance(R)
    sd = np.sqrt(np.diag(S))
    C = S / np.outer(sd, sd)
    lam, V = jacobi_eigh(C)
    if k is None:
        T, N = R.shape
        edge = marchenko_pastur_bounds(min(N / T, 1.0), market_adjusted_variance(lam))[1]
        k = max(int((lam > edge).sum()), 1)
    lam, V = lam[::-1][:k], V[:, ::-1][:, :k]
    common = (V * lam) @ V.T
    Cf = common + np.diag(1.0 - np.diag(common))
    return _to_cov(Cf, sd)
