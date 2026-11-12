"""Regularised least squares.

Ridge regression through the SVD
--------------------------------
With ``A = U diag(s) V^T`` and ``c = U^T b``, the ridge solution is

    x(lam) = V diag(f_i / s_i) c,      f_i = s_i^2 / (s_i^2 + lam)

The *filter factors* ``f_i`` show what ridge actually does: directions with large
singular values pass through almost unchanged and directions with tiny singular values
(the ones that amplify noise) are damped. One SVD gives the whole regularisation path,
the effective degrees of freedom ``sum(f_i)`` and the generalised cross-validation score

    GCV(lam) = (RSS / m) / (1 - dof / m)^2

which estimates leave-one-out prediction error without refitting.

LASSO by coordinate descent
---------------------------
Minimises ``(1 / 2m) ||b - A x - x0||^2 + lam ||x||_1``. Each coordinate update is a
1-D problem with a closed-form soft-thresholding solution. Optimality is certified by
the KKT conditions (see :func:`lasso_kkt_violation`).
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from numpy.typing import ArrayLike, NDArray

from .svd import jacobi_svd


def _prepare(A: ArrayLike, b: ArrayLike, fit_intercept: bool):
    A = np.asarray(A, dtype=float)
    b = np.asarray(b, dtype=float)
    if A.ndim != 2 or b.ndim != 1 or A.shape[0] != b.shape[0]:
        raise ValueError(f"incompatible shapes A{A.shape}, b{b.shape}")
    if fit_intercept:
        a_mean, b_mean = A.mean(axis=0), float(b.mean())
        return A - a_mean, b - b_mean, a_mean, b_mean
    return A, b, np.zeros(A.shape[1]), 0.0


# --------------------------------------------------------------------------- ridge


@dataclass(frozen=True)
class RidgePath:
    lambdas: NDArray[np.float64]
    coefs: NDArray[np.float64]  # (len(lambdas), n_features)
    intercepts: NDArray[np.float64]
    dof: NDArray[np.float64]  # effective degrees of freedom (includes intercept)
    rss: NDArray[np.float64]
    gcv: NDArray[np.float64]
    singular_values: NDArray[np.float64]

    @property
    def best(self) -> int:
        """Index of the lambda that minimises GCV."""
        return int(np.nanargmin(self.gcv))

    def filter_factors(self, lam: float) -> NDArray[np.float64]:
        s2 = self.singular_values**2
        return s2 / (s2 + lam) if lam > 0 else (s2 > 0).astype(float)


def ridge_path(
    A: ArrayLike, b: ArrayLike, lambdas: ArrayLike, fit_intercept: bool = False
) -> RidgePath:
    """Ridge solutions, degrees of freedom and GCV for every lambda from a single SVD."""
    Ac, bc, a_mean, b_mean = _prepare(A, b, fit_intercept)
    lambdas = np.atleast_1d(np.asarray(lambdas, dtype=float))
    if np.any(lambdas < 0):
        raise ValueError("lambdas must be non-negative")
    m = bc.shape[0]

    svd = jacobi_svd(Ac)
    s, c = svd.s, svd.U.T @ bc
    outside = max(float(bc @ bc - c @ c), 0.0)  # part of b no lambda can fit

    s2 = s**2
    denom = s2[None, :] + lambdas[:, None]
    pos = denom > 0
    f = np.divide(s2, denom, out=np.zeros_like(denom), where=pos)
    gain = np.divide(s, denom, out=np.zeros_like(denom), where=pos)

    coefs = (gain * c) @ svd.Vt
    rss = outside + (((1.0 - f) * c) ** 2).sum(axis=1)
    dof = f.sum(axis=1) + float(fit_intercept)
    with np.errstate(divide="ignore", invalid="ignore"):
        gcv = np.where(dof < m, (rss / m) / (1.0 - dof / m) ** 2, np.inf)
    return RidgePath(
        lambdas=lambdas,
        coefs=coefs,
        intercepts=b_mean - coefs @ a_mean,
        dof=dof,
        rss=rss,
        gcv=gcv,
        singular_values=s,
    )


def ridge(
    A: ArrayLike, b: ArrayLike, lam: float, fit_intercept: bool = False
) -> tuple[NDArray[np.float64], float]:
    """Single ridge solution. Returns ``(coef, intercept)``."""
    path = ridge_path(A, b, [lam], fit_intercept=fit_intercept)
    return path.coefs[0], float(path.intercepts[0])


# --------------------------------------------------------------------------- lasso


@dataclass(frozen=True)
class LassoFit:
    coef: NDArray[np.float64]
    intercept: float
    lam: float
    n_iter: int
    converged: bool


def _soft_threshold(z: float, gamma: float) -> float:
    return np.sign(z) * max(abs(z) - gamma, 0.0)


def lasso_lambda_max(A: ArrayLike, b: ArrayLike, fit_intercept: bool = True) -> float:
    """Smallest lambda at which every LASSO coefficient is exactly zero."""
    Ac, bc, *_ = _prepare(A, b, fit_intercept)
    return float(np.max(np.abs(Ac.T @ bc)) / Ac.shape[0])


def lasso_cd(
    A: ArrayLike,
    b: ArrayLike,
    lam: float,
    fit_intercept: bool = True,
    max_iter: int = 10_000,
    tol: float = 1e-10,
    coef_init: ArrayLike | None = None,
) -> LassoFit:
    """LASSO by cyclic coordinate descent with soft-thresholding."""
    Ac, bc, a_mean, b_mean = _prepare(A, b, fit_intercept)
    m, n = Ac.shape
    col_sq = (Ac**2).sum(axis=0) / m
    x = np.zeros(n) if coef_init is None else np.array(coef_init, dtype=float)
    r = bc - Ac @ x
    scale = max(np.sqrt(bc @ bc / m), np.finfo(float).tiny)

    converged, n_iter = False, 0
    while n_iter < max_iter:
        n_iter += 1
        max_step = 0.0
        for j in range(n):
            if col_sq[j] == 0.0:
                continue
            old = x[j]
            rho = Ac[:, j] @ r / m + col_sq[j] * old
            new = _soft_threshold(rho, lam) / col_sq[j]
            if new != old:
                r -= Ac[:, j] * (new - old)
                x[j] = new
                max_step = max(max_step, abs(new - old) * np.sqrt(col_sq[j]))
        if max_step <= tol * scale:
            converged = True
            break
    return LassoFit(
        coef=x, intercept=b_mean - float(a_mean @ x), lam=lam, n_iter=n_iter, converged=converged
    )


def lasso_path(
    A: ArrayLike,
    b: ArrayLike,
    lambdas: ArrayLike | None = None,
    n_lambdas: int = 60,
    ratio: float = 1e-4,
    fit_intercept: bool = True,
    **kwargs,
) -> tuple[NDArray[np.float64], NDArray[np.float64], NDArray[np.float64]]:
    """LASSO path with warm starts. Returns ``(lambdas, coefs, intercepts)``.

    Without explicit ``lambdas``, uses a log grid from ``lambda_max`` down to
    ``ratio * lambda_max``.
    """
    if lambdas is None:
        top = lasso_lambda_max(A, b, fit_intercept)
        lambdas = np.geomspace(top, top * ratio, n_lambdas)
    lambdas = np.sort(np.asarray(lambdas, dtype=float))[::-1]
    coefs, intercepts, warm = [], [], None
    for lam in lambdas:
        fit = lasso_cd(A, b, lam, fit_intercept=fit_intercept, coef_init=warm, **kwargs)
        coefs.append(fit.coef)
        intercepts.append(fit.intercept)
        warm = fit.coef
    return lambdas, np.array(coefs), np.array(intercepts)


def lasso_kkt_violation(
    A: ArrayLike, b: ArrayLike, coef: ArrayLike, lam: float, fit_intercept: bool = True
) -> float:
    """Largest violation of the LASSO optimality (KKT) conditions.

    With ``g = A^T (b - A x) / m``: active coordinates need ``g_j = lam * sign(x_j)``
    and inactive ones need ``|g_j| <= lam``. A value near zero certifies optimality.
    """
    Ac, bc, *_ = _prepare(A, b, fit_intercept)
    x = np.asarray(coef, dtype=float)
    g = Ac.T @ (bc - Ac @ x) / Ac.shape[0]
    active = x != 0
    viol_active = np.abs(g[active] - lam * np.sign(x[active]))
    viol_zero = np.maximum(np.abs(g[~active]) - lam, 0.0)
    return float(max(viol_active.max(initial=0.0), viol_zero.max(initial=0.0)))
