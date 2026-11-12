"""Polynomial trend fitting with the diagnostics the original project was missing.

The original script fitted ``price = c0 + c1 t`` on 8 points and reported the
eigenvalues of ``A^T A`` as "variance explained". Those eigenvalues depend only on the
day indices, never on prices: for t = 1..8 they are always 210.40 and 1.60. Their
ratio is the condition number of ``A^T A``, a measure of numerical sensitivity. The
quantity that measures variance explained is R^2, reported here together with
standard errors, leverage, a residual-orthogonality check and the Durbin-Watson
statistic (which shows why OLS standard errors on price levels can't be trusted).
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from numpy.typing import ArrayLike, NDArray
from scipy import stats

from .linalg import back_substitution, householder_qr, jacobi_svd, lstsq


def design_matrix(
    t: ArrayLike, degree: int, center: float = 0.0, scale: float = 1.0
) -> NDArray[np.float64]:
    """Polynomial (Vandermonde) design matrix in ``z = (t - center) / scale``."""
    z = (np.asarray(t, dtype=float) - center) / scale
    return np.vander(z, degree + 1, increasing=True)


def standardisation(t: ArrayLike) -> tuple[float, float]:
    """Affine map sending ``[min(t), max(t)]`` to ``[-1, 1]``."""
    t = np.asarray(t, dtype=float)
    lo, hi = float(t.min()), float(t.max())
    return (lo + hi) / 2.0, ((hi - lo) / 2.0) or 1.0


@dataclass(frozen=True)
class TrendFit:
    t: NDArray[np.float64]
    y: NDArray[np.float64]  # fitted target: price, or log price when ``log``
    degree: int
    log: bool
    center: float
    scale: float
    coef: NDArray[np.float64]  # coefficients in the standardised basis z
    fitted: NDArray[np.float64]
    residuals: NDArray[np.float64]
    r2: float
    adj_r2: float
    sigma: float  # residual standard error
    coef_se: NDArray[np.float64]  # classical OLS standard errors (assume iid errors)
    xtx_inv: NDArray[np.float64]  # (A^T A)^-1, computed as R^-1 R^-T
    cond: float  # condition number of the design matrix actually used
    leverage: NDArray[np.float64]  # diagonal of the hat matrix
    orthogonality: float  # max |A^T e| / (||A|| ||e||); ~eps when e is in C(A)-perp
    durbin_watson: float  # ~2 for uncorrelated residuals, ~0 for strongly autocorrelated

    @property
    def n(self) -> int:
        return self.y.size

    @property
    def dof(self) -> int:
        return self.n - (self.degree + 1)

    def design(self, t: ArrayLike) -> NDArray[np.float64]:
        return design_matrix(t, self.degree, self.center, self.scale)

    def predict(self, t: ArrayLike) -> NDArray[np.float64]:
        """Prediction in price units (exponentiated when the fit is in log space)."""
        y = self.design(t) @ self.coef
        return np.exp(y) if self.log else y

    def slope_at(self, t: float) -> float:
        """d(target)/dt at time ``t``: price per day, or log-growth per day."""
        z = (t - self.center) / self.scale
        k = np.arange(1, self.degree + 1)
        return float((k * self.coef[1:] * z ** (k - 1)).sum() / self.scale)

    def slope_ci(self, level: float = 0.95) -> tuple[float, float, float]:
        """Linear trends only: ``(slope, low, high)`` per day under iid-error assumptions."""
        if self.degree != 1:
            raise ValueError("slope_ci is defined for degree-1 trends")
        q = stats.t.ppf(0.5 + level / 2.0, self.dof)
        slope, se = self.coef[1] / self.scale, self.coef_se[1] / self.scale
        return float(slope), float(slope - q * se), float(slope + q * se)

    def prediction_interval(
        self, t: ArrayLike, level: float = 0.95
    ) -> tuple[NDArray, NDArray, NDArray]:
        """``(prediction, low, high)`` for new observations at ``t``, in price units.

        Uses ``sigma * sqrt(1 + a^T (A^T A)^-1 a)``: the leverage of the new point
        is why intervals widen as you extrapolate. Assumes iid errors.
        """
        a = self.design(t)
        q = stats.t.ppf(0.5 + level / 2.0, self.dof)
        y = a @ self.coef
        half = q * self.sigma * np.sqrt(1.0 + np.einsum("ij,jk,ik->i", a, self.xtx_inv, a))
        to_price = np.exp if self.log else (lambda v: v)
        return to_price(y), to_price(y - half), to_price(y + half)


def fit_trend(
    prices: ArrayLike,
    t: ArrayLike | None = None,
    degree: int = 1,
    log: bool = False,
    standardize: bool = True,
    solver: str = "qr",
) -> TrendFit:
    """Fit a polynomial trend by least squares and compute its diagnostics."""
    prices = np.asarray(prices, dtype=float).ravel()
    n, p = prices.size, degree + 1
    if n <= p:
        raise ValueError(f"need more than {p} points for a degree-{degree} fit, got {n}")
    if log and np.any(prices <= 0):
        raise ValueError("log trend requires positive prices")
    t = np.arange(1, n + 1, dtype=float) if t is None else np.asarray(t, dtype=float)
    y = np.log(prices) if log else prices

    center, scale = standardisation(t) if standardize else (0.0, 1.0)
    A = design_matrix(t, degree, center, scale)
    coef = lstsq(A, y, method=solver)
    fitted = A @ coef
    e = y - fitted

    rss = float(e @ e)
    tss = float(((y - y.mean()) ** 2).sum())
    r2 = 1.0 - rss / tss if tss > 0 else 1.0
    sigma2 = rss / (n - p)

    # Cov(coef) = sigma^2 (A^T A)^-1 = sigma^2 R^-1 R^-T, computed without forming A^T A.
    f = householder_qr(A)
    R_inv = back_substitution(f.R, np.eye(p))
    xtx_inv = R_inv @ R_inv.T
    se = np.sqrt(sigma2 * np.diag(xtx_inv))
    leverage = (f.q("reduced") ** 2).sum(axis=1)

    s = jacobi_svd(A).s
    e_norm = np.linalg.norm(e)
    orth = float(np.abs(A.T @ e).max() / (s[0] * e_norm)) if e_norm > 0 else 0.0
    dw = float((np.diff(e) ** 2).sum() / rss) if rss > 0 else float("nan")

    return TrendFit(
        t=t,
        y=y,
        degree=degree,
        log=log,
        center=center,
        scale=scale,
        coef=coef,
        fitted=fitted,
        residuals=e,
        r2=r2,
        adj_r2=1.0 - (1.0 - r2) * (n - 1) / (n - p),
        sigma=float(np.sqrt(sigma2)),
        coef_se=se,
        xtx_inv=xtx_inv,
        cond=float(s[0] / s[-1]),
        leverage=leverage,
        orthogonality=orth,
        durbin_watson=dw,
    )


def drift_from_returns(prices: ArrayLike, level: float = 0.95) -> tuple[float, float, float]:
    """Mean daily log return with a confidence interval: ``(mu, low, high)``.

    If log prices follow a random walk with drift, the residuals of a trend regression
    on price *levels* are themselves a random walk (Durbin-Watson near 0) and the OLS
    standard error of the slope is far too small. Differencing first gives roughly
    independent returns, whose mean estimates the same drift with honest uncertainty.
    """
    r = np.diff(np.log(np.asarray(prices, dtype=float)))
    mu, se = r.mean(), r.std(ddof=1) / np.sqrt(r.size)
    q = stats.t.ppf(0.5 + level / 2.0, r.size - 1)
    return float(mu), float(mu - q * se), float(mu + q * se)
