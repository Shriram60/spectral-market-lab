"""Sparse index tracking: replicate an index with a small long-only basket.

Tracking weights solve ``min ||r_index - R w||^2`` over fully invested long-only
portfolios (``w >= 0, sum(w) = 1``), a QP on the probability simplex.

Sparsity cannot come from an L1 penalty here: on the simplex ``||w||_1 = 1`` is constant.
So selection and weighting are separated. A long-only LASSO without the budget
constraint orders the stocks by when they enter its regularisation path. Each basket on
the path is then re-weighted by the simplex QP.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from numpy.typing import ArrayLike, NDArray

from .linalg import lasso_path
from .optimize import simplex_qp

TRADING_DAYS = 252


def tracking_error(index_returns: ArrayLike, portfolio_returns: ArrayLike) -> float:
    """Annualised standard deviation of the return difference."""
    d = np.asarray(index_returns, float) - np.asarray(portfolio_returns, float)
    return float(d.std(ddof=1) * np.sqrt(TRADING_DAYS))


def tracking_weights(R: ArrayLike, r_index: ArrayLike) -> NDArray[np.float64]:
    """Fully invested long-only weights minimising in-sample tracking variance."""
    R, r = np.asarray(R, float), np.asarray(r_index, float)
    m = R.shape[0]
    return simplex_qp(R.T @ R / m, -(R.T @ r) / m).x


@dataclass(frozen=True)
class TrackingPath:
    lambdas: NDArray[np.float64]
    weights: NDArray[np.float64]  # (n_lambdas, n_stocks)
    n_stocks: NDArray[np.int_]
    te_train: NDArray[np.float64]
    te_test: NDArray[np.float64]


def lasso_tracking_path(
    R_train: ArrayLike,
    r_train: ArrayLike,
    R_test: ArrayLike,
    r_test: ArrayLike,
    n_lambdas: int = 80,
    ratio: float = 1e-4,
) -> TrackingPath:
    """Baskets along the long-only LASSO path, re-weighted on the simplex and scored in
    and out of sample. Duplicate baskets along the path are collapsed."""
    R_train, R_test = np.asarray(R_train, float), np.asarray(R_test, float)
    r_train, r_test = np.asarray(r_train, float), np.asarray(r_test, float)
    lambdas, coefs, _ = lasso_path(
        R_train, r_train, n_lambdas=n_lambdas, ratio=ratio, fit_intercept=False, positive=True
    )
    rows, seen = [], set()
    for lam, w in zip(lambdas, coefs, strict=True):
        basket = tuple(np.flatnonzero(w > 0))
        if not basket or basket in seen:
            continue
        seen.add(basket)
        w = np.zeros_like(w)
        w[list(basket)] = tracking_weights(R_train[:, list(basket)], r_train)
        rows.append(
            (
                lam,
                w,
                len(basket),
                tracking_error(r_train, R_train @ w),
                tracking_error(r_test, R_test @ w),
            )
        )
    lam, W, k, te_tr, te_te = map(np.array, zip(*rows, strict=True))
    return TrackingPath(lam, W, k.astype(int), te_tr, te_te)


def random_basket_tracking(
    R_train: ArrayLike,
    r_train: ArrayLike,
    R_test: ArrayLike,
    r_test: ArrayLike,
    k: int,
    n_draws: int = 200,
    seed: int = 0,
) -> NDArray[np.float64]:
    """Out-of-sample tracking error of ``n_draws`` random k-stock baskets (simplex weights)."""
    R_train, R_test = np.asarray(R_train, float), np.asarray(R_test, float)
    rng = np.random.default_rng(seed)
    out = np.empty(n_draws)
    for i in range(n_draws):
        basket = rng.choice(R_train.shape[1], size=k, replace=False)
        w = tracking_weights(R_train[:, basket], r_train)
        out[i] = tracking_error(r_test, R_test[:, basket] @ w)
    return out


def greedy_tracking_path(
    R_train: ArrayLike,
    r_train: ArrayLike,
    R_test: ArrayLike,
    r_test: ArrayLike,
    max_stocks: int = 30,
) -> TrackingPath:
    """Forward stepwise selection under the budget constraint.

    At each step every remaining stock is tried, the enlarged basket is re-weighted by the
    simplex QP, and the stock giving the lowest in-sample tracking error is kept. Unlike
    the LASSO path, selection sees the same constraint the final weights obey. The Gram
    matrix is formed once, so each trial QP costs only a small sub-block.
    """
    R_train, R_test = np.asarray(R_train, float), np.asarray(R_test, float)
    r_train, r_test = np.asarray(r_train, float), np.asarray(r_test, float)
    m, n = R_train.shape
    H, g = R_train.T @ R_train / m, -(R_train.T @ r_train) / m
    c = r_train @ r_train / m
    basket: list[int] = []
    rows = []
    for _ in range(min(max_stocks, n)):
        best = None
        for j in set(range(n)) - set(basket):
            idx = [*basket, j]
            x = simplex_qp(H[np.ix_(idx, idx)], g[idx], check=False).x
            mse = x @ H[np.ix_(idx, idx)] @ x + 2 * g[idx] @ x + c
            if best is None or mse < best[0]:
                best = (mse, j, x)
        _, j, x = best
        basket.append(j)
        w = np.zeros(n)
        w[basket] = np.where(x > 1e-10, x, 0.0)  # rounding-level weights are not holdings
        w /= w.sum()
        rows.append(
            (
                np.nan,
                w,
                int((w > 0).sum()),
                tracking_error(r_train, R_train @ w),
                tracking_error(r_test, R_test @ w),
            )
        )
    lam, W, k, te_tr, te_te = map(np.array, zip(*rows, strict=True))
    return TrackingPath(lam, W, k.astype(int), te_tr, te_te)
