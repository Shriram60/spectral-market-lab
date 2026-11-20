"""Minimum-variance portfolios and a rolling out-of-sample backtest.

Global minimum variance (fully invested, shorting allowed) is the equality-constrained
QP ``min w^T S w  s.t.  1^T w = 1``. Its KKT system

    [ 2S  1 ] [ w  ]   [ 0 ]
    [ 1^T 0 ] [ nu ] = [ 1 ]

has the closed-form solution ``w = S^{-1} 1 / (1^T S^{-1} 1)``, computed here with one
Cholesky factorisation and two triangular solves. The long-only version adds ``w >= 0``
and is solved exactly by the simplex QP.

The in-sample variance of the optimiser's own portfolio, ``w^T S w``, is a *biased*
forecast: the optimiser picks the directions where ``S`` happens to underestimate risk.
The ratio of realised to predicted volatility measures that bias, and a well-cleaned
covariance brings it close to one.
"""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass

import numpy as np
from numpy.typing import ArrayLike, NDArray

from .linalg import back_substitution, cholesky, forward_substitution
from .optimize import simplex_qp

TRADING_DAYS = 252


def min_variance_weights(S: ArrayLike, long_only: bool = False) -> NDArray[np.float64]:
    S = np.asarray(S, dtype=float)
    if long_only:
        return simplex_qp(S, check=False).x
    L = cholesky(S)
    y = back_substitution(L.T, forward_substitution(L, np.ones(len(S))))  # S^{-1} 1
    return y / y.sum()


@dataclass(frozen=True)
class Backtest:
    returns: NDArray[np.float64]  # daily out-of-sample portfolio returns
    predicted_vol: NDArray[np.float64]  # annualised, one per rebalance
    realised_vol: NDArray[np.float64]  # annualised, one per holding period
    turnover: NDArray[np.float64]  # sum |w_new - w_drifted| at each rebalance
    gross_leverage: NDArray[np.float64]  # sum |w| at each rebalance

    @property
    def vol(self) -> float:
        return float(self.returns.std(ddof=1) * np.sqrt(TRADING_DAYS))

    @property
    def bias_ratio(self) -> float:
        """Realised over predicted volatility, pooled across holding periods."""
        return float(np.sqrt(np.mean(self.realised_vol**2) / np.mean(self.predicted_vol**2)))


def backtest(
    R: ArrayLike,
    estimator: Callable[[NDArray], NDArray] | None,
    window: int = 252,
    rebalance: int = 21,
    long_only: bool = False,
    start: int | None = None,
) -> Backtest:
    """Re-estimate every ``rebalance`` days from the trailing ``window`` days and hold.

    The out-of-sample period begins at row ``start`` (default ``window``); passing the
    same ``start`` to backtests with different windows makes them comparable.
    ``estimator=None`` is the equal-weight portfolio. Returns are simple returns
    ``exp(r) - 1``, and weights drift with prices between rebalances.
    """
    R = np.asarray(R, dtype=float)
    simple = np.expm1(R)
    T, N = R.shape
    out, pred, real, turn, lev = [], [], [], [], []
    w_prev = np.zeros(N)
    first = window if start is None else start
    if first < window:
        raise ValueError("start must leave a full estimation window")
    for t0 in range(first, T, rebalance):
        hist = R[t0 - window : t0]
        if estimator is None:
            w, p = np.full(N, 1.0 / N), np.nan
        else:
            S = estimator(hist)
            w = min_variance_weights(S, long_only)
            p = float(np.sqrt(w @ S @ w * TRADING_DAYS))
        turn.append(np.abs(w - w_prev).sum())
        lev.append(np.abs(w).sum())
        period = simple[t0 : t0 + rebalance]
        held = w.copy()
        r_period = []
        for day in period:
            r = held @ day
            r_period.append(r)
            held = held * (1 + day) / (1 + r)
        w_prev = held
        out.extend(r_period)
        pred.append(p)
        real.append(
            np.std(r_period, ddof=1) * np.sqrt(TRADING_DAYS) if len(r_period) > 1 else np.nan
        )
    real_arr, pred_arr = np.array(real), np.array(pred)
    ok = np.isfinite(real_arr)
    return Backtest(
        returns=np.array(out),
        predicted_vol=pred_arr[ok],
        realised_vol=real_arr[ok],
        turnover=np.array(turn),
        gross_leverage=np.array(lev),
    )
