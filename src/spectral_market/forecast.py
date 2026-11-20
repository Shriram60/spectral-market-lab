"""Walk-forward evaluation of rolling trend forecasts against a random walk.

A rolling least-squares trend forecast is a *fixed linear filter*. Re-index every
window as t = 0..w-1 and the design matrix ``A`` is identical for all windows, so the
h-step-ahead forecast is

    y_hat = a_h^T x_hat = a_h^T (A^T A)^-1 A^T y = w^T y,    w = Q R^-T a_h

where ``a_h`` is the design row at t = w - 1 + h. The weights are computed once from a
Householder QR and the whole backtest is a single matrix-vector product over a sliding
window view, so thousands of refits cost one dot product each.

Properties that make good tests: ``sum(w) = 1`` (a constant series is forecast exactly)
and ``w^T t^k = t_h^k`` for every ``k <= degree`` (polynomials are reproduced exactly).
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from numpy.lib.stride_tricks import sliding_window_view
from numpy.typing import ArrayLike, NDArray
from scipy import stats

from .linalg import forward_substitution, householder_qr
from .trend import design_matrix, standardisation


def trend_forecast_weights(window: int, horizon: int, degree: int) -> NDArray[np.float64]:
    """Weights ``w`` such that the rolling degree-``degree`` trend forecast is ``w @ y``."""
    if window <= degree:
        raise ValueError(f"window {window} too short for degree {degree}")
    t = np.arange(window, dtype=float)
    center, scale = standardisation(t)
    A = design_matrix(t, degree, center, scale)
    a_h = design_matrix([window - 1 + horizon], degree, center, scale)[0]
    f = householder_qr(A)
    z = forward_substitution(f.R.T, a_h)  # z = R^-T a_h
    return f.apply_q(np.concatenate([z, np.zeros(window - z.size)]))  # w = Q z


@dataclass(frozen=True)
class ForecastEval:
    window: int
    horizon: int
    degree: int
    n_forecasts: int
    rmse_model: float  # in units of the series (log price => approx. fractional error)
    rmse_random_walk: float
    directional_accuracy: float  # share of forecasts that called the direction right
    dm_stat: float  # Diebold-Mariano: > 0 means the model is worse than the random walk
    dm_pvalue: float

    @property
    def rmse_ratio(self) -> float:
        """Theil-style U: model RMSE / random-walk RMSE. Below 1 beats the random walk."""
        return self.rmse_model / self.rmse_random_walk


def loss_differential_test(d: ArrayLike, horizon: int) -> tuple[float, float]:
    """Diebold-Mariano test that a loss differential ``d_t`` has mean zero.

    Overlapping h-step forecasts give MA(h-1) loss differentials, so the long-run
    variance uses autocovariances up to lag h-1 (Bartlett weights keep it positive), and
    the Harvey-Leybourne-Newbold factor corrects the small-sample size.
    Returns ``(statistic, two-sided p-value)``; negative means the first model is better.
    """
    d = np.asarray(d, dtype=float)
    T = d.size
    dc = d - d.mean()
    lrv = dc @ dc / T
    for lag in range(1, horizon):
        lrv += 2.0 * (1.0 - lag / horizon) * (dc[lag:] @ dc[:-lag]) / T
    if lrv <= 0:
        return float("nan"), float("nan")
    dm = d.mean() / np.sqrt(lrv / T)
    hln = np.sqrt((T + 1 - 2 * horizon + horizon * (horizon - 1) / T) / T)
    stat = float(dm * hln)
    return stat, float(2.0 * stats.t.sf(abs(stat), df=T - 1))


def diebold_mariano(e_model: ArrayLike, e_bench: ArrayLike, horizon: int) -> tuple[float, float]:
    """Diebold-Mariano test on squared-error loss with the Harvey-Leybourne-Newbold fix."""
    d = np.asarray(e_model, float) ** 2 - np.asarray(e_bench, float) ** 2
    return loss_differential_test(d, horizon)


def walk_forward(series: ArrayLike, window: int, horizon: int, degree: int) -> ForecastEval:
    """Evaluate rolling trend forecasts of ``series`` (use log prices) out of sample.

    At each origin the model sees only the last ``window`` observations and forecasts
    ``horizon`` steps ahead. The benchmark is the random walk (no change).
    """
    y = np.asarray(series, dtype=float).ravel()
    n_fc = y.size - window - horizon + 1
    if n_fc < 10:
        raise ValueError("series too short for this window and horizon")
    w = trend_forecast_weights(window, horizon, degree)
    windows = sliding_window_view(y, window)[:n_fc]
    pred = windows @ w
    last = windows[:, -1]
    actual = y[window - 1 + horizon :]

    e_model, e_rw = actual - pred, actual - last
    moved = actual != last
    hits = np.sign(pred - last)[moved] == np.sign(actual - last)[moved]
    dm, p = diebold_mariano(e_model, e_rw, horizon)
    return ForecastEval(
        window=window,
        horizon=horizon,
        degree=degree,
        n_forecasts=n_fc,
        rmse_model=float(np.sqrt(np.mean(e_model**2))),
        rmse_random_walk=float(np.sqrt(np.mean(e_rw**2))),
        directional_accuracy=float(hits.mean()),
        dm_stat=dm,
        dm_pvalue=p,
    )
