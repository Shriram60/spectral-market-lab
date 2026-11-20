"""Statistical arbitrage on PCA residuals (Avellaneda & Lee, 2010).

Each stock's return is split into a part explained by the market's eigenportfolios and a
residual: ``r_i = alpha_i + sum_j beta_ij F_j + eps_i``. The cumulative residual
``X_i(t)`` is modelled as an Ornstein-Uhlenbeck process

    dX = kappa (m - X) dt + sigma dW,

estimated from a discrete AR(1) fit ``X_{k+1} = a + b X_k + zeta``:
``kappa = -ln(b) * 252``, ``m = a / (1 - b)``, ``sigma_eq = sqrt(var(zeta) / (1 - b^2))``.
The s-score ``s = (X - m) / sigma_eq`` says how many equilibrium standard deviations the
residual is from its mean. Positions are opened when ``|s|`` is large and closed when it
reverts, and every stock position is hedged with its factor exposure, so the P&L is the
residual return.

Parameters default to the paper's values (60-day regression window, thresholds 1.25 /
0.75 / 0.5, mean-reversion time under 30 trading days), so nothing is fitted to this
dataset.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np
from numpy.typing import ArrayLike, NDArray

from .linalg import jacobi_eigh, lstsq_qr
from .spectral import marchenko_pastur_bounds, market_adjusted_variance

TRADING_DAYS = 252


@dataclass(frozen=True)
class OUFit:
    kappa: NDArray[np.float64]  # annualised mean-reversion speed
    m: NDArray[np.float64]
    sigma_eq: NDArray[np.float64]
    s_score: NDArray[np.float64]
    valid: NDArray[np.bool_]  # 0 < b < 1, so the AR(1) is mean-reverting


def fit_ou(X: ArrayLike) -> OUFit:
    """AR(1) fit of each column of ``X`` (cumulative residuals) and its s-score."""
    X = np.asarray(X, dtype=float)
    x, y = X[:-1], X[1:]
    xm, ym = x.mean(axis=0), y.mean(axis=0)
    sxx = ((x - xm) ** 2).sum(axis=0)
    b = np.where(sxx > 0, ((x - xm) * (y - ym)).sum(axis=0) / np.where(sxx > 0, sxx, 1), 0.0)
    a = ym - b * xm
    zeta = y - (a + b * x)
    valid = (b > 0) & (b < 1)
    bs = np.where(valid, b, 0.5)
    kappa = np.where(valid, -np.log(bs) * TRADING_DAYS, 0.0)
    m = np.where(valid, a / (1 - bs), 0.0)
    var_eq = zeta.var(axis=0, ddof=1) / (1 - bs**2)
    sigma_eq = np.sqrt(np.where(valid, var_eq, np.inf))
    s = np.where(valid, (X[-1] - m) / sigma_eq, 0.0)
    return OUFit(kappa, m, sigma_eq, s, valid)


@dataclass(frozen=True)
class StatArbParams:
    pca_window: int = 252
    refit_every: int = 21
    regression_window: int = 60
    s_open: float = 1.25
    s_close_short: float = 0.75
    s_close_long: float = 0.5
    max_reversion_days: float = 30.0
    n_factors: int | None = None  # None: eigenvalues above the M-P edge, per window


@dataclass(frozen=True)
class StatArbResult:
    pnl: NDArray[np.float64]  # daily gross return per unit of capital in use
    positions: NDArray[np.int8]  # (days, stocks): +1 long, -1 short, 0 flat
    trades: NDArray[np.float64]  # (days,) units traded (stock + hedge legs)
    s_scores: NDArray[np.float64]  # (days, stocks); NaN where no valid OU fit
    tradeable: NDArray[np.bool_]  # (days, stocks): mean reversion fast enough to trade
    n_factors: list[int] = field(default_factory=list)

    def net(self, cost_bps: float) -> NDArray[np.float64]:
        """Daily return after a one-way cost of ``cost_bps`` per unit traded."""
        gross = np.abs(self.positions).sum(axis=1)
        capital = np.where(gross > 0, gross, 1.0)
        return self.pnl - cost_bps * 1e-4 * self.trades / capital


def _eigenportfolios(window: NDArray, k: int | None) -> tuple[NDArray, int]:
    T, N = window.shape
    sd = window.std(axis=0, ddof=1)
    Z = (window - window.mean(axis=0)) / sd
    lam, V = jacobi_eigh(Z.T @ Z / (T - 1))
    lam, V = lam[::-1], V[:, ::-1]
    if k is None:
        edge = marchenko_pastur_bounds(N / T, market_adjusted_variance(lam))[1]
        k = max(int((lam > edge).sum()), 1)
    return V[:, :k] / sd[:, None], k  # columns: capital weights of each eigenportfolio


def run_statarb(R: ArrayLike, params: StatArbParams | None = None) -> StatArbResult:
    """Daily signals at the close of day t, held over day t + 1. No look-ahead."""
    p = params or StatArbParams()
    R = np.asarray(R, dtype=float)
    simple = np.expm1(R)
    T, N = R.shape
    start = p.pca_window
    pos = np.zeros(N, dtype=np.int8)
    pnl, positions, trades, n_factors, scores, ok = [], [], [], [], [], []
    Q, hedge = None, np.zeros(N)  # hedge: stock-space holdings of the factor legs
    for t in range(start, T - 1):
        if (t - start) % p.refit_every == 0:
            Q, k = _eigenportfolios(R[t - p.pca_window : t], p.n_factors)
            n_factors.append(k)
        win = R[t - p.regression_window + 1 : t + 1]
        F = win @ Q
        A = np.column_stack([np.ones(len(win)), F])
        coef = lstsq_qr(A, win)  # (k + 1, N): intercept, betas
        eps = win - A @ coef
        ou = fit_ou(np.cumsum(eps, axis=0))
        tradeable = ou.valid & (ou.kappa > TRADING_DAYS / p.max_reversion_days)
        s = ou.s_score
        new = pos.copy()
        new[(pos == 0) & tradeable & (s < -p.s_open)] = 1
        new[(pos == 0) & tradeable & (s > p.s_open)] = -1
        new[(pos == 1) & (s > -p.s_close_long)] = 0
        new[(pos == -1) & (s < p.s_close_short)] = 0
        # Hedge legs: each stock position is offset by beta units of each eigenportfolio,
        # expressed in stock space so it stays comparable when the factor basis is refit.
        beta = coef[1:]
        hedge_new = -Q @ (beta @ new)
        traded = np.abs(new - pos).sum() + np.abs(hedge_new - hedge).sum()
        pos, hedge = new, hedge_new
        # Next-day P&L of hedged positions: stock return minus factor-implied return.
        r_next = simple[t + 1]
        resid = r_next - (r_next @ Q) @ beta
        gross = np.abs(pos).sum()
        pnl.append(float(pos @ resid / gross) if gross else 0.0)
        positions.append(pos.copy())
        trades.append(traded)
        scores.append(np.where(ou.valid, s, np.nan))
        ok.append(tradeable)
    return StatArbResult(
        pnl=np.array(pnl),
        positions=np.array(positions, dtype=np.int8),
        trades=np.array(trades),
        s_scores=np.array(scores),
        tradeable=np.array(ok),
        n_factors=n_factors,
    )


def performance(daily: ArrayLike) -> dict:
    """Annualised return, volatility, Sharpe ratio and maximum drawdown of daily returns."""
    r = np.asarray(daily, dtype=float)
    wealth = np.cumprod(1 + r)
    drawdown = 1 - wealth / np.maximum.accumulate(wealth)
    vol = r.std(ddof=1) * np.sqrt(TRADING_DAYS)
    ann = r.mean() * TRADING_DAYS
    return {
        "annual_return": float(ann),
        "annual_vol": float(vol),
        "sharpe": float(ann / vol) if vol > 0 else float("nan"),
        "max_drawdown": float(drawdown.max()),
    }
