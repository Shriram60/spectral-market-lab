"""A trend that updates over time: the local linear trend model and the Kalman filter.

The Phase 1 trend line assumes one fixed level and slope for the whole sample. Here both
are hidden states that drift:

    y_t     = level_t + e_t,                     e_t ~ N(0, r)
    level_t = level_{t-1} + slope_{t-1} + u_t,   u_t ~ N(0, q_level)
    slope_t = slope_{t-1} + v_t,                 v_t ~ N(0, q_slope)

i.e. ``x_t = F x_{t-1} + w_t``, ``y_t = H x_t + e_t`` with ``F = [[1, 1], [0, 1]]`` and
``H = [1, 0]``. The Kalman filter is the exact Bayesian update of the state mean and
covariance one observation at a time. With ``q_level = q_slope = 0`` the state never
moves and the filter is recursive least squares: after the last observation it returns
exactly the OLS line of Phase 1. The noise variances are fitted by maximising the
prediction-error log-likelihood, and the Rauch-Tung-Striebel smoother gives the
two-sided estimate.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from numpy.typing import ArrayLike, NDArray
from scipy.optimize import minimize

F = np.array([[1.0, 1.0], [0.0, 1.0]])
H = np.array([1.0, 0.0])


@dataclass(frozen=True)
class KalmanResult:
    filtered_mean: NDArray[np.float64]  # (T, 2): E[x_t | y_1..t]
    filtered_cov: NDArray[np.float64]  # (T, 2, 2)
    predicted_mean: NDArray[np.float64]  # (T, 2): E[x_t | y_1..t-1]
    predicted_cov: NDArray[np.float64]
    innovations: NDArray[np.float64]  # y_t - H x_{t|t-1}
    innovation_var: NDArray[np.float64]
    loglik: float


def _prior(y: NDArray, diffuse: float) -> tuple[NDArray, NDArray]:
    return np.array([y[0], 0.0]), np.eye(2) * diffuse


def kalman_filter(
    y: ArrayLike,
    q_level: float,
    q_slope: float,
    r: float,
    diffuse: float = 1e7,
    burn_in: int = 2,
) -> KalmanResult:
    """Filter ``y`` through the local linear trend model.

    The prior is centred on the first observation with a large ``diffuse`` variance; the
    first ``burn_in`` prediction errors (dominated by that prior) are excluded from the
    log-likelihood. Covariances are updated in Joseph form, which stays symmetric
    positive semidefinite in floating point.
    """
    y = np.asarray(y, dtype=float)
    T = y.size
    Q = np.diag([q_level, q_slope])
    m, P = _prior(y, diffuse)
    fm, fP = np.empty((T, 2)), np.empty((T, 2, 2))
    pm, pP = np.empty((T, 2)), np.empty((T, 2, 2))
    v, S = np.empty(T), np.empty(T)
    for t in range(T):
        if t > 0:
            m = F @ m
            P = F @ P @ F.T + Q
        pm[t], pP[t] = m, P
        v[t] = y[t] - H @ m
        S[t] = H @ P @ H + r
        K = P @ H / S[t]
        m = m + K * v[t]
        IKH = np.eye(2) - np.outer(K, H)
        P = IKH @ P @ IKH.T + r * np.outer(K, K)
        fm[t], fP[t] = m, P
    ok = slice(burn_in, None)
    loglik = -0.5 * float(np.sum(np.log(2 * np.pi * S[ok]) + v[ok] ** 2 / S[ok]))
    return KalmanResult(fm, fP, pm, pP, v, S, loglik)


def _inv2(A: NDArray) -> NDArray:
    """Inverse of a 2 x 2 matrix by the adjugate formula."""
    det = A[0, 0] * A[1, 1] - A[0, 1] * A[1, 0]
    if det == 0:
        raise np.linalg.LinAlgError("singular 2 x 2 matrix")
    return np.array([[A[1, 1], -A[0, 1]], [-A[1, 0], A[0, 0]]]) / det


def rts_smoother(res: KalmanResult) -> tuple[NDArray[np.float64], NDArray[np.float64]]:
    """Rauch-Tung-Striebel smoother: E[x_t | y_1..T] and its covariance."""
    T = len(res.filtered_mean)
    sm, sP = res.filtered_mean.copy(), res.filtered_cov.copy()
    for t in range(T - 2, -1, -1):
        Pp = res.predicted_cov[t + 1]
        G = res.filtered_cov[t] @ F.T @ _inv2(Pp)
        sm[t] = res.filtered_mean[t] + G @ (sm[t + 1] - res.predicted_mean[t + 1])
        sP[t] = res.filtered_cov[t] + G @ (sP[t + 1] - Pp) @ G.T
    return sm, sP


def _loglik(y: NDArray, ql: float, qs: float, r: float, diffuse: float, burn_in: int) -> float:
    """Same recursion as :func:`kalman_filter`, unrolled into scalar arithmetic.

    The likelihood is evaluated thousands of times during fitting; for a 2-state model
    plain floats are about 30x faster than NumPy's per-call overhead on 2 x 2 arrays.
    """
    m0, m1 = float(y[0]), 0.0
    p00, p01, p11 = diffuse, 0.0, diffuse
    total = 0.0
    log2pi = np.log(2 * np.pi)
    for t, yt in enumerate(y.tolist()):
        if t > 0:
            m0 += m1
            p00, p01, p11 = p00 + 2 * p01 + p11 + ql, p01 + p11, p11 + qs
        v = yt - m0
        S = p00 + r
        k0, k1 = p00 / S, p01 / S
        m0 += k0 * v
        m1 += k1 * v
        p00, p01, p11 = p00 - k0 * k0 * S, p01 - k0 * k1 * S, p11 - k1 * k1 * S
        if t >= burn_in:
            total += log2pi + np.log(S) + v * v / S
    return -0.5 * total


@dataclass(frozen=True)
class TrendFit:
    q_level: float
    q_slope: float
    r: float
    loglik: float
    signal_to_noise: float  # q_slope / q_level: how fast the drift moves vs the level


def fit_local_trend(y: ArrayLike, snr: float | None = None, r_floor: float = 1e-12) -> TrendFit:
    """Maximum-likelihood noise variances, optimised on the log scale from several starts.

    With ``snr`` given, ``q_slope = snr * q_level`` is held fixed and only the level and
    observation variances are fitted: the profile likelihood used to test whether a
    time-varying slope is supported by the data.
    """
    y = np.asarray(y, dtype=float)
    scale = float(np.var(np.diff(y)))

    def unpack(theta):
        if snr is None:
            ql, qs, r = np.exp(theta)
        else:
            ql, r = np.exp(theta)
            qs = snr * ql
        return ql, qs, r

    def nll(theta):
        ql, qs, r = unpack(theta)
        return -_loglik(y, ql, qs, r + r_floor, 1e7, 2)

    starts = [[0.0, -6.0, -2.0], [-2.0, -10.0, 0.0], [0.0, -14.0, -8.0]]
    if snr is not None:
        starts = [[0.0, -2.0], [-2.0, 0.0], [0.0, -8.0]]
    best = None
    for start in starts:
        theta0 = np.log(scale) + np.array(start)
        res = minimize(
            nll,
            theta0,
            method="Nelder-Mead",
            options={"xatol": 1e-4, "fatol": 1e-6, "maxiter": 4000},
        )
        if best is None or res.fun < best.fun:
            best = res
    ql, qs, r = unpack(best.x)
    return TrendFit(float(ql), float(qs), float(r), float(-best.fun), float(qs / ql))


def trend_forecast(res: KalmanResult, horizon: int) -> NDArray[np.float64]:
    """h-step forecast of y made at each t from the filtered state: level + h * slope."""
    return res.filtered_mean[:, 0] + horizon * res.filtered_mean[:, 1]
