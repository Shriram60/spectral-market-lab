"""Gaussian hidden Markov model for market regimes.

Daily returns are drawn from one of K Gaussian regimes, and the regime follows a Markov
chain with transition matrix A (``A[i, j] = P(s_t = j | s_{t-1} = i)``). Fitted by
Baum-Welch (EM):

* **E-step**: the forward-backward recursions give ``gamma_t(i) = P(s_t = i | x)`` and
  the expected transition counts. Each forward step is a vector-matrix product
  ``alpha_t = (alpha_{t-1} A) * b_t`` and is rescaled to sum to one, so the
  log-likelihood is ``sum_t log c_t`` and nothing underflows.
* **M-step**: weighted means, variances and normalised transition counts.

Viterbi gives the single most likely path. For forecasting only the *filtered*
probabilities ``P(s_t | x_1..t)`` are used, so nothing looks ahead.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from numpy.typing import ArrayLike, NDArray

from .linalg import lstsq_qr

LOG_2PI = np.log(2 * np.pi)


@dataclass(frozen=True)
class HMM:
    pi: NDArray[np.float64]  # initial distribution (K,)
    A: NDArray[np.float64]  # transition matrix (K, K), rows sum to one
    mu: NDArray[np.float64]  # (K,)
    sigma: NDArray[np.float64]  # (K,), states sorted by ascending sigma
    loglik: float = float("nan")
    n_iter: int = 0

    @property
    def k(self) -> int:
        return self.mu.size

    @property
    def n_params(self) -> int:
        K = self.k
        return (K - 1) + K * (K - 1) + 2 * K

    def bic(self, n_obs: int) -> float:
        return -2 * self.loglik + self.n_params * np.log(n_obs)

    @property
    def expected_duration(self) -> NDArray[np.float64]:
        """Mean days spent in each regime per visit: ``1 / (1 - A_ii)``."""
        stay = np.diag(self.A)
        return np.divide(1.0, 1.0 - stay, out=np.full(stay.shape, np.inf), where=stay < 1)

    @property
    def stationary(self) -> NDArray[np.float64]:
        """Long-run regime frequencies: the left eigenvector of A for eigenvalue 1."""
        K = self.k
        M = np.vstack([self.A.T - np.eye(K), np.ones(K)])
        b = np.r_[np.zeros(K), 1.0]
        return lstsq_qr(M, b)


def log_emissions(x: ArrayLike, mu: ArrayLike, sigma: ArrayLike) -> NDArray[np.float64]:
    x = np.asarray(x, dtype=float)[:, None]
    mu, sigma = np.asarray(mu, float)[None], np.asarray(sigma, float)[None]
    return -0.5 * (LOG_2PI + 2 * np.log(sigma) + ((x - mu) / sigma) ** 2)


def forward(model: HMM, x: ArrayLike) -> tuple[NDArray[np.float64], NDArray[np.float64], float]:
    """Scaled forward pass: filtered probabilities ``P(s_t | x_1..t)``, scales, loglik."""
    logb = log_emissions(x, model.mu, model.sigma)
    shift = logb.max(axis=1, keepdims=True)
    b = np.exp(logb - shift)
    T, K = b.shape
    alpha, c = np.empty((T, K)), np.empty(T)
    a = model.pi * b[0]
    for t in range(T):
        if t > 0:
            a = (alpha[t - 1] @ model.A) * b[t]
        c[t] = a.sum()
        alpha[t] = a / c[t]
    return alpha, c, float(np.log(c).sum() + shift.sum())


def forward_backward(model: HMM, x: ArrayLike) -> tuple[NDArray, NDArray, float]:
    """Posterior state probabilities ``gamma`` (T, K), expected transitions (K, K), loglik."""
    logb = log_emissions(x, model.mu, model.sigma)
    b = np.exp(logb - logb.max(axis=1, keepdims=True))
    alpha, c, loglik = forward(model, x)
    T, K = b.shape
    beta = np.ones((T, K))
    xi = np.zeros((K, K))
    for t in range(T - 2, -1, -1):
        bb = b[t + 1] * beta[t + 1]
        beta[t] = (model.A @ bb) / c[t + 1]
        xi += model.A * np.outer(alpha[t], bb) / c[t + 1]
    gamma = alpha * beta
    gamma /= gamma.sum(axis=1, keepdims=True)
    return gamma, xi, loglik


def _sorted(model: HMM) -> HMM:
    o = np.argsort(model.sigma)
    return HMM(
        model.pi[o], model.A[np.ix_(o, o)], model.mu[o], model.sigma[o], model.loglik, model.n_iter
    )


def baum_welch(
    x: ArrayLike,
    k: int,
    n_init: int = 10,
    max_iter: int = 500,
    tol: float = 1e-7,
    seed: int = 0,
    min_sigma: float = 1e-6,
) -> HMM:
    """EM from ``n_init`` random starts; returns the highest-likelihood fit."""
    x = np.asarray(x, dtype=float)
    rng = np.random.default_rng(seed)
    sd = x.std()
    best: HMM | None = None
    for _ in range(n_init):
        A = rng.dirichlet(np.ones(k), size=k) * 0.2 + np.eye(k) * 0.8
        model = HMM(
            pi=np.full(k, 1.0 / k),
            A=A / A.sum(axis=1, keepdims=True),
            mu=rng.choice(x, size=k),
            sigma=sd * rng.uniform(0.5, 2.0, size=k),
        )
        prev = -np.inf
        for it in range(1, max_iter + 1):
            gamma, xi, loglik = forward_backward(model, x)
            w = gamma.sum(axis=0)
            mu = gamma.T @ x / w
            var = (gamma * (x[:, None] - mu) ** 2).sum(axis=0) / w
            A = xi / xi.sum(axis=1, keepdims=True)
            model = HMM(gamma[0], A, mu, np.sqrt(np.maximum(var, min_sigma**2)), loglik, it)
            if loglik - prev < tol * abs(loglik):
                break
            prev = loglik
        _, _, loglik = forward(model, x)
        model = HMM(model.pi, model.A, model.mu, model.sigma, loglik, model.n_iter)
        if best is None or loglik > best.loglik:
            best = model
    return _sorted(best)


def viterbi(model: HMM, x: ArrayLike) -> NDArray[np.int_]:
    """Most likely state path, by dynamic programming in log space."""
    logb = log_emissions(x, model.mu, model.sigma)
    logA = np.log(model.A)
    T, K = logb.shape
    delta = np.log(model.pi + 1e-300) + logb[0]
    back = np.empty((T, K), dtype=int)
    for t in range(1, T):
        scores = delta[:, None] + logA
        back[t] = scores.argmax(axis=0)
        delta = scores.max(axis=0) + logb[t]
    path = np.empty(T, dtype=int)
    path[-1] = int(delta.argmax())
    for t in range(T - 1, 0, -1):
        path[t - 1] = back[t, path[t]]
    return path


def predicted_variance(model: HMM, filtered: ArrayLike) -> NDArray[np.float64]:
    """Variance of x_{t+1} given x_1..t: a mixture over the one-step-ahead regime."""
    p = np.asarray(filtered, dtype=float) @ model.A
    m = p @ model.mu
    return p @ (model.sigma**2 + model.mu**2) - m**2
