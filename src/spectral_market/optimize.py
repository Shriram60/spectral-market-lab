"""Quadratic programs over the probability simplex.

``min 0.5 x^T H x + g^T x  subject to  x >= 0, sum(x) = 1``

This is the shape of every long-only, fully invested portfolio problem in the project
(index tracking, minimum variance). It is solved exactly by a primal active-set method
(Nocedal & Wright, Algorithm 16.3). Each iteration fixes some weights at zero (the working
set) and solves the equality-constrained problem on the rest, which is the linear KKT
system

    [ H_FF  1 ] [ x_F ]   [ -g_F ]
    [ 1^T   0 ] [ nu  ] = [   1  ]

solved with Householder QR. If the solution leaves the simplex, the method steps to the
boundary and adds the blocking weight to the working set. If a fixed weight has a
negative multiplier, it is released. Termination is finite and the answer is exact to
rounding error; optimality is certified by the KKT conditions.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from numpy.typing import ArrayLike, NDArray

from .linalg import cholesky, lstsq_qr, lstsq_svd


def check_convex(H: ArrayLike, rtol: float = 1e-10) -> None:
    """Raise ``ValueError`` unless ``H`` is symmetric positive semidefinite.

    The active-set method is only exact for a convex objective. PSD is tested by a
    Cholesky factorisation of ``H + rtol * ||H|| * I``: the shift admits singular PSD
    matrices (e.g. a sample covariance with fewer days than stocks) while a genuinely
    negative eigenvalue still produces a non-positive pivot.
    """
    H = np.asarray(H, dtype=float)
    if H.ndim != 2 or H.shape[0] != H.shape[1]:
        raise ValueError(f"expected a square matrix, got shape {H.shape}")
    scale = max(np.abs(H).max(), np.finfo(float).tiny)
    if not np.allclose(H, H.T, rtol=0, atol=1e-12 * scale):
        raise ValueError("H is not symmetric")
    try:
        cholesky(H + rtol * scale * np.eye(len(H)))
    except np.linalg.LinAlgError as err:
        raise ValueError("H is not positive semidefinite; the QP is not convex") from err


def simplex_kkt_violation(H: ArrayLike, g: ArrayLike, x: ArrayLike) -> float:
    """KKT residual: on the support every gradient entry equals a common ``mu``;
    off the support entries must be ``>= mu``. Also reports infeasibility."""
    x = np.asarray(x, dtype=float)
    grad = np.asarray(H) @ x + np.asarray(g)
    support = x > 0
    mu = grad[support].mean()
    on = np.abs(grad[support] - mu).max(initial=0.0)
    off = np.maximum(mu - grad[~support], 0.0).max(initial=0.0)
    infeasible = max(abs(x.sum() - 1.0), max(-x.min(), 0.0))
    return float(max(on, off, infeasible))


@dataclass(frozen=True)
class SimplexQPResult:
    x: NDArray[np.float64]
    n_iter: int
    kkt_violation: float


def _equality_qp(H: NDArray, g: NDArray, free: NDArray) -> tuple[NDArray, float]:
    """Minimise over the free coordinates subject to their sum being 1."""
    k = free.size
    K = np.zeros((k + 1, k + 1))
    K[:k, :k] = H[np.ix_(free, free)]
    K[:k, k] = K[k, :k] = 1.0
    rhs = np.concatenate([-g[free], [1.0]])
    try:
        z = lstsq_qr(K, rhs)
        if not np.all(np.isfinite(z)):
            raise np.linalg.LinAlgError
    except np.linalg.LinAlgError:
        z = lstsq_svd(K, rhs)  # singular H on the free set: minimum-norm KKT solution
    return z[:k], -z[k]  # x_F and the budget multiplier mu (grad_F = mu on the support)


def simplex_qp(
    H: ArrayLike,
    g: ArrayLike | None = None,
    tol: float = 1e-12,
    max_iter: int = 1000,
    check: bool = True,
) -> SimplexQPResult:
    """Exact solution of the simplex-constrained convex QP by a primal active-set method.

    ``check=False`` skips the convexity test for callers that construct ``H`` as a
    Gram matrix and call the solver in a tight loop.
    """
    H = np.asarray(H, dtype=float)
    if check:
        check_convex(H)
    H = 0.5 * (H + H.T)
    n = H.shape[0]
    g = np.zeros(n) if g is None else np.asarray(g, dtype=float)
    scale = max(np.abs(H).max(), np.abs(g).max(), np.finfo(float).tiny)

    x = np.full(n, 1.0 / n)  # feasible start, every weight free
    fixed = np.zeros(n, dtype=bool)
    for it in range(1, max_iter + 1):
        free = np.flatnonzero(~fixed)
        target, mu = _equality_qp(H, g, free)
        step = target - x[free]
        if np.all(target >= -tol):
            x[free] = np.maximum(target, 0.0)
            x /= x.sum()
            grad = H @ x + g
            multipliers = grad[fixed] - mu
            if multipliers.size == 0 or multipliers.min() >= -tol * scale:
                return SimplexQPResult(x, it, simplex_kkt_violation(H, g, x))
            release = np.flatnonzero(fixed)[np.argmin(multipliers)]
            fixed[release] = False
            continue
        # Move toward the target until the first free weight hits zero.
        shrinking = step < 0
        ratios = np.full(free.size, np.inf)
        ratios[shrinking] = x[free][shrinking] / -step[shrinking]
        block = int(np.argmin(ratios))
        alpha = min(1.0, ratios[block])
        x[free] += alpha * step
        x[free[block]] = 0.0
        x = np.maximum(x, 0.0)
        x /= x.sum()
        fixed[free[block]] = True
    raise RuntimeError(f"active-set QP did not terminate in {max_iter} iterations")
