"""Singular value decomposition by one-sided Jacobi (Hestenes) rotations.

The idea: find an orthogonal ``V`` such that the columns of ``G = A V`` are mutually
orthogonal. Then the column norms of ``G`` are the singular values and the normalised
columns are the left singular vectors, giving ``A = U diag(s) V^T``.

Each Jacobi rotation orthogonalises one pair of columns. Two refinements make it fast
and accurate in NumPy:

* **QR preconditioning.** A tall ``A`` (m > n) is first reduced to its ``n x n``
  triangular factor ``R`` by Householder QR. Rotations then act on short columns, and
  ``U`` is recovered as ``Q U_R``.
* **Round-robin ordering.** Pairs are scheduled like a chess tournament, so each round
  is a set of disjoint pairs that can be rotated simultaneously as one vectorised
  operation (this is the "parallel Jacobi" ordering).

One-sided Jacobi is known for high *relative* accuracy in small singular values,
which matters later when separating signal from noise in covariance spectra.
"""

from __future__ import annotations

from dataclasses import dataclass
from functools import lru_cache

import numpy as np
from numpy.typing import ArrayLike, NDArray

from .qr import householder_qr

EPS = np.finfo(float).eps


@dataclass(frozen=True)
class SVD:
    """Thin SVD ``A = U @ diag(s) @ Vt`` with ``s`` in descending order."""

    U: NDArray[np.float64]
    s: NDArray[np.float64]
    Vt: NDArray[np.float64]
    sweeps: int

    def reconstruct(self) -> NDArray[np.float64]:
        return (self.U * self.s) @ self.Vt


@lru_cache(maxsize=64)
def _round_robin(n: int) -> tuple[tuple[NDArray, NDArray], ...]:
    """Disjoint (p, q) pair sets that together cover every pair p < q exactly once."""
    players = list(range(n)) + ([None] if n % 2 else [])
    size = len(players)
    rounds = []
    for _ in range(size - 1):
        pairs = [
            (min(a, b), max(a, b))
            for a, b in zip(players[: size // 2], reversed(players[size // 2 :]), strict=True)
            if a is not None and b is not None
        ]
        if pairs:
            P, Q = zip(*pairs, strict=True)
            rounds.append((np.array(P, dtype=int), np.array(Q, dtype=int)))
        players = [players[0], players[-1], *players[1:-1]]
    return tuple(rounds)


def _one_sided_jacobi(G: NDArray, tol: float, max_sweeps: int) -> tuple[NDArray, NDArray, int]:
    """Orthogonalise the columns of ``G`` in place. Returns ``(G, V, sweeps)``."""
    n = G.shape[1]
    V = np.eye(n)
    rounds = _round_robin(n)
    for sweep in range(1, max_sweeps + 1):
        rotated = False
        for P, Q in rounds:
            Gp, Gq = G[:, P], G[:, Q]
            alpha = np.einsum("ij,ij->j", Gp, Gp)
            beta = np.einsum("ij,ij->j", Gq, Gq)
            gamma = np.einsum("ij,ij->j", Gp, Gq)
            active = np.abs(gamma) > tol * np.sqrt(alpha * beta)
            if not active.any():
                continue
            rotated = True
            # Rotation angle that zeroes the (p, q) inner product; take the smaller root.
            zeta = (beta - alpha) / (2.0 * np.where(active, gamma, 1.0))
            t = np.where(zeta >= 0, 1.0, -1.0) / (np.abs(zeta) + np.hypot(1.0, zeta))
            c = np.where(active, 1.0 / np.hypot(1.0, t), 1.0)
            s = np.where(active, c * t, 0.0)
            G[:, P], G[:, Q] = c * Gp - s * Gq, s * Gp + c * Gq
            Vp, Vq = V[:, P], V[:, Q]
            V[:, P], V[:, Q] = c * Vp - s * Vq, s * Vp + c * Vq
        if not rotated:
            return G, V, sweep
    raise np.linalg.LinAlgError(f"Jacobi SVD did not converge in {max_sweeps} sweeps")


def _orthonormal_completion(U: NDArray, keep: NDArray) -> NDArray:
    """Replace columns of ``U`` not in ``keep`` with an orthonormal complement."""
    m, k = U.shape
    r = int(keep.sum())
    if r == k:
        return U
    out = np.empty_like(U)
    out[:, :r] = U[:, keep]
    if r == 0:
        out[:] = np.eye(m, k)
    else:
        # The trailing columns of a complete Q from QR of U_r span range(U_r)^perp.
        out[:, r:] = householder_qr(out[:, :r]).q("complete")[:, r:k]
    return out


def jacobi_svd(A: ArrayLike, tol: float | None = None, max_sweeps: int = 60) -> SVD:
    """Thin SVD of any real matrix by QR-preconditioned one-sided Jacobi."""
    A = np.asarray(A, dtype=float)
    if A.ndim != 2:
        raise ValueError(f"expected a 2-D matrix, got shape {A.shape}")
    m, n = A.shape
    if m < n:
        t = jacobi_svd(A.T, tol=tol, max_sweeps=max_sweeps)
        return SVD(U=t.Vt.T, s=t.s, Vt=t.U.T, sweeps=t.sweeps)
    if n == 0:
        return SVD(U=np.zeros((m, 0)), s=np.zeros(0), Vt=np.zeros((0, 0)), sweeps=0)

    factor = householder_qr(A) if m > n else None
    G = factor.R.copy() if factor is not None else A.copy()
    G, V, sweeps = _one_sided_jacobi(G, tol if tol is not None else G.shape[0] * EPS, max_sweeps)

    s = np.linalg.norm(G, axis=0)
    order = np.argsort(s)[::-1]
    s, G, V = s[order], G[:, order], V[:, order]
    nonzero = s > (s[0] * max(m, n) * EPS if s[0] > 0 else 0.0)
    U = np.zeros_like(G)
    U[:, nonzero] = G[:, nonzero] / s[nonzero]
    U = _orthonormal_completion(U, nonzero)

    if factor is not None:
        U = factor.apply_q(np.vstack([U, np.zeros((m - n, n))]))
    return SVD(U=U, s=s, Vt=V.T, sweeps=sweeps)


def singular_values(A: ArrayLike) -> NDArray[np.float64]:
    return jacobi_svd(A).s


def condition_number(A: ArrayLike) -> float:
    """2-norm condition number ``s_max / s_min`` (``inf`` if rank deficient)."""
    s = singular_values(A)
    return float(s[0] / s[-1]) if s[-1] > 0 else float("inf")


def numerical_rank(A: ArrayLike, rcond: float | None = None) -> int:
    s = singular_values(A)
    if s.size == 0 or s[0] == 0:
        return 0
    rcond = max(np.shape(A)) * EPS if rcond is None else rcond
    return int((s > rcond * s[0]).sum())
