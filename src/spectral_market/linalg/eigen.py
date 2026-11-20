"""Symmetric eigendecomposition by the cyclic Jacobi method.

Each Jacobi rotation ``J(p, q)`` zeroes one off-diagonal pair of a symmetric matrix via
``A <- J^T A J``. Rotations on disjoint index pairs commute, so a whole round-robin
round (the same schedule used by the SVD) is applied at once as one orthogonal matrix.
The off-diagonal Frobenius norm falls quadratically once it is small, and the product
of all rotations is the eigenvector matrix.

Jacobi is slower than tridiagonal QR for large matrices but is simple, numerically
robust, and computes small eigenvalues to high relative accuracy.
"""

from __future__ import annotations

import numpy as np
from numpy.typing import ArrayLike, NDArray

from .svd import EPS, _round_robin


def jacobi_eigh(
    S: ArrayLike, tol: float | None = None, max_sweeps: int = 50
) -> tuple[NDArray[np.float64], NDArray[np.float64]]:
    """Eigenvalues (ascending) and orthonormal eigenvectors (columns) of symmetric ``S``."""
    A = np.array(S, dtype=float, copy=True)
    if A.ndim != 2 or A.shape[0] != A.shape[1]:
        raise ValueError(f"expected a square matrix, got shape {A.shape}")
    n = A.shape[0]
    scale = np.linalg.norm(A)
    if not np.allclose(A, A.T, atol=1e-12 * max(scale, 1.0)):
        raise ValueError("matrix is not symmetric")
    A = 0.5 * (A + A.T)
    V = np.eye(n)
    tol = n * EPS if tol is None else tol

    for _ in range(max_sweeps):
        # Measured directly: ||A||^2 - ||diag A||^2 would cancel catastrophically near
        # convergence and stop the iteration at ~sqrt(eps) instead of ~eps.
        off = np.linalg.norm(A - np.diag(np.diag(A)))
        if off <= tol * scale or n < 2:
            w = np.diag(A).copy()
            order = np.argsort(w)
            return w[order], V[:, order]
        for P, Q in _round_robin(n):
            app, aqq, apq = A[P, P], A[Q, Q], A[P, Q]
            active = apq != 0.0
            theta = (aqq - app) / (2.0 * np.where(active, apq, 1.0))
            t = np.where(theta >= 0, 1.0, -1.0) / (np.abs(theta) + np.hypot(1.0, theta))
            c = np.where(active, 1.0 / np.hypot(1.0, t), 1.0)
            s = np.where(active, c * t, 0.0)
            J = np.eye(n)
            J[P, P], J[Q, Q], J[P, Q], J[Q, P] = c, c, s, -s
            A = J.T @ A @ J
            V = V @ J
    raise np.linalg.LinAlgError(f"Jacobi eigensolver did not converge in {max_sweeps} sweeps")
