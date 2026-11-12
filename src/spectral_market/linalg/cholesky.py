"""Cholesky factorisation, used to solve the normal equations."""

from __future__ import annotations

import numpy as np
from numpy.typing import ArrayLike, NDArray


def cholesky(M: ArrayLike) -> NDArray[np.float64]:
    """Return lower-triangular ``L`` with ``M = L @ L.T``.

    Only the lower triangle of ``M`` is read. Raises ``LinAlgError`` when a pivot is
    not strictly positive, which is how the normal equations fail in floating point:
    once ``cond(A.T @ A) = cond(A)**2`` passes ``1 / eps``, ``A.T @ A`` is no longer
    numerically positive definite even though it is in exact arithmetic.
    """
    M = np.asarray(M, dtype=float)
    if M.ndim != 2 or M.shape[0] != M.shape[1]:
        raise ValueError(f"expected a square matrix, got shape {M.shape}")
    n = M.shape[0]
    L = np.zeros_like(M)
    for j in range(n):
        pivot = M[j, j] - L[j, :j] @ L[j, :j]
        if not pivot > 0.0:
            raise np.linalg.LinAlgError(
                f"matrix is not numerically positive definite (pivot {j} = {pivot:.3e})"
            )
        L[j, j] = np.sqrt(pivot)
        L[j + 1 :, j] = (M[j + 1 :, j] - L[j + 1 :, :j] @ L[j, :j]) / L[j, j]
    return L
