"""Triangular solves: the final step of every direct least-squares method."""

from __future__ import annotations

import numpy as np
from numpy.typing import ArrayLike, NDArray


def _check_square(T: NDArray) -> int:
    if T.ndim != 2 or T.shape[0] != T.shape[1]:
        raise ValueError(f"expected a square matrix, got shape {T.shape}")
    return T.shape[0]


def back_substitution(R: ArrayLike, b: ArrayLike) -> NDArray[np.float64]:
    """Solve ``R x = b`` for upper-triangular ``R``.

    ``b`` may be a vector or a matrix of right-hand sides (one per column).
    Only the upper triangle of ``R`` is read.
    """
    R = np.asarray(R, dtype=float)
    n = _check_square(R)
    x = np.array(b, dtype=float, copy=True)
    if x.shape[0] != n:
        raise ValueError(f"R is {n}x{n} but b has {x.shape[0]} rows")
    for i in range(n - 1, -1, -1):
        if R[i, i] == 0.0:
            raise np.linalg.LinAlgError(f"singular triangular matrix (zero pivot at {i})")
        x[i] = (x[i] - R[i, i + 1 :] @ x[i + 1 :]) / R[i, i]
    return x


def forward_substitution(L: ArrayLike, b: ArrayLike) -> NDArray[np.float64]:
    """Solve ``L x = b`` for lower-triangular ``L``. Only the lower triangle is read."""
    L = np.asarray(L, dtype=float)
    n = _check_square(L)
    x = np.array(b, dtype=float, copy=True)
    if x.shape[0] != n:
        raise ValueError(f"L is {n}x{n} but b has {x.shape[0]} rows")
    for i in range(n):
        if L[i, i] == 0.0:
            raise np.linalg.LinAlgError(f"singular triangular matrix (zero pivot at {i})")
        x[i] = (x[i] - L[i, :i] @ x[:i]) / L[i, i]
    return x
