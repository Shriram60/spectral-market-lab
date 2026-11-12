"""Three ways to solve ``min ||A x - b||_2`` and why they are not equivalent.

=============  ==============================  ==================================
method         what it solves                   forward error (rough)
=============  ==============================  ==================================
``normal``     ``A^T A x = A^T b`` (Cholesky)   ``cond(A)**2 * eps``
``qr``         ``R x = Q^T b`` (Householder)    ``cond(A) * eps`` (+ residual term)
``svd``        ``x = V diag(1/s) U^T b``        ``cond(A) * eps``; handles rank loss
=============  ==============================  ==================================

All three give the orthogonal projection of ``b`` onto the column space of ``A`` in
exact arithmetic. In floating point the normal equations square the condition number
before solving, which is why they break first.
"""

from __future__ import annotations

import numpy as np
from numpy.typing import ArrayLike, NDArray

from .cholesky import cholesky
from .qr import householder_qr
from .svd import EPS, jacobi_svd
from .triangular import back_substitution, forward_substitution


def lstsq_normal(A: ArrayLike, b: ArrayLike) -> NDArray[np.float64]:
    """Least squares through the normal equations ``A^T A x = A^T b``."""
    A = np.asarray(A, dtype=float)
    L = cholesky(A.T @ A)
    return back_substitution(L.T, forward_substitution(L, A.T @ np.asarray(b, dtype=float)))


def lstsq_qr(A: ArrayLike, b: ArrayLike) -> NDArray[np.float64]:
    """Least squares through Householder QR, applying ``Q^T`` implicitly.

    Requires full column rank. Use :func:`lstsq_svd` when columns may be dependent.
    """
    f = householder_qr(A)
    n = f.R.shape[0]
    return back_substitution(f.R, f.apply_qt(b)[:n])


def lstsq_svd(A: ArrayLike, b: ArrayLike, rcond: float | None = None) -> NDArray[np.float64]:
    """Minimum-norm least squares through the pseudo-inverse ``V diag(1/s) U^T``.

    Singular values below ``rcond * s_max`` are treated as zero, so this also returns
    the minimum-norm solution when ``A`` is rank deficient.
    """
    A = np.asarray(A, dtype=float)
    svd = jacobi_svd(A)
    rcond = max(A.shape) * EPS if rcond is None else rcond
    keep = svd.s > (rcond * svd.s[0] if svd.s.size else 0.0)
    coeffs = svd.U[:, keep].T @ np.asarray(b, dtype=float)
    inv_s = 1.0 / svd.s[keep]
    scaled = inv_s[:, None] * coeffs if coeffs.ndim == 2 else inv_s * coeffs
    return svd.Vt[keep].T @ scaled


SOLVERS = {"normal": lstsq_normal, "qr": lstsq_qr, "svd": lstsq_svd}


def lstsq(A: ArrayLike, b: ArrayLike, method: str = "qr") -> NDArray[np.float64]:
    """Dispatch to ``"normal"``, ``"qr"`` or ``"svd"``."""
    solver = SOLVERS.get(method)
    if solver is None:
        raise ValueError(f"unknown method {method!r}; choose from {sorted(SOLVERS)}")
    return solver(A, b)
