"""Three QR factorisations with very different numerical behaviour.

* Classical Gram-Schmidt (CGS): the textbook algorithm. Loss of orthogonality grows
  like ``cond(A)**2 * eps``.
* Modified Gram-Schmidt (MGS): the same arithmetic reordered so that each projection
  is removed from the *updated* vectors. Loss of orthogonality grows like
  ``cond(A) * eps``.
* Householder reflections: orthogonal transformations applied to ``A``. Loss of
  orthogonality stays at ``O(eps)`` regardless of conditioning. This is what LAPACK
  (and therefore ``numpy.linalg.qr``) uses.

``experiments/orthogonality.py`` measures all three.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from numpy.typing import ArrayLike, NDArray


def _dependent(norm_left: float, norm_orig: float, m: int) -> bool:
    """True when what is left of a column after projection is only rounding noise."""
    return norm_left <= m * np.finfo(float).eps * norm_orig


def _as_tall(A: ArrayLike) -> NDArray[np.float64]:
    A = np.asarray(A, dtype=float)
    if A.ndim != 2:
        raise ValueError(f"expected a 2-D matrix, got shape {A.shape}")
    if A.shape[0] < A.shape[1]:
        raise ValueError(f"QR here requires m >= n, got shape {A.shape}")
    return A


def classical_gram_schmidt(A: ArrayLike) -> tuple[NDArray, NDArray]:
    """Reduced QR by classical Gram-Schmidt. Returns ``Q`` (m x n) and ``R`` (n x n)."""
    A = _as_tall(A)
    m, n = A.shape
    Q = np.zeros((m, n))
    R = np.zeros((n, n))
    for j in range(n):
        # Every projection coefficient is computed against the ORIGINAL column a_j.
        R[:j, j] = Q[:, :j].T @ A[:, j]
        v = A[:, j] - Q[:, :j] @ R[:j, j]
        R[j, j] = np.linalg.norm(v)
        if _dependent(R[j, j], np.linalg.norm(A[:, j]), m):
            raise np.linalg.LinAlgError(f"column {j} is linearly dependent on earlier columns")
        Q[:, j] = v / R[j, j]
    return Q, R


def modified_gram_schmidt(A: ArrayLike) -> tuple[NDArray, NDArray]:
    """Reduced QR by modified Gram-Schmidt. Returns ``Q`` (m x n) and ``R`` (n x n)."""
    A = _as_tall(A)
    V = A.copy()
    m, n = V.shape
    Q = np.zeros((m, n))
    R = np.zeros((n, n))
    for j in range(n):
        R[j, j] = np.linalg.norm(V[:, j])
        if _dependent(R[j, j], np.linalg.norm(A[:, j]), m):
            raise np.linalg.LinAlgError(f"column {j} is linearly dependent on earlier columns")
        Q[:, j] = V[:, j] / R[j, j]
        # Remove the q_j component from every REMAINING (already updated) column.
        R[j, j + 1 :] = Q[:, j] @ V[:, j + 1 :]
        V[:, j + 1 :] -= np.outer(Q[:, j], R[j, j + 1 :])
    return Q, R


@dataclass(frozen=True)
class HouseholderQR:
    """Compact Householder QR: ``A = Q R`` with ``Q = H_0 H_1 ... H_{k-1}``.

    ``Q`` is never formed unless asked for. Each reflector ``H_k = I - 2 v_k v_k^T`` is
    stored as the unit vector ``v_k`` in column ``k`` of ``reflectors`` (zero above row
    ``k``), which is enough to apply ``Q`` or ``Q^T`` in ``O(mn)`` operations.
    """

    reflectors: NDArray[np.float64]
    R: NDArray[np.float64]

    @property
    def shape(self) -> tuple[int, int]:
        return self.reflectors.shape

    def apply_qt(self, b: ArrayLike) -> NDArray[np.float64]:
        """Return ``Q^T b`` (length m) for a vector or matrix ``b``."""
        y = np.array(b, dtype=float, copy=True)
        for k in range(self.reflectors.shape[1]):
            v = self.reflectors[k:, k]
            y[k:] -= 2.0 * np.multiply.outer(v, v @ y[k:])
        return y

    def apply_q(self, y: ArrayLike) -> NDArray[np.float64]:
        """Return ``Q y`` (length m) for a vector or matrix ``y``."""
        b = np.array(y, dtype=float, copy=True)
        for k in range(self.reflectors.shape[1] - 1, -1, -1):
            v = self.reflectors[k:, k]
            b[k:] -= 2.0 * np.multiply.outer(v, v @ b[k:])
        return b

    def q(self, mode: str = "reduced") -> NDArray[np.float64]:
        """Form ``Q`` explicitly: ``m x n`` for ``"reduced"``, ``m x m`` for ``"complete"``."""
        m, n = self.shape
        cols = {"reduced": n, "complete": m}.get(mode)
        if cols is None:
            raise ValueError("mode must be 'reduced' or 'complete'")
        return self.apply_q(np.eye(m, cols))


def householder_qr(A: ArrayLike) -> HouseholderQR:
    """Householder QR of a tall matrix ``A`` (m >= n), returned in compact form."""
    W = _as_tall(A).copy()
    m, n = W.shape
    V = np.zeros((m, n))
    for k in range(min(m - 1, n)):
        x = W[k:, k]
        alpha = np.linalg.norm(x)
        if alpha == 0.0:
            continue  # column already zero below the diagonal; H_k = I
        v = x.copy()
        # Reflect x onto -sign(x_0) * ||x|| e_1 so that v_0 never suffers cancellation.
        v[0] += np.copysign(alpha, x[0])
        v /= np.linalg.norm(v)
        W[k:, k:] -= 2.0 * np.outer(v, v @ W[k:, k:])
        V[k:, k] = v
    return HouseholderQR(reflectors=V, R=np.triu(W[:n, :]))


def qr(A: ArrayLike, method: str = "householder") -> tuple[NDArray, NDArray]:
    """Reduced QR ``(Q, R)`` using ``"householder"``, ``"mgs"`` or ``"cgs"``."""
    if method == "householder":
        f = householder_qr(A)
        return f.q("reduced"), f.R
    if method == "mgs":
        return modified_gram_schmidt(A)
    if method == "cgs":
        return classical_gram_schmidt(A)
    raise ValueError(f"unknown QR method {method!r}")
