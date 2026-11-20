"""From-scratch numerical linear algebra (NumPy is used only for array arithmetic)."""

from .cholesky import cholesky
from .eigen import jacobi_eigh
from .lstsq import lstsq, lstsq_normal, lstsq_qr, lstsq_svd
from .qr import (
    HouseholderQR,
    classical_gram_schmidt,
    householder_qr,
    modified_gram_schmidt,
    qr,
)
from .regularized import (
    LassoFit,
    RidgePath,
    lasso_cd,
    lasso_kkt_violation,
    lasso_lambda_max,
    lasso_path,
    ridge,
    ridge_path,
)
from .svd import SVD, condition_number, jacobi_svd, numerical_rank, singular_values
from .triangular import back_substitution, forward_substitution

__all__ = [
    "SVD",
    "HouseholderQR",
    "LassoFit",
    "RidgePath",
    "back_substitution",
    "cholesky",
    "classical_gram_schmidt",
    "condition_number",
    "forward_substitution",
    "householder_qr",
    "jacobi_eigh",
    "jacobi_svd",
    "lasso_cd",
    "lasso_kkt_violation",
    "lasso_lambda_max",
    "lasso_path",
    "lstsq",
    "lstsq_normal",
    "lstsq_qr",
    "lstsq_svd",
    "modified_gram_schmidt",
    "numerical_rank",
    "qr",
    "ridge",
    "ridge_path",
    "singular_values",
]
