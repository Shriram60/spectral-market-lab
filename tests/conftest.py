import numpy as np
import pytest


@pytest.fixture
def rng():
    return np.random.default_rng(20261112)


def matrix_with_condition(rng, m, n, cond):
    """Random m x n matrix with prescribed 2-norm condition number."""
    U, _ = np.linalg.qr(rng.standard_normal((m, n)))
    V, _ = np.linalg.qr(rng.standard_normal((n, n)))
    s = np.geomspace(1.0, 1.0 / cond, n)
    return (U * s) @ V.T
