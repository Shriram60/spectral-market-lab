"""How much orthogonality each QR algorithm loses as the input gets ill-conditioned.

For matrices with prescribed condition number we measure ``||I - Q^T Q||_2``.
Theory predicts ``eps * cond^2`` for classical Gram-Schmidt, ``eps * cond`` for
modified Gram-Schmidt and ``O(eps)`` for Householder. Measurements use NumPy's norm so
the yardstick is independent of the code being measured.

Run: ``python -m spectral_market.experiments.orthogonality``
"""

from __future__ import annotations

import matplotlib.pyplot as plt
import numpy as np

from ..linalg import classical_gram_schmidt, householder_qr, modified_gram_schmidt
from ..plotting import INK_2, MUTED, SERIES, end_label, save, save_results, use_style

EPS = np.finfo(float).eps
METHODS = {
    "Classical Gram-Schmidt": lambda A: classical_gram_schmidt(A)[0],
    "Modified Gram-Schmidt": lambda A: modified_gram_schmidt(A)[0],
    "Householder": lambda A: householder_qr(A).q(),
}


def matrix_with_condition(rng: np.random.Generator, m: int, n: int, cond: float) -> np.ndarray:
    """``U diag(s) V^T`` with random orthonormal ``U, V`` and ``s`` from 1 down to 1/cond."""
    U = householder_qr(rng.standard_normal((m, n))).q()
    V = householder_qr(rng.standard_normal((n, n))).q()
    return (U * np.geomspace(1.0, 1.0 / cond, n)) @ V.T


def run(
    m: int = 100,
    n: int = 20,
    n_points: int = 27,
    max_log_cond: float = 13.0,
    trials: int = 5,
    seed: int = 0,
) -> tuple[np.ndarray, dict[str, np.ndarray]]:
    rng = np.random.default_rng(seed)
    conds = np.logspace(0, max_log_cond, n_points)
    loss = {name: np.empty(n_points) for name in METHODS}
    eye = np.eye(n)
    for i, cond in enumerate(conds):
        samples = {name: [] for name in METHODS}
        for _ in range(trials):
            A = matrix_with_condition(rng, m, n, cond)
            for name, method in METHODS.items():
                try:
                    Q = method(A)
                    samples[name].append(np.linalg.norm(eye - Q.T @ Q, 2))
                except np.linalg.LinAlgError:
                    samples[name].append(np.nan)
        for name in METHODS:
            loss[name][i] = np.nanmedian(samples[name])
    return conds, loss


def plot(conds: np.ndarray, loss: dict[str, np.ndarray]):
    fig, ax = plt.subplots(figsize=(7.6, 4.6))
    ref_x = conds[conds <= 1e9]
    for label, ref, x_lab in [
        (r"$\varepsilon\,\kappa^2$", EPS * ref_x**2, 1e8),
        (r"$\varepsilon\,\kappa$", EPS * conds, 3e12),
    ]:
        x = ref_x if ref.size == ref_x.size else conds
        ax.plot(x, ref, color=MUTED, lw=1.0, zorder=1)
        ax.annotate(
            label,
            (x_lab, np.interp(np.log10(x_lab), np.log10(x), ref)),
            xytext=(-4, 8),
            textcoords="offset points",
            color=MUTED,
            fontsize=10,
            ha="right",
        )
    for color, (name, y) in zip(SERIES, loss.items(), strict=False):
        ax.plot(conds, y, color=color, marker="o", ms=4.5, mec="white", mew=0.8, label=name)
        end_label(ax, conds[-1], y[-1], name.replace(" Gram-Schmidt", " G-S"))
    ax.set(
        xscale="log",
        yscale="log",
        ylim=(1e-17, 1e2),
        xlabel=r"condition number $\kappa(A)$",
        ylabel=r"loss of orthogonality  $\|I - Q^{\top}Q\|_2$",
    )
    ax.set_title("Same factorisation on paper, very different in floating point")
    ax.legend(loc="upper left")
    ax.text(
        0.99,
        -0.16,
        f"100x20 matrices, median of 5 draws per point; eps = {EPS:.1e}",
        transform=ax.transAxes,
        ha="right",
        fontsize=8,
        color=INK_2,
    )
    return fig


def main() -> None:
    use_style()
    conds, loss = run()
    path = save(plot(conds, loss), "orthogonality")
    probe = int(np.argmin(np.abs(np.log10(conds) - 8)))
    summary = {name: float(v[probe]) for name, v in loss.items()}
    save_results(
        "orthogonality",
        {
            "condition_numbers": conds.tolist(),
            "loss": {k: v.tolist() for k, v in loss.items()},
            "at_cond_1e8": summary,
        },
    )
    print(f"saved {path}")
    print(f"loss of orthogonality at cond = {conds[probe]:.0e}:")
    for name, v in summary.items():
        print(f"  {name:24s} {v:.2e}")


if __name__ == "__main__":
    main()
