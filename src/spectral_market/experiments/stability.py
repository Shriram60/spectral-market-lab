"""Normal equations vs QR vs SVD on polynomial trend fits of increasing degree.

A consistent system ``b = A x_true`` is built for one trading year (t = 1..250) so the
exact answer is known, and each solver's relative error in ``x`` is measured. This is
done twice: with raw day numbers, and with time rescaled to [-1, 1]. A third basis,
Legendre polynomials, is included in the conditioning panel.

Two lessons:
1. ``cond(A^T A) = cond(A)^2``, so the normal equations lose digits twice as fast as QR
   or SVD, and Cholesky eventually fails outright.
2. The basis matters more than the solver. Rescaling time changes the conditioning by
   many orders of magnitude, and an orthogonal-polynomial basis keeps it near 1.

Run: ``python -m spectral_market.experiments.stability``
"""

from __future__ import annotations

import matplotlib.pyplot as plt
import numpy as np

from ..linalg import lstsq
from ..plotting import GRID, INK_2, MUTED, SERIES, end_label, save, save_results, use_style
from ..trend import design_matrix, standardisation

EPS = np.finfo(float).eps
SOLVERS = {"normal": "Normal equations (Cholesky)", "qr": "Householder QR", "svd": "Jacobi SVD"}
SOLVER_COLORS = {"qr": SERIES[0], "normal": SERIES[1], "svd": SERIES[2]}
FAIL_Y = 1e19  # height at which Cholesky breakdowns are marked


def bases(t: np.ndarray) -> dict[str, callable]:
    center, scale = standardisation(t)
    z = (t - center) / scale
    return {
        "raw": lambda d: design_matrix(t, d),
        "scaled": lambda d: design_matrix(t, d, center, scale),
        "legendre": lambda d: np.polynomial.legendre.legvander(z, d),
    }


def run(n_days: int = 250, max_degree: int = 14, seed: int = 0) -> dict:
    rng = np.random.default_rng(seed)
    t = np.arange(1, n_days + 1, dtype=float)
    degrees = np.arange(1, max_degree + 1)
    builders = bases(t)
    out = {"degrees": degrees.tolist(), "cond": {}, "error": {}, "failed": {}}
    for name, build in builders.items():
        out["cond"][name] = [float(np.linalg.cond(build(d))) for d in degrees]
    for basis in ("raw", "scaled"):
        out["error"][basis] = {s: [] for s in SOLVERS}
        out["failed"][basis] = {s: [] for s in SOLVERS}
        for d in degrees:
            A = builders[basis](d)
            x_true = rng.standard_normal(d + 1)
            b = A @ x_true
            for solver in SOLVERS:
                try:
                    x = lstsq(A, b, solver)
                    err = float(np.linalg.norm(x - x_true) / np.linalg.norm(x_true))
                    failed = not np.isfinite(err)
                except np.linalg.LinAlgError:
                    err, failed = float("nan"), True
                out["error"][basis][solver].append(err)
                out["failed"][basis][solver].append(failed)
    return out


def _error_panel(ax, degrees, errors, failed, title):
    for solver, label in SOLVERS.items():
        err = np.array(errors[solver], dtype=float)
        bad = np.array(failed[solver])
        shown = np.clip(np.where(bad, np.nan, err), 1e-17, None)
        ax.plot(
            degrees,
            shown,
            color=SOLVER_COLORS[solver],
            marker="o",
            ms=4.5,
            mec="white",
            mew=0.8,
            label=label,
        )
        if bad.any():
            ax.plot(
                degrees[bad],
                np.full(bad.sum(), FAIL_Y),
                ls="none",
                marker="X",
                ms=8,
                color=SOLVER_COLORS[solver],
                mec="white",
                mew=0.8,
            )
    ax.axhline(1.0, color=MUTED, lw=1.0, zorder=1)
    ax.annotate(
        "100% error",
        (degrees[0], 1.0),
        xytext=(2, -10),
        textcoords="offset points",
        fontsize=8,
        color=MUTED,
    )
    ax.set(
        yscale="log",
        ylim=(1e-17, 1e21),
        xlabel="polynomial degree",
        xticks=degrees[1::2],
        yticks=10.0 ** np.arange(-16, 21, 4),
    )
    ax.set_title(title)


def plot(res: dict):
    degrees = np.array(res["degrees"])
    fig, axes = plt.subplots(1, 3, figsize=(15.5, 4.6), gridspec_kw={"wspace": 0.42})
    _error_panel(
        axes[0],
        degrees,
        res["error"]["raw"],
        res["failed"]["raw"],
        "(a) Raw day numbers t = 1...250",
    )
    axes[0].set_ylabel(r"relative error  $\|\hat x - x\| / \|x\|$")
    _error_panel(
        axes[1],
        degrees,
        res["error"]["scaled"],
        res["failed"]["scaled"],
        "(b) Time rescaled to [-1, 1]",
    )
    axes[0].legend(loc="lower right", fontsize=8.5)
    axes[0].annotate(
        "SVD drops tiny singular values,\nso its error stays below 100%",
        (11, 0.75),
        xytext=(10.2, 1e-7),
        fontsize=8,
        color=INK_2,
        ha="center",
        arrowprops={"arrowstyle": "-", "color": MUTED, "lw": 0.8},
    )
    axes[0].text(
        0.0,
        -0.2,
        r"X = Cholesky failed: $A^{\top}A$ is not numerically positive definite",
        transform=axes[0].transAxes,
        fontsize=8,
        color=INK_2,
    )

    ax = axes[2]
    labels = {"raw": "raw t", "scaled": "rescaled t", "legendre": "Legendre basis"}
    for color, (name, label) in zip(SERIES[3:6], labels.items(), strict=True):
        cond = np.array(res["cond"][name])
        ax.plot(degrees, cond, color=color, marker="o", ms=4.5, mec="white", mew=0.8, label=label)
        visible = np.flatnonzero(cond < 1e20)[-1]
        end_label(ax, degrees[visible], cond[visible], label)
    for level, text in [
        (1 / np.sqrt(EPS), r"$1/\sqrt{\varepsilon}$: normal equations exhausted"),
        (1 / EPS, r"$1/\varepsilon$: every method exhausted"),
    ]:
        ax.axhline(level, color=MUTED, lw=1.0, zorder=1)
        ax.annotate(
            text,
            (degrees[0], level),
            xytext=(0, 4),
            textcoords="offset points",
            fontsize=8,
            color=MUTED,
        )
    ax.set(
        yscale="log",
        ylim=(1, 1e20),
        xlabel="polynomial degree",
        xticks=degrees[1::2],
        ylabel=r"condition number $\kappa(A)$",
    )
    ax.set_title("(c) Conditioning of the design matrix")
    ax.legend(loc="center right", bbox_to_anchor=(1.0, 0.6), fontsize=8.5)
    for a in axes:
        a.grid(True, color=GRID)
    return fig


def first_failure(errors: list[float], failed: list[bool], degrees: list[int], tol: float) -> int:
    for d, e, f in zip(degrees, errors, failed, strict=True):
        if f or not e < tol:
            return d
    return -1


def main() -> None:
    use_style()
    res = run()
    path = save(plot(res), "stability")
    summary = {}
    for basis in ("raw", "scaled"):
        summary[basis] = {
            solver: first_failure(
                res["error"][basis][solver], res["failed"][basis][solver], res["degrees"], 1e-6
            )
            for solver in SOLVERS
        }
    res["first_degree_with_error_above_1e-6"] = summary
    save_results("stability", res)
    print(f"saved {path}")
    print("first polynomial degree whose coefficient error exceeds 1e-6 (-1 = never):")
    for basis, row in summary.items():
        print(f"  {basis:7s} " + "  ".join(f"{s}={d}" for s, d in row.items()))
    print(
        "condition number at degree 5: "
        + ", ".join(f"{k}={v[4]:.1e}" for k, v in res["cond"].items())
    )


if __name__ == "__main__":
    main()
