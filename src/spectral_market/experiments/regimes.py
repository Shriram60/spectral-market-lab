"""Market regimes from a Gaussian hidden Markov model of NIFTY 50 daily returns.

The number of regimes is chosen by BIC. Regimes are then related to the Phase 2
spectrum: the rolling market-mode share ``lambda_1 / N`` of the 51-stock correlation
matrix measures how much stocks move together. Finally the HMM's one-day-ahead
variance forecast (fitted on 2021-2022, filtered causally through 2023-2025) is
compared with an EWMA (RiskMetrics, lambda = 0.94) and a 21-day rolling variance.

Run: ``python -m spectral_market.experiments.regimes``
"""

from __future__ import annotations

import matplotlib.pyplot as plt
import numpy as np

from .. import spectral as sp
from ..forecast import loss_differential_test
from ..hmm import baum_welch, forward, predicted_variance, viterbi
from ..plotting import BAND, INK_2, SERIES, note, save, save_results, use_style
from ..universe import load_market

K_RANGE = (1, 2, 3)
N_INIT = 5
CORR_WINDOW = 63
SPLIT = "2023-01-01"
EWMA_LAMBDA = 0.94


def qlike(r2: np.ndarray, h: np.ndarray) -> np.ndarray:
    """Patton's robust variance-forecast loss (up to a constant); defined when r = 0."""
    return r2 / h + np.log(h)


def run() -> dict:
    m = load_market()
    x, T = m.index, len(m.index)
    fits = {k: baum_welch(x, k, n_init=N_INIT) for k in K_RANGE}
    bic = {k: float(f.bic(T)) for k, f in fits.items()}
    k_best = min(bic, key=bic.get)
    model = fits[k_best]
    path = viterbi(model, x)

    # Centred window: a descriptive comparison, so the measure should not lag the regime.
    lam1 = np.full(T, np.nan)
    half = CORR_WINDOW // 2
    for i in range(CORR_WINDOW, T + 1):
        lam1[i - 1 - half] = sp.pca_correlation(m.returns[i - CORR_WINDOW : i]).eigenvalues[0]
    lam1 /= m.returns.shape[1]

    # Out-of-sample variance forecasts for day t + 1 made at the close of day t.
    train, _ = m.split(SPLIT)
    n = int(train.sum())
    oos_model = baum_welch(x[train], k_best, n_init=N_INIT)
    alpha, _, _ = forward(oos_model, x)
    forecasts = {"HMM": predicted_variance(oos_model, alpha)[:-1]}
    ew, v = np.empty(T - 1), x[:21].var()
    for i in range(T - 1):
        v = EWMA_LAMBDA * v + (1 - EWMA_LAMBDA) * x[i] ** 2
        ew[i] = v
    forecasts["EWMA"] = ew
    forecasts["21-day rolling"] = np.array([x[max(0, i - 20) : i + 1].var() for i in range(T - 1)])
    r2 = x[1:] ** 2
    test = slice(n - 1, None)  # forecasts of days in the test period
    losses = {name: qlike(r2[test], h[test]) for name, h in forecasts.items()}
    comparison = {}
    for name, loss in losses.items():
        stat, p = (np.nan, np.nan)
        if name != "EWMA":
            stat, p = loss_differential_test(loss - losses["EWMA"], horizon=10)
        comparison[name] = {"qlike": float(loss.mean()), "dm_stat_vs_ewma": stat, "dm_p_vs_ewma": p}

    stats = []
    for s in range(model.k):
        sel = path == s
        stats.append(
            {
                "annual_vol": float(model.sigma[s] * np.sqrt(252)),
                "annual_mean": float(model.mu[s] * 252),
                "expected_duration_days": float(model.expected_duration[s]),
                "stationary_share": float(model.stationary[s]),
                "viterbi_days": int(sel.sum()),
                "mean_market_mode_share": float(np.nanmean(lam1[sel])),
            }
        )
    ok = np.isfinite(lam1)
    return {
        "dates": m.dates,
        "index": x,
        "path": path,
        "lam1": lam1,
        "model": model,
        "summary": {
            "bic": {str(k): v for k, v in bic.items()},
            "k_selected": k_best,
            "regimes": stats,
            "transition_matrix": model.A.tolist(),
            "corr_lam1_turbulent_indicator": float(
                np.corrcoef(lam1[ok], path[ok] == model.k - 1)[0, 1]
            ),
            "variance_forecast_oos_from": str(m.dates[n].date()),
            "variance_forecast": comparison,
        },
    }


def _shade(ax, dates, mask, color):
    edges = np.flatnonzero(np.diff(np.r_[0, mask.astype(int), 0]))
    for a, b in zip(edges[::2], edges[1::2], strict=True):
        ax.axvspan(dates[a], dates[min(b, len(dates) - 1)], color=color, lw=0, zorder=0)


def plot(res: dict):
    dates, path, s = res["dates"], res["path"], res["summary"]
    turbulent = path == res["model"].k - 1
    fig, (ax1, ax2) = plt.subplots(
        2,
        1,
        figsize=(11, 6.4),
        sharex=True,
        gridspec_kw={"height_ratios": [1.3, 1], "hspace": 0.12},
    )
    _shade(ax1, dates, turbulent, BAND)
    ax1.plot(dates, np.exp(np.cumsum(res["index"])), color=SERIES[0], lw=1.4)
    ax1.set(ylabel="NIFTY 50, growth of 1")
    calm, turb = s["regimes"][0], s["regimes"][-1]
    ax1.set_title(
        f"Two regimes (BIC): calm {100 * calm['annual_vol']:.0f}% vol, "
        f"turbulent {100 * turb['annual_vol']:.0f}% vol (shaded)"
    )
    note(
        ax1,
        f"mean stay: calm {calm['expected_duration_days']:.0f} days, "
        f"turbulent {turb['expected_duration_days']:.0f} days\n"
        f"time in turbulent regime: {100 * turb['stationary_share']:.0f}%",
        loc="upper left",
    )

    _shade(ax2, dates, turbulent, BAND)
    ax2.plot(dates, res["lam1"], color=SERIES[1], lw=1.4)
    for r, label in ((calm, "calm"), (turb, "turbulent")):
        ax2.axhline(r["mean_market_mode_share"], color=INK_2, lw=0.8, ls=":")
        ax2.annotate(
            f"{label} mean {r['mean_market_mode_share']:.2f}",
            (dates[-1], r["mean_market_mode_share"]),
            xytext=(6, 0),
            textcoords="offset points",
            va="center",
            fontsize=9,
            color=INK_2,
            annotation_clip=False,
        )
    ax2.set(ylabel=r"market mode share $\lambda_1 / N$")
    ax2.set_title(
        f"Stocks move together in the turbulent regime (centred {CORR_WINDOW}-day window)"
    )
    return fig


def main() -> None:
    use_style()
    res = run()
    path = save(plot(res), "regimes")
    save_results("regimes", res["summary"])
    s = res["summary"]
    print(f"saved {path}")
    print("BIC:", {k: round(v, 1) for k, v in s["bic"].items()}, "-> K =", s["k_selected"])
    for i, r in enumerate(s["regimes"]):
        print(
            f"regime {i}: vol {r['annual_vol']:.3f}, mean {r['annual_mean']:+.3f}, "
            f"stay {r['expected_duration_days']:.1f}d, share {r['stationary_share']:.2f}, "
            f"lambda1/N {r['mean_market_mode_share']:.3f}"
        )
    print(f"variance forecasts from {s['variance_forecast_oos_from']}:")
    for name, v in s["variance_forecast"].items():
        print(f"  {name:<15} QLIKE {v['qlike']:.4f}  DM vs EWMA p = {v['dm_p_vs_ewma']:.3f}")


if __name__ == "__main__":
    main()
