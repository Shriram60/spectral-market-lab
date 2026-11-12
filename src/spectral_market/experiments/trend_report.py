"""The original project redone: what a least-squares trend line can and cannot tell you.

1. The original 8-day fit (pinned to the last 8 sessions of 2025), with the numbers
   the first version should have reported: R^2, a confidence interval on the slope,
   prediction intervals and a residual-orthogonality check. It also shows that the
   eigenvalues of A^T A it reported are fixed by the day indices alone.
2. A 5-year log-linear trend, where R^2 looks respectable but the Durbin-Watson
   statistic shows the residuals are a random walk, so the OLS confidence interval is
   badly overconfident. The drift estimated from returns gives the honest interval.
3. Walk-forward evaluation: rolling trend forecasts against the no-change (random
   walk) forecast for 7 series, 5 window lengths, 3 polynomial degrees and 2 horizons.

Run: ``python -m spectral_market.experiments.trend_report``
"""

from __future__ import annotations

import matplotlib.pyplot as plt
import numpy as np

from ..data import PHASE1_TICKERS, load_close
from ..forecast import walk_forward
from ..linalg import jacobi_svd
from ..plotting import (
    BAND,
    INK_2,
    MUTED,
    SERIES,
    end_label,
    note,
    save,
    save_results,
    use_style,
)
from ..trend import design_matrix, drift_from_returns, fit_trend

TICKER = "MUTHOOTFIN.NS"
WINDOWS = [8, 20, 60, 120, 250]
DEGREES = {
    0: "rolling mean (degree 0)",
    1: "linear trend (degree 1)",
    2: "quadratic trend (degree 2)",
}
HORIZONS = [1, 5]
TRADING_DAYS = 252


def annualise(daily_log_growth: float) -> float:
    return float((np.exp(TRADING_DAYS * daily_log_growth) - 1.0) * 100.0)


# --------------------------------------------------------------------------- analyses


def eight_day(close) -> tuple[dict, object]:
    window = close.iloc[-8:]
    fit = fit_trend(window.values)
    raw = design_matrix(np.arange(1, 9), 1)
    eig = jacobi_svd(raw).s ** 2  # eigenvalues of A^T A are the squared singular values of A
    slope, lo, hi = fit.slope_ci()
    pred, plo, phi = fit.prediction_interval([9, 10])
    return {
        "dates": [d.strftime("%Y-%m-%d") for d in window.index],
        "prices": window.values.round(2).tolist(),
        "eigenvalues_AtA_raw": eig.tolist(),
        "cond_AtA_raw": float(eig[0] / eig[-1]),
        "cond_A_raw": float(np.sqrt(eig[0] / eig[-1])),
        "cond_A_standardised": fit.cond,
        "r2": fit.r2,
        "slope_per_day": slope,
        "slope_ci95": [lo, hi],
        "residual_orthogonality": fit.orthogonality,
        "forecast_day9_10": pred.tolist(),
        "prediction_interval95": [[a, b] for a, b in zip(plo, phi, strict=True)],
    }, fit


def five_year(close) -> tuple[dict, object]:
    fit = fit_trend(close.values, log=True)
    slope, lo, hi = fit.slope_ci()
    mu, rlo, rhi = drift_from_returns(close.values)
    return {
        "n_days": int(close.size),
        "r2": fit.r2,
        "durbin_watson": fit.durbin_watson,
        "ols_annual_growth_pct": annualise(slope),
        "ols_ci95_pct": [annualise(lo), annualise(hi)],
        "returns_annual_growth_pct": annualise(mu),
        "returns_ci95_pct": [annualise(rlo), annualise(rhi)],
        "ci_width_ratio": (rhi - rlo) / (hi - lo),
    }, fit


def walk_forward_grid(series: dict[str, np.ndarray]) -> list[dict]:
    rows = []
    for ticker, logp in series.items():
        for h in HORIZONS:
            for w in WINDOWS:
                for d in DEGREES:
                    ev = walk_forward(logp, window=w, horizon=h, degree=d)
                    rows.append(
                        {
                            "ticker": ticker,
                            "horizon": h,
                            "window": w,
                            "degree": d,
                            "n_forecasts": ev.n_forecasts,
                            "rmse_ratio": ev.rmse_ratio,
                            "directional_accuracy": ev.directional_accuracy,
                            "dm_stat": ev.dm_stat,
                            "dm_pvalue": ev.dm_pvalue,
                        }
                    )
    return rows


# --------------------------------------------------------------------------- figures


def plot_trend(close, r8: dict, fit8, r5: dict, fit5):
    fig, axes = plt.subplots(
        2, 2, figsize=(13.5, 8.6), gridspec_kw={"hspace": 0.42, "wspace": 0.22}
    )

    # (a) the original 8-day fit, with the uncertainty it never showed
    ax = axes[0, 0]
    grid = np.linspace(1, 10, 200)
    mid, lo, hi = fit8.prediction_interval(grid)
    ax.fill_between(grid, lo, hi, color=BAND, lw=0, label="95% prediction interval")
    ax.plot(grid, mid, color=SERIES[1], label="least-squares trend")
    ax.plot(
        np.arange(1, 9),
        r8["prices"],
        ls="none",
        marker="o",
        ms=7,
        color=SERIES[0],
        mec="white",
        mew=1.0,
        label="actual close",
        zorder=3,
    )
    ax.plot(
        [9, 10],
        r8["forecast_day9_10"],
        ls="none",
        marker="D",
        ms=7,
        color=SERIES[1],
        mec="white",
        mew=1.0,
        zorder=3,
    )
    end_label(ax, 10, r8["forecast_day9_10"][1], "forecast")
    labels = [d[5:].replace("-", "/") for d in r8["dates"]] + ["+1", "+2"]
    span = max(hi.max() - lo.min(), 1.0)
    ax.set(
        xticks=np.arange(1, 11),
        xlim=(0.5, 11.2),
        ylabel="price (INR)",
        ylim=(lo.min() - 0.45 * span, hi.max() + 0.05 * span),
    )
    ax.set_xticklabels(labels, fontsize=8.5)
    ax.set_title("(a) The original setup: last 8 sessions of 2025")
    lo_ci, hi_ci = r8["slope_ci95"]
    note(
        ax,
        f"R² = {r8['r2']:.3f}\nslope = {r8['slope_per_day']:+.2f} INR/day\n"
        f"95% CI [{lo_ci:+.1f}, {hi_ci:+.1f}]: includes zero",
        loc="lower left",
    )
    ax.legend(loc="lower right")

    # (b) residuals of the 8-day fit
    ax = axes[0, 1]
    ax.bar(np.arange(1, 9), fit8.residuals, width=0.72, color=SERIES[0])
    ax.axhline(0, color=INK_2, lw=0.8)
    ax.set(xticks=np.arange(1, 9), ylabel="residual (INR)")
    ax.set_xticklabels(labels[:8], fontsize=8.5)
    ax.set_title(r"(b) Residuals $e = b - A\hat{x}$, orthogonal to the column space of $A$")
    note(
        ax,
        r"$\max|A^{\top}e| \,/\, (\|A\|\,\|e\|)$" + f" = {r8['residual_orthogonality']:.0e}\n"
        r"$\kappa(A^{\top}A)$" + f" = {r8['cond_AtA_raw']:.2f} on raw days, for any stock",
        loc="lower left",
    )

    # (c) 5-year log-linear trend
    ax = axes[1, 0]
    dates = close.index
    ax.plot(dates, close.values, color=SERIES[0], lw=1.3, label="adjusted close")
    ax.plot(dates, np.exp(fit5.fitted), color=SERIES[1], label="log-linear trend")
    ax.set(yscale="log", ylabel="price (INR, log scale)")
    ax.set_title("(c) Five years: a trend line that looks convincing")
    lo5, hi5 = r5["ols_ci95_pct"]
    note(
        ax,
        f"R² = {r5['r2']:.2f}\ngrowth {r5['ols_annual_growth_pct']:.1f}%/yr, "
        f"OLS 95% CI [{lo5:.1f}, {hi5:.1f}]",
        loc="upper left",
    )
    ax.legend(loc="lower right")

    # (d) its residuals are a random walk
    ax = axes[1, 1]
    ax.plot(dates, fit5.residuals, color=SERIES[0], lw=1.3)
    ax.axhline(0, color=INK_2, lw=0.8)
    r_lo, r_hi = fit5.residuals.min(), fit5.residuals.max()
    ax.set(
        ylabel="residual (log price)",
        ylim=(r_lo - 0.45 * (r_hi - r_lo), r_hi + 0.08 * (r_hi - r_lo)),
    )
    ax.set_title("(d) ...whose residuals wander like a random walk")
    rlo, rhi = r5["returns_ci95_pct"]
    note(
        ax,
        f"Durbin-Watson = {r5['durbin_watson']:.3f}  (2 = independent errors)\n"
        f"honest 95% CI from returns: [{rlo:.1f}, {rhi:.1f}]%/yr,\n"
        f"{r5['ci_width_ratio']:.0f}x wider than the OLS interval",
        loc="lower left",
    )
    for a in axes[1]:
        a.tick_params(axis="x", labelsize=8.5)
    return fig


def plot_walk_forward(rows: list[dict]):
    fig, axes = plt.subplots(1, 2, figsize=(13.5, 4.8), sharey=True, gridspec_kw={"wspace": 0.28})
    x = np.arange(len(WINDOWS))
    offsets = {0: -0.12, 1: 0.0, 2: 0.12}
    for ax, h in zip(axes, HORIZONS, strict=True):
        ax.axhline(1.0, color=INK_2, lw=1.0, zorder=1)
        ax.annotate(
            "random walk (no-change forecast)",
            (x[0] - 0.3, 1.0),
            xytext=(0, -12),
            textcoords="offset points",
            fontsize=8.5,
            color=INK_2,
        )
        for color, (d, label) in zip(SERIES, DEGREES.items(), strict=False):
            sel = [r for r in rows if r["horizon"] == h and r["degree"] == d]
            ratios = np.array(
                [[r["rmse_ratio"] for r in sel if r["window"] == w] for w in WINDOWS]
            )  # (windows, tickers)
            ax.plot(
                np.repeat(x + offsets[d], ratios.shape[1]),
                ratios.ravel(),
                ls="none",
                marker="o",
                ms=4,
                color=color,
                alpha=0.35,
                mew=0,
            )
            med = np.median(ratios, axis=1)
            ax.plot(
                x + offsets[d],
                med,
                color=color,
                marker="o",
                ms=6.5,
                mec="white",
                mew=1.0,
                label=label,
            )
            end_label(ax, x[-1] + offsets[d], med[-1], f"degree {d}")
        ax.set(
            xticks=x,
            xticklabels=[str(w) for w in WINDOWS],
            yscale="log",
            xlabel="look-back window (trading days)",
            xlim=(-0.45, len(WINDOWS) - 0.2),
        )
        ax.set_yticks([0.8, 1, 2, 4, 8, 16])
        ax.set_yticklabels(["0.8", "1", "2", "4", "8", "16"])
        ax.set_title(f"({'ab'[HORIZONS.index(h)]}) {h}-day-ahead forecasts")
    axes[0].set_ylabel("RMSE relative to random walk\n(below 1 = better than no-change)")
    axes[0].set_ylim(0.75, 20)
    axes[0].legend(loc="upper left")
    axes[1].text(
        1.0,
        -0.2,
        "Lines: median over 7 series (MUTHOOTFIN, NIFTY 50, RELIANCE, TCS, "
        "HDFCBANK, INFY, ITC), 2021-2025. Dots: individual series.",
        transform=axes[1].transAxes,
        ha="right",
        fontsize=8,
        color=MUTED,
    )
    return fig


# --------------------------------------------------------------------------- main


def summarise(rows: list[dict]) -> list[dict]:
    out = []
    for h in HORIZONS:
        for w in WINDOWS:
            for d in DEGREES:
                sel = [
                    r for r in rows if r["horizon"] == h and r["window"] == w and r["degree"] == d
                ]
                out.append(
                    {
                        "horizon": h,
                        "window": w,
                        "degree": d,
                        "median_rmse_ratio": float(np.median([r["rmse_ratio"] for r in sel])),
                        "mean_directional_accuracy": float(
                            np.mean([r["directional_accuracy"] for r in sel])
                        ),
                        "significant_wins": sum(
                            r["dm_stat"] < 0 and r["dm_pvalue"] < 0.05 for r in sel
                        ),
                        "significant_losses": sum(
                            r["dm_stat"] > 0 and r["dm_pvalue"] < 0.05 for r in sel
                        ),
                        "n_series": len(sel),
                    }
                )
    return out


def main() -> None:
    use_style()
    close = load_close(TICKER)
    r8, fit8 = eight_day(close)
    r5, fit5 = five_year(close)
    series = {t: np.log(load_close(t).values) for t in PHASE1_TICKERS}
    rows = walk_forward_grid(series)
    summary = summarise(rows)

    p1 = save(plot_trend(close, r8, fit8, r5, fit5), "trend_diagnostics")
    p2 = save(plot_walk_forward(rows), "walk_forward")
    save_results(
        "trend_report",
        {
            "ticker": TICKER,
            "eight_day": r8,
            "five_year": r5,
            "walk_forward_summary": summary,
            "walk_forward": rows,
        },
    )
    print(f"saved {p1}\nsaved {p2}\n")

    print(f"8-day fit on {r8['dates'][0]}..{r8['dates'][-1]}")
    print(
        f"  eigenvalues of A^T A (raw days 1..8): {np.round(r8['eigenvalues_AtA_raw'], 2)}"
        f"  -> cond(A^T A) = {r8['cond_AtA_raw']:.2f}, identical for every stock"
    )
    print(f"  cond(A): raw {r8['cond_A_raw']:.2f} -> standardised {r8['cond_A_standardised']:.2f}")
    print(
        f"  R^2 = {r8['r2']:.4f}; slope {r8['slope_per_day']:+.2f}/day, "
        f"95% CI [{r8['slope_ci95'][0]:+.2f}, {r8['slope_ci95'][1]:+.2f}]"
    )
    for day, p, (lo, hi) in zip(
        (9, 10), r8["forecast_day9_10"], r8["prediction_interval95"], strict=True
    ):
        print(f"  day {day}: {p:.2f}  95% PI [{lo:.2f}, {hi:.2f}]")
    print(f"\n5-year log-linear fit: R^2 = {r5['r2']:.3f}, DW = {r5['durbin_watson']:.4f}")
    print(
        f"  OLS:     {r5['ols_annual_growth_pct']:.1f}%/yr, "
        f"95% CI [{r5['ols_ci95_pct'][0]:.1f}, {r5['ols_ci95_pct'][1]:.1f}]"
    )
    print(
        f"  returns: {r5['returns_annual_growth_pct']:.1f}%/yr, "
        f"95% CI [{r5['returns_ci95_pct'][0]:.1f}, {r5['returns_ci95_pct'][1]:.1f}]"
        f"  ({r5['ci_width_ratio']:.1f}x wider)"
    )
    print(
        "\nwalk-forward (median RMSE ratio vs random walk | mean directional accuracy | "
        "significant wins/losses out of 7)"
    )
    for s in summary:
        print(
            f"  h={s['horizon']} w={s['window']:3d} deg={s['degree']}: "
            f"{s['median_rmse_ratio']:6.2f} | {s['mean_directional_accuracy']:.3f} | "
            f"{s['significant_wins']}/{s['significant_losses']}"
        )
    total = len(rows)
    wins = sum(r["dm_stat"] < 0 and r["dm_pvalue"] < 0.05 for r in rows)
    beat = sum(r["rmse_ratio"] < 1 for r in rows)
    print(
        f"\n{total} configurations; {beat} have lower RMSE than the random walk; "
        f"{wins} beat it significantly (DM test, 5%)"
    )
    print(
        f"directional accuracy range: "
        f"{min(r['directional_accuracy'] for r in rows):.3f}-"
        f"{max(r['directional_accuracy'] for r in rows):.3f}"
    )


if __name__ == "__main__":
    main()
