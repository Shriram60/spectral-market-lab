"""Is there a time-varying trend? A local-linear-trend Kalman filter on the Phase 1 series.

For each series the slope's signal-to-noise ratio ``q_slope / q_level`` is fixed on a
grid and the remaining variances are fitted by maximum likelihood (profile likelihood).
The likelihood-ratio statistic against the unrestricted fit says which trend speeds the
data can rule out; under the null of a constant slope it is a 50:50 mixture of chi2(0)
and chi2(1) (boundary case), so the 5% critical value is 2.71.

Forecasts: parameters are fitted on 2021-2022 only, the filter runs causally through
2023-2025, and 5-day log-price forecasts are scored against the random walk.

Run: ``python -m spectral_market.experiments.kalman_trend``
"""

from __future__ import annotations

import matplotlib.pyplot as plt
import numpy as np

from ..data import PHASE1_TICKERS, load_close
from ..forecast import diebold_mariano
from ..kalman import fit_local_trend, kalman_filter, trend_forecast
from ..plotting import BAND, INK_2, MUTED, SERIES, save, save_results, use_style

SNR_GRID = np.logspace(-8, -2, 7)
SPLIT = "2023-01-01"
HORIZON = 5
LR_CRITICAL = 2.71  # 5% level, 0.5 chi2(0) + 0.5 chi2(1)
EXAMPLE = "^NSEI"
EXAMPLE_SNR = 1e-4
SKIP = 60  # days dominated by the diffuse prior, not plotted


def label(ticker: str) -> str:
    return "NIFTY 50" if ticker == "^NSEI" else ticker.removesuffix(".NS")


def evaluate(y: np.ndarray, n_train: int, snr: float | None) -> dict:
    fit = fit_local_trend(y[:n_train], snr=snr)
    res = kalman_filter(y, fit.q_level, fit.q_slope, fit.r)
    fc = trend_forecast(res, HORIZON)[:-HORIZON]
    target, rw = y[HORIZON:], y[:-HORIZON]
    test = slice(n_train, None)
    e, e_rw = target[test] - fc[test], target[test] - rw[test]
    stat, p = diebold_mariano(e, e_rw, HORIZON)
    return {
        "rmse_ratio": float(np.sqrt(np.mean(e**2) / np.mean(e_rw**2))),
        "dm_stat": stat,
        "dm_p": p,
    }


def run() -> dict:
    series = {}
    for ticker in PHASE1_TICKERS:
        close = load_close(ticker)
        y = np.log(close.to_numpy())
        n_train = int((close.index < SPLIT).sum())
        full = fit_local_trend(y)
        profile = np.array([fit_local_trend(y, snr=q).loglik for q in SNR_GRID])
        lr = np.maximum(2 * (full.loglik - profile), 0.0)
        series[ticker] = {
            "dates": close.index,
            "y": y,
            "mle": full,
            "lr": lr,
            "oos_mle": evaluate(y, n_train, None),
            "oos_grid": [evaluate(y, n_train, q) for q in SNR_GRID],
        }
    ex = series[EXAMPLE]
    tracks = {}
    for name, snr in (("MLE", None), ("trend-following", EXAMPLE_SNR)):
        fit = fit_local_trend(ex["y"], snr=snr)
        res = kalman_filter(ex["y"], fit.q_level, fit.q_slope, fit.r)
        tracks[name] = (res.filtered_mean[:, 1], np.sqrt(res.filtered_cov[:, 1, 1]), fit)
    return {"series": series, "tracks": tracks}


def summarise(res: dict) -> dict:
    out = {"snr_grid": SNR_GRID.tolist(), "horizon": HORIZON, "train_until": SPLIT, "series": {}}
    for ticker, s in res["series"].items():
        mle = s["mle"]
        rejected = SNR_GRID[s["lr"] > LR_CRITICAL]
        out["series"][label(ticker)] = {
            "q_level": mle.q_level,
            "q_slope": mle.q_slope,
            "r": mle.r,
            "snr_mle": mle.signal_to_noise,
            "lr_by_snr": s["lr"].tolist(),
            "smallest_rejected_snr": float(rejected.min()) if rejected.size else None,
            "oos_mle": s["oos_mle"],
            "oos_rmse_ratio_by_snr": [g["rmse_ratio"] for g in s["oos_grid"]],
            "oos_dm_p_by_snr": [g["dm_p"] for g in s["oos_grid"]],
        }
    ex = res["tracks"]
    out["example"] = {
        name: {
            "final_slope_annual": float(m[-1] * 252),
            "final_slope_se_annual": float(sd[-1] * 252),
        }
        for name, (m, sd, _) in ex.items()
    }
    return out


def plot(res: dict, s: dict):
    fig, axes = plt.subplots(1, 3, figsize=(15.5, 4.6), gridspec_kw={"wspace": 0.28})
    ax1, ax2, ax3 = axes
    for color, (ticker, d) in zip(SERIES, res["series"].items(), strict=False):
        kw = {"color": color, "marker": "o", "ms": 4, "mec": "white", "mew": 0.6, "lw": 1.4}
        ax1.plot(SNR_GRID, np.maximum(d["lr"], 0.05), label=label(ticker), **kw)
        ax2.plot(SNR_GRID, [g["rmse_ratio"] for g in d["oos_grid"]], **kw)
    band = ax1.axhspan(0.05, LR_CRITICAL, color=BAND, lw=0, zorder=0)
    crit = ax1.axhline(LR_CRITICAL, color=INK_2, lw=0.8, ls="--")
    ax1.set(
        xscale="log",
        yscale="log",
        ylim=(0.05, 400),
        xlabel=r"slope signal-to-noise $q_{\mathrm{slope}} / q_{\mathrm{level}}$",
        ylabel="likelihood-ratio statistic",
    )
    ax1.set_title("A moving trend is rejected by the likelihood")
    ax1.legend([band, crit], ["not rejected at 5%", "critical value 2.71"], loc="upper left")

    ax2.axhline(1.0, color=INK_2, lw=0.8)
    ax2.set(
        xscale="log",
        xlabel=r"slope signal-to-noise $q_{\mathrm{slope}} / q_{\mathrm{level}}$",
        ylabel=f"{HORIZON}-day RMSE / random walk RMSE",
    )
    ax2.set_title("...and chasing one hurts forecasts (2023-25)")
    ax2.legend(*ax1.get_legend_handles_labels(), loc="upper left", fontsize=8, ncol=2)

    dates = res["series"][EXAMPLE]["dates"][SKIP:]
    for color, (name, (m, sd, _)) in zip(
        (SERIES[0], SERIES[1]), res["tracks"].items(), strict=True
    ):
        m, sd = m[SKIP:], sd[SKIP:]
        ax3.fill_between(
            dates,
            100 * 252 * (m - 1.96 * sd),
            100 * 252 * (m + 1.96 * sd),
            color=color,
            alpha=0.15,
            lw=0,
        )
        ax3.plot(
            dates,
            100 * 252 * m,
            color=color,
            lw=1.4,
            label="maximum likelihood" if name == "MLE" else f"forced SNR = {EXAMPLE_SNR:g}",
        )
    ax3.axhline(0, color=MUTED, lw=0.8)
    ax3.set(ylabel="filtered drift, % per year (95% band)")
    ax3.set_title("NIFTY 50: estimated drift over time")
    ax3.legend(loc="lower right")
    ax3.xaxis.set_major_locator(plt.matplotlib.dates.YearLocator())
    ax3.xaxis.set_major_formatter(plt.matplotlib.dates.DateFormatter("%Y"))
    return fig


def main() -> None:
    use_style()
    res = run()
    s = summarise(res)
    path = save(plot(res, s), "kalman_trend")
    save_results("kalman_trend", s)
    print(f"saved {path}")
    for name, d in s["series"].items():
        o = d["oos_mle"]
        print(
            f"{name:<11} snr_mle {d['snr_mle']:.1e}  rejects snr >= {d['smallest_rejected_snr']}  "
            f"OOS RMSE/RW {o['rmse_ratio']:.4f} (p = {o['dm_p']:.2f})"
        )
    for name, e in s["example"].items():
        drift, se = e["final_slope_annual"], e["final_slope_se_annual"]
        print(f"NIFTY {name}: final drift {drift:+.3f} (se {se:.3f})")


if __name__ == "__main__":
    main()
