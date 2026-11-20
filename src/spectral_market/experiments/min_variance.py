"""Minimum-variance portfolios: does cleaning the covariance matrix pay off out of sample?

Four estimators feed the same optimiser: the sample covariance, Ledoit-Wolf shrinkage,
RMT eigenvalue clipping and a PCA factor model whose size is set by the Marchenko-Pastur
edge inside each window. The estimation window is varied from 84 to 504 days, i.e.
q = N/T from 0.61 down to 0.10, and every configuration is scored on the same
out-of-sample period (2023-2025), rebalanced monthly. Differences in realised variance
are tested with a Diebold-Mariano test on squared daily returns (HAC, 10 lags).

Run: ``python -m spectral_market.experiments.min_variance``
"""

from __future__ import annotations

import matplotlib.pyplot as plt
import numpy as np

from .. import covariance as cv
from ..forecast import diebold_mariano
from ..plotting import INK_2, MUTED, SERIES, end_label, note, save, save_results, use_style
from ..portfolio import backtest
from ..universe import load_market

WINDOWS = (84, 126, 189, 252, 378, 504)
REBALANCE = 21
ESTIMATORS = {
    "Sample": cv.sample_covariance,
    "Ledoit-Wolf": lambda X: cv.ledoit_wolf(X)[0],
    "RMT clipping": lambda X: cv.rmt_clipped(X)[0],
    "PCA factor model": lambda X: cv.factor_model(X),
}


def run() -> dict:
    m = load_market()
    R = m.returns
    N = R.shape[1]
    start = max(WINDOWS)
    res: dict = {"q": [N / w for w in WINDOWS], "oos_start": str(m.dates[start].date())}
    for long_only in (False, True):
        key = "long_only" if long_only else "unconstrained"
        table = {}
        for name, est in ESTIMATORS.items():
            runs = [
                backtest(R, est, window=w, rebalance=REBALANCE, long_only=long_only, start=start)
                for w in WINDOWS
            ]
            table[name] = runs
        res[key] = table
    res["equal_weight"] = backtest(R, None, window=start, rebalance=REBALANCE, start=start)
    return res


def summarise(res: dict) -> dict:
    out: dict = {
        "windows": list(WINDOWS),
        "q": res["q"],
        "oos_start": res["oos_start"],
        "rebalance_days": REBALANCE,
        "equal_weight_vol": res["equal_weight"].vol,
    }
    for key in ("unconstrained", "long_only"):
        block = {}
        sample = res[key]["Sample"]
        for name, runs in res[key].items():
            rows = []
            for i, (w, bt) in enumerate(zip(WINDOWS, runs, strict=True)):
                stat, p = diebold_mariano(bt.returns, sample[i].returns, horizon=10)
                rows.append(
                    {
                        "window": w,
                        "vol": bt.vol,
                        "vol_vs_sample": bt.vol / sample[i].vol - 1,
                        "dm_stat_vs_sample": stat,
                        "dm_p_vs_sample": p,
                        "bias_ratio": bt.bias_ratio,
                        "turnover": float(bt.turnover[1:].mean()),
                        "gross_leverage": float(bt.gross_leverage.mean()),
                    }
                )
            block[name] = rows
        out[key] = block
    return out


def plot(res: dict, s: dict):
    q = np.array(res["q"])
    fig, axes = plt.subplots(1, 3, figsize=(15, 4.6), sharex=True)
    panels = [
        (axes[0], "unconstrained", "vol", "Out-of-sample volatility (shorting allowed)"),
        (axes[1], "long_only", "vol", "Out-of-sample volatility (long-only)"),
        (axes[2], "unconstrained", "bias_ratio", "Realised / predicted risk (shorting allowed)"),
    ]
    for ax, key, field, title in panels:
        for color, (name, rows) in zip(SERIES, s[key].items(), strict=False):
            y = np.array([r[field] for r in rows]) * (100 if field == "vol" else 1)
            ax.plot(q, y, color=color, marker="o", ms=4.5, mec="white", mew=0.7, label=name)
        ax.set_title(title)
        ax.set_xlabel("q = N / T  (shorter window to the right)")
    eq = 100 * s["equal_weight_vol"]
    for ax in axes[:2]:
        ax.axhline(eq, color=MUTED, lw=1.0, ls="--")
        end_label(ax, q.max(), eq, "1/N", dy=6)
        ax.set_ylabel("annualised volatility (%)")
    for ax, key in zip(axes[:2], ("unconstrained", "long_only"), strict=True):
        vals = [r["vol"] for rows in s[key].values() for r in rows] + [s["equal_weight_vol"]]
        ax.set_ylim(100 * min(vals) * 0.96, 100 * max(vals) * 1.03)
    axes[2].axhline(1.0, color=INK_2, lw=0.8)
    axes[2].set_ylabel(r"$\sigma_{\mathrm{realised}} / \sigma_{\mathrm{predicted}}$")
    axes[1].legend(loc="center left")
    u = s["unconstrained"]  # annotate the shortest window, where noise is largest
    note(
        axes[0],
        f"T = {WINDOWS[0]} days, vs sample:\n"
        f"Ledoit-Wolf {100 * u['Ledoit-Wolf'][0]['vol_vs_sample']:+.0f}%, "
        f"RMT {100 * u['RMT clipping'][0]['vol_vs_sample']:+.0f}%",
        loc="upper left",
    )
    note(axes[2], f"out of sample from {s['oos_start']}", loc="lower right")
    return fig


def main() -> None:
    use_style()
    res = run()
    s = summarise(res)
    path = save(plot(res, s), "min_variance")
    save_results("min_variance", s)
    print(f"saved {path}")
    print(f"out of sample from {s['oos_start']}; equal weight vol {s['equal_weight_vol']:.4f}")
    for key in ("unconstrained", "long_only"):
        print(f"--- {key}")
        for name, rows in s[key].items():
            cells = "  ".join(
                f"{r['window']}:{r['vol']:.4f}"
                f"({100 * r['vol_vs_sample']:+.0f}%,p={r['dm_p_vs_sample']:.2f})"
                for r in rows
            )
            print(f"{name:<17} {cells}")


if __name__ == "__main__":
    main()
