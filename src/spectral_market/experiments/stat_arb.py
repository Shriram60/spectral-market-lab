"""PCA-residual statistical arbitrage (Avellaneda & Lee, 2010) on the NSE universe.

The paper's parameters are used unchanged: no threshold or window is tuned on this data,
so the whole 2022-2025 period is out of sample with respect to design choices. Costs are
charged one-way per unit traded on both the stock legs and the factor hedges.

Run: ``python -m spectral_market.experiments.stat_arb``
"""

from __future__ import annotations

import matplotlib.pyplot as plt
import numpy as np

from ..plotting import INK_2, MUTED, SERIES, end_label, note, save, save_results, use_style
from ..statarb import StatArbParams, performance, run_statarb
from ..universe import load_market

COSTS_BPS = (0, 2, 5, 10)
EXAMPLE_YEAR = 2024


def run() -> dict:
    m = load_market()
    p = StatArbParams()
    res = run_statarb(m.returns, p)
    dates = m.dates[p.pca_window + 1 :]  # P&L of signal day t is realised on day t + 1
    gross = np.abs(res.positions).sum(axis=1)
    capital = np.where(gross > 0, gross, 1.0)
    years = sorted(set(dates.year))
    by_year = {
        str(y): {
            str(c): performance(res.net(c)[np.asarray(dates.year == y)])["sharpe"]
            for c in COSTS_BPS
        }
        for y in years
    }
    nifty = np.expm1(m.index[p.pca_window + 1 :])
    trades_per_stock = np.abs(np.diff(res.positions, axis=0)).sum(axis=0)
    example = int(np.argmax(trades_per_stock))
    return {
        "res": res,
        "dates": dates,
        "params": p,
        "example": example,
        "example_name": m.names[example],
        "summary": {
            "period": [str(dates[0].date()), str(dates[-1].date())],
            "params": p.__dict__,
            "factors_per_refit": {
                "min": min(res.n_factors),
                "median": float(np.median(res.n_factors)),
                "max": max(res.n_factors),
            },
            "share_of_stock_days_tradeable": float(res.tradeable.mean()),
            "mean_open_positions": float(gross.mean()),
            "mean_daily_turnover": float((res.trades / capital).mean()),
            "performance_by_cost_bps": {str(c): performance(res.net(c)) for c in COSTS_BPS},
            "sharpe_by_year": by_year,
            "nifty_buy_and_hold": performance(nifty),
            "breakeven_cost_bps": float(res.pnl.mean() / (res.trades / capital).mean() * 1e4),
        },
    }


def plot(out: dict):
    res, dates, s, p = out["res"], out["dates"], out["summary"], out["params"]
    fig, axes = plt.subplots(
        1, 3, figsize=(16, 4.6), gridspec_kw={"width_ratios": [1.2, 1.2, 1], "wspace": 0.32}
    )

    # (a) the mechanism on one stock for one year.
    ax = axes[0]
    sel = np.asarray(dates.year == EXAMPLE_YEAR)
    d, sc = dates[sel], res.s_scores[sel, out["example"]]
    pos = res.positions[sel, out["example"]]
    for val, color in ((1, SERIES[2]), (-1, SERIES[1])):
        ax.fill_between(d, -3.5, 3.5, where=pos == val, color=color, alpha=0.15, lw=0, step="post")
    ax.plot(d, sc, color=SERIES[0], lw=1.2)
    for y, ls in (
        (p.s_open, "-"),
        (-p.s_open, "-"),
        (p.s_close_short, ":"),
        (-p.s_close_long, ":"),
    ):
        ax.axhline(y, color=MUTED, lw=0.8, ls=ls)
    ax.set(ylim=(-3.5, 3.5), ylabel="s-score")
    ax.set_title(f"Signal: {out['example_name']} residual s-score, {EXAMPLE_YEAR}")
    note(
        ax,
        "green: long residual   orange: short residual\nsolid: open ±1.25   dotted: close",
        loc="lower left",
    )
    ax.tick_params(axis="x", labelrotation=0)
    ax.xaxis.set_major_formatter(plt.matplotlib.dates.DateFormatter("%b"))

    # (b) cumulative performance against costs.
    ax = axes[1]
    for color, c in zip((SERIES[0], SERIES[3], SERIES[1]), (0, 2, 5), strict=True):
        wealth = np.cumprod(1 + res.net(c))
        ax.plot(dates, wealth, color=color, lw=1.6)
        end_label(ax, dates[-1], wealth[-1], f"{c} bps" if c else "no costs")
    ax.axhline(1, color=INK_2, lw=0.8)
    ax.set(ylabel="growth of 1 per unit of capital")
    ax.xaxis.set_major_locator(plt.matplotlib.dates.YearLocator())
    ax.xaxis.set_major_formatter(plt.matplotlib.dates.DateFormatter("%Y"))
    ax.set_title("Performance: the edge is smaller than trading costs")
    g = s["performance_by_cost_bps"]
    note(
        ax,
        f"Sharpe {g['0']['sharpe']:.2f} before costs, {g['5']['sharpe']:.2f} at 5 bps\n"
        f"break-even cost {s['breakeven_cost_bps']:.1f} bps",
        loc="upper left",
    )

    # (c) decay by year.
    ax = axes[2]
    years = list(s["sharpe_by_year"])
    x = np.arange(len(years))
    for j, (c, color) in enumerate(((0, SERIES[0]), (5, SERIES[1]))):
        vals = [s["sharpe_by_year"][y][str(c)] for y in years]
        ax.bar(
            x + (j - 0.5) * 0.38,
            vals,
            0.36,
            color=color,
            label="before costs" if c == 0 else f"{c} bps",
        )
    ax.axhline(0, color=INK_2, lw=0.8)
    ax.set_xticks(x, years)
    ax.grid(axis="x", visible=False)
    ax.set(ylabel="annualised Sharpe ratio")
    ax.set_title("Sharpe ratio by year")
    ax.legend(loc="upper right")
    return fig


def main() -> None:
    use_style()
    out = run()
    path = save(plot(out), "stat_arb")
    save_results("stat_arb", out["summary"])
    s = out["summary"]
    print(f"saved {path}")
    print(f"period {s['period'][0]} .. {s['period'][1]}, factors {s['factors_per_refit']}")
    print(
        f"tradeable stock-days {s['share_of_stock_days_tradeable']:.2f}, "
        f"open positions {s['mean_open_positions']:.1f}, "
        f"turnover {s['mean_daily_turnover']:.2f}/day"
    )
    for c, v in s["performance_by_cost_bps"].items():
        print(
            f"{c:>3} bps: return {v['annual_return']:+.3f} vol {v['annual_vol']:.3f} "
            f"Sharpe {v['sharpe']:+.2f} maxDD {v['max_drawdown']:.3f}"
        )
    print(f"break-even cost {s['breakeven_cost_bps']:.2f} bps")
    print(
        "Sharpe by year (0 / 5 bps):",
        {y: (round(v["0"], 2), round(v["5"], 2)) for y, v in s["sharpe_by_year"].items()},
    )
    print(f"NIFTY buy-and-hold Sharpe {s['nifty_buy_and_hold']['sharpe']:.2f}")


if __name__ == "__main__":
    main()
