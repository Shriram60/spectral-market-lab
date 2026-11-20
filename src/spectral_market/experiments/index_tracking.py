"""Sparse index tracking: how few stocks does it take to replicate the NIFTY 50?

Baskets are chosen on 2021-2023 and weighted by the exact simplex QP, with two
selection rules: the long-only LASSO path, and greedy forward selection that re-solves the
budget-constrained QP for every candidate. Every basket is then held, unchanged, through
2024-2025. The benchmark for each basket size is the distribution of random baskets of
the same size, weighted by the same QP, so the comparison isolates *selection*.

Run: ``python -m spectral_market.experiments.index_tracking``
"""

from __future__ import annotations

import matplotlib.pyplot as plt
import numpy as np

from ..plotting import BAND, INK_2, MUTED, SERIES, end_label, note, save, save_results, use_style
from ..tracking import (
    greedy_tracking_path,
    lasso_tracking_path,
    random_basket_tracking,
    tracking_weights,
)
from ..universe import load_market

SPLIT = "2024-01-01"
RANDOM_SIZES = (3, 5, 8, 10, 15, 20, 30, 40)
N_DRAWS = 200


def run() -> dict:
    m = load_market()
    train, test = m.split(SPLIT)
    R, r = m.returns, m.index
    args = (R[train], r[train], R[test], r[test])
    path = lasso_tracking_path(*args)
    greedy = greedy_tracking_path(*args, max_stocks=40)
    rand = {k: random_basket_tracking(*args, k, n_draws=N_DRAWS, seed=k) for k in RANDOM_SIZES}
    full = tracking_weights(R[train], r[train])
    te_full = float(np.std(r[test] - R[test] @ full, ddof=1) * np.sqrt(252))

    def at(p, k: int) -> dict:
        i = int(np.argmin(np.abs(p.n_stocks - k)))
        w = p.weights[i]
        top = [j for j in np.argsort(-w) if w[j] > 0]
        return {
            "n_stocks": int(p.n_stocks[i]),
            "te_train": float(p.te_train[i]),
            "te_test": float(p.te_test[i]),
            "basket": {m.names[j]: round(float(w[j]), 4) for j in top},
        }

    def best_at_most(p, k: int) -> float:
        """Test error of the path basket with the most stocks not exceeding ``k``."""
        ok = np.flatnonzero(p.n_stocks <= k)
        return float(p.te_test[ok[np.argmax(p.n_stocks[ok])]])

    def share_worse(p) -> dict:
        return {str(k): float((rand[k] > best_at_most(p, k)).mean()) for k in RANDOM_SIZES}

    return {
        "path": path,
        "greedy": greedy,
        "random": rand,
        "summary": {
            "train": [str(m.dates[train][0].date()), str(m.dates[train][-1].date())],
            "test": [str(m.dates[test][0].date()), str(m.dates[test][-1].date())],
            "lasso_n_stocks": path.n_stocks.tolist(),
            "lasso_te_train": path.te_train.tolist(),
            "lasso_te_test": path.te_test.tolist(),
            "greedy_te_train": greedy.te_train.tolist(),
            "greedy_te_test": greedy.te_test.tolist(),
            "random_median_te_test": {str(k): float(np.median(v)) for k, v in rand.items()},
            "share_of_random_baskets_worse_than_lasso": share_worse(path),
            "share_of_random_baskets_worse_than_greedy": share_worse(greedy),
            "greedy_basket_5": at(greedy, 5),
            "greedy_basket_10": at(greedy, 10),
            "lasso_basket_5": at(path, 5),
            "lasso_basket_10": at(path, 10),
            "all_stocks_te_test": te_full,
            "all_stocks_n_nonzero": int((full > 1e-10).sum()),
        },
    }


def plot(res: dict):
    path, greedy, rand, s = res["path"], res["greedy"], res["random"], res["summary"]
    fig, ax = plt.subplots(figsize=(8.4, 4.8))
    ks = np.array(RANDOM_SIZES)
    q25, q50, q75 = (np.array([np.percentile(rand[k], p) for k in ks]) for p in (25, 50, 75))
    ax.fill_between(ks, 100 * q25, 100 * q75, color=BAND, lw=0, label="random baskets, IQR")
    ax.plot(ks, 100 * q50, color=MUTED, lw=1.2, marker="o", ms=4, label="random baskets, median")
    for p, color, name in [(greedy, SERIES[0], "greedy"), (path, SERIES[1], "LASSO path")]:
        ax.plot(
            p.n_stocks,
            100 * p.te_test,
            color=color,
            marker="o",
            ms=4,
            mec="white",
            mew=0.7,
            label=f"{name} selection",
        )
        end_label(ax, p.n_stocks[-1], 100 * p.te_test[-1], name.split()[0], dy=-9 * (p is greedy))
    b = s["greedy_basket_10"]
    ax.annotate(
        f"{b['n_stocks']} stocks: {100 * b['te_test']:.1f}%",
        (b["n_stocks"], 100 * b["te_test"]),
        xytext=(-30, -34),
        textcoords="offset points",
        fontsize=9,
        color=INK_2,
        arrowprops={"arrowstyle": "-", "color": MUTED, "lw": 0.8},
    )
    ax.set(yscale="log", xlabel="stocks held", ylabel="annualised tracking error (%)", xlim=(0, 52))
    ax.set_yticks([1, 2, 5, 10, 20, 50], ["1", "2", "5", "10", "20", "50"])
    ax.set_title("Out-of-sample tracking error: selection must respect the budget")
    ax.legend(loc="upper right")
    note(ax, "selected and weighted on 2021-23, held unchanged through 2024-25", loc="lower left")
    return fig


def main() -> None:
    use_style()
    res = run()
    path = save(plot(res), "index_tracking")
    save_results("index_tracking", res["summary"])
    s = res["summary"]
    print(f"saved {path}")
    for key in ("lasso_basket_5", "greedy_basket_5", "lasso_basket_10", "greedy_basket_10"):
        b = s[key]
        print(f"{key:<17} TE train {b['te_train']:.4f}, test {b['te_test']:.4f}")
        print(f"   {b['basket']}")
    print(f"all stocks ({s['all_stocks_n_nonzero']} held): TE test {s['all_stocks_te_test']:.4f}")
    for k, v in s["random_median_te_test"].items():
        lw = s["share_of_random_baskets_worse_than_lasso"][k]
        gw = s["share_of_random_baskets_worse_than_greedy"][k]
        print(f"random {k:>2}: median {v:.4f}; worse than LASSO {lw:.0%}, than greedy {gw:.0%}")


if __name__ == "__main__":
    main()
