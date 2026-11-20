"""Unsupervised sector discovery by spectral clustering of the correlation graph.

The market mode is projected out first: it correlates every stock with every other and
drowns the sector structure. Stocks are then nodes of a graph weighted by positive
residual correlation, and the normalised Laplacian's low eigenvectors embed them for
k-means. The number of clusters comes from the eigengap; sector labels are used only to
score the result (adjusted Rand index), against a permutation null.

Run: ``python -m spectral_market.experiments.clustering``
"""

from __future__ import annotations

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.colors import LinearSegmentedColormap

from .. import spectral as sp
from ..plotting import (
    BAND,
    INK_2,
    MUTED,
    SERIES,
    SURFACE,
    end_label,
    note,
    save,
    save_results,
    use_style,
)
from ..universe import load_market

K_RANGE = range(2, 21)
N_PERMUTATIONS = 2000
# Diverging: blue (negative) - neutral gray - orange (positive).
DIVERGING = LinearSegmentedColormap.from_list("div", ["#2a78d6", "#f1f0ec", "#eb6834"])


def run(seed: int = 0) -> dict:
    m = load_market()
    X, truth = m.returns, m.sectors
    T = len(X)
    Z = sp.remove_market_mode(X)
    C_res = Z.T @ Z / T
    W_res = sp.correlation_affinity(C_res)
    W_raw = sp.correlation_affinity(sp.correlation(X))
    ev_res, _ = sp.spectral_embedding(W_res, 2)
    ev_raw, _ = sp.spectral_embedding(W_raw, 2)
    k_star = sp.eigengap_k(ev_res, max(K_RANGE))

    ari_res = np.array(
        [sp.adjusted_rand_index(sp.spectral_clustering(W_res, k), truth) for k in K_RANGE]
    )
    ari_raw = np.array(
        [sp.adjusted_rand_index(sp.spectral_clustering(W_raw, k), truth) for k in K_RANGE]
    )
    labels = sp.spectral_clustering(W_res, k_star)
    ari = sp.adjusted_rand_index(labels, truth)

    rng = np.random.default_rng(seed)
    null = np.array(
        [sp.adjusted_rand_index(rng.permutation(labels), truth) for _ in range(N_PERMUTATIONS)]
    )
    p_value = (1 + (null >= ari).sum()) / (1 + N_PERMUTATIONS)

    # Order clusters by size, stocks within a cluster by sector, for the heatmap.
    sizes = np.bincount(labels)
    rank = np.argsort(np.argsort(-sizes, kind="stable"), kind="stable")
    order = np.lexsort((np.array(m.names), truth, rank[labels]))
    clusters = []
    for c in np.argsort(-sizes, kind="stable"):
        members = np.flatnonzero(labels == c)
        sectors, counts = np.unique(truth[members], return_counts=True)
        clusters.append(
            {
                "members": [m.names[i] for i in members],
                "majority_sector": str(sectors[np.argmax(counts)]),
                "purity": float(counts.max() / counts.sum()),
            }
        )
    return {
        "C_res": C_res,
        "order": order,
        "labels": labels,
        "names": m.names,
        "ari_res": ari_res,
        "ari_raw": ari_raw,
        "null": null,
        "summary": {
            "n_sectors": len(set(truth)),
            "eigengap_k": k_star,
            "eigengap_k_without_market_removal": sp.eigengap_k(ev_raw, max(K_RANGE)),
            "laplacian_eigenvalues": ev_res[: max(K_RANGE) + 1].tolist(),
            "ari": ari,
            "ari_by_k": dict(zip(map(str, K_RANGE), ari_res.tolist(), strict=True)),
            "ari_by_k_without_market_removal": dict(
                zip(map(str, K_RANGE), ari_raw.tolist(), strict=True)
            ),
            "ari_null_97_5": float(np.percentile(null, 97.5)),
            "permutation_p_value": float(p_value),
            "clusters": clusters,
        },
    }


def plot(res: dict):
    s = res["summary"]
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(13, 5.6), gridspec_kw={"width_ratios": [1, 1.1]})

    ks = np.array(list(K_RANGE))
    ax1.axhspan(-0.1, s["ari_null_97_5"], color=BAND, lw=0, label="random labels, 97.5th pct")
    ax1.plot(
        ks,
        res["ari_res"],
        color=SERIES[0],
        marker="o",
        ms=5,
        mec="white",
        mew=0.8,
        label="market mode removed",
    )
    ax1.plot(
        ks,
        res["ari_raw"],
        color=SERIES[1],
        marker="o",
        ms=5,
        mec="white",
        mew=0.8,
        label="raw correlations",
    )
    end_label(ax1, ks[-1], res["ari_res"][-1], "removed")
    end_label(ax1, ks[-1], res["ari_raw"][-1], "raw")
    k = s["eigengap_k"]
    ax1.axvline(k, color=MUTED, lw=0.8, ls=":")
    ax1.annotate(
        f"eigengap k = {k}\nARI = {s['ari']:.2f}",
        (k, s["ari"]),
        xytext=(10, 18),
        textcoords="offset points",
        fontsize=9,
        color=INK_2,
        arrowprops={"arrowstyle": "-", "color": MUTED, "lw": 0.8},
    )
    ax1.set(
        xlabel="number of clusters k",
        ylabel="adjusted Rand index vs NSE sectors",
        ylim=(-0.05, 1),
        xticks=ks[::2],
    )
    ax1.set_title("Sectors emerge once the market mode is removed")
    ax1.legend(loc="upper left")
    note(
        ax1,
        f"{s['n_sectors']} NSE sectors; permutation p = {s['permutation_p_value']:.4f}\n"
        f"eigengap on raw correlations picks k = {s['eigengap_k_without_market_removal']}",
        loc="lower right",
    )

    o = res["order"]
    C = res["C_res"][np.ix_(o, o)]
    np.fill_diagonal(C, np.nan)
    lim = np.nanpercentile(np.abs(C), 99)
    ax2.imshow(C, cmap=DIVERGING, vmin=-lim, vmax=lim, interpolation="nearest")
    labels = res["labels"][o]
    for b in np.flatnonzero(np.diff(labels)) + 0.5:
        ax2.axhline(b, color=SURFACE, lw=1.6)
        ax2.axvline(b, color=SURFACE, lw=1.6)
    names = [res["names"][i] for i in o]
    ax2.set_xticks(range(len(o)), names, rotation=90, fontsize=6)
    ax2.set_yticks(range(len(o)), names, fontsize=6)
    ax2.grid(False)
    ax2.tick_params(length=0)
    ax2.set_title(f"Residual correlations, grouped into the {k} spectral clusters")
    sm = plt.cm.ScalarMappable(cmap=DIVERGING, norm=plt.Normalize(-lim, lim))
    fig.colorbar(sm, ax=ax2, fraction=0.04, pad=0.02, label="residual correlation")
    return fig


def main() -> None:
    use_style()
    res = run()
    path = save(plot(res), "clustering")
    save_results("clustering", res["summary"])
    s = res["summary"]
    print(f"saved {path}")
    k_raw = s["eigengap_k_without_market_removal"]
    print(f"eigengap k = {s['eigengap_k']} (raw correlations: {k_raw})")
    p = s["permutation_p_value"]
    print(f"ARI = {s['ari']:.3f}, null 97.5% = {s['ari_null_97_5']:.3f}, p = {p:.4f}")
    for c in s["clusters"]:
        print(f"  {c['majority_sector']:<12} {c['purity']:.2f}  {' '.join(c['members'])}")


if __name__ == "__main__":
    main()
