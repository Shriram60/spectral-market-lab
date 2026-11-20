"""Eigenvalue spectrum of the NSE correlation matrix against random matrix theory.

51 stocks x ~1235 daily log returns (2021-2025). The correlation spectrum is compared
with the Marchenko-Pastur law for pure noise in two ways: analytically, and empirically
by shuffling each stock's returns in time (keeps fat tails, destroys co-movement).
Eigenvalues above the noise edge are interpreted through their eigenportfolios.

Run: ``python -m spectral_market.experiments.spectrum``
"""

from __future__ import annotations

import matplotlib.pyplot as plt
import numpy as np

from .. import spectral as sp
from ..plotting import BAND, INK_2, MUTED, SERIES, note, save, save_results, use_style
from ..universe import load_market

N_SHUFFLES = 20


def run(seed: int = 0) -> dict:
    m = load_market()
    X, T, N = m.returns, *m.returns.shape
    q = N / T
    pca = sp.pca_correlation(X)
    lam = pca.eigenvalues
    sigma2 = sp.market_adjusted_variance(lam)
    rng = np.random.default_rng(seed)
    null = np.array(
        [sp.pca_correlation(sp.shuffled(X, rng)).eigenvalues for _ in range(N_SHUFFLES)]
    )  # (N_SHUFFLES, N), each row descending
    w1 = sp.eigenportfolio_weights(pca.eigenvectors[:, 0], X.std(axis=0))
    pc1 = X @ w1
    loadings = {}
    for k in range(1, 4):
        v = pca.eigenvectors[:, k]
        order = np.argsort(v)
        loadings[f"pc{k + 1}"] = {
            "negative": [m.names[i] for i in order[:5]],
            "positive": [m.names[i] for i in order[::-1][:5]],
        }
    return {
        "X": X,
        "dates": m.dates,
        "index": m.index,
        "pc1_returns": pc1,
        "eigenvalues": lam,
        "null": null,
        "q": q,
        "sigma2": sigma2,
        "ipr": sp.inverse_participation_ratio(pca.eigenvectors),
        "summary": {
            "n_stocks": N,
            "n_days": T,
            "q": q,
            "top_eigenvalues": lam[:6].tolist(),
            "explained_top": pca.explained[:3].tolist(),
            "mp_edge": sp.marchenko_pastur_bounds(q)[1],
            "mp_edge_market_adjusted": sp.marchenko_pastur_bounds(q, sigma2)[1],
            "n_signal": sp.n_signal(lam, q),
            "n_signal_market_adjusted": sp.n_signal(lam, q, sigma2),
            "shuffled_max_eigenvalue": float(null.max()),
            "pc1_index_correlation": float(np.corrcoef(pc1, m.index)[0, 1]),
            "pc1_ipr_times_n": float(sp.inverse_participation_ratio(pca.eigenvectors)[0] * N),
            "loadings": loadings,
        },
    }


def plot(res: dict):
    lam, q, s2, null = res["eigenvalues"], res["q"], res["sigma2"], res["null"]
    fig, (ax1, ax2) = plt.subplots(
        1, 2, figsize=(12.5, 4.6), gridspec_kw={"width_ratios": [1.25, 1]}
    )

    # Left: eigenvalues by rank against the noise null and the M-P edges.
    rank = np.arange(1, lam.size + 1)
    lo, hi = np.percentile(null, [2.5, 97.5], axis=0)
    ax1.fill_between(rank, lo, hi, color=BAND, lw=0, label="Shuffled returns, 95% band")
    edge = sp.marchenko_pastur_bounds(q)[1]
    edge_adj = sp.marchenko_pastur_bounds(q, s2)[1]
    ax1.axhline(edge, color=INK_2, lw=1.0, label=f"Marchenko-Pastur edge ({edge:.2f})")
    ax1.axhline(
        edge_adj, color=MUTED, lw=1.0, ls="--", label=f"market-adjusted edge ({edge_adj:.2f})"
    )
    signal = lam > edge
    ax1.plot(rank[~signal], lam[~signal], "o", color=SERIES[0], ms=4.5, mec="white", mew=0.6)
    ax1.plot(
        rank[signal],
        lam[signal],
        "o",
        color=SERIES[1],
        ms=7,
        mec="white",
        mew=0.8,
        label="NSE eigenvalues (above edge highlighted)",
    )
    for r, v, text in zip(
        rank[:4],
        lam[:4],
        ["market", "PSU vs IT/FMCG", "IT vs FMCG", "banks vs pharma"],
        strict=True,
    ):
        ax1.annotate(
            f"{text}",
            (r, v),
            xytext=(8, 0),
            textcoords="offset points",
            va="center",
            fontsize=9,
            color=INK_2,
        )
    ax1.set(yscale="log", xlim=(0, lam.size + 1), xlabel="rank", ylabel=r"eigenvalue $\lambda$")
    ax1.set_title("Five eigenvalues carry structure; the rest is noise")
    ax1.legend(loc="upper right")

    # Right: the market mode is the index.
    dates = res["dates"]
    pc1 = res["pc1_returns"]
    idx = res["index"]
    scale = idx.std() / pc1.std()
    ax2.plot(dates, np.exp(np.cumsum(idx)), color=SERIES[0], label="NIFTY 50")
    ax2.plot(
        dates,
        np.exp(np.cumsum(pc1 * scale)),
        color=SERIES[1],
        lw=1.4,
        label="Top eigenportfolio (vol-matched)",
    )
    ax2.set(ylabel="growth of 1")
    ax2.set_title("The top eigenvector is the market")
    ax2.legend(loc="upper left")
    s = res["summary"]
    note(
        ax2,
        f"daily correlation {s['pc1_index_correlation']:.3f}\n"
        f"explains {100 * s['explained_top'][0]:.0f}% of variance",
        loc="lower right",
    )
    plt.setp(ax2.get_xticklabels(), rotation=0)
    return fig


def main() -> None:
    use_style()
    res = run()
    path = save(plot(res), "spectrum")
    save_results("spectrum", res["summary"])
    s = res["summary"]
    print(f"saved {path}")
    print(f"N = {s['n_stocks']}, T = {s['n_days']}, q = {s['q']:.4f}")
    print(f"top eigenvalues {np.round(s['top_eigenvalues'], 2)}")
    print(f"signal eigenvalues: {s['n_signal']} (M-P), {s['n_signal_market_adjusted']} (adjusted)")
    print(f"largest shuffled eigenvalue {s['shuffled_max_eigenvalue']:.2f}")
    print(f"PC1 vs NIFTY 50 correlation {s['pc1_index_correlation']:.3f}")
    for k, v in s["loadings"].items():
        print(f"  {k}: {' '.join(v['negative'])}  <->  {' '.join(v['positive'])}")


if __name__ == "__main__":
    main()
