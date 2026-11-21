"""Bundle the experiment outputs into ``site/data.js`` for the project website.

Summary numbers come straight from ``results/*.json``. The few time series the site
plots (eigenvalues, the stat-arb equity curve, HMM regimes, Kalman slope tracks) are not
stored there, so the relevant experiments are re-run from the cached data.

Run from the repository root: ``python site/build_data.py``
"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np

from spectral_market import spectral as sp
from spectral_market.experiments import kalman_trend, regimes, spectrum, stat_arb

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "site" / "data.js"


def clean(o):
    """Make an object JSON-safe: arrays to lists, NaN to None, floats rounded."""
    if isinstance(o, dict):
        return {str(k): clean(v) for k, v in o.items()}
    if isinstance(o, list | tuple):
        return [clean(v) for v in o]
    if isinstance(o, np.ndarray):
        return clean(o.tolist())
    if isinstance(o, np.integer | np.bool_):
        return o.item()
    if isinstance(o, float | np.floating):
        o = float(o)
        return None if not np.isfinite(o) else float(f"{o:.6g}")
    return o


def dates(idx) -> list[str]:
    return [str(d.date()) for d in idx]


def main() -> None:
    data = {
        name.stem: json.loads(name.read_text().replace("NaN", "null"))
        for name in sorted((ROOT / "results").glob("*.json"))
    }

    s = spectrum.run()
    x = np.linspace(0.01, 3.2, 200)
    data["spectrum_series"] = {
        "eigenvalues": s["eigenvalues"],
        "null_max": s["null"][:, 0],
        "mp_x": x,
        "mp_pdf": sp.marchenko_pastur_pdf(x, s["q"]),
        "mp_pdf_adjusted": sp.marchenko_pastur_pdf(x, s["q"], s["sigma2"]),
    }

    a = stat_arb.run()
    curves = {str(c): np.cumprod(1 + a["res"].net(c)) for c in stat_arb.COSTS_BPS}
    m = regimes.load_market()
    nifty = np.exp(np.cumsum(m.index[a["params"].pca_window + 1 :]))
    data["stat_arb_series"] = {"dates": dates(a["dates"]), "equity": curves, "nifty": nifty}

    r = regimes.run()
    data["regimes_series"] = {
        "dates": dates(r["dates"]),
        "nifty": np.exp(np.cumsum(r["index"])),
        "turbulent": (r["path"] == r["model"].k - 1).astype(int),
        "lam1": r["lam1"],
    }

    k = kalman_trend.run()
    ex = k["series"][kalman_trend.EXAMPLE]
    skip = kalman_trend.SKIP
    data["kalman_series"] = {
        "dates": dates(ex["dates"][skip:]),
        "tracks": {
            name: {"slope": m_[skip:] * 252, "se": sd[skip:] * 252}
            for name, (m_, sd, _) in k["tracks"].items()
        },
    }

    payload = json.dumps(clean(data), separators=(",", ":"))
    OUT.write_text(f"window.SML = {payload};\n")
    print(f"wrote {OUT} ({OUT.stat().st_size / 1e3:.0f} kB)")


if __name__ == "__main__":
    main()
