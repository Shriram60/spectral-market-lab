"""Price data with a fixed date range and an on-disk cache, so results are reproducible.

The original script used "the last 8 trading days", which meant every run produced
different numbers. Here every dataset is pinned to explicit dates and cached as CSV
under ``data/`` (override with the ``SPECTRAL_MARKET_DATA`` environment variable).
"""

from __future__ import annotations

import os
from pathlib import Path

import pandas as pd

DEFAULT_START = "2021-01-01"
DEFAULT_END = "2026-01-01"  # exclusive, as in yfinance

# The original project's stock, a broad index, and large caps from different sectors.
PHASE1_TICKERS = [
    "MUTHOOTFIN.NS",
    "^NSEI",
    "RELIANCE.NS",
    "TCS.NS",
    "HDFCBANK.NS",
    "INFY.NS",
    "ITC.NS",
]


def data_dir() -> Path:
    return Path(os.environ.get("SPECTRAL_MARKET_DATA", "data"))


def _cache_path(ticker: str, start: str, end: str) -> Path:
    safe = ticker.replace("^", "IDX_").replace("/", "_")
    return data_dir() / f"{safe}_{start}_{end}.csv"


def load_close(
    ticker: str, start: str = DEFAULT_START, end: str = DEFAULT_END, refresh: bool = False
) -> pd.Series:
    """Daily adjusted close for ``ticker`` in ``[start, end)``, cached on first download."""
    path = _cache_path(ticker, start, end)
    if path.exists() and not refresh:
        return pd.read_csv(path, index_col=0, parse_dates=True).iloc[:, 0].rename(ticker)

    import yfinance as yf  # imported lazily so tests and cached runs never need the network

    raw = yf.download(ticker, start=start, end=end, interval="1d", auto_adjust=True, progress=False)
    if raw is None or raw.empty:
        raise RuntimeError(f"no data returned for {ticker} between {start} and {end}")
    close = raw["Close"]
    if isinstance(close, pd.DataFrame):  # yfinance returns a column MultiIndex
        close = close.iloc[:, 0]
    close = close.dropna().rename(ticker)
    close.index.name = "Date"
    path.parent.mkdir(parents=True, exist_ok=True)
    close.to_csv(path)
    return close
