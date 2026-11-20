"""The equity universe used from Phase 2 on.

51 large-cap NSE stocks drawn from recent NIFTY 50 constituents, restricted to names
with a complete price history over 2021-2025 (TATAMOTORS, LTIM, ETERNAL and JIOFIN are
excluded for listing, delisting or demerger reasons). Sector labels are hand-assigned
following NSE's sector names and are used only to *evaluate* unsupervised methods,
never as model inputs.

Using today's constituents over a past window introduces survivorship bias; results are
reported with that caveat.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd
from numpy.typing import NDArray

from .data import DEFAULT_END, DEFAULT_START, data_dir, load_close

SECTORS: dict[str, str] = {
    # Financial services
    "HDFCBANK.NS": "Financials",
    "ICICIBANK.NS": "Financials",
    "SBIN.NS": "Financials",
    "KOTAKBANK.NS": "Financials",
    "AXISBANK.NS": "Financials",
    "INDUSINDBK.NS": "Financials",
    "BAJFINANCE.NS": "Financials",
    "BAJAJFINSV.NS": "Financials",
    "HDFCLIFE.NS": "Financials",
    "SBILIFE.NS": "Financials",
    "SHRIRAMFIN.NS": "Financials",
    # Information technology
    "TCS.NS": "IT",
    "INFY.NS": "IT",
    "HCLTECH.NS": "IT",
    "WIPRO.NS": "IT",
    "TECHM.NS": "IT",
    # Oil, gas and consumable fuels
    "RELIANCE.NS": "Energy",
    "ONGC.NS": "Energy",
    "BPCL.NS": "Energy",
    "COALINDIA.NS": "Energy",
    # Power
    "NTPC.NS": "Power",
    "POWERGRID.NS": "Power",
    # Fast-moving consumer goods
    "HINDUNILVR.NS": "FMCG",
    "ITC.NS": "FMCG",
    "NESTLEIND.NS": "FMCG",
    "BRITANNIA.NS": "FMCG",
    "TATACONSUM.NS": "FMCG",
    # Automobiles
    "MARUTI.NS": "Auto",
    "M&M.NS": "Auto",
    "BAJAJ-AUTO.NS": "Auto",
    "EICHERMOT.NS": "Auto",
    "HEROMOTOCO.NS": "Auto",
    # Healthcare
    "SUNPHARMA.NS": "Healthcare",
    "CIPLA.NS": "Healthcare",
    "DRREDDY.NS": "Healthcare",
    "DIVISLAB.NS": "Healthcare",
    "APOLLOHOSP.NS": "Healthcare",
    # Metals and mining
    "TATASTEEL.NS": "Metals",
    "JSWSTEEL.NS": "Metals",
    "HINDALCO.NS": "Metals",
    "ADANIENT.NS": "Metals",
    # Construction materials
    "ULTRACEMCO.NS": "Cement",
    "GRASIM.NS": "Cement",
    # Consumer durables and services
    "ASIANPAINT.NS": "Consumer",
    "TITAN.NS": "Consumer",
    "TRENT.NS": "Consumer",
    # Industrials, services, telecom
    "LT.NS": "Industrials",
    "BEL.NS": "Industrials",
    "ADANIPORTS.NS": "Services",
    "INDIGO.NS": "Services",
    "BHARTIARTL.NS": "Telecom",
}

TICKERS: list[str] = list(SECTORS)
INDEX = "^NSEI"


def short_name(ticker: str) -> str:
    return ticker.removesuffix(".NS")


def load_universe(
    start: str = DEFAULT_START, end: str = DEFAULT_END, refresh: bool = False
) -> pd.DataFrame:
    """Adjusted closes (dates x tickers) for the whole universe, cached as one CSV."""
    path = data_dir() / f"universe_{start}_{end}.csv"
    if path.exists() and not refresh:
        prices = pd.read_csv(path, index_col=0, parse_dates=True)
    else:
        import yfinance as yf

        raw = yf.download(
            TICKERS, start=start, end=end, interval="1d", auto_adjust=True, progress=False
        )
        prices = raw["Close"][TICKERS]
        prices.index.name = "Date"
        path.parent.mkdir(parents=True, exist_ok=True)
        prices.to_csv(path)
    missing = prices.columns[prices.isna().any()].tolist()
    if missing:
        raise ValueError(f"incomplete price history for {missing}")
    return prices[TICKERS]


def log_returns(prices: pd.DataFrame | pd.Series) -> pd.DataFrame | pd.Series:
    return np.log(prices).diff().dropna()


def sector_labels(tickers: list[str]) -> np.ndarray:
    return np.array([SECTORS[t] for t in tickers])


@dataclass(frozen=True)
class Market:
    """Aligned daily log returns of the universe and of the NIFTY 50 index."""

    dates: pd.DatetimeIndex
    returns: NDArray[np.float64]  # (T, N)
    index: NDArray[np.float64]  # (T,)
    tickers: list[str]
    names: list[str]
    sectors: NDArray[np.str_]

    def split(self, date: str) -> tuple[NDArray[np.bool_], NDArray[np.bool_]]:
        """Boolean masks for days before and on/after ``date``."""
        before = np.asarray(self.dates < pd.Timestamp(date))
        return before, ~before


def load_market(start: str = DEFAULT_START, end: str = DEFAULT_END) -> Market:
    R = log_returns(load_universe(start, end))
    idx = log_returns(load_close(INDEX, start, end))
    R, idx = R.align(idx, join="inner", axis=0)
    return Market(
        dates=R.index,
        returns=R.to_numpy(),
        index=idx.to_numpy(),
        tickers=list(R.columns),
        names=[short_name(t) for t in R.columns],
        sectors=sector_labels(list(R.columns)),
    )
