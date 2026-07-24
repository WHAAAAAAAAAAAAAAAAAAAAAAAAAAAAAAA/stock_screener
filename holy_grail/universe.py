"""
Ticker universe for the screener. Starts as the S&P 500 constituent list
(large/mid-cap, liquid — a natural fit for the "holy grail" screen's own
market-cap and volume floors) scraped once from Wikipedia and cached
locally. Edit CACHE_PATH's file directly, or pass your own ticker list to
screener.run_screener(), to widen/narrow this later.
"""
from __future__ import annotations

import time
from pathlib import Path

import pandas as pd

CACHE_PATH = Path(__file__).resolve().parent.parent / "universe_sp500.csv"
CACHE_MAX_AGE_DAYS = 30
# Community-maintained mirror of the Wikipedia S&P 500 constituents table,
# updated on the same schedule. Used instead of scraping Wikipedia directly
# since plain HTTPS GETs to Wikipedia are unreliable from some sandboxed/
# corporate networks; raw.githubusercontent.com is broadly reachable.
SOURCE_URL = "https://raw.githubusercontent.com/datasets/s-and-p-500-companies/main/data/constituents.csv"


def _normalize_ticker(ticker: str) -> str:
    # yfinance wants BRK-B, the source list has BRK.B
    return ticker.strip().replace(".", "-")


def fetch_sp500_tickers() -> list[str]:
    constituents = pd.read_csv(SOURCE_URL)
    tickers = [_normalize_ticker(t) for t in constituents["Symbol"].tolist()]
    return sorted(set(tickers))


def get_sp500_tickers(force_refresh: bool = False) -> list[str]:
    if not force_refresh and CACHE_PATH.exists():
        age_days = (time.time() - CACHE_PATH.stat().st_mtime) / 86400
        if age_days <= CACHE_MAX_AGE_DAYS:
            return pd.read_csv(CACHE_PATH)["ticker"].tolist()

    tickers = fetch_sp500_tickers()
    pd.DataFrame({"ticker": tickers}).to_csv(CACHE_PATH, index=False)
    return tickers


if __name__ == "__main__":
    tks = get_sp500_tickers(force_refresh=True)
    print(f"{len(tks)} tickers cached to {CACHE_PATH}")
