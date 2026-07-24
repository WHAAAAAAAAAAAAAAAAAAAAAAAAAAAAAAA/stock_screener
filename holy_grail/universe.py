"""
Ticker universes for the screener.

- get_sp500_tickers(): S&P 500 constituents (kept for quick/small test runs).
- get_full_market_tickers(): every NASDAQ/NYSE/NYSE American/ARCA-listed
  common stock, sourced from NASDAQ Trader's own symbol directory (the
  closest free equivalent to "all US-listed stocks", which is what a real
  Finviz screen without an index restriction scans against). ETFs, test
  issues, and non-common-stock instrument types (warrants, units, rights,
  preferreds, notes, closed-end funds/trusts) are filtered out heuristically.

Both are cached locally so we're not re-fetching on every run.
"""
from __future__ import annotations

import time
from io import StringIO
from pathlib import Path

import pandas as pd

_ROOT = Path(__file__).resolve().parent.parent

SP500_CACHE_PATH = _ROOT / "universe_sp500.csv"
FULL_MARKET_CACHE_PATH = _ROOT / "universe_full_market.csv"
CACHE_MAX_AGE_DAYS = 30

# Community-maintained mirror of the Wikipedia S&P 500 constituents table,
# updated on the same schedule. Used instead of scraping Wikipedia directly
# since plain HTTPS GETs to Wikipedia are unreliable from some sandboxed/
# corporate networks; raw.githubusercontent.com is broadly reachable.
SP500_SOURCE_URL = "https://raw.githubusercontent.com/datasets/s-and-p-500-companies/main/data/constituents.csv"

# Official NASDAQ Trader symbol directory: nasdaqlisted.txt covers NASDAQ,
# otherlisted.txt covers NYSE/NYSE American/ARCA/BATS/IEX etc.
NASDAQ_LISTED_URL = "https://www.nasdaqtrader.com/dynamic/SymDir/nasdaqlisted.txt"
OTHER_LISTED_URL = "https://www.nasdaqtrader.com/dynamic/SymDir/otherlisted.txt"

_EXCLUDE_NAME_KEYWORDS = (
    "ETF", "ETN", "EXCHANGE TRADED", "FUND", "TRUST", "WARRANT", "WARRANTS",
    " RIGHT", "RIGHTS", " UNIT", "UNITS", "PREFERRED", " PFD", "DEPOSITARY",
    "DEPOSITARY SHARES", "NOTES", "NOTE ", "DEBENTURE", " BOND",
)


def _normalize_ticker(ticker: str) -> str:
    # yfinance wants BRK-B, source lists use BRK.B / BRK/A style separators
    return ticker.strip().replace(".", "-").replace("/", "-")


def _cached(path: Path, fetch_fn, force_refresh: bool) -> list[str]:
    if not force_refresh and path.exists():
        age_days = (time.time() - path.stat().st_mtime) / 86400
        if age_days <= CACHE_MAX_AGE_DAYS:
            return pd.read_csv(path)["ticker"].tolist()

    tickers = fetch_fn()
    pd.DataFrame({"ticker": tickers}).to_csv(path, index=False)
    return tickers


def fetch_sp500_tickers() -> list[str]:
    constituents = pd.read_csv(SP500_SOURCE_URL)
    tickers = [_normalize_ticker(t) for t in constituents["Symbol"].tolist()]
    return sorted(set(tickers))


def get_sp500_tickers(force_refresh: bool = False) -> list[str]:
    return _cached(SP500_CACHE_PATH, fetch_sp500_tickers, force_refresh)


def _looks_like_common_stock(name: str) -> bool:
    upper = f" {name.upper()} "
    return not any(kw in upper for kw in _EXCLUDE_NAME_KEYWORDS)


def _fetch_nasdaq_listed() -> pd.DataFrame:
    import urllib.request

    raw = urllib.request.urlopen(NASDAQ_LISTED_URL, timeout=30).read().decode("utf-8")
    df = pd.read_csv(StringIO(raw), sep="|")
    df = df[df["Test Issue"] == "N"]
    df = df[df["ETF"] == "N"]
    df = df[df["NextShares"] == "N"] if "NextShares" in df.columns else df
    return df[["Symbol", "Security Name"]].rename(columns={"Symbol": "symbol", "Security Name": "name"})


def _fetch_other_listed() -> pd.DataFrame:
    import urllib.request

    raw = urllib.request.urlopen(OTHER_LISTED_URL, timeout=30).read().decode("utf-8")
    df = pd.read_csv(StringIO(raw), sep="|")
    df = df[df["Test Issue"] == "N"]
    df = df[df["ETF"] == "N"]
    return df[["ACT Symbol", "Security Name"]].rename(columns={"ACT Symbol": "symbol", "Security Name": "name"})


def fetch_full_market_tickers() -> list[str]:
    nasdaq = _fetch_nasdaq_listed()
    other = _fetch_other_listed()
    combined = pd.concat([nasdaq, other], ignore_index=True).dropna(subset=["symbol", "name"])
    combined = combined[combined["name"].apply(_looks_like_common_stock)]
    tickers = [_normalize_ticker(s) for s in combined["symbol"].tolist()]
    # drop obviously malformed symbols (file footer lines, etc.)
    tickers = [t for t in tickers if t and t.replace("-", "").isalnum() and len(t) <= 6]
    return sorted(set(tickers))


def get_full_market_tickers(force_refresh: bool = False) -> list[str]:
    return _cached(FULL_MARKET_CACHE_PATH, fetch_full_market_tickers, force_refresh)


if __name__ == "__main__":
    import sys

    if "--full" in sys.argv:
        tks = get_full_market_tickers(force_refresh=True)
        print(f"{len(tks)} tickers cached to {FULL_MARKET_CACHE_PATH}")
    else:
        tks = get_sp500_tickers(force_refresh=True)
        print(f"{len(tks)} tickers cached to {SP500_CACHE_PATH}")
