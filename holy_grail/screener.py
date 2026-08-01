"""
Python replica of your saved Finviz "holy grail" screen (all 8 filters from
the saved screen, matched exactly):

    Market Cap        +Large (over $10B)
    Price             over $10
    Average Volume    500K to 1M
    Relative Volume   over 0.5
    Beta              under 2
    50-Day SMA        price above SMA50
    200-Day SMA       price above SMA200
    RSI (14)          not oversold (>40)

No Finviz API needed (free tier has none) - computed from bulk yfinance
OHLCV history plus a lightweight per-ticker market-cap lookup.

Market cap is fetched FIRST for the whole universe (cheap, one field, and
cached to disk for a week) so the expensive part - full price history +
technicals - only runs on tickers that already clear the market-cap floor.
That's what makes scanning the full market (thousands of tickers) practical
instead of just the S&P 500.
"""
from __future__ import annotations

import math
import time
import zlib
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import pandas as pd
import yfinance as yf

from . import indicators as ta
from .universe import get_full_market_tickers

SPY = "SPY"
MARKET_CAP_CACHE_PATH = Path(__file__).resolve().parent.parent / "market_cap_cache.csv"
MARKET_CAP_CACHE_MAX_AGE_DAYS = 7


@dataclass
class ScreenerParams:
    min_market_cap: float = 10e9
    min_price: float = 10.0
    avg_vol_min: float = 500_000
    avg_vol_max: float = 1_000_000
    min_relative_volume: float = 0.5
    max_beta: float = 2.0
    require_above_sma50: bool = True
    require_above_sma200: bool = True
    min_rsi: float = 40.0


def _bulk_history(tickers: list[str], period: str = "18mo", retries: int = 3) -> dict[str, pd.DataFrame]:
    """One batched yfinance call for the whole (already narrowed) universe (+SPY)."""
    all_tickers = list(dict.fromkeys(tickers + [SPY]))

    raw = None
    for attempt in range(retries):
        raw = yf.download(
            all_tickers, period=period, interval="1d", auto_adjust=False,
            group_by="ticker", threads=True, progress=False,
        )
        if raw is not None and not raw.empty and SPY in raw.columns.get_level_values(0):
            break
        wait = 30 * (attempt + 1)
        print(f"  bulk history download looked empty/rate-limited, retrying in {wait}s...")
        time.sleep(wait)

    out = {}
    for t in all_tickers:
        try:
            df = raw[t].dropna(how="all")
        except KeyError:
            continue
        if df.empty:
            continue
        out[t] = df.rename(columns=str.lower)
    return out


def _load_market_cap_cache() -> dict[str, tuple[float, str]]:
    if not MARKET_CAP_CACHE_PATH.exists():
        return {}
    df = pd.read_csv(MARKET_CAP_CACHE_PATH)
    return {row.ticker: (row.market_cap, row.fetched_at) for row in df.itertuples()}


def _save_market_cap_cache(cache: dict[str, tuple[float, str]]) -> None:
    df = pd.DataFrame(
        [{"ticker": t, "market_cap": cap, "fetched_at": fetched_at} for t, (cap, fetched_at) in cache.items()]
    )
    df.to_csv(MARKET_CAP_CACHE_PATH, index=False)


def _staleness_jitter_days(ticker: str, spread_days: int = 4) -> int:
    """
    Deterministic per-ticker jitter (0..spread_days-1) added to the cache
    staleness window. Without this, a big batch fetched all at once (e.g.
    the first full-market run) all expires at the same moment too, forcing
    a full slow refetch of the entire universe in one shot every ~7 days
    instead of a small rolling top-up. Uses crc32, not Python's built-in
    hash(), since str hashing is randomized per-process by default.
    """
    return zlib.crc32(ticker.encode()) % spread_days


def _fetch_market_caps_network(
    tickers: list[str], cache: dict[str, tuple[float, str]], max_workers: int = 12, retries: int = 3,
    save_every: int = 500,
) -> dict[str, float]:
    def _one(t: str):
        for attempt in range(retries):
            try:
                cap = yf.Ticker(t).fast_info.get("marketCap")
                return t, cap
            except Exception:
                time.sleep(2 * (attempt + 1))
        return t, None

    caps: dict[str, float] = {}
    done = 0
    with ThreadPoolExecutor(max_workers=max_workers) as pool:
        for t, cap in pool.map(_one, tickers):
            done += 1
            if cap:
                caps[t] = float(cap)
                cache[t] = (float(cap), datetime.now(timezone.utc).isoformat())
            if done % save_every == 0:
                print(f"  market cap fetch progress: {done} / {len(tickers)}")
                _save_market_cap_cache(cache)  # incremental checkpoint - survives an interrupted run
    return caps


def get_market_caps(
    tickers: list[str], cache_max_age_days: int = MARKET_CAP_CACHE_MAX_AGE_DAYS, max_workers: int = 12
) -> dict[str, float]:
    cache = _load_market_cap_cache()
    now = datetime.now(timezone.utc)

    fresh: dict[str, float] = {}
    to_fetch: list[str] = []
    for t in tickers:
        cached = cache.get(t)
        if cached is not None:
            cap, fetched_at = cached
            if not (isinstance(cap, float) and math.isnan(cap)):
                effective_max_age = cache_max_age_days + _staleness_jitter_days(t)
                age_days = (now - datetime.fromisoformat(fetched_at)).days
                if age_days <= effective_max_age:
                    fresh[t] = cap
                    continue
        to_fetch.append(t)

    print(f"Market cap: {len(fresh)} from cache, fetching {len(to_fetch)} fresh...")
    if to_fetch:
        newly_fetched = _fetch_market_caps_network(to_fetch, cache, max_workers=max_workers)
        _save_market_cap_cache(cache)  # final save, in case the last batch was smaller than save_every
        fresh.update(newly_fetched)

    return fresh


def _beta(returns: pd.Series, spy_returns: pd.Series) -> float:
    aligned = pd.concat([returns, spy_returns], axis=1, join="inner").dropna()
    if len(aligned) < 30:
        return float("nan")
    r, m = aligned.iloc[:, 0], aligned.iloc[:, 1]
    var_m = m.var(ddof=0)
    if var_m == 0 or np.isnan(var_m):
        return float("nan")
    return float(r.cov(m) / var_m)


def run_screener(tickers: list[str] | None = None, params: ScreenerParams | None = None) -> pd.DataFrame:
    p = params or ScreenerParams()
    tickers = tickers or get_full_market_tickers()

    print(f"Screening {len(tickers)} tickers. Checking market cap first (cheap + cached)...")
    market_caps = get_market_caps(tickers)
    qualified = [t for t, cap in market_caps.items() if cap >= p.min_market_cap]
    print(f"{len(qualified)} / {len(tickers)} clear the ${p.min_market_cap/1e9:.0f}B market-cap floor.")

    print(f"Downloading price history for {len(qualified)} tickers + SPY...")
    history = _bulk_history(qualified)
    if SPY not in history:
        raise RuntimeError("Failed to download SPY history (needed for beta) - aborting screener run.")
    spy_returns = history[SPY]["close"].pct_change()

    rows = []
    for t in qualified:
        df = history.get(t)
        if df is None or len(df) < 200:
            continue

        close, volume = df["close"], df["volume"]
        price = float(close.iloc[-1])
        sma50 = float(ta.sma(close, 50).iloc[-1])
        sma200 = float(ta.sma(close, 200).iloc[-1])
        rsi14 = float(ta.rsi(close, 14).iloc[-1])
        avg_vol = float(volume.tail(63).mean())
        rel_vol = float(volume.iloc[-1] / avg_vol) if avg_vol else float("nan")
        beta = _beta(close.pct_change(), spy_returns)
        market_cap = market_caps.get(t, float("nan"))

        checks = {
            "market_cap": market_cap >= p.min_market_cap if not np.isnan(market_cap) else False,
            "price": price >= p.min_price,
            "avg_volume": p.avg_vol_min <= avg_vol <= p.avg_vol_max,
            "relative_volume": rel_vol >= p.min_relative_volume if not np.isnan(rel_vol) else False,
            "beta": beta <= p.max_beta if not np.isnan(beta) else False,
            "above_sma50": (price > sma50) if p.require_above_sma50 else True,
            "above_sma200": (price > sma200) if p.require_above_sma200 else True,
            "rsi": rsi14 >= p.min_rsi if not np.isnan(rsi14) else False,
        }

        rows.append(
            {
                "ticker": t,
                "price": price,
                "market_cap": market_cap,
                "avg_volume": avg_vol,
                "relative_volume": rel_vol,
                "beta": beta,
                "sma50": sma50,
                "sma200": sma200,
                "rsi": rsi14,
                **{f"pass_{k}": v for k, v in checks.items()},
                "pass_all": all(checks.values()),
            }
        )

    result = pd.DataFrame(rows).set_index("ticker") if rows else pd.DataFrame(
        columns=["price", "market_cap", "avg_volume", "relative_volume", "beta", "sma50", "sma200", "rsi", "pass_all"]
    )
    if "pass_all" in result.columns:
        result = result.sort_values("pass_all", ascending=False)
    return result


def main():
    import argparse

    parser = argparse.ArgumentParser(description="Run the Finviz-equivalent screener.")
    parser.add_argument("--tickers", nargs="*", default=None, help="Override universe (default: full market)")
    args = parser.parse_args()

    result = run_screener(args.tickers)
    passed = result[result["pass_all"]]
    print(f"\n{len(passed)} / {len(result)} tickers pass all filters:\n")
    if len(passed):
        cols = ["price", "market_cap", "avg_volume", "relative_volume", "beta", "rsi"]
        print(passed[cols].to_string(float_format=lambda x: f"{x:,.2f}"))


if __name__ == "__main__":
    main()
