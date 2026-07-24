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
"""
from __future__ import annotations

import time
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass

import numpy as np
import pandas as pd
import yfinance as yf

from . import indicators as ta
from .universe import get_sp500_tickers

SPY = "SPY"


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
    """One batched yfinance call for the whole universe (+SPY), not N calls."""
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


def _fetch_market_caps(tickers: list[str], max_workers: int = 8, retries: int = 3) -> dict[str, float]:
    def _one(t: str):
        for attempt in range(retries):
            try:
                cap = yf.Ticker(t).fast_info.get("marketCap")
                return t, cap
            except Exception:
                time.sleep(2 * (attempt + 1))
        return t, None

    caps: dict[str, float] = {}
    with ThreadPoolExecutor(max_workers=max_workers) as pool:
        for t, cap in pool.map(_one, tickers):
            if cap:
                caps[t] = float(cap)
    return caps


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
    tickers = tickers or get_sp500_tickers()

    print(f"Downloading history for {len(tickers)} tickers + SPY...")
    history = _bulk_history(tickers)
    if SPY not in history:
        raise RuntimeError("Failed to download SPY history (needed for beta) - aborting screener run.")
    spy_returns = history[SPY]["close"].pct_change()

    print(f"Fetching market cap for {len(tickers)} tickers...")
    market_caps = _fetch_market_caps([t for t in tickers if t in history])

    rows = []
    for t in tickers:
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

    result = pd.DataFrame(rows).set_index("ticker")
    return result.sort_values("pass_all", ascending=False)


def main():
    import argparse

    parser = argparse.ArgumentParser(description="Run the Finviz-equivalent screener.")
    parser.add_argument("--tickers", nargs="*", default=None, help="Override universe (default: S&P 500)")
    args = parser.parse_args()

    result = run_screener(args.tickers)
    passed = result[result["pass_all"]]
    print(f"\n{len(passed)} / {len(result)} tickers pass all filters:\n")
    if len(passed):
        cols = ["price", "market_cap", "avg_volume", "relative_volume", "beta", "rsi"]
        print(passed[cols].to_string(float_format=lambda x: f"{x:,.2f}"))


if __name__ == "__main__":
    main()
