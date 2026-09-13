"""
Daily orchestrator: run the Finviz-equivalent screener, then run the
HolyGrailEngine on every ticker that passes it, and write the combined
result to docs/data/signals.json for the static dashboard to read.

This is the script the "HolyGrailDailyScan" Windows Scheduled Task runs
once a day (see scripts/daily_run.ps1).
"""
from __future__ import annotations

import argparse
import json
import math
from datetime import datetime, timezone
from pathlib import Path

import pandas as pd

from .backtest import extract_trades
from .engine import HolyGrailEngine, load_daily
from .screener import run_screener

# A closed trade this close to flat is neither a win nor a loss - usually a
# break-even stop-out, which the engine's BE ratchet produces by design.
FLAT_TRADE_PCT = 0.5
TRACK_RECORD_MONTHS = 12

OUTPUT_PATH = Path(__file__).resolve().parent.parent / "docs" / "data" / "signals.json"


def _clean(value):
    """JSON can't represent NaN — convert to None."""
    if isinstance(value, float) and math.isnan(value):
        return None
    if isinstance(value, (bool,)):
        return value
    if hasattr(value, "item"):  # numpy scalar
        value = value.item()
        return _clean(value)
    return value


def _ticker_signal(
    ticker: str, screener_row: pd.Series, engine: HolyGrailEngine, period: str
) -> tuple[dict, pd.DataFrame] | None:
    try:
        data = load_daily(ticker, period=period)
    except Exception as exc:
        print(f"  [{ticker}] failed to load price history: {exc}")
        return None
    if len(data) < 250:
        print(f"  [{ticker}] only {len(data)} bars, skipping")
        return None

    result = engine.run(data)
    row = result.iloc[-1]
    date = result.index[-1]

    # Full closed-trade history for this ticker, reusing the backtester's
    # reconstruction so the dashboard and the backtest always agree on what
    # counts as a trade and what it returned.
    trades = extract_trades(ticker, result, verbose=False)

    # If a position closed on the latest bar, surface what that trade did:
    # what it was bought at, sold at, and the resulting P/L.
    closed_today = None
    if not trades.empty:
        todays = trades[trades["exit_date"] == date]
        if not todays.empty:
            t = todays.iloc[-1]
            closed_today = {
                "direction": t["direction"],
                "entry_date": t["entry_date"].date().isoformat(),
                "entry_price": _clean(round(float(t["entry_price"]), 2)),
                "exit_price": _clean(round(float(t["exit_price"]), 2)),
                "pct_gain": _clean(round(float(t["pct_gain"]), 2)),
                "r_multiple": _clean(round(float(t["r_multiple"]), 2)),
                "exit_reason": t["exit_reason"],
                "bars_held": int(t["bars_held"]),
            }

    # "Setting up" score: how many of the bullish/bearish entry conditions
    # are already true right now, independent of whether an actual entry
    # signal (crossover/breakout/squeeze-fire/etc.) has fired yet. Only
    # meaningful for a FLAT ticker - a LONG LIVE / SHORT LIVE ticker already
    # entered, so "readiness" doesn't apply to it.
    long_conditions = {
        "weekly_bull": bool(row["weekly_bull_raw"]),
        "trend_bull": bool(row["trend_bull_raw"]),
        "macd_bull": bool(row["macd_bull_raw"]),
        "rsi_in_zone": bool(row["rsi_good_long"]),
    }
    short_conditions = {
        "weekly_bear": bool(row["weekly_bear_raw"]),
        "trend_bear": bool(row["trend_bear_raw"]),
        "macd_bear": bool(row["macd_bear_raw"]),
        "rsi_in_zone": bool(row["rsi_good_short"]),
    }

    signal = {
        "ticker": ticker,
        "as_of": date.date().isoformat(),
        "price": _clean(screener_row["price"]),
        "market_cap": _clean(screener_row["market_cap"]),
        "avg_volume": _clean(screener_row["avg_volume"]),
        "relative_volume": _clean(screener_row["relative_volume"]),
        "beta": _clean(screener_row["beta"]),
        "trend200": "BULL" if row["close"] > row["ema200"] else "BEAR",
        "weekly_warming_up": bool(row["weekly_warming_up"]),
        "weekly_state": (
            None if row["weekly_warming_up"] else ("BULL" if row["close"] > row["weekly_ema"] else "BEAR")
        ),
        "ema_cloud_state": "BULL" if row["ema_fast"] > row["ema_slow"] else "BEAR",
        "macd_state": "BULL" if row["macd_hist"] > 0 else "BEAR",
        "adx": _clean(round(row["adx"], 1)) if not math.isnan(row["adx"]) else None,
        "rsi": _clean(round(row["rsi"], 1)),
        "stoch_k": _clean(round(row["stoch_k"], 1)) if not math.isnan(row["stoch_k"]) else None,
        "bb_squeeze": bool(row["is_squeeze"]),
        "bear_divergence": bool(row["bear_div_active"]),
        "bull_divergence": bool(row["bull_div_active"]),
        "atr": _clean(round(row["atr"], 2)),
        "stop_price": _clean(round(row["stop_price"], 2)) if not math.isnan(row["stop_price"]) else None,
        "trade_status": row["trade_status"],
        "signal_count": int(row["signal_count"]),
        "entered_long_today": bool(row["enter_long"]),
        "entered_short_today": bool(row["enter_short"]),
        "exited_today": bool(row["exit_event"]) or bool(row["is_flip_long"]) or bool(row["is_flip_short"]),
        "long_readiness": sum(long_conditions.values()),
        "long_conditions": long_conditions,
        "short_readiness": sum(short_conditions.values()),
        "short_conditions": short_conditions,
        "closed_today": closed_today,
    }
    return signal, trades


def run_scan(tickers: list[str] | None = None, period: str = "3y") -> dict:
    print("Running screener...")
    screener_result = run_screener(tickers)
    passed = screener_result[screener_result["pass_all"]]
    print(f"{len(passed)} / {len(screener_result)} tickers passed the screener.")

    engine = HolyGrailEngine()
    signals = []
    all_trades = []
    print("Running Holy Grail engine on screener survivors...")
    for ticker, row in passed.iterrows():
        out = _ticker_signal(ticker, row, engine, period)
        if out is not None:
            sig, trades = out
            signals.append(sig)
            if not trades.empty:
                all_trades.append(trades)

    signals.sort(key=lambda s: (s["trade_status"] not in ("LONG LIVE", "SHORT LIVE"), s["ticker"]))

    return {
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "universe_size": len(screener_result),
        "screener_pass_count": len(passed),
        "track_record": _track_record(all_trades),
        "signals": signals,
    }


def _track_record(all_trades: list[pd.DataFrame]) -> dict:
    """
    Win / loss / flat counts over the trailing window, across every ticker
    currently on the dashboard.

    NOTE: these are the signals the engine WOULD have produced on the
    currently-screened tickers - not a record of trades actually taken.
    The screened list changes over time, so this is strategy performance,
    not a personal P/L statement. The dashboard labels it that way.
    """
    empty = {"wins": 0, "losses": 0, "flat": 0, "total": 0, "win_rate": None,
             "avg_pct": None, "months": TRACK_RECORD_MONTHS}
    if not all_trades:
        return empty

    trades = pd.concat(all_trades, ignore_index=True)
    cutoff = pd.Timestamp.now() - pd.DateOffset(months=TRACK_RECORD_MONTHS)
    trades = trades[trades["exit_date"] >= cutoff]
    if trades.empty:
        return empty

    flat = trades["pct_gain"].abs() < FLAT_TRADE_PCT
    wins = (trades["pct_gain"] >= FLAT_TRADE_PCT).sum()
    losses = (trades["pct_gain"] <= -FLAT_TRADE_PCT).sum()
    decided = wins + losses
    return {
        "wins": int(wins),
        "losses": int(losses),
        "flat": int(flat.sum()),
        "total": int(len(trades)),
        "win_rate": round(float(wins / decided * 100), 1) if decided else None,
        "avg_pct": round(float(trades["pct_gain"].mean()), 2),
        "months": TRACK_RECORD_MONTHS,
    }


def main():
    parser = argparse.ArgumentParser(description="Run the daily Holy Grail scan.")
    parser.add_argument("--tickers", nargs="*", default=None, help="Override universe (default: full market)")
    parser.add_argument("--period", default="3y", help="History window for the engine (default 3y)")
    parser.add_argument("--out", default=str(OUTPUT_PATH), help="Output JSON path")
    args = parser.parse_args()

    payload = run_scan(args.tickers, args.period)

    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(payload, indent=2))
    print(f"\nWrote {len(payload['signals'])} signals to {out_path}")


if __name__ == "__main__":
    main()
