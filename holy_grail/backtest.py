"""
Backtest the Holy Grail engine over full history for one or more tickers.

Reuses HolyGrailEngine unmodified — a "backtest" here is just running the
same bar-by-bar state machine used for live scanning over the full
downloaded history and reconstructing a trade log from its entry/exit flags.
"""
from __future__ import annotations

import argparse
import math
from pathlib import Path

import numpy as np
import pandas as pd

from .engine import HolyGrailEngine, HolyGrailParams, load_daily

OUTPUT_DIR = Path(__file__).resolve().parent.parent / "backtest_output"


def extract_trades(ticker: str, result: pd.DataFrame) -> pd.DataFrame:
    """Walk the bar-level engine output and reconstruct a closed-trade log."""
    trades = []
    open_dir = open_date = open_price = open_idx = None

    for idx, (date, row) in enumerate(result.iterrows()):
        if open_dir is not None and (row["exit_event"] or row["is_flip_long"] or row["is_flip_short"]):
            reason = (
                "hard_stop" if row["hard_stop_hit"] else
                "trail_stop" if row["trail_stop_hit"] else
                "full_exit" if row["full_exit"] else
                "time_exit" if row["time_exit"] else
                "flip"
            )
            trades.append(
                {
                    "ticker": ticker,
                    "direction": open_dir,
                    "entry_date": open_date,
                    "entry_price": open_price,
                    "exit_date": date,
                    "exit_price": row["close"],
                    "r_multiple": row["current_r"],
                    "exit_reason": reason,
                    "bars_held": idx - open_idx,
                }
            )
            open_dir = None

        if row["enter_long"]:
            open_dir, open_date, open_price, open_idx = "long", date, row["close"], idx
        elif row["enter_short"]:
            open_dir, open_date, open_price, open_idx = "short", date, row["close"], idx

    trades_df = pd.DataFrame(trades)

    if open_dir is not None:
        print(f"  [{ticker}] note: trade still open at end of data ({open_dir} from {open_date.date()}), excluded from stats")

    return trades_df


def summarize(trades: pd.DataFrame) -> dict:
    if trades.empty:
        return {"n_trades": 0}
    wins = trades[trades["r_multiple"] > 0]
    losses = trades[trades["r_multiple"] <= 0]
    equity = trades["r_multiple"].cumsum()
    running_max = equity.cummax()
    drawdown = equity - running_max
    return {
        "n_trades": len(trades),
        "win_rate": len(wins) / len(trades),
        "avg_r": trades["r_multiple"].mean(),
        "avg_win_r": wins["r_multiple"].mean() if len(wins) else float("nan"),
        "avg_loss_r": losses["r_multiple"].mean() if len(losses) else float("nan"),
        "total_r": trades["r_multiple"].sum(),
        "max_drawdown_r": drawdown.min(),
        "avg_bars_held": trades["bars_held"].mean(),
    }


def plot_equity_curve(trades: pd.DataFrame, out_path: Path) -> None:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    ordered = trades.sort_values("exit_date")
    equity = ordered["r_multiple"].cumsum()

    fig, ax = plt.subplots(figsize=(10, 5))
    ax.plot(ordered["exit_date"], equity, marker="o", markersize=3, linewidth=1)
    ax.axhline(0, color="gray", linewidth=0.8)
    ax.set_title("Holy Grail backtest — cumulative R multiple")
    ax.set_xlabel("Exit date")
    ax.set_ylabel("Cumulative R")
    ax.grid(alpha=0.3)
    fig.tight_layout()
    fig.savefig(out_path, dpi=120)
    plt.close(fig)


def run_backtest(tickers: list[str], period: str, params: HolyGrailParams | None = None) -> pd.DataFrame:
    engine = HolyGrailEngine(params)
    all_trades = []
    for ticker in tickers:
        print(f"Backtesting {ticker} ({period})...")
        try:
            data = load_daily(ticker, period=period)
        except Exception as exc:
            print(f"  [{ticker}] failed to load data: {exc}")
            continue
        if len(data) < 250:
            print(f"  [{ticker}] only {len(data)} bars of history, skipping (need 250+)")
            continue
        result = engine.run(data)
        trades = extract_trades(ticker, result)
        if not trades.empty:
            all_trades.append(trades)
    if not all_trades:
        return pd.DataFrame()
    return pd.concat(all_trades, ignore_index=True)


def main():
    parser = argparse.ArgumentParser(description="Backtest the Holy Grail engine.")
    parser.add_argument("tickers", nargs="+", help="Ticker symbols to backtest")
    parser.add_argument("--period", default="5y", help="yfinance history period (default 5y)")
    args = parser.parse_args()

    trades = run_backtest(args.tickers, args.period)

    OUTPUT_DIR.mkdir(exist_ok=True)
    if trades.empty:
        print("\nNo closed trades produced over this period.")
        return

    tag = "_".join(args.tickers[:5]) + ("_etc" if len(args.tickers) > 5 else "")
    csv_path = OUTPUT_DIR / f"trades_{tag}.csv"
    trades.to_csv(csv_path, index=False)
    print(f"\nSaved {len(trades)} trades to {csv_path}")

    print("\n=== Per-ticker summary ===")
    for ticker, group in trades.groupby("ticker"):
        s = summarize(group)
        print(
            f"{ticker:8s} n={s['n_trades']:3d}  win%={s['win_rate']*100:5.1f}  "
            f"avgR={s['avg_r']:+.2f}  totalR={s['total_r']:+.2f}  maxDD={s['max_drawdown_r']:+.2f}  "
            f"avgBars={s['avg_bars_held']:.1f}"
        )

    print("\n=== Combined ===")
    s = summarize(trades)
    print(
        f"n={s['n_trades']}  win%={s['win_rate']*100:.1f}  avgR={s['avg_r']:+.2f}  "
        f"avgWinR={s['avg_win_r']:+.2f}  avgLossR={s['avg_loss_r']:+.2f}  "
        f"totalR={s['total_r']:+.2f}  maxDD={s['max_drawdown_r']:+.2f}"
    )

    png_path = OUTPUT_DIR / f"equity_{tag}.png"
    plot_equity_curve(trades, png_path)
    print(f"\nSaved equity curve to {png_path}")


if __name__ == "__main__":
    main()
