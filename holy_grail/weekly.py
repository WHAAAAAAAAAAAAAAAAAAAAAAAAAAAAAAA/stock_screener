"""
Non-repainting weekly EMA, matching the Pine snippet:

    weeklyEma = request.security(syminfo.tickerid, "W",
                                  ta.ema(close, 50)[1], lookahead=barmerge.lookahead_on)

i.e. for any daily bar, the weekly EMA shown is the value from the last
*confirmed* (fully closed) weekly bar strictly before the current week —
it does not update until the current week closes.
"""
from __future__ import annotations

import pandas as pd

from .indicators import ema


def non_repainting_weekly_ema(daily_close: pd.Series, length: int = 50) -> pd.Series:
    weekly_close = daily_close.resample("W-FRI").last().dropna()
    weekly_ema = ema(weekly_close, length)
    confirmed = weekly_ema.shift(1)  # last CLOSED week's EMA, known as of the new week's start

    week_friday = daily_close.index.to_series().dt.to_period("W-FRI").dt.end_time.dt.normalize()
    return week_friday.map(confirmed).astype(float)
