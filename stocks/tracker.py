"""Paper trading: every live prediction is saved, then graded once the horizon has passed.
This is the real scoreboard — the backtest is a simulation, this is what actually happened."""
from __future__ import annotations

import numpy as np
import pandas as pd

from . import config, data


def grade(top_n: int = 10, min_price: float = config.MIN_PRICE,
          min_dollar_vol: float | None = None) -> pd.DataFrame:
    p = config.PRED_HISTORY_FILE
    if not p.exists():
        return pd.DataFrame()
    hist = pd.read_parquet(p)
    if hist.empty:
        return hist
    hist = hist[hist["close"] >= min_price]
    out = []
    daily = None
    hourly = None
    for h, g in hist.groupby("horizon"):
        n, kind = config.HORIZONS[h][1], config.HORIZONS[h][2]
        # the model's top-N each way per prediction date
        g = g.sort_values("p_up")
        picks = pd.concat([g.groupby("date").head(top_n).assign(side="DOWN"),
                           g.groupby("date").tail(top_n).assign(side="UP")])
        if kind == "daily":
            if daily is None:
                daily = data.load_daily(tickers=hist["ticker"].unique().tolist(),
                                        start=str(hist["date"].min().date()))
            px = daily
        else:
            if hourly is None:
                hourly = data.load_hourly(tickers=hist["ticker"].unique().tolist())
            px = hourly
        if px.empty:
            continue
        px = px.sort_values(["ticker", "date"])
        px["fwd_close"] = px.groupby("ticker")["close"].shift(-n)
        px["fwd_date"] = px.groupby("ticker")["date"].shift(-n)
        m = picks.merge(px[["ticker", "date", "fwd_close", "fwd_date"]], on=["ticker", "date"], how="left")
        out.append(m)
    if not out:
        return pd.DataFrame()
    df = pd.concat(out, ignore_index=True)
    df["actual_ret_%"] = 100 * (df["fwd_close"] / df["close"] - 1)
    df["graded"] = df["fwd_close"].notna()
    right = np.where(df["side"] == "UP", df["actual_ret_%"] > 0, df["actual_ret_%"] < 0).astype(float)
    df["right"] = np.where(df["graded"], right, np.nan)
    df["side_ret_%"] = np.where(df["side"] == "UP", 1, -1) * df["actual_ret_%"]
    df.to_parquet(config.TRACKER_FILE, index=False)
    return df


def summary(df: pd.DataFrame) -> pd.DataFrame:
    g = df[df["graded"]]
    if g.empty:
        return pd.DataFrame()
    return g.groupby(["horizon", "side"]).agg(picks=("right", "size"), hit_rate=("right", "mean"),
                                              avg_return_pct=("side_ret_%", "mean")).reset_index()
