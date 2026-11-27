"""Model features. Every feature for day t uses only what was known at the close of day t
(economic numbers only after their release date), so backtests can't peek at the future.

Feature names start with their group, which the app uses to explain picks:
  mom_   momentum (past returns, relative strength)
  risk_  volatility and beta
  trend_ moving averages, RSI, 52-week high/low
  liq_   volume and liquidity
  evt_   events (earnings, elections, wars, CEO changes, launches, your own types)
  mkt_   market and cross-asset series (S&P, VIX, oil, gold, bitcoin, sectors, ...)
  eco_   economy (FRED: inflation, jobs, rates, credit spreads, ...)
  cus_   your own series from data/custom_series/
  cal_   calendar (day of week, month, turn of month)
  intra_ intraday (hourly model only)
"""
from __future__ import annotations

from typing import Iterable

import numpy as np
import pandas as pd

from . import config

GROUP_NAMES = {
    "mom": "Momentum", "risk": "Volatility & beta", "trend": "Trend (moving averages, RSI)",
    "liq": "Volume & liquidity", "evt": "Events", "mkt": "Market & cross-asset",
    "eco": "Economy", "cus": "Your custom series", "cal": "Calendar", "intra": "Intraday",
    "sec": "Sector & peers", "ins": "Insider trading",
}
DAILY_HORIZONS = {h: v[1] for h, v in config.HORIZONS.items() if v[2] == "daily"}
F32 = np.float32


def group_of(feature: str) -> str:
    return feature.split("_", 1)[0]


# ======================================================================= stock features
def _stock_block(df: pd.DataFrame, spy_r: pd.Series | None, horizons: dict) -> pd.DataFrame:
    """Features for a chunk of tickers. df: long rows sorted by ticker, date."""
    g = df.groupby("ticker", sort=False)
    c = df["close"].astype("float64")
    lc = np.log(c.where(c > 0))
    r1 = g["close"].transform(lambda s: np.log(s.where(s > 0)).diff())
    out = pd.DataFrame({"date": df["date"].values, "ticker": df["ticker"].values}, index=df.index)

    def lag_ret(n):
        return lc - lc.groupby(df["ticker"]).shift(n)

    for n in (1, 5, 21, 63, 126, 252):
        out[f"mom_ret_{n}"] = lag_ret(n)
    out["mom_12_1"] = out["mom_ret_252"] - out["mom_ret_21"]

    def roll(s, n, fn, minp=None):
        r = getattr(s.groupby(df["ticker"].values, sort=False).rolling(n, min_periods=minp or max(2, n // 2)), fn)()
        return r.reset_index(level=0, drop=True).reindex(s.index)

    out["risk_vol_21"] = roll(r1, 21, "std") * np.sqrt(252)
    out["risk_vol_63"] = roll(r1, 63, "std") * np.sqrt(252)
    out["risk_vol_ratio"] = out["risk_vol_21"] / out["risk_vol_63"]
    rng = (df["high"] - df["low"]) / c
    out["risk_range_10"] = roll(rng, 10, "mean")
    out["risk_max_21"] = roll(r1, 21, "max")
    out["risk_min_21"] = roll(r1, 21, "min")

    if spy_r is not None:
        m = df["date"].map(spy_r)
        xy = roll(r1 * m, 63, "mean")
        x = roll(r1, 63, "mean")
        y = roll(m, 63, "mean")
        yy = roll(m * m, 63, "mean")
        var = yy - y * y
        out["risk_beta_63"] = (xy - x * y) / var.where(var > 0)
        mc = m.groupby(df["ticker"]).transform(lambda s: s.rolling(21, min_periods=10).sum())
        mc63 = m.groupby(df["ticker"]).transform(lambda s: s.rolling(63, min_periods=30).sum())
        out["mom_rel_21"] = out["mom_ret_21"] - mc
        out["mom_rel_63"] = out["mom_ret_63"] - mc63

    delta = c.groupby(df["ticker"]).diff()
    up = delta.clip(lower=0)
    dn = (-delta).clip(lower=0)
    au = up.groupby(df["ticker"]).transform(lambda s: s.ewm(alpha=1 / 14, min_periods=14).mean())
    ad = dn.groupby(df["ticker"]).transform(lambda s: s.ewm(alpha=1 / 14, min_periods=14).mean())
    out["trend_rsi_14"] = 100 - 100 / (1 + au / ad.replace(0, np.nan))
    sma50 = roll(c, 50, "mean", 40)
    sma200 = roll(c, 200, "mean", 150)
    out["trend_sma50"] = c / sma50 - 1
    out["trend_sma200"] = c / sma200 - 1
    out["trend_50_200"] = sma50 / sma200 - 1
    out["trend_hi_252"] = c / roll(df["high"].astype("float64"), 252, "max", 120) - 1
    out["trend_lo_252"] = c / roll(df["low"].astype("float64"), 252, "min", 120) - 1
    prev_c = c.groupby(df["ticker"]).shift(1)
    out["trend_gap"] = df["open"] / prev_c - 1

    v = df["volume"].astype("float64")
    v21 = roll(v, 21, "mean", 10)
    out["liq_vol_z"] = np.log((v + 1) / (v21 + 1))
    dv = roll(c * v, 21, "mean", 10)
    out["liq_dollar_vol"] = np.log10(dv + 1)
    out["liq_price"] = np.log10(c.clip(lower=0.01))
    out["liq_history"] = g.cumcount().clip(upper=2520).astype(F32)   # trading days on record
    out["close"] = c
    out["dollar_vol"] = dv

    # targets: forward log return over each horizon
    for h, n in horizons.items():
        fwd = lc.groupby(df["ticker"]).shift(-n) - lc
        out[f"y_ret_{h}"] = fwd
    return out


def stock_features(prices: pd.DataFrame, spy_close: pd.Series | None = None,
                   horizons: dict | None = None, chunk: int = 250, log=None) -> pd.DataFrame:
    horizons = DAILY_HORIZONS if horizons is None else horizons
    prices = prices.sort_values(["ticker", "date"]).reset_index(drop=True)
    spy_r = np.log(spy_close).diff() if spy_close is not None and len(spy_close) else None
    tickers = prices["ticker"].unique()
    parts = []
    for i in range(0, len(tickers), chunk):
        sub = prices[prices["ticker"].isin(tickers[i:i + chunk])]
        blk = _stock_block(sub, spy_r, horizons)
        num = blk.columns.difference(["date", "ticker", "close", "dollar_vol"])
        blk[num] = blk[num].astype(F32)
        parts.append(blk)
        if log and (i // chunk) % 4 == 0:
            log(f"  stock features {min(i + chunk, len(tickers)):,}/{len(tickers):,}")
    out = pd.concat(parts, ignore_index=True)
    del parts
    out = out[out["mom_ret_21"].notna()].reset_index(drop=True)    # drop each stock's first month
    num = out.columns.difference(["date", "ticker", "close", "dollar_vol"])
    out[num] = out[num].replace([np.inf, -np.inf], np.nan)
    # beat-the-market targets: stock's forward return minus SPY's over the same horizon
    if spy_close is not None and len(spy_close):
        sl = np.log(spy_close.dropna())
        for h, n in horizons.items():
            if f"y_ret_{h}" in out:
                spy_fwd = (sl.shift(-n) - sl)
                out[f"y_ex_{h}"] = (out[f"y_ret_{h}"] - out["date"].map(spy_fwd)).astype(F32)
    # cross-sectional ranks (where a stock stands vs all others that day)
    for col, name in (("mom_ret_21", "mom_rank_21"), ("mom_ret_252", "mom_rank_252"),
                      ("risk_vol_21", "risk_rank_vol")):
        out[name] = out.groupby("date")[col].rank(pct=True).astype(F32)
    return out


# ======================================================================= market & economy
def _series_freq_periods(idx: pd.DatetimeIndex) -> tuple[int, int]:
    """(periods per year, periods per 'one step' change) from observation spacing."""
    if len(idx) < 3:
        return 1, 1
    d = np.median(np.diff(idx.values).astype("timedelta64[D]").astype(int))
    if d >= 80:
        return 4, 1
    if d >= 25:
        return 12, 1
    if d >= 6:
        return 52, 4
    return 252, 21


def fred_daily(fred_raw: pd.DataFrame, calendar: pd.DatetimeIndex) -> pd.DataFrame:
    """Each FRED series becomes daily columns, visible only from its release date."""
    cols = {}
    for name, grp in fred_raw.groupby("name"):
        s = grp.set_index("obs_date")["value"].sort_index()
        s = s[~s.index.duplicated(keep="last")]
        lag = int(grp["lag_days"].iloc[0])
        kind = grp["kind"].iloc[0]
        per_year, step = _series_freq_periods(s.index)
        feats = {f"eco_{name}": s}
        if kind == "yoy":
            feats[f"eco_{name}_yoy"] = s.pct_change(per_year, fill_method=None) * 100
            feats[f"eco_{name}_yoy_chg"] = feats[f"eco_{name}_yoy"].diff(max(1, per_year // 4))
        elif kind == "chg":
            feats[f"eco_{name}_chg"] = s.diff(step)
            if per_year >= 52:
                feats[f"eco_{name}_chg_3m"] = s.diff(step * 3)
        for k, v in feats.items():
            v = v.copy()
            v.index = v.index + pd.Timedelta(days=lag)
            v = v[~v.index.duplicated(keep="last")]
            cols[k] = v.reindex(v.index.union(calendar)).ffill().reindex(calendar)
    if not cols:
        return pd.DataFrame(index=calendar)
    out = pd.DataFrame(cols, index=calendar)
    # Raw price-level series (CPI index, payrolls count, GDP $) trend forever and let a
    # tree model memorise the calendar; keep only their changes.
    trending = [f"eco_{n}" for n, (_, _, kind) in config.FRED_SERIES.items() if kind == "yoy"]
    trending += ["eco_payrolls", "eco_jobless_claims"]
    return out.drop(columns=[c for c in trending if c in out.columns])


LEVEL_SERIES = {"vix", "tnx", "irx"}   # yields / indexes where the level itself matters


def context_daily(ctx_wide: pd.DataFrame, calendar: pd.DatetimeIndex) -> pd.DataFrame:
    if ctx_wide is None or ctx_wide.empty:
        return pd.DataFrame(index=calendar)
    w = ctx_wide.reindex(ctx_wide.index.union(calendar)).ffill(limit=5).reindex(calendar)
    cols = {}
    for name in w.columns:
        s = w[name]
        if name in LEVEL_SERIES:
            cols[f"mkt_{name}"] = s
            cols[f"mkt_{name}_chg_5"] = s.diff(5)
            cols[f"mkt_{name}_chg_21"] = s.diff(21)
        else:
            ls = np.log(s.where(s > 0))
            cols[f"mkt_{name}_ret_5"] = ls.diff(5)
            cols[f"mkt_{name}_ret_21"] = ls.diff(21)
    if "spy" in w.columns:
        ls = np.log(w["spy"])
        cols["mkt_spy_ret_1"] = ls.diff()
        cols["mkt_spy_ret_63"] = ls.diff(63)
        cols["mkt_spy_vol_21"] = ls.diff().rolling(21).std() * np.sqrt(252)
        cols["mkt_spy_sma200"] = w["spy"] / w["spy"].rolling(200, min_periods=150).mean() - 1
    return pd.DataFrame(cols, index=calendar)


def custom_daily(custom: pd.DataFrame, calendar: pd.DatetimeIndex) -> pd.DataFrame:
    if custom is None or custom.empty:
        return pd.DataFrame(index=calendar)
    w = custom.reindex(custom.index.union(calendar)).ffill().reindex(calendar)
    cols = {}
    for name in w.columns:
        safe = "".join(ch if ch.isalnum() else "_" for ch in name.lower())
        cols[f"cus_{safe}"] = w[name]
        cols[f"cus_{safe}_chg_21"] = w[name].diff(21)
    return pd.DataFrame(cols, index=calendar)


def calendar_features(calendar: pd.DatetimeIndex) -> pd.DataFrame:
    d = pd.Series(calendar, index=calendar)
    month_end = d + pd.offsets.BMonthEnd(0)
    return pd.DataFrame({
        "cal_dow": d.dt.dayofweek, "cal_month": d.dt.month,
        "cal_days_to_month_end": (month_end - d).dt.days,
        "cal_day_of_month": d.dt.day,
    }, index=calendar).astype(F32)


# ======================================================================= events
def _days_since(dates: pd.DatetimeIndex, event_dates: Iterable, cap: int = 365, forward: bool = False):
    ev = np.sort(pd.to_datetime(pd.Series(list(event_dates))).dropna().unique())
    if len(ev) == 0:
        return np.full(len(dates), cap, dtype=F32)
    dv = dates.values
    if forward:
        pos = np.searchsorted(ev, dv, side="right")      # strictly after t
        ok = pos < len(ev)
        out = np.full(len(dv), cap, dtype=F32)
        out[ok] = (ev[pos[ok]] - dv[ok]).astype("timedelta64[D]").astype(int)
    else:
        pos = np.searchsorted(ev, dv, side="right") - 1  # on or before t
        ok = pos >= 0
        out = np.full(len(dv), cap, dtype=F32)
        out[ok] = (dv[ok] - ev[pos[ok]]).astype("timedelta64[D]").astype(int)
    return np.minimum(out, cap).astype(F32)


def event_types_for_model(events: pd.DataFrame, min_events: int = 3) -> tuple[list, list]:
    """Event types with enough examples: (market-wide types, ticker types)."""
    mkt = events[events["ticker"] == ""]["type"].value_counts()
    tkr = events[events["ticker"] != ""]["type"].value_counts()
    return (sorted(mkt[mkt >= min_events].index), sorted(t for t in tkr[tkr >= min_events].index
                                                         if t not in config.MODEL_SKIP_TICKER_TYPES))


def market_event_features(events: pd.DataFrame, calendar: pd.DatetimeIndex,
                          types: list[str]) -> pd.DataFrame:
    cols = {}
    mkt = events[events["ticker"] == ""]
    for t in types:
        d = mkt.loc[mkt["type"] == t, "date"]
        cols[f"evt_since_{t}"] = _days_since(calendar, d)
        if t in config.ADVANCE_KNOWN_EVENTS:
            cols[f"evt_until_{t}"] = _days_since(calendar, d, forward=True)
    return pd.DataFrame(cols, index=calendar)


def ticker_event_features(rows: pd.DataFrame, events: pd.DataFrame, earnings: pd.DataFrame,
                          types: list[str]) -> pd.DataFrame:
    """Per (ticker, date) event features. rows needs date, ticker."""
    from .events import explode_tickers

    orig_index = rows.index
    rows = rows.reset_index(drop=True)
    out = pd.DataFrame(index=rows.index)
    tk_events = explode_tickers(events[events["ticker"] != ""])
    for t in types + ["earnings"]:
        if t == "earnings":
            ev = earnings[["ticker", "date"]].copy() if earnings is not None and not earnings.empty else \
                pd.DataFrame(columns=["ticker", "date"])
        else:
            ev = tk_events.loc[tk_events["type"] == t, ["ticker", "date"]]
        since = np.full(len(rows), 365, dtype=F32)
        until = np.full(len(rows), 365, dtype=F32)
        if not ev.empty:
            by_tick = {k: v["date"] for k, v in ev.groupby("ticker")}
            for tick, idx in rows.groupby("ticker", sort=False).indices.items():
                e = by_tick.get(tick)
                if e is None or e.empty:
                    continue
                dts = pd.DatetimeIndex(rows["date"].values[idx])
                since[idx] = _days_since(dts, e)
                if t == "earnings":
                    until[idx] = _days_since(dts, e, forward=True)
        out[f"evt_tk_since_{t}"] = since
        if t == "earnings":
            out["evt_tk_until_earnings"] = until
    if earnings is not None and not earnings.empty and "surprise_pct" in earnings:
        e = earnings.dropna(subset=["surprise_pct"]).sort_values("date")[["ticker", "date", "surprise_pct"]]
        left = rows[["ticker", "date"]].reset_index().sort_values("date")
        m = pd.merge_asof(left, e, on="date", by="ticker", direction="backward")
        out.loc[m["index"].values, "evt_tk_last_surprise"] = m["surprise_pct"].clip(-100, 100).values.astype(F32)
    out.index = orig_index
    return out


# ======================================================================= assembly
def date_features(calendar: pd.DatetimeIndex, ctx_wide, fred_raw, custom, events, mkt_types) -> pd.DataFrame:
    parts = [context_daily(ctx_wide, calendar), fred_daily(fred_raw, calendar) if fred_raw is not None
             and len(fred_raw) else pd.DataFrame(index=calendar),
             custom_daily(custom, calendar), market_event_features(events, calendar, mkt_types),
             calendar_features(calendar)]
    df = pd.concat(parts, axis=1)
    df.index.name = "date"
    return df.replace([np.inf, -np.inf], np.nan).astype(F32)


def assemble(stock_rows: pd.DataFrame, date_feats: pd.DataFrame, events: pd.DataFrame,
             earnings: pd.DataFrame, tk_types: list[str], insider_trades: pd.DataFrame | None = None) -> pd.DataFrame:
    rows = stock_rows.reset_index(drop=True)
    df = rows.join(date_feats, on="date")
    parts = [df, ticker_event_features(rows, events, earnings, tk_types)]
    if insider_trades is not None and not insider_trades.empty:
        from .insiders import insider_features

        parts.append(insider_features(rows, insider_trades))
    return pd.concat(parts, axis=1)


def feature_columns(df: pd.DataFrame) -> list[str]:
    return [c for c in df.columns if group_of(c) in GROUP_NAMES]


# ======================================================================= hourly
def hourly_features(hourly: pd.DataFrame, daily_feats: pd.DataFrame | None,
                    market_hourly: pd.DataFrame | None) -> pd.DataFrame:
    """Hourly bars → features + next-bar target. daily_feats: latest daily stock features
    (as of the previous close) to give the hourly model the bigger picture."""
    df = hourly.sort_values(["ticker", "date"]).reset_index(drop=True)
    tick = df["ticker"]
    lc = np.log(df["close"].astype("float64").where(df["close"] > 0))
    r1 = lc.groupby(tick).diff()
    out = pd.DataFrame({"date": df["date"], "ticker": tick})
    for n in (1, 2, 3, 7, 14, 35):
        out[f"intra_ret_{n}"] = lc - lc.groupby(tick).shift(n)
    out["intra_vol_7"] = r1.groupby(tick).transform(lambda s: s.rolling(7, min_periods=4).std())
    out["intra_vol_35"] = r1.groupby(tick).transform(lambda s: s.rolling(35, min_periods=15).std())
    out["intra_vol_ratio"] = out["intra_vol_7"] / out["intra_vol_35"]
    v = df["volume"].astype("float64")
    out["intra_vol_z"] = np.log((v + 1) / (v.groupby(tick).transform(lambda s: s.rolling(35, min_periods=10).mean()) + 1))
    day = df["date"].dt.normalize()
    first_open = df.groupby([tick, day])["open"].transform("first")
    out["intra_from_open"] = df["close"] / first_open - 1
    out["intra_bar"] = df.groupby([tick, day]).cumcount().astype(F32)
    out["intra_last_bar"] = (df.groupby([tick, day]).cumcount(ascending=False) == 0).astype(F32)
    out["cal_hour"] = df["date"].dt.hour.astype(F32)
    out["cal_dow"] = df["date"].dt.dayofweek.astype(F32)
    hi = df.groupby([tick, day])["high"].cummax()
    lo = df.groupby([tick, day])["low"].cummin()
    out["intra_pos_in_range"] = (df["close"] - lo) / (hi - lo).replace(0, np.nan)
    out["close"] = df["close"].astype("float64")
    out["y_ret_1h"] = lc.groupby(tick).shift(-1) - lc
    if market_hourly is not None and not market_hourly.empty:
        mh = market_hourly.copy()
        mcols = {}
        if "SPY" in mh:
            ls = np.log(mh["SPY"])
            mcols["mkt_spy_h_ret_1"] = ls.diff()
            mcols["mkt_spy_h_ret_7"] = ls.diff(7)
        if "^VIX" in mh:
            mcols["mkt_vix_h"] = mh["^VIX"]
            mcols["mkt_vix_h_chg_7"] = mh["^VIX"].diff(7)
        if "QQQ" in mh:
            mcols["mkt_qqq_h_ret_1"] = np.log(mh["QQQ"]).diff()
        m = pd.DataFrame(mcols).sort_index()
        out = pd.merge_asof(out.sort_values("date"), m, left_on="date", right_index=True,
                            direction="backward").sort_values(["ticker", "date"])
        if "SPY" in mh:
            spy_fwd = ls.shift(-1) - ls
            out["y_ex_1h"] = out["y_ret_1h"] - out["date"].map(spy_fwd)
    if daily_feats is not None and not daily_feats.empty:
        keep = ["mom_ret_5", "mom_ret_21", "mom_ret_252", "risk_vol_21", "risk_beta_63",
                "trend_rsi_14", "trend_sma200", "trend_hi_252", "liq_dollar_vol", "liq_price",
                "evt_tk_until_earnings", "evt_tk_since_earnings", "mkt_vix", "mkt_spy_ret_21",
                "sec_rel_5", "sec_mom_21", "ins_net_ratio_90", "evt_until_fomc", "evt_until_cpi_release",
                "evt_until_jobs_report"]
        keep = [k for k in keep if k in daily_feats.columns]
        d = daily_feats[["date", "ticker", "dollar_vol"] + keep].copy()
        d["avail"] = d["date"] + pd.Timedelta(hours=16, minutes=1)   # usable after that day's close
        d = d.drop(columns="date").sort_values("avail")
        out = pd.merge_asof(out.sort_values("date"), d, left_on="date", right_on="avail",
                            by="ticker", direction="backward").drop(columns="avail")
        out = out.rename(columns={k: f"{k}" for k in keep})
        out = out.sort_values(["ticker", "date"])
    num = [c for c in out.columns if c not in ("date", "ticker", "close", "dollar_vol")]
    out[num] = out[num].replace([np.inf, -np.inf], np.nan).astype(F32)
    return out.reset_index(drop=True)
