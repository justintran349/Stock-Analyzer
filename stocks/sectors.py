"""Sector and industry for every stock (Yahoo, one request per stock, refreshed every 90 days)
and the sector/peer features built on them:

  sec_rel_5 / sec_rel_21     the stock's return minus its sector's median (beating its peers?)
  sec_mom_21 / sec_mom_63    the sector's median return (is the whole sector moving?)
  sec_ind_mom_5 / _21        same for the narrower industry
  sec_ind_rel_21             stock vs its industry
  sec_peer_earn_react_5      earnings spillover: how industry peers that reported in the last
                             5 days moved on their earnings day (beyond the market)
  sec_peers_reported_5       how many industry peers reported in the last 5 days
  sec_peers_upcoming_10      how many industry peers report in the next 10 days (dates are known)
"""
from __future__ import annotations

import time
from typing import Callable

import numpy as np
import pandas as pd

from . import config, data


def update_sectors(tickers: list[str] | None = None, max_age_days: int = 90,
                   log: Callable = print) -> pd.DataFrame:
    import logging

    import yfinance as yf

    logging.getLogger("yfinance").setLevel(logging.CRITICAL)
    if tickers is None:
        tickers = data.load_universe()["ticker"].tolist()
    old = load_sectors()
    fresh = set(old.loc[pd.to_datetime(old["fetched"]) > pd.Timestamp.now() - pd.Timedelta(days=max_age_days),
                        "ticker"]) if not old.empty else set()
    todo = [t for t in tickers if t not in fresh]
    log(f"Sectors: {len(todo):,} tickers to look up ({len(fresh):,} already known)")
    rows = []
    for i, t in enumerate(todo):
        info = {}
        for attempt in range(3):
            try:
                info = yf.Ticker(t).info or {}
                break
            except Exception as e:
                if "rate" in f"{type(e).__name__} {e}".lower() or "too many" in str(e).lower():
                    log(f"  Yahoo is rate-limiting; waiting {60 * (attempt + 1)}s ...")
                    time.sleep(60 * (attempt + 1))
                    continue
                break
        rows.append({"ticker": t, "sector": info.get("sector") or info.get("category") or "",
                     "industry": info.get("industry") or "", "quote_type": info.get("quoteType", ""),
                     "market_cap": info.get("marketCap"), "fetched": pd.Timestamp.now()})
        if (i + 1) % 100 == 0 or i == len(todo) - 1:
            old = _save(old, rows)
            rows = []
            log(f"  sectors {i + 1:,}/{len(todo):,}")
        time.sleep(0.3)
    return _save(old, rows)


def _save(old: pd.DataFrame, rows: list) -> pd.DataFrame:
    if not rows:
        return old
    new = pd.DataFrame(rows)
    keep = old[~old["ticker"].isin(new["ticker"])] if not old.empty else old
    out = pd.concat([keep, new], ignore_index=True) if not keep.empty else new
    out.to_csv(config.SECTORS_FILE, index=False)
    return out


def load_sectors() -> pd.DataFrame:
    if not config.SECTORS_FILE.exists():
        return pd.DataFrame(columns=["ticker", "sector", "industry", "quote_type", "market_cap", "fetched"])
    df = pd.read_csv(config.SECTORS_FILE, keep_default_na=False)
    df["sector"] = df["sector"].replace("", "Unknown")
    df["industry"] = df["industry"].replace("", "Unknown")
    return df


# ======================================================================= features
def add_sector_features(stock: pd.DataFrame, sectors: pd.DataFrame, earnings: pd.DataFrame | None,
                        log: Callable | None = None) -> pd.DataFrame:
    """stock: output of features.stock_features (all tickers, all dates). Adds sec_* columns in place."""
    if sectors is None or sectors.empty or stock.empty:
        return stock
    m = sectors.set_index("ticker")
    sec = stock["ticker"].map(m["sector"]).fillna("Unknown").values
    ind = stock["ticker"].map(m["industry"]).fillna("Unknown").values
    d = stock["date"].values
    f32 = np.float32

    def grp_median(col, key):
        return stock.groupby([d, key])[col].transform("median")

    for n, col in ((5, "mom_ret_5"), (21, "mom_ret_21")):
        sm = grp_median(col, sec)
        stock[f"sec_rel_{n}"] = (stock[col] - sm).astype(f32)
        im = grp_median(col, ind)
        stock[f"sec_ind_mom_{n}"] = im.astype(f32)
    stock["sec_mom_21"] = grp_median("mom_ret_21", sec).astype(f32)
    stock["sec_mom_63"] = grp_median("mom_ret_63", sec).astype(f32)
    stock["sec_ind_rel_21"] = (stock["mom_ret_21"] - stock["sec_ind_mom_21"]).astype(f32)
    # sectors with too few members are noisy → blank
    cnt = stock.groupby([d, ind])["ticker"].transform("size")
    stock.loc[cnt.values < 3, ["sec_ind_mom_5", "sec_ind_mom_21", "sec_ind_rel_21"]] = np.nan

    if earnings is not None and not earnings.empty:
        _peer_earnings(stock, ind, earnings)
    if log:
        log(f"  sector features added ({pd.Series(sec).nunique()} sectors, {pd.Series(ind).nunique()} industries)")
    return stock


def _peer_earnings(stock: pd.DataFrame, ind: np.ndarray, earnings: pd.DataFrame) -> None:
    """Earnings spillover inside each industry, excluding the stock's own report."""
    f32 = np.float32
    e = earnings[["ticker", "date"]].dropna().drop_duplicates()
    e = e.assign(rep=1.0)
    s = stock[["ticker", "date", "mom_ret_1"]].copy()
    s["ind"] = ind
    # reaction beyond the market: the stock's day return minus that day's median stock
    s["ab"] = s["mom_ret_1"] - s.groupby("date")["mom_ret_1"].transform("median")
    s = s.merge(e, on=["ticker", "date"], how="left")
    s["rep"] = s["rep"].fillna(0.0)
    s["rep_ab"] = s["ab"].fillna(0) * s["rep"]
    # industry-day totals, then 5-trading-day rolling sums along the calendar
    last = s["date"].max()
    cal = pd.DatetimeIndex(np.sort(s["date"].unique())).union(
        pd.bdate_range(last + pd.Timedelta(days=1), last + pd.Timedelta(days=21)))
    tot = s.groupby(["ind", "date"])[["rep", "rep_ab"]].sum()
    # future earnings dates (already announced) count toward "peers reporting soon"
    fut_e = e[(e["date"] > last) & (e["date"] <= cal[-1])].copy()
    if not fut_e.empty:
        imap = dict(zip(s["ticker"], s["ind"]))
        fut_e["ind"] = fut_e["ticker"].map(imap)
        fut_e = fut_e.dropna(subset=["ind"])
        ft = fut_e.groupby(["ind", "date"])["rep"].sum().to_frame().assign(rep_ab=0.0)
        tot = pd.concat([tot, ft]).groupby(level=[0, 1]).sum()
    wide_c = tot["rep"].unstack(0).reindex(cal).fillna(0)
    wide_s = tot["rep_ab"].unstack(0).reindex(cal).fillna(0)
    c5 = wide_c.rolling(5, min_periods=1).sum().stack()
    s5 = wide_s.rolling(5, min_periods=1).sum().stack()
    key = pd.MultiIndex.from_arrays([s["date"], s["ind"]])
    C = c5.reindex(key).values
    S = s5.reindex(key).values
    # the stock's own reports in the same window
    s = s.sort_values(["ticker", "date"])
    own_c = s.groupby("ticker")["rep"].transform(lambda x: x.rolling(5, min_periods=1).sum())
    own_s = s.groupby("ticker")["rep_ab"].transform(lambda x: x.rolling(5, min_periods=1).sum())
    own_c = own_c.sort_index().values
    own_s = own_s.sort_index().values
    s = s.sort_index()
    peers = C - own_c
    with np.errstate(invalid="ignore", divide="ignore"):
        react = np.where(peers > 0, (S - own_s) / peers, np.nan)
    unknown = s["ind"].values == "Unknown"
    react[unknown] = np.nan
    peers[unknown] = np.nan
    stock["sec_peer_earn_react_5"] = react.astype(f32)
    stock["sec_peers_reported_5"] = np.nan_to_num(peers).astype(f32)
    # upcoming: peers reporting in the next 10 trading days (earnings dates are announced ahead)
    fut = wide_c[::-1].rolling(10, min_periods=1).sum()[::-1].shift(-1).fillna(0).stack()
    U = fut.reindex(key).values
    stock["sec_peers_upcoming_10"] = np.nan_to_num(U).astype(f32)
