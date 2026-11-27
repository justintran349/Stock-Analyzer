"""Hidden links: does series X (oil, bitcoin, jobless claims, lumber, your own data...)
move *before* stock Y?

For every pair we correlate the driver's weekly change with the stock's return 1–4 weeks
later, first in the DISCOVERY period (before the 2-year test window), then check whether
the link held up in the CONFIRM period (the last 2 years). Testing thousands of pairs
guarantees some fake links by luck, so p-values are corrected (Benjamini–Hochberg) and the
confirm check is the real filter.
"""
from __future__ import annotations

import numpy as np
import pandas as pd
from scipy import stats

from . import config, data, features as F


def driver_table(ctx_wide: pd.DataFrame, fred_raw: pd.DataFrame, custom: pd.DataFrame) -> pd.DataFrame:
    """Weekly (Friday) changes of every candidate driver, using only public values."""
    if ctx_wide is None or ctx_wide.empty:
        return pd.DataFrame()
    cal = pd.bdate_range(ctx_wide.index.min(), ctx_wide.index.max())
    cols = {}
    w = ctx_wide.reindex(ctx_wide.index.union(cal)).ffill(limit=5).reindex(cal)
    for c in w.columns:
        s = w[c]
        cols[c] = s.diff() if c in F.LEVEL_SERIES else np.log(s.where(s > 0)).diff()
    if fred_raw is not None and len(fred_raw):
        for name, grp in fred_raw.groupby("name"):
            s = grp.set_index("obs_date")["value"].sort_index()
            s = s[~s.index.duplicated(keep="last")]
            if grp["kind"].iloc[0] == "yoy":
                per_year, _ = F._series_freq_periods(s.index)
                s = s.pct_change(per_year, fill_method=None) * 100
            s.index = s.index + pd.Timedelta(days=int(grp["lag_days"].iloc[0]))
            s = s[~s.index.duplicated(keep="last")]
            daily_s = s.reindex(s.index.union(cal)).ffill().reindex(cal)
            cols[f"eco:{name}"] = daily_s.diff()
    if custom is not None and not custom.empty:
        cu = custom.reindex(custom.index.union(cal)).ffill().reindex(cal)
        for c in cu.columns:
            cols[f"custom:{c}"] = cu[c].diff()
    daily = pd.DataFrame(cols, index=cal)
    weekly = daily.resample("W-FRI").sum(min_count=1)
    return weekly.loc[:, weekly.notna().sum() > 104]


def weekly_returns(prices: pd.DataFrame) -> pd.DataFrame:
    wide = prices.pivot_table(index="date", columns="ticker", values="close").sort_index()
    wk = wide.resample("W-FRI").last()
    return np.log(wk).diff()


def _corr_cols(X: pd.DataFrame, y: pd.Series) -> tuple[pd.Series, pd.Series]:
    """Correlation of each column of X with y plus number of overlapping weeks."""
    mask = X.notna() & y.notna().values[:, None]
    n = mask.sum()
    Xc = X.where(mask)
    yv = pd.DataFrame(np.repeat(y.values[:, None], X.shape[1], axis=1), index=X.index, columns=X.columns).where(mask)
    xm, ym = Xc - Xc.mean(), yv - yv.mean()
    r = (xm * ym).sum() / np.sqrt((xm ** 2).sum() * (ym ** 2).sum())
    return r, n


def _pvals(r: pd.Series, n: pd.Series) -> pd.Series:
    t = r * np.sqrt((n - 2) / (1 - r.clip(-0.9999, 0.9999) ** 2))
    return pd.Series(2 * stats.t.sf(np.abs(t), df=(n - 2).clip(lower=1)), index=r.index)


def bh_qvalues(p: np.ndarray) -> np.ndarray:
    p = np.asarray(p, dtype=float)
    n = np.sum(~np.isnan(p))
    order = np.argsort(np.where(np.isnan(p), np.inf, p))
    q = np.full(len(p), np.nan)
    prev = 1.0
    for rank in range(n, 0, -1):
        i = order[rank - 1]
        prev = min(prev, p[i] * n / rank)
        q[i] = prev
    return q


def scan(drivers: pd.DataFrame, rets: pd.DataFrame, lags=(1, 2, 3, 4),
         split_date: pd.Timestamp | None = None, min_weeks: int = 150,
         driver_names: list[str] | None = None) -> pd.DataFrame:
    """Every driver × every stock × lag. Returns one row per pair with discovery and
    confirm-period correlations."""
    if split_date is None:
        split_date = rets.index.max() - pd.DateOffset(days=int(365.25 * config.TEST_YEARS))
    drivers = drivers.reindex(rets.index)
    names = driver_names or list(drivers.columns)
    out = []
    disc_idx = rets.index < split_date
    conf_idx = ~disc_idx
    for d in names:
        x = drivers[d]
        for lag in lags:
            fut = rets.shift(-lag)   # stock return `lag` weeks after the driver's move
            r1, n1 = _corr_cols(fut[disc_idx], x[disc_idx])
            r2, n2 = _corr_cols(fut[conf_idx], x[conf_idx])
            df = pd.DataFrame({"driver": d, "ticker": r1.index, "lag_weeks": lag,
                               "r_discovery": r1.values, "weeks_discovery": n1.values,
                               "r_confirm": r2.reindex(r1.index).values,
                               "weeks_confirm": n2.reindex(r1.index).values})
            out.append(df[df["weeks_discovery"] >= min_weeks])
    if not out:
        return pd.DataFrame()
    res = pd.concat(out, ignore_index=True)
    res["p_discovery"] = _pvals(res["r_discovery"], res["weeks_discovery"]).values
    res["q_discovery"] = bh_qvalues(res["p_discovery"].values)
    res["p_confirm"] = _pvals(res["r_confirm"].fillna(0), res["weeks_confirm"].fillna(3)).values
    res["held_up"] = (np.sign(res["r_confirm"]) == np.sign(res["r_discovery"])) & (res["p_confirm"] < 0.05)
    res["strength"] = np.abs(res["r_discovery"]) * np.where(res["held_up"], 1, 0.3)
    return res.sort_values("strength", ascending=False).reset_index(drop=True)


def contemporaneous(drivers: pd.DataFrame, rets: pd.DataFrame, ticker: str) -> pd.Series:
    """Same-week correlation of a stock with every driver (what it moves *with*)."""
    y = rets[ticker]
    r, n = _corr_cols(drivers.reindex(rets.index), y)
    return r[n > 104].sort_values()
