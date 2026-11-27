"""Events you care about (elections, earnings, wars, launches, CEO news, ...) and
event studies: how stocks and the economy moved around them.

The list lives in data/events.csv. Edit it in the app (Events page) or in any
spreadsheet. Columns: date, type, ticker, title, notes
  - ticker blank  = market-wide event (election, war, shock)
  - ticker filled = company event (CEO change, product launch). Several tickers: "BA;GE"
  - type is free text: add new types whenever you like
"""
from __future__ import annotations

import numpy as np
import pandas as pd

from . import config

COLUMNS = ["date", "type", "ticker", "title", "notes"]

# Seed list. Dates are the first trading session the market could react
# (weekend news → following Monday).
SEED_EVENTS = [
    # --- US elections (presidential + midterms)
    ("2000-11-07", "election", "", "Presidential election: Bush vs Gore (result disputed until Dec 12)"),
    ("2002-11-05", "election", "", "Midterm elections"),
    ("2004-11-02", "election", "", "Presidential election: Bush re-elected"),
    ("2006-11-07", "election", "", "Midterm elections: Democrats win Congress"),
    ("2008-11-04", "election", "", "Presidential election: Obama"),
    ("2010-11-02", "election", "", "Midterm elections: Republicans win House"),
    ("2012-11-06", "election", "", "Presidential election: Obama re-elected"),
    ("2014-11-04", "election", "", "Midterm elections: Republicans win Senate"),
    ("2016-11-08", "election", "", "Presidential election: Trump"),
    ("2018-11-06", "election", "", "Midterm elections: Democrats win House"),
    ("2020-11-03", "election", "", "Presidential election: Biden"),
    ("2022-11-08", "election", "", "Midterm elections"),
    ("2024-11-05", "election", "", "Presidential election: Trump"),
    ("2026-11-03", "election", "", "Midterm elections"),
    ("2028-11-07", "election", "", "Presidential election (upcoming)"),
    # --- wars and geopolitical
    ("2001-09-17", "war", "", "9/11 attacks (markets reopened Sep 17)"),
    ("2001-10-08", "war", "", "US begins strikes in Afghanistan (Oct 7, Sunday)"),
    ("2003-03-20", "war", "", "US-led invasion of Iraq"),
    ("2011-03-21", "war", "", "NATO intervention in Libya (Mar 19)"),
    ("2014-03-03", "war", "", "Russia moves into Crimea"),
    ("2020-01-03", "war", "", "US strike kills Iranian general Soleimani"),
    ("2022-02-24", "war", "", "Russia invades Ukraine"),
    ("2023-10-09", "war", "", "Hamas attack on Israel (Oct 7, Saturday)"),
    ("2024-04-15", "war", "", "Iran's drone and missile attack on Israel (Apr 13)"),
    ("2025-06-13", "war", "", "Israel strikes Iran"),
    ("2025-06-23", "war", "", "US strikes Iranian nuclear sites (Jun 21-22)"),
    # --- market shocks and policy
    ("2008-09-15", "market_shock", "", "Lehman Brothers bankruptcy"),
    ("2011-08-08", "market_shock", "", "S&P downgrades US credit rating (Aug 5)"),
    ("2015-08-24", "market_shock", "", "China devaluation flash crash"),
    ("2020-03-11", "market_shock", "", "WHO declares COVID-19 a pandemic"),
    ("2020-11-09", "market_shock", "PFE", "Pfizer vaccine 90% effective"),
    ("2023-03-10", "market_shock", "", "Silicon Valley Bank collapse"),
    ("2025-04-03", "policy", "", "'Liberation Day' tariffs announced (Apr 2 after close)"),
    ("2025-04-09", "policy", "", "90-day tariff pause"),
    ("2008-11-25", "policy", "", "Fed announces first quantitative easing"),
    ("2022-03-16", "policy", "", "Fed's first rate hike of the 2022 cycle"),
    ("2024-09-18", "policy", "", "Fed's first rate cut of the 2024 cycle (50 bp)"),
    # --- product launches (NYSE companies + a few market-moving ones)
    ("2007-01-09", "product_launch", "AAPL", "iPhone announced"),
    ("2010-01-27", "product_launch", "AAPL", "iPad announced"),
    ("2019-11-12", "product_launch", "DIS", "Disney+ launches"),
    ("2021-05-20", "product_launch", "F", "Ford F-150 Lightning revealed (May 19 evening)"),
    ("2022-11-30", "product_launch", "", "ChatGPT released (AI boom starts)"),
    ("2023-03-14", "product_launch", "", "GPT-4 released"),
    ("2025-01-27", "product_launch", "", "DeepSeek R1 sell-off in AI stocks"),
    ("2021-06-07", "product_launch", "LLY", "Biogen Aduhelm approval lifts Alzheimer's drug makers"),
    ("2023-05-03", "product_launch", "LLY", "Lilly donanemab Alzheimer's trial results"),
    ("2017-06-15", "product_launch", "NKE", "Nike announces direct-to-consumer push"),
    # --- CEO changes and returns
    ("2018-10-01", "ceo_change", "GE", "Larry Culp replaces John Flannery as GE CEO"),
    ("2020-02-26", "ceo_change", "DIS", "Bob Chapek replaces Bob Iger as Disney CEO (Feb 25 after close)"),
    ("2022-11-21", "ceo_change", "DIS", "Bob Iger returns as Disney CEO (Nov 20, Sunday)"),
    ("2024-08-13", "ceo_change", "CMG", "Chipotle CEO Brian Niccol leaves for Starbucks"),
    ("2024-07-31", "ceo_change", "BA", "Kelly Ortberg named Boeing CEO"),
    ("2024-03-25", "ceo_change", "BA", "Boeing CEO Dave Calhoun to step down"),
    ("2024-09-20", "ceo_change", "NKE", "Elliott Hill returns to Nike as CEO (Sep 19 after close)"),
]


def seed_events() -> pd.DataFrame:
    return pd.DataFrame([(d, t, k, title, "") for d, t, k, title in SEED_EVENTS], columns=COLUMNS)


def load_events(include_earnings: bool = False, include_auto: bool = True) -> pd.DataFrame:
    """Your list (data/events.csv) plus, when include_auto, the scheduled economic calendar
    (Fed, CPI, jobs, ...) and insider cluster buys. include_earnings adds every earnings date."""
    if config.EVENTS_FILE.exists():
        df = pd.read_csv(config.EVENTS_FILE, dtype=str, keep_default_na=False)
    else:
        df = seed_events()
        df.to_csv(config.EVENTS_FILE, index=False)
    for c in COLUMNS:
        if c not in df:
            df[c] = ""
    df["date"] = pd.to_datetime(df["date"], errors="coerce")
    df = df.dropna(subset=["date"])
    df["type"] = df["type"].str.strip().str.lower().str.replace(" ", "_")
    df["ticker"] = df["ticker"].fillna("").str.upper().str.strip()
    extra = []
    if include_auto:
        from .insiders import insider_events
        from .macro_calendar import load_calendar

        extra += [load_calendar(), insider_events()]
    if include_earnings:
        extra.append(earnings_events())
    extra = [e for e in extra if e is not None and not e.empty]
    if extra:
        df = pd.concat([df] + extra, ignore_index=True)
    return df[COLUMNS].sort_values("date").reset_index(drop=True)


def save_events(df: pd.DataFrame) -> None:
    df = df.copy()
    df["date"] = pd.to_datetime(df["date"], errors="coerce").dt.strftime("%Y-%m-%d")
    df = df.dropna(subset=["date"])
    df["type"] = df["type"].fillna("custom").astype(str).str.strip().str.lower().str.replace(" ", "_")
    df["ticker"] = df["ticker"].fillna("").astype(str).str.upper().str.strip()
    for c in ("title", "notes"):
        df[c] = df[c].fillna("").astype(str)
    df[COLUMNS].sort_values("date").to_csv(config.EVENTS_FILE, index=False)


def earnings_events() -> pd.DataFrame:
    from .data import load_earnings

    e = load_earnings()
    if e.empty:
        return pd.DataFrame(columns=COLUMNS)
    surprise = e["surprise_pct"].map(lambda v: "" if pd.isna(v) else f"surprise {v:+.1f}%")
    return pd.DataFrame({"date": pd.to_datetime(e["date"]), "type": "earnings", "ticker": e["ticker"],
                         "title": e["ticker"] + " earnings", "notes": surprise})


def explode_tickers(ev: pd.DataFrame) -> pd.DataFrame:
    ev = ev.copy()
    ev["ticker"] = ev["ticker"].fillna("").str.split(r"[;, ]+")
    ev = ev.explode("ticker")
    ev["ticker"] = ev["ticker"].fillna("")
    return ev.reset_index(drop=True)


# ======================================================================= event study
def _returns_wide(prices: pd.DataFrame) -> pd.DataFrame:
    wide = prices.pivot_table(index="date", columns="ticker", values="close").sort_index()
    return np.log(wide).diff()


def event_study(events: pd.DataFrame, rets: pd.DataFrame, market: pd.Series,
                default_ticker: str | None = None, pre: int = 5, post: int = 20,
                est_window: tuple[int, int] = (-250, -30)) -> dict:
    """Classic event study.

    rets: wide daily log returns (date × ticker). market: daily log returns of SPY.
    For each event and ticker: abnormal return = return − beta × market (beta estimated
    on days −250…−30). If the ticker *is* the market, raw returns are used.
    Returns paths (one row per event, columns −pre…+post of cumulative abnormal return),
    a per-event summary table and averages.
    """
    dates = rets.index
    rows, paths = [], []
    for _, ev in events.iterrows():
        tick = ev["ticker"] or default_ticker
        if not tick or tick not in rets.columns:
            continue
        pos = dates.searchsorted(pd.Timestamp(ev["date"]))
        if pos >= len(dates) or pos - pre < 0:
            continue
        r = rets[tick]
        is_market = market is not None and r.equals(market)
        lo, hi = pos + est_window[0], pos + est_window[1]
        beta, alpha = 0.0, 0.0
        if not is_market and market is not None and lo >= 0:
            est = pd.concat([r.iloc[lo:hi], market.iloc[lo:hi]], axis=1).dropna()
            if len(est) > 60:
                cov = np.cov(est.iloc[:, 0], est.iloc[:, 1])
                beta = cov[0, 1] / cov[1, 1] if cov[1, 1] > 0 else 1.0
                alpha = 0.0   # alpha is noisy; leaving it out is standard for short windows
        end = min(pos + post + 1, len(dates))
        window = r.iloc[pos - pre:end]
        mkt = market.iloc[pos - pre:end] if market is not None else 0
        ab = (window - alpha - beta * mkt) if not is_market else window
        ab = ab.fillna(0).values
        idx = np.arange(-pre, -pre + len(ab))
        car = pd.Series(np.cumsum(ab), index=idx)
        car = car - (car.loc[-1] if -1 in car.index else 0)   # anchored at the day before
        paths.append(car.rename(f"{pd.Timestamp(ev['date']).date()} {tick}"))
        normal = r.iloc[max(0, pos - 120):pos - 5].abs().mean()
        day0 = ab[pre] if len(ab) > pre else np.nan
        pre_vol = r.iloc[max(0, pos - 21):pos].std() * np.sqrt(252)
        post_vol = r.iloc[pos + 1:pos + 22].std() * np.sqrt(252)
        rows.append({
            "date": pd.Timestamp(ev["date"]).date(), "type": ev.get("type", ""), "ticker": tick,
            "title": ev.get("title", ""),
            "day0_move_%": 100 * (np.exp(day0) - 1),
            "abs_day0_vs_normal": abs(day0) / normal if normal and normal > 0 else np.nan,
            **{f"car_{k}d_%": 100 * (np.exp(car.loc[k]) - 1) for k in (1, 5, 20) if k in car.index},
            "vol_before_%": 100 * pre_vol, "vol_after_%": 100 * post_vol,
            "beta": beta,
        })
    if not paths:
        return {"paths": pd.DataFrame(), "table": pd.DataFrame(), "mean": pd.Series(dtype=float),
                "ci": pd.Series(dtype=float), "n": 0}
    P = pd.concat(paths, axis=1).T
    mean = P.mean()
    n = P.notna().sum()
    ci = 1.96 * P.std() / np.sqrt(n.clip(lower=1))
    return {"paths": P, "table": pd.DataFrame(rows), "mean": mean, "ci": ci, "n": len(P)}


def cross_section_reaction(event_dates: list, rets: pd.DataFrame, market: pd.Series | None,
                           days: tuple[int, int] = (0, 5), min_events: int = 3) -> pd.DataFrame:
    """For market-wide events: which tickers/series react most on average?
    Abnormal (vs market) cumulative return from day `days[0]` to `days[1]`,
    averaged over events, with a t-statistic."""
    idx = rets.index
    if market is not None:
        mk = market.reindex(idx)
        # simple: excess return vs market (beta 1); robust and fast for thousands of tickers
        ab = rets.sub(mk, axis=0)
    else:
        ab = rets
    per_event = []
    for d in event_dates:
        pos = idx.searchsorted(pd.Timestamp(d))
        if pos + days[1] >= len(idx):
            continue
        per_event.append(ab.iloc[pos + days[0]:pos + days[1] + 1].sum(min_count=1))
    if not per_event:
        return pd.DataFrame()
    M = pd.concat(per_event, axis=1)
    n = M.notna().sum(axis=1)
    mean = M.mean(axis=1)
    sd = M.std(axis=1)
    t = mean / (sd / np.sqrt(n)).replace(0, np.nan)
    hit = (M > 0).sum(axis=1) / n
    out = pd.DataFrame({"avg_move_%": 100 * (np.exp(mean) - 1), "t_stat": t, "events": n,
                        "up_share": hit})
    return out[out["events"] >= min_events].sort_values("avg_move_%")


def event_vs_normal_moves(ticker_rets: pd.Series, dates: list) -> dict:
    """Options view: how big are event-day moves compared to a normal day?"""
    idx = ticker_rets.index
    moves = []
    for d in dates:
        pos = idx.searchsorted(pd.Timestamp(d))
        if pos < len(idx):
            moves.append(abs(ticker_rets.iloc[pos]))
    moves = pd.Series(moves).dropna()
    normal = ticker_rets.abs().median()
    return {"events": len(moves), "avg_abs_event_move_%": 100 * moves.mean() if len(moves) else np.nan,
            "median_abs_event_move_%": 100 * moves.median() if len(moves) else np.nan,
            "normal_day_abs_move_%": 100 * normal,
            "ratio": moves.mean() / normal if len(moves) and normal > 0 else np.nan}
