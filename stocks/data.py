"""Downloads and local storage: NYSE universe, Yahoo prices, FRED economy data, earnings dates.

Everything is saved under data/ as parquet so the app and the models never re-download
what they already have. Run `python scripts/download_data.py` on your Mac.
"""
from __future__ import annotations

import io
import time
from typing import Callable, Iterable

import numpy as np
import pandas as pd
import requests

from . import config

FIELDS = ["open", "high", "low", "close", "volume"]
UA = {"User-Agent": "Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) stock-analyzer"}


# ======================================================================= universe
NASDAQTRADER_OTHER = "https://www.nasdaqtrader.com/dynamic/SymDir/otherlisted.txt"
NASDAQTRADER_NASDAQ = "https://www.nasdaqtrader.com/dynamic/SymDir/nasdaqlisted.txt"
_JUNK = (r"warrant|\bright[s]?\b|\bunits?\b|preferred|% |\bnotes? due|debenture|"
         r"depositary shares,? each represent|\bpfd\b|trust preferred|subordinated")


def parse_symbol_file(text: str, exchange_col: str | None) -> pd.DataFrame:
    df = pd.read_csv(io.StringIO(text), sep="|", dtype=str)
    df = df[~df.iloc[:, 0].str.startswith("File Creation Time", na=False)]
    sym_col = "ACT Symbol" if "ACT Symbol" in df.columns else "Symbol"
    out = pd.DataFrame({
        "symbol": df[sym_col].str.strip(),
        "name": df["Security Name"].str.strip(),
        "exchange": df[exchange_col].str.strip() if exchange_col else "Q",
        "etf": df.get("ETF", pd.Series("N", index=df.index)).fillna("N").str.strip(),
        "test": df.get("Test Issue", pd.Series("N", index=df.index)).fillna("N").str.strip(),
    })
    return out


def clean_universe(df: pd.DataFrame, exchanges: Iterable[str], include_etfs: bool) -> pd.DataFrame:
    df = df[df["exchange"].isin(list(exchanges)) & (df["test"] != "Y")].copy()
    if not include_etfs:
        df = df[df["etf"] != "Y"]
    df = df[~df["symbol"].str.contains(r"[\$\^=]", regex=True, na=True)]
    df = df[~df["name"].str.contains(_JUNK, case=False, regex=True, na=False)]
    # Class shares: BRK.B on the exchange is BRK-B on Yahoo
    df["ticker"] = df["symbol"].str.replace(".", "-", regex=False)
    df = df[~df["ticker"].str.contains(r"-(?:WS|W|U|R|RT)$", regex=True)]
    return df.drop_duplicates("ticker")[["ticker", "name", "exchange", "etf"]].sort_values("ticker")


def update_universe(log: Callable = print) -> pd.DataFrame:
    frames = []
    r = requests.get(NASDAQTRADER_OTHER, headers=UA, timeout=30)
    r.raise_for_status()
    frames.append(parse_symbol_file(r.text, "Exchange"))
    if "Q" in config.EXCHANGES:
        r = requests.get(NASDAQTRADER_NASDAQ, headers=UA, timeout=30)
        r.raise_for_status()
        frames.append(parse_symbol_file(r.text, None))
    uni = clean_universe(pd.concat(frames), config.EXCHANGES, config.INCLUDE_ETFS)
    custom = config.DATA_DIR / "universe_extra.csv"   # optional: your own extra tickers
    if custom.exists():
        extra = pd.read_csv(custom)
        extra["ticker"] = extra["ticker"].str.upper().str.strip()
        for c in ("name", "exchange", "etf"):
            if c not in extra:
                extra[c] = "" if c != "etf" else "N"
        uni = pd.concat([uni, extra[uni.columns]]).drop_duplicates("ticker")
    uni.to_csv(config.UNIVERSE_FILE, index=False)
    log(f"Universe: {len(uni):,} tickers ({', '.join(config.EXCHANGES)})")
    return uni


def load_universe() -> pd.DataFrame:
    if config.UNIVERSE_FILE.exists():
        return pd.read_csv(config.UNIVERSE_FILE, keep_default_na=False)
    return pd.DataFrame(columns=["ticker", "name", "exchange", "etf"])


# ======================================================================= Yahoo
def _tidy_download(raw: pd.DataFrame, tickers: list[str]) -> pd.DataFrame:
    """yfinance returns wide frames whose column layout changes between versions.
    Turn whatever we get into long rows: date, ticker, open, high, low, close, volume."""
    if raw is None or raw.empty:
        return pd.DataFrame(columns=["date", "ticker"] + FIELDS)
    names = {"Open", "High", "Low", "Close", "Adj Close", "Volume"}
    if isinstance(raw.columns, pd.MultiIndex):
        lvl = 0 if set(raw.columns.get_level_values(0)) & names else 1
        long = raw.stack(level=1 - lvl, future_stack=True) if _has_future_stack() \
            else raw.stack(level=1 - lvl)
        long.index.names = ["date", "ticker"]
        long = long.reset_index()
    else:
        long = raw.reset_index().rename(columns={raw.index.name or "index": "date"})
        long["ticker"] = tickers[0]
    long.columns = [str(c).lower().replace(" ", "_") for c in long.columns]
    if "datetime" in long.columns and "date" not in long.columns:
        long = long.rename(columns={"datetime": "date"})
    if "close" not in long and "adj_close" in long:
        long["close"] = long["adj_close"]
    for c in FIELDS:
        if c not in long:
            long[c] = np.nan
    long = long.dropna(subset=["close"])
    d = pd.to_datetime(long["date"])
    if getattr(d.dt, "tz", None) is not None:
        d = d.dt.tz_convert("America/New_York").dt.tz_localize(None)
    long["date"] = d
    long[["open", "high", "low", "close"]] = long[["open", "high", "low", "close"]].astype("float32")
    long["volume"] = long["volume"].astype("float64")
    return long[["date", "ticker"] + FIELDS]


def _has_future_stack() -> bool:
    import inspect
    return "future_stack" in inspect.signature(pd.DataFrame.stack).parameters


def yahoo_download(tickers: list[str], interval: str = "1d", start: str | None = None,
                   period: str | None = None, batch: int = 50, log: Callable = print,
                   tries: int = 4) -> pd.DataFrame:
    """Download in batches. Tickers that Yahoo rate-limits are retried after a pause;
    tickers with no data (delisted, renamed) are skipped and counted."""
    import logging

    import yfinance as yf

    logging.getLogger("yfinance").setLevel(logging.CRITICAL)   # we print our own summary
    out = []
    no_data, gave_up = [], []
    for i in range(0, len(tickers), batch):
        pending = tickers[i:i + batch]
        for attempt in range(tries):
            try:
                raw = yf.download(pending, start=start, period=period, interval=interval,
                                  auto_adjust=True, actions=False, group_by="column",
                                  threads=4, progress=False)
                got = _tidy_download(raw, pending)
            except Exception as e:
                got = pd.DataFrame(columns=["date", "ticker"] + FIELDS)
                log(f"  {type(e).__name__}: {e}")
            out.append(got)
            returned = set(got["ticker"].unique())
            errors = dict(getattr(getattr(yf, "shared", None), "_ERRORS", {}) or {})
            missing = [t for t in pending if t not in returned]
            limited = [t for t in missing if "rate" in str(errors.get(t, "")).lower()
                       or "too many" in str(errors.get(t, "")).lower()]
            if not errors and len(missing) > len(pending) // 2:
                limited = missing          # can't tell why: assume throttling if most failed
            no_data += [t for t in missing if t not in limited]
            if not limited:
                break
            if attempt == tries - 1:
                gave_up += limited
                break
            wait = 60 * (attempt + 1)
            log(f"  Yahoo is rate-limiting ({len(limited)} tickers). Waiting {wait}s, then retrying ...")
            time.sleep(wait)
            pending = limited
        if (i // batch) % 10 == 0:
            log(f"  {interval}: {min(i + batch, len(tickers)):,}/{len(tickers):,} tickers")
        time.sleep(config.REQUEST_PAUSE)
    if no_data:
        log(f"  {len(no_data)} tickers had no data on Yahoo (new listings, renamed or delisted) — skipped")
    if gave_up:
        log(f"  {len(gave_up)} tickers still rate-limited after retries: re-run the same command later "
            f"to fill them in ({', '.join(gave_up[:10])}{' ...' if len(gave_up) > 10 else ''})")
    out = [o for o in out if not o.empty]
    if not out:
        return pd.DataFrame(columns=["date", "ticker"] + FIELDS)
    return pd.concat(out, ignore_index=True)


def _merge_incremental(old: pd.DataFrame, new: pd.DataFrame, log: Callable) -> tuple[pd.DataFrame, list[str]]:
    """Append new bars. If a split or dividend changed the adjusted history of a ticker,
    the overlap won't match: return those tickers so they get a full re-download."""
    if old.empty:
        return new, []
    both = old.merge(new[["date", "ticker", "close"]], on=["date", "ticker"], suffixes=("", "_new"))
    both["diff"] = (both["close_new"] / both["close"] - 1).abs()
    stale = sorted(both.loc[both["diff"] > 0.005, "ticker"].unique())
    merged = pd.concat([old, new], ignore_index=True)
    merged = merged.drop_duplicates(["date", "ticker"], keep="last")
    return merged.sort_values(["ticker", "date"]).reset_index(drop=True), stale


def update_prices(kind: str = "daily", tickers: list[str] | None = None, full: bool = False,
                  log: Callable = print) -> pd.DataFrame:
    """kind: 'daily' (since 2000) or 'hourly' (last 730 days, Yahoo's limit)."""
    path = config.DAILY_FILE if kind == "daily" else config.HOURLY_FILE
    interval = "1d" if kind == "daily" else "1h"
    if tickers is None:
        tickers = load_universe()["ticker"].tolist()
    old = pd.read_parquet(path) if path.exists() and not full else pd.DataFrame()
    have = set(old["ticker"].unique()) if not old.empty else set()
    missing = [t for t in tickers if t not in have]
    existing = [t for t in tickers if t in have]
    frames = []
    if missing:
        log(f"Full history for {len(missing):,} {kind} tickers ...")
        if kind == "daily":
            frames.append(yahoo_download(missing, "1d", start=config.HISTORY_START, log=log))
        else:
            frames.append(yahoo_download(missing, "1h", period=f"{config.HOURLY_DAYS}d", batch=50, log=log))
    stale: list[str] = []
    if existing:
        last = old["date"].max()
        start = (last - pd.Timedelta(days=7)).strftime("%Y-%m-%d")
        log(f"Updating {len(existing):,} {kind} tickers since {start} ...")
        new = yahoo_download(existing, interval, start=start, batch=100 if kind == "daily" else 50, log=log)
        old, stale = _merge_incremental(old, new, log)
    merged = pd.concat([old] + frames, ignore_index=True) if frames else old
    if stale:
        log(f"{len(stale)} tickers had a split/dividend adjustment; re-downloading their history")
        redo = yahoo_download(stale, interval,
                              start=config.HISTORY_START if kind == "daily" else None,
                              period=None if kind == "daily" else f"{config.HOURLY_DAYS}d", log=log)
        merged = pd.concat([merged[~merged["ticker"].isin(stale)], redo], ignore_index=True)
    if merged.empty:
        log("No price data came back. Check your internet connection.")
        return merged
    if kind == "hourly":   # keep the file from growing forever
        merged = merged[merged["date"] >= merged["date"].max() - pd.Timedelta(days=config.HOURLY_DAYS + 30)]
    merged = merged.drop_duplicates(["date", "ticker"], keep="last").sort_values(["ticker", "date"])
    merged.reset_index(drop=True).to_parquet(path, index=False)
    log(f"Saved {kind}: {merged['ticker'].nunique():,} tickers, {len(merged):,} rows, "
        f"{merged['date'].min():%Y-%m-%d} → {merged['date'].max():%Y-%m-%d %H:%M}")
    return merged


def update_yahoo_context(log: Callable = print) -> pd.DataFrame:
    syms = list(config.YAHOO_CONTEXT.values())
    raw = yahoo_download(syms, "1d", start=config.HISTORY_START, batch=50, log=log)
    inv = {v: k for k, v in config.YAHOO_CONTEXT.items()}
    raw["name"] = raw["ticker"].map(inv)
    raw.to_parquet(config.YAHOO_CONTEXT_FILE, index=False)
    log(f"Market/cross-asset series: {raw['name'].nunique()} of {len(syms)}")
    # hourly SPY/VIX for the hourly model's market features
    hr = yahoo_download(["SPY", "^VIX", "QQQ"], "1h", period=f"{config.HOURLY_DAYS}d", log=log)
    hr.to_parquet(config.CONTEXT_DIR / "hourly_market.parquet", index=False)
    return raw


# ======================================================================= FRED
FRED_CSV = "https://fred.stlouisfed.org/graph/fredgraph.csv?id={sid}&cosd={start}"


def fetch_fred(series_id: str, start: str = "1995-01-01") -> pd.Series:
    r = requests.get(FRED_CSV.format(sid=series_id, start=start), headers=UA, timeout=60)
    r.raise_for_status()
    df = pd.read_csv(io.StringIO(r.text))
    date_col = df.columns[0]
    s = pd.to_numeric(df[df.columns[1]], errors="coerce")
    s.index = pd.to_datetime(df[date_col])
    return s.dropna()


def update_fred(extra: dict | None = None, log: Callable = print) -> pd.DataFrame:
    series = dict(config.FRED_SERIES)
    series.update(load_extra_fred())
    if extra:
        series.update(extra)
    rows = []
    for name, (sid, lag, kind) in series.items():
        try:
            s = fetch_fred(sid)
            rows.append(pd.DataFrame({"name": name, "series_id": sid, "obs_date": s.index,
                                      "value": s.values, "lag_days": lag, "kind": kind}))
        except Exception as e:
            log(f"  FRED {sid}: {e}")
        time.sleep(0.3)
    df = pd.concat(rows, ignore_index=True)
    df.to_parquet(config.FRED_FILE, index=False)
    log(f"FRED economy series: {df['name'].nunique()} of {len(series)}")
    return df


def load_extra_fred() -> dict:
    """data/fred_extra.csv lets you add FRED series without touching code:
    columns name,series_id,lag_days,kind"""
    p = config.DATA_DIR / "fred_extra.csv"
    if not p.exists():
        return {}
    df = pd.read_csv(p)
    return {r["name"]: (r["series_id"], int(r.get("lag_days", 30)), r.get("kind", "level"))
            for _, r in df.iterrows()}


# ======================================================================= earnings
EARNINGS_CHECKED = config.DATA_DIR / "earnings_checked_v2.csv"
SEC_TICKERS = "https://www.sec.gov/files/company_tickers.json"
SEC_SUBMISSIONS = "https://data.sec.gov/submissions/{name}"


def _sec_headers() -> dict:
    return {"User-Agent": config.SEC_USER_AGENT or "stock-analyzer research contact@example.com",
            "Accept-Encoding": "gzip, deflate"}


def _sec_get(url: str, log: Callable, tries: int = 3):
    for attempt in range(tries):
        try:
            r = requests.get(url, headers=_sec_headers(), timeout=30)
            if r.status_code == 404:
                return None
            if r.status_code in (403, 429):
                if attempt == 0 and r.status_code == 403 and not config.SEC_USER_AGENT:
                    log("  SEC refused the request: set SEC_USER_AGENT in .env to your name and email.")
                time.sleep(10 * (attempt + 1))
                continue
            r.raise_for_status()
            return r.json()
        except Exception:
            time.sleep(3 * (attempt + 1))
    return None


def sec_earnings_dates(cik: int, log: Callable = print) -> pd.DataFrame:
    """Earnings release dates from SEC 8-K filings with Item 2.02 ('Results of Operations and
    Financial Condition'), the filing companies make when they announce quarterly results.
    The filing time decides the trading session: after 4 pm → next day's session."""
    name = f"CIK{int(cik):010d}.json"
    js = _sec_get(SEC_SUBMISSIONS.format(name=name), log)
    if not js:
        return pd.DataFrame(columns=["date"])
    pages = [js.get("filings", {}).get("recent", {})]
    for f in js.get("filings", {}).get("files", []):
        more = _sec_get(SEC_SUBMISSIONS.format(name=f["name"]), log)
        if more:
            pages.append(more)
        time.sleep(0.15)
    rows = []
    for pg in pages:
        forms = pg.get("form", [])
        items = pg.get("items", [""] * len(forms))
        acc = pg.get("acceptanceDateTime", [""] * len(forms))
        fdate = pg.get("filingDate", [""] * len(forms))
        for f, it, a, d in zip(forms, items, acc, fdate):
            if f in ("8-K", "8-K/A") and "2.02" in str(it):
                rows.append((a or d, d))
    if not rows:
        return pd.DataFrame(columns=["date"])
    df = pd.DataFrame(rows, columns=["accepted", "filed"])
    ts = pd.to_datetime(df["accepted"].str.replace("Z", "", regex=False), errors="coerce")
    ts = ts.fillna(pd.to_datetime(df["filed"], errors="coerce"))
    after_close = (ts.dt.hour >= 16)
    session = ts.dt.normalize() + pd.to_timedelta(after_close.astype(int), unit="D")
    session = session + pd.offsets.BDay(0)     # weekend → Monday
    out = pd.DataFrame({"date": session}).dropna().drop_duplicates()
    # amended/duplicate filings a few days apart are the same report
    out = out.sort_values("date")
    out = out[out["date"].diff().dt.days.fillna(999) > 20]
    return out


def upcoming_earnings_yahoo(ticker: str):
    """Next scheduled earnings date from Yahoo's calendar (an API call, not page scraping)."""
    import yfinance as yf

    try:
        cal = yf.Ticker(ticker).calendar
        dates = cal.get("Earnings Date") if isinstance(cal, dict) else None
        if dates:
            ds = [pd.Timestamp(d) for d in (dates if isinstance(dates, (list, tuple)) else [dates])]
            fut = [d for d in ds if d >= pd.Timestamp.now().normalize()]
            return min(fut) if fut else None
    except Exception:
        return None
    return None


def update_earnings(tickers: list[str] | None = None, limit: int = 100, log: Callable = print,
                    max_age_days: int = 7) -> pd.DataFrame:
    """Past earnings dates from SEC filings (since ~2001) + the next scheduled date from Yahoo.
    One SEC request per company (≈20–30 min the first time), resumable; companies checked in the
    last `max_age_days` are skipped. Foreign companies that don't file 8-Ks get upcoming dates only."""
    import logging

    logging.getLogger("yfinance").setLevel(logging.CRITICAL)
    if tickers is None:
        tickers = load_universe()["ticker"].tolist()
    cmap = _sec_get(SEC_TICKERS, log) or {}
    cik = {v["ticker"].upper().replace(".", "-"): int(v["cik_str"]) for v in cmap.values()} if cmap else {}
    if not cik:
        log("  Couldn't load the SEC ticker list; only upcoming dates from Yahoo will be fetched.")
    old = pd.read_parquet(config.EARNINGS_FILE) if config.EARNINGS_FILE.exists() else pd.DataFrame(
        columns=["ticker", "date", "eps_estimate", "eps_actual", "surprise_pct", "fetched"])
    checked = pd.read_csv(EARNINGS_CHECKED, parse_dates=["checked"]) if EARNINGS_CHECKED.exists() \
        else pd.DataFrame(columns=["ticker", "checked", "has_earnings"])
    cutoff = pd.Timestamp.now() - pd.Timedelta(days=max_age_days)
    fresh = set(checked.loc[pd.to_datetime(checked["checked"]) > cutoff, "ticker"])
    todo = [t for t in tickers if t not in fresh]
    log(f"Earnings dates: {len(todo):,} companies to check ({len(fresh):,} already up to date)")
    got, marks, none_found = [], [], 0
    for i, t in enumerate(todo):
        past = sec_earnings_dates(cik[t], log) if t in cik else pd.DataFrame(columns=["date"])
        nxt = upcoming_earnings_yahoo(t)
        dates = list(past["date"])
        if nxt is not None and pd.notna(nxt):
            dates.append(pd.Timestamp(nxt).normalize())
        if dates:
            got.append(pd.DataFrame({"ticker": t, "date": pd.to_datetime(dates), "eps_estimate": np.nan,
                                     "eps_actual": np.nan, "surprise_pct": np.nan, "fetched": pd.Timestamp.now()}))
            marks.append((t, pd.Timestamp.now(), True))
        else:
            none_found += 1
            marks.append((t, pd.Timestamp.now(), False))
        if (i + 1) % 100 == 0 or i == len(todo) - 1:
            log(f"  earnings {i + 1:,}/{len(todo):,} ({none_found:,} without earnings filings, e.g. funds)")
            old = _save_earnings(old, got)
            got = []
            checked = _save_checked(checked, marks)
            marks = []
        time.sleep(0.15)   # SEC allows 10 requests/second; stay well below
    _save_checked(checked, marks)
    return _save_earnings(old, got)


def _save_checked(checked: pd.DataFrame, marks: list) -> pd.DataFrame:
    if not marks:
        return checked
    new = pd.DataFrame(marks, columns=["ticker", "checked", "has_earnings"])
    keep = checked[~checked["ticker"].isin(new["ticker"])]
    out = pd.concat([keep, new], ignore_index=True) if not keep.empty else new
    out.to_csv(EARNINGS_CHECKED, index=False)
    return out


def _num_col(df: pd.DataFrame, name: str) -> pd.Series:
    return pd.to_numeric(df[name], errors="coerce") if name in df else pd.Series(np.nan, index=df.index)


def _save_earnings(old: pd.DataFrame, got: list) -> pd.DataFrame:
    if not got:
        return old
    new = pd.concat(got, ignore_index=True)
    keep = old[~old["ticker"].isin(new["ticker"].unique())]
    df = pd.concat([keep, new], ignore_index=True) if not keep.empty else new
    df = df.drop_duplicates(["ticker", "date"]).sort_values(["ticker", "date"])
    df.to_parquet(config.EARNINGS_FILE, index=False)
    return df


# ======================================================================= loaders
def load_daily(tickers: list[str] | None = None, start: str | None = None) -> pd.DataFrame:
    if not config.DAILY_FILE.exists():
        return pd.DataFrame(columns=["date", "ticker"] + FIELDS)
    filters = []
    if tickers is not None:
        filters.append(("ticker", "in", list(tickers)))
    if start is not None:
        filters.append(("date", ">=", pd.Timestamp(start)))
    return pd.read_parquet(config.DAILY_FILE, filters=filters or None)


def load_hourly(tickers: list[str] | None = None) -> pd.DataFrame:
    if not config.HOURLY_FILE.exists():
        return pd.DataFrame(columns=["date", "ticker"] + FIELDS)
    filters = [("ticker", "in", list(tickers))] if tickers is not None else None
    return pd.read_parquet(config.HOURLY_FILE, filters=filters)


def load_yahoo_context() -> pd.DataFrame:
    """Wide table of closes: index date, one column per context series."""
    if not config.YAHOO_CONTEXT_FILE.exists():
        return pd.DataFrame()
    df = pd.read_parquet(config.YAHOO_CONTEXT_FILE)
    wide = df.pivot_table(index="date", columns="name", values="close")
    return wide.sort_index()


def load_hourly_market() -> pd.DataFrame:
    p = config.CONTEXT_DIR / "hourly_market.parquet"
    if not p.exists():
        return pd.DataFrame()
    df = pd.read_parquet(p)
    return df.pivot_table(index="date", columns="ticker", values="close").sort_index()


def load_fred_raw() -> pd.DataFrame:
    if not config.FRED_FILE.exists():
        return pd.DataFrame(columns=["name", "series_id", "obs_date", "value", "lag_days", "kind"])
    return pd.read_parquet(config.FRED_FILE)


def load_earnings() -> pd.DataFrame:
    if not config.EARNINGS_FILE.exists():
        return pd.DataFrame(columns=["ticker", "date", "eps_estimate", "eps_actual", "surprise_pct"])
    return pd.read_parquet(config.EARNINGS_FILE)


def load_custom_series() -> pd.DataFrame:
    """Every CSV in data/custom_series/ with a date column and a value column becomes a
    series named after the file (e.g. google_trends_recession.csv). Assumed public the
    next day; add a `lag_days` column to change that."""
    out = {}
    for p in sorted(config.CUSTOM_DIR.glob("*.csv")):
        try:
            df = pd.read_csv(p)
            dcol = [c for c in df.columns if "date" in c.lower() or "week" in c.lower() or "month" in c.lower()]
            dcol = dcol[0] if dcol else df.columns[0]
            vcol = [c for c in df.columns if c != dcol and c != "lag_days"][0]
            lag = int(df["lag_days"].iloc[0]) if "lag_days" in df else 1
            s = pd.to_numeric(df[vcol], errors="coerce")
            s.index = pd.to_datetime(df[dcol], errors="coerce") + pd.Timedelta(days=lag)
            out[p.stem] = s[s.index.notna()].dropna().groupby(level=0).last()
        except Exception:
            continue
    return pd.DataFrame(out).sort_index() if out else pd.DataFrame()
