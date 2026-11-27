"""Insider buying and selling from the SEC's free Insider Transactions Data Sets
(Forms 3/4/5, quarterly files since 2006). Only open-market purchases (code P) and sales
(code S) are kept: option exercises, grants and gifts say little about what insiders think.

Features (prefix ins_), each counted from the day after the filing became public:
  ins_buy_value_90 / ins_sell_value_90   $ bought / sold by insiders in the last 90 days (log10)
  ins_net_ratio_90     (buys − sells) / (buys + sells): +1 = only buying, −1 = only selling
  ins_buyers_90        distinct insiders who bought in the last 90 days
  ins_exec_buy_90      the CEO/CFO/President bought in the last 90 days (1/0)
  ins_days_since_buy   days since the last insider purchase (capped at 365)
Event type insider_buy: a cluster (2+ insiders within 10 days) or a single buy ≥ $250k.

SEC asks automated downloads to identify themselves: set SEC_USER_AGENT in .env to
"Your Name your@email.com".
"""
from __future__ import annotations

import io
import time
import zipfile
from typing import Callable

import numpy as np
import pandas as pd
import requests

from . import config

URL = "https://www.sec.gov/files/structureddata/data/insider-transactions-data-sets/{y}q{q}_form345.zip"
TRADES_FILE = config.DATA_DIR / "insider_trades.parquet"
EXEC_TITLES = r"chief executive|\bceo\b|chief financial|\bcfo\b|president|chairman|chair\b"


def _quarters(start_year: int) -> list[tuple[int, int]]:
    now = pd.Timestamp.now()
    out = []
    for y in range(start_year, now.year + 1):
        for q in range(1, 5):
            q_end = pd.Timestamp(year=y, month=3 * q, day=1) + pd.offsets.MonthEnd(0)
            if q_end + pd.Timedelta(days=10) < now:    # SEC posts a few days after quarter end
                out.append((y, q))
    return out


def parse_zip(raw: bytes) -> pd.DataFrame:
    z = zipfile.ZipFile(io.BytesIO(raw))
    names = {n.split("/")[-1].upper(): n for n in z.namelist()}

    def read(name, cols):
        with z.open(names[name]) as f:
            df = pd.read_csv(f, sep="\t", dtype=str, usecols=lambda c: c in cols, quoting=3,
                             on_bad_lines="skip", encoding_errors="replace")
        return df

    sub = read("SUBMISSION.TSV", {"ACCESSION_NUMBER", "FILING_DATE", "DOCUMENT_TYPE", "ISSUERTRADINGSYMBOL"})
    own = read("REPORTINGOWNER.TSV", {"ACCESSION_NUMBER", "RPTOWNERCIK", "RPTOWNERNAME",
                                      "RPTOWNER_RELATIONSHIP", "RPTOWNER_TITLE"})
    tr = read("NONDERIV_TRANS.TSV", {"ACCESSION_NUMBER", "TRANS_DATE", "TRANS_CODE", "TRANS_SHARES",
                                     "TRANS_PRICEPERSHARE", "TRANS_ACQUIRED_DISP_CD"})
    tr = tr[tr["TRANS_CODE"].isin(["P", "S"])]
    sub = sub[sub["DOCUMENT_TYPE"].isin(["4", "4/A"])]
    own = own.drop_duplicates("ACCESSION_NUMBER")
    df = tr.merge(sub, on="ACCESSION_NUMBER").merge(own, on="ACCESSION_NUMBER", how="left")
    shares = pd.to_numeric(df["TRANS_SHARES"], errors="coerce")
    price = pd.to_numeric(df["TRANS_PRICEPERSHARE"], errors="coerce")
    rel = df["RPTOWNER_RELATIONSHIP"].fillna("").str.upper()
    title = df["RPTOWNER_TITLE"].fillna("")
    out = pd.DataFrame({
        "ticker": df["ISSUERTRADINGSYMBOL"].fillna("").str.upper().str.strip().str.replace(".", "-", regex=False),
        "filing_date": pd.to_datetime(df["FILING_DATE"], format="%d-%b-%Y", errors="coerce"),
        "trans_date": pd.to_datetime(df["TRANS_DATE"], format="%d-%b-%Y", errors="coerce"),
        "side": np.where(df["TRANS_CODE"] == "P", "buy", "sell"),
        "shares": shares, "price": price, "value": shares * price,
        "owner": df["RPTOWNERCIK"].fillna(df["RPTOWNERNAME"]),
        "owner_name": df["RPTOWNERNAME"],
        "is_officer": rel.str.contains("OFFICER"), "is_director": rel.str.contains("DIRECTOR"),
        "is_10pct": rel.str.contains("TENPERCENT"),
        "is_exec": title.str.contains(EXEC_TITLES, case=False, regex=True),
    })
    out = out[(out["ticker"] != "") & out["filing_date"].notna() & (out["value"] > 0)]
    # drop obvious data-entry errors (price per share off by 1000×)
    return out[out["value"] < 5e9]


def update_insiders(log: Callable = print, refresh_last: int = 1) -> pd.DataFrame:
    """Downloads each quarter once (≈10 MB each, ~80 files since 2006: ~1 GB total, 10–20 min the
    first time). The most recent `refresh_last` quarters are re-downloaded in case of updates."""
    ua = config.SEC_USER_AGENT or "stock-analyzer research contact@example.com"
    if not config.SEC_USER_AGENT:
        log("  Tip: set SEC_USER_AGENT='Your Name your@email.com' in .env — the SEC may block anonymous downloads.")
    qs = _quarters(config.INSIDER_START_YEAR)
    done = {p.stem for p in config.INSIDERS_DIR.glob("*.parquet")}
    todo = [(y, q) for y, q in qs if f"{y}q{q}" not in done] + [x for x in qs[-refresh_last:]]
    todo = list(dict.fromkeys(todo))
    log(f"Insider trades: {len(todo)} quarterly files to download ({len(done)} already saved)")
    for y, q in todo:
        for attempt in range(3):
            try:
                r = requests.get(URL.format(y=y, q=q), headers={"User-Agent": ua, "Accept-Encoding": "gzip, deflate"},
                                 timeout=120)
                if r.status_code == 404:
                    break
                if r.status_code == 403:
                    log("  SEC refused the download (403). Set SEC_USER_AGENT in .env to your name and email.")
                    return load_trades()
                r.raise_for_status()
                df = parse_zip(r.content)
                df.to_parquet(config.INSIDERS_DIR / f"{y}q{q}.parquet", index=False)
                log(f"  {y} Q{q}: {len(df):,} open-market trades")
                break
            except Exception as e:
                log(f"  {y} Q{q}: {type(e).__name__} {e}; retrying")
                time.sleep(10 * (attempt + 1))
        time.sleep(0.5)   # SEC allows ~10 requests/second; stay far below
    return combine()


def combine() -> pd.DataFrame:
    parts = [pd.read_parquet(p) for p in sorted(config.INSIDERS_DIR.glob("*.parquet"))]
    if not parts:
        return pd.DataFrame()
    df = pd.concat(parts, ignore_index=True).drop_duplicates()
    df.to_parquet(TRADES_FILE, index=False)
    return df


def load_trades() -> pd.DataFrame:
    if TRADES_FILE.exists():
        return pd.read_parquet(TRADES_FILE)
    return pd.DataFrame(columns=["ticker", "filing_date", "trans_date", "side", "shares", "price", "value",
                                 "owner", "owner_name", "is_officer", "is_director", "is_10pct", "is_exec"])


# ======================================================================= features
def insider_features(rows: pd.DataFrame, trades: pd.DataFrame, window: int = 90) -> pd.DataFrame:
    """rows: (ticker, date). Exact rolling windows: each trade adds its value on the day after
    filing and removes it `window` days later; a cumulative sum read as-of each row's date."""
    out = pd.DataFrame(index=rows.index)
    cols = ["ins_buy_value_90", "ins_sell_value_90", "ins_net_ratio_90", "ins_buyers_90",
            "ins_exec_buy_90", "ins_days_since_buy"]
    if trades is None or trades.empty:
        return out
    t = trades[trades["ticker"].isin(set(rows["ticker"]))].copy()
    if t.empty:
        for c in cols:
            out[c] = np.float32(np.nan) if c != "ins_days_since_buy" else np.float32(365)
        return out
    t["avail"] = t["filing_date"].dt.normalize() + pd.Timedelta(days=1)
    t["buy"] = np.where(t["side"] == "buy", t["value"], 0.0)
    t["sell"] = np.where(t["side"] == "sell", t["value"], 0.0)
    # one "buyer event" per owner per filing day
    t["buyer"] = ((t["side"] == "buy") & ~t.duplicated(["ticker", "owner", "avail", "side"])).astype(float)
    t["exec_buy"] = ((t["side"] == "buy") & t["is_exec"].fillna(False)).astype(float)
    daily = t.groupby(["ticker", "avail"])[["buy", "sell", "buyer", "exec_buy"]].sum().reset_index()
    plus = daily.rename(columns={"avail": "d"})
    minus = plus.copy()
    minus["d"] = minus["d"] + pd.Timedelta(days=window)
    minus[["buy", "sell", "buyer", "exec_buy"]] *= -1
    ev = pd.concat([plus, minus]).groupby(["ticker", "d"]).sum().sort_index()
    ev = ev.groupby(level=0).cumsum().reset_index().sort_values("d")
    left = rows[["ticker", "date"]].reset_index().sort_values("date")
    m = pd.merge_asof(left, ev, left_on="date", right_on="d", by="ticker", direction="backward")
    m = m.set_index("index").reindex(rows.index)
    buy = m["buy"].clip(lower=0).fillna(0)
    sell = m["sell"].clip(lower=0).fillna(0)
    out["ins_buy_value_90"] = np.log10(buy + 1).astype(np.float32)
    out["ins_sell_value_90"] = np.log10(sell + 1).astype(np.float32)
    with np.errstate(invalid="ignore", divide="ignore"):
        out["ins_net_ratio_90"] = np.where(buy + sell > 0, (buy - sell) / (buy + sell), np.nan).astype(np.float32)
    out["ins_buyers_90"] = m["buyer"].clip(lower=0).fillna(0).round().astype(np.float32)
    out["ins_exec_buy_90"] = (m["exec_buy"].fillna(0) > 0.5).astype(np.float32)
    # days since last purchase
    buys = daily[daily["buy"] > 0][["ticker", "avail"]].sort_values("avail")
    mb = pd.merge_asof(left, buys.rename(columns={"avail": "last_buy"}), left_on="date", right_on="last_buy",
                       by="ticker", direction="backward").set_index("index").reindex(rows.index)
    out["ins_days_since_buy"] = ((mb["date"] - mb["last_buy"]).dt.days.clip(upper=365).fillna(365)).astype(np.float32)
    return out


def insider_events(min_value: float = 250_000, cluster_days: int = 10) -> pd.DataFrame:
    """Notable insider purchases as company events (for the Events page)."""
    t = load_trades()
    if t.empty:
        return pd.DataFrame(columns=["date", "type", "ticker", "title", "notes"])
    b = t[t["side"] == "buy"].copy()
    b["day"] = b["filing_date"].dt.normalize() + pd.Timedelta(days=1)
    g = b.groupby(["ticker", "day"]).agg(value=("value", "sum"), owners=("owner", "nunique"),
                                         execs=("is_exec", "any")).reset_index().sort_values(["ticker", "day"])
    if g.empty:
        return pd.DataFrame(columns=["date", "type", "ticker", "title", "notes"])
    g = g.sort_values(["ticker", "day"]).reset_index(drop=True)
    roll = g.groupby("ticker", sort=True).rolling(f"{cluster_days}D", on="day")["owners"].sum()
    g["owners_10d"] = roll.values          # same (ticker, day) order as g
    hit = g[(g["value"] >= min_value) | (g["owners_10d"] >= 2)]
    return pd.DataFrame({
        "date": hit["day"], "type": "insider_buy", "ticker": hit["ticker"],
        "title": hit["ticker"] + " insider buying " + hit["value"].map(lambda v: f"${v / 1e6:.2f}M"),
        "notes": np.where(hit["execs"], "CEO/CFO/President bought", hit["owners_10d"].astype(int).astype(str)
                          + " insiders in 10 days")})
