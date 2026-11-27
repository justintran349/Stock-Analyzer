"""Options: daily chain snapshots, implied moves, trade ideas and paper tracking.

Free sources only have *today's* option prices, so this saves a snapshot every day for the most
traded stocks. Over time that becomes your own history of implied volatility and implied moves.

Files (data/options/):
  chains/YYYY-MM-DD.parquet   every contract within ±25% of the stock price, per snapshot day
  summary.parquet             one row per (day, ticker, expiry): at-the-money straddle, implied move,
                              ATM implied volatility, skew, put/call volume and open interest
  ideas.parquet               trade ideas you (or the daily job) chose to track, graded at expiry
"""
from __future__ import annotations

import time
from typing import Callable

import numpy as np
import pandas as pd

from . import config, data

CHAINS_DIR = config.OPTIONS_DIR / "chains"
CHAINS_DIR.mkdir(parents=True, exist_ok=True)
SUMMARY_FILE = config.OPTIONS_DIR / "summary.parquet"
IDEAS_FILE = config.OPTIONS_DIR / "ideas.parquet"
ALWAYS = ["SPY", "QQQ", "IWM"]
MACRO_NAMES = {"fomc": "Fed", "cpi_release": "CPI", "jobs_report": "jobs", "gdp_release": "GDP",
               "pce_release": "PCE", "ppi_release": "PPI", "retail_sales_release": "retail sales"}


def pick_tickers(n: int = config.OPTIONS_TOP_N) -> list[str]:
    """The n most-traded stocks (by average $ volume, last month) plus index ETFs and your
    extra tickers (data/options_watchlist.csv, column ticker)."""
    px = data.load_daily(start=(pd.Timestamp.now() - pd.Timedelta(days=45)).strftime("%Y-%m-%d"))
    out = list(ALWAYS)
    if not px.empty:
        px["dv"] = px["close"] * px["volume"]
        top = px.groupby("ticker")["dv"].mean().sort_values(ascending=False)
        out += [t for t in top.index if t not in out][:n]
    wl = config.DATA_DIR / "options_watchlist.csv"
    if wl.exists():
        out += [t for t in pd.read_csv(wl)["ticker"].str.upper() if t not in out]
    return out


def _mid(bid, ask, last):
    m = np.where((bid > 0) & (ask > 0), (bid + ask) / 2, last)
    return np.where(np.isfinite(m) & (m > 0), m, np.nan)


SQRT_2_PI = np.sqrt(2 / np.pi)     # expected |move| of a normal = 0.798 × σ√T


def summarize_chain(calls: pd.DataFrame, puts: pd.DataFrame, spot: float, days: float | None = None) -> dict:
    """Implied move from the at-the-money straddle, done carefully:

    - only strikes with a real two-sided quote (bid > 0, ask > 0, spread ≤ 60% of mid) are used
    - the straddle is interpolated between the strikes just below and just above the stock
      price, so it's priced *at* the stock price
    - if no good strikes sit within 2.5% of the price (thin expiries), or the straddle disagrees
      wildly with the options' own implied volatility, the move is taken from implied
      volatility instead: 0.798 × IV × √(days/365)
    """
    if not spot or spot <= 0 or calls.empty or puts.empty:
        return {}
    legs = []
    for d in (calls, puts):
        d = d.copy()
        bid, ask = d["bid"].fillna(0).values, d["ask"].fillna(0).values
        d["mid"] = _mid(bid, ask, d["lastPrice"].fillna(0).values)
        d["good"] = (bid > 0) & (ask >= bid) & ((ask - bid) <= 0.6 * np.where(d["mid"] > 0, d["mid"], np.inf))
        legs.append(d.set_index("strike"))
    c, p = legs
    common = c.index.intersection(p.index)
    if common.empty:
        return {}
    both = pd.DataFrame({"c": c.loc[common, "mid"], "p": p.loc[common, "mid"],
                         "good": c.loc[common, "good"] & p.loc[common, "good"],
                         "c_iv": c.loc[common, "impliedVolatility"], "p_iv": p.loc[common, "impliedVolatility"],
                         "c_sp": (c.loc[common, "ask"] - c.loc[common, "bid"]) / c.loc[common, "mid"],
                         "p_sp": (p.loc[common, "ask"] - p.loc[common, "bid"]) / p.loc[common, "mid"]})
    both = both[~both.index.duplicated()].sort_index()
    both["straddle"] = both["c"] + both["p"]
    cand = both[both["good"] & both["straddle"].notna()]
    if cand.empty:
        cand = both[both["straddle"] > 0]
    if cand.empty:
        return {}
    lo = cand[cand.index <= spot]
    hi = cand[cand.index >= spot]
    near = 0.025 * spot
    straddle = np.nan
    method = "straddle"
    if not lo.empty and not hi.empty and spot - lo.index[-1] <= 2 * near and hi.index[0] - spot <= 2 * near:
        k_lo, k_hi = lo.index[-1], hi.index[0]
        w = 0 if k_hi == k_lo else (spot - k_lo) / (k_hi - k_lo)
        straddle = (1 - w) * lo["straddle"].iloc[-1] + w * hi["straddle"].iloc[0]
        rows = cand.loc[[k_lo, k_hi]]
    else:
        k = cand.index[np.argmin(np.abs(cand.index - spot))]
        rows = cand.loc[[k]]
        if abs(k - spot) <= near:
            straddle = cand.loc[k, "straddle"] - abs(spot - k)   # remove the in-the-money part
    ivs = pd.concat([rows["c_iv"], rows["p_iv"]])
    ivs = ivs[(ivs > 0.01) & (ivs < 5)]
    atm_iv = float(ivs.mean()) if len(ivs) else np.nan
    t = max(days or 1, 1) / 365
    iv_move = 100 * SQRT_2_PI * atm_iv * np.sqrt(t) if pd.notna(atm_iv) else np.nan
    move = 100 * straddle / spot if pd.notna(straddle) and straddle > 0 else np.nan
    if days is not None and pd.notna(iv_move) and (pd.isna(move) or not 0.6 <= move / iv_move <= 1.6):
        move, method = iv_move, "iv"
    if pd.isna(move):
        return {}

    def iv_near(df, target):
        df = df[df["impliedVolatility"] > 0.01]
        if df.empty:
            return np.nan
        return float(df.iloc[np.argmin(np.abs(df.index - target))]["impliedVolatility"])

    k_atm = float(rows.index[np.argmin(np.abs(rows.index - spot))])
    return {
        "strike": k_atm, "straddle": spot * move / 100, "implied_move_pct": move, "method": method,
        "iv_move_pct": iv_move, "atm_iv": atm_iv,
        "skew": iv_near(p, 0.95 * spot) - iv_near(c, 1.05 * spot),
        "call_volume": calls["volume"].fillna(0).sum(), "put_volume": puts["volume"].fillna(0).sum(),
        "call_oi": calls["openInterest"].fillna(0).sum(), "put_oi": puts["openInterest"].fillna(0).sum(),
        "spread_pct": float(np.nanmean(rows[["c_sp", "p_sp"]].values)),
    }


def rebuild_summary(log: Callable = print) -> pd.DataFrame:
    """Recompute summary.parquet from the saved chains (after a fix to the implied-move math)."""
    old = load_summary()
    spots = old.groupby(["date", "ticker"])["spot"].first().to_dict() if not old.empty else {}
    out = []
    for f in sorted(CHAINS_DIR.glob("*.parquet")):
        ch = pd.read_parquet(f).rename(columns={"last": "lastPrice", "open_interest": "openInterest",
                                                "iv": "impliedVolatility"})
        day = pd.Timestamp(f.stem)
        px = data.load_daily(tickers=ch["ticker"].unique().tolist(),
                             start=(day - pd.Timedelta(days=10)).strftime("%Y-%m-%d"))
        closes = px[px["date"] <= day].sort_values("date").groupby("ticker")["close"].last().to_dict() \
            if not px.empty else {}
        for (t, e), g in ch.groupby(["ticker", "expiry"]):
            spot = spots.get((day, t)) or closes.get(t)
            days = (pd.Timestamp(e) - day).days
            sm = summarize_chain(g[g["type"] == "C"], g[g["type"] == "P"], spot, days)
            if sm:
                out.append({"date": day, "ticker": t, "expiry": pd.Timestamp(e), "spot": spot, "days": days, **sm})
    df = pd.DataFrame(out)
    if not df.empty:
        df.to_parquet(SUMMARY_FILE, index=False)
        log(f"Rebuilt option summary: {len(df):,} rows, {df['ticker'].nunique()} tickers, "
            f"{(df['method'] == 'iv').mean():.0%} from implied volatility (thin strikes)")
    return df


def snapshot(tickers: list[str] | None = None, log: Callable = print, pause: float = 0.35) -> pd.DataFrame:
    """Save today's chains for up to OPTIONS_MAX_DAYS out. ~300 stocks takes 15–30 minutes."""
    import logging

    import yfinance as yf

    logging.getLogger("yfinance").setLevel(logging.CRITICAL)
    tickers = tickers or pick_tickers()
    today = pd.Timestamp.now().normalize()
    last_close = {}
    px = data.load_daily(tickers=tickers, start=(today - pd.Timedelta(days=10)).strftime("%Y-%m-%d"))
    if not px.empty:
        last_close = px.sort_values("date").groupby("ticker")["close"].last().to_dict()
    chains, summ = [], []
    log(f"Option snapshots: {len(tickers)} tickers")
    for i, t in enumerate(tickers):
        for attempt in range(3):
            try:
                tk = yf.Ticker(t)
                exps = [e for e in tk.options if 0 <= (pd.Timestamp(e) - today).days <= config.OPTIONS_MAX_DAYS]
                spot = None
                try:
                    spot = float(tk.fast_info["lastPrice"])
                except Exception:
                    spot = last_close.get(t)
                for e in exps[:8]:
                    ch = tk.option_chain(e)
                    s = summarize_chain(ch.calls, ch.puts, spot, (pd.Timestamp(e) - today).days)
                    if s:
                        summ.append({"date": today, "ticker": t, "expiry": pd.Timestamp(e), "spot": spot,
                                     "days": (pd.Timestamp(e) - today).days, **s})
                    for side, df in (("C", ch.calls), ("P", ch.puts)):
                        if df.empty or not spot:
                            continue
                        df = df[(df["strike"] > 0.75 * spot) & (df["strike"] < 1.25 * spot)]
                        chains.append(pd.DataFrame({
                            "date": today, "ticker": t, "expiry": pd.Timestamp(e), "type": side,
                            "strike": df["strike"].values, "bid": df["bid"].values, "ask": df["ask"].values,
                            "last": df["lastPrice"].values, "volume": df["volume"].values,
                            "open_interest": df["openInterest"].values, "iv": df["impliedVolatility"].values}))
                    time.sleep(pause)
                break
            except Exception as e:
                if "rate" in f"{type(e).__name__} {e}".lower() or "too many" in str(e).lower():
                    log(f"  Yahoo is rate-limiting; waiting {60 * (attempt + 1)}s ...")
                    time.sleep(60 * (attempt + 1))
                    continue
                break
        if (i + 1) % 50 == 0:
            log(f"  options {i + 1}/{len(tickers)}")
    if chains:
        pd.concat(chains, ignore_index=True).to_parquet(CHAINS_DIR / f"{today:%Y-%m-%d}.parquet", index=False)
    new = pd.DataFrame(summ)
    if not new.empty:
        old = load_summary()
        allx = pd.concat([old[old["date"] != today], new], ignore_index=True) if not old.empty else new
        allx.to_parquet(SUMMARY_FILE, index=False)
        log(f"  saved {new['ticker'].nunique()} tickers, {len(new)} expiries")
    return new


def load_summary() -> pd.DataFrame:
    return pd.read_parquet(SUMMARY_FILE) if SUMMARY_FILE.exists() else pd.DataFrame()


def latest_summary() -> pd.DataFrame:
    s = load_summary()
    return s[s["date"] == s["date"].max()] if not s.empty else s


# ======================================================================= ideas
def historical_move(rets: pd.Series, n_days: int, event_dates: list | None = None) -> float:
    """Average absolute % move over n trading days. With event_dates: only windows that
    contain an event (e.g. past earnings), which is exactly what a straddle over the next
    event would have paid."""
    r = rets.dropna()
    if len(r) < 60 or n_days < 1:
        return np.nan
    cum = r.rolling(n_days).sum()
    if event_dates:
        idx = r.index
        vals = []
        for d in event_dates:
            pos = idx.searchsorted(pd.Timestamp(d))
            # window: from the close before the event to n_days later
            end = pos + n_days - 1
            if pos >= 1 and end < len(idx):
                vals.append(abs(r.iloc[pos:end + 1].sum()))
        if len(vals) >= 3:
            return 100 * (np.exp(np.mean(vals)) - 1)
        return np.nan
    last_year = cum.iloc[-252:].abs()
    return 100 * (np.exp(last_year.mean()) - 1)


def build_ideas(preds: dict, rets: pd.DataFrame, earnings: pd.DataFrame, macro: pd.DataFrame,
                summary: pd.DataFrame | None = None, max_days: int = 45) -> pd.DataFrame:
    """One row per (ticker, nearest expiry after its next event, or ~1 month out if none).

    preds: {"1w": latest weekly predictions, "1m": monthly} (ticker, p_up, exp_move, p_beat)
    rets: wide daily log returns. earnings: ticker, date. macro: scheduled market-wide events.
    """
    s = latest_summary() if summary is None else summary
    if s.empty:
        return pd.DataFrame()
    today = s["date"].max()
    s = s[(s["days"] >= 2) & (s["days"] <= max_days)]
    e = earnings[earnings["date"] > today] if earnings is not None and not earnings.empty else pd.DataFrame(
        columns=["ticker", "date"])
    next_e = e.sort_values("date").groupby("ticker")["date"].first()
    past_e = earnings[earnings["date"] <= today] if earnings is not None and not earnings.empty else e
    rows = []
    for t, g in s.groupby("ticker"):
        g = g.sort_values("days")
        ne = next_e.get(t)
        if ne is not None and pd.notna(ne) and (ne - today).days <= max_days:
            cand = g[g["expiry"] >= ne]
            if cand.empty:
                continue
            row = cand.iloc[0]
            event = f"earnings {ne:%b %d}"
            ev_dates = past_e.loc[past_e["ticker"] == t, "date"].tolist()
        else:
            row = g.iloc[(g["days"] - 30).abs().argsort()[:1]].iloc[0]
            event, ev_dates = "", None
        n_trading = int(np.busday_count(today.date(), row["expiry"].date()))
        r = rets[t] if t in rets.columns else pd.Series(dtype=float)
        hist = historical_move(r, max(n_trading, 1), ev_dates)
        hist_any = historical_move(r, max(n_trading, 1))
        # model: scale the closest horizon's expected move to the days until expiry
        h, h_days = ("1w", 5) if n_trading <= 12 else ("1m", 21)
        p = preds.get(h, pd.DataFrame())
        pr = p[p["ticker"] == t] if not p.empty else p
        model_move = pr["exp_move"].iloc[0] * np.sqrt(max(n_trading, 1) / h_days) if not pr.empty else np.nan
        p_up = pr["p_up"].iloc[0] if not pr.empty else np.nan
        p_beat = pr["p_beat"].iloc[0] if not pr.empty and "p_beat" in pr else np.nan
        macro_in = macro[(macro["date"] > today) & (macro["date"] <= row["expiry"])] if macro is not None else []
        expected = pd.Series([hist if event else hist_any, model_move], dtype=float).mean()
        implied = row["implied_move_pct"]
        ratio = expected / implied if implied and implied > 0 else np.nan
        rows.append({
            "ticker": t, "event": event,
            "macro_events": ", ".join(sorted({MACRO_NAMES.get(x, x) for x in macro_in["type"]})) if len(macro_in) else "",
            "expiry": row["expiry"].date(), "days": int(row["days"]), "spot": row["spot"], "strike": row["strike"],
            "straddle": row["straddle"], "implied_move_pct": implied, "atm_iv": row["atm_iv"],
            "skew": row.get("skew", np.nan), "spread_pct": row.get("spread_pct", np.nan),
            "hist_move_pct": hist if event else hist_any, "model_move_pct": model_move,
            "expected_move_pct": expected, "ratio": ratio, "p_up": p_up, "p_beat": p_beat,
            "method": row.get("method", "straddle"),
            "put_call_volume": row["put_volume"] / row["call_volume"] if row["call_volume"] else np.nan,
        })
    df = pd.DataFrame(rows)
    if df.empty:
        return df
    # Compare each stock with all the others. Across the whole market, expected moves can sit above
    # or below option prices for weeks (volatility regimes, the usual premium option sellers earn),
    # which says nothing about which *stock* is mispriced. So the verdict uses the ratio relative to
    # the median ratio across all stocks; the raw market-wide gap is reported separately.
    df["market_ratio"] = df["ratio"].median()
    df["rel_ratio"] = df["ratio"] / df["market_ratio"]
    df["idea"] = df.apply(suggest, axis=1)
    df["edge"] = (df["rel_ratio"] - 1).abs()
    return df.sort_values("edge", ascending=False).reset_index(drop=True)


def suggest(r: pd.Series) -> str:
    """Short trade structure from the *relative* ratio. Thresholds are wide: small gaps are noise."""
    ratio = r.get("rel_ratio", r["ratio"])
    if pd.isna(ratio):
        return ""
    lean = ""
    if pd.notna(r["p_up"]):
        lean = "up" if r["p_up"] >= 0.58 else "down" if r["p_up"] <= 0.42 else ""
    if ratio >= 1.2:
        return "Cheap → buy straddle" + (f" (lean {lean})" if lean else "")
    if ratio <= 0.8:
        return "Expensive → sell iron condor"
    if lean:
        return "Fair → " + ("call spread" if lean == "up" else "put spread")
    return "Fair → no edge"


# ======================================================================= paper tracking
def track(ideas: pd.DataFrame) -> int:
    """Log ideas as paper trades. Long-vol ideas: buy the straddle. Short-vol ideas: sell it."""
    if ideas.empty:
        return 0
    t = ideas.copy()
    t["tracked_on"] = pd.Timestamp.now().normalize()
    t["position"] = np.where(t["idea"].str.startswith("Expensive"), "short straddle",
                             np.where(t["idea"].str.startswith("Cheap"), "long straddle", "watch"))
    keep = ["tracked_on", "ticker", "event", "expiry", "spot", "strike", "straddle", "implied_move_pct",
            "expected_move_pct", "ratio", "rel_ratio", "p_up", "idea", "position"]
    keep = [k for k in keep if k in t]
    t = t[keep]
    t["expiry"] = pd.to_datetime(t["expiry"])
    old = pd.read_parquet(IDEAS_FILE) if IDEAS_FILE.exists() else pd.DataFrame()
    allx = pd.concat([old, t], ignore_index=True).drop_duplicates(["tracked_on", "ticker", "expiry"], keep="last") \
        if not old.empty else t
    allx.to_parquet(IDEAS_FILE, index=False)
    return len(t)


def grade_ideas() -> pd.DataFrame:
    """At expiry a straddle is worth |price − strike|. P&L uses the expiry-day close."""
    if not IDEAS_FILE.exists():
        return pd.DataFrame()
    ideas = pd.read_parquet(IDEAS_FILE)
    px = data.load_daily(tickers=ideas["ticker"].unique().tolist(),
                         start=str(ideas["tracked_on"].min().date()))
    if px.empty:
        for c in ("expiry_close", "actual_move_pct", "straddle_value", "pl_pct"):
            ideas[c] = np.nan
        ideas["graded"] = False
        return ideas
    px = px.sort_values("date")
    out = []
    for _, r in ideas.iterrows():
        p = px[(px["ticker"] == r["ticker"]) & (px["date"] <= r["expiry"]) & (px["date"] > r["tracked_on"])]
        done = r["expiry"] <= px["date"].max()
        final = p["close"].iloc[-1] if done and not p.empty else np.nan
        value = abs(final - r["strike"]) if pd.notna(final) else np.nan
        long_pl = 100 * (value - r["straddle"]) / r["straddle"] if pd.notna(value) and r["straddle"] else np.nan
        sign = -1 if r["position"] == "short straddle" else 1
        out.append({**r.to_dict(), "graded": pd.notna(final), "expiry_close": final,
                    "actual_move_pct": 100 * abs(final / r["spot"] - 1) if pd.notna(final) else np.nan,
                    "straddle_value": value, "pl_pct": sign * long_pl if r["position"] != "watch" else np.nan})
    return pd.DataFrame(out)
