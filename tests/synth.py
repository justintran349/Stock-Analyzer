"""Fake market data with planted patterns, so the whole pipeline can be tested offline.

Planted signals the models should find:
  - short-term reversal: stocks that fell over the last 5 days tend to bounce the next day
  - 'oil' stocks follow oil's move from the previous week
"""
from __future__ import annotations

import numpy as np
import pandas as pd


def make(data_dir, n_tickers: int = 60, start: str = "2004-01-01", end: str = "2026-11-20",
         hourly_days: int = 500, seed: int = 0):
    from stocks import config

    rng = np.random.default_rng(seed)
    dates = pd.bdate_range(start, end)
    T = len(dates)
    mkt = rng.normal(0.0003, 0.011, T)
    oil = rng.normal(0, 0.02, T)
    oil5 = pd.Series(oil).rolling(5).sum().shift(1).fillna(0).values
    rows = []
    tickers = [f"T{i:03d}" for i in range(n_tickers)]
    for i, t in enumerate(tickers):
        beta = rng.uniform(0.6, 1.4)
        oil_beta = 0.25 if i % 5 == 0 else 0.0
        eps = rng.normal(0, 0.018, T)
        r = np.zeros(T)
        for k in range(T):
            past5 = r[max(0, k - 5):k].sum()
            r[k] = beta * mkt[k] + oil_beta * oil5[k] - 0.08 * past5 + eps[k]
        close = 30 * np.exp(np.cumsum(r))
        first = rng.integers(0, T // 3) if i % 7 == 0 else 0
        df = pd.DataFrame({"date": dates[first:], "ticker": t,
                           "open": (close * np.exp(rng.normal(0, 0.004, T)))[first:],
                           "high": (close * np.exp(np.abs(rng.normal(0, 0.01, T))))[first:],
                           "low": (close * np.exp(-np.abs(rng.normal(0, 0.01, T))))[first:],
                           "close": close[first:],
                           "volume": rng.lognormal(14, 0.5, T)[first:]})
        rows.append(df)
    prices = pd.concat(rows, ignore_index=True)
    for c in ("open", "high", "low", "close"):
        prices[c] = prices[c].astype("float32")
    prices.to_parquet(config.DAILY_FILE, index=False)
    pd.DataFrame({"ticker": tickers, "name": [f"Fake Co {i}" for i in range(n_tickers)],
                  "exchange": "N", "etf": "N"}).to_csv(config.UNIVERSE_FILE, index=False)

    # context series
    spy = 100 * np.exp(np.cumsum(mkt))
    ctx = {"SPY": spy, "^VIX": 15 + 5 * np.abs(pd.Series(mkt).rolling(10).std().bfill().values / 0.011),
           "CL=F": 50 * np.exp(np.cumsum(oil)), "GC=F": 1000 * np.exp(np.cumsum(rng.normal(0, 0.01, T))),
           "^TNX": 3 + np.cumsum(rng.normal(0, 0.03, T)), "BTC-USD": 100 * np.exp(np.cumsum(rng.normal(0, 0.04, T)))}
    inv = {v: k for k, v in config.YAHOO_CONTEXT.items()}
    cr = []
    for sym, vals in ctx.items():
        cr.append(pd.DataFrame({"date": dates, "ticker": sym, "open": vals, "high": vals, "low": vals,
                                "close": vals, "volume": 0.0, "name": inv[sym]}))
    pd.concat(cr).to_parquet(config.YAHOO_CONTEXT_FILE, index=False)

    # FRED-like monthly + daily
    months = pd.date_range("2000-01-01", end, freq="MS")
    fred = [pd.DataFrame({"name": "cpi", "series_id": "CPIAUCSL", "obs_date": months,
                          "value": 170 * np.exp(np.cumsum(rng.normal(0.002, 0.002, len(months)))),
                          "lag_days": 45, "kind": "yoy"}),
            pd.DataFrame({"name": "unemployment", "series_id": "UNRATE", "obs_date": months,
                          "value": 5 + np.cumsum(rng.normal(0, 0.1, len(months))), "lag_days": 37, "kind": "chg"}),
            pd.DataFrame({"name": "hy_spread", "series_id": "BAMLH0A0HYM2", "obs_date": dates,
                          "value": 4 + np.cumsum(rng.normal(0, 0.02, T)), "lag_days": 1, "kind": "chg"})]
    pd.concat(fred).to_parquet(config.FRED_FILE, index=False)

    # earnings every ~63 trading days
    er = []
    for t in tickers:
        off = rng.integers(0, 63)
        for d in dates[off::63]:
            er.append({"ticker": t, "date": d, "eps_estimate": 1.0, "eps_actual": 1.0 + rng.normal(0, .1),
                       "surprise_pct": rng.normal(0, 8), "fetched": pd.Timestamp.now()})
    pd.DataFrame(er).to_parquet(config.EARNINGS_FILE, index=False)

    # hourly bars for the last `hourly_days` business days
    hd = dates[-hourly_days:]
    hrs = [pd.Timedelta(hours=9, minutes=30) + pd.Timedelta(hours=k) for k in range(7)]
    stamps = pd.DatetimeIndex([d + h for d in hd for h in hrs])
    H = len(stamps)
    mh = rng.normal(0, 0.003, H)
    hrows = []
    for i, t in enumerate(tickers[:30]):
        e = rng.normal(0, 0.006, H)
        r = np.zeros(H)
        for k in range(H):
            r[k] = mh[k] - 0.1 * r[max(0, k - 1)] + e[k]
        c = 30 * np.exp(np.cumsum(r))
        hrows.append(pd.DataFrame({"date": stamps, "ticker": t, "open": c, "high": c * 1.002, "low": c * 0.998,
                                   "close": c, "volume": rng.lognormal(12, 0.4, H)}))
    pd.concat(hrows).to_parquet(config.HOURLY_FILE, index=False)
    spyh = 100 * np.exp(np.cumsum(mh))
    pd.concat([pd.DataFrame({"date": stamps, "ticker": "SPY", "close": spyh}),
               pd.DataFrame({"date": stamps, "ticker": "^VIX", "close": 15 + rng.normal(0, 1, H)})]
              ).to_parquet(config.CONTEXT_DIR / "hourly_market.parquet", index=False)
    # sectors: oil stocks (every 5th) in Energy, the rest split into two industries
    pd.DataFrame({"ticker": tickers,
                  "sector": ["Energy" if i % 5 == 0 else "Tech" for i in range(n_tickers)],
                  "industry": ["Oil" if i % 5 == 0 else ("Software" if i % 2 else "Hardware") for i in range(n_tickers)],
                  "quote_type": "EQUITY", "market_cap": 1e9, "fetched": pd.Timestamp.now()}
                 ).to_csv(config.SECTORS_FILE, index=False)
    # insider trades: random buys/sells
    from stocks import insiders, macro_calendar
    ins = []
    for _ in range(400):
        t = tickers[rng.integers(0, n_tickers)]
        d = dates[rng.integers(300, T - 5)]
        side = "buy" if rng.random() < 0.4 else "sell"
        ins.append({"ticker": t, "filing_date": d, "trans_date": d - pd.Timedelta(days=2), "side": side,
                    "shares": 1000.0, "price": 30.0, "value": float(rng.lognormal(12, 1)),
                    "owner": f"o{rng.integers(0, 50)}", "owner_name": "x", "is_officer": True,
                    "is_director": False, "is_10pct": False, "is_exec": bool(rng.random() < 0.3)})
    pd.DataFrame(ins).to_parquet(config.INSIDERS_DIR / "2020q1.parquet", index=False)
    insiders.combine()
    # economic calendar
    rows = [(d, "fomc", "Fed rate decision (FOMC)") for d in macro_calendar.FOMC_BACKUP]
    rows += [(d.strftime("%Y-%m-%d"), "jobs_report", "Jobs report")
             for d in macro_calendar.jobs_report_rule(start, "2027-03-01")]
    pd.DataFrame(rows, columns=["date", "type", "title"]).to_csv(macro_calendar.CAL_FILE, index=False)
    # option snapshot for today
    from stocks import options
    today = pd.Timestamp.now().normalize()
    srows = []
    for t in tickers[:15]:
        spot = float(prices.loc[prices["ticker"] == t, "close"].iloc[-1])
        for days in (3, 10, 24, 45):
            mv = rng.uniform(2, 9)
            srows.append({"date": today, "ticker": t, "expiry": today + pd.Timedelta(days=days), "spot": spot,
                          "days": days, "strike": round(spot), "straddle": spot * mv / 100,
                          "implied_move_pct": mv, "atm_iv": 0.3, "skew": 0.02, "call_volume": 100.0,
                          "put_volume": 80.0, "call_oi": 1000.0, "put_oi": 900.0, "spread_pct": 0.05})
    pd.DataFrame(srows).to_parquet(options.SUMMARY_FILE, index=False)
    # make some earnings land in the future so ideas have events
    fut = pd.DataFrame({"ticker": tickers[:10], "date": [today + pd.Timedelta(days=int(k)) for k in rng.integers(2, 20, 10)],
                        "eps_estimate": 1.0, "eps_actual": np.nan, "surprise_pct": np.nan, "fetched": pd.Timestamp.now()})
    pd.concat([pd.read_parquet(config.EARNINGS_FILE), fut]).to_parquet(config.EARNINGS_FILE, index=False)
    return tickers
