"""Everyday refresh: new prices → today's predictions → option snapshots + ideas → grade old picks.

    python scripts/daily.py               # after the close (daily, week, month picks + options)
    python scripts/daily.py --hourly      # during market hours: refresh hourly bars + next-hour picks
    python scripts/daily.py --retrain     # also re-train (do this weekly or monthly)
    python scripts/daily.py --no-options  # skip the option-chain snapshot (saves ~20 min)

Schedule it with: bash scripts/install_daily_job.sh
"""
from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402
import requests  # noqa: E402

from stocks import config, data, insiders, macro_calendar, model, options, sectors, tracker  # noqa: E402


def _age_days(path) -> float:
    return (time.time() - path.stat().st_mtime) / 86400 if path.exists() else 1e9


def step(name, fn):
    print(f"\n— {name}")
    try:
        return fn()
    except Exception as e:
        print(f"  ⚠️  {name} failed: {type(e).__name__}: {e}")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--hourly", action="store_true", help="only refresh hourly data + next-hour picks")
    ap.add_argument("--retrain", action="store_true")
    ap.add_argument("--earnings", action="store_true", help="refresh earnings dates too")
    ap.add_argument("--no-options", action="store_true")
    a = ap.parse_args()
    tickers = data.load_universe()["ticker"].tolist()
    if a.hourly:
        data.update_prices("hourly", tickers)
        data.update_yahoo_context()
        model.predict_latest("1h")
        print("Next-hour picks updated.")
        return
    step("Market & economy", lambda: (data.update_yahoo_context(), data.update_fred()))
    step("Daily prices", lambda: data.update_prices("daily", tickers))
    if config.HOURLY_FILE.exists():
        step("Hourly prices", lambda: data.update_prices("hourly", tickers))
    if a.earnings or not config.EARNINGS_FILE.exists() or _age_days(config.EARNINGS_FILE) > 6:
        step("Earnings dates", lambda: data.update_earnings(tickers, max_age_days=7))
    if _age_days(macro_calendar.CAL_FILE) > 6:
        step("Economic calendar", macro_calendar.update_calendar)
    if config.SECTORS_FILE.exists() and _age_days(config.SECTORS_FILE) > 30:
        step("Sectors (new listings)", lambda: sectors.update_sectors(tickers))
    if any(config.INSIDERS_DIR.glob("*.parquet")) and _age_days(insiders.TRADES_FILE) > 30:
        step("Insider trades (new quarter)", insiders.update_insiders)
    if a.retrain:
        import subprocess
        subprocess.run([sys.executable, str(Path(__file__).parent / "train.py")], check=False)
    else:
        def score():
            live = model.load_inputs(start=model._live_start("1d"))
            for h in config.HORIZONS:
                if model.load_bundle(h) is not None:
                    p = model.predict_latest(h, inp=live)
                    print(f"  {h}: {len(p):,} stocks scored")
        step("Predictions", score)
    if not a.no_options:
        step("Option chains", lambda: options.snapshot())
        step("Option ideas", auto_track_ideas)
    g = step("Grading past picks", tracker.grade)
    if g is not None and not g.empty:
        print(tracker.summary(g).to_string(index=False))
    step("Phone alert", alert)


def auto_track_ideas(n: int = 10):
    """Paper-track the day's strongest options ideas so the Options page fills with results."""
    from stocks import events as ev_mod

    preds = {h: model.load_latest(h) for h in ("1w", "1m")}
    s = options.latest_summary()
    if s.empty:
        return
    tick = s["ticker"].unique().tolist()
    px = data.load_daily(tickers=tick, start=(pd.Timestamp.now() - pd.Timedelta(days=365 * 8)).strftime("%Y-%m-%d"))
    rets = np.log(px.pivot_table(index="date", columns="ticker", values="close")).diff()
    ideas = options.build_ideas(preds, rets, data.load_earnings(), macro_calendar.load_calendar(), s)
    if ideas.empty:
        return
    pick = ideas[ideas["idea"].str.contains("Cheap|Expensive")].head(n)
    print(f"  tracking {options.track(pick)} ideas")


def alert():
    """Phone notification with the top picks (install the ntfy app, set NTFY_TOPIC in .env)."""
    if not config.NTFY_TOPIC:
        return
    lines = []
    for h in ("1d", "1w"):
        p = model.load_latest(h)
        if p.empty:
            continue
        p = p[(p["close"] >= config.MIN_PRICE) & (p["dollar_vol"] >= config.MIN_DOLLAR_VOLUME)]
        up = ", ".join(f"{r.ticker} {r.p_up:.0%}" for r in p.head(3).itertuples())
        dn = ", ".join(f"{r.ticker} {1 - r.p_up:.0%}" for r in p.tail(3).iloc[::-1].itertuples())
        lines.append(f"{config.horizon_label(h)} ⬆ {up} | ⬇ {dn}")
    if lines:
        requests.post(f"https://ntfy.sh/{config.NTFY_TOPIC}", data="\n".join(lines).encode(),
                      headers={"Title": "Stock Analyzer picks"}, timeout=10)


if __name__ == "__main__":
    main()
