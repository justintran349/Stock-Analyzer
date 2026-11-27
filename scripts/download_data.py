"""Download everything the models need. Safe to re-run: it only fetches what's new.

    python scripts/download_data.py              # universe + daily prices + market + economy + calendar
    python scripts/download_data.py --hourly     # also hourly bars (needed for 'next hour')
    python scripts/download_data.py --earnings   # also earnings dates (slow the first time)
    python scripts/download_data.py --sectors    # sector/industry for every stock (slow the first time)
    python scripts/download_data.py --insiders   # SEC insider trades since 2006 (~1 GB, once)
    python scripts/download_data.py --options    # today's option chains for the most-traded stocks
    python scripts/download_data.py --all        # everything
    python scripts/download_data.py --limit 50   # quick try with 50 tickers

First full run takes 1.5–2.5 hours (mostly one-time); daily updates take minutes.
"""
from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from stocks import config, data, events, insiders, macro_calendar, options, sectors  # noqa: E402


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--hourly", action="store_true", help="also download hourly bars (last 730 days)")
    ap.add_argument("--earnings", action="store_true", help="also download earnings dates")
    ap.add_argument("--sectors", action="store_true", help="also look up sector/industry per stock")
    ap.add_argument("--insiders", action="store_true", help="also download SEC insider trades")
    ap.add_argument("--options", action="store_true", help="also snapshot today's option chains")
    ap.add_argument("--all", action="store_true")
    ap.add_argument("--limit", type=int, default=0, help="only the first N tickers (testing)")
    ap.add_argument("--full", action="store_true", help="re-download price history from scratch")
    ap.add_argument("--skip-universe", action="store_true")
    a = ap.parse_args()
    t0 = time.time()
    if a.skip_universe and config.UNIVERSE_FILE.exists():
        uni = data.load_universe()
    else:
        uni = data.update_universe()
    tickers = uni["ticker"].tolist()
    if a.limit:
        tickers = tickers[: a.limit]
    events.load_events(include_auto=False)   # creates data/events.csv with the starter list if missing
    steps = [("Market & cross-asset series (Yahoo)", lambda: data.update_yahoo_context()),
             ("Economy (FRED)", lambda: data.update_fred()),
             ("Economic calendar (Fed, CPI, jobs, GDP, ...)", lambda: macro_calendar.update_calendar()),
             (f"Daily prices since {config.HISTORY_START}", lambda: data.update_prices("daily", tickers, full=a.full))]
    if a.hourly or a.all:
        steps.append(("Hourly prices (last 730 days)", lambda: data.update_prices("hourly", tickers, full=a.full)))
    if a.earnings or a.all:
        steps.append(("Earnings dates", lambda: data.update_earnings(tickers)))
    if a.sectors or a.all:
        steps.append(("Sectors & industries", lambda: sectors.update_sectors(tickers)))
    if a.insiders or a.all:
        steps.append(("Insider trades (SEC)", lambda: insiders.update_insiders()))
    if a.options or a.all:
        steps.append(("Option chains (today)", lambda: options.snapshot(
            options.pick_tickers()[: (a.limit or config.OPTIONS_TOP_N) + 3])))
    for i, (name, fn) in enumerate(steps, 1):
        print(f"\n[{i}/{len(steps)}] {name}")
        try:
            fn()
        except Exception as e:   # one failing source shouldn't stop the rest
            print(f"  ⚠️  {name} failed: {type(e).__name__}: {e}\n  Everything else continues; re-run later.")
    print(f"\nDone in {(time.time() - t0) / 60:.1f} min. Next: python scripts/train.py")


if __name__ == "__main__":
    main()
