"""Train the models and run the 2-year backtest.

    python scripts/train.py                    # all horizons (hour, day, week, month)
    python scripts/train.py --horizons 1d 1w   # just some
    python scripts/train.py --quick            # one setting instead of the tuning grid
    python scripts/train.py --walk-forward     # re-train every quarter through the test window

What happens: train on data up to 3 years ago → tune on the year after → grade on the last
2 years (never seen) → refit on everything for live predictions. Results land in
models/<horizon>_metrics.json and show on the app's Backtest page.
Takes roughly 20–60 minutes for all horizons on a laptop with the full NYSE.
"""
from __future__ import annotations

import argparse
import gc
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from stocks import config, features as F, model  # noqa: E402


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--horizons", nargs="*", default=list(config.HORIZONS))
    ap.add_argument("--quick", action="store_true")
    ap.add_argument("--walk-forward", action="store_true")
    a = ap.parse_args()
    t0 = time.time()
    print("Loading data ...")
    inp = model.load_inputs()
    if inp.prices.empty:
        sys.exit("No price data. Run: python scripts/download_data.py")
    print(f"  {inp.prices['ticker'].nunique():,} tickers, {len(inp.prices):,} daily rows, "
          f"{inp.prices['date'].min():%Y-%m-%d} → {inp.prices['date'].max():%Y-%m-%d}")
    print(f"  events: {len(inp.events)} (model uses market types {inp.mkt_types}, "
          f"company types {inp.tk_types} + earnings)")
    results = {}
    # hourly first: it needs the raw daily prices, which are freed afterwards to save memory
    for h in [h for h in a.horizons if config.HORIZONS[h][2] == "hourly"]:
        results[h] = model.train_horizon(h, inp=inp, quick=a.quick, walk_forward=a.walk_forward)
    daily_h = [h for h in a.horizons if config.HORIZONS[h][2] == "daily"]
    if daily_h:
        print("Building daily features ...")
        stock = model.compute_stock(inp, log=print)
        inp.prices = inp.prices.iloc[0:0]
        gc.collect()
        print(f"  {len(stock):,} rows × {stock.shape[1]} columns "
              f"({stock.memory_usage(deep=False).sum() / 1e9:.1f} GB)")
        for h in daily_h:
            results[h] = model.train_horizon(h, inp=inp, stock=stock, quick=a.quick,
                                             walk_forward=a.walk_forward)
            gc.collect()
        del stock
    print("\nScoring today's predictions ...")
    live = model.load_inputs(start=model._live_start("1d"))
    for h in a.horizons:
        if results.get(h):
            p = model.predict_latest(h, inp=live)
            print(f"  {h}: {len(p):,} stocks scored")
    print(f"\nAll done in {(time.time() - t0) / 60:.1f} min. Open the app: streamlit run app.py")


if __name__ == "__main__":
    main()
