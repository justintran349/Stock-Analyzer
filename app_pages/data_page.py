from __future__ import annotations

import subprocess
import sys

import pandas as pd
import streamlit as st

from stocks import config, data, ui

ROOT = config.ROOT


def _file_row(name, path, date_col="date"):
    if not path.exists():
        return {"data": name, "status": "missing", "rows": 0, "tickers/series": None, "latest": ""}
    try:
        df = pd.read_parquet(path) if path.suffix == ".parquet" else pd.read_csv(path)
        key = "ticker" if "ticker" in df else "name" if "name" in df else None
        latest = pd.to_datetime(df[date_col]).max() if date_col in df else ""
        return {"data": name, "status": "ok", "rows": len(df), "tickers/series": df[key].nunique() if key else None,
                "latest": str(latest)[:16]}
    except Exception as e:
        return {"data": name, "status": f"error: {e}", "rows": 0, "tickers/series": 0, "latest": ""}


def _run(args: list[str]):
    box = st.empty()
    lines = []
    proc = subprocess.Popen([sys.executable] + args, cwd=ROOT, stdout=subprocess.PIPE, stderr=subprocess.STDOUT,
                            text=True, bufsize=1)
    for line in proc.stdout:
        lines.append(line.rstrip())
        box.code("\n".join(lines[-25:]))
    proc.wait()
    ui.clear()
    (st.success if proc.returncode == 0 else st.error)("Finished" if proc.returncode == 0 else "Failed: see log")


def render():
    ui.page_header("Data & models", "What's downloaded, model status, and where to add more data.",
                   kicker="Validate")
    st.dataframe(pd.DataFrame([
        _file_row("Daily prices", config.DAILY_FILE),
        _file_row("Hourly prices", config.HOURLY_FILE),
        _file_row("Market & cross-asset (Yahoo)", config.YAHOO_CONTEXT_FILE),
        _file_row("Economy (FRED)", config.FRED_FILE, "obs_date"),
        _file_row("Earnings dates", config.EARNINGS_FILE),
        _file_row("Events (your list)", config.EVENTS_FILE),
        _file_row("Universe", config.UNIVERSE_FILE, "none"),
        _file_row("Economic calendar (Fed, CPI, jobs...)", config.DATA_DIR / "macro_calendar.csv"),
        _file_row("Sectors & industries", config.SECTORS_FILE, "fetched"),
        _file_row("Insider trades (SEC)", config.DATA_DIR / "insider_trades.parquet", "filing_date"),
        _file_row("Option snapshots (summary)", config.OPTIONS_DIR / "summary.parquet"),
    ]), hide_index=True)
    import os
    keys = []
    if not os.getenv("FRED_API_KEY"):
        keys.append("**FRED_API_KEY** is not set: CPI, GDP, PCE, PPI and retail-sales release dates are skipped and "
                    "jobs-report dates are estimated. Free key: https://fred.stlouisfed.org/docs/api/api_key.html")
    if not config.SEC_USER_AGENT:
        keys.append("**SEC_USER_AGENT** is not set: the SEC may refuse insider-trade downloads. Put "
                    "`SEC_USER_AGENT=Your Name your@email.com` in .env.")
    for k in keys:
        st.warning(k)
    rows = []
    for h in config.HORIZONS:
        m = ui.metrics(h)
        rows.append({"horizon": config.horizon_label(h), "trained": m["trained_at"] if m else "not trained",
                     "test window": f"{m['dates']['test_start']} → {m['dates']['test_end']}" if m else "",
                     "test accuracy": f"{m['test']['accuracy']:.1%}" if m else "",
                     "same-day AUC": f"{m['test']['same_day_auc']:.3f}" if m else "",
                     "features": m["tuning"]["best"]["feature_set"] if m else ""})
    st.dataframe(pd.DataFrame(rows), hide_index=True)

    ui.section("Run", "Long jobs are better in the terminal, but these work too")
    st.caption("Long jobs are better in the VS Code terminal (you can keep using the app), but these work too.")
    c = st.columns(4)
    if c[0].button("Update prices + re-score", width="stretch"):
        _run(["scripts/daily.py"])
    if c[1].button("Next-hour refresh", width="stretch"):
        _run(["scripts/daily.py", "--hourly"])
    if c[2].button("Re-train (quick)", width="stretch"):
        _run(["scripts/train.py", "--quick"])
    if c[3].button("Reload app data", width="stretch"):
        ui.clear()
        st.rerun()

    ui.section("Add more data")
    t1, t2, t3 = st.tabs(["Economy series (FRED)", "Extra tickers", "Your own series"])
    with t1:
        st.caption("Any series from fred.stlouisfed.org — the ID is in its URL. **lag_days** = how long after the "
                   "observation date the number is published (monthly data: ~30–45). kind: level, chg or yoy.")
        p = config.DATA_DIR / "fred_extra.csv"
        df = pd.read_csv(p) if p.exists() else pd.DataFrame(
            [{"name": "oil_inventories", "series_id": "WCESTUS1", "lag_days": 7, "kind": "chg"}])
        ed = st.data_editor(df, num_rows="dynamic", hide_index=True, key="fred_ed")
        if st.button("Save FRED list"):
            ed.to_csv(p, index=False)
            st.success("Saved. Run 'Update prices + re-score' (or download_data.py) to fetch, then re-train.")
        st.caption("Built in: " + ", ".join(config.FRED_SERIES))
    with t2:
        st.caption("Add tickers outside the NYSE (e.g. AAPL, NVDA, MSFT, TSLA on NASDAQ). "
                   "Or set STOCK_EXCHANGES=N,Q in .env to pull all of NASDAQ too.")
        p = config.DATA_DIR / "universe_extra.csv"
        df = pd.read_csv(p) if p.exists() else pd.DataFrame({"ticker": ["AAPL", "NVDA", "MSFT", "TSLA"],
                                                             "name": ["Apple", "Nvidia", "Microsoft", "Tesla"]})
        ed = st.data_editor(df, num_rows="dynamic", hide_index=True, key="uni_ed")
        if st.button("Save extra tickers"):
            ed.to_csv(p, index=False)
            st.success("Saved. Run `python scripts/download_data.py` to download them.")
    with t3:
        st.markdown("Drop CSV files into `data/custom_series/` with a **date** column and a **value** column "
                    "(e.g. a Google Trends export, web traffic, weather, box office, anything). Each file becomes a "
                    "series on the Hidden links page and a model input after re-training. Optional `lag_days` "
                    "column: days after the date the value became public (default 1).")
        files = sorted(p.name for p in config.CUSTOM_DIR.glob("*.csv"))
        st.write(files or "No custom series yet.")
        up = st.file_uploader("…or upload one here", type="csv")
        if up is not None:
            (config.CUSTOM_DIR / up.name).write_bytes(up.getvalue())
            st.success(f"Saved data/custom_series/{up.name}")
