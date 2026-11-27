from __future__ import annotations

import numpy as np
import pandas as pd
import plotly.graph_objects as go
import streamlit as st

from stocks import config, data, links, ui


@st.cache_data(ttl=3600, show_spinner="Building weekly series ...")
def _drivers() -> pd.DataFrame:
    return links.driver_table(data.load_yahoo_context(), data.load_fred_raw(), data.load_custom_series())


@st.cache_data(ttl=3600, show_spinner="Building weekly stock returns ...")
def _weekly() -> pd.DataFrame:
    return links.weekly_returns(data.load_daily())


@st.cache_data(ttl=3600, show_spinner="Testing every pair (this can take a minute) ...")
def _scan(driver_names: tuple, tickers: tuple | None, lags: tuple) -> pd.DataFrame:
    rets = _weekly()
    if tickers:
        rets = rets[[t for t in tickers if t in rets.columns]]
    return links.scan(_drivers(), rets, lags=lags, driver_names=list(driver_names))


def _pair_chart(driver: str, ticker: str, lag: int):
    d = _drivers()[driver]
    r = _weekly()[ticker].shift(-lag)
    df = pd.concat([d, r], axis=1, keys=["x", "y"]).dropna()
    split = df.index.max() - pd.DateOffset(days=int(365.25 * config.TEST_YEARS))
    c = st.columns(2)
    fig = go.Figure()
    for name, part, color in (("before the test window", df[df.index < split], ui.SERIES[0]),
                              ("last 2 years", df[df.index >= split], ui.SERIES[1])):
        fig.add_trace(go.Scatter(x=part["x"], y=100 * part["y"], mode="markers", name=name,
                                 marker=dict(size=7, color=color, opacity=0.55, line=dict(width=1, color="white")),
                                 hovertemplate="%{x:.4f} → %{y:+.2f}%<extra></extra>"))
        if len(part) > 10:
            k, b = np.polyfit(part["x"], 100 * part["y"], 1)
            xs = np.linspace(part["x"].quantile(.02), part["x"].quantile(.98), 20)
            fig.add_trace(go.Scatter(x=xs, y=k * xs + b, mode="lines", line=dict(color=color, width=2),
                                     showlegend=False, hoverinfo="skip"))
    fig.update_xaxes(title=f"{driver}: weekly change")
    fig.update_yaxes(title=f"{ticker} return {lag} week(s) later (%)")
    with c[0]:
        ui.show(ui.style(fig, 380, "Each dot is one week", hover="closest"))
    roll = df["x"].rolling(104, min_periods=60).corr(df["y"])
    fig2 = go.Figure(go.Scatter(x=roll.index, y=roll, mode="lines", line=dict(color=ui.SERIES[0], width=2),
                                name="2-year rolling correlation"))
    fig2.add_hline(y=0, line=dict(color=ui.MUTED, width=1))
    fig2.add_vline(x=split, line=dict(color=ui.SERIES[1], dash="dot"))
    fig2.update_yaxes(title="correlation")
    with c[1]:
        ui.show(ui.style(fig2, 380, "Is the link stable over time?"))


def render():
    ui.page_header("Hidden links",
                   "Series that move before a stock, whether related (oil and airlines) or seemingly unrelated "
                   "(lumber, bitcoin, jobless claims, your own data). Each link is found on data before the last "
                   "2 years, then checked on the last 2 years. <b>Held up</b> means it was still there, same "
                   "direction, in the period it was never fitted on.", kicker="Research")
    ui.needs_data()
    D = _drivers()
    if D.empty:
        st.warning("No market/economy series yet. Run `python scripts/download_data.py`.")
        return
    mode = st.radio("Question", ["What moves before this stock?", "What does this series lead?",
                                 "Scan everything for the strongest links"], horizontal=True)
    c = st.columns([2, 2, 1])
    lags = tuple(c[1].multiselect("Lead time (weeks)", [1, 2, 3, 4], default=[1, 2]))
    held = c[2].toggle("Held up only", value=True)
    if not lags:
        return
    W = _weekly()
    if mode.startswith("What moves before"):
        t = st.session_state.get("ticker")
        tickers = list(W.columns)
        ticker = c[0].selectbox("Stock", tickers, index=tickers.index(t) if t in tickers else 0,
                                format_func=ui.ticker_label)
        res = _scan(tuple(D.columns), (ticker,), lags)
        ui.section(f"What {ticker} moves with", "Same-week correlation with every series")
        same = links.contemporaneous(D, W, ticker)
        if not same.empty:
            top = same if len(same) <= 12 else pd.concat([same.head(6), same.tail(6)])
            fig = go.Figure(go.Bar(x=top.values, y=top.index, orientation="h",
                                   marker_color=[ui.UP if v >= 0 else ui.DOWN for v in top.values]))
            fig.update_xaxes(title="correlation (same week)")
            ui.show(ui.style(fig, 360, hover="closest"))
    elif mode.startswith("What does"):
        driver = c[0].selectbox("Series", list(D.columns))
        res = _scan((driver,), None, lags)
    else:
        picks = c[0].multiselect("Series to scan (blank = all)", list(D.columns))
        res = _scan(tuple(picks or D.columns), None, lags)
    if res.empty:
        st.info("Not enough overlapping history.")
        return
    show = res[res["held_up"]] if held else res
    show = show.head(300)
    ui.section(f"{len(show):,} link{'' if len(show) == 1 else 's'}" + (" that held up" if held else ""),
               "Select a row to chart it")
    st.caption("r = correlation (−1…1; 0.1 is meaningful for weekly stock returns). q = chance the discovery is "
               "a fluke after correcting for testing many pairs at once.")
    sel = st.dataframe(show[["driver", "ticker", "lag_weeks", "r_discovery", "q_discovery", "r_confirm",
                             "p_confirm", "held_up", "weeks_discovery"]],
                       hide_index=True, on_select="rerun", selection_mode="single-row",
                       column_config={"r_discovery": st.column_config.NumberColumn("r (discovery)", format="%+.3f"),
                                      "r_confirm": st.column_config.NumberColumn("r (last 2 yrs)", format="%+.3f"),
                                      "q_discovery": st.column_config.NumberColumn("q", format="%.3f"),
                                      "p_confirm": st.column_config.NumberColumn("p (last 2 yrs)", format="%.3f")})
    rows = sel.selection.rows if sel and sel.selection else []
    r = show.iloc[rows[0]] if rows else (show.iloc[0] if not show.empty else None)
    if r is not None:
        ui.section(f"{r['driver']} → {r['ticker']}, {int(r['lag_weeks'])} week(s) later")
        _pair_chart(r["driver"], r["ticker"], int(r["lag_weeks"]))
        st.caption("Click a row in the table to chart it. Add your own series by dropping a CSV with date,value "
                   "columns into data/custom_series/ (e.g. a Google Trends export).")
