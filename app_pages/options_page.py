from __future__ import annotations

import numpy as np
import pandas as pd
import plotly.graph_objects as go
import streamlit as st

from stocks import config, data, macro_calendar, options, ui
from . import open_stock


@st.cache_data(ttl=600, show_spinner="Comparing option prices with history and the model ...")
def _ideas(max_days: int) -> pd.DataFrame:
    s = options.latest_summary()
    if s.empty:
        return pd.DataFrame()
    tick = s["ticker"].unique().tolist()
    px = data.load_daily(tickers=tick, start=(pd.Timestamp.now() - pd.Timedelta(days=365 * 8)).strftime("%Y-%m-%d"))
    rets = np.log(px.pivot_table(index="date", columns="ticker", values="close")).diff()
    preds = {h: ui.latest(h) for h in ("1w", "1m")}
    return options.build_ideas(preds, rets, data.load_earnings(), macro_calendar.load_calendar(), s, max_days)


def _ideas_tab():
    s = ui.options_summary()
    if s.empty:
        st.info("No option snapshots yet. Run `python scripts/download_data.py --options` (or let the daily job "
                "run after the close). The first snapshot is enough for ideas; history builds up every day.")
        return
    asof = s["date"].max()
    st.caption(f"Option prices from the snapshot of **{asof:%a %b %d}** · {s.loc[s['date'] == asof, 'ticker'].nunique()} "
               "stocks · compare what options charge (implied move) with what history and the model expect.")
    c = st.columns([1, 1, 1, 1.4])
    max_days = c[0].slider("Expiry within (days)", 7, 60, 45)
    only_events = c[1].toggle("Only stocks with earnings before expiry", value=False)
    min_edge = c[2].slider("Min gap vs fair (%)", 0, 60, 20, 5,
                           help="How far the expected move is from the implied move. 20% = options priced for ±5% "
                                "when ±6% is expected.") / 100
    kind = c[3].multiselect("Show", ["cheap (buy options)", "expensive (sell options)", "fair / directional"],
                            default=["cheap (buy options)", "expensive (sell options)"])
    ideas = _ideas(max_days)
    if ideas.empty:
        st.warning("No ideas: not enough price history for the snapshot tickers yet.")
        return
    d = ideas.copy()
    if only_events:
        d = d[d["event"] != ""]
    d = d[d["edge"] >= min_edge] if min_edge > 0 else d
    keep = pd.Series(False, index=d.index)
    if "cheap (buy options)" in kind:
        keep |= d["idea"].str.startswith("Cheap")
    if "expensive (sell options)" in kind:
        keep |= d["idea"].str.startswith("Expensive")
    if "fair / directional" in kind:
        keep |= d["idea"].str.startswith("Fair")
    d = d[keep]
    gauge = f"{ideas['ratio'].median():.2f}×"
    ui.note("<b>Market-wide:</b> expected move ÷ option-priced move, median across all stocks: " + gauge +
            ". Above 1 means the whole market's options are priced for less movement than history and the model "
            "expect (common after a volatile month); below 1, the reverse (normal: option sellers usually earn a "
            "premium). Because that gap affects every stock alike, ideas below compare each stock with "
            "<b>the others</b>: 'cheap' means cheap relative to the market today.")
    ui.section(f"{len(d)} ideas", "Distance from the dashed line = how far a stock is from the market-wide ratio")
    if d.empty:
        return
    fig = go.Figure()
    for label, mask, color in (("cheap vs market", d["rel_ratio"] >= 1, ui.UP),
                               ("expensive vs market", d["rel_ratio"] < 1, ui.DOWN)):
        x = d[mask]
        fig.add_trace(go.Scatter(x=x["implied_move_pct"], y=x["expected_move_pct"], mode="markers+text", name=label,
                                 text=x["ticker"], textposition="top center", textfont=dict(size=10),
                                 marker=dict(size=10, color=color, line=dict(width=1, color="white")),
                                 customdata=np.c_[x["expiry"].astype(str), x["event"], x["idea"]],
                                 hovertemplate="<b>%{text}</b> exp %{customdata[0]} %{customdata[1]}<br>"
                                               "options price ±%{x:.1f}% · expected ±%{y:.1f}%<br>"
                                               "%{customdata[2]}<extra></extra>"))
    m = float(np.nanmax([d["implied_move_pct"].max(), d["expected_move_pct"].max()])) * 1.05
    slope = float(ideas["ratio"].median())
    fig.add_trace(go.Scatter(x=[0, m], y=[0, m * slope], mode="lines", name=f"market-wide ratio ({slope:.2f}×)",
                             line=dict(color=ui.MUTED, dash="dot", width=1)))
    fig.add_trace(go.Scatter(x=[0, m], y=[0, m], mode="lines", name="1 : 1",
                             line=dict(color=ui.FAINT, dash="dash", width=1)))
    fig.update_xaxes(title="Move the options are priced for (±%)")
    fig.update_yaxes(title="Move expected from history + model (±%)")
    ui.show(ui.style(fig, 440, hover="closest"))
    show = d[["ticker", "idea", "event", "macro_events", "expiry", "days", "spot", "strike", "straddle",
              "implied_move_pct", "hist_move_pct", "model_move_pct", "expected_move_pct", "ratio", "rel_ratio",
              "p_up", "p_beat", "atm_iv", "spread_pct", "method"]]
    sel = st.dataframe(show, hide_index=True, on_select="rerun", selection_mode="multi-row",
                       column_config={
                           "straddle": st.column_config.NumberColumn("Straddle $", format="$%.2f"),
                           "spot": st.column_config.NumberColumn(format="$%.2f"),
                           "implied_move_pct": st.column_config.NumberColumn("Options price ±%", format="%.1f"),
                           "hist_move_pct": st.column_config.NumberColumn("History ±%", format="%.1f",
                                                                          help="Average move over the same number of "
                                                                               "days — around past earnings when "
                                                                               "earnings fall before expiry."),
                           "model_move_pct": st.column_config.NumberColumn("Model ±%", format="%.1f"),
                           "expected_move_pct": st.column_config.NumberColumn("Expected ±%", format="%.1f"),
                           "ratio": st.column_config.NumberColumn("Expected ÷ priced", format="%.2f"),
                           "rel_ratio": st.column_config.NumberColumn("vs market", format="%.2f",
                                                                      help="The ratio divided by today's market-wide "
                                                                           "ratio. Ideas use this: 1.2+ cheap, 0.8− "
                                                                           "expensive."),
                           "method": st.column_config.TextColumn("Priced from", help="straddle = at-the-money "
                                                                 "straddle; iv = implied volatility (strikes near "
                                                                 "the price were too thinly quoted)"),
                           "p_up": st.column_config.NumberColumn("Chance up", format="%.2f"),
                           "p_beat": st.column_config.NumberColumn("Beat S&P", format="%.2f"),
                           "atm_iv": st.column_config.NumberColumn("IV", format="%.2f"),
                           "spread_pct": st.column_config.NumberColumn("Bid-ask %", format="%.2f",
                                                                       help="Wide spreads (>0.10) eat the edge."),
                           "macro_events": st.column_config.TextColumn("Also before expiry"),
                           "idea": st.column_config.TextColumn("Idea", width="medium")})
    rows = sel.selection.rows if sel and sel.selection else []
    c = st.columns([1, 1, 3])
    if c[0].button(f"Paper-track {len(rows) or 'all shown'}", type="primary", icon=":material/bookmark_add:"):
        n = options.track(d.iloc[rows] if rows else d)
        st.success(f"Tracking {n} ideas. They're graded at expiry on the Results tab.")
    if rows and c[1].button("Open stock"):
        open_stock(d.iloc[rows[0]]["ticker"])
    st.caption("**Expensive** often means the options market knows something history doesn't (pending news, a "
               "deal, a lawsuit, an unusual earnings date). Check the news before selling premium. ")
    st.caption("How to read it: a **long straddle** (buy the at-the-money call and put) profits if the stock moves "
               "more than the straddle cost either way by expiry. **Selling** premium profits if it moves less — "
               "use defined-risk structures (iron condor/butterfly), since a naked short straddle has unlimited risk. "
               "'History' is what the stock actually moved over the same window in the past; 'Model' scales the "
               "model's expected move to the days until expiry. Wide bid-ask spreads can wipe out the gap.")


def _results_tab():
    g = options.grade_ideas()
    if g.empty:
        st.info("No tracked ideas yet. Track some on the Ideas tab — or let the daily job track the 10 strongest "
                "each day.")
        return
    done = g[g["graded"] & g["pl_pct"].notna()]
    c = st.columns(4)
    c[0].metric("Tracked ideas", len(g))
    c[1].metric("Expired & graded", len(done))
    if not done.empty:
        c[2].metric("Win rate", f"{(done['pl_pct'] > 0).mean():.0%}")
        c[3].metric("Avg P&L per trade", ui.pct(done["pl_pct"].mean()))
        by = done.groupby("position")["pl_pct"].agg(["count", "mean", lambda x: (x > 0).mean()])
        by.columns = ["trades", "avg P&L %", "win rate"]
        st.dataframe(by, column_config={"avg P&L %": st.column_config.NumberColumn(format="%+.1f"),
                                        "win rate": st.column_config.NumberColumn(format="%.2f")})
        eq = done.sort_values("expiry").set_index("expiry")["pl_pct"]
        fig = go.Figure(go.Scatter(x=eq.index, y=eq.cumsum(), mode="lines+markers",
                                   line=dict(color=ui.SERIES[0], width=2)))
        fig.update_yaxes(title="Cumulative P&L (% of premium, summed)")
        ui.show(ui.style(fig, 320, "Paper results by expiry date"))
    st.dataframe(g.sort_values("tracked_on", ascending=False)[
        ["tracked_on", "ticker", "event", "expiry", "position", "spot", "strike", "straddle", "implied_move_pct",
         "expected_move_pct", "expiry_close", "actual_move_pct", "straddle_value", "pl_pct"]],
        hide_index=True, column_config={c: st.column_config.NumberColumn(format="%.2f") for c in
                                        ["spot", "strike", "straddle", "implied_move_pct", "expected_move_pct",
                                         "expiry_close", "actual_move_pct", "straddle_value", "pl_pct"]})
    st.caption("Graded with the stock's closing price on expiry day: a straddle is then worth |price − strike|. "
               "Real fills, early exits and commissions will differ.")


def _iv_tab():
    s = ui.options_summary()
    if s.empty:
        st.caption("No snapshots yet.")
        return
    days = s["date"].nunique()
    st.caption(f"{days} day{'s' if days != 1 else ''} of snapshots saved. Once there are a few months, this history "
               "lets you study how implied volatility behaves before and after events.")
    t = st.selectbox("Ticker", sorted(s["ticker"].unique()), format_func=ui.ticker_label)
    x = s[s["ticker"] == t]
    piv = x[(x["days"] >= 20) & (x["days"] <= 40)].groupby("date")[["atm_iv", "implied_move_pct", "skew"]].mean()
    st.dataframe(x.sort_values(["date", "days"], ascending=[False, True]).head(40), hide_index=True)
    if len(piv) >= 2:
        fig = go.Figure(go.Scatter(x=piv.index, y=100 * piv["atm_iv"], mode="lines+markers",
                                   line=dict(color=ui.SERIES[0], width=2), name="1-month ATM IV"))
        fig.update_yaxes(title="Implied volatility (%)")
        ui.show(ui.style(fig, 300))


def render():
    ui.page_header("Options ideas", "Where option prices disagree with how much the stock has really moved around "
                   "similar events, and with what the model expects. Research only: options can lose 100% of "
                   "what you pay.", kicker="Predict")
    t1, t2, t3 = st.tabs(["Ideas", "Results (paper trading)", "Snapshot history"])
    with t1:
        _ideas_tab()
    with t2:
        _results_tab()
    with t3:
        _iv_tab()
