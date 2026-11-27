from __future__ import annotations

import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import streamlit as st

from stocks import config, ui

NICE = {"p_up": "Chance up", "p_beat": "Chance to beat S&P", "y_ex": "Return vs S&P (log)", "exp_ret": "Expected return %", "exp_move": "Expected move ±%",
        "conf": "Confidence", "tested_hit_rate": "Tested hit rate", "close": "Price",
        "dollar_vol": "$ volume/day", "mom_ret_21": "Last month return (log)", "risk_vol_21": "Volatility",
        "trend_rsi_14": "RSI", "evt_tk_until_earnings": "Days to earnings", "actual_%": "Actual return %",
        "y_ret": "Actual return (log)"}


def _source(h: str, src: str) -> pd.DataFrame:
    if src == "Latest predictions":
        d = ui.latest(h).copy()
    elif src == "Backtest (last 2 years)":
        d = ui.backtest(h).copy()
        if not d.empty:
            d["actual_%"] = 100 * (np.exp(d["y_ret"]) - 1)
            d["right"] = ((d["p_up"] >= .5) == (d["y_ret"] > 0)).astype(int)
    else:
        p = config.PRED_HISTORY_FILE
        d = pd.read_parquet(p) if p.exists() else pd.DataFrame()
        d = d[d["horizon"] == h] if not d.empty else d
    return d


def _presets(h: str):
    bt = ui.backtest(h)
    if bt.empty:
        st.caption("Run training to unlock preset charts.")
        return
    bt = bt.copy()
    bt["actual_%"] = 100 * (np.exp(bt["y_ret"]) - 1)
    c = st.columns(2)
    # 1. deciles
    bt["decile"] = bt.groupby("date")["p_up"].transform(lambda s: pd.qcut(s.rank(method="first"), 10, labels=False)
                                                         if len(s) >= 10 else np.nan) + 1
    dec = bt.dropna(subset=["decile"]).groupby("decile")["actual_%"].mean()
    fig = go.Figure(go.Bar(x=dec.index.astype(int), y=dec.values,
                           marker_color=[ui.UP if v >= 0 else ui.DOWN for v in dec.values],
                           hovertemplate="decile %{x}: %{y:+.2f}%<extra></extra>"))
    fig.update_xaxes(title="Model's ranking that day (1 = least likely up, 10 = most likely up)", dtick=1)
    fig.update_yaxes(title=f"Average actual return, {config.horizon_label(h).lower()} (%)")
    with c[0]:
        ui.show(ui.style(fig, 360, "Do higher-ranked stocks really do better?", hover="closest"))
        st.caption("A staircase rising left to right = the ranking works.")
    # 2. market mood
    mood = bt.groupby("date").agg(share_up=("p_up", lambda s: (s >= .5).mean()),
                                  actual=("actual_%", lambda s: (s > 0).mean()))
    fig = go.Figure()
    win = 20 if h in ("1d", "1h") else 4
    fig.add_trace(go.Scatter(x=mood.index, y=100 * mood["share_up"].rolling(win, min_periods=1).mean(),
                             mode="lines", name="predicted up",
                             line=dict(color=ui.SERIES[0], width=2)))
    fig.add_trace(go.Scatter(x=mood.index, y=100 * mood["actual"].rolling(win, min_periods=1).mean(), mode="lines",
                             name="actually went up", line=dict(color=ui.SERIES[1], width=2)))
    fig.update_yaxes(title="% of stocks")
    with c[1]:
        ui.show(ui.style(fig, 360, "Market mood: share of stocks predicted up (rolling average)"))
    # 3. expected move vs actual move
    bt["abs_actual"] = bt["actual_%"].abs()
    bt["move_bin"] = pd.qcut(bt["exp_move"], 10, duplicates="drop")
    mv = bt.groupby("move_bin", observed=True).agg(pred=("exp_move", "mean"), actual=("abs_actual", "mean"))
    fig = go.Figure()
    fig.add_trace(go.Scatter(x=mv["pred"], y=mv["actual"], mode="lines+markers", name="actual",
                             line=dict(color=ui.SERIES[2], width=2), marker=dict(size=8)))
    m = max(mv["pred"].max(), mv["actual"].max())
    fig.add_trace(go.Scatter(x=[0, m], y=[0, m], mode="lines", name="perfect", line=dict(color=ui.MUTED, dash="dot", width=1)))
    fig.update_xaxes(title="Predicted move ±%")
    fig.update_yaxes(title="Actual move ±%")
    with c[0]:
        ui.show(ui.style(fig, 360, "Expected move vs what happened (options view)", hover="closest"))
    # 4. hit rate over time
    hr = bt.assign(right=((bt["p_up"] >= .5) == (bt["y_ret"] > 0))).set_index("date")["right"]
    hr = hr.groupby(level=0).mean().rolling(20 if h == "1d" else 4, min_periods=1).mean()
    fig = go.Figure(go.Scatter(x=hr.index, y=100 * hr, mode="lines", line=dict(color=ui.SERIES[0], width=2),
                               name="hit rate"))
    fig.add_hline(y=50, line=dict(color=ui.MUTED, dash="dot", width=1))
    fig.update_yaxes(title="% of predictions right (rolling)")
    with c[1]:
        ui.show(ui.style(fig, 360, "Hit rate over the test window"))


def render():
    ui.page_header("Chart builder", "Preset charts of the model's predictions, or build your own from live "
                   "predictions, the 2-year backtest or saved prediction history.", kicker="Research")
    hs = ui.available_horizons()
    if not hs:
        st.info("Train the models first: `python scripts/train.py`")
        return
    c = st.columns([1, 2])
    h = c[0].selectbox("Horizon", hs, format_func=config.horizon_label, index=hs.index("1d") if "1d" in hs else 0)
    tab1, tab2 = st.tabs(["Preset charts", "Build your own"])
    with tab1:
        _presets(h)
    with tab2:
        src = st.radio("Data", ["Latest predictions", "Backtest (last 2 years)", "Prediction history"], horizontal=True)
        d = _source(h, src)
        if d.empty:
            st.info("No data for this source yet.")
            return
        f = st.columns(3)
        min_px = f[0].number_input("Min price", 0.0, 1000.0, float(config.MIN_PRICE))
        min_dv = f[1].number_input("Min $ volume/day (millions)", 0.0, 10000.0, 5.0)
        tick = f[2].multiselect("Only these tickers", sorted(d["ticker"].unique()))
        if "close" in d:
            d = d[d["close"] >= min_px]
        if "dollar_vol" in d:
            d = d[d["dollar_vol"].fillna(0) >= min_dv * 1e6]
        if tick:
            d = d[d["ticker"].isin(tick)]
        num = [col for col in d.columns if d[col].dtype.kind in "fi" and col not in ("horizon",)]
        k = st.columns(5)
        kind = k[0].selectbox("Chart", ["Scatter", "Histogram", "Box", "Top N bar", "Line over time"])
        x = k[1].selectbox("X", (["date"] if "date" in d and kind == "Line over time" else []) + num,
                           index=0, format_func=lambda v: NICE.get(v, v))
        y = k[2].selectbox("Y", num, index=num.index("exp_ret") if "exp_ret" in num else 0,
                           format_func=lambda v: NICE.get(v, v))
        color = k[3].selectbox("Color", ["none", "side", "ticker"] + num, format_func=lambda v: NICE.get(v, v))
        size = k[4].selectbox("Size", ["none"] + num, format_func=lambda v: NICE.get(v, v))
        if "p_up" in d:
            d["side"] = np.where(d["p_up"] >= .5, "leans up", "leans down")
        cmap = {"leans up": ui.UP, "leans down": ui.DOWN}
        col = None if color == "none" else color
        if len(d) > 30000 and kind == "Scatter":
            d = d.sample(30000, random_state=0)
            st.caption("Showing a random 30,000 points.")
        labels = {v: NICE.get(v, v) for v in d.columns}
        if kind == "Scatter":
            fig = px.scatter(d, x=x, y=y, color=col, size=None if size == "none" else d[size].abs(),
                             hover_name="ticker", color_discrete_map=cmap, labels=labels, opacity=0.7)
            hover = "closest"
        elif kind == "Histogram":
            fig = px.histogram(d, x=y, color=col, nbins=60, color_discrete_map=cmap, labels=labels, barmode="overlay")
            hover = "closest"
        elif kind == "Box":
            fig = px.box(d, x=col if col else None, y=y, color=col, color_discrete_map=cmap, labels=labels)
            hover = "closest"
        elif kind == "Top N bar":
            n = st.slider("N", 5, 50, 20)
            top = d.groupby("ticker")[y].mean().sort_values(ascending=False).head(n)
            fig = go.Figure(go.Bar(x=top.index, y=top.values, marker_color=ui.SERIES[0]))
            fig.update_yaxes(title=NICE.get(y, y))
            hover = "closest"
        else:
            if tick:
                fig = px.line(d.sort_values("date"), x="date", y=y, color="ticker", labels=labels)
            else:
                agg = d.groupby("date")[y].mean().reset_index()
                fig = px.line(agg, x="date", y=y, labels=labels)
                st.caption("Average across all stocks per date. Pick tickers above to see individual lines.")
            hover = "x unified"
        ui.show(ui.style(fig, 480, hover=hover))
        st.download_button("Download the data behind this chart", d.to_csv(index=False), f"chart_{h}.csv")
