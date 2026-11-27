from __future__ import annotations

import pandas as pd
import plotly.graph_objects as go
import streamlit as st

from stocks import config, tracker, ui


def render():
    ui.page_header("Tracker", "Every day's predictions are saved. Once the horizon passes, the top picks each way "
                   "are graded against what actually happened. This is the real scoreboard.", kicker="Validate")
    c = st.columns(3)
    n = c[0].slider("Top picks per side per day", 1, 50, 10)
    if not config.PRED_HISTORY_FILE.exists():
        st.info("No live predictions yet. They're saved every time `scripts/daily.py` or `scripts/train.py` runs.")
        return
    df = tracker.grade(top_n=n)
    if df.empty or not df["graded"].any():
        st.info("Predictions are saved but none have reached their horizon yet. Check back tomorrow.")
        if not df.empty:
            st.dataframe(df.tail(50), hide_index=True)
        return
    summ = tracker.summary(df)
    summ["horizon"] = summ["horizon"].map(config.horizon_label)
    st.dataframe(summ, hide_index=True,
                 column_config={"hit_rate": st.column_config.NumberColumn(format="%.3f"),
                                "avg_return_pct": st.column_config.NumberColumn(format="%+.2f")})
    g = df[df["graded"]].copy()
    fig = go.Figure()
    for i, (h, part) in enumerate(g.groupby("horizon")):
        daily = part.groupby("date")["side_ret_%"].mean().sort_index()
        fig.add_trace(go.Scatter(x=daily.index, y=daily.cumsum(), mode="lines", name=config.horizon_label(h),
                                 line=dict(color=ui.SERIES[i], width=2)))
    fig.update_yaxes(title="Cumulative avg pick return (%, simple sum)")
    ui.show(ui.style(fig, 380, "Paper-trading results"))
    st.dataframe(g.sort_values("date", ascending=False)[["date", "horizon", "ticker", "side", "p_up", "close",
                                                         "fwd_close", "actual_ret_%", "right"]].head(500),
                 hide_index=True)
