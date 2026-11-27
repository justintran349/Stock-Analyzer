from __future__ import annotations

import numpy as np
import pandas as pd
import plotly.graph_objects as go
import streamlit as st

from stocks import config, model, ui


def render():
    hs = ui.available_horizons()
    ui.page_header("Backtest", "The last 2 years, graded as if they never happened: the model never saw them "
                   "and nothing about them influenced its settings.", kicker="Validate")
    if not hs:
        st.info("Train first: `python scripts/train.py`")
        return
    h = st.segmented_control("Horizon", hs, format_func=config.horizon_label,
                             default="1d" if "1d" in hs else hs[0], key="bt_h") or hs[0]
    met = ui.metrics(h)
    bt = ui.backtest(h)
    if not met or bt.empty:
        st.warning("No backtest saved for this horizon.")
        return
    d = met["dates"]
    ui.note(f"Trained on data up to <b>{d['train_end']}</b> · tuned on {d['valid_start']} → "
            f"{d.get('valid_end', d['test_start'])} · graded on <b>{d['test_start']} → {d['test_end']}</b> "
            f"({met['rows']['test']:,} predictions it never saw)" +
            (" · re-trained every quarter (walk-forward)" if met.get("walk_forward") else ""))
    t = met["test"]
    c = st.columns(5)
    c[0].metric("Right direction", f"{t['accuracy']:.1%}", f"{100 * (t['accuracy'] - t['accuracy_always_up']):+.1f} pts vs always 'up'")
    c[1].metric("AUC", f"{t['auc']:.3f}", help="0.5 = coin flip, 1.0 = perfect. 0.53+ is useful in markets.")
    c[2].metric("Same-day ranking AUC", f"{t['same_day_auc']:.3f}",
                help="Can it rank stocks against each other on the same day? This is what picking stocks needs.")
    c[3].metric("Brier score", f"{t['brier']:.4f}", f"{t['brier'] - t['brier_baseline']:+.4f} vs baseline",
                delta_color="inverse", help="Lower is better.")
    c[4].metric("Return ranking (IC)", f"{t['return_rank_ic']:.3f}",
                help="Correlation between predicted and actual return ranks each day. 0.02–0.05 is good in practice.")
    if "beat" in t:
        b = t["beat"]
        c = st.columns(5)
        c[0].metric("Beat-the-S&P: right", f"{b['accuracy']:.1%}",
                    f"{100 * (b['accuracy'] - b['accuracy_coin']):+.1f} pts vs best guess",
                    help="Second model: does the stock do better than the S&P 500 over the horizon? "
                         "It ignores which way the market goes, so it measures pure stock-picking.")
        c[1].metric("Beat-the-S&P AUC", f"{b['auc']:.3f}")
        ls = b.get("strategy", {}).get("long_short", {})
        if ls:
            c[2].metric("Long/short ranked by beat model", f"{ls['total_return_%']:+.1f}%",
                        help="Top-N long and bottom-N short using the beat-the-S&P chance, with default settings.")

    ui.section("Strategy simulator", "Each test date: buy the N most-likely-up stocks and short the N most-likely-"
               "down, hold for the horizon, pay trading costs. Re-runs instantly on the saved test predictions.")
    s = st.columns(6)
    top_n = s[0].slider("Stocks each side", 1, 50, config.BACKTEST_TOP_N)
    cost = s[1].number_input("Cost per trade (bps)", 0.0, 50.0, float(config.COST_BPS), 1.0)
    min_px = s[2].number_input("Min price", 0.0, 500.0, float(config.MIN_PRICE))
    min_dv = s[3].selectbox("Min $ volume", [1e6, 5e6, 2e7, 1e8, 5e8], index=1, format_func=ui.money)
    rank_opts = ["p_up", "p_beat", "exp_ret"] if "p_beat" in bt else ["p_up", "exp_ret"]
    rank = s[4].selectbox("Rank by", rank_opts, format_func={"p_up": "chance up", "p_beat": "chance to beat S&P",
                                                            "exp_ret": "expected return"}.get)
    min_conf = s[5].slider("Min confidence", 0.5, 0.75, 0.5, 0.01)
    n = config.HORIZONS[h][1]
    ppy = 252 * 7 if config.HORIZONS[h][2] == "hourly" else 252 / n
    res = model.strategy(bt, top_n=top_n, cost_bps=cost, min_price=min_px, min_dollar_vol=min_dv,
                         rank_by=rank, min_conf=min_conf, periods_per_year=ppy)
    ts = res["series"]
    if ts.empty:
        st.warning("No trades with these filters.")
    else:
        names = {"long": f"Buy top {top_n}", "short": f"Short bottom {top_n}", "long_short": "Long + short",
                 "market": "S&P 500 (SPY)", "all_stocks": "Average stock"}
        fig = go.Figure()
        for i, (col, label) in enumerate(names.items()):
            if col in ts and ts[col].notna().any():
                eq = 100 * ((1 + ts[col].fillna(0)).cumprod() - 1)
                fig.add_trace(go.Scatter(x=ts.index, y=eq, mode="lines", name=label,
                                         line=dict(color=ui.SERIES[i], width=2.5 if col == "long_short" else 2,
                                                   dash="dot" if col in ("market", "all_stocks") else None)))
        fig.update_yaxes(title="Cumulative return (%)")
        ui.show(ui.style(fig, 430, "Growth over the test window"))
        stats = pd.DataFrame(res["stats"]).T.rename(index=names)
        stats.index.name = "Strategy"
        st.dataframe(stats, column_config={
            "total_return_%": st.column_config.NumberColumn("Total return", format="%+.1f%%"),
            "annual_return_%": st.column_config.NumberColumn("Annualized", format="%+.1f%%"),
            "sharpe": st.column_config.NumberColumn("Sharpe", format="%.2f"),
            "win_rate": st.column_config.NumberColumn("Winning periods", format="percent"),
            "max_drawdown_%": st.column_config.NumberColumn("Max drawdown", format="%.1f%%"),
            "pick_hit_rate": st.column_config.NumberColumn("Pick hit rate", format="percent")})
        if h == "1h":
            st.caption("Hourly trading pays costs every hour; even a real edge can disappear after costs. "
                       "Try cost = 0 to see the raw signal.")

    c1, c2 = st.columns(2)
    with c1:
        ct = pd.DataFrame(t["conf_table"])
        if not ct.empty:
            fig = go.Figure(go.Bar(x=ct["bin"], y=100 * ct["hit_rate"],
                                   marker_color=[ui.UP if v > .5 else ui.DOWN for v in ct["hit_rate"]],
                                   customdata=ct["picks"], hovertemplate="%{x}: %{y:.1f}% right "
                                                                         "(%{customdata:,} picks)<extra></extra>"))
            fig.add_hline(y=50, line=dict(color=ui.MUTED, dash="dot", width=1))
            fig.update_xaxes(title="Model confidence")
            fig.update_yaxes(title="% right in the test")
            ui.show(ui.style(fig, 360, "Hit rate by confidence", hover="closest"))
            st.caption("This is the 'tested hit rate' shown on each pick card.")
    with c2:
        cal = pd.DataFrame(t["calibration"])
        if not cal.empty:
            fig = go.Figure()
            fig.add_trace(go.Scatter(x=100 * cal["predicted"], y=100 * cal["actual"], mode="lines+markers",
                                     name="model", line=dict(color=ui.SERIES[0], width=2), marker=dict(size=8)))
            lo, hi = 100 * cal["predicted"].min(), 100 * cal["predicted"].max()
            fig.add_trace(go.Scatter(x=[lo, hi], y=[lo, hi], mode="lines", name="perfect",
                                     line=dict(color=ui.MUTED, dash="dot", width=1)))
            fig.update_xaxes(title="Predicted chance up (%)")
            fig.update_yaxes(title="Actually went up (%)")
            ui.show(ui.style(fig, 360, "Calibration: does 60% mean 60%?", hover="closest"))

    gi = met.get("group_importance") or {}
    if gi:
        ui.section("Which kinds of data mattered", "Drop in AUC when each group of inputs is scrambled")
        g = pd.Series(gi).sort_values()
        fig = go.Figure(go.Bar(x=g.values, y=g.index, orientation="h",
                               marker_color=[ui.SERIES[0] if v > 0 else ui.MUTED for v in g.values]))
        fig.update_xaxes(title="Drop in AUC when this group is scrambled (bigger = more important)")
        ui.show(ui.style(fig, 60 + 30 * len(g), hover="closest"))
    with st.expander("Tuning: settings tried on the validation year"):
        tr = pd.DataFrame(met["tuning"]["trials"])
        tr["params"] = tr["params"].astype(str)
        st.dataframe(tr, hide_index=True)
        st.caption(f"Chosen: **{met['tuning']['best']['feature_set']}**. Settings were picked on the validation "
                   "year only, so the test score above is a fair estimate. Re-train with "
                   "`python scripts/train.py --walk-forward` to simulate re-training every quarter.")
