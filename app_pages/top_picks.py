from __future__ import annotations

import html

import numpy as np
import pandas as pd
import plotly.graph_objects as go
import streamlit as st

from stocks import config, model, ui
from . import open_stock

DV_CHOICES = {"Any": 0, "$1M+": 1e6, "$5M+": 5e6, "$20M+": 2e7, "$100M+": 1e8, "$500M+": 5e8}


def _filters(h: str):
    with st.container(border=True):
        c = st.columns([1.3, 1, 1, 1.1, 1.3])
        sort = c[0].selectbox("Rank by", ["Chance up/down", "Chance to beat the S&P", "Expected return",
                                          "Expected move (options)"], key=f"sort_{h}",
                              help="'Beat the S&P' ignores which way the whole market goes: it picks stocks "
                                   "likely to do better (or worse) than the index.")
        min_conf = c[1].slider("Min confidence", 0.50, 0.80, 0.52, 0.01, key=f"conf_{h}",
                               help="How sure the model is, either way. 0.60 = 60% up or 60% down.")
        min_price = c[2].number_input("Min price ($)", 0.0, 1000.0, float(config.MIN_PRICE), 1.0, key=f"px_{h}")
        dv = c[3].selectbox("Avg. daily $ volume", list(DV_CHOICES), index=2, key=f"dv_{h}",
                            help="Liquid stocks have tighter spreads and real options markets.")
        earn = c[4].selectbox("Earnings in window", ["Any", "Within horizon", "Not within horizon"], key=f"earn_{h}",
                              help="Earnings inside the window make moves bigger.")
        c2 = st.columns([1.3, 1, 3.4])
        n = c2[0].slider("Picks per side", 3, 50, 10, key=f"n_{h}")
        search = c2[2].text_input("Tickers", key=f"search_{h}", placeholder="Filter by ticker, e.g. GE, F, BA")
        tested = c2[1].toggle("Proven levels", value=True, key=f"tested_{h}",
                              help="Only show confidence levels that were right more than half the time in the 2-year test.")
    return dict(min_conf=min_conf, min_price=min_price, min_dv=DV_CHOICES[dv], earn=earn, sort=sort,
                tested=tested, n=n, search=search)


def apply_filters(df: pd.DataFrame, f: dict, h: str) -> pd.DataFrame:
    d = df[(df["conf"] >= f["min_conf"]) & (df["close"] >= f["min_price"]) &
           (df["dollar_vol"].fillna(0) >= f["min_dv"])].copy()
    if "evt_tk_until_earnings" in d and f["earn"] != "Any":
        days = {"1h": 1, "1d": 1, "1w": 7, "1m": 30}[h]
        within = d["evt_tk_until_earnings"] <= days
        d = d[within] if f["earn"] == "Within horizon" else d[~within]
    if f["tested"] and d["tested_hit_rate"].notna().any():
        d = d[d["tested_hit_rate"].isna() | (d["tested_hit_rate"] > 0.5)]
    if f["search"].strip():
        want = [t.strip().upper() for t in f["search"].split(",") if t.strip()]
        d = d[d["ticker"].isin(want)]
    return d


def _card(r: pd.Series, h: str, rows: pd.DataFrame, side: str, beat_mode: bool = False):
    chance = r["p_up"] if side == "UP" else 1 - r["p_up"]
    word = ("beat S&P" if side == "UP" else "lag S&P") if beat_mode else ("chance up" if side == "UP" else "chance down")
    name = html.escape(r.get("name")) if isinstance(r.get("name"), str) else ""
    hr = r.get("tested_hit_rate")
    other = r.get("p_up_direction") if beat_mode else r.get("p_beat")
    stats = [("Price", f"${r['close']:,.2f}"), ("Exp. return", ui.pct(r["exp_ret"])),
             ("Exp. move", f"±{r['exp_move']:.1f}%"), ("Tested hit", "–" if pd.isna(hr) else f"{hr:.0%}")]
    tags = ""
    ue = r.get("evt_tk_until_earnings")
    if ue is not None and pd.notna(ue) and ue <= 30:
        tags += ui.pill(f"Earnings {int(ue)}d", "warn")
    if other is not None and pd.notna(other):
        tags += " " + ui.pill(f"{'Up' if beat_mode else 'Beat S&P'} {other:.0%}", "neutral")
    fr = rows[rows["ticker"] == r["ticker"]]
    why = " · ".join(model.reasons(fr.iloc[0])[:3]) if not fr.empty else ""
    with st.container(border=True):
        st.markdown(ui.pick_card_html(r["ticker"], name, chance, word, side, stats, html.escape(why), tags),
                    unsafe_allow_html=True)
        if st.button("Open analysis", key=f"an_{h}_{side}_{r['ticker']}", type="tertiary",
                     icon=":material/arrow_forward:"):
            open_stock(r["ticker"])


def render():
    ui.needs_data()
    hs = ui.available_horizons()
    if not hs:
        ui.page_header("Top picks", kicker="Predict")
        st.info("No trained models yet. Run `python scripts/train.py` in the terminal.")
        return
    labels = {h: config.horizon_label(h) for h in hs}
    h = st.session_state.get("tp_h") or ("1d" if "1d" in hs else hs[0])
    df = ui.latest(h)
    asof = pd.Timestamp(df["date"].max()) if not df.empty else None
    stamp = "" if asof is None else (f"{asof:%a %b %d, %Y}" + (f" · {asof:%H:%M} ET" if h == "1h" else " close"))
    ui.page_header("Top picks", "Stocks the models rate most likely to rise or fall, ranked and filtered.",
                   kicker="Predict", meta=f"As of <b>{stamp}</b><br>{len(df):,} stocks scored" if stamp else "")
    h = st.segmented_control("Horizon", hs, format_func=labels.get, default=h, key="tp_h",
                             label_visibility="collapsed") or hs[0]
    if df.empty:
        st.warning("No predictions saved for this horizon. Run `python scripts/daily.py`.")
        return
    met = ui.metrics(h)
    if met:
        t = met["test"]
        good = t["same_day_auc"] > 0.51 and t["strategy"].get("long_short", {}).get("total_return_%", 0) > 0
        ls = t["strategy"].get("long_short", {})
        ui.tiles([
            {"k": "Direction accuracy (2-yr test)", "v": f"{t['accuracy']:.1%}",
             "d": f"always 'up' scored {t['accuracy_always_up']:.1%}"},
            {"k": "Ranking skill (same-day AUC)", "v": f"{t['same_day_auc']:.3f}", "d": "0.500 = coin flip",
             "tone": "up" if t["same_day_auc"] > 0.51 else "down"},
            {"k": "Beat-the-S&P AUC", "v": f"{t['beat']['auc']:.3f}" if "beat" in t else "–",
             "d": "stock-picking only"},
            {"k": f"Top-{config.BACKTEST_TOP_N} long/short", "v": ui.pct(ls.get("total_return_%")) if ls else "–",
             "d": f"Sharpe {ls['sharpe']:.2f}" if ls and pd.notna(ls.get("sharpe")) else "",
             "tone": "up" if ls and ls.get("total_return_%", 0) > 0 else "down"},
        ])
        if not good:
            ui.note("This horizon did not clearly beat chance in the 2-year test. Treat its picks as weak.", "warn")
    rows = ui.latest_rows(h)
    if "evt_tk_until_earnings" not in df and not rows.empty and "evt_tk_until_earnings" in rows:
        df = df.merge(rows[["ticker", "evt_tk_until_earnings"]], on="ticker", how="left", suffixes=("", "_r"))
    f = _filters(h)
    if f["sort"].startswith("Chance to beat") and "p_beat" in df:
        df = df.copy()
        df["conf"] = df["beat_conf"]
        df["tested_hit_rate"] = df.get("beat_tested_hit_rate", np.nan)
    d = apply_filters(df, f, h)
    beat_mode = f["sort"].startswith("Chance to beat") and "p_beat" in d
    if beat_mode:   # reuse the same layout: 'up' = beats the S&P, 'down' = lags it
        d = d.copy()
        d["p_up_direction"] = d["p_up"]
        d["p_up"] = d["p_beat"]
        d["tested_hit_rate"] = d.get("beat_tested_hit_rate", np.nan)
        ui.note("Ranked by chance to beat the S&P 500 over the horizon, whatever the market does.")
    key = {"Chance up/down": "p_up", "Chance to beat the S&P": "p_up", "Expected return": "exp_ret",
           "Expected move (options)": "exp_move"}[f["sort"]]
    if key == "exp_move":
        ups = d.sort_values("exp_move", ascending=False).head(f["n"])
        ui.section(f"Largest expected moves", f"{len(d):,} stocks pass the filters · size of the move either way, "
                   "for comparing with option prices")
        cols = st.columns(2)
        for i, (_, r) in enumerate(ups.iterrows()):
            with cols[i % 2]:
                _card(r, h, rows, r["side"])
    else:
        ups = d[d["p_up"] >= 0.5].sort_values(key, ascending=False).head(f["n"])
        downs = d[d["p_up"] < 0.5].sort_values(key, ascending=True).head(f["n"])
        left, right = st.columns(2, gap="medium")
        with left:
            ui.section("Most likely to beat the S&P" if beat_mode else "Most likely up",
                       f"{len(d[d['p_up'] >= 0.5]):,} candidates pass the filters")
            if ups.empty:
                st.caption("None pass the filters.")
            for _, r in ups.iterrows():
                _card(r, h, rows, "UP", beat_mode)
        with right:
            ui.section("Most likely to lag the S&P" if beat_mode else "Most likely down",
                       f"{len(d[d['p_up'] < 0.5]):,} candidates pass the filters")
            if downs.empty:
                st.caption("None pass the filters.")
            for _, r in downs.iterrows():
                _card(r, h, rows, "DOWN", beat_mode)

    ui.section("Expected return vs expected move", "Every stock passing the filters · bubble size = trading volume")
    if not d.empty:
        fig = go.Figure()
        for side, color in (("UP", ui.UP), ("DOWN", ui.DOWN)):
            s = d[d["side"] == side]
            fig.add_trace(go.Scatter(
                x=s["exp_move"], y=s["exp_ret"], mode="markers", name=f"leans {side.lower()}",
                marker=dict(size=np.clip(np.log10(s["dollar_vol"].fillna(1e6)) * 2.2 - 8, 6, 22), color=color,
                            opacity=0.8, line=dict(width=1, color="#0B0E13")),
                customdata=np.c_[s["ticker"], 100 * s["p_up"]],
                hovertemplate="<b>%{customdata[0]}</b><br>chance up %{customdata[1]:.0f}%<br>"
                              "exp. return %{y:+.2f}%<br>exp. move ±%{x:.2f}%<extra></extra>"))
        fig.update_xaxes(title="Expected move either way (%)")
        fig.update_yaxes(title="Expected return (%)")
        ui.show(ui.style(fig, 420, hover="closest"))
        st.caption("Upper right: large move expected, leaning up. Lower right: large move, leaning down.")
    with st.expander("Full table", icon=":material/table:"):
        cols = ["ticker", "name", "close", "p_up", "p_beat", "exp_ret", "exp_move", "tested_hit_rate", "dollar_vol"]
        dd = d.assign(p_up=d["p_up_direction"]) if beat_mode else d
        show = dd[[c for c in cols if c in dd]]
        st.dataframe(show.sort_values("p_up", ascending=False), hide_index=True,
                     column_config={
                         "p_up": st.column_config.ProgressColumn("Chance up", min_value=0, max_value=1, format="%.2f"),
                         "p_beat": st.column_config.ProgressColumn("Beat S&P", min_value=0, max_value=1, format="%.2f"),
                         "exp_ret": st.column_config.NumberColumn("Exp. return %", format="%+.2f"),
                         "exp_move": st.column_config.NumberColumn("Exp. move ±%", format="%.2f"),
                         "tested_hit_rate": st.column_config.NumberColumn("Tested hit", format="%.2f"),
                         "close": st.column_config.NumberColumn("Price", format="$%.2f"),
                         "dollar_vol": st.column_config.NumberColumn("$ volume/day", format="%.0f")})
        st.download_button("Download CSV", show.to_csv(index=False), f"picks_{h}.csv")
