from __future__ import annotations

import re

import numpy as np
import pandas as pd
import plotly.graph_objects as go
import streamlit as st
from plotly.subplots import make_subplots

from stocks import config, events as ev_mod, model, ui

def _mentions(col: pd.Series, ticker: str) -> pd.Series:
    return col.str.contains(rf"(?:^|[;, ]){re.escape(ticker)}(?:$|[;, ])", regex=True)


RANGES = {"1M": 31, "6M": 183, "1Y": 365, "5Y": 1826, "Max": None}


def _price_chart(px: pd.DataFrame, ticker: str, rng: str, show_market_events: bool, evs: pd.DataFrame):
    if RANGES[rng]:
        px = px[px["date"] >= px["date"].max() - pd.Timedelta(days=RANGES[rng])]
    fig = go.Figure()
    if RANGES[rng] and RANGES[rng] <= 365:
        fig.add_trace(go.Candlestick(x=px["date"], open=px["open"], high=px["high"], low=px["low"],
                                     close=px["close"], name=ticker, increasing_line_color=ui.UP,
                                     decreasing_line_color=ui.DOWN))
        fig.update_layout(xaxis_rangeslider_visible=False)
    else:
        fig.add_trace(go.Scatter(x=px["date"], y=px["close"], mode="lines", name=ticker,
                                 line=dict(color=ui.UP, width=2)))
    lo, hi = px["date"].min(), px["date"].max()
    e = evs[(evs["date"] >= lo) & (evs["date"] <= hi)]
    e = e[_mentions(e["ticker"], ticker) | (show_market_events & (e["ticker"] == ""))]
    ymax = px["high"].max()
    for _, r in e.iterrows():
        color = ui.SERIES[3] if r["type"] == "earnings" else ui.SERIES[6]
        fig.add_vline(x=r["date"], line=dict(color=color, width=1, dash="dot"), opacity=0.6)
    if not e.empty:
        fig.add_trace(go.Scatter(x=e["date"], y=[ymax] * len(e), mode="markers", name="events",
                                 marker=dict(symbol="triangle-down", size=9,
                                             color=[ui.SERIES[3] if t == "earnings" else ui.SERIES[6] for t in e["type"]]),
                                 text=e["type"] + ": " + e["title"] + " " + e["notes"].fillna(""),
                                 hovertemplate="%{x|%b %d %Y}<br>%{text}<extra></extra>"))
    fig.update_yaxes(title="Price ($, split/dividend adjusted)")
    return ui.style(fig, 430, hover="x")


def _options_check(ticker: str, exp_moves: dict, hist_earn_move: float | None):
    """Live option chain from Yahoo → implied move from the at-the-money straddle."""
    try:
        import yfinance as yf

        tk = yf.Ticker(ticker)
        expiries = list(tk.options)[:8]
        if not expiries:
            st.info("No listed options for this ticker on Yahoo.")
            return
        spot = float(tk.history(period="5d")["Close"].iloc[-1])
        rows = []
        for exp in expiries:
            ch = tk.option_chain(exp)
            calls, puts = ch.calls, ch.puts
            k = calls.iloc[(calls["strike"] - spot).abs().argsort()[:1]]["strike"].iloc[0]
            c = calls[calls["strike"] == k].iloc[0]
            p = puts[puts["strike"] == k]
            if p.empty:
                continue
            p = p.iloc[0]
            mid = lambda o: (o["bid"] + o["ask"]) / 2 if o["bid"] > 0 and o["ask"] > 0 else o["lastPrice"]
            straddle = mid(c) + mid(p)
            days = (pd.Timestamp(exp) - pd.Timestamp.now().normalize()).days
            rows.append({"expiry": exp, "days": days, "strike": k, "straddle $": straddle,
                         "implied move ±%": 100 * straddle / spot,
                         "call IV": c["impliedVolatility"], "put IV": p["impliedVolatility"]})
        t = pd.DataFrame(rows)
        st.dataframe(t, hide_index=True,
                     column_config={"implied move ±%": st.column_config.NumberColumn(format="%.2f"),
                                    "straddle $": st.column_config.NumberColumn(format="$%.2f"),
                                    "call IV": st.column_config.NumberColumn(format="%.2f"),
                                    "put IV": st.column_config.NumberColumn(format="%.2f")})
        msgs = []
        for h, days in (("1w", 7), ("1m", 30)):
            if h in exp_moves and not t.empty:
                near = t.iloc[(t["days"] - days).abs().argsort()[:1]].iloc[0]
                im, mm = near["implied move ±%"], exp_moves[h]
                verdict = "options look **cheap** vs the model" if mm > im * 1.1 else \
                    "options look **expensive** vs the model" if mm < im * 0.9 else "roughly fair"
                msgs.append(f"{config.horizon_label(h)}: market prices ±{im:.1f}% ({near['expiry']}), "
                            f"model expects ±{mm:.1f}% → {verdict}")
        if hist_earn_move:
            msgs.append(f"Past earnings days moved this stock ±{hist_earn_move:.1f}% on average.")
        for m in msgs:
            st.markdown("- " + m)
        st.caption("The straddle (buy the call + put at the nearest strike) costs about the move the market "
                   "expects by expiry. Note: the model's 'move' is the average absolute move, which is close to "
                   "but not exactly what a straddle prices.")
    except Exception as e:
        st.warning(f"Couldn't load options ({type(e).__name__}: {e}). This needs internet on your Mac.")


def render():
    ui.needs_data()
    hs = ui.available_horizons()
    preds = {h: ui.latest(h) for h in hs}
    tickers = sorted(set().union(*[set(p["ticker"]) for p in preds.values() if not p.empty])) if preds else []
    if not tickers:
        tickers = ui.universe()["ticker"].tolist()
    default = st.session_state.get("ticker")
    idx = tickers.index(default) if default in tickers else 0
    head = st.container()
    ticker = st.selectbox("Stock", tickers, index=idx, format_func=ui.ticker_label, label_visibility="collapsed",
                          placeholder="Search a company or ticker, e.g. Boeing or BA")
    st.session_state["ticker"] = ticker
    px = ui.daily_prices((ticker,))
    secs = ui.sectors()
    srow = secs[secs["ticker"] == ticker] if not secs.empty else secs
    sub = ""
    peers_txt = ""
    if not srow.empty:
        sec, ind = srow["sector"].iloc[0], srow["industry"].iloc[0]
        sub = f"{sec} · {ind}"
        peers = secs.loc[(secs["industry"] == ind) & (secs["ticker"] != ticker), "ticker"].tolist()
        if peers and ind != "Unknown":
            peers_txt = "Peers: " + ", ".join(peers[:14]) + (" …" if len(peers) > 14 else "")
    meta = ""
    if not px.empty and len(px) > 1:
        last, prev = px["close"].iloc[-1], px["close"].iloc[-2]
        chg = 100 * (last / prev - 1)
        tone = ui.UP if chg >= 0 else ui.DOWN
        meta = (f"<b style='font-size:1.35rem'>${last:,.2f}</b> "
                f"<span style='color:{tone};font-weight:600'>{chg:+.2f}%</span><br>"
                f"close {px['date'].iloc[-1]:%b %d, %Y}")
    import html as _html
    name = _html.escape(ui.name_map().get(ticker, ""))
    with head:
        ui.page_header(f"{ticker} <span style='color:#8B93A1;font-weight:500;font-size:1.05rem'>{name}</span>",
                       sub, kicker="Stock analyzer", meta=meta)
    if peers_txt:
        st.caption(peers_txt)

    # ---- predictions for every horizon
    exp_moves = {}
    items = []
    for h in hs:
        p = preds[h]
        r = p[p["ticker"] == ticker]
        if r.empty:
            items.append({"k": config.horizon_label(h), "v": "–", "d": "not scored"})
            continue
        r = r.iloc[0]
        up = r["p_up"] >= 0.5
        chance = r["p_up"] if up else 1 - r["p_up"]
        d = f"exp. {ui.pct(r['exp_ret'])} · move ±{r['exp_move']:.1f}%"
        if pd.notna(r.get("p_beat")):
            d += f"<br>beat S&P {r['p_beat']:.0%}"
        if pd.notna(r.get("tested_hit_rate")):
            d += f" · tested hit {r['tested_hit_rate']:.0%}"
        items.append({"k": config.horizon_label(h), "v": f"{chance:.0%} {'up' if up else 'down'}",
                      "d": d, "tone": "up" if up else "down"})
        exp_moves[h] = r["exp_move"]
    ui.tiles(items)

    # ---- price chart with events
    ui.section("Price & events")
    evs = ui.events(include_earnings=True)
    c1, c2 = st.columns([3, 1])
    rng = c1.segmented_control("Range", list(RANGES), default="1Y", key="rng") or "1Y"
    mkt_ev = c2.toggle("Show market-wide events", value=False)
    if px.empty:
        st.warning("No price history for this ticker.")
        return
    ui.show(_price_chart(px, ticker, rng, mkt_ev, evs))
    st.caption("Markers: yellow = earnings, violet = your events (CEO changes, launches, insider buying, ...)")

    # ---- why
    ui.section("Why the model thinks this", "Each group of inputs switched to a typical value; the bar is how "
               "much the chance changes")
    if hs:
        h = st.segmented_control("Explain horizon", hs, format_func=config.horizon_label,
                                 default="1d" if "1d" in hs else hs[0], key="why_h") or hs[0]
        rows = ui.latest_rows(h)
        b = ui.bundle(h)
        row = rows[rows["ticker"] == ticker]
        if b is not None and not row.empty:
            for f in b["features"]:
                if f not in row:
                    row[f] = np.nan
            facts = model.reasons(row.iloc[0])
            if facts:
                st.markdown("**What stands out:** " + " · ".join(facts))
            ex = model.explain_row(b, row)
            fig = go.Figure(go.Bar(x=ex["effect_pts"], y=ex["group"], orientation="h",
                                   marker_color=[ui.UP if v >= 0 else ui.DOWN for v in ex["effect_pts"]],
                                   hovertemplate="%{y}: %{x:+.1f} pts<extra></extra>"))
            fig.update_xaxes(title="Change in chance of going up (percentage points)", zeroline=True)
            fig.update_yaxes(autorange="reversed")
            ui.show(ui.style(fig, 60 + 32 * len(ex), "What moved the chance most", hover="closest"))
            st.caption("Each bar: switch that group of inputs to a typical value, re-run the model, "
                       "and see how much the chance changes. Blue pushes up, orange pushes down.")

    # ---- track record on this stock
    ui.section("Track record on this stock", "Predictions vs what happened, 2-year test window")
    if hs:
        h2 = st.segmented_control("Track record horizon", [h for h in hs if h != "1h"] or hs,
                                  format_func=config.horizon_label, default="1d" if "1d" in hs else hs[0],
                                  key="trk_h") or hs[0]
        bt = ui.backtest(h2)
        s = bt[bt["ticker"] == ticker].sort_values("date") if not bt.empty else bt
        if s.empty:
            st.caption("This stock wasn't in the test set (too new, too cheap or too illiquid).")
        else:
            s = s.copy()
            s["actual_%"] = 100 * (np.exp(s["y_ret"]) - 1)
            s["right"] = (s["p_up"] >= 0.5) == (s["y_ret"] > 0)
            fig = make_subplots(rows=2, cols=1, shared_xaxes=True, vertical_spacing=0.06,
                                subplot_titles=("Model's chance of going up", "What actually happened (%)"))
            fig.add_trace(go.Scatter(x=s["date"], y=100 * s["p_up"], mode="lines", name="chance up",
                                     line=dict(color=ui.SERIES[0], width=2)), 1, 1)
            fig.add_hline(y=50, line=dict(color=ui.MUTED, width=1, dash="dot"), row=1, col=1)
            fig.add_trace(go.Bar(x=s["date"], y=s["actual_%"], name="actual return",
                                 marker_color=[ui.UP if v > 0 else ui.DOWN for v in s["actual_%"]]), 2, 1)
            fig.update_layout(showlegend=False)
            ui.show(ui.style(fig, 460))
            st.caption(f"{len(s)} test predictions · right {s['right'].mean():.0%} of the time · "
                       f"when it said up ≥ 60%: " +
                       (f"{(s.loc[s['p_up'] >= .6, 'y_ret'] > 0).mean():.0%} went up"
                        if (s["p_up"] >= .6).any() else "never happened"))

    # ---- events and options
    ui.section("Events", "Earnings and your events for this stock")
    rets = np.log(px.set_index("date")["close"]).diff()
    mine = evs[_mentions(evs["ticker"], ticker)]
    earn = mine[mine["type"] == "earnings"]
    hist_move = None
    if not earn.empty:
        past = earn[earn["date"] <= px["date"].max()]
        mv = ev_mod.event_vs_normal_moves(rets, past["date"].tolist())
        hist_move = mv["avg_abs_event_move_%"]
        nxt = earn[earn["date"] > px["date"].max()]
        c = st.columns(4)
        c[0].metric("Earnings reports on file", mv["events"])
        c[1].metric("Avg move on earnings day", ui.pct(mv["avg_abs_event_move_%"], sign=False))
        c[2].metric("Normal day move", ui.pct(mv["normal_day_abs_move_%"], sign=False))
        c[3].metric("Next earnings", f"{nxt['date'].min():%b %d}" if not nxt.empty else "unknown")
    other = mine[mine["type"] != "earnings"]
    if not other.empty:
        ctx = ui.context_returns()
        mkt = ctx["spy"] if "spy" in ctx else None
        wide = pd.DataFrame({ticker: rets})
        res = ev_mod.event_study(other, wide, mkt.reindex(wide.index) if mkt is not None else None)
        if res["n"]:
            tb = res["table"].drop(columns=["beta"])
            st.dataframe(tb, hide_index=True, column_config=ui.EVENT_TABLE_COLS)
    elif earn.empty:
        st.caption("No events on file for this stock. Add some on the Events page.")

    # ---- insiders
    trades = ui.insider_trades()
    mine_t = trades[trades["ticker"] == ticker] if not trades.empty else trades
    if not mine_t.empty:
        ui.section("Insider trading", "Open-market purchases and sales from SEC Form 4 filings")
        recent = mine_t[mine_t["filing_date"] >= mine_t["filing_date"].max() - pd.Timedelta(days=730)]
        q = recent.assign(month=recent["filing_date"].dt.to_period("M").dt.to_timestamp()) \
            .groupby(["month", "side"])["value"].sum().unstack(fill_value=0)
        fig = go.Figure()
        if "buy" in q:
            fig.add_trace(go.Bar(x=q.index, y=q["buy"] / 1e6, name="bought", marker_color=ui.UP))
        if "sell" in q:
            fig.add_trace(go.Bar(x=q.index, y=-q["sell"] / 1e6, name="sold", marker_color=ui.DOWN))
        fig.update_layout(barmode="relative")
        fig.update_yaxes(title="$ millions (sales shown below zero)")
        ui.show(ui.style(fig, 300, "Open-market insider trades, last 2 years"))
        st.dataframe(mine_t.sort_values("filing_date", ascending=False).head(25)[
            ["filing_date", "owner_name", "side", "shares", "price", "value", "is_exec"]],
            hide_index=True, column_config={"value": st.column_config.NumberColumn(format="$%.0f"),
                                            "price": st.column_config.NumberColumn(format="$%.2f"),
                                            "is_exec": st.column_config.CheckboxColumn("CEO/CFO/Pres.")})

    # ---- saved option snapshots
    osum = ui.options_summary()
    mine_o = osum[osum["ticker"] == ticker] if not osum.empty else osum
    if not mine_o.empty:
        ui.section("Options market", "From the daily option-chain snapshots")
        last = mine_o[mine_o["date"] == mine_o["date"].max()].sort_values("days")
        c1, c2 = st.columns(2)
        fig = go.Figure(go.Scatter(x=last["days"], y=last["implied_move_pct"], mode="lines+markers",
                                   line=dict(color=ui.SERIES[0], width=2), marker=dict(size=8),
                                   customdata=last["expiry"].dt.strftime("%b %d"),
                                   hovertemplate="%{customdata}: ±%{y:.1f}%<extra></extra>"))
        fig.update_xaxes(title="Days to expiry")
        fig.update_yaxes(title="Implied move ±%")
        with c1:
            ui.show(ui.style(fig, 320, f"Implied move by expiry ({last['date'].iloc[0]:%b %d})", hover="closest"))
        hist = mine_o[(mine_o["days"] >= 20) & (mine_o["days"] <= 40)].groupby("date")["atm_iv"].mean()
        with c2:
            if len(hist) >= 5:
                fig = go.Figure(go.Scatter(x=hist.index, y=100 * hist.values, mode="lines",
                                           line=dict(color=ui.SERIES[1], width=2)))
                fig.update_yaxes(title="ATM implied volatility (%)")
                ui.show(ui.style(fig, 320, "1-month implied volatility over time"))
            else:
                st.caption("Implied-volatility history builds up as the daily job saves snapshots "
                           f"({len(hist)} day{'s' if len(hist) != 1 else ''} so far).")

    with st.expander("Live options check (Yahoo)", icon=":material/candlestick_chart:"):
        st.caption("Compares what the options market expects with the model's expected move and past earnings moves.")
        if st.button("Load option chain", key="opt"):
            _options_check(ticker, exp_moves, hist_move)
