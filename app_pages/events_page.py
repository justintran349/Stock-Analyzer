from __future__ import annotations

import numpy as np
import pandas as pd
import plotly.graph_objects as go
import streamlit as st

from stocks import config, events as ev_mod, ui

ECON_SERIES = ["spy", "nasdaq100", "russell2000", "vix", "tnx", "oil", "gold", "copper", "dollar",
               "bitcoin", "tlt", "hyg", "xlk", "xlf", "xle", "xlv", "xli", "xly", "xlp", "xlu", "xlb",
               "xlre", "xlc", "semis", "homebuilders", "airlines", "transports", "retail", "shipping"]
NICE = {"spy": "S&P 500 (SPY)", "vix": "VIX (fear index, pts/100)", "tnx": "10-yr yield (pts/100)",
        "tlt": "Long bonds (TLT)", "hyg": "Junk bonds (HYG)", "xlk": "Tech", "xlf": "Financials",
        "xle": "Energy", "xlv": "Health care", "xli": "Industrials", "xly": "Consumer discretionary",
        "xlp": "Consumer staples", "xlu": "Utilities", "xlb": "Materials", "xlre": "Real estate",
        "xlc": "Communication"}


def _study_tab():
    evs = ui.events(include_earnings=True)
    types = sorted(evs["type"].unique())
    c = st.columns([2, 2, 1, 1])
    sel_types = c[0].multiselect("Event types", types, default=[t for t in ["war"] if t in types] or types[:1])
    sub = evs[evs["type"].isin(sel_types)]
    if "earnings" in sel_types and len(sub) > 3000:
        st.caption("Lots of earnings events: pick a ticker below to keep it fast.")
    titles = (sub["date"].dt.strftime("%Y-%m-%d") + " · " + sub["ticker"].replace("", "market") + " · " +
              sub["title"]).tolist()
    picked = c[1].multiselect("Specific events (blank = all of these types)", titles)
    if picked:
        sub = sub[titles_match(sub, picked)]
    pre = c[2].number_input("Days before", 1, 60, 5)
    post = c[3].number_input("Days after", 1, 120, 20)
    c2 = st.columns([2, 2])
    target = c2[0].radio("Measure the reaction of", ["The event's own stock (or S&P 500 if market-wide)",
                                                     "A stock or series I pick"], horizontal=False)
    tick = None
    if target.startswith("A stock"):
        opts = ["spy"] + ECON_SERIES[1:] + ui.universe()["ticker"].tolist()
        tick = c2[1].selectbox("Stock / series", opts, format_func=lambda t: NICE.get(t, t))
    if sub.empty:
        st.info("No events match.")
        return
    rets = ui.daily_returns_wide()
    ctx = ui.context_returns()
    if rets.empty:
        ui.needs_data()
    rets_all = rets.join(ctx, how="left") if not ctx.empty else rets
    market = rets_all["spy"] if "spy" in rets_all else None
    ev = ev_mod.explode_tickers(sub)
    if tick:
        ev["ticker"] = tick
        default = tick
    else:
        ev["ticker"] = ev["ticker"].replace("", "spy")
        default = "spy"
    if len(ev) > 2000:
        ev = ev.sample(2000, random_state=0)
        st.caption("Showing a random 2,000 events.")
    raw_series = bool(tick) and tick in ctx.columns      # VIX, oil, sectors...: raw moves
    res = ev_mod.event_study(ev, rets_all, None if raw_series else market, default_ticker=default,
                             pre=int(pre), post=int(post))
    if not res["n"]:
        st.warning("None of these events had price data for that stock/series.")
        return
    t = res["table"]
    m = st.columns(5)
    m[0].metric("Events", res["n"])
    m[1].metric("Avg reaction on the day", ui.pct(t["day0_move_%"].mean(), 2))
    m[2].metric("Event-day move vs normal", f"{t['abs_day0_vs_normal'].mean():.1f}×",
                help="Size of the event-day move (either way) compared with a normal day. >1.5× = options often pay.")
    m[3].metric("Avg after 5 days", ui.pct(t.get("car_5d_%", pd.Series(dtype=float)).mean(), 2))
    m[4].metric("Volatility after / before", f"{t['vol_after_%'].mean():.0f}% / {t['vol_before_%'].mean():.0f}%",
                help="Annualized volatility in the month after vs the month before. Higher after = options get pricier.")
    show_paths = st.toggle("Show each event's path", value=res["n"] <= 30)
    fig = go.Figure()
    x = res["mean"].index
    mean = 100 * (np.exp(res["mean"]) - 1)
    up = 100 * (np.exp(res["mean"] + res["ci"]) - 1)
    lo = 100 * (np.exp(res["mean"] - res["ci"]) - 1)
    if show_paths:
        for name, row in res["paths"].iterrows():
            fig.add_trace(go.Scatter(x=row.index, y=100 * (np.exp(row) - 1), mode="lines", name=name,
                                     line=dict(width=1, color="rgba(137,135,129,0.35)"), showlegend=False,
                                     hovertemplate=f"{name}<br>day %{{x}}: %{{y:+.1f}}%<extra></extra>"))
    fig.add_trace(go.Scatter(x=list(x) + list(x[::-1]), y=list(up) + list(lo[::-1]), fill="toself",
                             fillcolor="rgba(42,120,214,0.15)", line=dict(width=0), name="95% range of the average",
                             hoverinfo="skip"))
    fig.add_trace(go.Scatter(x=x, y=mean, mode="lines+markers", name="average",
                             line=dict(color=ui.UP, width=2.5), marker=dict(size=6),
                             hovertemplate="day %{x}: %{y:+.2f}%<extra>average</extra>"))
    fig.add_vline(x=0, line=dict(color=ui.MUTED, dash="dot"))
    fig.add_hline(y=0, line=dict(color=ui.MUTED, width=1))
    fig.update_xaxes(title="Trading days from the event")
    fig.update_yaxes(title="Cumulative return (%)" if raw_series or default == "spy"
                     else "Cumulative return beyond the market (%)")
    ui.show(ui.style(fig, 420, "Average path around the event", hover="closest"))
    st.caption("Stocks: return beyond what the market move explains (beta-adjusted). S&P 500 and series: raw return. "
               "If the shaded band crosses zero, the average effect could be luck.")
    st.dataframe(t.drop(columns=["beta"]), hide_index=True, column_config=ui.EVENT_TABLE_COLS)

    # economy-wide reaction
    mkt_events = sub[sub["ticker"] == ""]
    if len(mkt_events) >= 3:
        ui.section("Impact on the economy", "Average move of each market and sector after these events")
        k = st.slider("Days after the event", 0, 30, 5, key="xs_days")
        dates = mkt_events["date"].tolist()
        econ = ctx[[c for c in ECON_SERIES if c in ctx.columns]]
        er = ev_mod.cross_section_reaction(dates, econ, None, days=(0, k), min_events=2)
        if not er.empty:
            er = er.reset_index().rename(columns={"index": "series"})
            er["label"] = er["series"].map(lambda s: NICE.get(s, s))
            fig = go.Figure(go.Bar(x=er["avg_move_%"], y=er["label"], orientation="h",
                                   marker_color=[ui.UP if v >= 0 else ui.DOWN for v in er["avg_move_%"]],
                                   customdata=np.c_[er["up_share"] * 100, er["events"]],
                                   hovertemplate="%{y}: %{x:+.2f}%<br>up in %{customdata[0]:.0f}% of "
                                                 "%{customdata[1]} events<extra></extra>"))
            fig.update_xaxes(title=f"Average move, day 0 to day {k} (%)")
            ui.show(ui.style(fig, 40 + 24 * len(er), hover="closest"))
        ui.section("Stocks that react most", "Average move beyond the S&P 500 after these events")
        xs = ev_mod.cross_section_reaction(dates, rets, market, days=(0, k), min_events=3)
        if not xs.empty:
            xs = xs[np.abs(xs["t_stat"]) > 0].dropna()
            c = st.columns(2)
            c[0].caption("Outperformed the market")
            c[0].dataframe(xs.sort_values("avg_move_%", ascending=False).head(20))
            c[1].caption("Underperformed the market")
            c[1].dataframe(xs.head(20))
            st.caption("t-stat above ~2 (or below −2) means the pattern is unlikely to be pure luck — "
                       "though with thousands of stocks some will look strong by chance.")


def titles_match(sub: pd.DataFrame, picked: list) -> pd.Series:
    key = sub["date"].dt.strftime("%Y-%m-%d") + " · " + sub["ticker"].replace("", "market") + " · " + sub["title"]
    return key.isin(picked)


def _edit_tab():
    st.caption("Add a row for any event. Leave **ticker** blank for market-wide events. "
               "**type** is free text: type a new one (e.g. `fda_approval`, `fomc`, `pandemic`) and it "
               "becomes a new event type everywhere, including in the model after the next retrain "
               "(once it has 3+ events). Several tickers: `BA;GE`.")
    df = ev_mod.load_events(include_auto=False)
    edited = st.data_editor(df, num_rows="dynamic", hide_index=True,
                            column_config={"date": st.column_config.DateColumn("date", required=True),
                                           "type": st.column_config.TextColumn("type", required=True,
                                                                               help="; ".join(config.DEFAULT_EVENT_TYPES)),
                                           "ticker": st.column_config.TextColumn("ticker"),
                                           "title": st.column_config.TextColumn("title", width="large"),
                                           "notes": st.column_config.TextColumn("notes")},
                            key="ev_editor")
    if st.button("Save events", type="primary", icon=":material/save:"):
        ev_mod.save_events(edited)
        ui.events.clear()
        st.success(f"Saved {len(edited)} events to data/events.csv")
    st.caption("Not listed here because they're added automatically: earnings dates (Yahoo), the economic "
               "calendar — Fed decisions, CPI, jobs, GDP, PCE, PPI, retail sales — and insider buying (SEC).")


def _upcoming_tab():
    evs = ui.events(include_earnings=True)
    today = pd.Timestamp.now().normalize()
    fut = evs[(evs["date"] >= today) & (evs["date"] <= today + pd.Timedelta(days=45))].copy()
    if fut.empty:
        st.caption("Nothing in the next 45 days. Run `python scripts/download_data.py --earnings` for earnings dates.")
        return
    preds = ui.latest("1w")
    past = evs[(evs["type"] == "earnings") & (evs["date"] < today)]
    rets = ui.daily_returns_wide(start=(today - pd.Timedelta(days=365 * 6)).strftime("%Y-%m-%d"))
    rows = []
    for _, r in fut.iterrows():
        row = {"date": r["date"].date(), "type": r["type"], "ticker": r["ticker"] or "market", "title": r["title"]}
        if r["ticker"] and r["ticker"] in rets.columns:
            mv = ev_mod.event_vs_normal_moves(rets[r["ticker"]].dropna(),
                                              past.loc[past["ticker"] == r["ticker"], "date"].tolist())
            row["past avg event move ±%"] = mv["avg_abs_event_move_%"]
            row["normal day ±%"] = mv["normal_day_abs_move_%"]
            if not preds.empty:
                p = preds[preds["ticker"] == r["ticker"]]
                if not p.empty:
                    row["model week: chance up"] = p["p_up"].iloc[0]
                    row["model week: move ±%"] = p["exp_move"].iloc[0]
        rows.append(row)
    t = pd.DataFrame(rows)
    st.dataframe(t, hide_index=True,
                 column_config={c: st.column_config.NumberColumn(format="%.2f") for c in t.columns
                                if t[c].dtype.kind == "f"})
    st.caption("For options: an event stock whose past event moves are much bigger than a normal day is "
               "where option buyers have historically been paid — if the options aren't already priced for it "
               "(check on the Stock page).")


def render():
    ui.page_header("Events", "How stocks and the economy moved around elections, wars, Fed decisions, data "
                   "releases, earnings, CEO changes, launches, insider buying and any event you add.",
                   kicker="Research")
    ui.needs_data()
    t1, t2, t3 = st.tabs(["Event study", "Upcoming", "Edit the event list"])
    with t1:
        _study_tab()
    with t2:
        _upcoming_tab()
    with t3:
        _edit_tab()
