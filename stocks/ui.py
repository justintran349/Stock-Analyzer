"""Shared Streamlit helpers: cached loaders, chart styling, formatting."""
from __future__ import annotations

import re

import numpy as np
import pandas as pd
import plotly.graph_objects as go
import plotly.io as pio
import streamlit as st

from . import config, data, events as ev_mod, model

# ----------------------------------------------------------------- design tokens
# Colorblind-safe categorical order (fixed: series keep their color when filters change).
SERIES = ["#4C8DF6", "#F0883E", "#2EB67D", "#E3B341", "#D96BA6", "#8B7CF6", "#56B6C2", "#E5534B"]
UP, DOWN, NEUTRAL = "#4C8DF6", "#F0883E", "#8B93A1"   # up/down without relying on red/green
TEXT, MUTED, FAINT = "#E6E8EC", "#8B93A1", "#5C6573"
PANEL, BORDER = "#141920", "#232A35"
GRID, AXIS = "rgba(139,147,161,0.14)", "rgba(139,147,161,0.35)"

pio.templates["sa"] = go.layout.Template(layout=dict(
    colorway=SERIES,
    font=dict(family="Inter, -apple-system, system-ui, sans-serif", size=12, color="#C9CED6"),
    paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)",
    xaxis=dict(gridcolor=GRID, linecolor=AXIS, zeroline=False, showgrid=False, tickcolor=AXIS,
               title=dict(font=dict(size=11, color=MUTED))),
    yaxis=dict(gridcolor=GRID, linecolor="rgba(0,0,0,0)", zerolinecolor=AXIS, showgrid=True,
               title=dict(font=dict(size=11, color=MUTED))),
    hovermode="x unified",
    hoverlabel=dict(bgcolor="#1B212B", bordercolor=BORDER, font=dict(color=TEXT, size=12,
                                                                    family="Inter, system-ui, sans-serif")),
    margin=dict(l=8, r=8, t=36, b=8),
    legend=dict(orientation="h", yanchor="bottom", y=1.0, x=0, font=dict(size=11, color=MUTED),
                bgcolor="rgba(0,0,0,0)"),
))
pio.templates.default = "plotly_dark+sa"


def style(fig: go.Figure, height: int = 380, title: str | None = None, hover: str = "x unified") -> go.Figure:
    fig.update_layout(template="plotly_dark+sa", height=height, hovermode=hover)
    if title:
        fig.update_layout(title=dict(text=title, x=0, y=0.98, yanchor="top",
                                     font=dict(size=13, color=TEXT, family="Inter, system-ui, sans-serif")),
                          margin=dict(t=64), legend=dict(y=1.0, yanchor="bottom"))
    fig.update_traces(selector=dict(type="scatter", mode="lines"), line=dict(width=2))
    return fig


def show(fig, **kw):
    st.plotly_chart(fig, theme=None, config={"displaylogo": False,
                                             "modeBarButtonsToRemove": ["lasso2d", "select2d", "autoScale2d"]}, **kw)


# ----------------------------------------------------------------- layout components
CSS = """
<style>
:root { --sa-up:#4C8DF6; --sa-down:#F0883E; --sa-muted:#8B93A1; --sa-faint:#5C6573;
        --sa-panel:#141920; --sa-panel2:#11151C; --sa-border:#232A35; --sa-text:#E6E8EC; }
.block-container { padding-top: 1.6rem; padding-bottom: 3rem; max-width: 1480px; }
header[data-testid="stHeader"] { background: transparent; }
[data-testid="stToolbar"] [data-testid="stAppDeployButton"], .stAppDeployButton, footer { display: none !important; }
h1, h2, h3 { letter-spacing: -0.01em; }
/* numbers line up */
[data-testid="stMetricValue"], .sa-num, [data-testid="stDataFrame"] { font-variant-numeric: tabular-nums; }
/* bordered containers become panels */
[data-testid="stVerticalBlockBorderWrapper"] > div[data-testid="stVerticalBlock"],
div[data-testid="stVerticalBlockBorderWrapper"] { border-radius: 8px; }
/* metrics as tiles */
[data-testid="stMetric"] { background: var(--sa-panel); border: 1px solid var(--sa-border); border-radius: 8px;
                            padding: 12px 14px; }
[data-testid="stMetricLabel"] p { font-size: 0.72rem !important; text-transform: uppercase; letter-spacing: .06em;
                                  color: var(--sa-muted) !important; font-weight: 600; }
[data-testid="stMetricValue"] { font-size: 1.45rem !important; font-weight: 600; }
/* tabs */
.stTabs [data-baseweb="tab-list"] { gap: 4px; border-bottom: 1px solid var(--sa-border); }
.stTabs [data-baseweb="tab"] { padding: 6px 14px; font-weight: 500; }
/* sidebar */
section[data-testid="stSidebar"] [data-testid="stSidebarNavItems"] a span { font-weight: 500; }
section[data-testid="stSidebar"] [data-testid="stNavSectionHeader"] { text-transform: uppercase; letter-spacing: .08em;
                                                                    font-size: .68rem; color: var(--sa-faint); }
/* custom pieces */
.sa-head { display:flex; align-items:flex-end; justify-content:space-between; gap:16px;
           padding-bottom: 12px; margin-bottom: 18px; border-bottom: 1px solid var(--sa-border); }
.sa-head h1 { font-size: 1.55rem; margin: 0; padding: 0; font-weight: 700; }
.sa-kicker { font-size: .7rem; text-transform: uppercase; letter-spacing: .1em; color: var(--sa-up); font-weight: 600;
             margin-bottom: 4px; }
.sa-sub { color: var(--sa-muted); font-size: .88rem; margin-top: 6px; max-width: 900px; line-height: 1.45; }
.sa-meta { color: var(--sa-muted); font-size: .78rem; text-align: right; white-space: nowrap; }
.sa-meta b { color: var(--sa-text); font-weight: 600; }
.sa-section { margin: 26px 0 10px; }
.sa-section .t { font-size: 1.02rem; font-weight: 600; color: var(--sa-text); }
.sa-section .s { color: var(--sa-muted); font-size: .82rem; margin-top: 2px; }
.sa-pill { display:inline-block; padding: 2px 8px; border-radius: 999px; font-size: .68rem; font-weight: 600;
           font-family: Inter, system-ui, sans-serif; vertical-align: 2px; margin-left: 4px;
           letter-spacing: .02em; border: 1px solid transparent; }
.sa-pill.up   { color: #8DB6FA; background: rgba(76,141,246,.12); border-color: rgba(76,141,246,.35); }
.sa-pill.down { color: #F5AE77; background: rgba(240,136,62,.12); border-color: rgba(240,136,62,.35); }
.sa-pill.neutral { color: var(--sa-muted); background: rgba(139,147,161,.10); border-color: rgba(139,147,161,.25); }
.sa-pill.good { color: #6FD3A4; background: rgba(46,182,125,.12); border-color: rgba(46,182,125,.35); }
.sa-pill.warn { color: #EDC86A; background: rgba(227,179,65,.12); border-color: rgba(227,179,65,.35); }
.sa-card { padding: 2px 2px 0; }
.sa-card .row1 { display:flex; justify-content:space-between; align-items:flex-start; gap: 8px; }
.sa-card .tk { font-family: 'JetBrains Mono', ui-monospace, monospace; font-weight: 600; font-size: 1.02rem;
               color: var(--sa-text); }
.sa-card .nm { color: var(--sa-muted); font-size: .78rem; margin-top: 1px; white-space: nowrap; overflow: hidden;
               text-overflow: ellipsis; max-width: 260px; }
.sa-card .prob { text-align: right; }
.sa-card .prob .v { font-size: 1.35rem; font-weight: 700; line-height: 1.1; font-variant-numeric: tabular-nums; }
.sa-card .prob .v.up { color: var(--sa-up); } .sa-card .prob .v.down { color: var(--sa-down); }
.sa-card .prob .l { font-size: .68rem; color: var(--sa-muted); text-transform: uppercase; letter-spacing: .06em; }
.sa-bar { height: 4px; background: rgba(139,147,161,.15); border-radius: 2px; margin: 10px 0 10px; overflow: hidden; }
.sa-bar > div { height: 100%; border-radius: 2px; }
.sa-stats { display:grid; grid-template-columns: repeat(4, 1fr); gap: 6px; }
.sa-stats .k { font-size: .66rem; color: var(--sa-faint); text-transform: uppercase; letter-spacing: .06em; }
.sa-stats .x { font-size: .9rem; font-weight: 600; font-variant-numeric: tabular-nums; color: var(--sa-text); }
.sa-why { color: var(--sa-muted); font-size: .78rem; margin-top: 10px; line-height: 1.4; }
.sa-tiles { display:grid; grid-template-columns: repeat(auto-fit, minmax(170px, 1fr)); gap: 10px; margin: 4px 0 8px; }
.sa-tile { background: var(--sa-panel); border: 1px solid var(--sa-border); border-radius: 8px; padding: 12px 14px; }
.sa-tile .k { font-size: .68rem; color: var(--sa-muted); text-transform: uppercase; letter-spacing: .07em; font-weight: 600; }
.sa-tile .v { font-size: 1.35rem; font-weight: 650; margin-top: 4px; font-variant-numeric: tabular-nums; color: var(--sa-text); }
.sa-tile .v.up { color: var(--sa-up); } .sa-tile .v.down { color: var(--sa-down); }
.sa-tile .d { font-size: .76rem; color: var(--sa-muted); margin-top: 2px; }
.sa-note { border-left: 2px solid var(--sa-border); padding: 6px 12px; color: var(--sa-muted); font-size: .82rem;
           margin: 8px 0 14px; }
.sa-note.good { border-color: #2EB67D; } .sa-note.warn { border-color: #E3B341; }
.sa-brand { padding: 4px 0 12px; }
.sa-brand .n { font-weight: 700; font-size: 1.05rem; letter-spacing: -0.01em; }
.sa-brand .s { color: var(--sa-muted); font-size: .74rem; margin-top: 2px; }
.sa-dot { display:inline-block; width:7px; height:7px; border-radius:50%; margin-right:6px; vertical-align: 1px; }
</style>
"""


def inject_css():
    st.markdown(CSS, unsafe_allow_html=True)


def page_header(title: str, subtitle: str = "", kicker: str = "", meta: str = ""):
    k = f'<div class="sa-kicker">{kicker}</div>' if kicker else ""
    s = f'<div class="sa-sub">{subtitle}</div>' if subtitle else ""
    m = f'<div class="sa-meta">{meta}</div>' if meta else ""
    st.markdown(f'<div class="sa-head"><div>{k}<h1>{title}</h1>{s}</div>{m}</div>', unsafe_allow_html=True)


def section(title: str, subtitle: str = ""):
    s = f'<div class="s">{subtitle}</div>' if subtitle else ""
    st.markdown(f'<div class="sa-section"><div class="t">{title}</div>{s}</div>', unsafe_allow_html=True)


def pill(text: str, tone: str = "neutral") -> str:
    return f'<span class="sa-pill {tone}">{text}</span>'


def note(text: str, tone: str = ""):
    st.markdown(f'<div class="sa-note {tone}">{text}</div>', unsafe_allow_html=True)


def tiles(items: list[dict]):
    """items: dicts with k (label), v (value), d (detail, optional), tone ('up'/'down'/'')."""
    html = "".join(f'<div class="sa-tile"><div class="k">{i["k"]}</div>'
                   f'<div class="v {i.get("tone", "")}">{i["v"]}</div>'
                   + (f'<div class="d">{i["d"]}</div>' if i.get("d") else "") + "</div>" for i in items)
    st.markdown(f'<div class="sa-tiles">{html}</div>', unsafe_allow_html=True)


def pick_card_html(ticker: str, name: str, chance: float, word: str, side: str, stats: list[tuple],
                   why: str = "", tags: str = "") -> str:
    tone = "up" if side == "UP" else "down"
    color = UP if side == "UP" else DOWN
    width = max(0, min(100, (chance - 0.5) / 0.3 * 100))      # 50% → empty bar, 80%+ → full
    st_html = "".join(f'<div><div class="k">{k}</div><div class="x">{v}</div></div>' for k, v in stats)
    return (f'<div class="sa-card"><div class="row1"><div><div class="tk">{ticker} {tags}</div>'
            f'<div class="nm">{name}</div></div>'
            f'<div class="prob"><div class="v {tone}">{chance:.0%}</div><div class="l">{word}</div></div></div>'
            f'<div class="sa-bar"><div style="width:{width:.0f}%;background:{color}"></div></div>'
            f'<div class="sa-stats">{st_html}</div>'
            + (f'<div class="why sa-why">{why}</div>' if why else "") + "</div>")


# ----------------------------------------------------------------- cached loaders
@st.cache_data(ttl=600, show_spinner=False)
def latest(h: str) -> pd.DataFrame:
    return model.load_latest(h)


@st.cache_data(ttl=600, show_spinner=False)
def latest_rows(h: str) -> pd.DataFrame:
    return model.load_latest_rows(h)


@st.cache_data(ttl=600, show_spinner=False)
def metrics(h: str):
    return model.load_metrics(h)


@st.cache_resource(ttl=3600, show_spinner=False)
def bundle(h: str):
    return model.load_bundle(h)


@st.cache_data(ttl=600, show_spinner=False)
def backtest(h: str) -> pd.DataFrame:
    p = config.BACKTEST_DIR / f"{h}_test.parquet"
    return pd.read_parquet(p) if p.exists() else pd.DataFrame()


@st.cache_data(ttl=600, show_spinner="Loading prices ...")
def daily_prices(tickers: tuple | None = None, start: str | None = None) -> pd.DataFrame:
    return data.load_daily(tickers=list(tickers) if tickers else None, start=start)


@st.cache_data(ttl=600, show_spinner="Loading prices ...")
def daily_returns_wide(start: str | None = None) -> pd.DataFrame:
    px = data.load_daily(start=start)
    if px.empty:
        return pd.DataFrame()
    wide = px.pivot_table(index="date", columns="ticker", values="close").sort_index()
    return np.log(wide).diff()


@st.cache_data(ttl=600, show_spinner=False)
def context() -> pd.DataFrame:
    return data.load_yahoo_context()


@st.cache_data(ttl=600, show_spinner=False)
def context_returns() -> pd.DataFrame:
    """Daily log returns of the market/cross-asset series (levels → changes for VIX/yields)."""
    w = data.load_yahoo_context()
    if w.empty:
        return w
    out = {}
    for c in w.columns:
        out[c] = w[c].diff() / 100 if c in ("vix", "tnx", "irx") else np.log(w[c]).diff()
    return pd.DataFrame(out)


@st.cache_data(ttl=600, show_spinner=False)
def universe() -> pd.DataFrame:
    return data.load_universe()


@st.cache_data(ttl=60, show_spinner=False)
def events(include_earnings: bool = False) -> pd.DataFrame:
    return ev_mod.load_events(include_earnings=include_earnings)


@st.cache_data(ttl=3600, show_spinner=False)
def sectors() -> pd.DataFrame:
    from .sectors import load_sectors
    return load_sectors()


@st.cache_data(ttl=3600, show_spinner=False)
def insider_trades() -> pd.DataFrame:
    from .insiders import load_trades
    return load_trades()


@st.cache_data(ttl=600, show_spinner=False)
def options_summary() -> pd.DataFrame:
    from .options import load_summary
    return load_summary()


EVENT_TABLE_COLS = {
    "date": st.column_config.DateColumn("Date"), "type": "Type", "ticker": "Ticker", "title": "Event",
    "day0_move_%": st.column_config.NumberColumn("Day-0 move", format="%+.2f%%"),
    "abs_day0_vs_normal": st.column_config.NumberColumn("vs normal day", format="%.1f×"),
    "car_1d_%": st.column_config.NumberColumn("+1 day", format="%+.2f%%"),
    "car_5d_%": st.column_config.NumberColumn("+5 days", format="%+.2f%%"),
    "car_20d_%": st.column_config.NumberColumn("+20 days", format="%+.2f%%"),
    "vol_before_%": st.column_config.NumberColumn("Vol. before", format="%.0f%%"),
    "vol_after_%": st.column_config.NumberColumn("Vol. after", format="%.0f%%"),
}


def clear():
    st.cache_data.clear()
    st.cache_resource.clear()


# ----------------------------------------------------------------- formatting
def pct(v, digits=1, sign=True):
    if v is None or (isinstance(v, float) and np.isnan(v)):
        return "–"
    return f"{v:+.{digits}f}%" if sign else f"{v:.{digits}f}%"


def money(v):
    if v is None or pd.isna(v):
        return "–"
    for unit, div in (("B", 1e9), ("M", 1e6), ("K", 1e3)):
        if abs(v) >= div:
            return f"${v / div:.1f}{unit}"
    return f"${v:,.0f}"


_NAME_JUNK = re.compile(
    r"\s*[-,]?\s*(new\s+)?(class [a-z]\s+)?(common stock|ordinary shares|common shares|subordinate voting shares|"
    r"american depositary shares?.*|ads.*|depositary shares.*|shares of beneficial interest|units?)\b.*$", re.I)
# well-known names people type that aren't in the official company name
ALIASES = {"GOOGL": "Google", "GOOG": "Google", "META": "Facebook", "BRK-B": "Berkshire", "BRK-A": "Berkshire",
           "KO": "Coke", "JPM": "Chase", "TSM": "TSMC", "BABA": "Alibaba", "XOM": "Exxon", "WMT": "Walmart",
           "LLY": "Lilly", "UNH": "UnitedHealth", "BAC": "BofA", "MCD": "McDonalds", "HD": "Home Depot",
           "PG": "P&G", "JNJ": "J&J", "BA": "Boeing", "F": "Ford", "GM": "General Motors", "T": "AT&T"}


def clean_name(name: str) -> str:
    """'NVIDIA Corporation - Common Stock' → 'NVIDIA Corporation'."""
    if not isinstance(name, str):
        return ""
    n = _NAME_JUNK.sub("", name).strip(" ,-")
    n = re.sub(r"\s*\(The\)", "", n)
    return n or name


@st.cache_data(ttl=3600, show_spinner=False)
def name_map() -> dict:
    u = universe()
    if u.empty:
        return {}
    return {t: clean_name(n) for t, n in zip(u["ticker"], u["name"])}


def ticker_label(t: str) -> str:
    """Shown in every stock picker. Typing filters on the whole label, so either the
    ticker (NVDA) or any part of the company name (nvidia) finds the stock."""
    n = name_map().get(t, "")
    a = ALIASES.get(t)
    if a and a.lower() not in n.lower():
        n = f"{n} ({a})" if n else a
    return f"{t} — {n}" if n else t


def all_tickers() -> list[str]:
    """Every stock that has predictions, falling back to the whole universe."""
    seen = []
    for h in available_horizons():
        p = latest(h)
        if not p.empty:
            seen = sorted(set(seen) | set(p["ticker"]))
    return seen or sorted(universe()["ticker"].tolist())


def needs_data():
    if not config.DAILY_FILE.exists():
        st.info("No data yet. In the VS Code terminal run:\n\n"
                "```bash\npython scripts/download_data.py --all\npython scripts/train.py\n```")
        st.stop()


def available_horizons() -> list[str]:
    return [h for h in config.HORIZONS if (config.MODELS_DIR / f"{h}.joblib").exists()]
