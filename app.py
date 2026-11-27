"""Stock Analyzer — streamlit run app.py"""
import pandas as pd
import streamlit as st

st.set_page_config(page_title="Stock Analyzer", page_icon=":material/candlestick_chart:", layout="wide",
                   initial_sidebar_state="expanded")

from app_pages import (PAGES, backtest, charts, data_page, events_page, links_page, options_page,  # noqa: E402
                       stock, top_picks, tracker_page)
from stocks import config, ui  # noqa: E402

ui.inject_css()

PAGES["top"] = st.Page(top_picks.render, title="Top picks", icon=":material/insights:", url_path="top", default=True)
PAGES["stock"] = st.Page(stock.render, title="Stock analyzer", icon=":material/query_stats:", url_path="stock")
PAGES["options"] = st.Page(options_page.render, title="Options ideas", icon=":material/candlestick_chart:",
                           url_path="options")
PAGES["events"] = st.Page(events_page.render, title="Events", icon=":material/event:", url_path="events")
PAGES["links"] = st.Page(links_page.render, title="Hidden links", icon=":material/hub:", url_path="links")
PAGES["charts"] = st.Page(charts.render, title="Chart builder", icon=":material/bar_chart:", url_path="charts")
PAGES["backtest"] = st.Page(backtest.render, title="Backtest", icon=":material/science:", url_path="backtest")
PAGES["tracker"] = st.Page(tracker_page.render, title="Tracker", icon=":material/fact_check:", url_path="tracker")
PAGES["data"] = st.Page(data_page.render, title="Data & models", icon=":material/database:", url_path="data")

with st.sidebar:
    last = ""
    try:
        p = ui.latest("1d")
        if not p.empty:
            last = f"{pd.Timestamp(p['date'].max()):%b %d, %Y}"
    except Exception:
        pass
    n_models = len(ui.available_horizons())
    dot = ui.UP if n_models else ui.DOWN
    st.markdown(
        f'<div class="sa-brand"><div class="n">Stock Analyzer</div>'
        f'<div class="s"><span class="sa-dot" style="background:{dot}"></span>'
        f'{n_models}/{len(config.HORIZONS)} models · data {last or "not loaded"}</div></div>',
        unsafe_allow_html=True)

nav = st.navigation({
    "Predict": [PAGES["top"], PAGES["stock"], PAGES["options"]],
    "Research": [PAGES["events"], PAGES["links"], PAGES["charts"]],
    "Validate": [PAGES["backtest"], PAGES["tracker"], PAGES["data"]],
})
st.sidebar.caption("Research tool, not financial advice. Past patterns can stop working.")
nav.run()
