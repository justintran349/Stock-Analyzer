"""Streamlit pages. app.py registers them in PAGES so pages can link to each other."""
PAGES: dict = {}


def open_stock(ticker: str):
    import streamlit as st

    st.session_state["ticker"] = ticker
    if "stock" in PAGES:
        st.switch_page(PAGES["stock"])
