import numpy as np
import pandas as pd
import pytest

from stocks import config, data, events, features as F, links, model
from tests import synth


@pytest.fixture(scope="module")
def inp():
    synth.make(None, n_tickers=25, start="2012-01-01", hourly_days=300)
    return model.load_inputs()


def test_tidy_download_layouts():
    idx = pd.date_range("2024-01-01", periods=3)
    cols = pd.MultiIndex.from_product([["Close", "High", "Low", "Open", "Volume"], ["AAA", "BBB"]])
    raw = pd.DataFrame(np.arange(30).reshape(3, 10) + 1.0, index=idx, columns=cols)
    out = data._tidy_download(raw, ["AAA", "BBB"])
    assert set(out["ticker"]) == {"AAA", "BBB"} and len(out) == 6
    raw2 = raw.swaplevel(axis=1)            # (ticker, field) layout
    out2 = data._tidy_download(raw2, ["AAA", "BBB"])
    assert len(out2) == 6 and out2["close"].notna().all()


def test_universe_cleaning():
    text = ("ACT Symbol|Security Name|Exchange|CQS Symbol|ETF|Round Lot Size|Test Issue|NASDAQ Symbol\n"
            "BRK.B|Berkshire Class B|N|BRK.B|N|100|N|BRK.B\n"
            "ABR$D|Arbor 6.375% Preferred|N|ABR-D|N|100|N|ABR-D\n"
            "XYZ.WS|XYZ Warrants|N|XYZ.WS|N|100|N|XYZ.WS\n"
            "SPY|SPDR S&P 500|P|SPY|Y|100|N|SPY\n"
            "GE|GE Aerospace|N|GE|N|100|N|GE\n"
            "File Creation Time: 1124202617:00|||||||\n")
    uni = data.clean_universe(data.parse_symbol_file(text, "Exchange"), ["N"], False)
    assert uni["ticker"].tolist() == ["BRK-B", "GE"]


def test_no_lookahead_in_fred():
    cal = pd.bdate_range("2020-01-01", "2020-06-30")
    raw = pd.DataFrame({"name": "testser", "series_id": "X", "obs_date": pd.date_range("2019-01-01", "2020-05-01", freq="MS"),
                        "value": np.arange(17.0) + 100, "lag_days": 45, "kind": "chg"})
    eco = F.fred_daily(raw, cal)
    # April's value (obs Apr 1) is public May 16 → on May 15 we still see March's
    assert eco.loc["2020-05-15", "eco_testser"] == raw.set_index("obs_date").loc["2020-03-01", "value"]
    assert eco.loc["2020-05-18", "eco_testser"] == raw.set_index("obs_date").loc["2020-04-01", "value"]


def test_event_features_until_and_since():
    cal = pd.DatetimeIndex(["2024-11-01", "2024-11-05", "2024-11-06"])
    since = F._days_since(cal, ["2024-11-05"])
    until = F._days_since(cal, ["2024-11-05"], forward=True)
    assert list(since) == [365, 0, 1] and list(until) == [4, 365, 365]


def test_features_and_targets(inp):
    st = F.stock_features(inp.prices, model.spy_close(inp))
    t = st[st["ticker"] == "T001"].set_index("date")
    # target = next day's log return
    lc = np.log(t["close"])
    assert np.allclose(t["y_ret_1d"].iloc[:-1].values, (lc.shift(-1) - lc).iloc[:-1].values, atol=1e-5)
    assert np.isnan(t["y_ret_1d"].iloc[-1])
    # momentum uses the past only
    assert np.allclose(t["mom_ret_5"].iloc[10], lc.iloc[10] - lc.iloc[5], atol=1e-5)


def test_train_backtest_predict(inp):
    res = model.train_horizon("1d", inp=inp, quick=True, log=lambda *a: None)
    assert res and res["test"]["rows"] > 500
    d = res["dates"]
    assert d["train_end"] < d["valid_start"] < d["test_start"]   # no overlap
    assert res["test"]["same_day_auc"] > 0.52   # finds the planted reversal pattern
    p = model.predict_latest("1d", inp=inp, log=lambda *a: None)
    assert len(p) == inp.prices["ticker"].nunique()
    assert p["p_up"].between(0, 1).all()
    rows = model.load_latest_rows("1d")
    ex = model.explain_row(model.load_bundle("1d"), rows.iloc[[0]])
    assert len(ex) > 3


def test_hourly(inp):
    res = model.train_horizon("1h", inp=inp, quick=True, log=lambda *a: None)
    assert res and res["test"]["rows"] > 100


def test_event_study_and_links(inp):
    rets = np.log(inp.prices.pivot_table(index="date", columns="ticker", values="close")).diff()
    spy = np.log(inp.ctx["spy"]).diff().reindex(rets.index)
    ev = pd.DataFrame({"date": pd.to_datetime(["2016-11-08", "2020-11-03"]), "type": "election",
                       "ticker": ["T001", "T002"], "title": "x", "notes": ""})
    r = events.event_study(ev, rets, spy)
    assert r["n"] == 2 and len(r["mean"]) == 26
    D = links.driver_table(inp.ctx, inp.fred, None)
    W = links.weekly_returns(inp.prices)
    res = links.scan(D, W, lags=(1,), driver_names=["oil"])
    top = res[res["held_up"]].head(5)["ticker"].tolist()
    assert sum(int(t[1:]) % 5 == 0 for t in top) >= 4   # planted oil-sensitive stocks


def test_events_roundtrip():
    df = events.load_events()
    assert (df["type"] == "election").sum() >= 10
    df.loc[len(df)] = [pd.Timestamp("2026-01-05"), "My Type", "ge", "test", ""]
    events.save_events(df)
    again = events.load_events()
    assert "my_type" in set(again["type"]) and "GE" in set(again["ticker"])


def test_new_features_present(inp):
    st = model.compute_stock(inp, log=None)
    for c in ("sec_rel_5", "sec_mom_21", "sec_peer_earn_react_5", "sec_peers_upcoming_10", "y_ex_1d"):
        assert c in st.columns, c
    assert st["sec_peers_reported_5"].max() >= 1
    rows = st[st["date"] == st["date"].max()]
    from stocks import insiders
    ins = insiders.insider_features(rows.reset_index(drop=True), inp.insiders)
    assert {"ins_buy_value_90", "ins_net_ratio_90", "ins_days_since_buy"} <= set(ins.columns)
    assert "fomc" in inp.mkt_types and "jobs_report" in inp.mkt_types


def test_insider_window_exact():
    from stocks import insiders
    t = pd.DataFrame({"ticker": ["A", "A"], "filing_date": pd.to_datetime(["2024-01-01", "2024-03-15"]),
                      "side": ["buy", "sell"], "value": [1e6, 3e6], "owner": ["x", "y"], "is_exec": [True, False]})
    rows = pd.DataFrame({"ticker": "A", "date": pd.to_datetime(["2024-01-01", "2024-01-02", "2024-03-31",
                                                                "2024-04-05"])})
    f = insiders.insider_features(rows, t)
    assert f["ins_buy_value_90"].iloc[0] == 0                      # filed that day: not public until next day
    assert abs(f["ins_buy_value_90"].iloc[1] - 6) < 1e-3           # $1M → log10 6
    assert f["ins_net_ratio_90"].iloc[2] == pytest.approx(-0.5)    # 1M buy, 3M sell
    assert f["ins_buy_value_90"].iloc[3] == 0                      # buy dropped out after 90 days


def test_fomc_parsers():
    from stocks import macro_calendar as mc
    cal = ("<h4>2026 FOMC Meetings</h4><div>January</div><div>27-28</div><div>March</div><div>17-18*</div>"
           "<h4>2024 FOMC Meetings</h4><div>Apr/May</div><div>30-1</div>")
    got = mc.parse_fomc_calendar_page(cal)
    assert pd.Timestamp("2026-01-28") in got and pd.Timestamp("2026-03-18") in got and pd.Timestamp("2024-05-01") in got
    hist = "<h5>January 24-25 Meeting - 2012</h5> <h5>July 31-August 1 Meeting - 2012</h5> <h5>March 13 Meeting - 2012</h5>"
    got = mc.parse_fomc_historical_page(hist, 2012)
    assert set(got) == {pd.Timestamp("2012-01-25"), pd.Timestamp("2012-08-01"), pd.Timestamp("2012-03-13")}
    jobs = mc.jobs_report_rule("2026-10-01", "2026-11-01")
    assert pd.Timestamp("2026-11-06") in jobs     # Oct 2026 report


def test_options_summary_and_ideas(inp):
    from stocks import macro_calendar, options
    calls = pd.DataFrame({"strike": [95.0, 100.0, 105.0], "bid": [6.0, 2.0, 0.5], "ask": [6.4, 2.2, 0.7],
                          "lastPrice": [6.2, 2.1, 0.6], "volume": [10, 50, 5], "openInterest": [100, 500, 50],
                          "impliedVolatility": [0.35, 0.30, 0.28]})
    puts = calls.assign(bid=[0.5, 1.8, 5.5], ask=[0.7, 2.0, 5.9], impliedVolatility=[0.40, 0.31, 0.27])
    s = options.summarize_chain(calls, puts, 100.0)
    assert s["strike"] == 100 and abs(s["implied_move_pct"] - 4.0) < 1e-6 and s["skew"] > 0
    # thin expiry: nearest two-sided strike far from the price → falls back to implied volatility
    far = summarize = options.summarize_chain(calls[calls.strike == 105], puts[puts.strike == 105], 90.0, 30)
    assert far["method"] == "iv" and abs(far["implied_move_pct"] - 100 * 0.798 * 0.275 * (30 / 365) ** 0.5) < 0.2
    # interpolation between strikes around the price
    mid = options.summarize_chain(calls, puts, 102.5, None)
    assert 4.0 < mid["implied_move_pct"] * 102.5 / 100 < 6.6
    model.train_horizon("1w", inp=inp, quick=True, log=lambda *a: None)
    preds = {"1w": model.predict_latest("1w", inp=inp, log=lambda *a: None)}
    assert "p_beat" in preds["1w"]
    rets = np.log(inp.prices.pivot_table(index="date", columns="ticker", values="close")).diff()
    ideas = options.build_ideas(preds, rets, inp.earnings, macro_calendar.load_calendar())
    assert len(ideas) == 15 and ideas["idea"].str.len().gt(0).all()
    assert abs(ideas["rel_ratio"].median() - 1) < 0.2      # verdicts are relative to the market
    assert (ideas["event"] != "").sum() >= 5
    assert options.track(ideas.head(3)) == 3
    g = options.grade_ideas()
    assert len(g) == 3


def test_beat_model_metrics(inp):
    m = model.load_metrics("1d")
    assert "beat" in m["test"] and m["test"]["beat"]["auc"] > 0.5


def test_sec_earnings_parsing(monkeypatch):
    sub = {"filings": {"recent": {
        "form": ["8-K", "8-K", "10-Q", "8-K", "8-K/A"],
        "items": ["2.02,9.01", "5.02", "", "2.02", "2.02"],
        "acceptanceDateTime": ["2026-10-29T16:05:00.000Z", "2026-09-01T10:00:00.000Z", "2026-10-30T12:00:00.000Z",
                               "2026-07-30T07:00:00.000Z", "2026-08-02T09:00:00.000Z"],
        "filingDate": ["2026-10-29", "2026-09-01", "2026-10-30", "2026-07-30", "2026-08-02"]}, "files": []}}
    monkeypatch.setattr(data, "_sec_get", lambda url, log, tries=3: sub)
    d = data.sec_earnings_dates(123)
    got = sorted(d["date"].dt.strftime("%Y-%m-%d"))
    assert got == ["2026-07-30", "2026-10-30"]      # after-close → next day; amendment dropped
