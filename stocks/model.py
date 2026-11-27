"""Training, the 2-year backtest, and live predictions.

For each horizon (next hour / day / week / month) three gradient-boosted models are fit:
  - direction: chance the stock is up at the end of the horizon (calibrated)
  - size:      expected return (%)
  - move:      expected absolute move (%), i.e. how much it will swing either way —
               the number that matters for options (straddles, strangles)

How the 2-year test works (scripts/train.py):
  1. TRAIN on everything up to 3 years ago.
  2. TUNE: try several model settings and feature sets; keep whichever does best on the
     year between 3 and 2 years ago (the validation year).
  3. TEST: the chosen model predicts every day of the last 2 years. It has never seen
     these days and nothing about them influenced its settings, so the score is honest.
     (If we tuned on the test years, the score would look great and mean nothing.)
  4. LIVE: refit on everything (including the last 2 years) with the chosen settings for
     today's predictions. Probabilities are calibrated so a 60% means ~60%.
Optional --walk-forward re-trains every quarter through the test window, the way you'd
really use it.
"""
from __future__ import annotations

import json
import time
import warnings
from dataclasses import dataclass, field
from typing import Callable

import joblib
import numpy as np
import pandas as pd
from sklearn.ensemble import HistGradientBoostingClassifier, HistGradientBoostingRegressor
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import brier_score_loss, log_loss, roc_auc_score

from . import config, data, events as ev_mod, features as F

warnings.filterwarnings("ignore", message="Unknown solver options")

PARAM_GRID = [
    dict(learning_rate=0.05, max_leaf_nodes=31, min_samples_leaf=400, l2_regularization=1.0, max_iter=400),
    dict(learning_rate=0.03, max_leaf_nodes=15, min_samples_leaf=1500, l2_regularization=5.0, max_iter=500),
    dict(learning_rate=0.08, max_leaf_nodes=63, min_samples_leaf=200, l2_regularization=0.5, max_iter=300),
]
FEATURE_SETS = {
    "all": None,                                             # every group
    "no_economy": {"eco", "cus"},                            # drop FRED + custom
    "stock_and_events": {"eco", "cus", "mkt"},               # drop all macro/cross-asset
}
CONF_BINS = [0.5, 0.52, 0.54, 0.56, 0.58, 0.60, 0.65, 0.70, 1.0001]


# ======================================================================= calibration
class PlattCalibrator:
    """Maps raw model probabilities to honest ones (a 60% happens ~60% of the time)."""

    def __init__(self):
        self.lr = LogisticRegression(C=1e6)

    @staticmethod
    def _logit(p):
        p = np.clip(np.asarray(p, dtype=float), 1e-4, 1 - 1e-4)
        return np.log(p / (1 - p)).reshape(-1, 1)

    def fit(self, p, y):
        self.lr.fit(self._logit(p), np.asarray(y).astype(int))
        return self

    def transform(self, p):
        return self.lr.predict_proba(self._logit(p))[:, 1]


# ======================================================================= inputs
@dataclass
class Inputs:
    prices: pd.DataFrame
    ctx: pd.DataFrame
    fred: pd.DataFrame
    custom: pd.DataFrame
    events: pd.DataFrame
    earnings: pd.DataFrame
    mkt_types: list = field(default_factory=list)
    tk_types: list = field(default_factory=list)
    sectors: pd.DataFrame = field(default_factory=pd.DataFrame)
    insiders: pd.DataFrame = field(default_factory=pd.DataFrame)


def load_inputs(start: str | None = None, tickers: list[str] | None = None) -> Inputs:
    from .insiders import load_trades
    from .sectors import load_sectors

    prices = data.load_daily(tickers=tickers, start=start)
    ctx = data.load_yahoo_context()
    events = ev_mod.load_events()
    mkt_types, tk_types = F.event_types_for_model(events)
    trades = load_trades()
    if start is not None and not trades.empty:
        trades = trades[trades["filing_date"] >= pd.Timestamp(start) - pd.Timedelta(days=120)]
    return Inputs(prices=prices, ctx=ctx, fred=data.load_fred_raw(), custom=data.load_custom_series(),
                  events=events, earnings=data.load_earnings(), mkt_types=mkt_types, tk_types=tk_types,
                  sectors=load_sectors(), insiders=trades)


def compute_stock(inp: Inputs, horizons: dict | None = None, prices: pd.DataFrame | None = None,
                  log: Callable | None = print) -> pd.DataFrame:
    """Per-stock daily features (+ sector/peer features, which need every stock at once)."""
    from .sectors import add_sector_features

    px = inp.prices if prices is None else prices
    stock = F.stock_features(px, spy_close(inp), horizons=horizons, log=log)
    if not inp.sectors.empty:
        stock = add_sector_features(stock, inp.sectors, inp.earnings, log=log)
    return stock


def spy_close(inp: Inputs) -> pd.Series | None:
    if not inp.ctx.empty and "spy" in inp.ctx:
        return inp.ctx["spy"].dropna()
    p = inp.prices[inp.prices["ticker"] == "SPY"]
    return p.set_index("date")["close"] if not p.empty else None


def build_date_features(inp: Inputs, calendar: pd.DatetimeIndex) -> pd.DataFrame:
    return F.date_features(calendar, inp.ctx, inp.fred, inp.custom, inp.events, inp.mkt_types)


# ======================================================================= datasets
@dataclass
class Split:
    train: pd.DataFrame
    valid: pd.DataFrame
    test: pd.DataFrame
    features: list
    dates: dict


def _sample_dates(dates: pd.DatetimeIndex, every: int) -> pd.DatetimeIndex:
    return dates[::every] if every > 1 else dates


def daily_split(inp: Inputs, horizon: str, stock: pd.DataFrame | None = None,
                log: Callable = print) -> Split:
    n = config.HORIZONS[horizon][1]
    if stock is None:
        stock = compute_stock(inp, log=log)
    y = f"y_ret_{horizon}"
    ok = stock[y].notna() & stock["mom_ret_63"].notna() & (stock["close"] >= 1) & \
        (stock["dollar_vol"] >= 1_000_000)
    lab = stock.loc[ok]
    cal = pd.DatetimeIndex(np.sort(lab["date"].unique()))
    end = cal[-1]
    test_start = end - pd.DateOffset(days=int(365.25 * config.TEST_YEARS))
    valid_start = test_start - pd.DateOffset(days=int(365.25 * config.VALID_YEARS))
    # purge: drop the last n days before each boundary so targets don't overlap the next set
    i_test = cal.searchsorted(test_start)
    i_val = cal.searchsorted(valid_start)
    train_end = cal[max(0, i_val - n - 1)]
    valid_end = cal[max(0, i_test - n - 1)]
    every = config.TRAIN_SAMPLE_EVERY.get(horizon, 1)
    tr_dates = _sample_dates(cal[cal <= train_end], every)
    va_dates = _sample_dates(cal[(cal >= cal[i_val]) & (cal <= valid_end)], max(1, every // 2))
    te_dates = _sample_dates(cal[cal >= cal[i_test]], n)        # one rebalance per horizon
    tr = lab[lab["date"].isin(tr_dates)]
    if len(tr) > config.MAX_TRAIN_ROWS:
        tr = tr.sample(config.MAX_TRAIN_ROWS, random_state=config.RANDOM_STATE)
    va = lab[lab["date"].isin(va_dates)]
    te = lab[lab["date"].isin(te_dates)]
    dfeat = build_date_features(inp, pd.DatetimeIndex(np.sort(stock["date"].unique())))
    parts = []
    for part in (tr, va, te):
        parts.append(F.assemble(part.sort_values(["ticker", "date"]), dfeat, inp.events, inp.earnings,
                                inp.tk_types, inp.insiders))
    feats = F.feature_columns(parts[0])
    log(f"  {horizon}: train {len(parts[0]):,} rows (to {train_end:%Y-%m-%d}) · "
        f"valid {len(parts[1]):,} ({cal[i_val]:%Y-%m-%d}–{valid_end:%Y-%m-%d}) · "
        f"test {len(parts[2]):,} ({cal[i_test]:%Y-%m-%d}–{end:%Y-%m-%d}) · {len(feats)} features")
    return Split(parts[0], parts[1], parts[2], feats,
                 {"train_end": str(train_end.date()), "valid_start": str(cal[i_val].date()),
                  "valid_end": str(valid_end.date()), "test_start": str(cal[i_test].date()),
                  "test_end": str(end.date())})


def hourly_frame(inp: Inputs, log: Callable = print) -> pd.DataFrame:
    hourly = data.load_hourly()
    if hourly.empty:
        return pd.DataFrame()
    start = hourly["date"].min() - pd.Timedelta(days=420)
    px = inp.prices[inp.prices["date"] >= start]
    px = px[px["ticker"].isin(hourly["ticker"].unique())]
    stock = compute_stock(inp, horizons={}, prices=px, log=log)
    cal = pd.DatetimeIndex(np.sort(stock["date"].unique()))
    dfeat = build_date_features(inp, cal)
    dfeat = dfeat[[c for c in ("mkt_vix", "mkt_spy_ret_21", "evt_until_fomc", "evt_until_cpi_release",
                               "evt_until_jobs_report") if c in dfeat.columns]]
    daily = F.assemble(stock, dfeat, inp.events, inp.earnings, [], inp.insiders)
    return F.hourly_features(hourly, daily, data.load_hourly_market())


def hourly_split(frame: pd.DataFrame, log: Callable = print) -> Split:
    ok = frame["y_ret_1h"].notna() & frame["intra_ret_35"].notna()
    if "dollar_vol" in frame:
        ok &= frame["dollar_vol"].fillna(0) >= 1_000_000
    lab = frame.loc[ok]
    end = lab["date"].max()
    test_start = end - pd.DateOffset(months=config.HOURLY_TEST_MONTHS)
    valid_start = test_start - pd.DateOffset(months=config.HOURLY_VALID_MONTHS)
    tr = lab[lab["date"] < valid_start - pd.Timedelta(days=1)]
    if len(tr) > config.MAX_TRAIN_ROWS:
        tr = tr.sample(config.MAX_TRAIN_ROWS, random_state=config.RANDOM_STATE)
    va = lab[(lab["date"] >= valid_start) & (lab["date"] < test_start - pd.Timedelta(days=1))]
    if len(va) > 600_000:
        va = va.sample(600_000, random_state=config.RANDOM_STATE)
    te = lab[lab["date"] >= test_start]
    feats = F.feature_columns(frame)
    log(f"  1h: train {len(tr):,} · valid {len(va):,} · test {len(te):,} rows "
        f"(test {test_start:%Y-%m-%d}–{end:%Y-%m-%d}) · {len(feats)} features")
    return Split(tr, va, te, feats, {"train_end": str((valid_start - pd.Timedelta(days=1)).date()),
                                     "valid_start": str(valid_start.date()),
                                     "test_start": str(test_start.date()), "test_end": str(end.date())})


# ======================================================================= fitting
def _clf(params):
    return HistGradientBoostingClassifier(**params, early_stopping=True, validation_fraction=0.1,
                                          n_iter_no_change=25, random_state=config.RANDOM_STATE)


def _reg(params, loss="squared_error"):
    p = dict(params)
    return HistGradientBoostingRegressor(**p, loss=loss, early_stopping=True, validation_fraction=0.1,
                                         n_iter_no_change=25, random_state=config.RANDOM_STATE)


def _feature_subset(features: list, set_name: str) -> list:
    drop = FEATURE_SETS[set_name]
    return features if not drop else [f for f in features if F.group_of(f) not in drop]


def daily_auc(df: pd.DataFrame, p: np.ndarray, y: np.ndarray) -> float:
    """Average AUC within each date: can the model rank stocks against each other on the
    same day, regardless of whether the whole market went up or down?"""
    tmp = pd.DataFrame({"d": df["date"].values, "p": p, "y": y})
    aucs = []
    for _, g in tmp.groupby("d"):
        if g["y"].nunique() == 2 and len(g) >= 20:
            aucs.append(roc_auc_score(g["y"], g["p"]))
    return float(np.mean(aucs)) if aucs else float("nan")


def _winsor(y: pd.Series, lo: float, hi: float) -> pd.Series:
    return y.clip(lo, hi)


def tune(split: Split, horizon: str, quick: bool = False, log: Callable = print) -> dict:
    ycol = f"y_ret_{horizon}"
    ytr = (split.train[ycol] > 0).astype(int)
    yva = (split.valid[ycol] > 0).astype(int)
    results = []

    def run(params, fset):
        cols = _feature_subset(split.features, fset)
        t0 = time.time()
        m = _clf(params).fit(split.train[cols], ytr)
        p = m.predict_proba(split.valid[cols])[:, 1]
        res = {"params": params, "feature_set": fset, "n_features": len(cols),
               "valid_auc": float(roc_auc_score(yva, p)) if yva.nunique() == 2 else float("nan"),
               "valid_daily_auc": daily_auc(split.valid, p, yva.values),
               "valid_logloss": float(log_loss(yva, np.clip(p, 1e-4, 1 - 1e-4), labels=[0, 1])),
               "iters": int(m.n_iter_), "secs": round(time.time() - t0, 1)}
        results.append(res)
        log(f"    {fset:<17} lr={params['learning_rate']} leaves={params['max_leaf_nodes']} "
            f"→ valid AUC {res['valid_auc']:.4f}, same-day AUC {res['valid_daily_auc']:.4f} "
            f"({res['secs']}s)")
        return res

    grid = PARAM_GRID[:1] if quick else PARAM_GRID
    for params in grid:
        run(params, "all")
    score = lambda r: (np.nan_to_num(r["valid_daily_auc"], nan=0.5) + np.nan_to_num(r["valid_auc"], nan=0.5)) / 2
    best = max(results, key=score)
    if not quick:
        for fset in FEATURE_SETS:
            if fset != "all":
                run(best["params"], fset)
        best = max(results, key=score)
    log(f"  chosen: {best['feature_set']} with {best['params']}")
    return {"best": best, "trials": results}


def fit_models(train: pd.DataFrame, cols: list, ycol: str, params: dict) -> dict:
    """direction (clf), expected return, expected move, and beat-the-market (clf_beat)."""
    y = train[ycol]
    lo, hi = y.quantile([0.005, 0.995])
    m = {"features": cols,
         "clf": _clf(params).fit(train[cols], (y > 0).astype(int)),
         "reg_ret": _reg(params).fit(train[cols], _winsor(y, lo, hi)),
         "reg_move": _reg(params).fit(train[cols], _winsor(y.abs(), 0, max(abs(lo), abs(hi))))}
    xcol = ycol.replace("y_ret_", "y_ex_")
    if xcol in train and train[xcol].notna().sum() > 1000:
        ok = train[xcol].notna()
        m["clf_beat"] = _clf(params).fit(train.loc[ok, cols], (train.loc[ok, xcol] > 0).astype(int))
    return m


def calibrate_models(m: dict, df: pd.DataFrame, ycol: str) -> dict:
    """Fit the probability calibrators on rows the models weren't trained on."""
    cols = m["features"]
    m["cal"] = PlattCalibrator().fit(m["clf"].predict_proba(df[cols])[:, 1], df[ycol] > 0)
    xcol = ycol.replace("y_ret_", "y_ex_")
    if "clf_beat" in m and xcol in df:
        ok = df[xcol].notna()
        m["cal_beat"] = PlattCalibrator().fit(m["clf_beat"].predict_proba(df.loc[ok, cols])[:, 1],
                                              df.loc[ok, xcol] > 0)
    return m


def predict_frame(models: dict, df: pd.DataFrame) -> pd.DataFrame:
    cols = models["features"]
    raw = models["clf"].predict_proba(df[cols])[:, 1]
    out = pd.DataFrame(index=df.index)
    out["p_raw"] = raw
    out["p_up"] = models["cal"].transform(raw) if models.get("cal") is not None else raw
    out["exp_ret"] = 100 * (np.exp(models["reg_ret"].predict(df[cols])) - 1)
    out["exp_move"] = 100 * (np.exp(np.maximum(models["reg_move"].predict(df[cols]), 0)) - 1)
    if "clf_beat" in models:
        rb = models["clf_beat"].predict_proba(df[cols])[:, 1]
        out["p_beat_raw"] = rb
        out["p_beat"] = models["cal_beat"].transform(rb) if models.get("cal_beat") is not None else rb
    return out


# ======================================================================= evaluation
def market_forward(inp: Inputs | None, horizon: str, dates: pd.Series) -> pd.Series:
    n, kind = config.HORIZONS[horizon][1], config.HORIZONS[horizon][2]
    if kind == "hourly":
        mh = data.load_hourly_market()
        if mh.empty or "SPY" not in mh:
            return pd.Series(np.nan, index=dates.index)
        s = np.log(mh["SPY"].dropna())
    else:
        sc = spy_close(inp) if inp is not None else None
        if sc is None:
            return pd.Series(np.nan, index=dates.index)
        s = np.log(sc)
    fwd = s.shift(-n) - s
    return dates.map(fwd)


def conf_table(df: pd.DataFrame) -> pd.DataFrame:
    """How often picks at each confidence level were right in the test."""
    d = df.copy()
    d["conf"] = np.maximum(d["p_up"], 1 - d["p_up"])
    d["side_up"] = d["p_up"] >= 0.5
    d["right"] = np.where(d["side_up"], d["y_ret"] > 0, d["y_ret"] < 0)
    d["side_ret"] = np.where(d["side_up"], 1, -1) * (np.exp(d["y_ret"]) - 1) * 100
    d["bin"] = pd.cut(d["conf"], CONF_BINS, right=False)
    t = d.groupby("bin", observed=True).agg(picks=("right", "size"), hit_rate=("right", "mean"),
                                            avg_return_pct=("side_ret", "mean"),
                                            avg_conf=("conf", "mean")).reset_index()
    t["low"] = [b.left for b in t["bin"]]
    t["high"] = [b.right for b in t["bin"]]
    t["bin"] = t["bin"].astype(str)
    return t


def calibration_curve(df: pd.DataFrame, bins: int = 10) -> pd.DataFrame:
    d = df[["p_up", "y_ret"]].copy()
    d["b"] = pd.qcut(d["p_up"], bins, duplicates="drop")
    return d.groupby("b", observed=True).agg(predicted=("p_up", "mean"),
                                             actual=("y_ret", lambda s: (s > 0).mean()),
                                             n=("p_up", "size")).reset_index(drop=True)


def strategy(df: pd.DataFrame, top_n: int = config.BACKTEST_TOP_N, cost_bps: float = config.COST_BPS,
             min_price: float = config.MIN_PRICE, min_dollar_vol: float = config.MIN_DOLLAR_VOLUME,
             rank_by: str = "p_up", min_conf: float = 0.0, periods_per_year: float = 252) -> dict:
    """Each test date: buy the top-N most-likely-up, short the top-N most-likely-down
    (equal weight), hold for the horizon, pay costs. Compared with just holding SPY."""
    d = df[(df["close"] >= min_price) & (df["dollar_vol"] >= min_dollar_vol)].copy()
    d["r"] = np.exp(d["y_ret"]) - 1
    rows = []
    cost = 2 * cost_bps / 1e4
    for date, g in d.groupby("date"):
        g = g.sort_values(rank_by, ascending=False)
        longs = g.head(top_n)
        shorts = g.tail(top_n).iloc[::-1]
        if min_conf > 0:
            pc = rank_by if rank_by in ("p_up", "p_beat") else "p_up"
            longs = longs[longs[pc] >= min_conf]
            shorts = shorts[shorts[pc] <= 1 - min_conf]
        lr = longs["r"].mean() - cost if len(longs) else np.nan
        sr = -shorts["r"].mean() - cost if len(shorts) else np.nan
        mkt = np.exp(g["mkt_fwd"].iloc[0]) - 1 if "mkt_fwd" in g and pd.notna(g["mkt_fwd"].iloc[0]) else np.nan
        rows.append({"date": date, "long": lr, "short": sr,
                     "long_short": np.nanmean([lr, sr]) if not (np.isnan(lr) and np.isnan(sr)) else np.nan,
                     "market": mkt, "all_stocks": g["r"].mean(),
                     "long_hit": (longs["r"] > 0).mean() if len(longs) else np.nan,
                     "short_hit": (shorts["r"] < 0).mean() if len(shorts) else np.nan,
                     "n_long": len(longs), "n_short": len(shorts)})
    ts = pd.DataFrame(rows).set_index("date") if rows else pd.DataFrame()
    stats = {}
    for col in ("long", "short", "long_short", "market", "all_stocks"):
        if ts.empty or col not in ts:
            continue
        r = ts[col].fillna(0)
        eq = (1 + r).cumprod()
        years = len(r) / periods_per_year
        stats[col] = {
            "total_return_%": 100 * (eq.iloc[-1] - 1),
            "annual_return_%": 100 * (eq.iloc[-1] ** (1 / years) - 1) if years > 0 and eq.iloc[-1] > 0 else np.nan,
            "sharpe": float(r.mean() / r.std() * np.sqrt(periods_per_year)) if r.std() > 0 else np.nan,
            "win_rate": float((ts[col].dropna() > 0).mean()),
            "max_drawdown_%": 100 * float((eq / eq.cummax() - 1).min()),
        }
    if not ts.empty:
        stats["long"]["pick_hit_rate"] = float(ts["long_hit"].mean())
        stats["short"]["pick_hit_rate"] = float(ts["short_hit"].mean())
    return {"series": ts, "stats": stats}


def evaluate(df: pd.DataFrame, horizon: str) -> dict:
    y = (df["y_ret"] > 0).astype(int)
    p = df["p_up"].values
    n = config.HORIZONS[horizon][1]
    ppy = 252 * 7 if config.HORIZONS[horizon][2] == "hourly" else 252 / n
    ics = [g["exp_ret"].corr(g["y_ret"], method="spearman") for _, g in df.groupby("date") if len(g) > 20]
    ic = np.nanmean(ics) if ics else np.nan
    mv = df["exp_move"].corr(100 * (np.exp(df["y_ret"].abs()) - 1), method="spearman")
    strat = strategy(df, periods_per_year=ppy)
    beat = {}
    if "p_beat" in df and "y_ex" in df and df["y_ex"].notna().any():
        d2 = df[df["y_ex"].notna()]
        yb = (d2["y_ex"] > 0).astype(int)
        pb = d2["p_beat"].values
        tmp = d2.assign(p_up=d2["p_beat"], y_ret=d2["y_ex"])
        beat = {"accuracy": float(((pb >= 0.5) == yb).mean()),
                "accuracy_coin": float(max(yb.mean(), 1 - yb.mean())),
                "auc": float(roc_auc_score(yb, pb)) if yb.nunique() == 2 else float("nan"),
                "brier": float(brier_score_loss(yb, pb)),
                "conf_table": conf_table(tmp).to_dict("records"),
                "calibration": calibration_curve(tmp).to_dict("records"),
                "strategy": strategy(df, periods_per_year=ppy, rank_by="p_beat")["stats"]}
    out = {
        "rows": int(len(df)), "dates": int(df["date"].nunique()),
        "accuracy": float(((p >= 0.5) == y).mean()),
        "base_rate_up": float(y.mean()),
        "accuracy_always_up": float(max(y.mean(), 1 - y.mean())),
        "auc": float(roc_auc_score(y, p)) if y.nunique() == 2 else float("nan"),
        "same_day_auc": daily_auc(df, p, y.values),
        "brier": float(brier_score_loss(y, p)),
        "brier_baseline": float(brier_score_loss(y, np.full(len(y), y.mean()))),
        "return_rank_ic": float(ic),
        "move_rank_corr": float(mv),
        "strategy": strat["stats"],
        "conf_table": conf_table(df).to_dict("records"),
        "calibration": calibration_curve(df).to_dict("records"),
    }
    if beat:
        out["beat"] = beat
    return out


# ======================================================================= train one horizon
def _keep_cols(df: pd.DataFrame) -> list:
    keep = ["date", "ticker", "close", "dollar_vol", "mom_ret_21", "risk_vol_21", "trend_rsi_14",
            "evt_tk_until_earnings"]
    return [c for c in keep if c in df.columns]


def train_horizon(horizon: str, inp: Inputs | None = None, stock: pd.DataFrame | None = None,
                  hourly: pd.DataFrame | None = None, quick: bool = False,
                  walk_forward: bool = False, log: Callable = print) -> dict:
    t0 = time.time()
    log(f"\n=== {config.horizon_label(horizon)} ({horizon}) ===")
    ycol = f"y_ret_{horizon}"
    if config.HORIZONS[horizon][2] == "hourly":
        if hourly is None:
            hourly = hourly_frame(inp, log=log)
        if hourly.empty:
            log("  no hourly data yet; run scripts/download_data.py --hourly")
            return {}
        split = hourly_split(hourly, log=log)
    else:
        split = daily_split(inp, horizon, stock=stock, log=log)
    if len(split.train) < 1000 or len(split.valid) < 200 or len(split.test) < 200:
        log("  not enough data to train this horizon")
        return {}

    # 1-2. tune on the validation year
    tuning = tune(split, horizon, quick=quick, log=log)
    params, fset = tuning["best"]["params"], tuning["best"]["feature_set"]
    cols = _feature_subset(split.features, fset)

    # 3. fit on train, calibrate on validation, grade on the untouched test window
    xcol = f"y_ex_{horizon}"
    models = calibrate_models(fit_models(split.train, cols, ycol, params), split.valid, ycol)
    test = split.test[_keep_cols(split.test)].copy()
    test = test.join(predict_frame(models, split.test))
    test["y_ret"] = split.test[ycol].values
    if xcol in split.test:
        test["y_ex"] = split.test[xcol].values
    if walk_forward:
        test = _walk_forward(split, cols, ycol, params, horizon, test, log)
    test["mkt_fwd"] = market_forward(inp, horizon, test["date"]).values
    metrics = evaluate(test, horizon)
    s = metrics["strategy"]
    log(f"  TEST ({split.dates['test_start']} → {split.dates['test_end']}): accuracy "
        f"{metrics['accuracy']:.1%} (always-up {metrics['accuracy_always_up']:.1%}), "
        f"AUC {metrics['auc']:.3f}, same-day AUC {metrics['same_day_auc']:.3f}")
    if "beat" in metrics:
        b = metrics["beat"]
        log(f"  beat-the-S&P model: right {b['accuracy']:.1%} of the time, AUC {b['auc']:.3f}")
    if "long_short" in s:
        log(f"  top-{config.BACKTEST_TOP_N} long/short: {s['long_short']['total_return_%']:+.1f}% "
            f"(Sharpe {s['long_short']['sharpe']:.2f}) · longs {s['long']['total_return_%']:+.1f}% · "
            f"SPY {s.get('market', {}).get('total_return_%', float('nan')):+.1f}%")
    val_pred = predict_frame(models, split.valid)

    # 4. refit on everything for live use; calibrate on validation + test predictions
    everything = pd.concat([split.train, split.valid, split.test], ignore_index=True)
    if len(everything) > config.MAX_TRAIN_ROWS * 1.3:
        everything = everything.sample(int(config.MAX_TRAIN_ROWS * 1.3), random_state=config.RANDOM_STATE)
    bundle = fit_models(everything, cols, ycol, params)
    bundle["cal"] = PlattCalibrator().fit(np.concatenate([val_pred["p_raw"].values, test["p_raw"].values]),
                                          np.concatenate([split.valid[ycol].values > 0, test["y_ret"].values > 0]))
    if "clf_beat" in bundle and "p_beat_raw" in test and xcol in split.valid:
        vb = split.valid[xcol].notna().values
        tb = test["y_ex"].notna().values
        bundle["cal_beat"] = PlattCalibrator().fit(
            np.concatenate([val_pred["p_beat_raw"].values[vb], test["p_beat_raw"].values[tb]]),
            np.concatenate([split.valid[xcol].values[vb] > 0, test["y_ex"].values[tb] > 0]))
    bundle.update({"medians": everything[cols].median(numeric_only=True).to_dict(), "params": params,
                   "feature_set": fset, "horizon": horizon,
                   "trained_through": str(pd.Timestamp(everything["date"].max())),
                   "trained_at": str(pd.Timestamp.now().round("s"))})
    joblib.dump(bundle, config.MODELS_DIR / f"{horizon}.joblib", compress=3)
    test.to_parquet(config.BACKTEST_DIR / f"{horizon}_test.parquet", index=False)
    importance = _group_importance(models, split.test.sample(min(len(split.test), 20000),
                                                             random_state=1), ycol)
    out = {"horizon": horizon, "label": config.horizon_label(horizon), "dates": split.dates,
           "rows": {"train": len(split.train), "valid": len(split.valid), "test": len(split.test)},
           "tuning": tuning, "test": metrics, "group_importance": importance,
           "walk_forward": walk_forward, "trained_at": bundle["trained_at"],
           "minutes": round((time.time() - t0) / 60, 1)}
    with open(config.MODELS_DIR / f"{horizon}_metrics.json", "w") as f:
        json.dump(out, f, indent=1, default=_json_default)
    log(f"  saved models/{horizon}.joblib ({out['minutes']} min)")
    return out


def _walk_forward(split: Split, cols, ycol, params, horizon, test, log, months: int = 3) -> pd.DataFrame:
    """Re-train every `months` through the test window using only data before each block."""
    log(f"  walk-forward: re-training every {months} months through the test window")
    allrows = pd.concat([split.train, split.valid, split.test], ignore_index=True)
    n = config.HORIZONS[horizon][1]
    start, end = split.test["date"].min(), split.test["date"].max()
    blocks = pd.date_range(start, end + pd.DateOffset(months=months), freq=f"{months}MS")
    if len(blocks) == 0 or blocks[0] > start:
        blocks = blocks.insert(0, start)
    preds = []
    gap = pd.Timedelta(days=int(n * 1.5) + 2) if config.HORIZONS[horizon][2] == "daily" else pd.Timedelta(days=1)
    for b0, b1 in zip(blocks[:-1], blocks[1:]):
        blk = split.test[(split.test["date"] >= b0) & (split.test["date"] < b1)]
        if blk.empty:
            continue
        cal_start = b0 - pd.DateOffset(years=1) if config.HORIZONS[horizon][2] == "daily" \
            else b0 - pd.DateOffset(months=config.HOURLY_VALID_MONTHS)
        tr = allrows[allrows["date"] < cal_start - gap]
        if len(tr) > config.MAX_TRAIN_ROWS:
            tr = tr.sample(config.MAX_TRAIN_ROWS, random_state=config.RANDOM_STATE)
        va = allrows[(allrows["date"] >= cal_start) & (allrows["date"] < b0 - gap)]
        p = predict_frame(calibrate_models(fit_models(tr, cols, ycol, params), va, ycol), blk)
        preds.append(p)
        log(f"    block {b0:%Y-%m-%d}: trained on {len(tr):,} rows")
    if preds:
        P = pd.concat(preds)
        test = test.copy()
        test.loc[P.index.intersection(test.index), P.columns] = P
    return test


def _group_importance(models: dict, sample: pd.DataFrame, ycol: str) -> dict:
    """Permutation importance per feature group on test rows (AUC drop when shuffled)."""
    y = (sample[ycol] > 0).astype(int)
    if y.nunique() < 2:
        return {}
    cols = models["features"]
    base = roc_auc_score(y, models["clf"].predict_proba(sample[cols])[:, 1])
    rng = np.random.default_rng(0)
    out = {}
    for g in sorted({F.group_of(c) for c in cols}):
        X = sample[cols].copy()
        gc = [c for c in cols if F.group_of(c) == g]
        perm = rng.permutation(len(X))
        X[gc] = X[gc].values[perm]
        out[F.GROUP_NAMES.get(g, g)] = float(base - roc_auc_score(y, models["clf"].predict_proba(X)[:, 1]))
    return dict(sorted(out.items(), key=lambda kv: -kv[1]))


def _json_default(o):
    if isinstance(o, (np.integer,)):
        return int(o)
    if isinstance(o, (np.floating,)):
        return None if np.isnan(o) else float(o)
    if isinstance(o, (pd.Timestamp,)):
        return str(o)
    return str(o)


# ======================================================================= live predictions
def load_bundle(horizon: str) -> dict | None:
    p = config.MODELS_DIR / f"{horizon}.joblib"
    return joblib.load(p) if p.exists() else None


def load_metrics(horizon: str) -> dict | None:
    p = config.MODELS_DIR / f"{horizon}_metrics.json"
    return json.load(open(p)) if p.exists() else None


def latest_rows(horizon: str, inp: Inputs, log: Callable = print) -> pd.DataFrame:
    """Feature rows for the most recent bar of every ticker."""
    if config.HORIZONS[horizon][2] == "hourly":
        frame = hourly_frame(inp, log=log)
        if frame.empty:
            return frame
        last = frame["date"].max()
        return frame[frame["date"] >= last - pd.Timedelta(hours=1)].groupby("ticker").tail(1)
    stock = compute_stock(inp, horizons={}, log=log)
    last = stock["date"].max()
    rows = stock[stock["date"] >= last - pd.Timedelta(days=4)].groupby("ticker").tail(1)
    dfeat = build_date_features(inp, pd.DatetimeIndex(np.sort(stock["date"].unique())))
    return F.assemble(rows, dfeat, inp.events, inp.earnings, inp.tk_types, inp.insiders)


def tested_hit_rate(metrics: dict | None, conf: np.ndarray, key: str | None = None) -> np.ndarray:
    src = (metrics or {}).get("test", {})
    if key:
        src = src.get(key, {})
    if not src or "conf_table" not in src:
        return np.full(len(conf), np.nan)
    t = pd.DataFrame(src["conf_table"])
    out = np.full(len(conf), np.nan)
    for _, r in t.iterrows():
        m = (conf >= r["low"]) & (conf < r["high"])
        out[m] = r["hit_rate"] if r["picks"] >= 30 else np.nan
    return out


def predict_latest(horizon: str, inp: Inputs | None = None, rows: pd.DataFrame | None = None,
                   save: bool = True, log: Callable = print) -> pd.DataFrame:
    bundle = load_bundle(horizon)
    if bundle is None:
        return pd.DataFrame()
    if rows is None:
        inp = inp or load_inputs(start=_live_start(horizon))
        rows = latest_rows(horizon, inp, log=log)
    if rows.empty:
        return pd.DataFrame()
    for c in bundle["features"]:
        if c not in rows:
            rows[c] = np.nan
    preds = predict_frame(bundle, rows)
    out = rows[_keep_cols(rows)].copy().join(preds)
    out["conf"] = np.maximum(out["p_up"], 1 - out["p_up"])
    out["side"] = np.where(out["p_up"] >= 0.5, "UP", "DOWN")
    met = load_metrics(horizon)
    out["tested_hit_rate"] = tested_hit_rate(met, out["conf"].values)
    if "p_beat" in out:
        out["beat_conf"] = np.maximum(out["p_beat"], 1 - out["p_beat"])
        out["beat_tested_hit_rate"] = tested_hit_rate(met, out["beat_conf"].values, key="beat")
    out["horizon"] = horizon
    uni = data.load_universe()
    if not uni.empty:
        out = out.merge(uni[["ticker", "name"]], on="ticker", how="left")
    out = out.sort_values("p_up", ascending=False).reset_index(drop=True)
    if save:
        out.to_parquet(config.PRED_DIR / f"latest_{horizon}.parquet", index=False)
        feat_cols = [c for c in rows.columns if F.group_of(c) in F.GROUP_NAMES]
        rows[["date", "ticker"] + feat_cols].to_parquet(config.PRED_DIR / f"latest_rows_{horizon}.parquet",
                                                        index=False)
        _append_history(out)
    return out


def _live_start(horizon: str) -> str:
    """Daily features need ~1.5 years of history; no need to load all 25."""
    return (pd.Timestamp.now() - pd.Timedelta(days=600)).strftime("%Y-%m-%d")


def _append_history(pred: pd.DataFrame) -> None:
    keep = ["date", "ticker", "horizon", "close", "p_up", "p_beat", "exp_ret", "exp_move", "conf",
            "tested_hit_rate"]
    new = pred[[c for c in keep if c in pred]].copy()
    new["predicted_at"] = pd.Timestamp.now().round("s")
    p = config.PRED_HISTORY_FILE
    if p.exists():
        old = pd.read_parquet(p)
        new = pd.concat([old, new]).drop_duplicates(["date", "ticker", "horizon"], keep="last")
    new.to_parquet(p, index=False)


def load_latest(horizon: str) -> pd.DataFrame:
    p = config.PRED_DIR / f"latest_{horizon}.parquet"
    return pd.read_parquet(p) if p.exists() else pd.DataFrame()


def load_latest_rows(horizon: str) -> pd.DataFrame:
    p = config.PRED_DIR / f"latest_rows_{horizon}.parquet"
    return pd.read_parquet(p) if p.exists() else pd.DataFrame()


# ======================================================================= explanations
def explain_row(bundle: dict, row: pd.DataFrame) -> pd.DataFrame:
    """How much each feature group moved this stock's chance of going up: switch the group
    to typical (median) values, re-run the model, and measure the difference."""
    cols = bundle["features"]
    X = row[cols].iloc[[0]].copy()
    base = float(bundle["cal"].transform(bundle["clf"].predict_proba(X)[:, 1])[0])
    rows = []
    for g in sorted({F.group_of(c) for c in cols}):
        gc = [c for c in cols if F.group_of(c) == g]
        Xn = X.copy()
        for c in gc:
            Xn[c] = bundle["medians"].get(c, np.nan)
        pn = float(bundle["cal"].transform(bundle["clf"].predict_proba(Xn)[:, 1])[0])
        rows.append({"group": F.GROUP_NAMES.get(g, g), "effect_pts": 100 * (base - pn)})
    return pd.DataFrame(rows).sort_values("effect_pts", key=np.abs, ascending=False)


def reasons(row: pd.Series, universe_rows: pd.DataFrame | None = None) -> list[str]:
    """Plain-English facts about the stock that a trader would look at."""
    out = []

    def pct(v):
        return f"{100 * (np.exp(v) - 1):+.1f}%"

    if pd.notna(row.get("mom_ret_21")):
        rank = row.get("mom_rank_21")
        extra = ""
        if pd.notna(rank):
            extra = f" (top {max(1, round(100 * (1 - rank)))}% of stocks)" if rank >= 0.5 else \
                f" (bottom {max(1, round(100 * rank))}% of stocks)"
        out.append(f"{pct(row['mom_ret_21'])} over the last month{extra}")
    if pd.notna(row.get("mom_ret_252")):
        out.append(f"{pct(row['mom_ret_252'])} over the last year")
    rsi = row.get("trend_rsi_14")
    if pd.notna(rsi):
        tag = " — overbought" if rsi >= 70 else " — oversold" if rsi <= 30 else ""
        out.append(f"RSI {rsi:.0f}{tag}")
    s200 = row.get("trend_sma200")
    if pd.notna(s200):
        out.append(f"{abs(s200):.0%} {'above' if s200 >= 0 else 'below'} its 200-day average")
    hi = row.get("trend_hi_252")
    if pd.notna(hi) and hi > -0.03:
        out.append("near its 52-week high")
    vz = row.get("liq_vol_z")
    if pd.notna(vz) and vz > 0.7:
        out.append(f"volume {np.exp(vz):.1f}× normal")
    vol = row.get("risk_vol_21")
    if pd.notna(vol):
        out.append(f"recent volatility {vol:.0%} a year")
    ue = row.get("evt_tk_until_earnings")
    if pd.notna(ue) and ue <= 30:
        out.append(f"earnings in {int(ue)} days")
    se = row.get("evt_tk_since_earnings")
    if pd.notna(se) and se <= 5:
        sur = row.get("evt_tk_last_surprise")
        out.append(f"reported earnings {int(se)} days ago" + (f" (surprise {sur:+.1f}%)" if pd.notna(sur) else ""))
    for c in row.index:
        if c.startswith("evt_since_") and pd.notna(row[c]) and row[c] <= 14:
            out.append(f"{c.replace('evt_since_', '').replace('_', ' ')} event {int(row[c])} days ago")
        if c.startswith("evt_until_") and pd.notna(row[c]) and row[c] <= 14:
            out.append(f"{c.replace('evt_until_', '').replace('_', ' ')} in {int(row[c])} days")
        if c.startswith("evt_tk_since_") and c != "evt_tk_since_earnings" and pd.notna(row[c]) and row[c] <= 30:
            out.append(f"{c.replace('evt_tk_since_', '').replace('_', ' ')} {int(row[c])} days ago")
    vix = row.get("mkt_vix")
    if pd.notna(vix):
        out.append(f"VIX {vix:.0f}" + (" (fearful market)" if vix >= 25 else " (calm market)" if vix <= 14 else ""))
    return out
