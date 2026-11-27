"""Project-wide settings. Edit these instead of hunting through the code."""
from __future__ import annotations

import os
from pathlib import Path

try:
    from dotenv import load_dotenv

    load_dotenv()
except ImportError:  # python-dotenv is optional
    pass

ROOT = Path(__file__).resolve().parent.parent
DATA_DIR = Path(os.getenv("STOCKS_DATA_DIR", ROOT / "data"))
PRICES_DIR = DATA_DIR / "prices"
CONTEXT_DIR = DATA_DIR / "context"
BACKTEST_DIR = DATA_DIR / "backtest"
PRED_DIR = DATA_DIR / "predictions"
CUSTOM_DIR = DATA_DIR / "custom_series"   # drop your own date,value CSVs here
MODELS_DIR = Path(os.getenv("STOCKS_MODELS_DIR", ROOT / "models"))
for _d in (DATA_DIR, PRICES_DIR, CONTEXT_DIR, BACKTEST_DIR, PRED_DIR, CUSTOM_DIR, MODELS_DIR):
    _d.mkdir(parents=True, exist_ok=True)

DAILY_FILE = PRICES_DIR / "daily.parquet"
HOURLY_FILE = PRICES_DIR / "hourly.parquet"
UNIVERSE_FILE = DATA_DIR / "universe.csv"
YAHOO_CONTEXT_FILE = CONTEXT_DIR / "yahoo_context.parquet"
FRED_FILE = CONTEXT_DIR / "fred.parquet"
EARNINGS_FILE = DATA_DIR / "earnings.parquet"
EVENTS_FILE = DATA_DIR / "events.csv"
TRACKER_FILE = DATA_DIR / "tracker.parquet"
PRED_HISTORY_FILE = PRED_DIR / "history.parquet"

# ---------------------------------------------------------------- universe
# Which exchanges to include. N = NYSE, A = NYSE American, P = NYSE Arca.
# Add "Q" to also pull NASDAQ (Apple, Nvidia, Microsoft, Tesla, ...).
EXCHANGES = [x.strip() for x in os.getenv("STOCK_EXCHANGES", "N").split(",") if x.strip()]
INCLUDE_ETFS = os.getenv("INCLUDE_ETFS", "0") == "1"

# ---------------------------------------------------------------- history
HISTORY_START = os.getenv("HISTORY_START", "2000-01-01")   # ~25 years
HOURLY_DAYS = 729            # Yahoo serves hourly bars for the last 730 days only

# Market and cross-asset series pulled from Yahoo (free). name -> Yahoo symbol
YAHOO_CONTEXT = {
    "spy": "SPY", "sp500": "^GSPC", "nasdaq100": "^NDX", "russell2000": "^RUT",
    "vix": "^VIX", "tnx": "^TNX", "irx": "^IRX",
    "oil": "CL=F", "gold": "GC=F", "copper": "HG=F", "natgas": "NG=F", "silver": "SI=F",
    "dollar": "DX-Y.NYB", "eurusd": "EURUSD=X", "usdjpy": "JPY=X",
    "bitcoin": "BTC-USD", "tlt": "TLT", "hyg": "HYG",
    # sector ETFs
    "xlk": "XLK", "xlf": "XLF", "xle": "XLE", "xlv": "XLV", "xli": "XLI", "xly": "XLY",
    "xlp": "XLP", "xlu": "XLU", "xlb": "XLB", "xlre": "XLRE", "xlc": "XLC",
    # "seemingly unrelated" starters
    "shipping": "BDRY", "homebuilders": "XHB", "semis": "SMH", "retail": "XRT",
    "airlines": "JETS", "transports": "IYT", "lumber": "WOOD", "corn": "CORN", "wheat": "WEAT",
}

# FRED economic series (free, no key). name -> (FRED id, publication lag in days, kind)
# Lag = days after the observation date that the number is actually public. The model
# only sees a value once it's public, so it can't peek at data released later.
# kind: "level" (use as is), "yoy" (also add year-over-year % change), "chg" (also add change)
FRED_SERIES = {
    "cpi": ("CPIAUCSL", 45, "yoy"),
    "core_cpi": ("CPILFESL", 45, "yoy"),
    "unemployment": ("UNRATE", 37, "chg"),
    "payrolls": ("PAYEMS", 37, "chg"),
    "jobless_claims": ("ICSA", 5, "chg"),
    "fed_funds": ("DFF", 1, "chg"),
    "treasury_10y": ("DGS10", 1, "chg"),
    "treasury_2y": ("DGS2", 1, "chg"),
    "curve_10y_2y": ("T10Y2Y", 1, "level"),
    "curve_10y_3m": ("T10Y3M", 1, "level"),
    "hy_spread": ("BAMLH0A0HYM2", 1, "chg"),
    "real_rate_10y": ("DFII10", 1, "level"),
    "breakeven_10y": ("T10YIE", 1, "level"),
    "industrial_prod": ("INDPRO", 47, "yoy"),
    "retail_sales": ("RSAFS", 46, "yoy"),
    "housing_starts": ("HOUST", 48, "yoy"),
    "consumer_sentiment": ("UMCSENT", 30, "chg"),
    "m2": ("M2SL", 55, "yoy"),
    "gdp": ("GDPC1", 120, "yoy"),
    "pce_inflation": ("PCEPILFE", 60, "yoy"),
    "financial_stress": ("STLFSI4", 8, "level"),
    "dollar_index": ("DTWEXBGS", 1, "chg"),
    "mortgage_30y": ("MORTGAGE30US", 2, "chg"),
}

# ---------------------------------------------------------------- model
# horizon key -> (label, bars ahead, bar type)
HORIZONS = {
    "1h": ("Next hour", 1, "hourly"),
    "1d": ("Next day", 1, "daily"),
    "1w": ("Next week", 5, "daily"),
    "1m": ("Next month", 21, "daily"),
}

TEST_YEARS = 2.0          # last 2 years are held out and graded as if they never happened
VALID_YEARS = 1.0         # the year before that is used to tune settings
HOURLY_TEST_MONTHS = 6    # hourly data only goes back 2 years, so its test window is shorter
HOURLY_VALID_MONTHS = 3

# Keep training sets a size a laptop can handle. Daily rows are sampled every N days
# (neighbouring days are nearly identical for week/month targets anyway).
TRAIN_SAMPLE_EVERY = {"1d": 3, "1w": 5, "1m": 10}
MAX_TRAIN_ROWS = int(os.getenv("MAX_TRAIN_ROWS", "1500000"))
RANDOM_STATE = 7

# Tradability filters used in the backtest and as Top Picks defaults
MIN_PRICE = 5.0
MIN_DOLLAR_VOLUME = 5_000_000      # average daily $ volume over the last month
BACKTEST_TOP_N = 10
COST_BPS = 5                       # cost per trade side, in basis points

# Event types whose dates are known ahead of time (the model may use "days until")
ADVANCE_KNOWN_EVENTS = {"election", "earnings", "fomc", "cpi_release", "jobs_report", "gdp_release",
                        "pce_release", "ppi_release", "retail_sales_release"}
DEFAULT_EVENT_TYPES = ["election", "earnings", "war", "product_launch", "ceo_change",
                       "market_shock", "policy"]
# Event types the model gets through dedicated features instead of generic "days since"
MODEL_SKIP_TICKER_TYPES = {"earnings", "insider_buy"}

# ---------------------------------------------------------------- new data sources
SEC_USER_AGENT = os.getenv("SEC_USER_AGENT", "")       # SEC asks for "Your Name your@email.com"
INSIDER_START_YEAR = int(os.getenv("INSIDER_START_YEAR", "2006"))
OPTIONS_TOP_N = int(os.getenv("OPTIONS_TOP_N", "300"))  # most-traded stocks to snapshot daily
OPTIONS_MAX_DAYS = 70                                    # expiries up to ~10 weeks out
OPTIONS_DIR = DATA_DIR / "options"
OPTIONS_DIR.mkdir(parents=True, exist_ok=True)
SECTORS_FILE = DATA_DIR / "sectors.csv"
INSIDERS_DIR = DATA_DIR / "insiders"
INSIDERS_DIR.mkdir(parents=True, exist_ok=True)

REQUEST_PAUSE = float(os.getenv("YAHOO_PAUSE", "2.0"))   # seconds between Yahoo batches
NTFY_TOPIC = os.getenv("NTFY_TOPIC", "")


def horizon_label(h: str) -> str:
    return HORIZONS[h][0]
