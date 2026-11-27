# 📈 Stock Analyzer

Same idea as the NBA props app, for the stock market. A Streamlit app plus models trained
on **25 years of free Yahoo data for every NYSE stock**, along with market, cross-asset and economy
series and your own list of events. It predicts which stocks are **most likely to go up or down
over the next hour, day, week, or month**, plus how big the move should be (useful for options).
It also tests every model on the **last 2 years as if they never happened**.

> Research tool, not financial advice. Past patterns can stop working, and a backtest is a simulation.
> The Tracker page shows how live picks actually do.

## Setup (once)

Open this folder in VS Code (**File → Open Folder…**), then in the terminal (**Ctrl+`**):

```bash
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
cp .env.example .env               # optional settings (exchanges, phone alerts, memory)
```

Two free keys make the new data sources complete. Add them to `.env`:

```
FRED_API_KEY=...                         # exact CPI/jobs/GDP/PCE/PPI/retail release dates (1-minute signup:
                                         # https://fred.stlouisfed.org/docs/api/api_key.html)
SEC_USER_AGENT=Your Name your@email.com  # the SEC asks downloads to identify themselves (insider trades)
```

## First run

```bash
python scripts/download_data.py --limit 50   # 2-minute test with 50 stocks (optional)
python scripts/download_data.py --all        # everything: ~2,000 NYSE stocks since 2000, hourly bars,
                                             # market + economy, earnings, sectors, insider trades,
                                             # economic calendar, today's option chains (≈1.5–2.5 h once)
python scripts/train.py                      # train + tune + 2-year backtest (≈20–60 min)
streamlit run app.py                         # the dashboard
```

After that, `python scripts/daily.py` updates prices and predictions in a few minutes, then saves today's option chains (≈20 min; skip with `--no-options`). To run it automatically,
use `bash scripts/install_daily_job.sh`: it runs weekdays after the close, plus next-hour picks every hour
during market hours. As with the NBA app, background jobs can't read `~/Desktop`, so move the folder to
`~/Projects` first if you use the scheduled job.

Retrain weekly or monthly with `python scripts/train.py` (or `scripts/daily.py --retrain`).

## What gets downloaded (all free)

| Data | Source | Notes |
|---|---|---|
| **Every NYSE common stock** (~2,000) | nasdaqtrader.com symbol list | Preferreds, warrants, units and test issues are removed. Add NASDAQ with `STOCK_EXCHANGES=N,Q` in `.env`, or add single tickers on the Data page |
| **Daily prices since 2000** | Yahoo | Split and dividend adjusted. Updates are incremental; tickers with a new split get re-downloaded automatically |
| **Hourly prices** | Yahoo | Yahoo only serves the **last 730 days** of hourly data, so the next-hour model trains on 2 years, not 25 |
| **Market & cross-asset** | Yahoo | S&P 500, Nasdaq 100, Russell 2000, VIX, yields, oil, gold, copper, nat gas, silver, dollar, euro, yen, bitcoin, long bonds, junk bonds, 11 sector ETFs, plus "unrelated" series: shipping, homebuilders, semis, retail, airlines, transports, lumber, corn, wheat |
| **Economy** | FRED | CPI, core CPI, PCE, unemployment, payrolls, jobless claims, Fed funds, 2y/10y yields, yield curves, junk-bond spreads, real rates, inflation expectations, industrial production, retail sales, housing starts, consumer sentiment, M2, GDP, financial stress, dollar index, mortgage rates |
| **Earnings dates + surprises** | Yahoo | Past and upcoming. Slow the first time (one request per stock). Resumable |
| **Economic calendar** | federalreserve.gov + FRED | Every Fed decision since 2000 plus CPI, jobs report, GDP, PCE, PPI and retail sales release dates, past and scheduled. Becomes events and "days until the Fed/CPI/jobs report" inputs |
| **Sectors & industries** | Yahoo | One lookup per stock, refreshed every 90 days. Powers sector-relative and peer features |
| **Insider trades** | SEC Form 4 data sets | Every open-market insider purchase and sale since 2006 (≈1 GB once, then a new file each quarter) |
| **Option chains** | Yahoo | Saved **every day** for the ~300 most-traded stocks: at-the-money straddle, implied move, implied volatility, skew, put/call volume. Free sources have no option history, so this builds your own |
| **Your events** | `data/events.csv` | Starts with elections, wars, earnings, product launches, CEO changes and returns, market shocks and Fed policy. Edit it on the Events page |

**Add more:** FRED series on the Data page (any ID from fred.stlouisfed.org), extra tickers, and
**your own series**: drop any CSV with `date,value` columns into `data/custom_series/`, such as a Google
Trends export, weather, box office or web traffic. Each one shows up on Hidden links and becomes a model input
at the next retrain.

## How the models work

Each horizon (**next hour / day / week / month**) has three gradient-boosted models (scikit-learn, as in the
NBA app):

- **Direction**: chance the stock is up at the end of the horizon, calibrated so a 60% happens about 60% of the time
- **Return**: expected % return
- **Move**: expected size of the move either way. For options, compare it with the straddle price.
- **Beat the S&P**: chance the stock does better than the S&P 500 over the horizon. This ignores which way the
  whole market goes, so it measures pure stock-picking. On a day the market falls, most "up" picks lose, but
  "beat the S&P" picks can still be right.

**Inputs.** Every input for day *t* uses only what was public at the close of day *t*. Economy numbers
appear only after their release date: CPI for March isn't visible until mid-April.

- **Momentum:** 1-day to 12-month returns, return vs the S&P, rank vs all other stocks
- **Volatility & beta:** 1- and 3-month volatility, daily range, biggest up and down days, beta to the S&P
- **Trend:** RSI, distance from the 50- and 200-day averages, 52-week high and low, overnight gap
- **Volume & liquidity:** volume vs normal, $ volume, price
- **Events:** days until and since earnings, last earnings surprise, days since and until each market-wide event
  type (elections, wars, shocks, policy, ...), days since each company event (CEO change, launch, ...).
  **Any new event type you add with 3+ examples becomes an input automatically.**
- **Economic calendar:** days until and since the next Fed decision, CPI report, jobs report, GDP, PCE, PPI and retail sales
- **Sector & peers:** return vs its sector and industry, sector and industry momentum, and **earnings
  spillover**: how industry peers that reported in the last 5 days moved on their earnings, and how many peers
  report in the next 10 days
- **Insider trading:** $ bought and sold by insiders in the last 90 days, net buying ratio, number of distinct
  buyers, whether the CEO/CFO/President bought, days since the last purchase. Each trade counts from the day after its
  filing became public.
- **Market & cross-asset:** 1-week and 1-month moves of every series above, VIX level, S&P trend
- **Economy:** every FRED series, as level, change or year-over-year, depending on the series
- **Calendar:** day of week, month, turn of month
- **Hourly model:** last 1 to 35 bars, intraday volatility, position in the day's range, hour of day, SPY/VIX/QQQ
  hourly moves, plus the stock's daily picture as of yesterday's close

## The 2-year test, done honestly

```
2000 ──────────── TRAIN ──────────── 3 yrs ago ── TUNE ── 2 yrs ago ──── TEST ──── today
                                                                    (never seen)
```

1. **Train** on everything up to 3 years ago.
2. **Tune** on the year after that. Several model settings and three feature sets are tried: everything,
   no economy data, and stock + events only. The best one on that year wins.
3. **Test** on the last 2 years. The chosen model predicts every day of them without ever having seen them,
   and nothing about those 2 years influenced its settings.
4. **Live:** refit on everything with the chosen settings to make today's picks.

You asked to "train the model to be successful based off" the test years. The model is tuned to succeed on
the **validation year**, then graded on the 2 test years. If it were tuned on the test years directly, the
backtest would look great and mean nothing, because the settings would have been picked by peeking at the
answers. `python scripts/train.py --walk-forward` goes further: it re-trains every quarter through the test
window, the way you would really use it.

A gap equal to the horizon is left between each period, so a training target never overlaps the next period.
Hourly data only goes back 2 years, so the hourly test is the **last 6 months** (tuned on the 3 months before).

**Survivorship bias:** the stock list is today's NYSE, so companies that went bankrupt since 2000 are missing.
This mostly flatters the long 25-year history. The last-2-years test is much less affected.

## Pages

| Page | What it does |
|---|---|
| **🔥 Top Picks** | Most likely up and down, or most likely to **beat / lag the S&P**, for the chosen horizon. **Filters:** confidence, price, $ volume, earnings inside the window, rank by chance, expected return or expected move, and a switch to hide confidence levels that lost in testing. Each card shows chance, expected return, expected move, **tested hit rate**, and plain-English reasons. Scatter of every stock: expected return vs expected move |
| **🔎 Stock analyzer** | Sector, industry and peers; predictions for all 4 horizons; insider buying and selling chart; saved implied-move and implied-volatility charts; a candlestick chart with earnings and event markers, **why** (each group of inputs switched off to see how much it moved the chance), the model's 2-year track record on that stock, past earnings move vs a normal day, and an **options check**: live option chain → implied move from the at-the-money straddle vs the model's expected move |
| **💵 Options ideas** | For every stock with a saved option snapshot, it compares the **move options are priced for** (straddle ÷ price) with the **move history says to expect** (around past earnings when earnings fall before expiry) and the **model's expected move**. It flags options as **cheap** (long straddle/strangle) or **expensive** (defined-risk premium selling), with a directional lean from the model. **Paper-track** ideas, and they're graded at expiry (straddle worth \|price − strike\|). The daily job tracks the 10 strongest each day automatically. Snapshot history shows implied volatility over time |
| **📅 Events** | **Event study:** average path before and after any event type, with a 95% band and each event's line. Day-0 move vs a normal day, and volatility before vs after. **Impact on the economy:** how the S&P, VIX, yields, oil, gold, dollar, bitcoin and every sector reacted. **Stocks that react most**, with t-stats. **Upcoming:** next 45 days of events and earnings, with past event moves and the model's expected move. **Edit the event list** in a table |
| **🧭 Hidden links** | Does series X move *before* stock Y? Every driver × every stock × 1–4 weeks of lead time. Found before the test window, checked in the last 2 years ("held up"), with correction for testing thousands of pairs. Charts each link and its stability over time |
| **📊 Chart builder** | Presets: does a higher rank give better returns (deciles), market mood, predicted vs actual move, hit rate over time. **Build your own:** scatter, histogram, box, top-N or line from latest predictions, backtest predictions or prediction history, using any columns. CSV download |
| **🧪 Backtest** | Accuracy vs always guessing "up", AUC, same-day ranking AUC, Brier score, return IC. **Strategy simulator:** buy the top N and short the bottom N, with costs, price/volume filters and minimum confidence; it re-runs instantly. Growth chart vs the S&P, hit rate by confidence, calibration, which data mattered, and tuning results |
| **📒 Tracker** | Live picks graded after the fact. The real scoreboard |
| **⚙️ Data & models** | What's downloaded, model status, buttons to update/re-score/re-train, and editors for extra FRED series, extra tickers and your own CSV series |

## How to read the numbers

- **Stock direction is hard.** In the 2-year test, a **same-day ranking AUC of 0.52–0.55** is a real edge, and
  60%+ accuracy for daily direction is very unlikely to hold up. If something looks too good, check the
  strategy with costs.
- **The next-hour model pays costs every hour.** Even with a real signal, it often loses after costs.
  The Backtest page shows both, so set cost to 0 to see the raw signal.
- **For options**, the move model is often more useful than direction: the size of a move is far more predictable
  than which way it goes. Use the Stock page's options check to see whether the market already prices it in.
- **Hidden links:** most strong-looking links between unrelated series are luck. Only trust ones that **held up**
  in the last 2 years.

## Memory

Training on the full NYSE uses about 4–6 GB of RAM. On an 8 GB Mac, set `MAX_TRAIN_ROWS=800000` in `.env`,
or start history later (`HISTORY_START=2005-01-01`).

## Tests

`pytest` runs the whole pipeline on made-up data with planted patterns. It never touches your real data:
the model has to find the patterns, and the Hidden links scan has to find the planted oil-sensitive stocks.

## Project layout

```
app.py                  navigation
app_pages/              one file per page
stocks/
  config.py             settings: exchanges, series lists, horizons, test window, filters
  data.py               downloads: universe, Yahoo prices, FRED, earnings; loaders
  events.py             event list + event studies
  features.py           all model inputs (no look-ahead)
  model.py              training, tuning, 2-year backtest, live predictions, explanations
  links.py              hidden-links scanner
  macro_calendar.py     Fed / CPI / jobs / GDP / PCE / PPI / retail release dates
  sectors.py            sector & industry lookup, sector/peer/earnings-spillover features
  insiders.py           SEC insider trades → features + insider_buy events
  options.py            daily option snapshots, implied moves, ideas, paper-trade grading
  tracker.py            grading live picks
  ui.py                 shared chart style + cached loaders
scripts/                download_data, train, daily, install_daily_job.sh
data/                   events.csv, custom_series/, downloaded data (parquet)
models/                 trained models + test metrics
tests/                  pipeline tests on synthetic data
```

## Ideas for next

- VIX term structure (VIX9D / VIX3M / VIX6M) as market-fear-timing inputs
- Fundamentals from SEC XBRL filings (earnings, revenue, margins, debt since ~2009)
- FINRA daily short-sale volume
- Use the saved option snapshots as model inputs once there's a year of history
- News sentiment (GDELT)
