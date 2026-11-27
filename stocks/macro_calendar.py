"""Scheduled economic releases: Fed decisions (FOMC), CPI, jobs report, GDP, PCE, PPI,
retail sales. Their dates are public in advance, so the model may use "days until".

Sources (all free):
  - FOMC: the Fed's own meeting calendars (federalreserve.gov), with a built-in backup list
  - Everything else: FRED's release calendar API. Needs a free FRED API key in .env
    (FRED_API_KEY=...; get one in a minute at https://fred.stlouisfed.org/docs/api/api_key.html).
    Without a key, jobs-report dates are estimated with the BLS rule and the rest are skipped.

Saved to data/macro_calendar.csv (date, type, title) and merged into the event list as
market-wide events automatically.
"""
from __future__ import annotations

import os
import re
import time
from typing import Callable

import pandas as pd
import requests

from . import config

CAL_FILE = config.DATA_DIR / "macro_calendar.csv"
UA = {"User-Agent": "Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) stock-analyzer"}

# FRED series whose *release* dates we want → event type, title
RELEASES = {
    "CPIAUCSL": ("cpi_release", "CPI inflation report"),
    "PAYEMS": ("jobs_report", "Jobs report (nonfarm payrolls)"),
    "GDPC1": ("gdp_release", "GDP report"),
    "PCEPI": ("pce_release", "PCE inflation report"),
    "PPIFIS": ("ppi_release", "Producer prices (PPI)"),
    "RSAFS": ("retail_sales_release", "Retail sales report"),
}
MONTHS = ["January", "February", "March", "April", "May", "June", "July", "August", "September",
          "October", "November", "December"]
_M = "|".join(MONTHS)

# Backup: decision dates (day 2 of each meeting) 2000–2027 including emergency moves,
# used if the Fed's site can't be read. Updated Nov 2026.
FOMC_BACKUP = """
2000-02-02 2000-03-21 2000-05-16 2000-06-28 2000-08-22 2000-10-03 2000-11-15 2000-12-19
2001-01-03 2001-01-31 2001-03-20 2001-04-18 2001-05-15 2001-06-27 2001-08-21 2001-09-17 2001-10-02
2001-11-06 2001-12-11 2002-01-30 2002-03-19 2002-05-07 2002-06-26 2002-08-13 2002-09-24 2002-11-06
2002-12-10 2003-01-29 2003-03-18 2003-05-06 2003-06-25 2003-08-12 2003-09-16 2003-10-28 2003-12-09
2004-01-28 2004-03-16 2004-05-04 2004-06-30 2004-08-10 2004-09-21 2004-11-10 2004-12-14 2005-02-02
2005-03-22 2005-05-03 2005-06-30 2005-08-09 2005-09-20 2005-11-01 2005-12-13 2006-01-31 2006-03-28
2006-05-10 2006-06-29 2006-08-08 2006-09-20 2006-10-25 2006-12-12 2007-01-31 2007-03-21 2007-05-09
2007-06-28 2007-08-07 2007-08-17 2007-09-18 2007-10-31 2007-12-11 2008-01-22 2008-01-30 2008-03-18
2008-04-30 2008-06-25 2008-08-05 2008-09-16 2008-10-08 2008-10-29 2008-12-16 2009-01-28 2009-03-18
2009-04-29 2009-06-24 2009-08-12 2009-09-23 2009-11-04 2009-12-16 2010-01-27 2010-03-16 2010-04-28
2010-06-23 2010-08-10 2010-09-21 2010-11-03 2010-12-14 2011-01-26 2011-03-15 2011-04-27 2011-06-22
2011-08-09 2011-09-21 2011-11-02 2011-12-13 2012-01-25 2012-03-13 2012-04-25 2012-06-20 2012-08-01
2012-09-13 2012-10-24 2012-12-12 2013-01-30 2013-03-20 2013-05-01 2013-06-19 2013-07-31 2013-09-18
2013-10-30 2013-12-18 2014-01-29 2014-03-19 2014-04-30 2014-06-18 2014-07-30 2014-09-17 2014-10-29
2014-12-17 2015-01-28 2015-03-18 2015-04-29 2015-06-17 2015-07-29 2015-09-17 2015-10-28 2015-12-16
2016-01-27 2016-03-16 2016-04-27 2016-06-15 2016-07-27 2016-09-21 2016-11-02 2016-12-14 2017-02-01
2017-03-15 2017-05-03 2017-06-14 2017-07-26 2017-09-20 2017-11-01 2017-12-13 2018-01-31 2018-03-21
2018-05-02 2018-06-13 2018-08-01 2018-09-26 2018-11-08 2018-12-19 2019-01-30 2019-03-20 2019-05-01
2019-06-19 2019-07-31 2019-09-18 2019-10-30 2019-12-11 2020-01-29 2020-03-03 2020-03-15 2020-04-29
2020-06-10 2020-07-29 2020-09-16 2020-11-05 2020-12-16 2021-01-27 2021-03-17 2021-04-28 2021-06-16
2021-07-28 2021-09-22 2021-11-03 2021-12-15 2022-01-26 2022-03-16 2022-05-04 2022-06-15 2022-07-27
2022-09-21 2022-11-02 2022-12-14 2023-02-01 2023-03-22 2023-05-03 2023-06-14 2023-07-26 2023-09-20
2023-11-01 2023-12-13 2024-01-31 2024-03-20 2024-05-01 2024-06-12 2024-07-31 2024-09-18 2024-11-07
2024-12-18 2025-01-29 2025-03-19 2025-05-07 2025-06-18 2025-07-30 2025-09-17 2025-10-29 2025-12-10
2026-01-28 2026-03-18 2026-04-29 2026-06-17 2026-07-29 2026-09-16 2026-10-28 2026-12-09
2027-01-27 2027-03-17 2027-04-28 2027-06-09 2027-07-28 2027-09-15 2027-11-17
""".split()


# ======================================================================= FOMC
def _text(html: str) -> str:
    t = re.sub(r"<script.*?</script>|<style.*?</style>", " ", html, flags=re.S | re.I)
    t = re.sub(r"<[^>]+>", " ", t)
    return re.sub(r"\s+", " ", t.replace("&nbsp;", " "))


def parse_fomc_calendar_page(html: str) -> list[pd.Timestamp]:
    """fomccalendars.htm: one panel per year ('2026 FOMC Meetings'), entries like
    'January 27-28', 'March 17-18*', 'Jan/Feb 31-1', 'Apr/May 30-1'."""
    txt = _text(html)
    out = []
    parts = re.split(r"(\d{4}) FOMC Meetings", txt)
    for i in range(1, len(parts) - 1, 2):
        year = int(parts[i])
        body = parts[i + 1]
        for m in re.finditer(r"\b(" + _M + r"|Jan/Feb|Feb/Mar|Mar/Apr|Apr/May|May/Jun|Jun/Jul|Jul/Aug|"
                             r"Aug/Sep|Sep/Oct|Oct/Nov|Nov/Dec)\s+(\d{1,2})(?:-(\d{1,2}))?\*?(?:\s|$)", body):
            mon, d1, d2 = m.group(1), int(m.group(2)), m.group(3)
            if "/" in mon:
                mon = mon.split("/")[1]
            mon_idx = [x[:3] for x in MONTHS].index(mon[:3]) + 1
            day = int(d2) if d2 else d1
            try:
                out.append(pd.Timestamp(year=year, month=mon_idx, day=day))
            except ValueError:
                continue
    return out


def parse_fomc_historical_page(html: str, year: int) -> list[pd.Timestamp]:
    """fomchistoricalYYYY.htm: headings like 'January 24-25 Meeting - 2012',
    'July 31-August 1 Meeting - 2012', 'March 13 Meeting - 2012', 'January 21 (unscheduled) - 2008'."""
    txt = _text(html)
    out = []
    pat = (r"\b(" + _M + r")\s+(\d{1,2})(?:\s*-\s*(?:(" + _M + r")\s+)?(\d{1,2}))?\s*"
           r"(?:\((?:unscheduled|Conference Call|conference call)\)\s*)?(?:Meeting|Conference Call)?\s*-\s*" + str(year))
    for m in re.finditer(pat, txt):
        mon = m.group(3) or m.group(1)
        day = int(m.group(4) or m.group(2))
        try:
            out.append(pd.Timestamp(year=year, month=MONTHS.index(mon) + 1, day=day))
        except ValueError:
            continue
    return out


def fetch_fomc_dates(log: Callable = print) -> list[pd.Timestamp]:
    dates: set = set(pd.Timestamp(d) for d in FOMC_BACKUP)
    base = "https://www.federalreserve.gov/monetarypolicy/"
    got = 0
    try:
        r = requests.get(base + "fomccalendars.htm", headers=UA, timeout=30)
        r.raise_for_status()
        new = parse_fomc_calendar_page(r.text)
        dates |= set(new)
        got += len(new)
    except Exception as e:
        log(f"  Fed calendar page: {e} (using built-in list)")
    first_cal_year = 2021
    for y in range(int(config.HISTORY_START[:4]), first_cal_year):
        try:
            r = requests.get(base + f"fomchistorical{y}.htm", headers=UA, timeout=30)
            if r.ok:
                new = parse_fomc_historical_page(r.text, y)
                dates |= set(new)
                got += len(new)
        except Exception:
            pass
        time.sleep(0.3)
    log(f"  FOMC: {len(dates)} decision dates ({got} read from federalreserve.gov)")
    return sorted(dates)


# ======================================================================= FRED releases
def fetch_release_dates(series_id: str, api_key: str) -> list[pd.Timestamp]:
    api = "https://api.stlouisfed.org/fred/"
    r = requests.get(api + "series/release", params={"series_id": series_id, "api_key": api_key,
                                                      "file_type": "json"}, timeout=30)
    r.raise_for_status()
    rid = r.json()["releases"][0]["id"]
    r = requests.get(api + "release/dates", params={
        "release_id": rid, "api_key": api_key, "file_type": "json", "realtime_start": "1999-01-01",
        "realtime_end": "9999-12-31", "include_release_dates_with_no_data": "true",
        "sort_order": "asc", "limit": 10000}, timeout=30)
    r.raise_for_status()
    return [pd.Timestamp(x["date"]) for x in r.json().get("release_dates", [])]


def jobs_report_rule(start: str, end: str) -> list[pd.Timestamp]:
    """Fallback without a FRED key: BLS releases the jobs report on the third Friday after the
    week containing the 12th (usually the first Friday of the next month). Shutdowns and
    holidays occasionally move it; with a FRED key you get the exact dates."""
    out = []
    for m in pd.date_range(start, end, freq="MS"):
        d12 = m + pd.Timedelta(days=11)
        week_end = d12 + pd.Timedelta(days=(5 - d12.dayofweek) % 7)    # Saturday of that week
        fri = week_end + pd.Timedelta(days=6)                           # following Friday
        out.append(fri + pd.Timedelta(days=14))                         # third Friday after
    return out


# ======================================================================= build
def update_calendar(log: Callable = print) -> pd.DataFrame:
    rows = [(d, "fomc", "Fed rate decision (FOMC)") for d in fetch_fomc_dates(log)]
    key = os.getenv("FRED_API_KEY", "").strip()
    if key:
        for sid, (typ, title) in RELEASES.items():
            try:
                ds = fetch_release_dates(sid, key)
                rows += [(d, typ, title) for d in ds]
                log(f"  {title}: {len(ds)} dates")
            except Exception as e:
                log(f"  {title}: {e}")
            time.sleep(0.5)
    else:
        log("  No FRED_API_KEY in .env: using estimated jobs-report dates and skipping CPI/GDP/PCE/PPI/retail. "
            "Get a free key at https://fred.stlouisfed.org/docs/api/api_key.html")
        end = (pd.Timestamp.now() + pd.DateOffset(months=6)).strftime("%Y-%m-%d")
        rows += [(d, "jobs_report", "Jobs report (estimated date)")
                 for d in jobs_report_rule(config.HISTORY_START, end)]
    df = pd.DataFrame(rows, columns=["date", "type", "title"])
    df = df[df["date"] >= pd.Timestamp(config.HISTORY_START)].drop_duplicates(["date", "type"]).sort_values("date")
    df["date"] = df["date"].dt.strftime("%Y-%m-%d")
    df.to_csv(CAL_FILE, index=False)
    log(f"Economic calendar: {len(df):,} scheduled releases saved")
    return df


def load_calendar() -> pd.DataFrame:
    if not CAL_FILE.exists():
        return pd.DataFrame(columns=["date", "type", "ticker", "title", "notes"])
    df = pd.read_csv(CAL_FILE, dtype=str)
    df["date"] = pd.to_datetime(df["date"])
    df["ticker"] = ""
    df["notes"] = "scheduled"
    return df[["date", "type", "ticker", "title", "notes"]]
