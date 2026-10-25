"""The macro release calendar: official U.S. release dates from FRED, and which moments they touch.

Releases (FRED release ids, checked 2026-10-24 on fred.stlouisfed.org/release?rid=...):
  jobs  50  Employment Situation, U.S. Bureau of Labor Statistics     08:30 ET (before the open)
  cpi   10  Consumer Price Index, U.S. Bureau of Labor Statistics     08:30 ET (before the open)
  gdp   53  Gross Domestic Product, U.S. Bureau of Economic Analysis  08:30 ET (before the open)
  fomc 101  FOMC Press Release, Federal Reserve Board                 14:00 ET (during trading)

The pre-open releases move the whole day, so they are split by day. The FOMC decision lands mid-session,
so it is also split by whether a moment's hold spans 14:00. The release *times* are the agencies'
standard times; FRED gives the dates. (An unscheduled FOMC action, such as those of March 2020, can fall
at another hour; such a day still counts as an FOMC day.)

The FRED API needs a free key (fred.stlouisfed.org -> My Account -> API Keys), read from the FRED_API_KEY
environment variable or a Colab secret of that name; never put it in code.
"""
from __future__ import annotations

import json
import os
import urllib.parse
import urllib.request

import numpy as np
import pandas as pd

RELEASES = {
    "jobs": (50, "08:30", "Employment Situation (BLS)"),
    "cpi": (10, "08:30", "Consumer Price Index (BLS)"),
    "gdp": (53, "08:30", "Gross Domestic Product (BEA)"),
    "fomc": (101, "14:00", "FOMC Press Release (Federal Reserve)"),
}
API = "https://api.stlouisfed.org/fred/release/dates"


def fred_key(explicit=None):
    key = explicit or os.environ.get("FRED_API_KEY")
    if not key:
        try:
            from google.colab import userdata  # only inside Colab

            key = userdata.get("FRED_API_KEY")
        except Exception:  # noqa: BLE001
            key = None
    if not key:
        raise RuntimeError("FRED needs a free API key: create one at fred.stlouisfed.org (My Account -> API Keys), "
                           "then add it as the environment variable or Colab secret FRED_API_KEY")
    return key


def fetch_calendar(start="2016-01-01", end="2026-12-31", api_key=None, opener=None) -> pd.DataFrame:
    """One row per (date, event) with the release time, from FRED. `opener` exists for tests."""
    key = fred_key(api_key)
    opener = opener or urllib.request.urlopen
    rows = []
    for event, (rid, at, name) in RELEASES.items():
        q = urllib.parse.urlencode({"release_id": rid, "api_key": key, "file_type": "json", "limit": 10000,
                                    "realtime_start": start, "realtime_end": end,
                                    "include_release_dates_with_no_data": "false", "sort_order": "asc"})
        with opener(f"{API}?{q}") as resp:
            data = json.loads(resp.read().decode())
        for d in data.get("release_dates", []):
            rows.append({"date": d["date"], "event": event, "time_et": at, "release": name})
    cal = pd.DataFrame(rows, columns=["date", "event", "time_et", "release"])
    cal = cal[(cal["date"] >= start) & (cal["date"] <= end)].drop_duplicates(["date", "event"])
    return cal.sort_values(["date", "event"]).reset_index(drop=True)


def load_or_fetch(path, start="2016-01-01", end="2026-12-31", log=print) -> pd.DataFrame:
    """The calendar from `path` if it was fetched before, else from FRED (and saved there)."""
    if os.path.exists(path):
        cal = pd.read_csv(path, dtype=str)
        log(f"event calendar: {len(cal):,} releases loaded from {path}")
        return cal
    cal = fetch_calendar(start, end)
    os.makedirs(os.path.dirname(os.path.abspath(path)), exist_ok=True)
    cal.to_csv(path, index=False)
    log(f"event calendar: {len(cal):,} releases fetched from FRED and saved to {path}")
    return cal


def moment_masks(test, panel, calendar: pd.DataFrame, holds, lag=1):
    """{name: bool mask over the test moments}. Day masks for every event, an "any release" day mask,
    "no release" days, and for FOMC, per hold, whether the hold spans the 14:00 release."""
    dates = pd.DatetimeIndex(panel.days[test.day]).strftime("%Y-%m-%d").to_numpy()
    masks = {}
    any_day = np.zeros(len(test), bool)
    for event in RELEASES:
        on = np.isin(dates, calendar.loc[calendar["event"] == event, "date"].to_numpy())
        masks[f"{event} days"] = on
        any_day |= on
    masks["any release day"] = any_day
    masks["no release day"] = ~any_day
    fomc = masks["fomc days"]
    release_bar = (14 * 60 - (9 * 60 + 30)) // 5  # the bar that starts at 14:00
    bar = panel.bar[test.t].astype(int)
    for h in holds:
        first, last = bar + lag + 1, bar + lag + h  # the hold's bars
        masks[f"fomc, hold spans 14:00 ({h * 5} min)"] = fomc & (first <= release_bar) & (last >= release_bar)
    return masks
