"""The thesis bars -> one panel of market-relative 5-minute returns, (bars x stocks).

Source: the SIP cache entropy-gate-pipeline wrote to Google Drive,
`<data_dir>/5Min/<SYM>.parquet` (index (symbol, timestamp), a `close` column) with a
`.meta.json` naming the feed. Rules carried over from the thesis (`src/data.py`, `src/peers.py`):

- Regular hours, on a fixed 5-minute grid per trading day; a missing bar stays NaN, so it never
  shifts later bars. A day whose last bar starts before 13:00 is an early close. A final day whose
  last bar starts before 15:30 is still in progress and is dropped.
- Returns are within the day: a day's first bar has none (no overnight return).
- Relative to the market: each bar's log return minus the average over the stocks that traded in
  both bars (`relative_returns` in the thesis).
- Feeds never mix: loading refuses a cache whose metadata names another feed.
"""
from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd

EASTERN = "America/New_York"
OPEN = pd.Timedelta(hours=9, minutes=30)
FULL_CLOSE = pd.Timedelta(hours=16)
EARLY_CLOSE = pd.Timedelta(hours=13)
IN_PROGRESS = pd.Timedelta(hours=15, minutes=30)
BAR = pd.Timedelta(minutes=5)
THESIS_SIP_DIR = "/content/drive/MyDrive/entropy-gate-pipeline-data/data/bars_sip"


@dataclass
class Panel:
    """Market-relative returns on the grid. Row i is the bar starting at `times[i]`."""
    times: pd.DatetimeIndex  # US/Eastern, naive
    day: np.ndarray  # (T,) day number, 0..n_days-1
    bar: np.ndarray  # (T,) bar of the day, 0 = 09:30
    symbols: list
    returns: np.ndarray  # (T, S) float32, NaN where a stock lacks this bar or the previous one

    @property
    def days(self) -> pd.DatetimeIndex:
        return pd.DatetimeIndex(self.times[np.r_[0, np.flatnonzero(np.diff(self.day)) + 1]].normalize())

    def save(self, path):
        Path(path).parent.mkdir(parents=True, exist_ok=True)
        np.savez(path, times=self.times.asi8, day=self.day, bar=self.bar,
                 symbols=np.array(self.symbols), returns=self.returns)

    @classmethod
    def load(cls, path):
        z = np.load(path, allow_pickle=False)
        return cls(pd.DatetimeIndex(z["times"]), z["day"], z["bar"], z["symbols"].tolist(), z["returns"])


def read_closes(data_dir, symbol, timeframe="5Min", expect_feed="sip") -> pd.Series:
    """One symbol's closes, indexed by naive US/Eastern bar start, regular hours only."""
    path = Path(data_dir) / timeframe / f"{symbol}.parquet"
    meta = path.with_suffix(".meta.json")
    if expect_feed and meta.exists():
        feed = json.loads(meta.read_text()).get("feed")
        if feed and feed != expect_feed:
            raise ValueError(f"{path} holds {feed!r} bars, expected {expect_feed!r}")
    df = pd.read_parquet(path, columns=["close"])
    ts = df.index.get_level_values("timestamp") if isinstance(df.index, pd.MultiIndex) else df.index
    ts = pd.DatetimeIndex(ts)
    ts = (ts.tz_convert(EASTERN) if ts.tz is not None else ts.tz_localize("UTC").tz_convert(EASTERN)).tz_localize(None)
    s = pd.Series(df["close"].to_numpy(np.float64), index=ts)
    s = s[~s.index.duplicated(keep="last")]
    tod = s.index - s.index.normalize()
    return s[(tod >= OPEN) & (tod < FULL_CLOSE)].sort_index()


def build_grid(last_bar_by_day: pd.Series) -> pd.DatetimeIndex:
    """The complete 5-minute grid for these days, given each day's latest bar start."""
    last_bar_by_day = last_bar_by_day.sort_index()
    if (last_bar_by_day.iloc[-1] - last_bar_by_day.index[-1]) < IN_PROGRESS:
        last_bar_by_day = last_bar_by_day.iloc[:-1]
    early = ((last_bar_by_day - last_bar_by_day.index) < EARLY_CLOSE).to_numpy()
    offsets = pd.timedelta_range(OPEN, FULL_CLOSE - BAR, freq=BAR)
    grid = last_bar_by_day.index.values[:, None] + offsets.values[None, :]
    keep = ~(early[:, None] & (offsets.values[None, :] >= EARLY_CLOSE.to_timedelta64()))
    return pd.DatetimeIndex(grid[keep].ravel())


def relative_returns(log_close: np.ndarray, day: np.ndarray) -> np.ndarray:
    """Within-day log returns minus the average over stocks with both bars (the thesis market)."""
    r = np.full_like(log_close, np.nan)
    r[1:] = log_close[1:] - log_close[:-1]
    r[np.r_[True, day[1:] != day[:-1]]] = np.nan  # first bar of each day: no overnight return
    finite = np.isfinite(r)
    count = finite.sum(axis=1, keepdims=True)
    market = np.divide(np.where(finite, r, 0.0).sum(axis=1, keepdims=True), count,
                       out=np.full((len(r), 1), np.nan, dtype=r.dtype), where=count > 0)
    return np.where(finite, r - market, np.nan).astype(np.float32)


def build_panel(data_dir, symbols, timeframe="5Min", expect_feed="sip", progress=None) -> Panel:
    """Read every symbol once, put all on one grid, and return market-relative returns."""
    series, last = {}, None
    for i, sym in enumerate(symbols):
        path = Path(data_dir) / timeframe / f"{sym}.parquet"
        if not path.exists():
            continue
        s = read_closes(data_dir, sym, timeframe, expect_feed)
        if s.empty:
            continue
        series[sym] = s
        lb = pd.Series(s.index, index=s.index).groupby(s.index.normalize()).max()
        last = lb if last is None else pd.concat([last, lb]).groupby(level=0).max()
        if progress and (i + 1) % 50 == 0:
            progress(f"read {i + 1}/{len(symbols)}")
    if not series:
        raise FileNotFoundError(f"no {timeframe} parquet files for these symbols under {data_dir}")
    grid = build_grid(last)
    names = list(series)
    log_close = np.full((len(grid), len(names)), np.nan, dtype=np.float64)
    for j, sym in enumerate(names):
        log_close[:, j] = np.log(series[sym].reindex(grid).to_numpy())
    days = grid.normalize()
    day = pd.factorize(days)[0].astype(np.int32)
    bar = ((grid - days) - OPEN) // BAR
    return Panel(grid, day, np.asarray(bar, dtype=np.int16), names, relative_returns(log_close, day))


def load_universe(path="universe/sp500.csv") -> pd.DataFrame:
    return pd.read_csv(path)
