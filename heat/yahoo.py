"""Yahoo Finance daily bars -> the daily panel.

Daily adjusted closes (splits and dividends) for the universe, from Yahoo Finance via the `yfinance`
library, cached to one parquet file so the download happens once. The panel holds each stock's daily log
return relative to the market (the equal-weighted mean over the stocks with a return that day), as the
5-minute panel does. One row per trading day, and the whole series is one session: a lookback or hold
runs across days.

Limits to state in any write-up:
- Survivorship: Yahoo has no prices for companies that left the index or disappeared, so the universe is
  today's S&P 500 traced back; a stock enters the panel when its prices start.
- Data errors: a daily log return beyond +/-100% (x2.7 or /2.7 in one day) is treated as a data error and
  dropped. Real moves of that size in S&P 500 stocks are vanishingly rare; Yahoo glitches are not.
- Yahoo's terms allow personal research use; the data is not redistributed here.
"""
from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd

from .data import Panel

MAX_ABS_LOG_RETURN = 1.0


def yahoo_symbol(symbol: str) -> str:
    """Yahoo writes class shares with a dash: BRK.B -> BRK-B."""
    return symbol.replace(".", "-")


def fetch_daily(symbols, start="2000-01-01", end=None, chunk=100, download=None, log=print) -> pd.DataFrame:
    """Adjusted daily closes, (dates x symbols), NaN before a stock's first price. `download` exists for
    tests; by default yfinance.download."""
    if download is None:
        import yfinance as yf

        download = yf.download
    symbols = list(symbols)
    if end is None:  # yfinance's end is exclusive: this leaves out today, which may still be trading
        end = pd.Timestamp.today().normalize().strftime("%Y-%m-%d")
    to_yahoo = {s: yahoo_symbol(s) for s in symbols}
    frames, missing = [], []
    for i in range(0, len(symbols), chunk):
        batch = symbols[i:i + chunk]
        tickers = [to_yahoo[s] for s in batch]
        data = download(tickers, start=start, end=end, auto_adjust=True, actions=False, progress=False,
                        threads=True, group_by="column")
        close = data["Close"] if isinstance(data.columns, pd.MultiIndex) else data[["Close"]]
        if not isinstance(data.columns, pd.MultiIndex):
            close.columns = tickers
        close = close.rename(columns={to_yahoo[s]: s for s in batch})
        got = [s for s in batch if s in close.columns and close[s].notna().any()]
        missing += [s for s in batch if s not in got]
        frames.append(close[got])
        log(f"yahoo: {min(i + chunk, len(symbols))}/{len(symbols)} symbols")
    closes = pd.concat(frames, axis=1).sort_index()
    closes.index = pd.DatetimeIndex(closes.index).tz_localize(None).normalize()
    if missing:
        log(f"yahoo: no data for {len(missing)} symbols: {', '.join(missing[:12])}{' ...' if len(missing) > 12 else ''}")
    return closes


def cached_daily(path, symbols, start="2000-01-01", end=None, log=print) -> pd.DataFrame:
    """The closes from `path` if downloaded before, else from Yahoo (and saved there)."""
    path = Path(path)
    if path.exists():
        closes = pd.read_parquet(path)
        log(f"yahoo: {closes.shape[1]} symbols x {closes.shape[0]:,} days loaded from {path.name}")
        return closes
    closes = fetch_daily(symbols, start, end, log=log)
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(".tmp.parquet")
    closes.to_parquet(tmp)
    tmp.replace(path)
    log(f"yahoo: saved {closes.shape[1]} symbols x {closes.shape[0]:,} days to {path}")
    return closes


def daily_panel(closes: pd.DataFrame, min_stocks=20) -> Panel:
    """Daily log returns relative to the equal-weighted market, one row per trading day (days with fewer
    than `min_stocks` prices are dropped). One session: holds and lookbacks run across days."""
    closes = closes.sort_index()
    closes = closes[closes.notna().sum(axis=1) >= min_stocks]
    logc = np.log(closes.where(closes > 0).to_numpy(np.float64))
    r = np.full_like(logc, np.nan)
    r[1:] = logc[1:] - logc[:-1]
    r[np.abs(r) > MAX_ABS_LOG_RETURN] = np.nan  # Yahoo glitches, not market moves
    finite = np.isfinite(r)
    count = finite.sum(axis=1, keepdims=True)
    market = np.divide(np.where(finite, r, 0.0).sum(axis=1, keepdims=True), count,
                       out=np.full((len(r), 1), np.nan), where=count > 0)
    rel = np.where(finite, r - market, np.nan).astype(np.float32)
    T = len(closes)
    times = pd.DatetimeIndex(closes.index) + pd.Timedelta(hours=16)  # the close
    return Panel(times, np.arange(T, dtype=np.int32), np.arange(T, dtype=np.int64), list(closes.columns), rel,
                 session=np.zeros(T, dtype=np.int32))
