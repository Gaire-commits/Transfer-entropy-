"""Market data -> arrays the pipeline accepts, following the thesis data conventions.

Primary source: the 5-minute SIP bars already cached by entropy-gate-pipeline on Google
Drive (503 S&P 500 stocks, 2016-01-04 to 2026-10-12). Nothing is downloaded:

    market = market_returns(THESIS_SIP_DIR, cache_file=".../market_5min.parquet")  # once
    closes = load_cached_closes(THESIS_SIP_DIR, ["NVDA", "AMD"])
    sessions = day_sessions(closes, x="NVDA", y="AMD", market=market)
    report(sessions, Config(k=6, l=6, ref_cols=(2,)))

Same rules as the thesis (`src/data.py` there): regular hours only, a fixed 5-minute grid per
trading day with missing bars kept as gaps (NaN) so a gap never shifts later bars, early closes
end at 13:00, and a final day still in progress is dropped. Returns are within the day only, so
none spans the overnight gap. The market is the thesis's: the equal-weighted mean 5-minute return
across the universe at each bar (`leadlag.py`, `peers.py`), passed as a reference column and
regressed out, so the analysis is market-neutral. (The SIP cache holds stocks only, no SPY.)

Fallback: `fetch_bars` downloads from Alpaca (`pip install alpaca-py`), by default the same
5-minute SIP bars (the free plan serves SIP older than 15 minutes). Keys are read from the env
vars APCA_API_KEY_ID / APCA_API_SECRET_KEY or Colab secrets, never from code. Crypto needs none.
"""
import json
import os
import re
import warnings
from pathlib import Path

import numpy as np
import pandas as pd

KEY_ENV, SECRET_ENV = "APCA_API_KEY_ID", "APCA_API_SECRET_KEY"
EASTERN = "America/New_York"
OPEN = pd.Timedelta(hours=9, minutes=30)
FULL_CLOSE = pd.Timedelta(hours=16)
EARLY_CLOSE = pd.Timedelta(hours=13)
THESIS_SIP_DIR = "/content/drive/MyDrive/entropy-gate-pipeline-data/data/bars_sip"


# ---- cached thesis bars ------------------------------------------------------
def _eastern(ts):
    ts = pd.DatetimeIndex(ts)
    return ts.tz_convert(EASTERN) if ts.tz is not None else ts.tz_localize("UTC").tz_convert(EASTERN)


def load_cached_closes(data_dir, symbols, timeframe="5Min", expect_feed="sip"):
    """Close prices from the thesis parquet cache (<data_dir>/<timeframe>/<SYM>.parquet), as a
    wide frame on US/Eastern timestamps, NaN where a symbol has no bar. Refuses a cache whose
    metadata says it holds another feed, mirroring the thesis rule that feeds never mix."""
    cols = {}
    for sym in symbols:
        path = Path(data_dir) / timeframe / f"{sym}.parquet"
        if not path.exists():
            raise FileNotFoundError(f"{path} not found (is Google Drive mounted?)")
        meta = path.with_suffix(".meta.json")
        if expect_feed and meta.exists():
            feed = json.loads(meta.read_text()).get("feed")
            if feed and feed != expect_feed:
                raise ValueError(f"{path} holds {feed!r} bars, expected {expect_feed!r}")
        df = pd.read_parquet(path)
        ts = df.index.get_level_values("timestamp") if isinstance(df.index, pd.MultiIndex) else df.index
        s = pd.Series(df["close"].to_numpy(float), index=_eastern(ts))
        cols[sym] = s[~s.index.duplicated(keep="last")]
    return pd.DataFrame(cols).sort_index()


# ---- the fixed session grid --------------------------------------------------
def session_grid(closes, minutes=5):
    """Reindex onto a complete bar grid per trading day (regular hours), NaN where a bar is
    missing. A day whose last bar starts before 13:00 is an early close. A final day whose last
    bar starts before 15:30 is treated as still in progress and dropped."""
    idx = _eastern(closes.index)
    closes = closes.set_axis(idx)
    t = idx.tz_localize(None)
    tod = t - t.normalize()
    closes = closes[(tod >= OPEN) & (tod < FULL_CLOSE)]
    t = closes.index.tz_localize(None)
    days = t.normalize()

    last_bar = pd.Series(t, index=t).groupby(days).max()
    if (last_bar.iloc[-1] - last_bar.index[-1]) < pd.Timedelta(hours=15, minutes=30):
        last_bar = last_bar.iloc[:-1]
    early = (last_bar - last_bar.index) < EARLY_CLOSE

    step = pd.Timedelta(minutes=minutes)
    offsets = pd.timedelta_range(OPEN, FULL_CLOSE - step, freq=step)
    grid = last_bar.index.values[:, None] + offsets.values[None, :]
    keep = ~(early.values[:, None] & (offsets.values[None, :] >= EARLY_CLOSE.to_timedelta64()))
    grid_index = pd.DatetimeIndex(grid[keep].ravel())
    return closes.set_axis(t).reindex(grid_index).set_axis(grid_index.tz_localize(EASTERN))


def completeness(grid):
    """Share of grid slots that have a real bar, per symbol."""
    return grid.notna().mean()


def _within_day_returns(grid):
    """Log returns on the grid, NaN at each day's first bar (no overnight return) and next
    to every missing bar."""
    r = np.log(grid).diff()
    first = grid.index.tz_localize(None).normalize()
    r[np.r_[True, first[1:] != first[:-1]]] = np.nan
    return r


def market_returns(data_dir, symbols=None, timeframe="5Min", min_count=20, cache_file=None):
    """The thesis market: equal-weighted mean within-day 5-minute log return across every cached
    stock with a bar at that slot (slots with fewer than `min_count` stocks are NaN).

    Reads every parquet file in <data_dir>/<timeframe> unless `symbols` is given -- a few minutes
    for the 503-stock universe -- so pass `cache_file` to save the result and reuse it."""
    if cache_file and Path(cache_file).exists():
        cached = pd.read_parquet(cache_file)["market"]
        return cached.set_axis(_eastern(cached.index))
    folder = Path(data_dir) / timeframe
    symbols = symbols or sorted(p.name[: -len(".parquet")] for p in folder.glob("*.parquet"))
    total = count = None
    for sym in symbols:
        r = _within_day_returns(session_grid(load_cached_closes(data_dir, [sym], timeframe)))[sym]
        total = r.fillna(0.0) if total is None else total.add(r.fillna(0.0), fill_value=0.0)
        n = r.notna().astype(float)
        count = n if count is None else count.add(n, fill_value=0.0)
    market = (total / count).where(count >= min_count).rename("market")
    if cache_file:
        Path(cache_file).parent.mkdir(parents=True, exist_ok=True)
        market.to_frame().to_parquet(cache_file)
    return market


def day_sessions(closes, x, y, refs=(), market=None, minutes=5, min_valid=30):
    """{"YYYY-MM-DD": (n, c) array of within-day log returns}, columns [X, Y, refs..., market].

    `market` is an optional return series (e.g. from `market_returns`), appended as the last
    column; with refs=() it is column 2, so use Config(ref_cols=(2,)).

    Each day's first bar has no return (no overnight return); a missing bar makes the returns
    on either side of it NaN, and the pipeline drops every window that touches one. Days with
    fewer than `min_valid` complete rows are skipped."""
    returns = _within_day_returns(session_grid(closes[[x, y, *refs]], minutes))
    if market is not None:
        returns["market"] = market.reindex(returns.index)
    sessions, skipped = {}, []
    for day, chunk in returns.groupby(returns.index.tz_localize(None).normalize()):
        r = chunk.to_numpy(float)[1:]  # the first bar of a day never has a return
        if np.isfinite(r).all(axis=1).sum() >= min_valid:
            sessions[str(day.date())] = r
        else:
            skipped.append(str(day.date()))
    if skipped:
        warnings.warn(f"skipped {len(skipped)} day(s) with under {min_valid} complete bars: "
                      + ", ".join(skipped[:5]) + (" ..." if len(skipped) > 5 else ""))
    if not sessions:
        raise ValueError("no trading day had enough complete bars")
    return sessions


def last_sessions(sessions, frac=0.2):
    """IDs of the most recent sessions, to hold out as test (keys sort by date)."""
    keys = sorted(sessions)
    return keys[-max(1, int(round(frac * len(keys)))):]


# ---- continuous series (daily bars, crypto) ----------------------------------
def log_returns(prices):
    return np.log(prices).diff().dropna()


def to_array(returns, x, y, refs=()):
    """Columns [X, Y, refs...] -- the order the pipeline expects."""
    return returns[[x, y, *refs]].to_numpy(float)


# ---- Alpaca download (fallback) ----------------------------------------------
def _credentials(api_key, secret_key):
    api_key = api_key or os.environ.get(KEY_ENV)
    secret_key = secret_key or os.environ.get(SECRET_ENV)
    if not (api_key and secret_key):
        try:
            from google.colab import userdata  # only exists inside Colab

            api_key = api_key or userdata.get(KEY_ENV)
            secret_key = secret_key or userdata.get(SECRET_ENV)
        except Exception:
            pass
    return api_key, secret_key


def _timeframe(spec):
    from alpaca.data.timeframe import TimeFrame, TimeFrameUnit

    m = re.fullmatch(r"(\d+)(Min|Hour|Day|Week|Month)", spec)
    if not m:
        raise ValueError(f"timeframe like '5Min', '1Hour', '1Day'; got {spec!r}")
    return TimeFrame(int(m.group(1)), TimeFrameUnit(m.group(2)))


def fetch_bars(symbols, start, end, timeframe="5Min", asset_class="stock", feed="sip",
               adjustment="all", api_key=None, secret_key=None):
    """Close prices from Alpaca as a wide frame (US/Eastern timestamps x symbols), NaN where a
    symbol has no bar. Pass the result to `day_sessions` (intraday) or `log_returns` (daily)."""
    symbols = list(symbols)
    tf = _timeframe(timeframe)
    start, end = pd.Timestamp(start).to_pydatetime(), pd.Timestamp(end).to_pydatetime()
    key, secret = _credentials(api_key, secret_key)

    if asset_class == "crypto":
        from alpaca.data.historical import CryptoHistoricalDataClient
        from alpaca.data.requests import CryptoBarsRequest

        client = CryptoHistoricalDataClient(key, secret)
        bars = client.get_crypto_bars(CryptoBarsRequest(
            symbol_or_symbols=symbols, timeframe=tf, start=start, end=end))
    elif asset_class == "stock":
        from alpaca.data.enums import Adjustment, DataFeed
        from alpaca.data.historical import StockHistoricalDataClient
        from alpaca.data.requests import StockBarsRequest

        if not (key and secret):
            raise RuntimeError(f"stock data needs Alpaca keys: set {KEY_ENV} and {SECRET_ENV} "
                               "(env vars, or Colab secrets), or pass api_key/secret_key")
        client = StockHistoricalDataClient(key, secret)
        bars = client.get_stock_bars(StockBarsRequest(
            symbol_or_symbols=symbols, timeframe=tf, start=start, end=end,
            feed=DataFeed(feed), adjustment=Adjustment(adjustment)))
    else:
        raise ValueError("asset_class must be 'stock' or 'crypto'")

    df = bars.df
    if df.empty:
        raise ValueError("Alpaca returned no bars for this request")
    prices = df["close"].unstack(level="symbol")
    missing = [s for s in symbols if s not in prices.columns]
    if missing:
        raise ValueError(f"no bars returned for {missing}")
    prices = prices[symbols].sort_index()
    return prices.set_axis(_eastern(prices.index))
