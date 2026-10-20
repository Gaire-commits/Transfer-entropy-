"""Alpaca market data -> arrays the pipeline accepts.

    pip install alpaca-py

Stock data needs API keys (free paper-trading keys work with feed="iex"). They are
read, in order, from the arguments, the env vars APCA_API_KEY_ID / APCA_API_SECRET_KEY,
or Colab secrets with the same names. Crypto bars need no keys.

Daily bars -> one continuous series:
    prices = fetch_bars(["AAPL", "MSFT", "SPY"], "2016-01-01", "2025-12-31", "1Day")
    data = to_array(log_returns(prices), x="AAPL", y="MSFT", refs=["SPY"])
    report(data, Config(ref_cols=(2,)))

Intraday bars -> one session per trading day (no window spans the overnight gap):
    prices = fetch_bars(["AAPL", "MSFT", "SPY"], "2025-06-01", "2025-09-30", "1Min")
    sessions = intraday_sessions(prices, x="AAPL", y="MSFT", refs=["SPY"])
    report(sessions, Config(ref_cols=(2,)), test_ids=last_sessions(sessions, 0.2))
"""
import os
import re

import numpy as np
import pandas as pd

KEY_ENV, SECRET_ENV = "APCA_API_KEY_ID", "APCA_API_SECRET_KEY"


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
        raise ValueError(f"timeframe like '1Min', '5Min', '1Hour', '1Day'; got {spec!r}")
    return TimeFrame(int(m.group(1)), TimeFrameUnit(m.group(2)))


def fetch_bars(symbols, start, end, timeframe="1Day", asset_class="stock", feed="iex",
               adjustment="all", api_key=None, secret_key=None):
    """Close prices as a wide DataFrame (UTC timestamps x symbols), rows aligned on
    timestamps where every symbol traded."""
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
    return prices[symbols].dropna().sort_index()


def log_returns(prices):
    return np.log(prices).diff().dropna()


def to_array(returns, x, y, refs=()):
    """Columns [X, Y, refs...] -- the order the pipeline expects."""
    return returns[[x, y, *refs]].to_numpy(float)


def intraday_sessions(prices, x, y, refs=(), tz="America/New_York", min_bars=300):
    """{trading date: (n, c) array of within-day log returns}. Each day's first return
    is dropped, so no return spans the overnight gap. Days with fewer than `min_bars`
    returns are skipped; the default 300 drops half days (about 210 one-minute bars)."""
    local = prices.tz_convert(tz) if prices.index.tz is not None else prices
    sessions = {}
    for day, chunk in local.groupby(local.index.date):
        r = log_returns(chunk)
        if len(r) >= min_bars:
            sessions[str(day)] = to_array(r, x, y, refs)
    if not sessions:
        raise ValueError(f"no trading day had at least {min_bars} bars")
    return sessions


def last_sessions(sessions, frac=0.2):
    """IDs of the most recent sessions, to hold out as test (keys sort by date)."""
    keys = sorted(sessions)
    return keys[-max(1, int(round(frac * len(keys)))):]
