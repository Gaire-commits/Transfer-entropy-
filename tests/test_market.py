"""Market-data helpers, tested offline (no network, no keys)."""
import numpy as np
import pandas as pd
import pytest

from learned_te import Config, report, simulate
from learned_te.market import fetch_bars, intraday_sessions, last_sessions, log_returns, to_array


def _minute_prices(days=("2025-06-02", "2025-06-03", "2025-06-04"), bars=390, seed=0):
    rng = np.random.default_rng(seed)
    idx = pd.DatetimeIndex([], tz="UTC")
    for d in days:
        open_ = pd.Timestamp(f"{d} 09:30", tz="America/New_York")
        idx = idx.append(pd.date_range(open_, periods=bars, freq="1min").tz_convert("UTC"))
    walk = np.exp(np.cumsum(rng.normal(0, 1e-3, (len(idx), 3)), axis=0)) * 100
    return pd.DataFrame(walk, index=idx, columns=["AAA", "BBB", "SPY"])


def test_to_array_orders_columns_x_y_refs():
    r = log_returns(_minute_prices())
    a = to_array(r, x="BBB", y="AAA", refs=["SPY"])
    assert np.allclose(a[:, 0], r["BBB"]) and np.allclose(a[:, 1], r["AAA"]) and np.allclose(a[:, 2], r["SPY"])


def test_intraday_sessions_never_span_overnight():
    p = _minute_prices()
    p.iloc[390:] *= 1.5  # a big overnight jump between day 1 and day 2
    sessions = intraday_sessions(p, x="AAA", y="BBB", refs=["SPY"])
    assert list(sessions) == ["2025-06-02", "2025-06-03", "2025-06-04"]
    for s in sessions.values():
        assert s.shape == (389, 3)  # first return of each day dropped
        assert np.abs(s).max() < 0.1  # the jump is not in any return


def test_short_days_are_skipped():
    p = pd.concat([_minute_prices(("2025-06-02",)), _minute_prices(("2025-06-03",), bars=100)])
    assert list(intraday_sessions(p, "AAA", "BBB")) == ["2025-06-02"]


def test_last_sessions_holds_out_most_recent():
    s = {f"2025-06-{d:02d}": None for d in range(1, 11)}
    assert last_sessions(s, 0.2) == ["2025-06-09", "2025-06-10"]


def test_stock_data_without_keys_fails_clearly(monkeypatch):
    monkeypatch.delenv("APCA_API_KEY_ID", raising=False)
    monkeypatch.delenv("APCA_API_SECRET_KEY", raising=False)
    with pytest.raises(RuntimeError, match="needs Alpaca keys"):
        fetch_bars(["AAPL"], "2025-01-01", "2025-02-01")


def test_default_min_bars_drops_half_days():
    full = _minute_prices(("2025-07-02",))
    half = _minute_prices(("2025-07-03",), bars=210)  # 9:30-13:00 early close
    assert list(intraday_sessions(pd.concat([full, half]), "AAA", "BBB")) == ["2025-07-02"]


def test_half_day_session_is_skipped_not_fatal():
    # Regression: a 209-sample half day left a 49-sample val block and crashed report().
    sessions = {f"2025-07-{d:02d}": simulate("coupled", n=389, seed=d) for d in range(1, 13)}
    sessions["2025-07-03"] = simulate("coupled", n=209, seed=99)
    with pytest.warns(UserWarning, match="2025-07-03"):
        rep = report(sessions, Config(n_surrogates=20, n_bootstrap=100), test_ids=["2025-07-11", "2025-07-12"])
    assert rep["forward"].significant


def test_report_accepts_sessions():
    sessions = {f"day{i:02d}": simulate("coupled", n=600, seed=i) for i in range(10)}
    rep = report(sessions, Config(n_surrogates=20, n_bootstrap=100), test_ids=["day08", "day09"])
    assert rep["forward"].significant and rep["delta_te"] > 0
