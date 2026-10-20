"""Market-data helpers, tested offline (no network, no keys, no Drive)."""
import json

import numpy as np
import pandas as pd
import pytest

from learned_te import Config, report, run, simulate
from learned_te.market import (day_sessions, fetch_bars, last_sessions, load_cached_closes, market_returns,
                               session_grid)
from learned_te.multiple import benjamini_hochberg
from learned_te.split import day_split
from learned_te.windows import windows_from_blocks

FULL, EARLY, NEXT = "2025-07-02", "2025-07-03", "2025-07-07"  # 2025-07-03 closed at 13:00


def _bars(day, symbol, close_at="16:00", seed=0, drop=()):
    """One symbol's 5-minute bars for a day, in the thesis cache format (UTC MultiIndex)."""
    ts = pd.date_range(f"{day} 09:30", f"{day} {close_at}", freq="5min", inclusive="left",
                       tz="America/New_York")
    ts = ts.delete(list(drop))
    close = 100 * np.exp(np.cumsum(np.random.default_rng(seed).normal(0, 1e-3, len(ts))))
    idx = pd.MultiIndex.from_arrays([[symbol] * len(ts), ts.tz_convert("UTC")], names=["symbol", "timestamp"])
    return pd.DataFrame({"close": close, "volume": 1.0}, index=idx)


def _write_cache(root, symbol, frames, feed="sip"):
    folder = root / "5Min"
    folder.mkdir(parents=True, exist_ok=True)
    pd.concat(frames).to_parquet(folder / f"{symbol}.parquet")
    (folder / f"{symbol}.meta.json").write_text(json.dumps({"feed": feed}))


@pytest.fixture
def cache(tmp_path):
    for i, sym in enumerate(["AAA", "BBB", "SPY"]):
        drop = (10,) if sym == "BBB" else ()  # BBB misses the 10:20 bar on the full day
        _write_cache(tmp_path, sym, [_bars(FULL, sym, seed=i, drop=drop),
                                     _bars(EARLY, sym, close_at="13:00", seed=10 + i),
                                     _bars(NEXT, sym, seed=20 + i)])  # an early close is never the last day
    return tmp_path


def test_load_cached_closes_reads_thesis_format(cache):
    closes = load_cached_closes(cache, ["AAA", "BBB", "SPY"])
    assert list(closes.columns) == ["AAA", "BBB", "SPY"]
    assert str(closes.index.tz) == "America/New_York"
    assert closes["BBB"].isna().sum() == 1


def test_load_refuses_other_feed(tmp_path):
    _write_cache(tmp_path, "AAA", [_bars(FULL, "AAA")], feed="iex")
    with pytest.raises(ValueError, match="iex"):
        load_cached_closes(tmp_path, ["AAA"])


def test_session_grid_keeps_gaps_and_early_close(cache):
    grid = session_grid(load_cached_closes(cache, ["AAA", "BBB"]))
    days = grid.groupby(grid.index.tz_localize(None).normalize()).size()
    assert days.tolist() == [78, 42, 78]  # full day 9:30-15:55, early close 9:30-12:55
    gap = grid.loc[pd.Timestamp(f"{FULL} 10:20", tz="America/New_York")]
    assert np.isnan(gap["BBB"]) and np.isfinite(gap["AAA"])  # gap restored, not shifted


def test_unfinished_last_day_is_dropped(tmp_path):
    for sym in ["AAA", "BBB"]:
        _write_cache(tmp_path, sym, [_bars(FULL, sym), _bars("2025-07-07", sym, close_at="11:00")])
    grid = session_grid(load_cached_closes(tmp_path, ["AAA", "BBB"]))
    assert grid.index.tz_localize(None).normalize().unique().tolist() == [pd.Timestamp(FULL)]


def test_day_sessions_no_overnight_and_gaps_propagate(cache):
    s = day_sessions(load_cached_closes(cache, ["AAA", "BBB", "SPY"]), "AAA", "BBB", refs=["SPY"])
    assert list(s) == [FULL, EARLY, NEXT]
    assert s[FULL].shape == (77, 3) and s[EARLY].shape == (41, 3)
    # BBB's missing 10:20 bar makes the returns into and out of it NaN, nothing else.
    assert np.isnan(s[FULL][:, 1]).sum() == 2 and np.isfinite(s[FULL][:, [0, 2]]).all()


def test_market_is_equal_weighted_mean_and_cached(cache, tmp_path):
    syms = ["AAA", "BBB", "SPY"]
    m = market_returns(cache, symbols=syms, min_count=2, cache_file=tmp_path / "out" / "m.parquet")
    s = day_sessions(load_cached_closes(cache, syms), "AAA", "BBB", refs=["SPY"])[FULL]
    expected = np.nanmean(s, axis=1)  # thesis: mean over the stocks that have a bar
    got = m[m.index.tz_localize(None).normalize() == pd.Timestamp(FULL)].to_numpy()[1:]
    assert np.allclose(got, expected)
    again = market_returns("/nonexistent", cache_file=tmp_path / "out" / "m.parquet")
    assert np.allclose(again.to_numpy(), m.to_numpy(), equal_nan=True)  # read back, not recomputed


def test_market_needs_min_count(cache):
    m = market_returns(cache, symbols=["AAA", "BBB", "SPY"], min_count=3)
    assert m.isna().sum() > m.groupby(m.index.tz_localize(None).normalize()).ngroups  # BBB's gap


def test_day_sessions_appends_market_column(cache):
    syms = ["AAA", "BBB", "SPY"]
    m = market_returns(cache, symbols=syms, min_count=2)
    s = day_sessions(load_cached_closes(cache, ["AAA", "BBB"]), "AAA", "BBB", market=m)[FULL]
    assert s.shape == (77, 3) and np.isfinite(s[:, 2]).all()


def test_windows_never_touch_a_gap():
    block = np.arange(40, dtype=float)[:, None].repeat(2, axis=1)
    block[20, 1] = np.nan
    w = windows_from_blocks([block], 3, 3)
    assert np.isfinite(w.yh).all() and np.isfinite(w.xh).all() and np.isfinite(w.target).all()
    assert len(w) == (40 - 4 + 1) - 4  # the 4 windows containing index 20 are gone


def test_day_split_is_chronological_with_embargo():
    sessions = {f"2025-07-{d:02d}": np.full((5, 2), d, float) for d in range(1, 21)}
    sp = day_split(sessions, embargo=1)
    first = lambda blocks: [int(b[0, 0]) for b in blocks]
    tr, va, te = first(sp.train), first(sp.val), first(sp.test)
    assert max(tr) < min(va) - 1 and max(va) < min(te) - 1  # one embargo day between sets
    assert te == [17, 18, 19, 20]


def test_benjamini_hochberg():
    p = [0.001, 0.008, 0.039, 0.041, 0.042, 0.06, 0.074, 0.205, 0.212, 0.216]
    assert benjamini_hochberg(p, q=0.05).tolist() == [True, True] + [False] * 8
    assert not benjamini_hochberg([0.5, 0.9]).any()


def test_last_sessions_holds_out_most_recent():
    s = {f"2025-06-{d:02d}": None for d in range(1, 11)}
    assert last_sessions(s, 0.2) == ["2025-06-09", "2025-06-10"]


def test_stock_data_without_keys_fails_clearly(monkeypatch):
    monkeypatch.delenv("APCA_API_KEY_ID", raising=False)
    monkeypatch.delenv("APCA_API_SECRET_KEY", raising=False)
    with pytest.raises(RuntimeError, match="needs Alpaca keys"):
        fetch_bars(["AAPL"], "2025-01-01", "2025-02-01")


def _day_like_sessions(n_days=40, bars=77, with_gaps=False, kind="coupled"):
    out = {}
    for d in range(n_days):
        s = simulate(kind, n=bars, seed=d)
        if with_gaps:
            s[np.random.default_rng(d).integers(0, bars, 2), 1] = np.nan
        out[f"2025-{1 + d // 28:02d}-{1 + d % 28:02d}"] = s
    return out


def test_day_split_finds_coupling_in_77_bar_days_with_gaps():
    r = run(sessions=_day_like_sessions(with_gaps=True), cfg=Config(n_surrogates=30, n_bootstrap=200))
    assert r.significant and r.ci[0] > 0


def test_day_split_no_false_positive():
    r = run(sessions=_day_like_sessions(kind="independent"), cfg=Config(n_surrogates=30, n_bootstrap=200))
    assert not r.significant


def test_report_accepts_day_sessions():
    rep = report(_day_like_sessions(), Config(n_surrogates=20, n_bootstrap=100))
    assert rep["forward"].significant and rep["delta_te"] > 0
