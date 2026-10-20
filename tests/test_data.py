"""Panel building from a cache in the thesis format (no Drive, no network)."""
import json

import numpy as np
import pandas as pd
import pytest

from heat.data import Panel, build_panel

FULL, EARLY, NEXT, PARTIAL = "2025-07-02", "2025-07-03", "2025-07-07", "2025-07-08"


def _write(root, sym, days, feed="sip", drop=(), seed=0):
    frames = []
    for day, close in days:
        ts = pd.date_range(f"{day} 09:30", f"{day} {close}", freq="5min", inclusive="left", tz="America/New_York")
        frames.append(pd.DataFrame({"close": 100 + np.random.default_rng(seed).normal(0, 0.1, len(ts)).cumsum()},
                                   index=pd.MultiIndex.from_arrays([[sym] * len(ts), ts.tz_convert("UTC")],
                                                                   names=["symbol", "timestamp"])))
    df = pd.concat(frames).drop(index=[df_i for df_i in drop])
    folder = root / "5Min"
    folder.mkdir(parents=True, exist_ok=True)
    df.to_parquet(folder / f"{sym}.parquet")
    (folder / f"{sym}.meta.json").write_text(json.dumps({"feed": feed}))


@pytest.fixture
def cache(tmp_path):
    days = [(FULL, "16:00"), (EARLY, "13:00"), (NEXT, "16:00"), (PARTIAL, "11:00")]
    gap = ("BBB", pd.Timestamp(f"{FULL} 10:20", tz="America/New_York").tz_convert("UTC"))
    for i, sym in enumerate(["AAA", "BBB", "CCC"]):
        _write(tmp_path, sym, days, drop=[gap] if sym == "BBB" else (), seed=i)
    return tmp_path


def test_grid_early_close_and_unfinished_day(cache):
    p = build_panel(cache, ["AAA", "BBB", "CCC"])
    per_day = pd.Series(p.day).value_counts().sort_index().tolist()
    assert per_day == [78, 42, 78]  # early close 9:30-12:55; the 11:00 day is still "in progress"
    assert p.days.strftime("%Y-%m-%d").tolist() == [FULL, EARLY, NEXT]


def test_gap_stays_a_gap_and_no_overnight_return(cache):
    p = build_panel(cache, ["AAA", "BBB", "CCC"])
    first_bars = np.flatnonzero(p.bar == 0)
    assert np.isnan(p.returns[first_bars]).all()  # no return spans the night
    row = np.flatnonzero(p.times == pd.Timestamp(f"{FULL} 10:20"))[0]
    b = p.symbols.index("BBB")
    assert np.isnan(p.returns[row, b]) and np.isnan(p.returns[row + 1, b])  # into and out of the gap
    assert np.isfinite(p.returns[row + 2, b])


def test_returns_are_market_relative(cache):
    p = build_panel(cache, ["AAA", "BBB", "CCC"])
    full = np.isfinite(p.returns).all(axis=1)
    assert np.allclose(p.returns[full].sum(axis=1), 0, atol=1e-6)


def test_refuses_other_feed(tmp_path):
    _write(tmp_path, "AAA", [(FULL, "16:00"), (NEXT, "16:00")], feed="iex")
    with pytest.raises(ValueError, match="iex"):
        build_panel(tmp_path, ["AAA"])


def test_panel_round_trip(cache, tmp_path):
    p = build_panel(cache, ["AAA", "BBB", "CCC"])
    p.save(tmp_path / "out" / "panel.npz")
    q = Panel.load(tmp_path / "out" / "panel.npz")
    assert q.symbols == p.symbols and (q.times == p.times).all()
    assert np.allclose(q.returns, p.returns, equal_nan=True)
