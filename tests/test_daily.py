"""heat_v3 on daily bars: Yahoo parsing, the daily panel, holds across days, the embargo, block intervals,
and the controls (hidden negative leaders found by TE peers as an edge; nothing planted, no edge)."""
import copy

import numpy as np
import pandas as pd
import pytest

from heat import config
from heat.dataset import build_moments, split_by_date
from heat.evaluate import day_weights
from heat.peers import monthly_peers
from heat.run import run_experiment
from heat.synthetic import synthetic_daily
from heat.yahoo import daily_panel, fetch_daily, yahoo_symbol


def test_yahoo_symbols_and_download_parsing():
    assert yahoo_symbol("BRK.B") == "BRK-B" and yahoo_symbol("AAPL") == "AAPL"
    idx = pd.bdate_range("2020-01-01", periods=5)

    def download(tickers, **kw):
        assert kw["auto_adjust"] and kw["end"]  # adjusted prices, and today left out
        cols = pd.MultiIndex.from_product([["Close", "Open"], tickers])
        df = pd.DataFrame(np.arange(5 * len(cols), dtype=float).reshape(5, -1) + 1, index=idx, columns=cols)
        df[("Close", "GONE")] = np.nan if "GONE" in tickers else None
        return df

    closes = fetch_daily(["AAPL", "BRK.B", "GONE"], download=download, log=lambda m: None)
    assert list(closes.columns) == ["AAPL", "BRK.B"]  # mapped back; the symbol with no data dropped


def test_daily_panel_is_market_relative_and_drops_glitches():
    idx = pd.bdate_range("2020-01-01", periods=30)
    c = pd.DataFrame(np.exp(np.cumsum(np.random.default_rng(0).normal(0, 0.01, (30, 25)), axis=0)) * 50,
                     index=idx, columns=[f"X{i}" for i in range(25)])
    c.iloc[10, 3] *= 5.0  # a Yahoo glitch: x5 for one day
    p = daily_panel(c)
    assert p.returns.shape == (30, 25) and (p.session == 0).all() and (np.diff(p.day) == 1).all()
    full = np.isfinite(p.returns).all(axis=1)
    assert np.allclose(np.nansum(p.returns[full], axis=1), 0, atol=1e-5)
    assert np.isnan(p.returns[10, 3]) and np.isnan(p.returns[11, 3])  # the jump in and out is dropped


def test_daily_holds_run_across_days_and_never_past_the_end():
    p = synthetic_daily(n_days=400, n_clusters=4, planted=False)
    peers = monthly_peers(p, 3, 60)
    m = build_moments(p, peers, lookback=20, stride=1, lag=1, holds=(1, 10))
    i = len(m) // 2
    t, s = m.t[i], m.s[i]
    assert np.isclose(m.y[i, 1], np.nansum(p.returns[t + 2: t + 12, s]), atol=1e-6)  # days t+2 .. t+11
    late = m.t > len(p.day) - 1 - 1 - 10
    assert np.isnan(m.y[late, 1]).all()


def test_embargo_keeps_labels_out_of_the_next_set():
    p = synthetic_daily(n_days=600, n_clusters=4, planted=False)
    peers = monthly_peers(p, 3, 60)
    m = build_moments(p, peers, 20, 1, 1, (1, 10))
    d = p.days
    tr, va, te = split_by_date(m, p, d[300], d[450], embargo_days=11)
    assert tr.day.max() + 1 + 10 < va.day.min() and va.day.max() + 1 + 10 < te.day.min()


def test_block_bootstrap_widens_intervals_for_autocorrelated_days():
    rng = np.random.default_rng(1)
    x = np.convolve(rng.normal(size=600), np.ones(10) / 10, mode="same")  # strongly autocorrelated by day
    spread = lambda block: np.std(day_weights(600, 1000, np.random.default_rng(2), block) @ x / 600)
    assert spread(21) > 1.8 * spread(1)


def _daily_cfg(panel):
    cfg = copy.deepcopy(config.load("configs/heat_v3.yaml"))
    d = panel.days
    cfg["peers"]["peer_days"] = 500
    cfg["split"].update(train_end=str(d[1799].date()), val_end=str(d[2299].date()))
    cfg["train"].update(archs=["bilstm"], size=32, batch=1024, samples_per_epoch=40000, max_epochs=6, patience=2,
                        val_samples=15000)
    cfg["evaluation"].update(bootstrap=300, heat_samples=0)
    return cfg


def test_daily_negative_leaders_are_an_edge_found_by_te_not_correlation():
    p = synthetic_daily(n_days=2800, cross=-1, beta=0.5, seed=3)
    t, _ = run_experiment(p, _daily_cfg(p), log=lambda m: None)
    row = t[t["hold"] == "1 d"].iloc[0]
    assert row["verdict"] == "EDGE" and row["beats_correlation_peers"], row["verdict"]
    assert row["G_peers"] > 0.005 and abs(row["G_corr"]) < 0.002


def test_daily_nothing_planted_no_edge():
    p = synthetic_daily(n_days=2800, planted=False, seed=3)
    t, _ = run_experiment(p, _daily_cfg(p), log=lambda m: None)
    assert (t["verdict"] != "EDGE").all()
