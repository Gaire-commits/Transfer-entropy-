"""Synthetic panels with known answers, for the positive and negative controls.

Stocks come in clusters that share a factor, so correlation picks cluster-mates as peers. With
`planted`, each cluster's first stock is a leader: the others take `beta` of its idiosyncratic move
`lags` bars later (after the 1-bar execution lag, so it is tradeable). Without it, nothing transfers.
"""
import numpy as np
import pandas as pd

from .data import BAR, OPEN, Panel


def synthetic_panel(n_days=160, n_clusters=8, cluster_size=5, beta=0.5, lags=(2, 3), planted=True,
                    gap_share=0.005, start="2020-01-02", seed=0, cross=None) -> Panel:
    """cross=+1 or -1: instead, each cluster's leader drives the NEXT cluster's stocks, with that sign.
    Leader and followers then don't move together at the same moment, so correlation can't pick the
    leader as a peer; only a lagged measure such as transfer entropy can."""
    rng = np.random.default_rng(seed)
    S, B = n_clusters * cluster_size, 78
    days = pd.bdate_range(start, periods=n_days)
    offsets = pd.timedelta_range(OPEN, periods=B, freq=BAR)
    times = pd.DatetimeIndex((days.values[:, None] + offsets.values[None, :]).ravel())
    T = len(times)
    cluster = np.repeat(np.arange(n_clusters), cluster_size)
    market = rng.normal(0, 1e-3, (T, 1))
    factor = rng.normal(0, 1e-3, (T, n_clusters))[:, cluster]
    idio = rng.normal(0, 1e-3, (T, S))
    r = 0.6 * market + 0.8 * factor + idio
    bar = np.tile(np.arange(B), n_days)
    if cross is not None:
        for c in range(n_clusters):
            lead, nxt = c * cluster_size, ((c + 1) % n_clusters) * cluster_size
            for f in range(nxt + 1, nxt + cluster_size):
                for lag in lags:
                    r[lag:, f] += cross * beta * idio[:-lag, lead] * (bar[lag:] >= lag)
    elif planted:
        leaders = np.arange(n_clusters) * cluster_size
        for c, lead in enumerate(leaders):
            for f in range(lead + 1, lead + cluster_size):
                for lag in lags:
                    r[lag:, f] += beta * idio[:-lag, lead] * (bar[lag:] >= lag)  # within the day
    r[bar == 0] = np.nan  # no overnight return
    r[rng.random((T, S)) < gap_share] = np.nan
    finite = np.isfinite(r)
    mkt = np.where(finite, r, 0).sum(1, keepdims=True) / np.maximum(finite.sum(1, keepdims=True), 1)
    rel = np.where(finite, r - mkt, np.nan).astype(np.float32)
    return Panel(times, np.repeat(np.arange(n_days), B).astype(np.int32), bar.astype(np.int16),
                 [f"S{i:03d}" for i in range(S)], rel)
