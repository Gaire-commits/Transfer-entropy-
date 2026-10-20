"""Correlated and random peers, chosen point-in-time (the thesis E12 rule, `src/peers.py`).

On the first trading day of each month, each stock's `k` stocks whose market-relative 5-minute
returns moved most closely with its own over the previous `peer_days` trading days. Only past days
are used, and the choice is kept for the month. A stock is eligible if it has a return in at least
`min_share` of those bars. The control draws `k` random eligible peers per stock, redrawn monthly.
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from .data import Panel


def pick_peers(history: np.ndarray, k: int, min_share: float = 0.8):
    """Each stock's `k` most correlated other stocks over `history` (bars, stocks), best first.
    -1 where a stock is not eligible. Returns (peers, correlations, eligible)."""
    n = history.shape[1]
    finite = np.isfinite(history)
    ok = finite.mean(axis=0) >= min_share
    x = np.where(finite, history, 0.0)
    x = x - x.mean(axis=0)
    sd = np.sqrt((x ** 2).sum(axis=0))
    z = np.divide(x, sd, out=np.zeros_like(x), where=sd > 0)
    corr = z.T @ z
    corr[:, ~ok] = -np.inf
    np.fill_diagonal(corr, -np.inf)
    peers = np.full((n, k), -1)
    rho = np.full((n, k), np.nan)
    if ok.sum() > k:
        idx = np.flatnonzero(ok)
        order = np.argsort(-corr[idx], axis=1)[:, :k]
        peers[idx] = order
        rho[idx] = np.take_along_axis(corr[idx], order, axis=1)
    return peers, rho, ok


def random_peers(ok: np.ndarray, k: int, rng: np.random.Generator) -> np.ndarray:
    n = len(ok)
    out = np.full((n, k), -1)
    pool = np.flatnonzero(ok)
    if len(pool) <= k:
        return out
    for i in pool:
        out[i] = rng.choice(pool[pool != i], size=k, replace=False)
    return out


@dataclass
class PeerTable:
    """`correlated[m]` and `random[m]` are (stocks, k) for month m; `month_of_day[d]` maps a day
    to its month (-1 before the first month with enough history)."""
    month_start_day: np.ndarray
    month_of_day: np.ndarray
    correlated: np.ndarray  # (months, stocks, k)
    random: np.ndarray
    rho: np.ndarray


def monthly_peers(panel: Panel, k=5, peer_days=60, min_share=0.8, seed=0) -> PeerTable:
    rng = np.random.default_rng(seed)
    days = panel.days
    n_days = len(days)
    first_of_month = np.flatnonzero(np.r_[True, days.month[1:] != days.month[:-1]])
    first_of_month = first_of_month[first_of_month >= peer_days]
    day_rows = np.r_[0, np.flatnonzero(np.diff(panel.day)) + 1, len(panel.day)]

    corr, rand, rhos = [], [], []
    for d0 in first_of_month:
        hist = panel.returns[day_rows[d0 - peer_days]: day_rows[d0]]  # strictly before the month
        p, rho, ok = pick_peers(hist, k, min_share)
        corr.append(p)
        rhos.append(rho)
        rand.append(random_peers(ok, k, rng))
    month_of_day = np.full(n_days, -1)
    for m, d0 in enumerate(first_of_month):
        end = first_of_month[m + 1] if m + 1 < len(first_of_month) else n_days
        month_of_day[d0:end] = m
    S = panel.returns.shape[1]
    stack = lambda xs, fill: np.stack(xs) if xs else np.full((0, S, k), fill)
    return PeerTable(first_of_month, month_of_day, stack(corr, -1), stack(rand, -1), stack(rhos, np.nan))
