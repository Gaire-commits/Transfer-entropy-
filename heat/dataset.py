"""Moments (stock, signal bar) and their inputs and labels, in the thesis E5/E12 geometry.

A moment is a stock at a signal bar t, one every `stride` bars. Input: the last `lookback` bars of
market-relative returns of the stock and of its `k` peers (bars t-lookback+1..t). Trade: enter
`lag` bars after the signal and hold h bars, so the label for hold h is the sum of the stock's
returns over bars t+lag+1..t+lag+h. Intraday means intraday: the lookback, the lag and the hold
all sit inside one trading day, or that label is NaN. Every bar of the stock's own lookback and of
the hold must have a trade; a missing peer bar is read as 0 (no move relative to the market).
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd
import torch

from .data import Panel
from .peers import PeerTable

VARIANTS = ("own", "peers", "random")  # no peers / the experiment's peers / random peers (the control)


def variants_for(peers_cfg: dict) -> tuple:
    """heat_v1: own / peers (by correlation) / random. heat_v2 (select: te): own / peers (by transfer
    entropy) / random / corr (heat_v1's correlation peers, as a further control)."""
    return VARIANTS + (("corr",) if peers_cfg.get("select", "corr") == "te" else ())


@dataclass
class Moments:
    t: np.ndarray  # (n,) panel row of the signal bar
    s: np.ndarray  # (n,) stock column
    day: np.ndarray  # (n,) day number
    month: np.ndarray  # (n,) peer month
    y: np.ndarray  # (n, H) float32 hold returns, NaN where the hold does not fit or a bar is missing

    def __len__(self):
        return len(self.t)

    def subset(self, mask):
        return Moments(self.t[mask], self.s[mask], self.day[mask], self.month[mask], self.y[mask])


def build_moments(panel: Panel, peers: PeerTable, lookback=12, stride=6, lag=1, holds=(6, 12, 24, 48)) -> Moments:
    R = panel.returns
    T, S = R.shape
    finite = np.isfinite(R)
    day, bar = panel.day.astype(np.int64), panel.bar.astype(np.int64)
    last_bar = pd.Series(bar).groupby(day).transform("max").to_numpy()

    rows = np.flatnonzero((bar >= lookback) & ((bar - lookback) % stride == 0)
                          & (bar + lag + min(holds) <= last_bar) & (peers.month_of_day[day] >= 0))
    # Every bar of the own lookback must have a trade (cumulative counts make this one subtraction).
    cnt = np.vstack([np.zeros((1, S), np.int64), np.cumsum(finite, axis=0)])
    own_ok = (cnt[rows + 1] - cnt[rows + 1 - lookback]) == lookback

    csum = np.vstack([np.zeros((1, S)), np.cumsum(np.where(finite, R, 0.0), axis=0)])
    y = np.full((len(rows), S, len(holds)), np.nan, np.float32)
    for j, h in enumerate(holds):
        a, b = rows + lag + 1, rows + lag + h  # first and last bar of the hold
        fits = b < T
        b_c = np.minimum(b, T - 1)
        fits &= (day[b_c] == day[rows]) & (bar[b_c] <= last_bar[rows])
        full = (cnt[b_c + 1] - cnt[a]) == h
        val = (csum[b_c + 1] - csum[a]).astype(np.float32)
        y[:, :, j] = np.where(fits[:, None] & full, val, np.nan)

    keep = own_ok & np.isfinite(y).any(axis=2)
    r_idx, s_idx = np.nonzero(keep)
    t = rows[r_idx]
    return Moments(t=t, s=s_idx, day=day[t], month=peers.month_of_day[day[t]], y=y[r_idx, s_idx])


def split_by_date(m: Moments, panel: Panel, train_end, val_end, embargo_days=1, test_end=None):
    """Chronological train / val / test by trading day, with `embargo_days` dropped before val
    and before test. Labels never leave their day, so nothing crosses a boundary. `test_end`
    (optional) stops the test days there, so a check run can stay inside the training years."""
    dates = panel.days
    d_train = np.searchsorted(dates, pd.Timestamp(train_end), side="right")  # last train day + 1
    d_val = np.searchsorted(dates, pd.Timestamp(val_end), side="right")
    train = m.subset(m.day < d_train)
    val = m.subset((m.day >= d_train + embargo_days) & (m.day < d_val))
    d_end = np.searchsorted(dates, pd.Timestamp(test_end), side="right") if test_end else len(dates)
    test = m.subset((m.day >= d_val + embargo_days) & (m.day < d_end))
    return train, val, test


class Gather:
    """Builds input batches on the device straight from the panel: (B, lookback, channels).

    Channels: the stock's and its k peers' returns (1 + k). With TE features (heat_v2), also:
    - each peer's signed TE strength into the stock at each hold (k x holds), constant over the lookback;
    - the TE-weighted peer signal per hold: sum over peers of strength x that peer's return, at every bar
      (holds channels). A positive relation pushes it with the peer, a negative one against, wherever
      the peer sits among the k slots, so the model doesn't have to discover that interaction itself.
    The no-peer model gets zeros there, so every variant has the same input shape."""

    def __init__(self, panel: Panel, peers: PeerTable, lookback: int, scale: float, device="cpu", feat_scale=None):
        R = np.nan_to_num(panel.returns / scale, nan=0.0).astype(np.float32)
        S = R.shape[1]
        self.R = torch.from_numpy(np.hstack([R, np.zeros((len(R), 1), np.float32)])).to(device)  # col S = zeros
        fix = lambda p: torch.from_numpy(np.where(p >= 0, p, S)).long().to(device)
        names = ("peers", "random", "corr")
        self.tables = {n: fix(peers.table(n)) for n in names}
        self.k = peers.correlated.shape[2]
        self.none = S
        self.offsets = torch.arange(lookback - 1, -1, -1, device=device)
        self.device = device
        self.lookback = lookback
        self.feats = {}
        if peers.features:
            fs = feat_scale or 1.0
            self.feats = {n: torch.from_numpy(peers.features[n] / fs).float().to(device) for n in names}
        self.n_holds = len(peers.feature_holds) if self.feats else 0
        self.n_feat = self.k * self.n_holds
        self.raw = {}  # unscaled strengths, for the weighted signal
        if peers.features:
            self.raw = {n: torch.from_numpy(peers.features[n]).float().to(device) for n in names}
        self.channels = 1 + self.k + self.n_feat + self.n_holds

    def features(self, s, month, variant):
        """(B, lookback, n_feat) signed TE strengths of the variant's peers, or None without features."""
        if not self.n_feat:
            return None
        if variant == "own":
            return torch.zeros(len(s), self.lookback, self.n_feat, device=self.device)
        f = self.feats[variant][month, s].reshape(len(s), 1, self.n_feat)
        return f.expand(len(s), self.lookback, self.n_feat)

    def stocks(self, s, month, variant):
        if variant == "own":
            return torch.cat([s[:, None], torch.full((len(s), self.k), self.none, device=self.device)], 1)
        return torch.cat([s[:, None], self.tables[variant][month, s]], 1)

    def weighted_signal(self, x, s, month, variant):
        """(B, lookback, holds): sum_k strength_k(hold) x peer_k's return, bar by bar."""
        if variant == "own":
            return torch.zeros(len(s), self.lookback, self.n_holds, device=self.device)
        w = self.raw[variant][month, s]  # (B, k, holds)
        return torch.einsum("blk,bkh->blh", x[:, :, 1:1 + self.k], w)

    def __call__(self, t, s, month, variant):
        times = t[:, None] - self.offsets[None, :]  # (B, L), oldest first
        x = self.R[times[:, :, None], self.stocks(s, month, variant)[:, None, :]]
        f = self.features(s, month, variant)
        if f is None:
            return x
        return torch.cat([x, f, self.weighted_signal(x, s, month, variant)], dim=2)


def to_device(m: Moments, device):
    as_long = lambda a: torch.from_numpy(np.asarray(a, np.int64)).to(device)
    return as_long(m.t), as_long(m.s), as_long(m.month), torch.from_numpy(m.y).to(device)
