"""Transfer entropy between every pair of stocks, point-in-time, on the timescale the models predict.

For a source stock i and a target stock j, at signal bar t (as in the moments):
  x_i = i's return over the last hour (bars t-11..t), the same window the models read;
  x_j = j's own return over the last hour;
  y_j = j's return over the hold h after a one-bar lag (bars t+2..t+1+h), h = 30 min ... 4 h.
TE(i -> j, h) is the information x_i carries about y_j beyond what x_j already carries (so j's own
reversal is in the baseline). With Gaussian returns this is exact and closed-form:

    rho = partial correlation of (y_j, x_i) given x_j,    TE = -1/2 ln(1 - rho^2)   (nats).

TE has no sign; rho does. rho > 0: when i rose, j rose after it (a positive relation); rho < 0: j fell
(a negative relation). Selection uses TE, so both kinds count equally; the sign goes to the models as a
feature. Every month is estimated from the previous `peer_days` trading days only.
"""
from __future__ import annotations

import numpy as np

from .data import Panel


def window_returns(panel: Panel, lookback=12, lag=1, hold=6):
    """(past, future, rows): for every bar t where the last-hour window and the hold both fit inside
    t's day, each stock's last-hour return and hold return (NaN -> 0: no move against the market)."""
    R = np.nan_to_num(panel.returns.astype(np.float64))
    T = len(R)
    csum = np.vstack([np.zeros((1, R.shape[1])), np.cumsum(R, axis=0)])
    day, bar = panel.day.astype(np.int64), panel.bar.astype(np.int64)
    last = np.zeros(T, np.int64)
    np.maximum.at(last, day, bar)  # last[d] = last bar index of day d
    rows = np.flatnonzero((bar >= lookback) & (bar + lag + hold <= last[day]))
    past = csum[rows + 1] - csum[rows + 1 - lookback]
    future = csum[rows + lag + hold + 1] - csum[rows + lag + 1]
    return past, future, rows


def partial_rho(past: np.ndarray, future: np.ndarray, ok: np.ndarray) -> np.ndarray:
    """(S, S) matrix rho[j, i] = partial correlation of (future_j, past_i | past_j). NaN where i or j is
    not eligible, and on the diagonal."""
    def z(a):
        a = a - a.mean(axis=0)
        sd = a.std(axis=0)
        return np.divide(a, sd, out=np.zeros_like(a), where=sd > 0)

    n = len(past)
    X, Y = z(past), z(future)
    c_yx = Y.T @ X / n  # c_yx[j, i] = corr(future_j, past_i)
    c_xx = X.T @ X / n  # c_xx[j, i] = corr(past_j, past_i)
    r = np.diag(c_yx)[:, None]  # corr(future_j, past_j): j's own carry-over (e.g. its reversal)
    denom = np.sqrt(np.clip((1 - r ** 2) * (1 - c_xx ** 2), 1e-12, None))
    rho = np.clip((c_yx - r * c_xx) / denom, -0.999999, 0.999999)
    rho[~ok, :] = np.nan
    rho[:, ~ok] = np.nan
    np.fill_diagonal(rho, np.nan)
    return rho


def te_from_rho(rho):
    return -0.5 * np.log1p(-np.square(rho))


def monthly_rho(panel: Panel, month_start_day, peer_days=60, lookback=12, lag=1, holds=(6, 12, 24, 48),
                min_share=0.8):
    """{hold: (months, S, S) partial correlations}, each month from the `peer_days` days before it."""
    finite = np.isfinite(panel.returns)
    day_rows = np.r_[0, np.flatnonzero(np.diff(panel.day)) + 1, len(panel.day)]
    per_hold = {h: window_returns(panel, lookback, lag, h) for h in holds}
    out = {h: np.full((len(month_start_day), panel.returns.shape[1], panel.returns.shape[1]), np.nan, np.float32)
           for h in holds}
    for m, d0 in enumerate(month_start_day):
        lo, hi = day_rows[d0 - peer_days], day_rows[d0]  # bars strictly before the month
        ok = finite[lo:hi].mean(axis=0) >= min_share
        for h, (past, future, rows) in per_hold.items():
            sel = (rows >= lo) & (rows + lag + h < hi)  # the hold must also end before the month
            if sel.sum() > 30:
                out[h][m] = partial_rho(past[sel], future[sel], ok)
    return out


def pick_te_peers(rho_select: np.ndarray, k: int) -> np.ndarray:
    """Each stock's k sources with the highest TE into it (positive or negative), best first; -1 where
    there are not enough eligible sources."""
    te = te_from_rho(rho_select)
    te = np.where(np.isfinite(te), te, -np.inf)
    S = len(te)
    peers = np.full((S, k), -1)
    for j in range(S):
        if np.isfinite(te[j]).sum() >= k:
            order = np.argsort(-te[j])[:k]
            if np.all(np.isfinite(te[j, order])):
                peers[j] = order
    return peers
