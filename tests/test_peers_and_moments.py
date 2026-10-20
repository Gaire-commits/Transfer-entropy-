"""Point-in-time peers, the moment geometry, and the input batches: no look-ahead anywhere."""
import numpy as np
import torch

from heat.dataset import Gather, build_moments, split_by_date
from heat.peers import monthly_peers
from heat.synthetic import synthetic_panel


def _setup(**kw):
    panel = synthetic_panel(n_days=120, **kw)
    return panel, monthly_peers(panel, k=4, peer_days=60)


def test_correlated_peers_are_cluster_mates_and_random_are_not_self():
    panel, peers = _setup()
    cluster = np.arange(len(panel.symbols)) // 5
    p = peers.correlated[0]
    assert (cluster[p] == cluster[:, None]).mean() > 0.95
    r = peers.random[0]
    assert not (r == np.arange(len(r))[:, None]).any()
    assert all(len(set(row)) == 4 for row in r)


def test_peers_use_only_days_before_the_month():
    panel, peers = _setup()
    d0 = peers.month_start_day[0]
    later = panel.day >= d0
    shuffled = panel.returns.copy()
    shuffled[later] = np.random.default_rng(1).permutation(shuffled[later])  # scramble the month and after
    panel.returns = shuffled
    again = monthly_peers(panel, k=4, peer_days=60)
    assert (again.correlated[0] == peers.correlated[0]).all()


def test_moment_geometry():
    panel, peers = _setup()
    m = build_moments(panel, peers, lookback=12, stride=6, lag=1, holds=(6, 48))
    bars = panel.bar[m.t]
    assert bars.min() >= 12 and ((bars - 12) % 6 == 0).all()
    assert (peers.month_of_day[m.day] >= 0).all()
    # Hold label = sum of the stock's returns over bars t+2 .. t+1+h, inside the day.
    i = np.flatnonzero(np.isfinite(m.y[:, 0]))[5]
    t, s = m.t[i], m.s[i]
    assert np.isclose(m.y[i, 0], panel.returns[t + 2: t + 8, s].sum(), atol=1e-6)
    # A 4-hour hold from a late signal does not fit the day: NaN, never next-day returns.
    late = bars > 77 - 1 - 48
    assert np.isnan(m.y[late, 1]).all()


def test_inputs_never_see_the_future():
    panel, peers = _setup()
    m = build_moments(panel, peers, 12, 6, 1, (6,))
    g = Gather(panel, peers, 12, 1e-3)
    i = len(m) // 2
    t, s, mo = (torch.tensor([int(a[i])]) for a in (m.t, m.s, m.month))
    before = g(t, s, mo, "peers").clone()
    g.R[int(t) + 1:] = 123.0  # everything after the signal bar, for every stock
    assert torch.equal(g(t, s, mo, "peers"), before)
    g.R[int(t)] = 123.0  # the signal bar itself is the newest input
    assert (g(t, s, mo, "peers")[0, -1] == 123.0).all()


def test_variants_differ_only_in_peer_channels():
    panel, peers = _setup()
    m = build_moments(panel, peers, 12, 6, 1, (6,))
    g = Gather(panel, peers, 12, 1e-3)
    t, s, mo = (torch.from_numpy(a[:300]).long() for a in (m.t, m.s, m.month))
    own, cor, ran = (g(t, s, mo, v) for v in ("own", "peers", "random"))
    assert torch.equal(own[:, :, 0], cor[:, :, 0]) and torch.equal(cor[:, :, 0], ran[:, :, 0])
    assert (own[:, :, 1:] == 0).all() and not torch.equal(cor[:, :, 1:], ran[:, :, 1:])


def test_split_is_chronological_with_embargo():
    panel, peers = _setup()
    m = build_moments(panel, peers, 12, 6, 1, (6,))
    d = panel.days
    tr, va, te = split_by_date(m, panel, d[80], d[100], embargo_days=1)
    assert tr.day.max() == 80 and va.day.min() == 82 and va.day.max() == 100 and te.day.min() == 102
