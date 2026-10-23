"""Transfer entropy between stocks: the closed form is right, point-in-time, finds negative relations
with the right sign, and finds lagged links that correlation cannot."""
import numpy as np

from heat.peers import monthly_peers
from heat.synthetic import synthetic_panel
from heat.te import partial_rho, pick_te_peers, te_from_rho, window_returns


def test_partial_rho_matches_regression():
    rng = np.random.default_rng(0)
    n, S = 4000, 4
    past = rng.normal(size=(n, S))
    future = 0.3 * past + rng.normal(size=(n, S))
    future[:, 2] += -0.4 * past[:, 0]  # stock 0 -> stock 2, negative
    rho = partial_rho(past, future, np.ones(S, bool))
    # Direct: residualise future_2 and past_0 on past_2, then correlate.
    x = np.column_stack([np.ones(n), past[:, 2]])
    res = lambda v: v - x @ np.linalg.lstsq(x, v, rcond=None)[0]
    direct = np.corrcoef(res(future[:, 2]), res(past[:, 0]))[0, 1]
    assert np.isclose(rho[2, 0], direct, atol=1e-6) and rho[2, 0] < -0.3
    assert np.isnan(np.diag(rho)).all()
    assert np.isclose(te_from_rho(rho[2, 0]), -0.5 * np.log(1 - direct ** 2))


def test_window_returns_stay_inside_the_day():
    p = synthetic_panel(n_days=5, n_clusters=2)
    past, future, rows = window_returns(p, lookback=12, lag=1, hold=48)
    bars = p.bar[rows]
    assert bars.min() >= 12 and (bars + 1 + 48 <= 77).all()
    t, s = rows[3], 1
    assert np.isclose(future[3, s], np.nansum(p.returns[t + 2: t + 50, s]))
    assert np.isclose(past[3, s], np.nansum(p.returns[t - 11: t + 1, s]))


def test_te_peers_find_negative_cross_cluster_leaders_that_correlation_misses():
    p = synthetic_panel(n_days=130, n_clusters=6, cluster_size=5, cross=-1, beta=0.6, seed=4)
    t = monthly_peers(p, k=5, peer_days=60, select="te")
    m = 0
    cluster = np.arange(30) // 5
    leader_of = lambda j: ((cluster[j] - 1) % 6) * 5  # followers' leader sits in the previous cluster
    followers = [j for j in range(30) if j % 5 != 0]
    te_hit = np.mean([leader_of(j) in t.te[m, j] for j in followers])
    assert te_hit > 0.9
    # Correlation fills its first 4 slots with the stock's own cluster-mates and never ranks the leader
    # among them; it lands in the leftover 5th slot only by accident of the market-relative structure.
    assert not any(leader_of(j) in t.correlated[m, j, :4] for j in followers)
    # ...and the TE feature carries the negative sign of the relation.
    signs = [t.features["peers"][m, j, list(t.te[m, j]).index(leader_of(j)), 0] for j in followers
             if leader_of(j) in t.te[m, j]]
    assert np.mean(np.array(signs) < 0) > 0.9


def test_te_is_point_in_time():
    p = synthetic_panel(n_days=130, n_clusters=4, cross=1, seed=1)
    a = monthly_peers(p, k=3, peer_days=60, select="te")
    later = p.day >= a.month_start_day[0]
    p.returns[later] = np.random.default_rng(9).permutation(p.returns[later])  # scramble the month on
    b = monthly_peers(p, k=3, peer_days=60, select="te")
    assert (a.te[0] == b.te[0]).all() and np.allclose(a.features["peers"][0], b.features["peers"][0])


def test_pick_te_peers_ranks_by_strength_regardless_of_sign():
    rho = np.full((3, 3), np.nan)
    rho[0, 1], rho[0, 2] = 0.05, -0.20  # the negative link is the stronger one
    rho[1, 0], rho[1, 2] = 0.10, 0.01
    rho[2, 0], rho[2, 1] = -0.02, 0.03
    assert pick_te_peers(rho, 1)[:, 0].tolist() == [2, 0, 1]
