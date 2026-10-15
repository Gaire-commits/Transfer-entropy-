"""Guards for the leakage guarantees of the pipeline."""
import numpy as np
import pytest

from learned_te import Config, run, simulate
from learned_te.preprocess import CausalPreprocessor
from learned_te.split import time_split, window_span
from learned_te.windows import make_windows, windows_from_blocks


def test_split_has_gaps_and_no_overlap():
    s = np.arange(1000, dtype=float)[:, None].repeat(2, axis=1)
    sp = time_split(s, gap=25)
    tr, va, te = sp.train[0][:, 0], sp.val[0][:, 0], sp.test[0][:, 0]
    assert va[0] - tr[-1] > 25 and te[0] - va[-1] > 25


@pytest.mark.parametrize("k,l,u", [(1, 1, 1), (3, 5, 1), (5, 2, 3)])
def test_windows_stay_inside_block(k, l, u):
    # Encode the time index in the values; every window must use only past indices.
    block = np.arange(200, dtype=float)[:, None].repeat(2, axis=1)
    yh, xh, tgt = make_windows(block, k, l, u)
    assert yh.min() >= 0 and xh.min() >= 0 and tgt.max() <= 199
    assert np.all(yh.max(axis=1) < tgt) and np.all(xh.max(axis=1) <= tgt - u)
    assert len(tgt) == 200 - window_span(k, l, u) + 1


def test_windows_never_cross_blocks():
    a = np.zeros((100, 2))
    b = np.full((100, 2), 1000.0)
    w = windows_from_blocks([a, b], 3, 3)
    for bid, val in [(0, 0.0), (1, 1000.0)]:
        m = w.block_id == bid
        assert np.all(w.yh[m] == val) and np.all(w.xh[m] == val) and np.all(w.target[m] == val)


def test_preprocessing_uses_train_statistics_only():
    rng = np.random.default_rng(0)
    train = [rng.normal(0, 1, (500, 2))]
    p1 = CausalPreprocessor().fit(train)
    p2 = CausalPreprocessor().fit(train)
    p2.transform([rng.normal(50, 9, (500, 2))])  # transforming other data changes nothing
    assert np.allclose(p1.mean, p2.mean) and np.allclose(p1.std, p2.std)


def test_transform_before_fit_raises():
    with pytest.raises(RuntimeError):
        CausalPreprocessor().transform([np.zeros((10, 2))])


def test_filter_is_causal():
    rng = np.random.default_rng(1)
    s = rng.normal(0, 1, (2000, 2))
    prep = CausalPreprocessor(band=(0.01, 0.2), fs=1.0, filter_warmup=0).fit([s])
    changed = s.copy()
    changed[1500:] += 100.0  # perturb only the future
    a, b = prep.transform([s])[0], prep.transform([changed])[0]
    assert np.allclose(a[:1500], b[:1500])


def test_independent_series_not_significant_and_coupled_is():
    cfg = Config(n_surrogates=50, n_bootstrap=100)
    assert not run(simulate("independent", seed=3), cfg).significant
    r = run(simulate("coupled", seed=3), cfg)
    assert r.significant and r.ci[0] > 0


def test_over_penalised_model_is_not_a_false_positive():
    # lambda=100 makes f_XY worse than f_Y; that must not read as transfer.
    cfg = Config(model="invariant", inv_lambda=100.0, n_surrogates=20, n_bootstrap=100)
    r = run(simulate("independent", seed=1), cfg)
    assert not r.significant and r.te_eff <= 0.005


def test_null_bias_is_small_under_coupling():
    # A refit-based null must not go strongly negative when X really matters.
    r = run(simulate("coupled", seed=4), Config(n_surrogates=50, n_bootstrap=100))
    assert abs(r.te - r.te_eff) < 0.01
