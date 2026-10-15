"""The noise-invariance penalty should flatten leak growth without erasing real transfer."""
import pytest

from learned_te import Config, simulate
from learned_te.models import make_predictors
from learned_te.split import time_split
from learned_te.windows import windows_from_blocks


def _fit(kind, lam, penalty="monotone", seed=1):
    sp = time_split(simulate(kind, seed=seed), gap=20)
    tr, va, te = (windows_from_blocks(b, 3, 3) for b in (sp.train, sp.val, sp.test))
    m = make_predictors(Config(model="invariant", inv_lambda=lam, inv_penalty=penalty)).fit(tr, va)
    return m, te


def test_penalty_removes_growth_with_noise_on_leak():
    plain, te = _fit("proxy_leak", 0.0)
    inv, _ = _fit("proxy_leak", 10.0)
    g0, g1 = list(plain.env_gains(te).values()), list(inv.env_gains(te).values())
    assert g0[-1] > g0[0]  # without the penalty the leak grows with noise
    assert g1[-1] - g1[0] < 0.5 * (g0[-1] - g0[0])  # with it, growth is at least halved
    assert inv.te(te) < plain.te(te)


def test_monotone_penalty_keeps_real_transfer():
    plain, te = _fit("coupled", 0.0)
    inv, _ = _fit("coupled", 10.0)
    assert inv.te(te) > 0.9 * plain.te(te)


def test_frozen_refit_does_not_need_val():
    m, te = _fit("independent", 10.0)
    clone = m.refit_frozen(te)  # any windows; only the frozen epoch count is used
    assert clone.epochs == m.epochs and clone.f_xy is not m.f_xy


def test_unknown_penalty_rejected():
    with pytest.raises(ValueError):
        make_predictors(Config(model="invariant", inv_penalty="l2"))
