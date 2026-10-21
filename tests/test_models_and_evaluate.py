"""Model shapes, the book's accounting, false-discovery control and the registered verdict."""
import numpy as np
import pandas as pd
import pytest
import torch

from heat.evaluate import benjamini_hochberg, book_positions, screen_pairs, verdicts
from heat.models import make_model


@pytest.mark.parametrize("arch", ["bilstm", "resnet1d", "transformer"])
def test_models_map_batches_to_one_output_per_hold(arch):
    m = make_model(arch, channels=6, holds=4, lookback=12, size=16)
    assert m(torch.randn(8, 12, 6)).shape == (8, 4)


def test_unknown_arch_rejected():
    with pytest.raises(ValueError):
        make_model("gru", 6, 4, 12, 16)


def test_book_longs_top_and_shorts_bottom_quintile():
    t = np.repeat([0, 1], 25)
    pred = np.tile(np.arange(25, dtype=float), 2)
    y = (pred - 12) / 100  # the prediction is right, so long and short both make money
    pos, ret = book_positions(pred, y, t, q=0.2, min_stocks=20)
    assert pos.sum() == 2 * 10  # 5 long + 5 short per moment
    # longs earn +0.08..+0.12, shorts earn -(-0.12..-0.08); 1.0 per moment, two moments
    assert (ret[pos] > 0).all() and np.isclose(ret[pos].sum(), 2.0)


def test_book_skips_thin_moments():
    pos, _ = book_positions(np.arange(10.0), np.arange(10.0), np.zeros(10, int), q=0.2, min_stocks=20)
    assert not pos.any()


def test_benjamini_hochberg():
    p = [0.001, 0.008, 0.039, 0.041, 0.042, 0.06, 0.074, 0.205, 0.212, 0.216]
    assert benjamini_hochberg(p, 0.05).tolist() == [True, True] + [False] * 8


def _cell(**over):
    base = dict(net_bps_peers_ci=(0.5, 2.0), net_bps_peers_minus_own_ci=(0.2, 1.0),
                G_peers_minus_random_ci=(1e-4, 1e-3), G_peers_ci=(1e-4, 1e-3),
                net_bps_peers_first_half=1.0, net_bps_peers_second_half=1.0, p_net_peers=0.001)
    base.update(over)
    return base


def test_verdict_needs_every_check():
    t = verdicts(pd.DataFrame([_cell(), _cell(net_bps_peers_minus_own_ci=(-0.1, 1.0)),
                               _cell(net_bps_peers_second_half=-0.5)]))
    assert t["verdict"].tolist()[0] == "EDGE"
    assert "beats_no_peers" in t["verdict"][1] and "both_halves_positive" in t["verdict"][2]


def test_heat_screen_prefers_solid_pairs_over_lucky_small_ones():
    rng = np.random.default_rng(0)
    rows = []
    for src, n, mean, noise in ((1, 2000, 0.05, 0.05), (2, 60, 0.30, 3.0), (3, 30, 0.9, 0.01)):
        rows.append(pd.DataFrame({"source": src, "target": 0, "base": 1.0,
                                  "delta": mean + rng.normal(0, noise, n)}))
    out = screen_pairs(pd.concat(rows), min_count=50, z=3.0).set_index("source")
    assert 3 not in out.index  # too few moments
    assert out.loc[2, "heat_share"] > out.loc[1, "heat_share"]  # the lucky pair has the bigger share...
    assert out.loc[1, "lcb"] > 0 > out.loc[2, "lcb"]  # ...but only the solid one clears the bar
    assert out.index[0] == 1  # and it ranks first
    assert np.isclose(out.loc[1, "se"], 0.05 / np.sqrt(2000), rtol=0.1)  # the error bar is the right size
