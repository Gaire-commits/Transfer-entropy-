"""heat_v2 controls: hidden NEGATIVE leaders in another cluster (invisible to correlation) must be found by
transfer-entropy peers, as an edge that also beats correlation peers; with nothing planted, no edge."""
import copy

import pytest

from heat import config
from heat.run import run_experiment
from heat.synthetic import synthetic_panel


def _run(**kw):
    cfg = copy.deepcopy(config.load("configs/heat_v2.yaml"))
    panel = synthetic_panel(n_days=480, n_clusters=8, cluster_size=5, seed=3, **kw)
    d = panel.days
    cfg["split"].update(train_end=str(d[389].date()), val_end=str(d[429].date()))
    cfg["train"].update(archs=["bilstm"], size=32, batch=1024, samples_per_epoch=30000, max_epochs=6, patience=2,
                        val_samples=15000)
    cfg["evaluation"].update(bootstrap=300, heat_samples=0)
    return run_experiment(panel, cfg, log=lambda m: None)[0]


@pytest.fixture(scope="module")
def negative():
    return _run(cross=-1, beta=0.6)


def test_negative_leaders_are_an_edge_found_by_te_not_correlation(negative):
    row = negative[negative["hold"] == "30 min"].iloc[0]
    assert row["verdict"] == "EDGE" and row["beats_correlation_peers"]
    assert row["G_peers"] > 0.005 and abs(row["G_corr"]) < 0.002 and abs(row["G_random"]) < 0.002


def test_nothing_planted_no_edge():
    t = _run(planted=False)
    assert (t["verdict"] != "EDGE").all()
