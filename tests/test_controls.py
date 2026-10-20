"""Positive and negative controls: the whole experiment on synthetic panels with known answers.
A planted leader must show up as an edge and in the heat map; with nothing planted, no edge."""
import copy

import pytest

from heat import config
from heat.run import run_experiment
from heat.synthetic import synthetic_panel


def _run(planted):
    cfg = copy.deepcopy(config.load("configs/heat_v1.yaml"))
    panel = synthetic_panel(n_days=230, planted=planted, seed=3)
    d = panel.days
    cfg["split"].update(train_end=str(d[149].date()), val_end=str(d[189].date()))
    cfg["train"].update(archs=["resnet1d"], size=32, batch=1024, samples_per_epoch=30000, max_epochs=6,
                        patience=2, val_samples=15000)
    cfg["evaluation"].update(bootstrap=300, heat_samples=15000)
    return run_experiment(panel, cfg, log=lambda m: None)


@pytest.fixture(scope="module")
def planted():
    return _run(True)


def test_planted_leader_is_an_edge_at_30_minutes(planted):
    table, _ = planted
    row = table[table["hold"] == "30 min"].iloc[0]
    assert row["verdict"] == "EDGE" and row["heat_transfer"]


def test_heat_map_finds_the_planted_leaders(planted):
    _, heat = planted
    top = heat[heat["hold"] == "30 min"].head(10)
    leaders = {f"S{i:03d}" for i in range(0, 40, 5)}
    assert top["source"].isin(leaders).mean() >= 0.8


def test_nothing_planted_no_edge():
    table, _ = _run(False)
    assert (table["verdict"] != "EDGE").all() and not table["heat_transfer"].any()
