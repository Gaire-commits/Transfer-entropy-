"""The visual report: built from a run's outputs, self-contained, and right about the numbers it shows."""
import copy
import json
import re

import numpy as np
import pandas as pd
import pytest

from heat import config
from heat.run import run_experiment
from heat.synthetic import synthetic_panel
from heat.viz import hold_label, load_results, sector_layout, write_report

SECTORS = ["Technology", "Financials", "Health Care", "Energy"]


@pytest.fixture(scope="module")
def outputs(tmp_path_factory):
    out = tmp_path_factory.mktemp("viz")
    panel = synthetic_panel(n_days=230, n_clusters=6, cluster_size=5, planted=True, seed=3)
    d = panel.days
    cfg = copy.deepcopy(config.load("configs/heat_v1.yaml"))
    cfg["split"].update(train_end=str(d[149].date()), val_end=str(d[189].date()))
    cfg["train"].update(archs=["resnet1d", "bilstm"], size=16, batch=1024, samples_per_epoch=12000, max_epochs=2,
                        patience=1, val_samples=8000)
    cfg["evaluation"].update(bootstrap=100, heat_samples=8000)
    run_experiment(panel, cfg, out, log=lambda m: None)
    panel.save(out / "panel.npz")
    syms = panel.symbols
    uni = pd.DataFrame({"symbol": syms, "name": [f"{s} Corp" for s in syms],
                        "sector": [SECTORS[(i // 5) % len(SECTORS)] for i in range(len(syms))]})
    uni.to_csv(out / "universe.csv", index=False)
    return out


def _payload(html):
    return json.loads(re.search(r'<script id="data" type="application/json">(.*?)</script>', html, re.S)
                      .group(1).replace("<\\/", "</"))


def test_report_is_self_contained(outputs):
    html = write_report(outputs, outputs / "universe.csv").read_text()
    assert not re.search(r'(src|href)\s*=\s*"https?://', html) and "cdn." not in html  # works offline
    assert "/*__DATA__*/" not in html  # the placeholder was filled


def test_payload_matches_the_run(outputs):
    html = write_report(outputs, outputs / "universe.csv").read_text()
    p = _payload(html)
    cells, _, complete = load_results(outputs)
    assert p["meta"]["complete"] is True and complete
    assert p["meta"]["archs"] == ["resnet1d", "bilstm"] and p["meta"]["holds"] == ["30 min", "1 h", "2 h", "4 h"]
    assert len(p["cells"]) == len(cells) == 8
    row = next(c for c in p["cells"] if c["arch"] == "resnet1d" and c["hold"] == "30 min")
    src = cells[(cells["arch"] == "resnet1d") & (cells["hold_bars"] == 6)].iloc[0]
    assert row["G_peers"][0] == pytest.approx(src["G_peers"]) and row["net"][0] == pytest.approx(src["net_bps_peers"])
    assert row["verdict"] == src["verdict"]
    # Flow keys use the report's hold labels (a hold with no pair clearing the bar has no key).
    assert {k.split("|")[1] for k in p["flows"]} <= set(p["meta"]["holds"]) and p["meta"]["screened"]


def test_hold_label_mapping():
    assert [hold_label(h) for h in ("30 min", "60 min", "120 min", "240 min")] == ["30 min", "1 h", "2 h", "4 h"]


def test_flows_are_real_pairs_within_range(outputs):
    p = _payload(write_report(outputs, outputs / "universe.csv").read_text())
    n = len(p["stocks"])
    for rows in p["flows"].values():
        assert rows and all(0 <= s < n and 0 <= t < n and s != t and n_ >= 50 for s, t, v, n_ in rows)
        assert all(v > 0 for _, _, v, _ in rows)  # only pairs that cleared the bar
        assert [r[2] for r in rows] == sorted((r[2] for r in rows), reverse=True)  # strongest first


def test_flows_are_exactly_the_runs_heat_map(outputs):
    # Fidelity, not model quality (a planted leader being found is test_controls' job).
    p = _payload(write_report(outputs, outputs / "universe.csv").read_text())
    heat = pd.read_csv(outputs / "heat_flows.csv")
    key = next(iter(p["flows"]))  # any (arch, hold) with pairs that clear the bar
    arch, hold = key.split("|")
    minutes = {"30 min": "30 min", "1 h": "60 min", "2 h": "120 min", "4 h": "240 min"}[hold]
    sub = heat[(heat["arch"] == arch) & (heat["hold"] == minutes) & (heat["lcb"] > 0)]
    sub = sub.sort_values("heat_share", ascending=False)
    shown = [(p["stocks"][s]["s"], p["stocks"][t]["s"], v, n) for s, t, v, n in p["flows"][key]]
    top = list(zip(sub["source"], sub["target"], sub["heat_share"].round(5), sub["n"]))[: len(shown)]
    assert [(a, b, n) for a, b, _, n in shown] == [(a, b, n) for a, b, _, n in top]
    assert np.allclose([v for *_, v, _ in shown], [v for *_, v, _ in top], atol=1e-5)


def test_sector_layout_keeps_stocks_apart_and_inside():
    stocks = [f"X{i}" for i in range(120)]
    sectors = [SECTORS[i % 4] for i in range(120)]
    pos, centers, order = sector_layout(stocks, sectors)
    xy = np.array(list(pos.values()))
    assert xy[:, 0].min() > 0 and xy[:, 0].max() < 1000 and xy[:, 1].min() > 0 and xy[:, 1].max() < 860
    d = np.linalg.norm(xy[:, None] - xy[None], axis=2) + np.eye(len(xy)) * 1e9
    assert d.min() > 5  # no stack of identical dots
    assert set(order) == set(SECTORS)


def test_partial_run_says_pending(outputs, tmp_path):
    (tmp_path / "cells_partial.csv").write_text((outputs / "cells.csv").read_text().replace("EDGE", "pending"))
    df = pd.read_csv(tmp_path / "cells_partial.csv").drop(columns=["verdict"])
    df.to_csv(tmp_path / "cells_partial.csv", index=False)
    (tmp_path / "heat_flows.csv").write_text((outputs / "heat_flows.csv").read_text())
    p = _payload(write_report(tmp_path, outputs / "universe.csv").read_text())
    assert p["meta"]["complete"] is False and all(c["verdict"] == "pending" for c in p["cells"])


def test_missing_results_fail_clearly(tmp_path):
    with pytest.raises(FileNotFoundError, match="score at least one architecture"):
        load_results(tmp_path)
