"""Scoring one architecture at a time: saved at once, never re-scored by accident, verdict only
when every registered architecture is in."""
import copy

from heat import config
from heat.run import run_experiment
from heat.synthetic import synthetic_panel


def _cfg(panel):
    cfg = copy.deepcopy(config.load("configs/heat_v1.yaml"))
    d = panel.days
    cfg["split"].update(train_end=str(d[109].date()), val_end=str(d[129].date()))
    cfg["train"].update(archs=["resnet1d", "bilstm"], size=8, batch=512, samples_per_epoch=2000,
                        max_epochs=1, patience=1, val_samples=1000)
    cfg["evaluation"].update(bootstrap=50, heat_samples=0)
    return cfg


def test_one_arch_at_a_time(tmp_path):
    panel = synthetic_panel(n_days=150, n_clusters=5)
    cfg = _cfg(panel)
    quiet = lambda m: None

    table, _ = run_experiment(panel, cfg, tmp_path, log=quiet, archs=["resnet1d"])
    assert "verdict" not in table and (tmp_path / "cells_partial.csv").exists()
    assert not (tmp_path / "cells.csv").exists()
    assert "PARTIAL: 1 of 2" in (tmp_path / "report.md").read_text()
    first = (tmp_path / "variants" / "resnet1d_peers.pt").stat().st_mtime_ns

    logs = []
    table, _ = run_experiment(panel, cfg, tmp_path, log=logs.append)  # resume: only bilstm runs
    assert any("resnet1d: already scored, skipped" in m for m in logs)
    assert (tmp_path / "variants" / "resnet1d_peers.pt").stat().st_mtime_ns == first  # not retrained
    assert (tmp_path / "cells.csv").exists() and len(table) == 8 and table["verdict"].notna().all()
    assert "PARTIAL" not in (tmp_path / "report.md").read_text()
