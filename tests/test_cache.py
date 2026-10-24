"""Nothing is computed twice, and nothing stale is ever reused."""
import copy

import numpy as np

import heat.train as train_mod
import heat.run as run_mod
from heat import config
from heat.cache import cached_peers, load_peers, panel_fingerprint, peers_key, save_peers
from heat.run import build_peers, run_experiment
from heat.synthetic import synthetic_panel


def _cfg(panel, v="configs/heat_v2.yaml"):
    cfg = copy.deepcopy(config.load(v))
    d = panel.days
    cfg["peers"]["peer_days"] = 60
    cfg["split"].update(train_end=str(d[109].date()), val_end=str(d[129].date()))
    cfg["train"].update(archs=["resnet1d"], size=8, batch=512, samples_per_epoch=1500, max_epochs=1, patience=1,
                        val_samples=800)
    cfg["evaluation"].update(bootstrap=30, heat_samples=0)
    return cfg


def test_peer_cache_round_trip_and_keys(tmp_path):
    p = synthetic_panel(n_days=150, n_clusters=5, cross=-1)
    cfg = _cfg(p)
    t = build_peers(p, cfg)
    save_peers(tmp_path / "x.npz", t)
    u = load_peers(tmp_path / "x.npz")
    assert (u.te == t.te).all() and (u.correlated == t.correlated).all() and (u.random == t.random).all()
    assert all(np.array_equal(u.features[k], t.features[k]) for k in t.features) and u.feature_holds == t.feature_holds
    other = copy.deepcopy(cfg); other["peers"]["k"] = 4
    assert peers_key(p, cfg) != peers_key(p, other)  # different settings, different cache entry
    q = synthetic_panel(n_days=150, n_clusters=5, cross=-1, seed=9)
    assert panel_fingerprint(p) != panel_fingerprint(q)  # different data, different cache entry


def test_cached_peers_builds_once(tmp_path):
    p = synthetic_panel(n_days=150, n_clusters=5)
    cfg = _cfg(p)
    calls = []
    build = lambda: calls.append(1) or build_peers(p, cfg)
    cached_peers(p, cfg, tmp_path, build, log=lambda m: None)
    cached_peers(p, cfg, tmp_path, build, log=lambda m: None)
    assert len(calls) == 1


def test_finished_variants_are_reused_not_retrained(tmp_path, monkeypatch):
    p = synthetic_panel(n_days=150, n_clusters=5, cross=-1)
    cfg = _cfg(p)
    first, _ = run_experiment(p, cfg, tmp_path / "out", log=lambda m: None, cache_dir=tmp_path / "cache")
    # Simulate a dropped session after training: the scored rows vanish, the trained variants remain.
    (tmp_path / "out" / "rows_resnet1d.json").unlink()
    calls = []
    real = train_mod.train_variant
    monkeypatch.setattr(run_mod, "train_variant", lambda *a, **k: calls.append(a[1]) or real(*a, **k))
    again, _ = run_experiment(p, cfg, tmp_path / "out", log=lambda m: None, cache_dir=tmp_path / "cache")
    assert calls == []  # all four variants came from disk
    assert np.allclose(first["G_peers"], again["G_peers"]) and np.allclose(first["net_bps_peers"], again["net_bps_peers"])
    # Change a training setting: the saved variants no longer match and are retrained.
    cfg2 = copy.deepcopy(cfg); cfg2["train"]["lr"] = 0.002
    (tmp_path / "out" / "rows_resnet1d.json").unlink()
    run_experiment(p, cfg2, tmp_path / "out", log=lambda m: None, cache_dir=tmp_path / "cache")
    assert sorted(calls) == ["corr", "own", "peers", "random"]
