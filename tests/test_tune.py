"""Tuning: validation-only, default always a candidate, adoption needs a gain beyond seed noise, and the
candidate config differs from the registered one only where it says it does."""
import copy
import inspect

import numpy as np
import pytest

from heat import config
from heat.dataset import Gather, build_moments, split_by_date
from heat.models import DEFAULTS, make_model
from heat.peers import monthly_peers
from heat.run import run_experiment
from heat.synthetic import synthetic_panel
from heat.tune import SPACE, candidate_config, default_setting, sample_setting, tune_arch


@pytest.fixture(scope="module")
def setup():
    cfg = config.load("configs/heat_v1.yaml")
    panel = synthetic_panel(n_days=150, n_clusters=5, seed=2)
    peers = monthly_peers(panel, 5, 60, 0.8)
    mc = cfg["moments"]
    m = build_moments(panel, peers, mc["lookback"], mc["stride"], mc["lag"], mc["holds"])
    d = panel.days
    train, val, _ = split_by_date(m, panel, d[109], d[129])
    gather = Gather(panel, peers, mc["lookback"], 1e-3)
    base = {**cfg["train"], "size": 16, "batch": 512, "samples_per_epoch": 3000, "val_samples": 2000}
    return cfg, panel, gather, train, val, base


def test_tuner_cannot_see_test_days():
    params = inspect.signature(tune_arch).parameters
    assert not any("test" in p for p in params)  # train and val moments only


def test_default_is_trial_zero_and_samples_are_valid():
    base = {"size": 64, "lr": 1e-3, "weight_decay": 1e-4}
    for arch in DEFAULTS:
        assert default_setting(arch, base)["dropout"] == DEFAULTS[arch]["dropout"]
        rng = np.random.default_rng(0)
        for _ in range(50):
            s = sample_setting(arch, rng)
            assert 3e-4 <= s["lr"] <= 3e-3 and 0 <= s["dropout"] <= 0.3 and set(SPACE[arch]) <= set(s)
            hp = {k: s[k] for k in ("dropout", "depth", "kernel", "heads") if k in s}
            make_model(arch, 6, 4, 12, s["size"], hp)  # every sampled setting builds


def test_tune_returns_a_setting_that_trains(setup):
    cfg, panel, gather, train, val, base = setup
    quiet = lambda m: None
    result, trials = tune_arch("resnet1d", gather, train, val, cfg["moments"]["holds"], cfg["moments"]["lookback"],
                               base, n_trials=3, eta=3, budgets=(1, 2), seed=0, log=quiet)
    assert (trials["trial"] == 0).any()  # the default was a candidate
    assert set(trials["rung"]) == {0, 1} and trials[trials["rung"] == 1]["trial"].nunique() == 1  # 3 -> 1
    assert result["seed_spread"] >= 0 and set(result["confirm"]) == {"default", "tuned"}
    s = result["setting"]
    hp = {k: s[k] for k in ("dropout", "depth", "kernel", "heads") if k in s}
    make_model("resnet1d", 6, 4, 12, s["size"], hp)
    if not result["adopt_tuned"]:
        assert result["setting"] == default_setting("resnet1d", base)


def test_gain_inside_seed_noise_is_not_adopted(setup, monkeypatch):
    import heat.tune as tune
    cfg, panel, gather, train, val, base = setup
    scores = iter([0.50, 0.40, 0.45, 0.50,           # rung 0: default, two others (3 trials) -> best is trial 1
                   0.30, 0.299, 0.305, 0.301])       # confirm: default x2 seeds, tuned x2 seeds (spread .006 > gain)
    monkeypatch.setattr(tune, "_score", lambda *a, **k: (next(scores), {v: 0.0 for v in tune.VARIANTS}))
    result, _ = tune.tune_arch("bilstm", gather, train, val, cfg["moments"]["holds"], cfg["moments"]["lookback"],
                               base, n_trials=3, eta=3, budgets=(1,), seed=0, log=lambda m: None)
    assert not result["adopt_tuned"] and result["setting"] == tune.default_setting("bilstm", base)


def test_candidate_config_changes_only_what_it_says():
    base_text = open("configs/heat_v1.yaml").read()
    results = {a: {"setting": {"size": 32, "lr": 0.002, "weight_decay": 0.001, **DEFAULTS[a]}, "adopt_tuned": a == "bilstm"}
               for a in ("bilstm", "resnet1d", "transformer")}
    text = candidate_config(base_text, results, "2026-10-21")
    import yaml
    v1, v2 = yaml.safe_load(base_text), yaml.safe_load(text)
    assert v2["experiment"] == "heat_v2" and "per_arch" in v2["train"]
    per_arch = v2["train"].pop("per_arch")
    v2["experiment"] = v1["experiment"]
    assert v1 == v2  # nothing else moved: data, split, moments, evaluation, cost, protocol
    assert per_arch["bilstm"]["size"] == 32 and "default kept" in text


def test_per_arch_settings_are_applied_by_the_run(setup, tmp_path):
    cfg, panel, *_ = setup
    cfg = copy.deepcopy(cfg)
    d = panel.days
    cfg["split"].update(train_end=str(d[109].date()), val_end=str(d[129].date()))
    cfg["train"].update(archs=["resnet1d"], size=8, batch=512, samples_per_epoch=1500, max_epochs=1, patience=1,
                        val_samples=800, per_arch={"resnet1d": {"size": 16, "depth": 2, "kernel": 5, "dropout": 0.0}})
    cfg["evaluation"].update(bootstrap=30, heat_samples=0)
    run_experiment(panel, cfg, tmp_path, log=lambda m: None)
    import torch
    sd = torch.load(tmp_path / "resnet1d_peers.pt")
    assert sd["stem.weight"].shape[0] == 16  # width from per_arch, not the base size of 8
    assert sum(k.startswith("blocks.") and k.endswith("net.0.weight") for k in sd) == 2  # depth 2
