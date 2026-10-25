"""The release calendar and the event-day split."""
import copy
import io
import json

import numpy as np
import pandas as pd
import pytest

from heat import config
from heat.event_analysis import event_table
from heat.events import RELEASES, fetch_calendar, fred_key, moment_masks
from heat.run import build_peers, run_experiment
from heat.dataset import build_moments, split_by_date
from heat.synthetic import synthetic_panel


def test_fetch_calendar_parses_fred(monkeypatch):
    seen = []

    def opener(url):
        seen.append(url)
        rid = int(url.split("release_id=")[1].split("&")[0])
        dates = {50: ["2024-01-05", "2024-02-02"], 10: ["2024-01-11"], 53: ["2024-01-25"], 101: ["2024-01-31", "2030-01-01"]}[rid]
        return io.BytesIO(json.dumps({"release_dates": [{"release_id": rid, "date": d} for d in dates]}).encode())

    cal = fetch_calendar("2016-01-01", "2026-12-31", api_key="k" * 32, opener=opener)
    assert len(seen) == 4 and all("file_type=json" in u and "api_key=" in u for u in seen)
    assert cal["event"].value_counts().to_dict() == {"jobs": 2, "cpi": 1, "gdp": 1, "fomc": 1}  # 2030 dropped
    assert (cal.loc[cal.event == "fomc", "time_et"] == "14:00").all() and (cal.loc[cal.event == "jobs", "time_et"] == "08:30").all()


def test_missing_key_says_how_to_get_one(monkeypatch):
    monkeypatch.delenv("FRED_API_KEY", raising=False)
    with pytest.raises(RuntimeError, match="FRED_API_KEY"):
        fred_key()


def test_release_ids_are_the_checked_ones():
    assert {k: v[0] for k, v in RELEASES.items()} == {"jobs": 50, "cpi": 10, "gdp": 53, "fomc": 101}


def test_masks_days_and_the_fomc_window():
    p = synthetic_panel(n_days=130, n_clusters=4)
    cfg = config.load("configs/heat_v1.yaml")
    peers = build_peers(p, cfg)
    m = build_moments(p, peers, 12, 6, 1, (6, 48))
    d = p.days
    _, _, test = split_by_date(m, p, d[90], d[100])
    day = d[110].strftime("%Y-%m-%d")
    cal = pd.DataFrame({"date": [day, day], "event": ["fomc", "jobs"]})
    masks = moment_masks(test, p, cal, (6, 48))
    on_day = pd.DatetimeIndex(d[test.day]).strftime("%Y-%m-%d") == day
    assert (masks["fomc days"] == on_day).all() and (masks["jobs days"] == on_day).all()
    assert (masks["no release day"] == ~on_day).all() and not masks["cpi days"].any()
    w = masks["fomc, hold spans 14:00 (30 min)"]
    bars = p.bar[test.t[w]].astype(int)
    assert w.any() and ((bars + 2 <= 54) & (bars + 7 >= 54)).all() and on_day[w].all()  # 30-min hold covers bar 54


def test_split_shows_transfer_concentrated_on_release_days(tmp_path):
    # The planted link exists every day but is three times stronger on "release days". (A link that exists
    # ONLY on release days is diluted: TE is estimated over all days and the models don't see the calendar.)
    rng = np.random.default_rng(1)
    n_days = 480
    event = rng.random(n_days) < 0.3
    p = synthetic_panel(n_days=n_days, n_clusters=8, cluster_size=5, seed=3, cross=-1, beta=0.4,
                        active_days=1.0 + 2.0 * event)
    cfg = copy.deepcopy(config.load("configs/heat_v2.yaml"))
    d = p.days
    cfg["split"].update(train_end=str(d[389].date()), val_end=str(d[429].date()))
    cfg["train"].update(archs=["bilstm"], size=32, batch=1024, samples_per_epoch=30000, max_epochs=6, patience=2,
                        val_samples=15000)
    cfg["evaluation"].update(bootstrap=200, heat_samples=0)
    run_experiment(p, cfg, tmp_path, log=lambda m: None, cache_dir=tmp_path / "cache")
    cal = pd.DataFrame({"date": pd.DatetimeIndex(d[event]).strftime("%Y-%m-%d"), "event": "jobs"})
    t = event_table(tmp_path, p, cfg, cal, tmp_path / "cache", bootstrap=200, log=lambda m: None)
    row = lambda subset: t[(t.subset == subset) & (t.hold == "30 min")].iloc[0]
    on, off = row("jobs days"), row("no release day")
    assert on["G_peers_minus_random_ci"][0] > 0  # the gain shows up on release days...
    # ...and is much smaller (here even negative: the models learn an average strength, too much for
    # ordinary days) on the other days. The split must make that gap visible.
    assert on["G_peers"] - off["G_peers"] > 0.003
    assert (tmp_path / "events.md").exists() and "exploratory" in (tmp_path / "events.md").read_text()
