"""Exploratory: are transfer and the edge different on macro release days?

Uses the models a run already trained (their saved test predictions), so nothing is retrained. For each
architecture, hold and subset of test moments (each release type's days, any-release days, no-release
days, and for FOMC the holds that span the 14:00 decision), it recomputes the same numbers as the
registered evaluation, with day-block intervals on that subset's days.

This was not part of any registered rule, so it gives no verdict: it is a description of where any
transfer lives, to be reported as exploratory, with many subsets looked at.
"""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd

from . import cache
from .dataset import build_moments, split_by_date, variants_for
from .evaluate import evaluate_arch
from .events import moment_masks
from .run import build_peers


def _ci(c, fmt="{:+.5f}"):
    return "[" + fmt.format(c[0]) + ", " + fmt.format(c[1]) + "]"


def event_table(out_dir, panel, cfg, calendar, cache_dir=None, bootstrap=1000, min_days=5, log=print):
    out = Path(out_dir)
    mc, sc = cfg["moments"], cfg["split"]
    holds = mc["holds"]
    peers = cache.cached_peers(panel, cfg, cache_dir, lambda: build_peers(panel, cfg), log)
    moments = build_moments(panel, peers, mc["lookback"], mc["stride"], mc["lag"], holds)
    _, _, test = split_by_date(moments, panel, sc["train_end"], sc["val_end"], sc["embargo_days"], sc.get("test_end"))
    masks = moment_masks(test, panel, calendar, holds, mc["lag"])
    panel_fp = cache.panel_fingerprint(panel)
    ecfg = {**cfg["evaluation"], "bootstrap": bootstrap}
    rows = []
    for arch in cfg["train"]["archs"]:
        tca = {**cfg["train"], **cfg["train"].get("per_arch", {}).get(arch, {})}
        preds = {}
        for v in variants_for(cfg["peers"]):
            hit = cache.load_variant(out / "variants", arch, v, cache.variant_key(panel_fp, {**cfg, "train": tca},
                                                                                 arch, v, len(test)))
            if hit is None:
                break
            preds[v] = hit[1]
        if len(preds) < len(variants_for(cfg["peers"])):
            log(f"{arch}: no saved predictions matching this run's settings; run it first")
            continue
        rng = np.random.default_rng([cfg["train"]["seed"], 7])
        for name, m in masks.items():
            n_days = len(np.unique(test.day[m]))
            if n_days < min_days:
                continue
            for r in evaluate_arch({v: p[m] for v, p in preds.items()}, test.subset(m), holds, panel.days, ecfg, rng):
                if name.startswith("fomc, hold spans") and not name.endswith(f"({r['hold_bars'] * 5} min)"):
                    continue  # that subset is defined for one hold only
                row = {"arch": arch, "subset": name, "hold": r["hold"], "days": n_days, "moments": r["moments"],
                       "G_peers": r["G_peers"], "G_peers_ci": r["G_peers_ci"],
                       "G_peers_minus_random": r["G_peers_minus_random"], "G_peers_minus_random_ci": r["G_peers_minus_random_ci"],
                       "net_bps_peers": r["net_bps_peers"], "net_bps_peers_ci": r["net_bps_peers_ci"],
                       "gross_bps_peers": r["gross_bps_peers"]}
                if "G_peers_minus_corr" in r:
                    row["G_peers_minus_corr"] = r["G_peers_minus_corr"]
                    row["G_peers_minus_corr_ci"] = r["G_peers_minus_corr_ci"]
                rows.append(row)
        log(f"{arch}: {sum(r['arch'] == arch for r in rows)} subset cells")
    table = pd.DataFrame(rows)
    if len(table):
        table.to_csv(out / "events.csv", index=False)
        (out / "events.md").write_text(events_md(table, cfg, calendar))
    return table


def events_md(table, cfg, calendar):
    counts = calendar.groupby("event").size().to_dict()
    lines = [f"# {cfg['experiment']}: transfer and the edge on macro release days (exploratory)", "",
             "> Not a registered test: no verdict. Many subsets are looked at, so single cells whose interval "
             "clears zero are expected by chance. Read it for patterns that hold across architectures.", "",
             "Releases in the calendar (official dates via FRED): "
             + ", ".join(f"{k} {v}" for k, v in sorted(counts.items())) + ". Intervals: day-block, "
             f"{int(cfg['evaluation']['level'] * 100)}%.", ""]
    te = "G_peers_minus_corr" in table
    head = "| subset | hold | arch | days | G peers (nats) | G peers − random |" + (" G TE − correlation |" if te else "") \
        + " net bps | gross bps |"
    lines += [head, "|" + "---|" * (head.count("|") - 1)]
    order = {s: i for i, s in enumerate(dict.fromkeys(table["subset"]))}
    for _, r in table.sort_values(["subset", "hold", "arch"], key=lambda c: c.map(order) if c.name == "subset" else c).iterrows():
        lines.append(f"| {r['subset']} | {r['hold']} | {r['arch']} | {r['days']} | {r['G_peers']:+.5f} {_ci(r['G_peers_ci'])} | "
                     f"{r['G_peers_minus_random']:+.5f} {_ci(r['G_peers_minus_random_ci'])} | "
                     + (f"{r['G_peers_minus_corr']:+.5f} {_ci(r['G_peers_minus_corr_ci'])} | " if te else "")
                     + f"{r['net_bps_peers']:+.2f} {_ci(r['net_bps_peers_ci'], '{:+.2f}')} | {r['gross_bps_peers']:+.2f} |")
    clear = table[table["G_peers_minus_random_ci"].map(lambda c: c[0] > 0)]
    lines += ["", f"**Cells where TE peers beat random peers (interval above zero): {len(clear)} of {len(table)}.**"]
    return "\n".join(lines) + "\n"
