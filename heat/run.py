"""The whole experiment on one panel: peers -> moments -> split -> train -> test once -> report."""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd
import torch

from .dataset import VARIANTS, Gather, build_moments, split_by_date
from .device import resolve, to_cpu_state
from .evaluate import evaluate_arch, heat_flows, verdicts
from .peers import monthly_peers
from .train import predict, train_variant


def pick_device(name="auto"):
    """'auto' = a CUDA GPU if there is one, else the CPU. TPUs are chosen explicitly ('tpu' or 'xla'),
    normally by the notebook after its device self-test, because only that test shows they work."""
    if name != "auto":
        return resolve(name)
    if torch.cuda.is_available():
        return "cuda"
    return "cpu"  # MPS lacks some ops used here; CPU is the safe local default


def _saved(out, arch):
    return out is not None and (out / f"rows_{arch}.json").exists()


def run_experiment(panel, cfg, out_dir=None, device="auto", log=print, archs=None, rerun=False):
    """Train, predict and score each architecture in `archs` (default: the config's). Each finished
    architecture is saved at once (rows_<arch>.json, heat_<arch>.csv), and one already saved is
    skipped unless `rerun`, so a dropped session resumes without scoring the test days twice.
    The verdict (false-discovery control over every registered cell) is made only when all of the
    config's architectures are in."""
    device = pick_device(device)
    out = Path(out_dir) if out_dir else None
    if out:
        out.mkdir(parents=True, exist_ok=True)
    mc, sc, tc, ec = cfg["moments"], cfg["split"], cfg["train"], cfg["evaluation"]
    holds = mc["holds"]
    registered = list(tc["archs"])
    wanted = list(archs or registered)
    todo = [a for a in wanted if rerun or not _saved(out, a)]
    for a in wanted:
        if a not in todo:
            log(f"{a}: already scored, skipped (pass rerun to score again)")

    rows, heat = {}, {}
    if todo:
        log(f"peers: k={cfg['peers']['k']}, {cfg['peers']['peer_days']} days, monthly")
        peers = monthly_peers(panel, cfg["peers"]["k"], cfg["peers"]["peer_days"], cfg["peers"]["min_share"],
                              cfg["peers"]["seed"])
        moments = build_moments(panel, peers, mc["lookback"], mc["stride"], mc["lag"], holds)
        train, val, test = split_by_date(moments, panel, sc["train_end"], sc["val_end"], sc["embargo_days"],
                                         sc.get("test_end"))
        log(f"moments: train {len(train):,}, val {len(val):,}, test {len(test):,}")
        if not (len(train) and len(val) and len(test)):
            raise ValueError("a split is empty: check the split dates against the data's days")
        tr_rows = np.unique(train.t)
        scale = float(np.nanstd(panel.returns[tr_rows[0]: tr_rows[-1] + 1]))  # train days only
        gather = Gather(panel, peers, mc["lookback"], scale, device)
        if out:
            counts = {"train": len(train), "val": len(val), "test": len(test)}
            (out / "run.json").write_text(json.dumps({"device": device, "input_scale": scale, "moments": counts}))

    for arch in todo:
        # The bootstrap stream depends on the architecture, not on the order archs are run in.
        rng = np.random.default_rng([tc["seed"], registered.index(arch) if arch in registered else 99])
        models, preds = {}, {}
        tca = {**tc, **tc.get("per_arch", {}).get(arch, {})}  # tuned settings, if the config has them
        for variant in VARIANTS:
            model, info = train_variant(arch, variant, gather, train, val, holds, mc["lookback"], tca, device, log)
            models[variant] = (model, info)
            preds[variant] = predict(model, gather, test, variant, info["y_scale"], device=device)
            if out:
                torch.save(to_cpu_state(model), out / f"{arch}_{variant}.pt")
        rows[arch] = [{"arch": arch, **r} for r in evaluate_arch(preds, test, holds, panel.days, ec, rng)]
        flows = []
        if ec.get("heat_samples"):
            model, info = models["peers"]
            for j, h in enumerate(holds):
                f = heat_flows(model, gather, test, info["y_scale"], j, panel.symbols, ec["heat_samples"],
                               device=device, rng=rng)
                flows.append(f.assign(arch=arch, hold=f"{h * 5} min"))
        heat[arch] = pd.concat(flows) if flows else pd.DataFrame()
        if out:
            pd.DataFrame(rows[arch]).to_json(out / f"rows_{arch}.json", orient="records")
            heat[arch].to_csv(out / f"heat_{arch}.csv", index=False)
        log(f"{arch}: scored and saved")
    return finalize(out, cfg, rows, heat)


def finalize(out, cfg, rows=None, heat=None):
    """Combine the scored architectures (this run's, then any saved). Returns (table, heat). The
    table carries verdicts only when every registered architecture is in; otherwise it is the raw
    numbers, and the report is marked partial."""
    registered = list(cfg["train"]["archs"])
    rows, heat = dict(rows or {}), dict(heat or {})
    for a in registered:
        if a not in rows and _saved(out, a):
            saved = pd.read_json(out / f"rows_{a}.json", orient="records")
            for col in [c for c in saved.columns if c.endswith("_ci")]:
                saved[col] = saved[col].map(tuple)
            rows[a] = saved.to_dict("records")
            h = out / f"heat_{a}.csv"
            heat[a] = pd.read_csv(h) if h.exists() and h.stat().st_size > 1 else pd.DataFrame()
    have = [a for a in registered if rows.get(a)] + [a for a in rows if a not in registered]
    table = pd.DataFrame([r for a in have for r in rows[a]])
    parts = [heat[a] for a in have if a in heat and len(heat[a])]
    heat_all = pd.concat(parts) if parts else pd.DataFrame()
    complete = all(a in have for a in registered)
    if complete:
        table = verdicts(table, cfg["evaluation"]["fdr_q"])
    if out is not None and have:
        table.to_csv(out / ("cells.csv" if complete else "cells_partial.csv"), index=False)
        heat_all.to_csv(out / "heat_flows.csv", index=False)
        run_json = out / "run.json"
        counts = json.loads(run_json.read_text()).get("moments", {}) if run_json.exists() else {}
        (out / "report.md").write_text(report_md(table, heat_all, cfg, counts, complete, have))
    return table, heat_all


def _ci(c, fmt="{:+.2f}"):
    return "[" + fmt.format(c[0]) + ", " + fmt.format(c[1]) + "]"


def report_md(table, heat, cfg, counts, complete=True, have=()):
    ec = cfg["evaluation"]
    lines = [f"# {cfg['experiment']}: transfer between correlated stocks", ""]
    if cfg.get("check"):
        lines += [f"> **CHECK RUN, not the registered result.** Scored on {cfg['check']}; the registered "
                  "test days were not used.", ""]
    if not complete:
        missing = [a for a in cfg["train"]["archs"] if a not in have]
        lines += [f"> **PARTIAL: {len(have)} of {len(cfg['train']['archs'])} architectures scored** (missing: "
                  f"{', '.join(missing)}). No verdict until all are in: the false-discovery control is "
                  "registered over every cell.", ""]
    n = lambda k: f"{counts[k]:,}" if k in counts else "?"
    lines += [f"Moments: train {n('train')}, val {n('val')}, test {n('test')} (scored days: after "
              f"{cfg['split']['val_end']}). Intervals: {int(ec['level'] * 100)}% day-block. "
              f"Cost: {ec['cost_bps']:.0f} bps per position.", "",
              "| arch | hold | G peers (nats) | G peers − random | net bps (peers) | net − no peers | IC | verdict |",
              "|---|---|---|---|---|---|---|---|"]
    for _, r in table.iterrows():
        verdict = r["verdict"] if complete else "(pending)"
        lines.append(f"| {r['arch']} | {r['hold']} | {r['G_peers']:+.5f} {_ci(r['G_peers_ci'], '{:+.5f}')} | "
                     f"{r['G_peers_minus_random']:+.5f} {_ci(r['G_peers_minus_random_ci'], '{:+.5f}')} | "
                     f"{r['net_bps_peers']:+.2f} {_ci(r['net_bps_peers_ci'])} | "
                     f"{r['net_bps_peers_minus_own']:+.2f} {_ci(r['net_bps_peers_minus_own_ci'])} | "
                     f"{r['ic_peers']:+.4f} | {verdict} |")
    if complete:
        lines += ["", f"**Cells with an edge: {int((table['verdict'] == 'EDGE').sum())} of {len(table)}.** "
                  "Cells with heat transfer (correlated peers beat random peers and no peers): "
                  f"{int(table['heat_transfer'].sum())} of {len(table)}."]
    if len(heat):
        if "lcb" in heat:
            clear = heat[heat["lcb"] > 0]
            top = clear.sort_values("lcb", ascending=False).head(20)
            lines += ["", "## Where the heat goes", "",
                      f"Pairs whose share of the follower's error clears zero by 3 standard errors: **{len(clear):,} of "
                      f"{len(heat):,}** (a screening bar across many pairs, not a test)."]
            if len(top):
                lines += ["", "| arch | hold | source | target | moments | heat share | lower bound |", "|---|---|---|---|---|---|---|"]
                lines += [f"| {r.arch} | {r.hold} | {r.source} | {r.target} | {r.n} | {r.heat_share:+.4f} | {r.lcb:+.4f} |"
                          for r in top.itertuples()]
        else:  # outputs from before the screening bar existed
            top = heat.sort_values("heat_share", ascending=False).head(20)
            lines += ["", "## Where the heat goes (unscreened: small pairs can top this list by luck)", "",
                      "| arch | hold | source | target | moments | heat share |", "|---|---|---|---|---|---|"]
            lines += [f"| {r.arch} | {r.hold} | {r.source} | {r.target} | {r.n} | {r.heat_share:+.4f} |"
                      for r in top.itertuples()]
    return "\n".join(lines) + "\n"
