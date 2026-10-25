"""The whole experiment on one panel: peers -> moments -> split -> train -> test once -> report."""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd
import torch

from .dataset import Gather, build_moments, split_by_date, variants_for
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


def apply_check(cfg):
    """The --check run's settings: train to 2019, validate on 2020, score 2021, inside the training years."""
    cfg["split"].update(train_end="2019-12-31", val_end="2020-12-31", test_end="2021-12-31")
    cfg["train"].update(samples_per_epoch=300_000, max_epochs=3, val_samples=100_000)
    cfg["evaluation"].update(bootstrap=300, heat_samples=50_000)
    cfg["check"] = "2021 (inside the registered training years)"
    return cfg


def build_peers(panel, cfg):
    """The month-by-month peer tables the config asks for (E12's correlation rule, or transfer entropy)."""
    pc, mc = cfg["peers"], cfg["moments"]
    return monthly_peers(panel, pc["k"], pc["peer_days"], pc["min_share"], pc["seed"],
                         select=pc.get("select", "corr"), lookback=mc["lookback"], lag=mc["lag"],
                         holds=tuple(mc["holds"]), select_hold=pc.get("select_hold", mc["holds"][0]))


def make_gather(panel, peers, train, cfg, device):
    """Inputs on the device. Return and TE-feature scales come from the training days only."""
    tr_rows = np.unique(train.t)
    scale = float(np.nanstd(panel.returns[tr_rows[0]: tr_rows[-1] + 1]))
    feat_scale = None
    if peers.features:
        f = peers.features["peers"][np.unique(train.month)]
        f = f[f != 0]
        feat_scale = float(f.std()) if f.size else 1.0
    return Gather(panel, peers, cfg["moments"]["lookback"], scale, device, feat_scale), scale


def _saved(out, arch):
    return out is not None and (out / f"rows_{arch}.json").exists()


def run_experiment(panel, cfg, out_dir=None, device="auto", log=print, archs=None, rerun=False, cache_dir=None):
    """Train, predict and score each architecture in `archs` (default: the config's). Each finished
    architecture is saved at once (rows_<arch>.json, heat_<arch>.csv), and one already saved is
    skipped unless `rerun`, so a dropped session resumes without scoring the test days twice.
    The verdict (false-discovery control over every registered cell) is made only when all of the
    config's architectures are in.

    Nothing is computed twice: the peer tables come from `cache_dir` when this panel and these settings
    were seen before, and each trained variant (weights + test predictions) is saved to
    <out>/variants the moment it is done and reused on a rerun (unless `rerun`)."""
    from . import cache
    from .models import make_model

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
        peers = cache.cached_peers(panel, cfg, cache_dir, lambda: build_peers(panel, cfg), log)
        panel_fp = cache.panel_fingerprint(panel)
        moments = build_moments(panel, peers, mc["lookback"], mc["stride"], mc["lag"], holds)
        train, val, test = split_by_date(moments, panel, sc["train_end"], sc["val_end"], sc["embargo_days"],
                                         sc.get("test_end"))
        log(f"moments: train {len(train):,}, val {len(val):,}, test {len(test):,}")
        if not (len(train) and len(val) and len(test)):
            raise ValueError("a split is empty: check the split dates against the data's days")
        gather, scale = make_gather(panel, peers, train, cfg, device)
        if out:
            counts = {"train": len(train), "val": len(val), "test": len(test)}
            info = {"device": device, "input_scale": scale, "moments": counts}
            if peers.te is not None:  # heat_v2: how often a TE peer is a negative relation
                pick = cfg["peers"].get("select_hold", holds[0])
                f = peers.features["peers"][:, :, :, list(peers.feature_holds).index(pick)]
                f = f[peers.te >= 0]
                info["te_peers_negative_share"] = float((f < 0).mean()) if f.size else None
            (out / "run.json").write_text(json.dumps(info))

    for arch in todo:
        # The bootstrap stream depends on the architecture, not on the order archs are run in.
        rng = np.random.default_rng([tc["seed"], registered.index(arch) if arch in registered else 99])
        models, preds = {}, {}
        tca = {**tc, **tc.get("per_arch", {}).get(arch, {})}  # tuned settings, if the config has them
        vdir = out / "variants" if out else None
        cfg_a = {**cfg, "train": tca}
        for variant in variants_for(cfg["peers"]):
            vkey = cache.variant_key(panel_fp, cfg_a, arch, variant, len(test))
            hit = None if (rerun or vdir is None) else cache.load_variant(vdir, arch, variant, vkey)
            if hit is not None:
                state, preds[variant], info = hit
                hp = {k: tca[k] for k in ("dropout", "depth", "kernel", "heads") if k in tca}
                model = make_model(arch, gather.channels, len(holds), mc["lookback"], tca["size"], hp)
                model.load_state_dict(state)
                model = model.to(device).eval()
                log(f"  {arch}/{variant}: reused (trained earlier with these exact settings)")
            else:
                model, info = train_variant(arch, variant, gather, train, val, holds, mc["lookback"], tca, device, log)
                preds[variant] = predict(model, gather, test, variant, info["y_scale"], device=device)
                if vdir is not None:  # saved at once: a dropped session loses at most this one variant
                    cache.save_variant(vdir, arch, variant, vkey, to_cpu_state(model), preds[variant], info)
            models[variant] = (model, info)
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
        info = json.loads(run_json.read_text()) if run_json.exists() else {}
        (out / "report.md").write_text(report_md(table, heat_all, cfg, info.get("moments", {}), complete, have,
                                                 info.get("te_peers_negative_share")))
    return table, heat_all


def _ci(c, fmt="{:+.2f}"):
    return "[" + fmt.format(c[0]) + ", " + fmt.format(c[1]) + "]"


def report_md(table, heat, cfg, counts, complete=True, have=(), negative_share=None):
    ec = cfg["evaluation"]
    by_te = cfg["peers"].get("select", "corr") == "te"
    lines = [f"# {cfg['experiment']}: transfer between stocks "
             f"(peers chosen by {'transfer entropy' if by_te else 'correlation'})", ""]
    if cfg.get("check"):
        lines += [f"> **CHECK RUN, not the registered result.** Scored on {cfg['check']}; the registered "
                  "test days were not used.", ""]
    if not complete:
        missing = [a for a in cfg["train"]["archs"] if a not in have]
        lines += [f"> **PARTIAL: {len(have)} of {len(cfg['train']['archs'])} architectures scored** (missing: "
                  f"{', '.join(missing)}). No verdict until all are in: the false-discovery control is "
                  "registered over every cell.", ""]
    if negative_share is not None:
        lines += [f"Of the transfer-entropy peers, **{negative_share:.0%} are negative relations** (the stock moved "
                  "against the peer) and the rest positive.", ""]
    n = lambda k: f"{counts[k]:,}" if k in counts else "?"
    lines += [f"Moments: train {n('train')}, val {n('val')}, test {n('test')} (scored days: after "
              f"{cfg['split']['val_end']}). Intervals: {int(ec['level'] * 100)}% day-block. "
              f"Cost: {ec['cost_bps']:.0f} bps per position.", "",
              "| arch | hold | G peers (nats) | G peers − random | net bps (peers) | net − no peers | IC | verdict |",
              "|---|---|---|---|---|---|---|---|"]
    if "G_peers_minus_corr" in table:  # heat_v2: peers are chosen by transfer entropy
        lines[-2] = lines[-2].replace("G peers − random |", "G peers − random | G TE − correlation peers |")
        lines[-1] = lines[-1] + "---|"
    for _, r in table.iterrows():
        verdict = r["verdict"] if complete else "(pending)"
        lines.append(f"| {r['arch']} | {r['hold']} | {r['G_peers']:+.5f} {_ci(r['G_peers_ci'], '{:+.5f}')} | "
                     f"{r['G_peers_minus_random']:+.5f} {_ci(r['G_peers_minus_random_ci'], '{:+.5f}')} | "
                     + (f"{r['G_peers_minus_corr']:+.5f} {_ci(r['G_peers_minus_corr_ci'], '{:+.5f}')} | "
                        if "G_peers_minus_corr" in table else "")
                     + f"{r['net_bps_peers']:+.2f} {_ci(r['net_bps_peers_ci'])} | "
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
