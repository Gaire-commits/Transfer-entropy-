"""Tune the deep models on the training and validation days. The test days are not touched.

    python scripts/tune.py --out-dir <outputs>/heat_v1 --archs bilstm      # one architecture per session
    python scripts/tune.py --out-dir <outputs>/heat_v1 --check --trials 4  # a quick run on 2019-2020

Reads the panel saved by scripts/run.py (builds it if missing). Each architecture is saved to
<out-dir>/tuning/ as soon as it is done and skipped on the next run. When all architectures in the
config are tuned it writes tuning/report.md and tuning/heat_v2.yaml, the candidate config to
register before the registered run scores the test days.
"""
import argparse
import datetime
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from heat import config  # noqa: E402
from heat.data import THESIS_SIP_DIR, Panel, build_panel, load_universe  # noqa: E402
from heat.dataset import Gather, build_moments, split_by_date  # noqa: E402
from heat.dataset import variants_for  # noqa: E402
from heat.run import build_peers, make_gather, pick_device  # noqa: E402
from heat.tune import candidate_config, tune_arch  # noqa: E402


def write_outputs(tdir, cfg, base_text, archs, args):
    results = {a: json.loads((tdir / f"{a}.json").read_text()) for a in archs if (tdir / f"{a}.json").exists()}
    if not results:
        return
    lines = ["# Tuning report", "",
             "Validation days only; the registered test days were not used. Score = mean validation loss over the "
             "no-peer, correlated-peer and random-peer models (lower is better).", "",
             "| arch | default | tuned winner | gain | seed spread | decision |", "|---|---|---|---|---|---|"]
    for a, r in results.items():
        lines.append(f"| {a} | {r['default_val_loss']:.5f} | {r['tuned_val_loss']:.5f} | {r['improvement']:+.5f} | "
                     f"{r['seed_spread']:.5f} | {'tuned adopted' if r['adopt_tuned'] else 'default kept'} |")
    for a, r in results.items():
        lines += ["", f"**{a}** chosen: `{r['setting']}`"]
    complete = all(a in results for a in cfg["train"]["archs"])
    if not complete:
        lines += ["", "> PARTIAL: architectures still to tune: "
                  + ", ".join(a for a in cfg["train"]["archs"] if a not in results)]
    (tdir / "report.md").write_text("\n".join(lines) + "\n")
    if complete and not args.check:
        date = datetime.date.today().isoformat()
        (tdir / "heat_v2.yaml").write_text(candidate_config(base_text, results, date))
        print(f"\nwrote {tdir / 'heat_v2.yaml'}: register it (commit, freeze its hash) before scoring the test days")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default="configs/heat_v1.yaml")
    ap.add_argument("--data-dir", default=THESIS_SIP_DIR)
    ap.add_argument("--out-dir", required=True)
    ap.add_argument("--device", default="auto")
    ap.add_argument("--archs", nargs="*")
    ap.add_argument("--trials", type=int, default=12)
    ap.add_argument("--budgets", type=int, nargs="+", default=[1, 3, 9],
                    help="epochs per successive-halving rung (each epoch is `samples_per_epoch` samples)")
    ap.add_argument("--samples-per-epoch", type=int, default=500_000)
    ap.add_argument("--val-samples", type=int, default=200_000)
    ap.add_argument("--check", action="store_true",
                    help="quick run: train to 2019, validate on 2020, small budgets; writes no candidate config")
    ap.add_argument("--universe")
    ap.add_argument("--panel", help="where the panel lives (default <out-dir>/panel.npz); keep it off Drive: "
                                    "it is ~420 MB and cheap to rebuild")
    ap.add_argument("--rerun", action="store_true")
    args = ap.parse_args()

    base_text = Path(args.config).read_text()
    cfg = config.load(args.config)
    out = Path(args.out_dir)
    sc, mc, tc = cfg["split"], cfg["moments"], cfg["train"]
    if args.check:
        sc = {**sc, "train_end": "2019-12-31", "val_end": "2020-12-31"}
        tdir = out / "tuning_check"
        args.samples_per_epoch, args.val_samples = min(args.samples_per_epoch, 200_000), min(args.val_samples, 100_000)
    else:
        tdir = out / "tuning"
    tdir.mkdir(parents=True, exist_ok=True)

    panel_path = Path(args.panel) if args.panel else out / "panel.npz"
    if panel_path.exists():
        panel = Panel.load(panel_path)
    else:
        symbols = load_universe(args.universe or cfg["data"]["universe"])["symbol"].tolist()
        panel = build_panel(args.data_dir, symbols, cfg["data"]["timeframe"], cfg["data"]["feed"], progress=print)
        panel.save(panel_path)
    print(f"panel: {panel.returns.shape[0]:,} bars x {len(panel.symbols)} stocks")

    wanted = args.archs or tc["archs"]
    todo = [a for a in wanted if args.rerun or not (tdir / f"{a}.json").exists()]
    for a in wanted:
        if a not in todo:
            print(f"{a}: already tuned, skipped (--rerun to tune again)")
    if todo:
        device = pick_device(args.device)
        from heat.cache import cached_peers
        peers = cached_peers(panel, cfg, out / "cache", lambda: build_peers(panel, cfg))
        moments = build_moments(panel, peers, mc["lookback"], mc["stride"], mc["lag"], mc["holds"])
        # Train and val only: the third return value (the test days) is dropped on the spot.
        train, val, _ = split_by_date(moments, panel, sc["train_end"], sc["val_end"], sc["embargo_days"])
        print(f"moments: train {len(train):,}, val {len(val):,}  (the test days are not used)")
        gather, _ = make_gather(panel, peers, train, cfg, device)
        base = {**tc, "samples_per_epoch": args.samples_per_epoch, "val_samples": args.val_samples}
        for arch in todo:
            result, trials = tune_arch(arch, gather, train, val, mc["holds"], mc["lookback"], base,
                                       n_trials=args.trials, budgets=tuple(args.budgets), seed=tc["seed"],
                                       device=device, variants=variants_for(cfg["peers"]))
            trials.to_csv(tdir / f"trials_{arch}.csv", index=False)
            (tdir / f"{arch}.json").write_text(json.dumps(result, indent=2))
    write_outputs(tdir, cfg, base_text, tc["archs"], args)
    print((tdir / "report.md").read_text())


if __name__ == "__main__":
    main()
