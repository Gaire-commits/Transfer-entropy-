"""Run a registered experiment on the thesis bars.

    python scripts/run.py --config configs/heat_v1.yaml \
        --data-dir /content/drive/MyDrive/entropy-gate-pipeline-data/data/bars_sip \
        --out-dir  /content/drive/MyDrive/entropy-gate-pipeline-data/outputs/heat_v1

The panel (every stock's market-relative returns on the grid) is built once and saved to
<out-dir>/panel.npz; later runs load it. --archs runs only some architectures (e.g. one per Colab
session); each is saved as soon as it is scored and skipped next time, and the verdict appears once
all the config's architectures are in. --rerun scores already-scored ones again (disclose it).
--check is a quick run that stays inside the training years.
"""
import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from heat import config  # noqa: E402
from heat.data import THESIS_SIP_DIR, Panel, build_panel, load_universe  # noqa: E402
from heat.run import run_experiment  # noqa: E402
from heat.viz import write_report  # noqa: E402


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default="configs/heat_v1.yaml")
    ap.add_argument("--data-dir", default=THESIS_SIP_DIR)
    ap.add_argument("--out-dir", required=True)
    ap.add_argument("--device", default="auto")
    ap.add_argument("--archs", nargs="*", help="override the config's architectures (e.g. one per session)")
    ap.add_argument("--check", action="store_true",
                    help="a quick run inside the training years (train to 2019, val 2020, score 2021); "
                         "never touches the registered test days, writes to <out-dir>/check")
    ap.add_argument("--universe", help="symbols CSV (default: the config's universe)")
    ap.add_argument("--panel", help="a panel.npz built earlier (e.g. by another experiment's run); "
                                    "default <out-dir>/panel.npz, built if missing")
    ap.add_argument("--rerun", action="store_true",
                    help="score already-scored architectures again, retraining them (disclose it in the write-up)")
    ap.add_argument("--cache-dir", help="where peer/TE tables are cached (default <out-dir>/cache); shared by the "
                                        "check run and the registered run")
    args = ap.parse_args()

    cfg = config.load(args.config)
    out = Path(args.out_dir)
    cache_dir = Path(args.cache_dir) if args.cache_dir else out / "cache"
    panel_path = Path(args.panel) if args.panel else out / "panel.npz"
    if args.check:
        cfg["split"].update(train_end="2019-12-31", val_end="2020-12-31", test_end="2021-12-31")
        cfg["train"].update(samples_per_epoch=300_000, max_epochs=3, val_samples=100_000)
        cfg["evaluation"].update(bootstrap=300, heat_samples=50_000)
        cfg["check"] = "2021 (inside the registered training years)"
        out = out / "check"
    if panel_path.exists():
        panel = Panel.load(panel_path)
        print(f"panel loaded: {panel.returns.shape[0]:,} bars x {len(panel.symbols)} stocks")
    else:
        symbols = load_universe(args.universe or cfg["data"]["universe"])["symbol"].tolist()
        panel = build_panel(args.data_dir, symbols, cfg["data"]["timeframe"], cfg["data"]["feed"], progress=print)
        panel.save(panel_path)
        print(f"panel built and saved: {panel.returns.shape[0]:,} bars x {len(panel.symbols)} stocks")

    # Architectures already scored in <out> are skipped, so a dropped session just resumes.
    run_experiment(panel, cfg, out, args.device, archs=args.archs, rerun=args.rerun, cache_dir=cache_dir)
    print((out / "report.md").read_text())
    try:  # the pictures must never break a run
        ev = cfg["evaluation"]
        meta = {"experiment": cfg["experiment"], "cost_bps": ev["cost_bps"], "level": ev["level"],
                "check": bool(cfg.get("check"))}
        print("visual report:", write_report(out, args.universe or cfg["data"]["universe"], meta, panel_path))
    except Exception as exc:  # noqa: BLE001
        print(f"(visual report skipped: {exc})")


if __name__ == "__main__":
    main()
