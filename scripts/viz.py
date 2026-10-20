"""Write the visual report for a run: <out-dir>/viz.html (self-contained, opens in any browser).

    python scripts/viz.py --out-dir <outputs>/heat_v1/check
    python scripts/viz.py --out-dir <outputs>/heat_v2 --panel <outputs>/heat_v1/panel.npz
"""
import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from heat import config  # noqa: E402
from heat.viz import write_report  # noqa: E402


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out-dir", required=True)
    ap.add_argument("--config", default="configs/heat_v1.yaml")
    ap.add_argument("--universe", help="symbols CSV with sector and name (default: the config's universe)")
    ap.add_argument("--panel", help="panel.npz for the 'who moves with whom' map (default: <out-dir>/panel.npz)")
    args = ap.parse_args()
    cfg = config.load(args.config)
    ev = cfg["evaluation"]
    meta = {"experiment": cfg["experiment"], "cost_bps": ev["cost_bps"], "level": ev["level"],
            "check": Path(args.out_dir).name == "check"}
    print(write_report(args.out_dir, args.universe or cfg["data"]["universe"], meta, args.panel))


if __name__ == "__main__":
    main()
