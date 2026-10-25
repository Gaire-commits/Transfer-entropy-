"""Exploratory: transfer and the edge on macro release days, from a run's saved predictions (no training).

    python scripts/events.py --config configs/heat_v2.yaml --out-dir <outputs>/heat_v2 --check
    python scripts/events.py --config configs/heat_v2.yaml --out-dir <outputs>/heat_v2

Release dates come from FRED (needs FRED_API_KEY) and are saved to <out-dir>/events_calendar.csv the first
time, so later runs need no key. Writes events.md and events.csv next to the run's results.
"""
import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from heat import config  # noqa: E402
from heat.data import THESIS_SIP_DIR, Panel, build_panel, load_universe  # noqa: E402
from heat.event_analysis import event_table  # noqa: E402
from heat.events import load_or_fetch  # noqa: E402
from heat.run import apply_check  # noqa: E402


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default="configs/heat_v2.yaml")
    ap.add_argument("--data-dir", default=THESIS_SIP_DIR)
    ap.add_argument("--out-dir", required=True, help="the run's output folder (as given to scripts/run.py)")
    ap.add_argument("--check", action="store_true", help="analyse the check run (<out-dir>/check)")
    ap.add_argument("--panel")
    ap.add_argument("--cache-dir")
    ap.add_argument("--calendar", help="release calendar CSV (default <out-dir>/events_calendar.csv)")
    ap.add_argument("--universe")
    ap.add_argument("--bootstrap", type=int, default=1000)
    args = ap.parse_args()

    cfg = config.load(args.config)
    out = Path(args.out_dir)
    target = out / "check" if args.check else out
    if args.check:
        apply_check(cfg)
    panel_path = Path(args.panel) if args.panel else out / "panel.npz"
    if panel_path.exists():
        panel = Panel.load(panel_path)
    else:
        symbols = load_universe(args.universe or cfg["data"]["universe"])["symbol"].tolist()
        panel = build_panel(args.data_dir, symbols, cfg["data"]["timeframe"], cfg["data"]["feed"], progress=print)
        panel.save(panel_path)
    calendar = load_or_fetch(args.calendar or str(out / "events_calendar.csv"))
    table = event_table(target, panel, cfg, calendar, Path(args.cache_dir) if args.cache_dir else out / "cache",
                        bootstrap=args.bootstrap)
    if len(table):
        print((target / "events.md").read_text())


if __name__ == "__main__":
    main()
