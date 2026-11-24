"""One way to get the panel, whatever the config's data source: a saved panel, the thesis 5-minute SIP
cache (source: sip, the default), or Yahoo Finance daily bars (source: yahoo)."""
from __future__ import annotations

from pathlib import Path

from .data import Panel, build_panel, load_universe


def get_panel(cfg, data_dir, panel_path, universe=None, log=print) -> Panel:
    panel_path = Path(panel_path)
    if panel_path.exists():
        panel = Panel.load(panel_path)
        check_panel(panel, cfg, panel_path)
        log(f"panel loaded: {panel.returns.shape[0]:,} rows x {len(panel.symbols)} stocks")
        return panel
    d = cfg["data"]
    symbols = load_universe(universe or d["universe"])["symbol"].tolist()
    if d.get("source", "sip") == "yahoo":
        from .yahoo import cached_daily, daily_panel

        start = d.get("start", "2000-01-01")
        closes = cached_daily(Path(data_dir) / f"yahoo_daily_from_{start}.parquet", symbols, start, log=log)
        panel = daily_panel(closes)
    else:
        panel = build_panel(data_dir, symbols, d["timeframe"], d["feed"], progress=log)
    panel.save(panel_path)
    log(f"panel built and saved: {panel.returns.shape[0]:,} rows x {len(panel.symbols)} stocks")
    return panel


def check_panel(panel, cfg, where="the panel"):
    """Stop if the panel's bars don't match the config's: a daily panel run with 5-minute settings (or the
    reverse) produces numbers that mean nothing, without any error to say so."""
    daily_panel = panel.session is not None
    daily_cfg = cfg["data"].get("timeframe", "5Min") == "1Day"
    if daily_panel != daily_cfg:
        have = "daily" if daily_panel else "5-minute"
        want = "daily" if daily_cfg else "5-minute"
        raise SystemExit(f"{where} holds {have} bars but {cfg['experiment']} expects {want} bars. Pass the config "
                         f"that matches the data (--config), or the panel built for this experiment (--panel).")
