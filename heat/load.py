"""One way to get the panel, whatever the config's data source: a saved panel, the thesis 5-minute SIP
cache (source: sip, the default), or Yahoo Finance daily bars (source: yahoo)."""
from __future__ import annotations

from pathlib import Path

from .data import Panel, build_panel, load_universe


def get_panel(cfg, data_dir, panel_path, universe=None, log=print) -> Panel:
    panel_path = Path(panel_path)
    if panel_path.exists():
        panel = Panel.load(panel_path)
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
