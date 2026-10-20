"""The visual report: transfer by hold, who leads whom, and how the companies relate.

`write_report(out_dir, universe_csv)` reads a run's outputs (cells.csv or cells_partial.csv, heat_flows.csv,
and the panel if it is there) and writes one self-contained HTML file (viz.html, no network needed).
Python shapes the data and lays the stocks out; the page (viz_template.html) draws it.

Layout: stocks sit in sector clusters around a ring, so position says "which industry" and nothing else
needs a colour. Colour is reserved for magnitude (one blue ramp).
"""
from __future__ import annotations

import ast
import json
import math
from pathlib import Path

import numpy as np
import pandas as pd

HOLD_LABELS = {6: "30 min", 12: "1 h", 24: "2 h", 48: "4 h"}
TEMPLATE = Path(__file__).with_name("viz_template.html")
MAX_FLOWS = 3000  # strongest source -> target pairs kept per (architecture, hold)


def hold_label(text) -> str:
    """'60 min' (how a run labels a hold) -> '1 h' (how the report labels it)."""
    bars = int(str(text).split()[0]) // 5
    return HOLD_LABELS.get(bars, f"{bars * 5} min")


def _tuple_cols(df):
    for col in [c for c in df.columns if c.endswith("_ci")]:
        df[col] = df[col].map(lambda v: tuple(ast.literal_eval(v)) if isinstance(v, str) else tuple(v))
    return df


def load_results(out_dir):
    """(cells, heat, complete). `cells` always has heat_transfer; verdict is 'pending' until complete."""
    out = Path(out_dir)
    complete = (out / "cells.csv").exists()
    path = out / ("cells.csv" if complete else "cells_partial.csv")
    if not path.exists():
        raise FileNotFoundError(f"no cells.csv or cells_partial.csv in {out}: score at least one architecture first")
    cells = _tuple_cols(pd.read_csv(path))
    if "verdict" not in cells:
        cells["verdict"] = "pending"
    lo = lambda col: cells[col].map(lambda c: c[0])
    cells["heat_transfer"] = (lo("G_peers_ci") > 0) & (lo("G_peers_minus_random_ci") > 0)
    heat_path = out / "heat_flows.csv"
    heat = pd.read_csv(heat_path) if heat_path.exists() and heat_path.stat().st_size > 1 else pd.DataFrame()
    return cells, heat, complete


def sector_layout(stocks, sectors, width=1000, height=860):
    """Positions (x, y) for each stock: sector clusters on an ring, golden-angle spirals inside."""
    order = pd.Series(sectors).value_counts().index.tolist()
    n_sec = len(order)
    cx, cy = width / 2, height / 2
    ring = 0 if n_sec == 1 else min(width, height) * 0.38
    pos, centers = {}, {}
    for k, sec in enumerate(order):
        ang = -math.pi / 2 + 2 * math.pi * k / max(n_sec, 1)
        sx, sy = cx + ring * math.cos(ang), cy + ring * math.sin(ang)
        members = [s for s, sc in zip(stocks, sectors) if sc == sec]
        step = 13.0 if n_sec > 1 else 20.0
        for i, s in enumerate(members):
            r, th = step * math.sqrt(i + 0.5), i * 2.399963  # golden angle
            pos[s] = (round(sx + r * math.cos(th), 1), round(sy + r * math.sin(th), 1))
        centers[sec] = (round(sx, 1), round(sy, 1), len(members), round(step * math.sqrt(len(members) + 0.5), 1))
    return pos, centers, order


def peer_relations(panel, universe: pd.DataFrame, k=5, peer_days=60, min_share=0.8):
    """The latest month's correlated peers (E12's rule): edges (a, b, rho) and how often a peer shares the
    stock's sector, against what random peers would give."""
    from .peers import monthly_peers

    pt = monthly_peers(panel, k, peer_days, min_share)
    if not len(pt.correlated):
        return None
    peers, rho = pt.correlated[-1], pt.rho[-1]
    sector = dict(zip(universe["symbol"], universe["sector"]))
    syms = panel.symbols
    edges, same = [], []
    for i in range(len(syms)):
        for j in range(peers.shape[1]):
            p = peers[i, j]
            if p >= 0:
                edges.append([syms[i], syms[p], round(float(rho[i, j]), 3)])
                same.append(sector.get(syms[i]) == sector.get(syms[p]))
    secs = pd.Series([sector.get(s) for s in syms]).value_counts(normalize=True)
    return {"edges": edges, "same_sector": float(np.mean(same)) if same else None,
            "random_same_sector": float((secs ** 2).sum())}


def build_payload(cells, heat, complete, universe, cfg_meta, peers=None, top_stocks=None):
    sector = dict(zip(universe["symbol"], universe["sector"]))
    name = dict(zip(universe["symbol"], universe.get("name", universe["symbol"])))
    holds = sorted(cells["hold_bars"].unique())
    archs = list(dict.fromkeys(cells["arch"]))
    hold_labels = [HOLD_LABELS.get(int(h), f"{int(h) * 5} min") for h in holds]

    flows, used = {}, set()
    for (arch, hold), g in (heat.groupby(["arch", "hold"]) if len(heat) else []):
        g = g.sort_values("heat_share", ascending=False).head(MAX_FLOWS)
        flows[f"{arch}|{hold_label(hold)}"] = g[["source", "target", "heat_share", "n"]].values.tolist()
        used.update(g["source"].tolist() + g["target"].tolist())
    peer_syms = set(sum(([e[0], e[1]] for e in peers["edges"]), [])) if peers else set()
    stocks = sorted(used | peer_syms)
    secs = [sector.get(s, "Unknown") for s in stocks]
    pos, centers, order = sector_layout(stocks, secs)
    index = {s: i for i, s in enumerate(stocks)}

    def cell_row(r):
        trip = lambda v, ci: [r[v], ci[0], ci[1]]
        return {"arch": r["arch"], "hold": HOLD_LABELS.get(int(r["hold_bars"]), r["hold"]),
                "G_peers": trip("G_peers", r["G_peers_ci"]), "G_random": trip("G_random", r["G_random_ci"]),
                "G_diff": trip("G_peers_minus_random", r["G_peers_minus_random_ci"]),
                "net": trip("net_bps_peers", r["net_bps_peers_ci"]), "net_own": r["net_bps_own"],
                "gross": r["gross_bps_peers"], "ic": r["ic_peers"], "verdict": r["verdict"],
                "heat_transfer": bool(r["heat_transfer"]), "moments": int(r["moments"])}

    payload = {
        "meta": {**cfg_meta, "complete": complete, "archs": archs, "holds": hold_labels,
                 "days": int(cells["days"].iloc[0]) if "days" in cells else None},
        "cells": [cell_row(r) for _, r in cells.iterrows()],
        "stocks": [{"s": s, "name": name.get(s, s), "sector": sector.get(s, "Unknown"), "x": pos[s][0], "y": pos[s][1]}
                   for s in stocks],
        "sectors": [{"name": s, "x": centers[s][0], "y": centers[s][1], "n": centers[s][2], "r": centers[s][3]}
                    for s in order],
        "flows": {k: [[index[a], index[b], round(float(v), 5), int(n)] for a, b, v, n in rows] for k, rows in flows.items()},
        "peers": None,
    }
    if peers:
        payload["peers"] = {"edges": [[index[a], index[b], rho] for a, b, rho in peers["edges"]],
                            "same_sector": peers["same_sector"], "random_same_sector": peers["random_same_sector"]}
    return payload


def render(payload) -> str:
    data = json.dumps(payload, separators=(",", ":")).replace("</", "<\\/")
    return TEMPLATE.read_text().replace("/*__DATA__*/null", data)


def write_report(out_dir, universe_csv, cfg_meta=None, panel_path=None, dest=None):
    """Write <out_dir>/viz.html (or `dest`). Returns its path."""
    out = Path(out_dir)
    cells, heat, complete = load_results(out)
    universe = pd.read_csv(universe_csv)
    peers = None
    panel_path = Path(panel_path) if panel_path else out / "panel.npz"
    if not panel_path.exists() and out.name in ("check",) and (out.parent / "panel.npz").exists():
        panel_path = out.parent / "panel.npz"
    if panel_path.exists():
        from .data import Panel

        peers = peer_relations(Panel.load(panel_path), universe)
    meta = {"experiment": "heat", "check": out.name == "check", "cost_bps": 2.0, "level": 0.99, **(cfg_meta or {})}
    dest = Path(dest) if dest else out / "viz.html"
    dest.write_text(render(build_payload(cells, heat, complete, universe, meta, peers)))
    return dest
