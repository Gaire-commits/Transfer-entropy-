"""Test-day evaluation: transfer (the TE-style gain), the trading book, the verdict, and the heat map.

All intervals resample whole test days (moments on one day share that day's market), as in the
thesis. Transfer for a variant is the predictive gain over the no-peer model on the same moments,
G = 1/2 ln(MSE_own / MSE_variant) in nats: the information the peers' past adds about the stock's
hold return beyond the stock's own past (including its own reversal), estimated out of sample.
"""
from __future__ import annotations

import numpy as np
import pandas as pd
import torch

from .dataset import Moments


# ---- day-block bootstrap ------------------------------------------------------
def day_weights(n_days, n_boot, rng):
    draws = rng.integers(0, n_days, (n_boot, n_days))
    return np.stack([np.bincount(d, minlength=n_days) for d in draws]).astype(np.float64)


def _day_sums(values, day_codes, n_days):
    out = np.zeros(n_days)
    np.add.at(out, day_codes, values)
    return out


def interval(boot, level):
    tail = (1 - level) / 2 * 100
    return float(np.percentile(boot, tail)), float(np.percentile(boot, 100 - tail))


def benjamini_hochberg(p, q=0.05):
    p = np.asarray(p, float)
    m = len(p)
    order = np.argsort(p)
    passed = p[order] <= q * np.arange(1, m + 1) / m
    keep = np.zeros(m, bool)
    if passed.any():
        keep[order[: np.flatnonzero(passed).max() + 1]] = True
    return keep


# ---- the quintile book ----------------------------------------------------------
def book_positions(pred, y, t, q=0.2, min_stocks=20):
    """Long the top `q`, short the bottom `q` of predictions at each signal bar. Returns a mask of
    positions and their signed returns (long y, short -y). Moments with fewer than `min_stocks`
    valid stocks are skipped."""
    ok = np.isfinite(y) & np.isfinite(pred)
    df = pd.DataFrame({"t": t[ok], "p": pred[ok]})
    size = df.groupby("t")["p"].transform("size").to_numpy()
    pct = df.groupby("t")["p"].rank(pct=True, method="first").to_numpy()
    long_, short_ = (pct > 1 - q) & (size >= min_stocks), (pct <= q) & (size >= min_stocks)
    pos = np.zeros(len(y), bool)
    sign = np.zeros(len(y))
    idx = np.flatnonzero(ok)
    pos[idx] = long_ | short_
    sign[idx] = np.where(long_, 1.0, np.where(short_, -1.0, 0.0))
    return pos, sign * np.nan_to_num(y)


def rank_ic_by_moment(pred, y, t, min_stocks=20):
    ok = np.isfinite(y) & np.isfinite(pred)
    df = pd.DataFrame({"t": t[ok], "p": pred[ok], "y": y[ok]})
    df["rp"] = df.groupby("t")["p"].rank()
    df["ry"] = df.groupby("t")["y"].rank()
    g = df.groupby("t")
    ic = g.apply(lambda d: d["rp"].corr(d["ry"]) if len(d) >= min_stocks else np.nan, include_groups=False)
    return ic.dropna()


# ---- one architecture, all holds ----------------------------------------------
def evaluate_arch(preds: dict, test: Moments, holds, panel_days, ecfg, rng):
    """`preds[variant]` is (n, H) on `test`. Returns one row per hold with every reported number."""
    level, n_boot, cost = ecfg["level"], ecfg["bootstrap"], ecfg["cost_bps"] * 1e-4
    days, codes = np.unique(test.day, return_inverse=True)
    n_days = len(days)
    W = day_weights(n_days, n_boot, rng)
    half = np.arange(n_days) < n_days // 2  # first half of the test days
    rows = []
    for j, h in enumerate(holds):
        y = test.y[:, j]
        ok = np.isfinite(y)
        err = {v: np.where(ok, (preds[v][:, j] - np.nan_to_num(y)) ** 2, 0.0) for v in preds}
        sse = {v: _day_sums(err[v], codes, n_days) for v in err}
        gain = lambda v, w: 0.5 * np.log((w @ sse["own"]) / (w @ sse[v]))
        ones = np.ones(n_days)
        row = {"hold_bars": h, "hold": f"{h * 5} min", "moments": int(ok.sum()), "days": n_days}
        for v in ("peers", "random"):
            row[f"G_{v}"] = float(gain(v, ones))
            row[f"G_{v}_ci"] = interval(gain(v, W), level)
        diff = 0.5 * np.log((W @ sse["random"]) / (W @ sse["peers"]))
        row["G_peers_minus_random"] = float(0.5 * np.log(sse["random"].sum() / sse["peers"].sum()))
        row["G_peers_minus_random_ci"] = interval(diff, level)

        net = {}
        for v in preds:
            pos, ret = book_positions(preds[v][:, j], y, test.t, ecfg["q"], ecfg["min_stocks"])
            s_ret, s_cnt = _day_sums(ret, codes, n_days), _day_sums(pos.astype(float), codes, n_days)
            net[v] = (W @ s_ret) / (W @ s_cnt) - cost
            gross = s_ret.sum() / max(s_cnt.sum(), 1)
            row[f"gross_bps_{v}"] = gross * 1e4
            row[f"net_bps_{v}"] = (gross - cost) * 1e4
            row[f"net_bps_{v}_ci"] = tuple(x * 1e4 for x in interval(net[v], level))
            row[f"positions_{v}"] = int(s_cnt.sum())
            if v == "peers":
                for name, m in (("first_half", half), ("second_half", ~half)):
                    row[f"net_bps_peers_{name}"] = (s_ret[m].sum() / max(s_cnt[m].sum(), 1) - cost) * 1e4
            ic = rank_ic_by_moment(preds[v][:, j], y, test.t, ecfg["min_stocks"])
            row[f"ic_{v}"] = float(ic.mean()) if len(ic) else np.nan
        d = net["peers"] - net["own"]
        row["net_bps_peers_minus_own"] = row["net_bps_peers"] - row["net_bps_own"]
        row["net_bps_peers_minus_own_ci"] = tuple(x * 1e4 for x in interval(d, level))
        row["p_net_peers"] = float((1 + np.sum(net["peers"] <= 0)) / (n_boot + 1))
        rows.append(row)
    return rows


def verdicts(table: pd.DataFrame, q=0.05) -> pd.DataFrame:
    """The registered rule (configs: `protocol`), applied to every (arch, hold) cell."""
    t = table.copy()
    t["fdr"] = benjamini_hochberg(t["p_net_peers"], q)
    checks = {
        "net_positive": t["net_bps_peers_ci"].map(lambda c: c[0] > 0),
        "beats_no_peers": t["net_bps_peers_minus_own_ci"].map(lambda c: c[0] > 0),
        "beats_random_peers": t["G_peers_minus_random_ci"].map(lambda c: c[0] > 0),
        "both_halves_positive": (t["net_bps_peers_first_half"] > 0) & (t["net_bps_peers_second_half"] > 0),
        "fdr": t["fdr"],
    }
    for k, v in checks.items():
        t[k] = v.astype(bool)
    failed = lambda r: [k for k in checks if not r[k]]
    t["verdict"] = t.apply(lambda r: "EDGE" if not failed(r) else "NO EDGE (failed: " + ", ".join(failed(r)) + ")", axis=1)
    t["heat_transfer"] = t["G_peers_minus_random_ci"].map(lambda c: c[0] > 0) & t["G_peers_ci"].map(lambda c: c[0] > 0)
    return t


# ---- where the heat goes ----------------------------------------------------------
def heat_flows(model, gather, test: Moments, y_scale, hold_index, symbols, max_samples=200_000,
               batch=8192, device="cpu", rng=None, min_count=50, z=3.0):
    """Which peer's past carries information about which stock: for each peer slot, swap that
    peer's history for the slot-mate of another stock at the same signal bar and measure how much
    the stock's squared error rises. Aggregated by (source = the peer, target = the stock) as the
    share of the target's error that the source explains."""
    rng = rng or np.random.default_rng(0)
    y = test.y[:, hold_index]
    idx = np.flatnonzero(np.isfinite(y))
    if len(idx) > max_samples:
        idx = np.sort(rng.choice(idx, max_samples, replace=False))
    m = test.subset(idx)
    y = m.y[:, hold_index]
    t = torch.from_numpy(m.t).long().to(device)
    s = torch.from_numpy(m.s).long().to(device)
    mo = torch.from_numpy(m.month).long().to(device)
    stocks = gather.stocks(s, mo, "peers")  # (n, 1 + k)
    scale = float(y_scale[hold_index])

    def errors(stk):
        out = []
        model.eval()
        with torch.no_grad():
            for i in range(0, len(t), batch):
                sl = slice(i, i + batch)
                times = t[sl, None] - gather.offsets[None, :]
                x = gather.R[times[:, :, None], stk[sl, None, :]]
                out.append(model(x)[:, hold_index].cpu().numpy() * scale)
        return (np.concatenate(out) - y) ** 2

    base = errors(stocks)
    # A permutation of samples within each signal bar: the swapped history is from the same moment.
    order = np.argsort(m.t, kind="stable")
    groups = np.split(order, np.flatnonzero(np.diff(m.t[order])) + 1)
    perm = np.empty(len(m.t), np.int64)
    for g in groups:
        perm[g] = rng.permutation(g)
    perm_t = torch.from_numpy(perm).to(device)

    frames = []
    for k in range(gather.k):
        swapped = stocks.clone()
        swapped[:, k + 1] = stocks[perm_t, k + 1]
        delta = errors(swapped) - base
        src = stocks[:, k + 1].cpu().numpy()
        frames.append(pd.DataFrame({"source": src, "target": m.s, "delta": delta, "base": base}))
    df = pd.concat(frames)
    agg = screen_pairs(df[df["source"] != gather.none], min_count, z)
    agg["source"] = [symbols[i] for i in agg["source"]]
    agg["target"] = [symbols[i] for i in agg["target"]]
    return agg


def screen_pairs(df: pd.DataFrame, min_count=50, z=3.0) -> pd.DataFrame:
    """Per (source, target): n, heat_share = sum(delta) / sum(base), its standard error, and the lower
    bound lcb = share - z * se, strongest bound first. A screening bar, not inference: thousands of
    pairs are looked at, so a pair counts only when its share is z standard errors above zero, and a
    small pair with a lucky share no longer tops the list."""
    df = df.assign(dd=df["delta"] ** 2, db=df["delta"] * df["base"], bb=df["base"] ** 2)
    agg = df.groupby(["source", "target"]).agg(n=("delta", "size"), delta=("delta", "sum"), base=("base", "sum"),
                                                dd=("dd", "sum"), db=("db", "sum"), bb=("bb", "sum"))
    agg = agg[agg["n"] >= min_count]
    share = agg["delta"] / agg["base"]
    # Standard error of a ratio of sums (linearised): residuals r = delta - share * base.
    rss = (agg["dd"] - 2 * share * agg["db"] + share ** 2 * agg["bb"]).clip(lower=0)
    agg["heat_share"] = share
    agg["se"] = np.sqrt(rss * agg["n"] / (agg["n"] - 1).clip(lower=1)) / agg["base"]
    agg["lcb"] = agg["heat_share"] - z * agg["se"]
    agg = agg.reset_index()
    return agg.sort_values("lcb", ascending=False)[["source", "target", "n", "heat_share", "se", "lcb"]]
