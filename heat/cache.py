"""Don't compute anything twice.

- The peer tables (with transfer entropy, ~80 s for the S&P 500) are cached by a key made of the panel's
  fingerprint and the settings that shape them; any change to either gives a new key, so a stale table
  is never reused.
- Each trained variant (model weights + its test predictions) is saved the moment it is done, keyed by
  everything that shapes it, so a rerun picks up exactly where the last one stopped.
"""
from __future__ import annotations

import hashlib
import json
from pathlib import Path

import numpy as np

from .peers import PeerTable


def panel_fingerprint(panel) -> str:
    """Cheap but specific: shape, symbols, first/last bar, and a strided sample of the returns."""
    h = hashlib.sha256()
    h.update(json.dumps([list(panel.returns.shape), panel.symbols, str(panel.times[0]), str(panel.times[-1])]).encode())
    sample = np.nan_to_num(panel.returns[:: max(1, len(panel.returns) // 4000)], nan=-9.0)
    h.update(np.ascontiguousarray(sample).tobytes())
    return h.hexdigest()[:16]


def key(*parts) -> str:
    return hashlib.sha256(json.dumps(parts, sort_keys=True, default=str).encode()).hexdigest()[:16]


def peers_key(panel, cfg) -> str:
    pc, mc = cfg["peers"], cfg["moments"]
    return key(panel_fingerprint(panel), pc, mc["lookback"], mc["lag"], list(mc["holds"]))


def save_peers(path, t: PeerTable):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    arrays = {"month_start_day": t.month_start_day, "month_of_day": t.month_of_day, "correlated": t.correlated,
              "random": t.random, "rho": t.rho, "feature_holds": np.array(t.feature_holds, dtype=np.int64)}
    if t.te is not None:
        arrays["te"] = t.te
    for name, f in t.features.items():
        arrays[f"feat_{name}"] = f
    tmp = path.with_suffix(".tmp.npz")
    np.savez_compressed(tmp, **arrays)
    tmp.replace(path)  # whole file or nothing: a killed save never leaves a half-written cache


def load_peers(path) -> PeerTable:
    z = np.load(path, allow_pickle=False)
    feats = {k[5:]: z[k] for k in z.files if k.startswith("feat_")}
    return PeerTable(z["month_start_day"], z["month_of_day"], z["correlated"], z["random"], z["rho"],
                     te=z["te"] if "te" in z.files else None, features=feats,
                     feature_holds=tuple(int(h) for h in z["feature_holds"]))


def cached_peers(panel, cfg, cache_dir, build, log=print):
    """The peer tables from `cache_dir` if this exact panel and these settings were seen before,
    else `build()` them and save them there."""
    if cache_dir is None:
        return build()
    path = Path(cache_dir) / f"peers_{peers_key(panel, cfg)}.npz"
    if path.exists():
        log(f"peers: loaded from cache ({path.name})")
        return load_peers(path)
    t = build()
    save_peers(path, t)
    log(f"peers: built and cached ({path.name})")
    return t


def variant_key(panel_fp, cfg, arch, variant, n_test) -> str:
    """Everything that shapes a trained variant and its test predictions."""
    tc = {k: v for k, v in cfg["train"].items() if k != "archs"}
    return key(panel_fp, cfg["peers"], cfg["moments"], cfg["split"], tc, arch, variant, n_test)


def save_variant(dirpath, arch, variant, vkey, state, preds, info):
    d = Path(dirpath)
    d.mkdir(parents=True, exist_ok=True)
    import torch

    torch.save(state, d / f"{arch}_{variant}.pt")
    tmp = d / f"{arch}_{variant}.tmp.npz"
    np.savez_compressed(tmp, key=np.array(vkey), preds=preds, y_scale=info["y_scale"],
                        best_epoch=np.array(info["best_epoch"]), val_loss=np.array(info["val_loss"]))
    tmp.replace(d / f"{arch}_{variant}.npz")


def load_variant(dirpath, arch, variant, vkey):
    """(state, preds, info) if this exact variant was trained before, else None."""
    d = Path(dirpath)
    npz, pt = d / f"{arch}_{variant}.npz", d / f"{arch}_{variant}.pt"
    if not (npz.exists() and pt.exists()):
        return None
    z = np.load(npz, allow_pickle=False)
    if str(z["key"]) != vkey:
        return None  # trained under other settings or data: never reused
    import torch

    info = {"y_scale": z["y_scale"], "best_epoch": int(z["best_epoch"]), "val_loss": float(z["val_loss"])}
    return torch.load(pt, map_location="cpu"), z["preds"], info
