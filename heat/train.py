"""Training. The three variants of an architecture (own / peers / random) get the same initial
weights, the same batches in the same order, and the same budget; only the peer channels differ.
Early stopping uses the validation days only; the test days are never seen here."""
from __future__ import annotations

import copy
import time

import numpy as np
import torch

from .dataset import Gather, Moments, to_device
from .models import make_model


def masked_mse(pred, y):
    ok = torch.isfinite(y)
    diff = torch.where(ok, pred - torch.nan_to_num(y), torch.zeros_like(pred))
    return (diff ** 2).sum() / ok.sum().clamp(min=1)


def _loss_on(model, gather, data, variant, y_scale, batch):
    t, s, mo, y = data
    model.eval()
    total, n = 0.0, 0
    with torch.no_grad():
        for i in range(0, len(t), batch):
            sl = slice(i, i + batch)
            pred = model(gather(t[sl], s[sl], mo[sl], variant))
            yy = y[sl] / y_scale
            ok = torch.isfinite(yy)
            total += float(((pred - torch.nan_to_num(yy)) ** 2)[ok].sum())
            n += int(ok.sum())
    return total / max(n, 1)


def train_variant(arch, variant, gather: Gather, train: Moments, val: Moments, holds, lookback, tcfg,
                  device="cpu", log=print):
    """Returns (model, info). `tcfg`: size, lr, weight_decay, batch, samples_per_epoch, max_epochs,
    patience, val_samples, seed."""
    torch.manual_seed(tcfg["seed"])  # same init for every variant of an architecture
    hp = {k: tcfg[k] for k in ("dropout", "depth", "kernel", "heads") if k in tcfg}
    model = make_model(arch, 1 + gather.k, len(holds), lookback, tcfg["size"], hp).to(device)
    opt = torch.optim.AdamW(model.parameters(), lr=tcfg["lr"], weight_decay=tcfg["weight_decay"])
    y_scale = torch.from_numpy(np.nanstd(train.y, axis=0).astype(np.float32)).to(device)

    rng = np.random.default_rng(tcfg["seed"])  # same batches for every variant
    val_idx = rng.permutation(len(val))[: tcfg["val_samples"]]
    vdata = to_device(val.subset(np.sort(val_idx)), device)
    tr = to_device(train, device)

    best = (np.inf, None, 0)
    stale = 0
    for epoch in range(1, tcfg["max_epochs"] + 1):
        model.train()
        order = rng.choice(len(train), size=min(tcfg["samples_per_epoch"], len(train)), replace=False)
        start = time.time()
        for i in range(0, len(order), tcfg["batch"]):
            idx = torch.from_numpy(order[i:i + tcfg["batch"]]).to(device)
            x = gather(tr[0][idx], tr[1][idx], tr[2][idx], variant)
            loss = masked_mse(model(x), tr[3][idx] / y_scale)
            opt.zero_grad()
            loss.backward()
            opt.step()
        v = _loss_on(model, gather, vdata, variant, y_scale, tcfg["batch"])
        log(f"  {arch}/{variant} epoch {epoch}: val {v:.5f} ({time.time() - start:.0f}s)")
        if v < best[0] - 1e-6:
            best, stale = (v, copy.deepcopy(model.state_dict()), epoch), 0
        else:
            stale += 1
            if stale >= tcfg["patience"]:
                break
    model.load_state_dict(best[1])
    return model, {"best_epoch": best[2], "val_loss": best[0], "y_scale": y_scale.cpu().numpy()}


def predict(model, gather: Gather, m: Moments, variant, y_scale, batch=8192, device="cpu", x_hook=None):
    """Predictions in return units, (n, H). `x_hook(x, sl)` may alter each input batch."""
    t, s, mo, _ = to_device(m, device)
    scale = torch.as_tensor(y_scale, device=device)
    out = []
    model.eval()
    with torch.no_grad():
        for i in range(0, len(t), batch):
            sl = slice(i, i + batch)
            x = gather(t[sl], s[sl], mo[sl], variant)
            if x_hook is not None:
                x = x_hook(x, sl)
            out.append((model(x) * scale).cpu().numpy())
    return np.concatenate(out) if out else np.zeros((0, len(scale)), np.float32)
