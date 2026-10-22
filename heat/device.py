"""Which chip to train on: NVIDIA GPU (cuda), Google TPU, or CPU.

TPUs are reached two ways, depending on the runtime:
- "tpu": PyTorch's native TPU backend (the runtime registers it as a device type named "tpu");
- "xla": the torch_xla library ("xla" device), which needs `sync()` after each step.
Neither can be tested off a TPU, so `self_test` tries each candidate on a real training step of each
architecture, checks its numbers against the CPU's, times it, and the notebook uses the fastest one
that passes. A chip that fails or disagrees is skipped with its reason printed.
"""
from __future__ import annotations

import time

import numpy as np
import torch

CANDIDATES = ("cuda", "tpu", "xla", "cpu")


def resolve(name: str) -> str:
    """A device string torch accepts. 'xla' imports torch_xla so the device type exists."""
    if name.startswith("xla"):
        import torch_xla  # noqa: F401  (registers the 'xla' device type)
        try:
            return str(torch_xla.device())
        except AttributeError:  # older torch_xla
            import torch_xla.core.xla_model as xm
            return str(xm.xla_device())
    return name


def sync(device) -> None:
    """Run what has been queued. Needed for torch_xla (it builds a graph lazily); a no-op elsewhere."""
    if str(device).startswith("xla"):
        import torch_xla
        if hasattr(torch_xla, "sync"):
            torch_xla.sync()
        else:
            import torch_xla.core.xla_model as xm
            xm.mark_step()


def to_cpu_state(model) -> dict:
    """A CPU copy of the weights: safe to keep, compare and save whatever device trained them."""
    return {k: v.detach().to("cpu").clone() for k, v in model.state_dict().items()}


def _probe(name):
    """(device string, None) if a tiny tensor round-trips on this device, else (None, reason)."""
    try:
        if name == "cuda" and not torch.cuda.is_available():
            return None, "no NVIDIA GPU"
        dev = resolve(name)
        x = torch.arange(6, dtype=torch.float32, device=dev)
        if float((x * 2).sum().to("cpu")) != 30.0:
            return None, "wrong arithmetic"
        return dev, None
    except Exception as exc:  # noqa: BLE001
        return None, f"{type(exc).__name__}: {str(exc).splitlines()[0][:120] if str(exc) else ''}"


def _train_steps(arch, dev, steps, batch, seed=0, rows=20000, stocks=60, lookback=12, k=5, size=64):
    """Time `steps` training steps of `arch` on `dev` with the real input path (gather from a panel on
    the device by index), plus a fixed-input forward pass to compare against the CPU."""
    from .models import make_model

    g = torch.Generator().manual_seed(seed)
    R = torch.randn(rows, stocks + 1, generator=g).to(dev)
    torch.manual_seed(seed)
    model = make_model(arch, 1 + k, 4, lookback, size).to(dev)
    opt = torch.optim.AdamW(model.parameters(), lr=1e-3)
    offsets = torch.arange(lookback - 1, -1, -1).to(dev)
    t_all = torch.randint(lookback, rows, (steps + 1, batch), generator=g)
    s_all = torch.randint(0, stocks, (steps + 1, batch, 1 + k), generator=g)
    y_all = torch.randn(steps + 1, batch, 4, generator=g)

    def step(i):
        t, s, y = t_all[i].to(dev), s_all[i].to(dev), y_all[i].to(dev)
        x = R[(t[:, None] - offsets[None, :])[:, :, None], s[:, None, :]]
        loss = ((model(x) - y) ** 2).mean()
        opt.zero_grad()
        loss.backward()
        opt.step()
        sync(dev)
        return loss

    float(step(0).detach().to("cpu"))  # the first step compiles on a TPU: not timed
    start = time.perf_counter()
    for i in range(1, steps + 1):
        loss = step(i)
    final = float(loss.detach().to("cpu"))
    per_step = (time.perf_counter() - start) / steps

    torch.manual_seed(seed + 1)
    ref = make_model(arch, 1 + k, 4, lookback, size).eval()
    probe = make_model(arch, 1 + k, 4, lookback, size).to(dev).eval()
    probe.load_state_dict(ref.state_dict())
    xin = torch.randn(256, lookback, 1 + k, generator=g)
    with torch.no_grad():
        want = ref(xin).numpy()
        got = probe(xin.to(dev)).to("cpu").numpy()
        sync(dev)
    return per_step, final, float(np.max(np.abs(got - want)) / (np.max(np.abs(want)) + 1e-9))


def self_test(archs=("bilstm", "resnet1d", "transformer"), candidates=CANDIDATES, steps=15, batch=4096,
              tol=0.05, log=print):
    """Try every candidate device on every architecture. Returns ({arch: best device}, rows), where a row
    is (arch, device, seconds per step or None, note). A device counts only if training runs and its
    output agrees with the CPU's to within `tol` (relative; TPUs compute in lower precision)."""
    alive = {}
    for name in candidates:
        dev, why = _probe(name)
        if dev is None:
            log(f"  {name}: not available ({why})")
        else:
            alive[name] = dev
    rows, best = [], {}
    for arch in archs:
        timings = {}
        for name, dev in alive.items():
            try:
                per_step, final, err = _train_steps(arch, dev, steps, batch)
                if not np.isfinite(final):
                    rows.append((arch, name, None, "training gave NaN"))
                elif err > tol:
                    rows.append((arch, name, None, f"output differs from CPU by {err:.1%}"))
                else:
                    timings[name] = per_step
                    rows.append((arch, name, per_step, f"agrees with CPU (max difference {err:.2%})"))
            except Exception as exc:  # noqa: BLE001
                rows.append((arch, name, None, f"failed: {type(exc).__name__}: {str(exc).splitlines()[0][:100] if str(exc) else ''}"))
        best[arch] = alive[min(timings, key=timings.get)] if timings else "cpu"
        for a, name, sec, note in rows:
            if a == arch:
                log(f"  {arch:11s} {name:5s} " + (f"{sec * 1000:8.1f} ms/step" if sec else "        --") + f"  {note}")
        log(f"  -> {arch} trains on {best[arch]}")
    return best, rows
