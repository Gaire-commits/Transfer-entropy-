"""Choosing the chip: unavailable ones are skipped with a reason, disagreeing ones are rejected, and the
training code runs unchanged on any device the self-test accepts."""
import numpy as np
import torch

import heat.device as device
from heat.device import resolve, self_test, sync, to_cpu_state
from heat.models import make_model


def test_cpu_passes_and_unavailable_devices_are_skipped():
    log = []
    best, rows = self_test(archs=("resnet1d",), candidates=("cuda", "xla", "cpu"), steps=2, batch=64, log=log.append)
    assert best == {"resnet1d": "cpu"}
    if not torch.cuda.is_available():
        assert any(m.startswith("  cuda: not available") for m in log)
    assert any(r[1] == "cpu" and r[2] is not None for r in rows)


def test_device_that_disagrees_with_cpu_is_rejected(monkeypatch):
    monkeypatch.setattr(device, "_train_steps", lambda *a, **k: (0.001, 1.0, 0.5))  # fast but 50% off
    best, rows = self_test(archs=("bilstm",), candidates=("cpu",), log=lambda m: None)
    assert best == {"bilstm": "cpu"} and rows[0][2] is None and "differs from CPU" in rows[0][3]


def test_fastest_passing_device_wins(monkeypatch):
    monkeypatch.setattr(device, "_probe", lambda name: (name, None))
    times = {"cuda": 0.05, "tpu": 0.01, "cpu": 0.5}
    monkeypatch.setattr(device, "_train_steps", lambda arch, dev, *a, **k: (times[dev], 1.0, 0.001))
    best, _ = self_test(archs=("transformer",), candidates=("cuda", "tpu", "cpu"), log=lambda m: None)
    assert best == {"transformer": "tpu"}


def test_helpers_on_cpu():
    assert resolve("cpu") == "cpu"
    sync("cpu")  # no-op
    m = make_model("resnet1d", 6, 4, 12, 8)
    st = to_cpu_state(m)
    assert all(v.device.type == "cpu" for v in st.values())
    st[next(iter(st))].add_(1.0)  # a copy: changing it leaves the model alone
    assert not torch.equal(st[next(iter(st))], m.state_dict()[next(iter(st))])
