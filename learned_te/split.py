"""Steps 1-2: group split, then contiguous time split with gaps."""
from dataclasses import dataclass

import numpy as np


@dataclass
class Split:
    train: list  # list of 2-D blocks; windows are never built across blocks
    val: list
    test: list
    gap: int


def window_span(k, l, u=1):
    """Number of consecutive samples one (history, target) window touches."""
    return max(k, l + u - 1) + 1


def autocorr_time(series, thresh=0.05, max_lag=500):
    """First lag where |ACF| < thresh, maximised over channels."""
    lags = []
    for col in np.atleast_2d(series.T):
        z = col - col.mean()
        denom = z @ z
        lag = max_lag
        for h in range(1, min(max_lag, len(z) - 1)):
            if abs(z[:-h] @ z[h:] / denom) < thresh:
                lag = h
                break
        lags.append(lag)
    return max(lags)


def choose_gap(series, span, train_frac):
    """G >= max(window span, autocorrelation time), estimated on the train region only."""
    train_region = series[: int(train_frac * len(series))]
    return max(span, autocorr_time(train_region))


def _check(block, name):
    if len(block) < 50:
        raise ValueError(f"{name} block has only {len(block)} samples; series too short for this gap")
    return block


def time_split(series, gap, frac=(0.6, 0.2, 0.2)):
    """[train) gap [val) gap [test) on one continuous series. Gap samples are discarded."""
    n = len(series)
    i1 = int(frac[0] * n)
    i2 = int((frac[0] + frac[1]) * n)
    return Split(
        train=[_check(series[:i1], "train")],
        val=[_check(series[i1 + gap : i2], "val")],
        test=[_check(series[i2 + gap :], "test")],
        gap=gap,
    )


def group_split(sessions, test_ids, gap, train_frac=0.75):
    """Hold out whole sessions for test; time-split the rest into train/val with a gap."""
    train, val, test = [], [], []
    for sid, s in sessions.items():
        if sid in test_ids:
            test.append(_check(s, f"test session {sid}"))
            continue
        i1 = int(train_frac * len(s))
        train.append(_check(s[:i1], f"train session {sid}"))
        val.append(_check(s[i1 + gap :], f"val session {sid}"))
    if not test:
        raise ValueError("no test sessions selected")
    return Split(train, val, test, gap)
