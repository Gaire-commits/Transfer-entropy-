"""Steps 1-2: group split, then contiguous time split with gaps."""
import warnings
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


MIN_BLOCK = 50  # smallest train/val/test block, in samples


def _check(block, name):
    if len(block) < MIN_BLOCK:
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


def day_split(sessions, test_ids=(), embargo=1, frac=(0.6, 0.2, 0.2)):
    """Whole sessions (e.g. trading days) go to train, val or test, in time order.

    Session ids must sort chronologically (e.g. "2025-07-03"). Without `test_ids`
    the last frac[2] of sessions is test. `embargo` sessions are dropped before val
    and before test, so the sets are separated in time. Here `gap` counts sessions,
    not samples. Every block is a whole day, so windows never cross a day boundary.
    """
    keys = sorted(sessions)
    if test_ids:
        test_keys = sorted(set(test_ids) & set(keys))
        rest = [k for k in keys if k < test_keys[0]] if test_keys else []
        rest = rest[: max(0, len(rest) - embargo)]  # embargo before test
    else:
        n_test = max(1, int(round(frac[2] * len(keys))))
        test_keys = keys[-n_test:]
        rest = keys[: max(0, len(keys) - n_test - embargo)]
    n_train = int(round(frac[0] / (frac[0] + frac[1]) * len(rest)))
    train_keys, val_keys = rest[:n_train], rest[n_train + embargo :]
    if not (train_keys and val_keys and test_keys):
        raise ValueError(f"too few sessions for a train/val/test split with embargo={embargo}")
    pick = lambda ks: [sessions[k] for k in ks]
    return Split(pick(train_keys), pick(val_keys), pick(test_keys), embargo)


def group_split(sessions, test_ids, gap, train_frac=0.75):
    """Hold out whole sessions for test; time-split the rest into train/val with a gap.

    Sessions too short for this gap (e.g. half trading days) are skipped with a
    warning instead of failing the whole run.
    """
    train, val, test, skipped = [], [], [], []
    for sid, s in sessions.items():
        if sid in test_ids:
            if len(s) >= MIN_BLOCK:
                test.append(s)
            else:
                skipped.append(sid)
            continue
        i1 = int(train_frac * len(s))
        if i1 < MIN_BLOCK or len(s) - i1 - gap < MIN_BLOCK:
            skipped.append(sid)
            continue
        train.append(s[:i1])
        val.append(s[i1 + gap :])
    if skipped:
        shown = ", ".join(map(str, skipped[:5])) + (" ..." if len(skipped) > 5 else "")
        warnings.warn(f"skipped {len(skipped)} session(s) too short for gap={gap}: {shown}")
    if not train:
        raise ValueError(f"no train/val session is long enough for gap={gap}")
    if not test:
        raise ValueError("no usable test sessions")
    return Split(train, val, test, gap)
