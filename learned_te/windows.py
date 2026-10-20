"""Step 5: build (history, target) windows inside each block, never across blocks."""
from dataclasses import dataclass

import numpy as np

from .split import window_span


@dataclass
class Windows:
    yh: np.ndarray  # (n, k)  Y_t, Y_{t-1}, ..., Y_{t-k+1}
    xh: np.ndarray  # (n, l)  X_{t+1-u}, ..., X_{t+1-u-l+1}
    target: np.ndarray  # (n,) Y_{t+1}
    block_id: np.ndarray  # (n,) which block each sample came from

    def __len__(self):
        return len(self.target)

    @property
    def xy(self):
        return np.column_stack([self.yh, self.xh])


def make_windows(block, k, l, u=1):
    x, y = block[:, 0], block[:, 1]
    span = window_span(k, l, u)
    t = np.arange(span - 2, len(block) - 1)  # last history index; target at t+1
    yh = np.stack([y[t - j] for j in range(k)], axis=1)
    xh = np.stack([x[t + 1 - u - j] for j in range(l)], axis=1)
    return yh, xh, y[t + 1]


def windows_from_blocks(blocks, k, l, u=1):
    """Windows touching a gap (NaN) are dropped, so a missing bar never shifts the
    others; blocks too short for one window contribute nothing."""
    parts = []
    for b in blocks:
        if len(b) < window_span(k, l, u):
            parts.append((np.empty((0, k)), np.empty((0, l)), np.empty(0)))
            continue
        yh, xh, tgt = make_windows(b, k, l, u)
        ok = np.isfinite(yh).all(axis=1) & np.isfinite(xh).all(axis=1) & np.isfinite(tgt)
        parts.append((yh[ok], xh[ok], tgt[ok]))
    if not sum(len(p[2]) for p in parts):
        raise ValueError("no complete windows: every window touches a gap or a block edge")
    return Windows(
        yh=np.concatenate([p[0] for p in parts]),
        xh=np.concatenate([p[1] for p in parts]),
        target=np.concatenate([p[2] for p in parts]),
        block_id=np.concatenate([np.full(len(p[2]), i) for i, p in enumerate(parts)]),
    )
