"""Multiple-testing control for scans over many stock pairs (proposal step 3)."""
import numpy as np


def benjamini_hochberg(p_values, q=0.05):
    """Boolean mask of discoveries at false-discovery rate q (Benjamini-Hochberg 1995).

    Surrogate p-values cannot go below 1 / (n_surrogates + 1), so with m tests and
    rate q you need n_surrogates + 1 > m / q for even the smallest p to pass.
    """
    p = np.asarray(p_values, float)
    m = len(p)
    order = np.argsort(p)
    passed = p[order] <= q * np.arange(1, m + 1) / m
    keep = np.zeros(m, bool)
    if passed.any():
        keep[order[: np.flatnonzero(passed).max() + 1]] = True
    return keep
