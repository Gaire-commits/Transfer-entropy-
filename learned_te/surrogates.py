"""Step 8 null models. Both are generated inside a single split (val) only."""
import numpy as np
from sklearn.neighbors import NearestNeighbors

from .windows import Windows, windows_from_blocks


def neighbor_index(w, n_neighbors=10):
    """Each window's nearest neighbors in Y-history space (excluding itself). Depends only
    on Y history, which surrogates never change, so it is computed once per null."""
    nn = NearestNeighbors(n_neighbors=n_neighbors + 1).fit(w.yh)
    return nn.kneighbors(w.yh, return_distance=False)[:, 1:]


def local_permutation(w, rng, n_neighbors=10, neighbors=None):
    """Swap X history only between samples with similar Y history.

    A plain shuffle also breaks the X <-> observed-Y-history dependence and makes
    the null too easy; this keeps it. It tests X _||_ Y_{t+1} | observed Y history,
    so a leak through the *unobserved* true Y_t is still (correctly, for that
    hypothesis) significant -- the diagnostics, not the null, have to catch it.
    """
    idx = neighbors if neighbors is not None else neighbor_index(w, n_neighbors)
    pick = idx[np.arange(len(w)), rng.integers(0, idx.shape[1], len(w))]
    return Windows(w.yh, w.xh[pick], w.target, w.block_id)


def phase_randomize(block, rng):
    """Multivariate FT surrogate: same random phases on every channel.

    Preserves each spectrum and the full linear cross-spectrum (incl. lag-0
    shared noise), destroys everything else.
    """
    n = len(block)
    F = np.fft.rfft(block, axis=0)
    phases = rng.uniform(0, 2 * np.pi, F.shape[0])
    phases[0] = 0.0
    if n % 2 == 0:
        phases[-1] = 0.0
    return np.fft.irfft(F * np.exp(1j * phases)[:, None], n=n, axis=0)


def null_distribution(models, train_blocks, val_blocks, train_w, val_w, k, l, u, kind,
                      n_surrogates, rng):
    """Null of the *whole estimator*: refit on a surrogate train set with the real run's
    frozen hyperparameters, score on a surrogate val set. Test data is never used.

    (Scoring a fixed model on permuted X is wrong: if the model learned to use X,
    the null comes out negative and T_eff = T_hat - mean(null) is inflated.)
    """
    if kind not in ("local", "phase"):
        raise ValueError("surrogate must be 'local' or 'phase'")
    out = np.empty(n_surrogates)
    if kind == "local":
        nb_tr, nb_va = neighbor_index(train_w), neighbor_index(val_w)
    for b in range(n_surrogates):
        if kind == "local":
            s_tr = local_permutation(train_w, rng, neighbors=nb_tr)
            s_va = local_permutation(val_w, rng, neighbors=nb_va)
            m = models.refit_frozen(s_tr, refit_y=False)  # Y-only inputs are unchanged
        else:
            s_tr = windows_from_blocks([phase_randomize(bl, rng) for bl in train_blocks], k, l, u)
            s_va = windows_from_blocks([phase_randomize(bl, rng) for bl in val_blocks], k, l, u)
            m = models.refit_frozen(s_tr)
        out[b] = m.te(s_va)
    return out
