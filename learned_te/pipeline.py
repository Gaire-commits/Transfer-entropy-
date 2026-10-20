"""Steps 0-9 in order. The test block is touched exactly once, in step 9."""
from dataclasses import dataclass, field, replace

import numpy as np

from .models import gain, make_predictors
from .preprocess import CausalPreprocessor
from .split import choose_gap, day_split, group_split, time_split, window_span
from .surrogates import null_distribution
from .windows import windows_from_blocks


@dataclass(frozen=True)
class Config:
    # Step 0: everything here is fixed before the data is looked at.
    k: int = 3
    l: int = 3
    u: int = 1  # source delay: X history ends at t+1-u
    k_grid: tuple = ()  # if set, k and l are tuned on val (step 7)
    l_grid: tuple = ()
    model: str = "ridge"  # "ridge", "mlp", or "invariant" (noise-invariance penalty)
    inv_lambda: float = 10.0
    inv_sigmas: tuple = (0.0, 0.5, 1.0)  # added Y-noise std per environment
    inv_penalty: str = "monotone"  # "monotone" or "vrex"
    inv_hidden: int = 0  # 0 = linear, >0 = two-layer tanh MLP
    gap: int | None = None  # samples (one series) or sessions (day split); None -> automatic
    session_split: str = "days"  # sessions: "days" = whole sessions per set; "within" = split each
    frac: tuple = (0.6, 0.2, 0.2)
    band: tuple | None = None  # causal filter band in Hz, e.g. (1, 40)
    fs: float = 1.0
    ref_cols: tuple = ()  # reference channels for common-mode regression
    surrogate: str = "local"  # "local" (conditional permutation) or "phase"
    n_surrogates: int = 200
    alpha: float = 0.05
    n_bootstrap: int = 500
    seed: int = 0


@dataclass
class Result:
    te: float  # raw test-set T_hat (nats)
    te_eff: float  # T_hat minus mean null T_hat
    ci: tuple  # block-bootstrap CI for te_eff (whole days with the day split)
    threshold: float  # frozen on val
    p_value: float
    significant: bool
    shuffled_target_te: float  # step-10 sanity check, should be ~0
    k: int
    l: int
    gap: int
    n_windows: dict = field(default_factory=dict)


def _block_bootstrap(e_y, e_xy, block_len, n_boot, rng):
    n = len(e_y)
    n_blocks = int(np.ceil(n / block_len))
    starts_max = n - block_len
    out = np.empty(n_boot)
    for b in range(n_boot):
        starts = rng.integers(0, starts_max + 1, n_blocks)
        idx = (starts[:, None] + np.arange(block_len)).ravel()[:n]
        out[b] = gain(e_y[idx].mean(), e_xy[idx].mean())
    return out


def _day_bootstrap(e_y, e_xy, block_id, n_boot, rng):
    """Resample whole test days: windows on the same day share that day's market."""
    rows = [np.flatnonzero(block_id == d) for d in np.unique(block_id)]
    out = np.empty(n_boot)
    for b in range(n_boot):
        idx = np.concatenate([rows[i] for i in rng.integers(0, len(rows), len(rows))])
        out[b] = gain(e_y[idx].mean(), e_xy[idx].mean())
    return out


def _select_history(cfg, train_blocks, val_blocks):
    """Step 7: choose k by f_Y val MSE, then l by f_XY val MSE. Val only."""
    if not cfg.k_grid and not cfg.l_grid:
        return cfg.k, cfg.l
    ks = cfg.k_grid or (cfg.k,)
    ls = cfg.l_grid or (cfg.l,)

    def val_mse(k, l, which):
        tr = windows_from_blocks(train_blocks, k, l, cfg.u)
        va = windows_from_blocks(val_blocks, k, l, cfg.u)
        m = make_predictors(cfg).fit(tr, va)
        e_y, e_xy = m.sq_errors(va)
        return (e_y if which == "y" else e_xy).mean()

    k = min(ks, key=lambda k: val_mse(k, ls[0], "y"))
    l = min(ls, key=lambda l: val_mse(k, l, "xy"))
    return k, l


def run(series=None, cfg=Config(), sessions=None, test_ids=()):
    """Estimate T_{X->Y}. Pass one (n, c) array as `series`, or a dict of sessions + test_ids."""
    rng = np.random.default_rng(cfg.seed)
    max_k = max(cfg.k_grid or (cfg.k,))
    max_l = max(cfg.l_grid or (cfg.l,))
    span = window_span(max_k, max_l, cfg.u)

    # Steps 1-2: split (group first if sessions are given), with gaps.
    by_day = sessions is not None and cfg.session_split == "days"
    if by_day:
        gap = cfg.gap or 1  # embargo, in sessions
        split = day_split(sessions, test_ids, gap, cfg.frac)
    elif sessions is not None:
        any_train = next(s for sid, s in sessions.items() if sid not in test_ids)
        gap = cfg.gap or choose_gap(any_train, span, 0.75)
        split = group_split(sessions, set(test_ids), gap)
    else:
        gap = cfg.gap or choose_gap(series, span, cfg.frac[0])
        split = time_split(series, gap, cfg.frac)

    # Steps 3-4: fit preprocessing on train, apply frozen transform to all.
    prep = CausalPreprocessor(cfg.band, cfg.fs, ref_cols=cfg.ref_cols).fit(split.train)
    tr_b, va_b, te_b = (prep.transform(b) for b in (split.train, split.val, split.test))

    # Step 7 (needs train/val windows, so it sits before the final fit).
    k, l = _select_history(cfg, tr_b, va_b)

    # Step 5: windows inside each block.
    tr, va, te = (windows_from_blocks(b, k, l, cfg.u) for b in (tr_b, va_b, te_b))

    # Step 6: paired models, early stopping / alpha on val.
    models = make_predictors(cfg).fit(tr, va)

    # Step 8: estimator null from train/val surrogates, then freeze threshold and bias.
    null = null_distribution(models, tr_b, va_b, tr, va, k, l, cfg.u, cfg.surrogate,
                             cfg.n_surrogates, rng)
    # Both are clipped at 0: if f_XY ends up worse than f_Y (e.g. an over-strong
    # invariance penalty), a negative null must not turn a negative T_hat into
    # a positive T_eff or a "significant" result.
    threshold = max(float(np.quantile(null, 1 - cfg.alpha)), 0.0)
    bias = max(float(null.mean()), 0.0)

    # Step 9: one-shot test evaluation.
    e_y, e_xy = models.sq_errors(te)
    t_hat = gain(e_y.mean(), e_xy.mean())
    if by_day:
        boot = _day_bootstrap(e_y, e_xy, te.block_id, cfg.n_bootstrap, rng) - bias
    else:
        boot = _block_bootstrap(e_y, e_xy, gap, cfg.n_bootstrap, rng) - bias
    p = (1 + np.sum(null >= t_hat)) / (len(null) + 1)

    # Step 10 sanity check: permuting targets must kill any gain.
    perm = te.target[rng.permutation(len(te))]
    shuffled = replace(te, target=perm)
    shuffled_te = models.te(shuffled)

    return Result(
        te=float(t_hat),
        te_eff=float(t_hat - bias),
        ci=(float(np.quantile(boot, 0.025)), float(np.quantile(boot, 0.975))),
        threshold=threshold,
        p_value=float(p),
        significant=bool(t_hat > threshold),
        shuffled_target_te=float(shuffled_te),
        k=k,
        l=l,
        gap=gap,
        n_windows={"train": len(tr), "val": len(va), "test": len(te)},
    )


def swap_xy(series):
    s = series.copy()
    s[:, [0, 1]] = s[:, [1, 0]]
    return s
