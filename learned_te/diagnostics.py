"""Step 10: leak diagnostics. These are run and reported, never used for tuning.

`data` is either one (n, c) array or a dict of sessions {id: (n, c) array}; with
sessions, `test_ids` names the sessions held out for test.
"""
from dataclasses import replace

import numpy as np

from .pipeline import Config, run, swap_xy


def _map(data, f):
    return {k: f(v) for k, v in data.items()} if isinstance(data, dict) else f(data)


def _run(data, cfg, test_ids):
    if isinstance(data, dict):
        return run(cfg=cfg, sessions=data, test_ids=test_ids)
    return run(data, cfg)


def both_directions(data, cfg, test_ids=()):
    fwd = _run(data, cfg, test_ids)
    rev = _run(_map(data, swap_xy), cfg, test_ids)
    return fwd, rev, fwd.te_eff - rev.te_eff


def time_reversal(data, cfg, test_ids=()):
    """Real directed coupling should flip the sign of Delta T under time reversal."""
    _, _, d_orig = both_directions(data, cfg, test_ids)
    _, _, d_rev = both_directions(_map(data, lambda s: s[::-1].copy()), cfg, test_ids)
    return d_orig, d_rev


def gap_sweep(data, cfg, test_ids=(), base_gap=None, multiples=(1, 2, 4, 8)):
    """T_hat should be stable as the gap grows; a downward drift means leakage.
    Gaps too large for the data are skipped."""
    base_gap = base_gap or _run(data, cfg, test_ids).gap
    out = {}
    for m in multiples:
        try:
            out[m * base_gap] = _run(data, replace(cfg, gap=m * base_gap), test_ids).te
        except ValueError:  # a block became too short
            break
    return out


def noise_sweep(data, cfg, test_ids=(), sigmas=(0.0, 0.5, 1.0), channel=1, seed=123):
    """Add white noise to the target channel. Real transfer should not *grow* with it;
    leakage through a noisy conditioning history does."""
    rng = np.random.default_rng(seed)

    def add(s, sigma):
        noisy = s.copy()
        noisy[:, channel] += rng.normal(0, sigma * np.nanstd(s[:, channel]), len(s))
        return noisy

    return {sig: _run(_map(data, lambda s: add(s, sig)), cfg, test_ids).te for sig in sigmas}


def report(data, cfg=Config(), test_ids=()):
    fwd, rev, delta = both_directions(data, cfg, test_ids)
    d_orig, d_reversed = time_reversal(data, cfg, test_ids)
    noise = noise_sweep(data, cfg, test_ids)
    gaps = gap_sweep(data, cfg, test_ids, base_gap=fwd.gap)

    # Tolerances scale with the estimate's own uncertainty, so the checks mean the
    # same thing for large simulated effects and for tiny effects in returns.
    tol = (fwd.ci[1] - fwd.ci[0]) / 2
    noise_vals = list(noise.values())
    checks = {
        "significant": fwd.significant,
        # With shuffled targets f_XY's extra variance makes the gain <= 0; any
        # *positive* gain here means the targets leak into the inputs.
        "shuffled_target_no_gain": fwd.shuffled_target_te < tol,
        "net_direction_positive": delta > 0 and fwd.ci[0] > rev.te_eff,
        "flips_under_time_reversal": bool(np.sign(d_orig) != np.sign(d_reversed)),
        "does_not_grow_with_target_noise": noise_vals[-1] <= noise_vals[0] + tol,
        "stable_across_gaps": len(gaps) > 1 and (max(gaps.values()) - min(gaps.values())) < 2 * tol,
    }

    if not checks["significant"]:
        verdict = "no evidence of X->Y transfer"
    elif all(checks.values()):
        verdict = "directed X->Y transfer supported"
    else:
        failed = [k for k, v in checks.items() if not v]
        verdict = "dependence present, direction unresolved (failed: " + ", ".join(failed) + ")"

    return {
        "forward": fwd,
        "reverse": rev,
        "delta_te": delta,
        "time_reversal": (d_orig, d_reversed),
        "noise_sweep": noise,
        "gap_sweep": gaps,
        "tolerance": tol,
        "checks": checks,
        "verdict": verdict,
    }
