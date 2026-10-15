"""Step 10: leak diagnostics. These are run and reported, never used for tuning."""
from dataclasses import replace

import numpy as np

from .pipeline import Config, run, swap_xy


def both_directions(series, cfg):
    fwd = run(series, cfg)
    rev = run(swap_xy(series), cfg)
    return fwd, rev, fwd.te_eff - rev.te_eff


def time_reversal(series, cfg):
    """Real directed coupling should flip the sign of Delta T under time reversal."""
    _, _, d_orig = both_directions(series, cfg)
    _, _, d_rev = both_directions(series[::-1].copy(), cfg)
    return d_orig, d_rev


def gap_sweep(series, cfg, gaps=(None, 50, 100, 200)):
    """T_hat should be stable as the gap grows; a downward drift means leakage."""
    return {g: run(series, replace(cfg, gap=g)).te for g in gaps}


def noise_sweep(series, cfg, sigmas=(0.0, 0.5, 1.0), channel=1, seed=123):
    """Add white noise to the target channel. Real transfer should not *grow* with it;
    leakage through a noisy conditioning history does."""
    rng = np.random.default_rng(seed)
    scale = series[:, channel].std()
    out = {}
    for s in sigmas:
        noisy = series.copy()
        noisy[:, channel] += rng.normal(0, s * scale, len(series))
        out[s] = run(noisy, cfg).te
    return out


def report(series, cfg=Config()):
    fwd, rev, delta = both_directions(series, cfg)
    d_orig, d_reversed = time_reversal(series, cfg)
    noise = noise_sweep(series, cfg)
    gaps = gap_sweep(series, cfg)

    noise_vals = list(noise.values())
    checks = {
        "significant": fwd.significant,
        # With shuffled targets f_XY's extra variance makes the gain <= 0; any
        # *positive* gain here means the targets leak into the inputs.
        "shuffled_target_no_gain": fwd.shuffled_target_te < 0.01,
        "net_direction_positive": delta > 0 and fwd.ci[0] > rev.te_eff,
        "flips_under_time_reversal": np.sign(d_orig) != np.sign(d_reversed),
        "does_not_grow_with_target_noise": noise_vals[-1] <= noise_vals[0] + 0.01,
        "stable_across_gaps": (max(gaps.values()) - min(gaps.values())) < 0.02,
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
        "checks": checks,
        "verdict": verdict,
    }
