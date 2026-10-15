"""Effect of the noise-invariance penalty on each scenario.

lambda = 0 is the identical torch model without the penalty (the control).

    python3 compare_invariance.py
"""
from dataclasses import replace

from learned_te import SCENARIOS, Config, run, simulate

base = Config(model="invariant", n_surrogates=30, n_bootstrap=200)
variants = {
    "no penalty (lambda=0)": replace(base, inv_lambda=0.0),
    "monotone, lambda=10": replace(base, inv_penalty="monotone", inv_lambda=10.0),
    "monotone, lambda=100": replace(base, inv_penalty="monotone", inv_lambda=100.0),
    "vrex,     lambda=100": replace(base, inv_penalty="vrex", inv_lambda=100.0),
}

seeds = (1, 2, 3)
print(f"{'scenario':<14}{'variant':<24}{'T_eff (mean over seeds)':>24}{'significant':>13}")
for name, (truth, _) in SCENARIOS.items():
    for label, cfg in variants.items():
        rs = [run(simulate(name, seed=s), cfg) for s in seeds]
        mean_te = sum(r.te_eff for r in rs) / len(rs)
        sig = sum(r.significant for r in rs)
        print(f"{name:<14}{label:<24}{mean_te:>+24.4f}{f'{sig}/{len(rs)}':>13}")
    print(f"{'':<14}truth: {'X->Y coupling' if truth else 'no coupling'}\n")
