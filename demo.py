"""Run the full pipeline + diagnostics on every simulated scenario.

    python3 demo.py            # ridge (fast)
    python3 demo.py mlp        # nonlinear learned estimator
"""
import sys

from learned_te import SCENARIOS, Config, report, simulate

model = sys.argv[1] if len(sys.argv) > 1 else "ridge"
cfg = Config(model=model, n_surrogates=100, n_bootstrap=300)

for name, (truth, desc) in SCENARIOS.items():
    r = report(simulate(name, seed=1), cfg)
    f, rv = r["forward"], r["reverse"]
    print(f"\n=== {name}: {desc}")
    print(f"truth: {'X->Y coupling' if truth else 'no X->Y coupling'}")
    print(f"T_eff X->Y = {f.te_eff:+.4f}  CI [{f.ci[0]:+.4f}, {f.ci[1]:+.4f}]  p = {f.p_value:.3f}"
          f"   (k={f.k}, l={f.l}, gap={f.gap})")
    print(f"T_eff Y->X = {rv.te_eff:+.4f}   delta = {r['delta_te']:+.4f}")
    d0, d1 = r["time_reversal"]
    print(f"delta original / time-reversed = {d0:+.4f} / {d1:+.4f}")
    print("noise sweep (sigma: T_hat):", {s: round(v, 4) for s, v in r["noise_sweep"].items()})
    print("gap sweep   (gap:   T_hat):", {g: round(v, 4) for g, v in r["gap_sweep"].items()})
    print("verdict:", r["verdict"])
