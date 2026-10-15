# learned_te

Leakage-safe learned transfer entropy estimator:
T̂_{X→Y} = ½ ln(MSE of f_Y / MSE of f_XY), evaluated once on a held-out test block.

| Step | What | Where |
|---|---|---|
| 0 | Freeze design (k, l, u, grids, model, seed) | `pipeline.Config` |
| 1–2 | Group split, then time split with gap G ≥ max(window span, ACF time) | `split.py` |
| 3–4 | Causal filter, common-mode regression, scaling — fit on train only | `preprocess.py` |
| 5 | Windows built inside each block, never across | `windows.py` |
| 6 | f_Y and f_XY, same family and budget, early stopping on val | `models.py` |
| 7 | k, l tuned on val | `pipeline._select_history` |
| 8 | Estimator null: refit on surrogate train with frozen hyperparameters, score on surrogate val | `surrogates.py` |
| 9 | One-shot test: T̂, T_eff, block-bootstrap CI, p-value | `pipeline.run` |
| 10 | Diagnostics: reverse direction, time reversal, noise sweep, gap sweep, shuffled targets | `diagnostics.py` |

```bash
python3 demo.py
```

```bash
python3 -m pytest -q tests
```

Input: an `(n, c)` array with columns `[X, Y, ref...]`, or `run(sessions={...}, test_ids=[...])`
for multi-session data. Use `Config(ref_cols=(2,))` to regress out a reference channel and
`Config(model="mlp")` for the nonlinear estimator.

## Noise-invariance training (`model="invariant"`)

[`invariance.py`](learned_te/invariance.py) trains f_Y and f_XY jointly (torch) with

L = MSE_Y + MSE_XY + λ · penalty(g_1 … g_E),  g_e = ½ ln(MSE_Y^e / MSE_XY^e)

where environment e adds white noise of std σ_e to the Y channel. Leakage makes g_e grow
with σ_e, while real transfer does not. `inv_penalty="monotone"` (default) penalizes only that
growth. `"vrex"` penalizes any variation, which also suppresses real transfer. MSE_Y is
detached in the penalty, so the penalty cannot be met by making f_Y worse. The estimator null
refits with the penalty and frozen epoch count.

```bash
python3 compare_invariance.py
```

Results (T_eff in nats, mean over 3 seeds; significant runs out of 3):

| Scenario (truth) | λ = 0 | monotone λ = 10 | monotone λ = 100 | V-REx λ = 100 |
|---|---|---|---|---|
| coupled (X→Y) | +0.186 (3/3) | +0.187 (3/3) | +0.192 (3/3) | +0.182 (3/3) |
| independent (none) | −0.001 (0/3) | −0.001 (0/3) | −0.001 (0/3) | −0.001 (0/3) |
| proxy_leak (none) | +0.024 (3/3) | +0.017 (3/3) | +0.010 (3/3) | +0.022 (3/3) |
| shared_noise (none) | +0.005 (3/3) | +0.006 (3/3) | +0.006 (3/3) | +0.006 (3/3) |

The monotone penalty cuts the proxy leak by more than half without affecting real transfer,
but it does not remove it, and it does nothing for shared noise (whose spurious gain doesn't
grow with added Y noise). The leak cases are still significant, so use the diagnostics in
`report()`, not significance alone.

Note: `surrogate="phase"` preserves all linear cross-structure, so with `model="ridge"` it
only tests for nonlinear transfer.
