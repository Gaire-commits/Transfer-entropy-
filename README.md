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

## Market data (Alpaca)

[`market.py`](learned_te/market.py) fetches bars with `alpaca-py` and turns them into log
returns in the `[X, Y, refs...]` column order the pipeline expects.

```bash
pip install alpaca-py
```

Stock data needs Alpaca API keys (free paper-trading keys work with the default `feed="iex"`).
They are read from the env vars `APCA_API_KEY_ID` / `APCA_API_SECRET_KEY`, or from Colab
secrets with the same names. Never paste keys into a notebook cell. Crypto bars need no keys.

```python
from learned_te import Config, report
from learned_te.market import fetch_bars, log_returns, to_array, intraday_sessions, last_sessions

# Daily: one continuous series. SPY is a reference column, so the market factor
# (shared noise for any two stocks) is regressed out.
prices = fetch_bars(["AAPL", "MSFT", "SPY"], "2016-01-01", "2025-12-31", "1Day")
data = to_array(log_returns(prices), x="AAPL", y="MSFT", refs=["SPY"])
rep = report(data, Config(ref_cols=(2,)))

# Intraday: one session per trading day, so no window spans the overnight gap,
# and the most recent days are held out whole as test.
prices = fetch_bars(["AAPL", "MSFT", "SPY"], "2025-06-01", "2025-09-30", "1Min")
sessions = intraday_sessions(prices, x="AAPL", y="MSFT", refs=["SPY"])
rep = report(sessions, Config(ref_cols=(2,)), test_ids=last_sessions(sessions, 0.2))

# Crypto, no keys needed
prices = fetch_bars(["BTC/USD", "ETH/USD"], "2025-01-01", "2025-10-01", "1Hour", asset_class="crypto")
rep = report(to_array(log_returns(prices), "BTC/USD", "ETH/USD"))
print(rep["verdict"])
```

Things to keep in mind with returns:
- Effects are small (often around 0.001 nats), so you need long histories. A few hundred
  daily bars is too few; use years of daily data or intraday bars.
- Markets are non-stationary. A result that holds on one period may not hold on another, so
  re-run on separate periods.
- `report()` checks are scaled to the estimate's own confidence interval, so they stay
  meaningful at these small effect sizes.
- Like any TE result, a positive finding is predictive information, not causation, and not a
  trading signal on its own.

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
