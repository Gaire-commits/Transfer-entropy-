# learned_te

[![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/Gaire-commits/Transfer-entropy-/blob/main/notebooks/learned_te_colab.ipynb)

Leakage-safe learned transfer entropy estimator:
T̂_{X→Y} = ½ ln(MSE of f_Y / MSE of f_XY), evaluated once on a held-out test block.

The transfer-entropy stage of the thesis that follows
[entropy-gate-pipeline](https://github.com/Gaire-commits/entropy-gate-pipeline): does one stock's
recent 5-minute move predict another's next move, beyond the target's own past and the market?
The [Colab notebook](notebooks/learned_te_colab.ipynb) runs the synthetic controls, then the
proposal's pairs (NVDA–AMD, V–MA, MSFT–GOOGL) on the thesis's cached SIP bars.

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

## Market data: the thesis's 5-minute SIP bars

[`market.py`](learned_te/market.py) reads the bars entropy-gate-pipeline already cached on Google
Drive (`My Drive/entropy-gate-pipeline-data/data/bars_sip/5Min/<SYM>.parquet`, 503 S&P 500 stocks,
2016-01-04 to 2026-10-12) and applies the thesis data rules:

- **Fixed session grid.** Regular hours on a 5-minute grid per trading day. Missing bars stay as
  gaps (NaN) and every window touching one is dropped, so a gap never shifts later bars. Early
  closes end at 13:00, and a final day still in progress is dropped.
- **Within-day returns.** No return spans the overnight gap.
- **Market-neutral.** The market is the equal-weighted mean return across the universe at each bar,
  as in the thesis (`leadlag.py`, `peers.py`). It is computed once, saved, and regressed out as a
  reference column. The SIP cache holds stocks only, so SPY is not used.
- **Inference by day.** With sessions, whole days go to train, validation or test in time order
  (`day_split`, one embargo day between sets; the last 20% of days is the hold-out), and
  confidence intervals resample whole test days.
- **Feeds never mix.** Loading refuses a cache whose metadata says it is not SIP.
- **Many pairs.** [`multiple.py`](learned_te/multiple.py) applies Benjamini-Hochberg
  false-discovery control across directions.

```python
from learned_te import Config, report
from learned_te.market import THESIS_SIP_DIR, day_sessions, load_cached_closes, market_returns

market = market_returns(THESIS_SIP_DIR, cache_file=".../outputs/transfer_entropy/market_5min.parquet")
closes = load_cached_closes(THESIS_SIP_DIR, ["NVDA", "AMD"])
sessions = day_sessions(closes, x="NVDA", y="AMD", market=market)  # columns [NVDA, AMD, market]
rep = report(sessions, Config(k=6, l=6, ref_cols=(2,), n_surrogates=200))
print(rep["verdict"])
```

`k = 6` bars is 30 minutes of the target's own history, so its own short-term reversal (the
thesis's E5 result) is already in the baseline and cannot pass for transfer.

For symbols outside the cache, `fetch_bars` downloads the same 5-minute SIP bars from Alpaca
(`pip install alpaca-py`; keys from env vars or Colab secrets, never from code).

Keep in mind: effects in returns are tiny, markets are non-stationary, and a positive result is
predictive information, not causation or a trading signal. `report()` checks scale with the
estimate's own confidence interval, so they stay meaningful at these sizes.

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
