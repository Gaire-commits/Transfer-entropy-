# Heat transfer between correlated stocks

[![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/Gaire-commits/Transfer-entropy-/blob/main/notebooks/heat_colab.ipynb)

The transfer-entropy stage of Kshitiz Gaire's thesis, following
[entropy-gate-pipeline](https://github.com/Gaire-commits/entropy-gate-pipeline). That project found a real
short-term reversal (a stock that lagged the market over 30 minutes recovers a little, building over the
first hour and lasting to 4 hours: E5, E9, E10), and found that a stock's 5 most correlated peers, as a
linear rank signal, predict no better than random peers (E12). This asks the question again with deep
models and transfer entropy: does a stock's next 30 minutes to 4 hours follow the recent moves of the
stocks it moves with, beyond its own reversal, and is that an edge after costs?

## The experiment (HEAT1, registered in [`configs/heat_v1.yaml`](configs/heat_v1.yaml), 2026-10-20)

| | |
|---|---|
| Data | Thesis 5-minute SIP bars, S&P 500 (503 stocks), 2016–2026, returns relative to the equal-weighted market |
| Moments | A stock at a signal bar, every 30 minutes (E5/E12 geometry): about 12 million |
| Input | The last hour (12 bars) of the stock and its 5 most correlated peers (previous 60 days, chosen monthly, E12's rule) |
| Holds | Enter one bar after the signal; hold 30 min, 1 h, 2 h, 4 h, all within the day |
| Models | BiLSTM, CNN-ResNet (1-D), Transformer; one output per hold |
| Controls | The same model, initial weights, batches and budget with **no peers** and with **5 random peers** |
| Transfer | G = ½ ln(MSE no peers / MSE with peers), nats, on test moments |
| Split | Whole days: train 2016–2021, validation 2022–2023, **test 2024 onward, scored once** |
| Trading | Quintile long-short on predictions at each moment, 2 bps per position |
| Intervals | 99% day-block bootstrap (2,000 draws) |

**Edge** in a cell (architecture × hold, 12 cells) requires all five:
1. net of 2 bps above zero;
2. above the no-peer model's net (the edge is the peers, not the reversal);
3. G with correlated peers above G with random peers (E12's control);
4. positive in both halves of the test days;
5. Benjamini-Hochberg at q = 0.05 over the 12 cells.

**Heat map** ([`evaluate.heat_flows`](heat/evaluate.py)): for each source → target pair, the share of the
target's error that the source's last hour removes. The source's history is swapped with another stock's
peer history from the same signal bar. Each pair also gets a standard error, and a pair is shown only if its share is
3 standard errors above zero (`evaluate.screen_pairs`). Thousands of pairs are looked at, so without this
the top of the list is pairs with few moments and a lucky share. This is a screening bar for the heat
map, not a test; the registered verdict doesn't use it.

## Running it

Open the notebook with the badge above (GPU runtime). In order:

1. **Controls**: the tests, including a planted leader that must be found and a null that must not.
2. **Check run**: trains to 2019, validates on 2020, scores 2021, inside the training years. Run it as
   often as you like; it never touches the test days. It also draws the visual report.
3. **Tuning** (optional, validation days only): see below.
4. **Registered run**: one architecture per cell. Each is saved to Drive as soon as it's scored and
   skipped if run again, so a dropped session resumes and the test days are not scored twice by accident
   (`--rerun` does that on purpose, to be disclosed). The verdict appears once all three are in.
5. **Result and visual report.**

```bash
python scripts/run.py --out-dir <outputs>/heat_v1 --check
```

```bash
python scripts/tune.py --out-dir <outputs>/heat_v1 --archs bilstm
```

```bash
python scripts/run.py --config configs/heat_v2.yaml --out-dir <outputs>/heat_v2 --panel <outputs>/heat_v1/panel.npz --archs bilstm
```

```bash
python -m pytest -q tests
```

**Drive and the panel.** The panel (`panel.npz`, ~420 MB) is rebuilt from the bars in a few minutes, so the
notebook keeps it on the machine's own disk (`--panel /content/heat_panel.npz`) and sends only the small
result files to Drive. One huge upload can hold up everything queued behind it. After each script the
notebook flushes Drive with a time limit, so a slow upload can't freeze it.

## Tuning the deep models

[`heat/tune.py`](heat/tune.py), run by [`scripts/tune.py`](scripts/tune.py): random search with successive
halving over learning rate, weight decay, width, depth, dropout and each architecture's own knobs (kernel
size for the ResNet, attention heads for the Transformer).

- **Validation days only.** The tuner is given the training and validation moments and nothing else; the
  test days are never loaded. That is what lets tuned settings be registered before the test is scored.
- **The default is always a candidate**, so tuning cannot do worse than the registered default on validation.
- **No tuning advantage for the peers.** A setting is scored by its mean validation loss over the no-peer,
  correlated-peer and random-peer models, and all three share it.
- **Adopted only if it beats noise.** The winner and the default are each retrained with two seeds; the
  tuned setting is used only if its mean validation loss beats the default's by more than the larger
  seed spread. Otherwise the default is kept.
- **Register before scoring.** The tuner writes `tuning/heat_v2.yaml`, a copy of `heat_v1` with only the
  experiment name and `train.per_arch` changed (a test checks that nothing else moved). It is committed
  and hash-frozen before the registered run, which then uses it. `heat_v1` was never scored on the test
  days, so `heat_v2` supersedes it with the test still unseen.

## The visual report

[`heat/viz.py`](heat/viz.py) writes `viz.html` for any run: one self-contained file, no internet needed,
opens in a browser and in the notebook. Filters at the top: architecture (or all three averaged), hold,
and how many links to show. Hover anything for the numbers; every chart has a table view; light and dark
themes.

| Chart | Shows |
|---|---|
| Transfer by hold | Gain from correlated peers vs random peers per architecture, with day-block intervals |
| Does it pay after costs? | Net bps per position with intervals, and the verdict |
| Who leads whom | Leader × follower matrix of how much of the follower's error the leader's last hour removes, grouped by sector |
| The flow network | Arrows from leader to follower, stocks clustered by sector, darker = sends more heat |
| Industries leading industries | The same heat averaged between sectors |
| Strongest pairs | The top pairs, across all four holds side by side |
| Who moves with whom | The correlated-peer links themselves: how many stay in a sector and how many cross it |

Position carries the sector and one blue ramp carries magnitude, so nothing needs more colours than the
chart palette can keep apart.

```bash
python scripts/viz.py --out-dir <outputs>/heat_v1/check
```

## Code

| File | What |
|---|---|
| [`heat/data.py`](heat/data.py) | Thesis cache → one panel of market-relative 5-minute returns on the fixed session grid (gaps kept, no overnight returns, early closes, feeds never mix) |
| [`heat/peers.py`](heat/peers.py) | Point-in-time correlated peers and random peers, monthly (E12's rule) |
| [`heat/dataset.py`](heat/dataset.py) | Moments, hold labels inside the day, whole-day splits, input batches built on the GPU |
| [`heat/models.py`](heat/models.py) | BiLSTM, ResNet1D, Transformer |
| [`heat/train.py`](heat/train.py) | Training with identical budgets per variant, early stopping on validation days |
| [`heat/evaluate.py`](heat/evaluate.py) | Transfer G, quintile book, rank IC, day-block intervals, the verdict, the heat map |
| [`heat/run.py`](heat/run.py) | The experiment end to end, saved per architecture |
| [`heat/tune.py`](heat/tune.py) | Hyperparameter tuning on validation days, and the candidate `heat_v2` config |
| [`heat/viz.py`](heat/viz.py), [`viz_template.html`](heat/viz_template.html) | The visual report |
| [`heat/synthetic.py`](heat/synthetic.py) | Synthetic panels with planted leaders, for the controls |

**Tests** (`tests/`, about 5 minutes): no look-ahead in peers, inputs and labels; the grid rules; the
book's accounting; the verdict logic; the controls (a planted leader is an edge at 30 minutes and leads
the heat map; with nothing planted, no edge); resuming; tuning (validation only, default always a
candidate, adoption needs a gain beyond seed noise, the candidate config differs only where it says);
the visual report (self-contained, matches the run's files); and the frozen config.

A positive result is predictive information, not causation. Like the reversal, it would also need to
hold in a forward test before it is called tradeable.
