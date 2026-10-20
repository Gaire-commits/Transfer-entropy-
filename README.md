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
peer history from the same signal bar.

## Running it

Open the notebook with the badge above (GPU runtime). It runs the controls, then a **check run** that
trains to 2019, validates on 2020 and scores 2021, inside the training years, then the registered run,
one architecture per cell. Each architecture is saved to Drive as soon as it's scored and skipped if
run again, so a dropped session resumes, and the test days are not scored twice by accident (`--rerun`
does that on purpose, to be disclosed). The verdict appears once all three architectures are in.

```bash
python scripts/run.py --out-dir <outputs>/heat_v1 --check
```

```bash
python scripts/run.py --out-dir <outputs>/heat_v1 --archs bilstm
```

```bash
python -m pytest -q tests
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
| [`heat/synthetic.py`](heat/synthetic.py) | Synthetic panels with planted leaders, for the controls |

**Tests** (`tests/`, about 2 minutes): no look-ahead in peers, inputs and labels; the grid rules; the
book's accounting; the verdict logic; the controls (a planted leader is an edge at 30 minutes and leads
the heat map; with nothing planted, no edge); resuming; and the frozen config.

A positive result is predictive information, not causation. Like the reversal, it would also need to
hold in a forward test before it is called tradeable.
