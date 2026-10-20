"""Hyperparameter tuning for the deep models, on training and validation days only.

The test days are never an input here: every function takes train and val moments, and nothing else.
That is what lets a tuned config be registered before the test days are scored.

Method (random search with successive halving):
- Trial 0 is the registered default, so tuning cannot do worse than the default on validation.
- The rest are drawn at random from the search space. All start on a small budget; the best third
  survive to a bigger one, and so on, until the last rung trains for the full epoch budget.
- A setting is scored by its mean validation loss over the three variants (no peers, correlated
  peers, random peers). One setting is shared by all three, so "same model, same budget" holds and
  the peer models get no tuning advantage over their controls.
- Confirmation: the winner and the default are each retrained with two seeds. The tuned setting is
  adopted only if its mean validation loss beats the default's by more than the larger of the two
  seed spreads; otherwise the default is kept. A gain inside seed noise is not a gain.
"""
from __future__ import annotations

import math

import numpy as np
import pandas as pd

from .dataset import VARIANTS, Gather, Moments
from .models import DEFAULTS
from .train import train_variant

SIZES = (32, 64, 128)  # all divisible by every head count below
SPACE = {
    "bilstm": {"depth": (1, 2, 3)},
    "resnet1d": {"depth": (2, 3, 4, 5), "kernel": (3, 5)},
    "transformer": {"depth": (1, 2, 3, 4), "heads": (2, 4, 8)},
}


def default_setting(arch: str, base: dict) -> dict:
    """The registered heat_v1 settings for this architecture."""
    return {"size": base["size"], "lr": base["lr"], "weight_decay": base["weight_decay"], **DEFAULTS[arch]}


def _log_uniform(rng, lo, hi):
    return float(f"{math.exp(rng.uniform(math.log(lo), math.log(hi))):.3g}")


def sample_setting(arch: str, rng: np.random.Generator) -> dict:
    s = {"size": int(rng.choice(SIZES)), "lr": _log_uniform(rng, 3e-4, 3e-3),
         "weight_decay": _log_uniform(rng, 1e-5, 1e-2), "dropout": round(float(rng.uniform(0.0, 0.3)), 2)}
    for key, choices in SPACE[arch].items():
        s[key] = int(rng.choice(choices))
    return s


def _score(arch, setting, epochs, gather, train, val, holds, lookback, base, seed, device, log):
    """Mean over the three variants of the best validation loss, with `epochs` epochs and no early stop."""
    t = {**base, **setting, "max_epochs": epochs, "patience": epochs, "seed": seed}
    losses = {}
    for variant in VARIANTS:
        _, info = train_variant(arch, variant, gather, train, val, holds, lookback, t, device, lambda m: None)
        losses[variant] = info["val_loss"]
    return float(np.mean(list(losses.values()))), losses


def tune_arch(arch, gather: Gather, train: Moments, val: Moments, holds, lookback, base, n_trials=12,
              eta=3, budgets=(1, 3, 9), seed=0, device="cpu", log=print):
    """Returns (result, trials). `result` has the chosen setting, whether it beat the default beyond
    seed noise, and the confirmation numbers; `trials` is every (trial, rung) score."""
    rng = np.random.default_rng([seed, 17])
    settings = [default_setting(arch, base)] + [sample_setting(arch, rng) for _ in range(n_trials - 1)]
    alive = list(range(len(settings)))
    rows = []
    for rung, epochs in enumerate(budgets):
        scored = []
        for i in alive:
            score, losses = _score(arch, settings[i], epochs, gather, train, val, holds, lookback, base,
                                   base["seed"], device, log)
            scored.append((score, i))
            rows.append({"arch": arch, "trial": i, "rung": rung, "epochs": epochs, "val_loss": score,
                         **{f"val_{v}": losses[v] for v in VARIANTS}, **settings[i]})
            log(f"  {arch} trial {i:2d} rung {rung} ({epochs} ep): val {score:.5f}")
        scored.sort()
        if rung < len(budgets) - 1:
            alive = [i for _, i in scored[: max(1, math.ceil(len(scored) / eta))]]
    best_i = scored[0][1]

    # Confirmation: same budget as the last rung, two seeds each, default vs the winner.
    last = budgets[-1]
    seeds = (base["seed"], base["seed"] + 1)
    conf = {}
    for name, i in (("default", 0), ("tuned", best_i)):
        conf[name] = [_score(arch, settings[i], last, gather, train, val, holds, lookback, base, s, device, log)[0]
                      for s in seeds]
    d_mean, t_mean = float(np.mean(conf["default"])), float(np.mean(conf["tuned"]))
    spread = max(abs(conf["default"][0] - conf["default"][1]), abs(conf["tuned"][0] - conf["tuned"][1]))
    adopt = best_i != 0 and (d_mean - t_mean) > spread
    chosen = settings[best_i] if adopt else settings[0]
    result = {
        "arch": arch, "adopt_tuned": bool(adopt), "setting": chosen, "winner_trial": int(best_i),
        "winner_setting": settings[best_i], "default_val_loss": d_mean, "tuned_val_loss": t_mean,
        "seed_spread": spread, "improvement": d_mean - t_mean, "trials": n_trials, "budgets": list(budgets),
        "confirm": conf,
    }
    log(f"{arch}: default {d_mean:.5f}, tuned {t_mean:.5f}, spread {spread:.5f} -> "
        + ("TUNED adopted" if adopt else "default kept"))
    return result, pd.DataFrame(rows)


# ---- the candidate registration -------------------------------------------------------------------
def candidate_config(base_text: str, results: dict, date: str, base_name="heat_v1", new_name="heat_v2") -> str:
    """The registered config with the tuned per-architecture settings added, as text. The rule is
    unchanged: only the experiment name, a header, and `train.per_arch` differ from the base."""
    block = ["  per_arch:  # tuned on validation days only (see tuning/report.md); the test days were not used"]
    for arch, r in results.items():
        s = r["setting"]
        inner = ", ".join(f"{k}: {s[k]}" for k in sorted(s))
        note = "tuned" if r["adopt_tuned"] else "default kept: no gain beyond seed noise"
        block.append(f"    {arch}: {{{inner}}}  # {note}")
    marker = "\nevaluation:"
    assert base_text.count(marker) == 1 and base_text.count(f"experiment: {base_name}") == 1
    text = base_text.replace(f"experiment: {base_name}", f"experiment: {new_name}")
    text = text.replace(marker, "\n".join(block) + "\n" + marker)
    header = (f"# {new_name.upper()} = {base_name.upper()} with hyperparameters tuned on the validation days ({date}).\n"
              f"# The protocol, data, split, peers, controls and cost are unchanged. {base_name} was never scored on the\n"
              f"# test days; this supersedes it before any test scoring. Tuning used train and validation days only.\n#\n")
    return header + text
