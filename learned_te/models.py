"""Step 6: the paired predictors f_Y and f_XY, trained with identical budgets.

Early stopping and alpha selection use the val block only. sklearn's built-in
`early_stopping=True` is deliberately NOT used: it carves a random (shuffled)
validation subset out of the training windows, which leaks through autocorrelation.
"""
import copy

import numpy as np
from sklearn.linear_model import Ridge
from sklearn.neural_network import MLPRegressor

RIDGE_ALPHAS = (1e-3, 1e-2, 1e-1, 1.0, 10.0, 100.0)


def gain(mse_y, mse_xy):
    """Gaussian-style TE proxy: 0.5 * ln(MSE without X / MSE with X), in nats."""
    return 0.5 * np.log(mse_y / mse_xy)


def _mse(model, X, y):
    return float(np.mean((model.predict(X) - y) ** 2))


def _fit_ridge(Xtr, ytr, Xva, yva):
    """Returns (model, frozen hyperparameters)."""
    best = None
    for alpha in RIDGE_ALPHAS:
        m = Ridge(alpha=alpha).fit(Xtr, ytr)
        score = _mse(m, Xva, yva)
        if best is None or score < best[0]:
            best = (score, m, {"alpha": alpha})
    return best[1], best[2]


def _new_mlp(seed, hidden):
    return MLPRegressor(hidden_layer_sizes=(hidden, hidden), alpha=1e-4,
                        learning_rate_init=1e-3, random_state=seed)


def _fit_mlp(Xtr, ytr, Xva, yva, seed, hidden=32, max_epochs=200, patience=15):
    m = _new_mlp(seed, hidden)
    best_score, best_model, best_epoch, stale = np.inf, None, 0, 0
    for epoch in range(1, max_epochs + 1):
        m.partial_fit(Xtr, ytr)  # sklearn shuffles within the train set only
        score = _mse(m, Xva, yva)
        if score < best_score - 1e-6:
            best_score, best_model, best_epoch, stale = score, copy.deepcopy(m), epoch, 0
        else:
            stale += 1
            if stale >= patience:
                break
    return best_model, {"epochs": best_epoch, "hidden": hidden}


def _refit(kind, hp, Xtr, ytr, seed):
    """Fit with hyperparameters frozen from the real run -- no val data involved."""
    if kind == "ridge":
        return Ridge(alpha=hp["alpha"]).fit(Xtr, ytr)
    m = _new_mlp(seed, hp["hidden"])
    for _ in range(hp["epochs"]):
        m.partial_fit(Xtr, ytr)
    return m


def make_predictors(cfg):
    if cfg.model == "invariant":
        from .invariance import InvariantPairedPredictors

        return InvariantPairedPredictors(seed=cfg.seed, lam=cfg.inv_lambda, sigmas=cfg.inv_sigmas,
                                         penalty=cfg.inv_penalty, hidden=cfg.inv_hidden)
    return PairedPredictors(cfg.model, cfg.seed)


class PairedPredictors:
    """f_Y sees Y history only; f_XY sees Y and X history. Same family, budget and seed."""

    def __init__(self, kind="ridge", seed=0, **kw):
        if kind not in ("ridge", "mlp"):
            raise ValueError("kind must be 'ridge' or 'mlp'")
        self.kind, self.seed, self.kw = kind, seed, kw

    def _fit_one(self, Xtr, ytr, Xva, yva):
        if self.kind == "ridge":
            return _fit_ridge(Xtr, ytr, Xva, yva)
        return _fit_mlp(Xtr, ytr, Xva, yva, self.seed, **self.kw)

    def fit(self, train, val):
        self.f_y, self.hp_y = self._fit_one(train.yh, train.target, val.yh, val.target)
        self.f_xy, self.hp_xy = self._fit_one(train.xy, train.target, val.xy, val.target)
        return self

    def refit_frozen(self, train, refit_y=True):
        """A copy trained on (surrogate) train windows with this run's frozen hyperparameters."""
        clone = copy.copy(self)
        if refit_y:
            clone.f_y = _refit(self.kind, self.hp_y, train.yh, train.target, self.seed)
        clone.f_xy = _refit(self.kind, self.hp_xy, train.xy, train.target, self.seed)
        return clone

    def sq_errors(self, w):
        e_y = (self.f_y.predict(w.yh) - w.target) ** 2
        e_xy = (self.f_xy.predict(w.xy) - w.target) ** 2
        return e_y, e_xy

    def te(self, w):
        e_y, e_xy = self.sq_errors(w)
        return gain(e_y.mean(), e_xy.mean())
