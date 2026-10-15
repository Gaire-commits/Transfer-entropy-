"""Noise-invariance training for the paired predictors.

Environments are copies of the training windows with extra white noise of std
sigma_e added to the Y channel (history and target). For fixed models, each
environment gives a gain

    g_e = 0.5 * ln(MSE_Y^e / MSE_XY^e).

Leakage through a noisy conditioning history makes g_e *grow* with sigma_e;
real transfer is diluted by target noise and does not. The loss is

    L = MSE_Y + MSE_XY  (clean data)  +  lambda * penalty(g_1..g_E)

  penalty = "monotone": sum_e relu(g_{e+1} - g_e)   only growth is penalised (default)
  penalty = "vrex":     Var_e(g_e)                  any change is penalised

f_Y is trained first and frozen, so the penalty acts only on f_XY and cannot be
satisfied by making the Y-only baseline worse or by cutting its training short.
"""
import copy

import numpy as np
import torch
from torch import nn

torch.set_num_threads(1)  # tiny full-batch models; threading overhead dominates


def _net(d_in, hidden):
    if hidden == 0:
        return nn.Linear(d_in, 1)
    return nn.Sequential(nn.Linear(d_in, hidden), nn.Tanh(),
                         nn.Linear(hidden, hidden), nn.Tanh(), nn.Linear(hidden, 1))


def _tensors(w):
    f = lambda a: torch.as_tensor(np.asarray(a), dtype=torch.float32)
    return f(w.yh), f(w.xh), f(w.target)


def _mse(pred, target):
    return torch.mean((pred.squeeze(-1) - target) ** 2)


class InvariantPairedPredictors:
    """Same interface as models.PairedPredictors (fit, sq_errors, te, refit_frozen)."""

    def __init__(self, seed=0, lam=10.0, sigmas=(0.0, 0.5, 1.0), penalty="monotone",
                 hidden=0, lr=1e-2, max_epochs=600, patience=40):
        if penalty not in ("monotone", "vrex"):
            raise ValueError("penalty must be 'monotone' or 'vrex'")
        self.seed, self.lam, self.sigmas, self.penalty = seed, lam, tuple(sigmas), penalty
        self.hidden, self.lr, self.max_epochs, self.patience = hidden, lr, max_epochs, patience

    # ---- objective -------------------------------------------------------
    def _env_gains(self, f_y, f_xy, yh, xh, tgt, gen):
        gains = []
        for s in self.sigmas:
            yh_e, tgt_e = yh, tgt
            if s > 0:
                yh_e = yh + s * torch.randn(yh.shape, generator=gen)
                tgt_e = tgt + s * torch.randn(tgt.shape, generator=gen)
            m_y = _mse(f_y(yh_e), tgt_e).detach()
            m_xy = _mse(f_xy(torch.cat([yh_e, xh], 1)), tgt_e)
            gains.append(0.5 * (torch.log(m_y) - torch.log(m_xy)))
        return torch.stack(gains)

    def _penalty(self, g):
        if self.penalty == "vrex":
            return g.var()
        return torch.relu(g[1:] - g[:-1]).sum()

    def _y_objective(self, f_y, f_xy, data, gen):
        yh, _, tgt = data
        return _mse(f_y(yh), tgt)

    def _xy_objective(self, f_y, f_xy, data, gen):
        yh, xh, tgt = data
        risk = _mse(f_xy(torch.cat([yh, xh], 1)), tgt)
        return risk + self.lam * self._penalty(self._env_gains(f_y, f_xy, yh, xh, tgt, gen))

    # ---- training --------------------------------------------------------
    # Two stages, so the penalty can never shorten f_Y's training (a shared early-
    # stopping point left f_Y undertrained at large lambda, which reads as transfer):
    #   1. f_Y alone, early-stopped on its own val MSE, then frozen;
    #   2. f_XY with the penalty against frozen f_Y, early-stopped on its val objective.
    def _stage(self, net, params, objective, f_y, f_xy, tr, va, epochs):
        opt = torch.optim.Adam(params, lr=self.lr)
        gen = torch.Generator().manual_seed(self.seed)
        best, stale = (np.inf, None, 0), 0
        for epoch in range(1, (epochs or self.max_epochs) + 1):
            opt.zero_grad()
            objective(f_y, f_xy, tr, gen).backward()
            opt.step()
            if va is None:
                continue
            with torch.no_grad():  # fixed val noise -> comparable across epochs
                v = objective(f_y, f_xy, va, torch.Generator().manual_seed(self.seed + 1)).item()
            if v < best[0] - 1e-6:
                best, stale = (v, copy.deepcopy(net.state_dict()), epoch), 0
            else:
                stale += 1
                if stale >= self.patience:
                    break
        if va is None:
            return epochs
        net.load_state_dict(best[1])
        return best[2]

    def _train(self, train, val=None, epochs=None):
        torch.manual_seed(self.seed)
        tr = _tensors(train)
        va = _tensors(val) if val is not None else None
        k, l = tr[0].shape[1], tr[1].shape[1]
        f_y, f_xy = _net(k, self.hidden), _net(k + l, self.hidden)
        ep_y, ep_xy = epochs or (None, None)

        ep_y = self._stage(f_y, f_y.parameters(), self._y_objective, f_y, f_xy, tr, va, ep_y)
        f_y.requires_grad_(False)

        if self.hidden == 0:  # start f_XY at f_Y with zero X weights, i.e. at gain = 0
            with torch.no_grad():
                f_xy.weight.zero_()
                f_xy.weight[:, :k] = f_y.weight
                f_xy.bias.copy_(f_y.bias)
        ep_xy = self._stage(f_xy, f_xy.parameters(), self._xy_objective, f_y, f_xy, tr, va, ep_xy)

        if va is not None:
            self.epochs = (ep_y, ep_xy)
        self.f_y, self.f_xy = f_y.eval(), f_xy.eval()
        return self

    def fit(self, train, val):
        return self._train(train, val)

    def refit_frozen(self, train, refit_y=True):
        """Retrain from scratch on (surrogate) train data for the frozen epoch counts, no val."""
        clone = copy.copy(self)
        return clone._train(train, epochs=self.epochs)

    # ---- evaluation ------------------------------------------------------
    def sq_errors(self, w):
        yh, xh, tgt = _tensors(w)
        with torch.no_grad():
            e_y = (self.f_y(yh).squeeze(-1) - tgt) ** 2
            e_xy = (self.f_xy(torch.cat([yh, xh], 1)).squeeze(-1) - tgt) ** 2
        return e_y.numpy().astype(float), e_xy.numpy().astype(float)

    def te(self, w):
        e_y, e_xy = self.sq_errors(w)
        return 0.5 * np.log(e_y.mean() / e_xy.mean())

    def env_gains(self, w, seed=0):
        """Gain of the trained models at each noise level (a diagnostic, not used for tuning)."""
        with torch.no_grad():
            g = self._env_gains(self.f_y, self.f_xy, *_tensors(w), torch.Generator().manual_seed(seed))
        return dict(zip(self.sigmas, g.numpy().round(4).tolist()))
