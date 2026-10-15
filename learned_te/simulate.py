"""Synthetic systems with known ground truth, including the leak cases.

Every generator returns an (n, 2) array with columns [X_obs, Y_obs].
"""
import numpy as np

SCENARIOS = {
    # name: (true X->Y coupling?, description)
    "coupled": (True, "Y_{t+1} = aY_t + cX_t + eta  (real transfer)"),
    "independent": (False, "X, Y independent AR(1)"),
    "proxy_leak": (False, "X = Y + nu, Y observed with heavy noise (spurious TE)"),
    "shared_noise": (False, "independent X, Y plus a common slow noise source"),
}


def _ar1(n, a, rng, scale=1.0):
    e = rng.normal(0.0, scale, n)
    x = np.zeros(n)
    for t in range(1, n):
        x[t] = a * x[t - 1] + e[t]
    return x


def simulate(kind, n=6000, a=0.8, c=0.5, obs_noise=None, seed=0):
    rng = np.random.default_rng(seed)
    if kind == "coupled":
        x = _ar1(n, a, rng)
        eta = rng.normal(0.0, 1.0, n)
        y = np.zeros(n)
        for t in range(1, n):
            y[t] = a * y[t - 1] + c * x[t - 1] + eta[t]
    elif kind == "independent":
        x, y = _ar1(n, a, rng), _ar1(n, a, rng)
    elif kind == "proxy_leak":
        y = _ar1(n, a, rng)
        x = y + rng.normal(0.0, 0.5, n)
    elif kind == "shared_noise":
        common = _ar1(n, 0.95, rng, scale=0.6)
        x, y = _ar1(n, a, rng) + common, _ar1(n, a, rng) + common
    else:
        raise ValueError(f"unknown scenario {kind!r}; choose from {list(SCENARIOS)}")

    if obs_noise is None:
        obs_noise = 1.0 if kind == "proxy_leak" else 0.3
    x_obs = x + rng.normal(0.0, obs_noise, n)
    y_obs = y + rng.normal(0.0, obs_noise, n)
    return np.column_stack([x_obs, y_obs])
