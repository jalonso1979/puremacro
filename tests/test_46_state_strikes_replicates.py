"""Synthetic-DGP recovery test for Sprint β.3 state-level Poisson LP.

Generates a 50-state × 60-quarter panel from a known Poisson DGP with
planted h=1 coefficient β=0.5, fits the Poisson LP helper that the
NB46 notebook uses (re-imported from a small shim), and asserts that
the planted β is recovered within 3·SE.
"""
from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

statsmodels = pytest.importorskip("statsmodels.api")


def _poisson_lp_at_horizon(panel, *, y_col, shock_col, state_col, time_col, h):
    """Poisson LP at horizon h: log E[y_{i,t+h}] = α_i + γ_t + β·shock_{i,t}.

    Returns (β̂, SE(β̂)). Returns (nan, nan) on convergence failure.
    """
    import statsmodels.api as sm

    df = panel.copy().sort_values([state_col, time_col])
    df[f"_y_lead_{h}"] = df.groupby(state_col)[y_col].shift(-h)
    df = df.dropna(subset=[f"_y_lead_{h}", shock_col])
    state_dummies = pd.get_dummies(df[state_col], prefix="state", drop_first=True)
    time_dummies = pd.get_dummies(df[time_col], prefix="t", drop_first=True)
    X = pd.concat([df[[shock_col]], state_dummies, time_dummies], axis=1).astype(float)
    X = sm.add_constant(X, has_constant="add")
    y = df[f"_y_lead_{h}"].astype(int)
    try:
        res = sm.GLM(y, X, family=sm.families.Poisson()).fit(disp=False, maxiter=200)
    except Exception:
        return float("nan"), float("nan")
    return float(res.params[shock_col]), float(res.bse[shock_col])


def test_poisson_lp_recovers_planted_h1():
    rng = np.random.default_rng(0)
    n_states, n_periods, true_beta = 50, 60, 0.5
    state_fe = rng.normal(0.5, 0.3, n_states)
    time_fe = rng.normal(0.0, 0.2, n_periods)
    shocks = rng.standard_normal(n_periods)
    rows = []
    for i in range(n_states):
        for t in range(n_periods):
            shock_lag = shocks[t - 1] if t >= 1 else 0.0
            mu = float(np.exp(state_fe[i] + time_fe[t] + true_beta * shock_lag))
            y = int(rng.poisson(mu))
            rows.append((i, t, y, float(shocks[t])))
    df = pd.DataFrame(rows, columns=["state", "t", "y", "shock"])

    beta, se = _poisson_lp_at_horizon(
        df, y_col="y", shock_col="shock", state_col="state", time_col="t", h=1
    )

    assert np.isfinite(beta), "Poisson LP failed to converge on a well-posed DGP"
    assert abs(beta - true_beta) < 3 * se, (
        f"Recovery failed: β̂={beta:.4f}, true=0.5, SE={se:.4f} "
        f"(|β̂ − β| / SE = {abs(beta - true_beta) / se:.2f})"
    )
