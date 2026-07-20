"""Aggregates and inequality over the stationary agent distribution
(VFIToolkit EvalFnOnAgentDist_AggVars_Case1 + Lorenz / Gini / percentiles).

Given the stationary measure mu(a,z), a solved policy, and the grids, evaluate
a user function at each state's REALIZED choice and integrate over mu. The eval
function shares the engine's xp-threaded convention:

    no decision:   fn(aprime, a, z, *params, xp=np)
    with decision: fn(d, aprime, a, z, *params, xp=np)

where at each (a,z): aprime = a_grid[policy_aprime[a,z]] (realized next state),
a = a_grid[a], z = z_grid[z], d = d_grid[policy_d[a,z]]. Everything is numpy
(the distribution and policy are numpy). A partially-broadcast result is
expanded up to the full (n_a, n_z) shape (mirrors returnfn.build_return_tensor).
"""
from __future__ import annotations

import numpy as np


def evaluate_on_grid(fn, policy_aprime, a_grid, z_grid, *, params=None,
                     policy_d=None, d_grid=None):
    """Evaluate ``fn`` at each state's realized policy choice -> (n_a, n_z) array.

    ``a_grid`` is a 1-D array (single endogenous state) or a list/tuple of 1-D
    arrays (K states, C-order product). ``fn`` receives the K components as
    separate positional args, VFIToolkit order: ``[d,] a'_1..a'_K, a_1..a_K, z,
    *params`` (a'_j is the realized next value, a_j the current value).
    """
    pol = np.asarray(policy_aprime)
    n_a, n_z = pol.shape
    pvals = tuple((params or {}).values())
    if isinstance(a_grid, (list, tuple)):
        grids = [np.asarray(g, dtype=float) for g in a_grid]
        mesh = np.meshgrid(*grids, indexing="ij")
        comps = [m.reshape(-1) for m in mesh]
    else:
        comps = [np.asarray(a_grid, dtype=float)]
    ap_comps = [c[pol] for c in comps]            # each (n_a, n_z) realized next value
    a_comps = [c[:, None] for c in comps]         # each (n_a, 1) current value
    if isinstance(z_grid, (list, tuple)):
        zg = [np.asarray(g, dtype=float) for g in z_grid]
        if len(zg) == 1:
            z_comps = [zg[0]]
        else:
            zmesh = np.meshgrid(*zg, indexing="ij")
            z_comps = [m.reshape(-1) for m in zmesh]
    else:
        z_comps = [np.asarray(z_grid, dtype=float)]
    z_args = [c[None, :] for c in z_comps]
    if policy_d is None or d_grid is None:
        vals = fn(*ap_comps, *a_comps, *z_args, *pvals, xp=np)
    else:
        d = np.asarray(d_grid, dtype=float)[np.asarray(policy_d)]   # (n_a, n_z)
        vals = fn(d, *ap_comps, *a_comps, *z_args, *pvals, xp=np)
    vals = np.asarray(vals, dtype=float)
    if vals.shape != (n_a, n_z):
        vals = vals + np.zeros((n_a, n_z))
    return vals


def aggregate(fn, mu, policy_aprime, a_grid, z_grid, *, params=None,
              policy_d=None, d_grid=None):
    """Integral of ``fn`` over the agent distribution: sum(mu * fn_values)."""
    vals = evaluate_on_grid(fn, policy_aprime, a_grid, z_grid, params=params,
                            policy_d=policy_d, d_grid=d_grid)
    return float(np.sum(np.asarray(mu, dtype=float) * vals))


def lorenz_and_gini(mu, values):
    """Lorenz curve (pop_share, value_share) and Gini for a NONNEGATIVE value.

    States are sorted by ``values``; shares are mu-weighted. Returns
    (pop_share, value_share, gini), each share array prefixed with 0. Gini is
    1 - (area-under-Lorenz)*2 by the trapezoidal rule (0 = perfect equality).
    """
    w = np.asarray(mu, dtype=float).reshape(-1)
    v = np.asarray(values, dtype=float).reshape(-1)
    if np.any(v < 0) or np.any(np.isnan(v)):
        raise ValueError("Lorenz/Gini require finite nonnegative values")
    order = np.argsort(v, kind="stable")
    w = w[order]
    w = w / w.sum()
    v = v[order]
    pop = np.concatenate(([0.0], np.cumsum(w)))
    total = float(np.sum(w * v))
    if total <= 0.0:
        return pop, pop.copy(), 0.0  # all-zero value -> perfect equality
    val = np.concatenate(([0.0], np.cumsum(w * v) / total))
    gini = 1.0 - float(np.sum((pop[1:] - pop[:-1]) * (val[1:] + val[:-1])))
    return pop, val, gini


def weighted_quantile(mu, values, q):
    """Weighted quantile(s) ``q`` in [0,1] of ``values`` under weights ``mu``.

    Returns a float for scalar q, else an array. Uses the weighted CDF; ties at
    zero-weight gaps resolve to the upper value (np.interp convention).
    """
    w = np.asarray(mu, dtype=float).reshape(-1)
    v = np.asarray(values, dtype=float).reshape(-1)
    order = np.argsort(v, kind="stable")
    v = v[order]
    w = w[order]
    cdf = np.cumsum(w) / w.sum()
    qq = np.atleast_1d(np.asarray(q, dtype=float))
    out = np.interp(qq, cdf, v)
    return float(out[0]) if np.ndim(q) == 0 else out


__all__ = ["evaluate_on_grid", "aggregate", "lorenz_and_gini", "weighted_quantile"]
