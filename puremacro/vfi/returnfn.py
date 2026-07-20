"""Build the discrete return tensor R(d, a', a, z) from a user return function.

The return function is xp-generic: it is called with broadcast grid arrays in
the active namespace plus a trailing ``xp`` keyword, so the SAME function builds
the tensor under numpy (oracle), mlx (Apple GPU) and cupy (NVIDIA GPU). Model
parameters are passed positionally, in ``params`` dict-insertion order, after
the state arguments (so the numba scalar path, which cannot take kwargs, could
share the convention). Signature:

    no decision:   return_fn(aprime, a, z, *params, xp=np)
    with decision: return_fn(d, aprime, a, z, *params, xp=np)

Infeasible choices (e.g. negative consumption) should return -inf or a large
negative number; the VFI driver raises if a state has no finite-payoff choice
at all.
"""
from __future__ import annotations

import numpy as np


def build_return_tensor(return_fn, a_grid, z_grid, params, d_grid=None, xp=np):
    """Return tensor of shape (n_d, n_a', n_a, n_z) on backend ``xp``.

    ``a_grid`` and/or ``z_grid`` may each be a 1-D array (single state) OR a
    list/tuple of 1-D arrays (multiple endogenous / exogenous states), flattened
    to the C-order product. The return function receives the components as
    separate positional args, in VFIToolkit order
    ``[d,] a'_1..a'_K, a_1..a_K, z_1..z_M, *params``. When ``d_grid is None`` a
    singleton decision axis (n_d=1) is prepended. A partially broadcast result is
    expanded up to the full shape.
    """
    def _components(grid):
        if isinstance(grid, (list, tuple)):
            gg = [np.asarray(g, dtype=float) for g in grid]
            if len(gg) == 1:
                return [xp.array(gg[0])], int(gg[0].shape[0])
            mesh = np.meshgrid(*gg, indexing="ij")    # C-order; last varies fastest
            n = int(np.prod([g.shape[0] for g in gg]))
            return [xp.array(m.reshape(-1)) for m in mesh], n
        g = np.asarray(grid, dtype=float)
        return [xp.array(g)], int(g.shape[0])

    a_comps, n_a = _components(a_grid)
    z_comps, n_z = _components(z_grid)
    pvals = tuple(params.values())

    if d_grid is None:
        ap_args = [c.reshape(n_a, 1, 1) for c in a_comps]
        a_args = [c.reshape(1, n_a, 1) for c in a_comps]
        z_args = [c.reshape(1, 1, n_z) for c in z_comps]
        R = return_fn(*ap_args, *a_args, *z_args, *pvals, xp=xp)
        target = (n_a, n_a, n_z)
        if tuple(R.shape) != target:
            R = R + xp.zeros(target, dtype=R.dtype)
        return R.reshape(1, n_a, n_a, n_z)

    n_d = int(np.asarray(d_grid).shape[0])
    D = xp.array(np.asarray(d_grid, dtype=float))
    d = D.reshape(n_d, 1, 1, 1)
    ap_args = [c.reshape(1, n_a, 1, 1) for c in a_comps]
    a_args = [c.reshape(1, 1, n_a, 1) for c in a_comps]
    z_args = [c.reshape(1, 1, 1, n_z) for c in z_comps]
    R = return_fn(d, *ap_args, *a_args, *z_args, *pvals, xp=xp)
    target = (n_d, n_a, n_a, n_z)
    if tuple(R.shape) != target:
        R = R + xp.zeros(target, dtype=R.dtype)
    return R


__all__ = ["build_return_tensor"]
