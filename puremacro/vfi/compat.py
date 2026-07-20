"""VFIToolkit-signature compatibility shim.

A thin wrapper mapping VFIToolkit's MATLAB-style ``ValueFnIter_Case1`` signature
onto the native :class:`~puremacro.vfi.problem.VFIProblem`, to ease 1:1 porting
of existing models. For new code, prefer ``VFIProblem`` directly.
"""
from __future__ import annotations

import numpy as np

from puremacro.vfi.problem import VFIProblem


def value_fn_iter_case1(n_d, n_a, n_z, d_grid, a_grid, z_grid, pi_z, return_fn,
                        beta, *, params=None, backend="numpy", **options):
    """Case-1 infinite-horizon solve with a VFIToolkit-style signature.

    Maps ``(n_d, n_a, n_z, d_grid, a_grid, z_grid, pi_z, ReturnFn, beta)`` onto a
    :class:`VFIProblem` and returns ``(V, Policy)``:

    - ``V`` has shape ``(n_a, n_z)``.
    - ``Policy`` is the a'-index array ``(n_a, n_z)`` when there is no decision
      (``n_d`` in {0, 1, None}), else ``np.stack([policy_d, policy_aprime])`` of
      shape ``(2, n_a, n_z)`` (decision index on top, next-asset index below).

    ``ReturnFn`` follows the engine convention: ``(aprime, a, z, *params, xp=np)``
    with no decision, ``(d, aprime, a, z, *params, xp=np)`` with one. Extra
    keyword ``options`` are forwarded to ``VFIProblem.options`` (howard,
    n_howard, tol, max_iter); ``backend`` selects the solve backend. The sizes
    ``n_d/n_a/n_z`` are validated against the grids (a common porting footgun).
    For new code, prefer the native ``VFIProblem`` API.
    """
    a_grid = np.asarray(a_grid)
    z_grid = np.asarray(z_grid)
    if int(n_a) != a_grid.shape[0]:
        raise ValueError(f"n_a ({n_a}) != len(a_grid) ({a_grid.shape[0]})")
    if int(n_z) != z_grid.shape[0]:
        raise ValueError(f"n_z ({n_z}) != len(z_grid) ({z_grid.shape[0]})")
    has_d = (n_d is not None) and (int(n_d) > 1)
    if has_d and int(n_d) != np.asarray(d_grid).shape[0]:
        raise ValueError(
            f"n_d ({n_d}) != len(d_grid) ({np.asarray(d_grid).shape[0]})"
        )
    prob = VFIProblem(
        a_grid=a_grid, z_grid=z_grid, P_z=pi_z, return_fn=return_fn, beta=beta,
        params=params or {}, d_grid=(np.asarray(d_grid) if has_d else None),
        options=options or {},
    )
    sol = prob.solve(backend)
    if has_d:
        policy = np.stack([sol.policy_d, sol.policy_aprime])
    else:
        policy = sol.policy_aprime
    return sol.V, policy


__all__ = ["value_fn_iter_case1"]
