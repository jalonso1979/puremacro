"""xp-generic VFI driver: pure Bellman iteration + Howard policy improvement.

Runs unchanged under numpy / mlx / cupy. The exogenous expectation is the
matmul V @ P_z.T; the policy gathers use take_along_axis (portable across the
three namespaces). The numba backend has a separate compiled twin in
kernels_numba.solve_vfi_numba.
"""
from __future__ import annotations

import numpy as np

from puremacro.vfi.kernels import bellman_step


def solve_vfi(R, P_z, beta, *, howard=True, n_howard=20, tol=1e-8,
              max_iter=10_000, xp=np):
    """Solve V(a,z) = max_{d,a'} R(d,a',a,z) + beta E[V(a',z')|z].

    R and P_z must live on backend ``xp``. Returns
    (V (n_a,n_z), flat_idx (n_a,n_z), n_ap, n_iter, sup_norm). flat_idx encodes
    k = d*n_a' + a'. With ``howard`` the value is iterated under the greedy
    policy ``n_howard`` times per outer step (a gather + matmul, no max), which
    sharply cuts the outer-iteration count. ``n_a' == n_a`` (a' is chosen on the
    a-grid), so EV = V @ P_z.T is reused as EV(a', z).
    Infeasible choices may be encoded as -inf (or a large negative number); the
    solver raises RuntimeError if some state has no feasible action at all.
    """
    n_d, n_ap, n_a, n_z = R.shape
    Rflat = R.reshape(n_d * n_ap, n_a, n_z)
    V = xp.zeros((n_a, n_z))
    PzT = P_z.T
    sup = float("inf")
    for it in range(1, max_iter + 1):
        EV = V @ PzT
        V_new, flat_idx = bellman_step(R, EV, beta, xp=xp)
        if not bool(xp.all(xp.isfinite(V_new))):
            raise RuntimeError(
                "VFI: a state has no feasible action (value is -inf). Every "
                "(a, z) needs at least one finite-payoff choice; check the "
                "return function's feasibility region."
            )
        sup = float(xp.max(xp.abs(V_new - V)))
        V = V_new
        if howard and n_howard:
            R_pol = xp.take_along_axis(
                Rflat, xp.expand_dims(flat_idx, 0), axis=0
            )[0]
            aprime_idx = flat_idx % n_ap
            for _ in range(n_howard):
                EV_pol = xp.take_along_axis(V @ PzT, aprime_idx, axis=0)
                V = R_pol + beta * EV_pol
        if sup < tol:
            return V, flat_idx, n_ap, it, sup
    raise RuntimeError(
        f"solve_vfi did not converge in {max_iter} iterations "
        f"(sup-norm {sup:.3e} > tol {tol:.1e})"
    )


__all__ = ["solve_vfi"]
