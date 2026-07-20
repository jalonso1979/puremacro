"""Numba-compiled VFI driver (CPU JIT). Mirrors puremacro.vfi.solve.solve_vfi
and is validated against the numpy oracle in the cross-backend tests.

The return tensor is built in numpy (fast broadcasting, any numpy return_fn);
numba accelerates the hot fixed-point loop -- the same split the companion
model uses (kernels_numba.bellman_iterate_numba). A numba-built return tensor
is a deferred optimisation. Imported lazily (optional [backend] extra).

Tie-breaking matches numpy.argmax (first maximiser wins) via strict ``>``, so
the integer policy is identical to the oracle's.

The all-infeasible-state guard (raise RuntimeError when best == -inf) mirrors
the guard in solve_vfi so that -inf values never silently poison the sup-norm.
"""
from __future__ import annotations

import numpy as np
from numba import njit


@njit(cache=True)
def solve_vfi_numba(R, P_z, beta, howard, n_howard, tol, max_iter):
    n_d, n_ap, n_a, n_z = R.shape
    V = np.zeros((n_a, n_z))
    flat_pol = np.zeros((n_a, n_z), dtype=np.int64)
    EV = np.zeros((n_ap, n_z))
    sup = np.inf
    for _it in range(max_iter):
        for iap in range(n_ap):
            for iz in range(n_z):
                s = 0.0
                for jz in range(n_z):
                    s += V[iap, jz] * P_z[iz, jz]
                EV[iap, iz] = s
        sup = 0.0
        for ia in range(n_a):
            for iz in range(n_z):
                best = -np.inf
                bestk = 0
                for idd in range(n_d):
                    for iap in range(n_ap):
                        q = R[idd, iap, ia, iz] + beta * EV[iap, iz]
                        if q > best:
                            best = q
                            bestk = idd * n_ap + iap
                if best == -np.inf:
                    raise RuntimeError(
                        "solve_vfi_numba: a state has no feasible action (-inf)"
                    )
                diff = abs(best - V[ia, iz])
                if diff > sup:
                    sup = diff
                V[ia, iz] = best
                flat_pol[ia, iz] = bestk
        if howard:
            for _h in range(n_howard):
                for iap in range(n_ap):
                    for iz in range(n_z):
                        s = 0.0
                        for jz in range(n_z):
                            s += V[iap, jz] * P_z[iz, jz]
                        EV[iap, iz] = s
                for ia in range(n_a):
                    for iz in range(n_z):
                        k = flat_pol[ia, iz]
                        idd = k // n_ap
                        iap = k % n_ap
                        V[ia, iz] = R[idd, iap, ia, iz] + beta * EV[iap, iz]
        if sup < tol:
            return V, flat_pol, _it + 1, sup
    raise RuntimeError("solve_vfi_numba did not converge")


@njit(cache=True)
def _dc_fill_numba(R, EV, beta, idd, iz, n_ap, n_a, cand_val, cand_pol):
    """Fill cand_val[ia], cand_pol[ia] = max_{a'} (R[idd,a',ia,iz]+beta EV[a',iz])
    for every ia, by monotone divide-and-conquer (assumes the argmax is
    non-decreasing in ia). Q is evaluated on the fly -- never materialized.
    Explicit LIFO stack of (lo, hi, alo, ahi)."""
    cap = 4 * (n_a + 2)
    st_lo = np.empty(cap, dtype=np.int64)
    st_hi = np.empty(cap, dtype=np.int64)
    st_alo = np.empty(cap, dtype=np.int64)
    st_ahi = np.empty(cap, dtype=np.int64)
    sp = 0
    st_lo[sp] = 0; st_hi[sp] = n_ap - 1; st_alo[sp] = 0; st_ahi[sp] = n_a - 1
    sp += 1
    while sp > 0:
        sp -= 1
        lo = st_lo[sp]; hi = st_hi[sp]; alo = st_alo[sp]; ahi = st_ahi[sp]
        if alo > ahi:
            continue
        amid = (alo + ahi) // 2
        best = -np.inf
        bestj = lo
        for j in range(lo, hi + 1):
            q = R[idd, j, amid, iz] + beta * EV[j, iz]
            if q > best:                       # strict => smallest maximiser (first-max)
                best = q
                bestj = j
        cand_val[amid] = best
        cand_pol[amid] = bestj
        st_lo[sp] = lo; st_hi[sp] = bestj; st_alo[sp] = alo; st_ahi[sp] = amid - 1
        sp += 1
        st_lo[sp] = bestj; st_hi[sp] = hi; st_alo[sp] = amid + 1; st_ahi[sp] = ahi
        sp += 1


@njit(cache=True)
def solve_vfi_dc_numba(R, P_z, beta, howard, n_howard, tol, max_iter):
    """Divide-and-conquer VFI (monotone-policy accelerator). Same contract as
    solve_vfi_numba -- returns (V, flat_pol, n_iter, sup) and is EXACT (equals
    brute force) when the optimal a'-policy is monotone non-decreasing in a
    (supermodular / concave problems). The greedy step is O(n_ap log n_a) per
    (d,z) instead of O(n_ap*n_a). Tie-break matches numpy.argmax (first-max:
    smallest d, then smallest a')."""
    n_d, n_ap, n_a, n_z = R.shape
    V = np.zeros((n_a, n_z))
    flat_pol = np.zeros((n_a, n_z), dtype=np.int64)
    EV = np.zeros((n_ap, n_z))
    cand_val = np.empty(n_a)
    cand_pol = np.empty(n_a, dtype=np.int64)
    Vrow = np.empty(n_a)
    Krow = np.empty(n_a, dtype=np.int64)
    sup = np.inf
    for _it in range(max_iter):
        for iap in range(n_ap):
            for iz in range(n_z):
                s = 0.0
                for jz in range(n_z):
                    s += V[iap, jz] * P_z[iz, jz]
                EV[iap, iz] = s
        sup = 0.0
        for iz in range(n_z):
            for ia in range(n_a):
                Vrow[ia] = -np.inf
                Krow[ia] = 0
            for idd in range(n_d):                  # one DC sweep per (iz, d), merge by >
                _dc_fill_numba(R, EV, beta, idd, iz, n_ap, n_a,
                               cand_val, cand_pol)
                for ia in range(n_a):
                    if cand_val[ia] > Vrow[ia]:     # strict => smaller d wins ties
                        Vrow[ia] = cand_val[ia]
                        Krow[ia] = idd * n_ap + cand_pol[ia]
            for ia in range(n_a):
                if Vrow[ia] == -np.inf:
                    raise RuntimeError(
                        "solve_vfi_dc_numba: a state has no feasible action (-inf)"
                    )
                diff = abs(Vrow[ia] - V[ia, iz])
                if diff > sup:
                    sup = diff
                V[ia, iz] = Vrow[ia]
                flat_pol[ia, iz] = Krow[ia]
        if howard:
            for _h in range(n_howard):
                for iap in range(n_ap):
                    for iz in range(n_z):
                        s = 0.0
                        for jz in range(n_z):
                            s += V[iap, jz] * P_z[iz, jz]
                        EV[iap, iz] = s
                for ia in range(n_a):
                    for iz in range(n_z):
                        k = flat_pol[ia, iz]
                        idd = k // n_ap
                        iap = k % n_ap
                        V[ia, iz] = R[idd, iap, ia, iz] + beta * EV[iap, iz]
        if sup < tol:
            return V, flat_pol, _it + 1, sup
    raise RuntimeError("solve_vfi_dc_numba did not converge")


__all__ = ["solve_vfi_numba", "solve_vfi_dc_numba"]
