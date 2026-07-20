"""Numba-compiled counterparts of the hot nested-DMP kernels.

CPU JIT (``@njit``) and parallel (``@njit(parallel=True)`` + ``prange``)
implementations of the Bellman VFI workhorse. These mirror
``kernels.bellman_iterate`` exactly and are validated against it (the
NumPy oracle) in the cross-backend tests. Imported lazily — importing
this module requires the optional ``[backend]`` extra (numba).

numba.cuda (NVIDIA GPU) kernels are deferred to a later phase (no NVIDIA
hardware on the Apple-Silicon dev machine).
"""
from __future__ import annotations

import numpy as np
from numba import njit, prange


@njit(cache=True)
def bellman_iterate_numba(flow, P, beta, tol, max_iter):
    """JIT VFI: V = max(flow + beta * (P @ V), 0). Mirrors kernels.bellman_iterate."""
    n = flow.shape[0]
    V = np.zeros(n)
    converged = False
    for _ in range(max_iter):
        cont = flow + beta * (P @ V)
        V_new = np.maximum(cont, 0.0)
        diff = np.max(np.abs(V_new - V))
        V = V_new
        if diff < tol:
            converged = True
            break
    if not converged:
        raise RuntimeError("bellman_iterate_numba did not converge")
    return V


@njit(parallel=True, cache=True)
def bellman_iterate_batch_numba(flow_batch, P_batch, beta, tol, max_iter):
    """Parallel VFI over a batch of independent problems (prange over batch)."""
    B, n = flow_batch.shape
    out = np.zeros((B, n))
    for b in prange(B):
        out[b] = bellman_iterate_numba(flow_batch[b], P_batch[b], beta, tol, max_iter)
    return out


__all__ = ["bellman_iterate_numba", "bellman_iterate_batch_numba"]
