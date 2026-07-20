"""xp-generic VFI inner kernel: one greedy Bellman maximisation step.

Written once against a numpy-like namespace ``xp`` (operators + reshape + max +
argmax), so it runs unchanged under numpy (oracle), mlx and cupy. reshape is
used instead of None-indexing for portability.
"""
from __future__ import annotations

import numpy as np


def bellman_step(R, EV, beta, *, xp=np):
    """One greedy Bellman step.

    R  : return tensor (n_d, n_a', n_a, n_z).
    EV : expected continuation (n_a', n_z) = E[V(a', z') | z].
    Returns (V_new (n_a, n_z), flat_idx (n_a, n_z)) where flat_idx encodes the
    argmax over the flattened (d, a') choice as k = d*n_a' + a'.
    """
    n_d, n_ap, n_a, n_z = R.shape
    Q = R + beta * EV.reshape(1, n_ap, 1, n_z)
    Qflat = Q.reshape(n_d * n_ap, n_a, n_z)
    V_new = xp.max(Qflat, axis=0)
    flat_idx = xp.argmax(Qflat, axis=0)
    return V_new, flat_idx


__all__ = ["bellman_step"]
