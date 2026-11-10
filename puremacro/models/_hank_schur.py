"""Checked stable policies from real reduced HANK equation pencils."""
from __future__ import annotations

from typing import Any

import numpy as np
from scipy.linalg import ordqz, solve


def _solve_schur_policy(F0: np.ndarray, F1: np.ndarray, ns: int, *,
                        output: str = "real") -> tuple[np.ndarray, np.ndarray, dict[str, Any]]:
    """Solve F0 y_t + F1 E_t y_{t+1} = 0 with the first ns entries as states.

    Real Schur form retains conjugate eigenvalues in 2 by 2 blocks. Solving
    the entire stable T,S blocks is essential; their diagonals alone do not
    describe a rotating stable mode. The complex form is a comparison path.
    """
    if output not in ("real", "complex"):
        raise ValueError("schur_output must be 'real' or 'complex'")
    S, T, alpha, beta, _, Z = ordqz(-F0, F1, sort="iuc", output=output)
    with np.errstate(divide="ignore", invalid="ignore"):
        modulus = abs(alpha/beta)
    stable = int(np.sum(modulus < 1-1e-8))
    if np.isnan(modulus).any() or np.any(abs(modulus-1) < 1e-8) or stable != ns:
        raise ValueError(f"projected stochastic equilibrium fails determinacy: {stable} stable roots for {ns} states")
    Zx = Z[:ns, :ns]
    condition = float(np.linalg.cond(Zx))
    if not np.isfinite(condition) or condition > 1e12:
        raise ValueError("stable Schur subspace does not identify a well-conditioned state policy")
    G = solve(Zx.T, Z[ns:, :ns].T).T
    H = solve(Zx.T, (Zx@solve(T[:ns, :ns], S[:ns, :ns])).T).T
    if not np.isfinite(G).all() or not np.isfinite(H).all():
        raise ValueError("nonfinite Schur policy")
    if max(float(abs(G.imag).max()), float(abs(H.imag).max())) > 1e-8:
        raise ValueError("non-real Schur policy")
    G, H = G.real, H.real
    stack = np.vstack([np.eye(ns), G])
    residual = float(np.max(abs(F0@stack+F1@stack@H)))
    radius = float(np.max(abs(np.linalg.eigvals(H))))
    if not np.isfinite([residual, radius]).all() or residual > 1e-8 or radius >= 1-1e-8:
        raise ValueError("projected stochastic equilibrium failed equation or stability checks")
    return H, G, dict(schur_output=output, equilibrium_residual_max_abs=residual,
                     projected_equilibrium_residual_max_abs=residual,
                     spectral_radius=radius, stable_roots=stable,
                     state_projection_condition=condition)
