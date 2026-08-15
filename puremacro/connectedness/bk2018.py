"""Baruník & Křehlík (2018) Frequency-Domain Connectedness Index."""
from __future__ import annotations

from typing import Dict, Any, Tuple
import numpy as np
from ..var.estimate import estimate_var


def barunik_krehlik(
    returns_panel: np.ndarray,
    var_lags: int = 4,
    fevd_horizon: int = 100,
    freq_bounds: Tuple[float, float] = (np.pi / 4, np.pi),
) -> Dict[str, Any]:
    """Frequency-domain spillover index decomposition (Baruník & Křehlík 2018)."""
    returns_panel = np.asarray(returns_panel, dtype=float)
    A_list, c, Sigma, resid, _ = estimate_var(returns_panel, var_lags)

    n = Sigma.shape[0]
    sigma_jj = np.diag(Sigma)

    n_freq = 300
    w_grid = np.linspace(freq_bounds[0], freq_bounds[1], n_freq)
    dw = w_grid[1] - w_grid[0]

    num_int = np.zeros((n, n))
    den_int = np.zeros(n)

    for w in w_grid:
        A_w = np.zeros((n, n), dtype=complex)
        for l in range(1, var_lags + 1):
            A_w += A_list[l - 1] * np.exp(-1j * w * l)

        Psi_w = np.linalg.inv(np.eye(n, dtype=complex) - A_w)
        PS_w = Psi_w @ Sigma

        for i in range(n):
            for j in range(n):
                num_int[i, j] += (np.abs(PS_w[i, j]) ** 2 / max(sigma_jj[j], 1e-12)) * dw
            den_int[i] += np.real((Psi_w[i, :] @ Sigma @ Psi_w[i, :].conj().T)) * dw

    theta_w = num_int / np.maximum(den_int[:, None], 1e-12)
    row_sums = np.sum(theta_w, axis=1, keepdims=True)
    row_sums = np.where(row_sums == 0, 1.0, row_sums)
    theta_w = theta_w / row_sums

    off_diag_sum = np.sum(theta_w) - np.sum(np.diag(theta_w))
    total_spillover = float(100.0 * off_diag_sum / n)

    return {
        "total_spillover": total_spillover,
        "band_matrix": theta_w,
    }
