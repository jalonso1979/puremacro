"""Abadie, Diamond & Hainmueller (2010) Synthetic Control Method."""
from __future__ import annotations

from typing import Dict, Any
import numpy as np
import scipy.optimize


def synthetic_control(
    Y_matrix: np.ndarray,
    treated_idx: int,
    treat_time: int,
) -> Dict[str, Any]:
    """Synthetic Control Method for micro/macro policy evaluation."""
    Y_matrix = np.asarray(Y_matrix, dtype=float)
    T, N_total = Y_matrix.shape

    donor_mask = np.ones(N_total, dtype=bool)
    donor_mask[treated_idx] = False

    y_treated = Y_matrix[:, treated_idx]
    Y_donors = Y_matrix[:, donor_mask]
    N_donors = Y_donors.shape[1]

    T_pre = treat_time - 1
    y_pre = y_treated[:T_pre]
    Y_donors_pre = Y_donors[:T_pre, :]

    def objective(w):
        return np.sum((y_pre - Y_donors_pre @ w) ** 2)

    w0 = np.ones(N_donors) / N_donors
    bounds = [(0.0, 1.0) for _ in range(N_donors)]
    constraints = {"type": "eq", "fun": lambda w: np.sum(w) - 1.0}

    res_opt = scipy.optimize.minimize(objective, w0, bounds=bounds, constraints=constraints, method="SLSQP")
    w_opt = res_opt.x

    y_synthetic = Y_donors @ w_opt
    att_path = y_treated - y_synthetic
    att_post = float(np.mean(att_path[treat_time:]))

    return {
        "weights": w_opt,
        "y_treated": y_treated,
        "y_synthetic": y_synthetic,
        "att_path": att_path,
        "att_post": att_post,
        "treat_time": treat_time,
    }
