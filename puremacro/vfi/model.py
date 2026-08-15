"""Dynare-like Declarative Automated Dynamic Programming Model Solver."""
from __future__ import annotations

import time
from typing import Callable, Optional, Dict, Any, List
import numpy as np

from .discretize import tauchen, rouwenhorst
from .solve import solve_vfi_howard, simulate_vfi_panel


class Model:
    """Declarative Dynare-like VFI Model interface.

    Example
    -------
    >>> m = Model("Neoclassical Growth")
    >>> m.set_param("beta", 0.96).set_param("gamma", 2.0).set_param("alpha", 0.36).set_param("delta", 0.10)
    >>> m.add_state("k", 0.1, 15.0, 50).add_shock("z", 0.90, 0.15, 5)
    >>> m.set_utility(lambda c, p: (c**(1 - p['gamma'])) / (1 - p['gamma']))
    >>> m.set_budget(lambda k, kp, z, p: np.exp(z) * (k**p['alpha']) + (1 - p['delta']) * k - kp)
    >>> res = m.solve()
    """

    def __init__(self, name: str = "VFI Model"):
        self.name = name
        self.params: Dict[str, float] = {}
        self.states: List[Dict[str, Any]] = []
        self.shocks: List[Dict[str, Any]] = []
        self.utility_fn: Optional[Callable] = None
        self.budget_fn: Optional[Callable] = None

    def set_param(self, name: str, val: float) -> Model:
        self.params[name] = val
        return self

    def add_state(self, name: str, min_val: float, max_val: float, n_points: int) -> Model:
        self.states.append({"name": name, "min": min_val, "max": max_val, "n": n_points})
        return self

    def add_shock(self, name: str, rho: float, sigma: float, n_points: int, method: str = "tauchen") -> Model:
        self.shocks.append({"name": name, "rho": rho, "sigma": sigma, "n": n_points, "method": method})
        return self

    def set_utility(self, fn: Callable) -> Model:
        self.utility_fn = fn
        return self

    def set_budget(self, fn: Callable) -> Model:
        self.budget_fn = fn
        return self

    def solve(self, howard_steps: int = 20, tol: float = 1e-7) -> Dict[str, Any]:
        t_start = time.time()

        if not self.shocks:
            z_grid = np.array([0.0])
            P_z = np.array([[1.0]])
        else:
            shk = self.shocks[0]
            if shk["method"] == "rouwenhorst":
                z_grid, P_z = rouwenhorst(shk["n"], shk["rho"], shk["sigma"])
            else:
                z_grid, P_z = tauchen(shk["n"], shk["rho"], shk["sigma"])

        st = self.states[0]
        a_grid = np.linspace(st["min"], st["max"], st["n"])

        def return_wrapper(a_val, ap_val, z_val):
            c_val = self.budget_fn(a_val, ap_val, z_val, self.params)
            if c_val <= 0 or not np.isreal(c_val) or np.isnan(c_val):
                return -1e10
            return float(self.utility_fn(c_val, self.params))

        res_vfi = solve_vfi_howard(return_wrapper, a_grid, z_grid, P_z, self.params["beta"], howard_steps=howard_steps, tol=tol)
        res_sim = simulate_vfi_panel(res_vfi["policy_a"], a_grid, z_grid, P_z, n_agents=2000, T_periods=100)

        elapsed = time.time() - t_start

        return {
            "model_name": self.name,
            "V": res_vfi["V"],
            "policy_a": res_vfi["policy_a"],
            "policy_idx": res_vfi["policy_idx"],
            "a_grid": a_grid,
            "z_grid": z_grid,
            "P_z": P_z,
            "n_iter": res_vfi["n_iter"],
            "elapsed": elapsed,
            "mean_assets": res_sim["mean_assets"],
            "gini": res_sim["gini"],
        }
