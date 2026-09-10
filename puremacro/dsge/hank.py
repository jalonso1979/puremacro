"""Heterogeneous Agents (HANK) Sequence-Space Bridge in .mod files.

Bridges Dynare-style .mod specifications carrying a `hetagent_block` with the
Sequence-Space Jacobian (SSJ) engine from Auclert, Bardóczy, Rognlie & Straub
(2021, Econometrica).

Key Features
------------
1. Parses `hetagent_block; ... end;` in .mod files and links microeconomic
   household decision blocks with aggregate DSGE equilibrium conditions.
2. Computes microeconomic stationary distributions D*(a) and consumption Jacobians
   (J_C_r, J_C_Y) via the Fake-News Algorithm.
3. Solves coupled general equilibrium MIT transition paths in sequence space
   without curse-of-dimensionality state space explosion.
4. Provides HANKResult with transition paths, distribution diagnostics, summary
   tables, and publication-ready matplotlib figures.
"""
from __future__ import annotations

from pathlib import Path
from typing import Any, Mapping, Sequence

import numpy as np
import pandas as pd

from puremacro.dsge._parser import parse_mod_to_dag, ParsedModelDAG
from puremacro.dsge._results import HANKResult
from puremacro.models.hank_sequence_space import (
    solve_hank_sequence_space,
    solve_two_asset_hank_sequence_space,
    solve_nonlinear_transition,
    fake_news_algorithm,
    _solve_household_block,
    _solve_two_asset_household_block,
    _consumption_jacobians,
    _two_asset_jacobians,
    _ge_matrices,
    _local_mpc,
    _transaction_cost,
    TwoAssetSequenceSpaceHANKResult,
)


class HANKModel:
    """Coupled Heterogeneous-Agent DSGE Model representation."""

    def __init__(
        self,
        dag: ParsedModelDAG,
        params: dict[str, float] | None = None,
    ) -> None:
        self.dag = dag
        self.hetagent_config = dict(dag.hetagent_block or {})
        
        # Merge declared parameters with explicit overrides
        p = dict(dag.parameter_values)
        if params:
            p.update(params)
        self.params = p

        self.variables = list(dag.variables)
        self.shocks = list(dag.shocks)

        # Determine 1-asset vs 2-asset configuration
        assets_val = self.hetagent_config.get("assets", 1)
        model_type = str(self.hetagent_config.get("model", "")).lower()
        self.is_two_asset = (
            int(float(assets_val)) == 2
            or "two_asset" in model_type
            or "2_asset" in model_type
        )

        # Baseline calibration parameters
        self.beta = float(self.params.get("beta", 0.985))
        self.gamma = float(self.params.get("gamma", 1.0))
        self.r_ss = float(self.params.get("r_ss", self.params.get("r_b_ss", 0.01)))
        self.r_b_ss = float(self.params.get("r_b_ss", self.r_ss))
        self.r_a_ss = float(self.params.get("r_a_ss", 0.03))
        self.phi_pi = float(self.params.get("phi_pi", 1.5))
        self.kappa = float(self.params.get("kappa", 0.1))
        self.chi_0 = float(self.hetagent_config.get("chi_0", self.params.get("chi_0", 0.25)))
        self.chi_1 = float(self.hetagent_config.get("chi_1", self.params.get("chi_1", 1.0)))

        # Solve stationary distribution on initialization
        self.steady_state: dict[str, float] = {}
        self.asset_grid: np.ndarray = np.array([])
        self.liquid_asset_grid: np.ndarray | None = None
        self.asset_distribution: np.ndarray = np.array([])
        self.marginal_distribution_b: np.ndarray | None = None
        self.joint_distribution: np.ndarray | None = None
        self.deposit_distribution: np.ndarray | None = None
        self.mpc_distribution: np.ndarray | None = None
        self._hh_block = None
        self._solve_steady_state()

    def _solve_steady_state(self) -> None:
        """Solve the microeconomic stationary distribution and aggregate steady state."""
        if self.is_two_asset:
            n_a = int(self.hetagent_config.get("n_a", 25))
            n_b = int(self.hetagent_config.get("n_b", 25))
            a_max = float(self.hetagent_config.get("a_max", 30.0))
            b_max = float(self.hetagent_config.get("b_max", 15.0))
            b_min = float(self.hetagent_config.get("b_min", 0.0))

            hh = _solve_two_asset_household_block(
                beta=self.beta,
                gamma=self.gamma,
                r_b_ss=self.r_b_ss,
                r_a_ss=self.r_a_ss,
                w_ss=1.0,
                n_a=n_a,
                n_b=n_b,
                a_max=a_max,
                b_max=b_max,
                b_min=b_min,
                chi_0=self.chi_0,
                chi_1=self.chi_1,
            )
            self._hh_block = hh
            self.asset_grid = hh.a_grid.copy()
            self.liquid_asset_grid = hh.b_grid.copy()
            self.asset_distribution = hh.marginal_distribution_a.copy()
            self.marginal_distribution_b = hh.marginal_distribution_b.copy()
            self.joint_distribution = hh.joint_distribution.copy()
            self.deposit_distribution = hh.deposit_distribution.copy()

            # Local MPC across wealth grid
            mpc_a = np.clip(1.0 - (hh.a_grid / (hh.a_grid[-1] + 1e-4)), 0.05, 0.95)
            self.mpc_distribution = mpc_a

            c_ss = float(hh.C_ss)
            self.steady_state = {
                "Y": c_ss,
                "C": c_ss,
                "D": float(hh.D_flow_ss),
                "r_b": self.r_b_ss,
                "r_a": self.r_a_ss,
                "r": self.r_b_ss,
                "pi": 0.0,
                "i": self.r_b_ss,
                "w": 1.0,
                "A": float(hh.A_ss),
                "B": float(hh.B_ss),
            }
            for v in self.variables:
                if v not in self.steady_state:
                    self.steady_state[v] = 0.0
            return

        n_a = int(self.hetagent_config.get("n_a", 100))
        a_max = float(self.hetagent_config.get("a_max", 50.0))

        hh = _solve_household_block(
            beta=self.beta,
            gamma=self.gamma,
            r_ss=self.r_ss,
            n_a=n_a,
            a_max=a_max,
        )
        self._hh_block = hh
        self.asset_grid = hh.a_grid.copy()
        self.asset_distribution = np.sum(hh.D_ss, axis=1)
        mpc_mat = _local_mpc(hh.c_ss, hh.a_grid, hh.r_ss)
        weight = np.sum(hh.D_ss, axis=1)
        mpc_cond = np.zeros(n_a)
        mask = weight > 1e-12
        mpc_cond[mask] = np.sum(hh.D_ss[mask] * mpc_mat[mask], axis=1) / weight[mask]
        mpc_cond[~mask] = np.mean(mpc_mat[~mask], axis=1)
        self.mpc_distribution = mpc_cond

        c_ss = float(hh.C_ss)
        y_ss = c_ss  # Market clearing Y = C in baseline
        self.steady_state = {
            "Y": y_ss,
            "C": c_ss,
            "r": self.r_ss,
            "pi": 0.0,
            "i": self.r_ss,
            "w": 1.0,
            "B": float(hh.B),
        }
        # Populate any other variables from .mod file with steady state
        for v in self.variables:
            if v not in self.steady_state:
                self.steady_state[v] = 0.0

    def compute_jacobians(self, T: int = 300) -> dict[str, np.ndarray]:
        """Compute sequence-space Jacobians via Fake-News Algorithm."""
        if self._hh_block is None:
            self._solve_steady_state()
        if self.is_two_asset:
            return _two_asset_jacobians(self._hh_block, T=T)
        J_C_r, J_C_Y = _consumption_jacobians(self._hh_block, T=T)
        return {
            "J_C_r": J_C_r,
            "J_C_Y": J_C_Y,
        }

    def simulate(
        self,
        shock: str = "eps_m",
        magnitude: float = -0.0025,
        rho: float = 0.5,
        horizon: int = 40,
        nonlinear: bool = False,
    ) -> HANKResult:
        """Simulate general equilibrium transition dynamics for an MIT shock."""
        shock_path = magnitude * (rho ** np.arange(horizon))

        if self.is_two_asset:
            res_2a = solve_two_asset_hank_sequence_space(
                T=horizon,
                beta=self.beta,
                gamma=self.gamma,
                r_b_ss=self.r_b_ss,
                r_a_ss=self.r_a_ss,
                phi_pi=self.phi_pi,
                kappa=self.kappa,
                chi_0=self.chi_0,
                chi_1=self.chi_1,
                shock_magnitude=magnitude,
                shock_rho=rho,
                n_a=int(self.hetagent_config.get("n_a", 25)),
                n_b=int(self.hetagent_config.get("n_b", 25)),
                a_max=float(self.hetagent_config.get("a_max", 30.0)),
                b_max=float(self.hetagent_config.get("b_max", 15.0)),
                b_min=float(self.hetagent_config.get("b_min", 0.0)),
            )
            dY = res_2a.irf_output[:horizon]
            dC = res_2a.irf_consumption[:horizon]
            dD = res_2a.irf_deposit[:horizon]
            dr_b = res_2a.irf_rate_b[:horizon]
            dr_a = res_2a.irf_rate_a[:horizon]
            dpi = res_2a.irf_inflation[:horizon]
            di = self.phi_pi * dpi + shock_path

            path_data: dict[str, np.ndarray] = {
                "Y": dY,
                "C": dC,
                "D": dD,
                "r_b": dr_b,
                "r_a": dr_a,
                "r": dr_b,
                "pi": dpi,
                "i": di,
            }
            if shock in self.shocks:
                path_data[shock] = shock_path
            for v in self.variables:
                if v not in path_data:
                    path_data[v] = np.zeros(horizon)

            df_paths = pd.DataFrame(path_data)
            J_dict = {
                "J_C_rb": res_2a.jacobian_c_rb,
                "J_C_ra": res_2a.jacobian_c_ra,
                "J_C_Y": res_2a.jacobian_c_y,
                "J_D_rb": res_2a.jacobian_d_rb,
                "J_D_ra": res_2a.jacobian_d_ra,
                "J_D_Y": res_2a.jacobian_d_y,
                "J_C_r": res_2a.jacobian_c_rb,
            }

            return HANKResult(
                steady_state=self.steady_state,
                transition_paths=df_paths,
                jacobians=J_dict,
                asset_distribution=self.asset_distribution,
                asset_grid=self.asset_grid,
                liquid_asset_grid=self.liquid_asset_grid,
                joint_distribution=self.joint_distribution,
                marginal_distribution_b=self.marginal_distribution_b,
                deposit_distribution=self.deposit_distribution,
                mpc_distribution=self.mpc_distribution,
                shock_name=shock,
                horizon=horizon,
                model_name=self.hetagent_config.get("model", "hank_two_asset"),
                converged=res_2a.converged,
            )

        if nonlinear:
            res_ssj = solve_nonlinear_transition(
                beta=self.beta,
                gamma=self.gamma,
                r_ss=self.r_ss,
                phi_pi=self.phi_pi,
                kappa=self.kappa,
                horizon=horizon,
                shock_seq=shock_path,
                n_a=int(self.hetagent_config.get("n_a", 50)),
                a_max=float(self.hetagent_config.get("a_max", 30.0)),
            )
            dY = res_ssj.irf_output_nonlinear[:horizon]
            dC = res_ssj.irf_consumption_nonlinear[:horizon]
            dr = res_ssj.irf_rate_nonlinear[:horizon]
            dpi = res_ssj.irf_inflation_nonlinear[:horizon]
            converged = res_ssj.converged
            J_dict = {
                "J_C_r": res_ssj.jacobian_c_r,
                "J_C_Y": res_ssj.jacobian_c_y,
            }
        else:
            res_ssj = solve_hank_sequence_space(
                T=horizon,
                beta=self.beta,
                gamma=self.gamma,
                r_ss=self.r_ss,
                phi_pi=self.phi_pi,
                kappa=self.kappa,
                shock_magnitude=magnitude,
                shock_rho=rho,
                n_a=int(self.hetagent_config.get("n_a", 50)),
                a_max=float(self.hetagent_config.get("a_max", 30.0)),
            )
            dY = res_ssj.irf_output[:horizon]
            dC = res_ssj.irf_consumption[:horizon]
            dr = res_ssj.irf_rate[:horizon]
            dpi = res_ssj.irf_inflation[:horizon]
            converged = True
            J_dict = {
                "J_C_r": res_ssj.jacobian_c_r,
                "J_C_Y": res_ssj.jacobian_c_y,
            }

        # Build paths DataFrame matching variables in the .mod file
        di = self.phi_pi * dpi + shock_path

        path_data: dict[str, np.ndarray] = {
            "Y": dY,
            "C": dC,
            "r": dr,
            "pi": dpi,
            "i": di,
        }
        if shock in self.shocks:
            path_data[shock] = shock_path

        # Add remaining declared variables
        for v in self.variables:
            if v not in path_data:
                if v == "w":
                    path_data[v] = np.zeros(horizon)
                elif v == "K":
                    path_data[v] = np.zeros(horizon)
                elif v == "I":
                    path_data[v] = dY - dC
                else:
                    path_data[v] = np.zeros(horizon)

        df_paths = pd.DataFrame(path_data)

        return HANKResult(
            steady_state=self.steady_state,
            transition_paths=df_paths,
            jacobians=J_dict,
            asset_distribution=self.asset_distribution,
            asset_grid=self.asset_grid,
            mpc_distribution=self.mpc_distribution,
            shock_name=shock,
            horizon=horizon,
            model_name=self.hetagent_config.get("model", "hank_sequence_space"),
            converged=converged,
        )


def load_hank_mod(source: str | Path, **param_overrides: Any) -> HANKModel:
    """Load and parse a Dynare .mod file containing a hetagent_block."""
    if isinstance(source, Path) or (isinstance(source, str) and "\n" not in source and Path(source).exists()):
        text = Path(source).read_text(encoding="utf-8")
    else:
        text = str(source)

    dag = parse_mod_to_dag(text)
    if dag.hetagent_block is None:
        raise ValueError(
            "load_hank_mod: the supplied .mod file does not contain a hetagent_block; "
            "use load_mod for standard representative-agent DSGE models."
        )

    return HANKModel(dag, params=param_overrides)


def solve_hank_bridge(
    source: str | Path,
    shock: str = "eps_m",
    magnitude: float = -0.0025,
    rho: float = 0.5,
    horizon: int = 40,
    nonlinear: bool = False,
    **param_overrides: Any,
) -> HANKResult:
    """High-level interface to solve a HANK .mod model and return transition dynamics."""
    model = load_hank_mod(source, **param_overrides)
    return model.simulate(
        shock=shock,
        magnitude=magnitude,
        rho=rho,
        horizon=horizon,
        nonlinear=nonlinear,
    )


__all__ = [
    "HANKModel",
    "HANKResult",
    "load_hank_mod",
    "solve_hank_bridge",
]
