"""Gertler-Karadi (2011) DSGE model with financial frictions and banking moral hazard.

Implementation of the canonical macro-finance banking friction framework:
    Gertler, M., & Karadi, P. (2011). "A model of unconventional monetary policy".
    Journal of Monetary Economics, 58(1), 17-34.

And occasionally binding leverage / credit policy constraints via OccBin:
    Guerrieri, L., & Iacoviello, M. (2015). "OccBin: A toolkit for solving dynamic
    models with occasionally binding constraints easily".
    Journal of Monetary Economics, 70, 22-38.

Features
--------
1. Endogenous Financial Friction:
   Bankers face an agency problem / moral hazard where they can divert a fraction
   ``lambda_b`` of bank assets. Depositors enforce an incentive compatibility
   constraint binding bank leverage:
       phi_t = eta_t / (lambda_b - nu_t)
   where eta_t is the marginal continuation value of bank net worth, and nu_t
   is the marginal value of expanding bank assets.

2. Macroeconomic Amplification & Capital Quality Shocks:
   A negative shock to capital quality xi_t reduces physical capital productivity,
   depresses the asset price Q_t, causes massive bank capital losses, shrinks
   bank net worth N_t by the leverage multiplier (phi ~ 4x), spikes the credit
   spread E_t[R_{k,t+1} - R_{t+1}], and induces a persistent slump in investment
   and output.

3. Dual Solvers:
   - Klein QZ linear solver (puremacro.dsge.klein / build_dynare) for unconstrained
     first-order perturbations.
   - OccBin piecewise linear backward recursion (puremacro.dsge.occbin) for
     occasionally binding credit policy interventions or regulatory leverage caps.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Mapping, Sequence, cast

import warnings
import numpy as np
import pandas as pd
import scipy.optimize

from puremacro.dsge.build import LinearModel
from puremacro.dsge.dynare import build_dynare
from puremacro.dsge.occbin import OccBinConstraint, OccBinResult, solve_occbin


# ---------------------------------------------------------------------------
# Canonical Calibration (Gertler & Karadi 2011 Table 1)
# ---------------------------------------------------------------------------

GK2011_PARAMS: dict[str, float] = {
    # Households
    "beta": 0.99,         # Subjective discount factor
    "sigma": 1.0,         # Intertemporal elasticity of substitution (log utility)
    "h": 0.815,           # Habit persistence parameter
    "varphi": 0.276,      # Inverse Frisch elasticity of labor supply
    "epsilon": 4.167,     # Elasticity of substitution across retail goods (markup ~ 1.316)

    # Intermediate goods & capital production
    "alpha": 0.33,        # Effective capital share
    "delta": 0.025,       # Steady-state physical depreciation rate (quarterly 2.5%)
    "eta_i": 1.728,       # Investment adjustment cost parameter
    "zeta": 7.2,          # Elasticity of marginal depreciation wrt capacity utilization

    # Financial intermediaries (banks)
    "theta_b": 0.972,     # Banker quarterly survival probability
    "lambda_b": 0.381,    # Divertable fraction of assets (moral hazard parameter)
    "omega_b": 0.002,     # Start-up transfer fraction to newly entering bankers

    # Retail price stickiness & monetary policy
    "gamma": 0.779,       # Calvo price stickiness probability
    "gamma_p": 0.241,     # Price indexation parameter
    "kappa_pi": 1.5,      # Taylor rule inflation response
    "kappa_y": 0.125,     # Taylor rule output response
    "rho_i": 0.8,         # Taylor rule interest rate smoothing
    "g_share": 0.20,      # Government spending share of steady-state output

    # Shock persistences
    "rho_xi": 0.66,       # Capital quality shock persistence
    "rho_a": 0.95,        # Total factor productivity (TFP) persistence

    # Unconventional credit policy
    "nu_g": 10.0,         # Central bank credit policy intervention intensity
}


# ---------------------------------------------------------------------------
# Steady-State Solution
# ---------------------------------------------------------------------------

def solve_steady_state(params: Mapping[str, float] | None = None) -> dict[str, float]:
    """Solve the exact deterministic steady state of the Gertler-Karadi (2011) model.

    Parameters
    ----------
    params : Mapping[str, float], optional
        Parameter overrides. Defaults to ``GK2011_PARAMS``.

    Returns
    -------
    dict[str, float]
        Dictionary of all steady-state variables, asset prices, banking multipliers,
        and calibrated constants.
    """
    p = dict(GK2011_PARAMS)
    if params is not None:
        p.update(params)

    if not (0.0 < float(p["beta"]) < 1.0):
        raise ValueError(f"beta must lie in (0, 1) for a well-defined steady state, got {p['beta']!r}")
    if not (0.0 < float(p["theta_b"]) < 1.0):
        raise ValueError(f"theta_b (banker survival probability) must lie in (0, 1), got {p['theta_b']!r}")
    beta = p["beta"]
    R = 1.0 / beta
    theta_b = p["theta_b"]
    lambda_b = p["lambda_b"]
    omega_b = p["omega_b"]
    alpha = p["alpha"]
    delta = p["delta"]
    epsilon = p["epsilon"]
    h = p["h"]
    varphi = p["varphi"]
    g_share = p["g_share"]
    gamma = p["gamma"]
    zeta = p["zeta"]

    # 1. Solve endogenous credit spread (prem) and leverage (phi)
    def res_prem(prem_val: float) -> float:
        phi_nw = (1.0 - theta_b * R) / (theta_b * prem_val + omega_b)
        denom = 1.0 - (theta_b / (1.0 - theta_b)) * beta * prem_val * phi_nw
        if denom <= 0.0:
            return 1e6
        eta_val = 1.0 / denom
        nu_val = beta * eta_val * prem_val
        phi_ic = eta_val / (lambda_b - nu_val)
        return phi_nw - phi_ic

    try:
        sol_prem = scipy.optimize.brentq(res_prem, 1e-6, 0.05)
    except ValueError as exc:
        raise ValueError(
            "solve_steady_state: no steady-state credit spread in (0, 5%) for these "
            "parameters (check beta < 1, 0 < theta_b < 1, lambda_b, omega_b)"
        ) from exc
    prem_ss = float(sol_prem)
    Rk_ss = R + prem_ss
    phi_ss = (1.0 - theta_b * R) / (theta_b * prem_ss + omega_b)
    eta_ss = 1.0 / (1.0 - (theta_b / (1.0 - theta_b)) * beta * prem_ss * phi_ss)
    nu_ss = beta * eta_ss * prem_ss
    Omega_ss = eta_ss

    # 2. Intermediate goods & factor markets
    Pm_ss = (epsilon - 1.0) / epsilon
    Z_ss = Rk_ss - (1.0 - delta)

    kl_ratio = (Pm_ss * alpha / Z_ss) ** (1.0 / (1.0 - alpha))
    yl_ratio = kl_ratio ** alpha
    wl = Pm_ss * (1.0 - alpha) * yl_ratio

    # Steady-state labor hours normalized to 1/3 (canonical RBC convention)
    L_ss = 1.0 / 3.0
    K_ss = kl_ratio * L_ss
    Y_ss = yl_ratio * L_ss
    I_ss = delta * K_ss
    G_ss = g_share * Y_ss
    C_ss = Y_ss - I_ss - G_ss
    w_ss = wl

    # 3. Household marginal utility & labor preference calibration
    rho_c_ss = (1.0 - beta * h) / ((1.0 - h) * C_ss)
    chi_ss = rho_c_ss * w_ss / (L_ss ** varphi)

    # 4. Financial intermediaries (banks)
    Q_ss = 1.0
    S_ss = K_ss
    N_ss = (Q_ss * S_ss) / phi_ss
    B_ss = (Q_ss * S_ss) - N_ss
    Ne_ss = theta_b * ((Rk_ss - R) * phi_ss + R) * N_ss
    Nn_ss = omega_b * Q_ss * S_ss

    # 5. Prices, utilization, policy
    U_ss = 1.0
    Pi_ss = 1.0
    Rn_ss = R * Pi_ss
    xi_ss = 1.0
    a_ss = 1.0
    psi_ss = 0.0

    # Calvo Phillips curve slope
    kappa_p = ((1.0 - gamma) * (1.0 - beta * gamma) / gamma) * (
        (1.0 - alpha) / (1.0 - alpha + alpha * epsilon)
    )

    # Utilization parameters
    delta_0 = delta - Z_ss / (1.0 + zeta)
    b_util = Z_ss

    return {
        # Core macro quantities
        "Y": Y_ss,
        "C": C_ss,
        "I": I_ss,
        "K": K_ss,
        "L": L_ss,
        "G": G_ss,
        "w": w_ss,
        "rho_c": rho_c_ss,
        "Q": Q_ss,
        "R": R,
        "Rk": Rk_ss,
        "prem": prem_ss,
        "spread_ann": prem_ss * 40000.0,
        "Pi": Pi_ss,
        "Rn": Rn_ss,
        "xi": xi_ss,
        "a": a_ss,
        "psi": psi_ss,
        "Pm": Pm_ss,
        "Z": Z_ss,
        "U": U_ss,
        # Banking sector
        "N": N_ss,
        "Ne": Ne_ss,
        "Nn": Nn_ss,
        "B": B_ss,
        "phi": phi_ss,
        "nu": nu_ss,
        "eta": eta_ss,
        "Omega": Omega_ss,
        # Derived structural constants
        "chi": chi_ss,
        "kappa": kappa_p,
        "delta_0": delta_0,
        "b_util": b_util,
        "Pm_ss": Pm_ss,
        "Rn_ss": Rn_ss,
        "Y_ss": Y_ss,
        "prem_ss": prem_ss,
        "phi_ss": phi_ss,
    }


# ---------------------------------------------------------------------------
# Equilibrium Equations (Non-Linear Lead-Lag Form for build_dynare)
# ---------------------------------------------------------------------------

GK_VARIABLES = [
    "Y", "C", "I", "K", "L", "w", "rho_c", "Q", "R", "Rk", "prem",
    "N", "Ne", "Nn", "phi", "nu", "eta", "Omega", "Pm", "Z", "U", "Pi", "Rn", "xi", "a", "psi"
]

GK_SHOCKS = ["eps_xi", "eps_a", "eps_r"]


def _gk_equations_ref(lead, curr, lag, shocks_v, p):
    """Equilibrium conditions in reference regime (no credit policy, active friction)."""
    delta_U = p.delta_0 + (p.b_util / (1.0 + p.zeta)) * (curr.U ** (1.0 + p.zeta))
    inv_cost = 0.5 * p.eta_i * (curr.I / lag.I - 1.0) ** 2
    inv_deriv = p.eta_i * (curr.I / lag.I - 1.0) * (curr.I / lag.I)
    inv_lead_deriv = (
        p.beta * (lead.rho_c / curr.rho_c) * p.eta_i * (lead.I / curr.I - 1.0) * ((lead.I / curr.I) ** 2)
    )

    return [
        # 1. Marginal utility of consumption
        curr.rho_c - (1.0 / (curr.C - p.h * lag.C) - p.beta * p.h / (lead.C - p.h * curr.C)),
        # 2. Euler equation for deposits
        p.beta * (lead.rho_c / curr.rho_c) * (curr.Rn / lead.Pi) - 1.0,
        # 3. Real risk-free rate definition
        curr.R - (lag.Rn / curr.Pi),
        # 4. Labor supply FOC
        curr.rho_c * curr.w - p.chi * (curr.L ** p.varphi),
        # 5. Production function
        curr.Y - curr.a * ((curr.U * curr.xi * lag.K) ** p.alpha) * (curr.L ** (1.0 - p.alpha)),
        # 6. Intermediate goods wage / labor demand
        curr.w - curr.Pm * (1.0 - p.alpha) * curr.Y / curr.L,
        # 7. Rental rate of capital
        curr.Z - curr.Pm * p.alpha * curr.Y / (curr.U * curr.xi * lag.K),
        # 8. Capacity utilization FOC
        curr.Z - p.b_util * (curr.U ** p.zeta),
        # 9. Capital accumulation
        curr.K - (curr.xi * lag.K * (1.0 - delta_U) + curr.I),
        # 10. Gross return on capital
        curr.Rk - (curr.Z * curr.U - delta_U + curr.Q) * curr.xi / lag.Q,
        # 11. Expected credit spread E_t[R_{k,t+1} - R_{t+1}]
        curr.prem - (lead.Rk - lead.R),
        # 12. Tobin Q / Investment FOC
        curr.Q - (1.0 + inv_cost + inv_deriv - inv_lead_deriv),
        # 13. Banker asset expansion value multiplier (nu)
        curr.nu - p.beta * (lead.rho_c / curr.rho_c) * lead.Omega * (lead.Rk - lead.R),
        # 14. Banker net worth value multiplier (eta)
        curr.eta - p.beta * (lead.rho_c / curr.rho_c) * lead.Omega * lead.R,
        # 15. Banker continuation multiplier (Omega)
        curr.Omega - (1.0 - p.theta_b + p.theta_b * (curr.eta + curr.nu * curr.phi)),
        # 16. Moral hazard incentive constraint (endogenous leverage)
        curr.phi * (p.lambda_b - curr.nu) - curr.eta,
        # 17. Bank balance sheet with public credit policy psi
        (1.0 - curr.psi) * curr.Q * curr.K - curr.phi * curr.N,
        # 18. Existing banker net worth
        curr.Ne - p.theta_b * ((curr.Rk - curr.R) * lag.phi + curr.R) * lag.N,
        # 19. New banker start-up transfer
        curr.Nn - p.omega_b * curr.Q * lag.K,
        # 20. Total bank net worth
        curr.N - (curr.Ne + curr.Nn),
        # 21. Resource constraint
        curr.Y - curr.C - curr.I - (p.g_share * p.Y_ss),
        # 22. New Keynesian Phillips Curve with indexation
        (
            ((curr.Pi - 1.0) - p.gamma_p * (lag.Pi - 1.0))
            - p.beta * ((lead.Pi - 1.0) - p.gamma_p * (curr.Pi - 1.0))
            - p.kappa * (curr.Pm - p.Pm_ss)
        ),
        # 23. Taylor rule
        (
            curr.Rn
            - (
                (lag.Rn ** p.rho_i)
                * (
                    (
                        p.Rn_ss
                        * (curr.Pi ** p.kappa_pi)
                        * ((curr.Y / p.Y_ss) ** p.kappa_y)
                    )
                    ** (1.0 - p.rho_i)
                )
                * np.exp(shocks_v.eps_r)
            )
        ),
        # 24. Capital quality shock process
        np.log(curr.xi) - p.rho_xi * np.log(lag.xi) - shocks_v.eps_xi,
        # 25. TFP shock process
        np.log(curr.a) - p.rho_a * np.log(lag.a) - shocks_v.eps_a,
        # 26. In reference regime: credit policy is inactive (psi = 0)
        curr.psi,
    ]


def _gk_equations_cons_credit_policy(lead, curr, lag, shocks_v, p):
    """Equilibrium conditions in constrained regime under credit policy intervention."""
    eqs = _gk_equations_ref(lead, curr, lag, shocks_v, p)
    # Replace equation 26 (index 25) with active credit policy rule
    eqs[25] = curr.psi - p.nu_g * (curr.prem - p.prem_ss)
    return eqs


def _gk_equations_cons_leverage_cap(lead, curr, lag, shocks_v, p):
    """Equilibrium conditions in constrained regime under hard leverage cap."""
    eqs = _gk_equations_ref(lead, curr, lag, shocks_v, p)
    # Replace equation 16 (index 15) with hard leverage cap
    eqs[15] = curr.phi - p.phi_max
    return eqs


# ---------------------------------------------------------------------------
# Model Builder
# ---------------------------------------------------------------------------

def build_gertler_karadi_model(
    params: Mapping[str, float] | None = None,
    regime: str = "reference",
    constraint_type: str = "credit_policy",
    check_steady_state: bool = True,
) -> LinearModel:
    """Compile the Gertler-Karadi (2011) DSGE model into a solved LinearModel.

    Parameters
    ----------
    params : Mapping[str, float], optional
        Parameter overrides.
    regime : {'reference', 'constrained'}, default 'reference'
        Which regime to compile.
    constraint_type : {'credit_policy', 'leverage_cap'}, default 'credit_policy'
        Regime switch specification for the constrained regime.
    check_steady_state : bool, default True
        Whether to verify steady-state residual check.

    Returns
    -------
    LinearModel
        Solved DSGE model equipped with Klein decision rules and QZ matrices.
    """
    p_dict = dict(GK2011_PARAMS)
    if params is not None:
        p_dict.update(params)

    ss = solve_steady_state(p_dict)
    full_params = dict(p_dict)
    full_params.update({
        "chi": ss["chi"],
        "kappa": ss["kappa"],
        "delta_0": ss["delta_0"],
        "b_util": ss["b_util"],
        "Pm_ss": ss["Pm_ss"],
        "Rn_ss": ss["Rn_ss"],
        "Y_ss": ss["Y_ss"],
        "prem_ss": ss["prem_ss"],
        "phi_ss": ss["phi_ss"],
        "phi_max": full_params.get("phi_max", ss["phi_ss"]),
    })

    if regime == "reference":
        eq_fn = _gk_equations_ref
    elif regime == "constrained":
        if constraint_type == "credit_policy":
            eq_fn = _gk_equations_cons_credit_policy
        elif constraint_type == "leverage_cap":
            eq_fn = _gk_equations_cons_leverage_cap
        else:
            raise ValueError(f"unknown constraint_type {constraint_type!r}; expected 'credit_policy' or 'leverage_cap'")
    else:
        raise ValueError(f"unknown regime {regime!r}; expected 'reference' or 'constrained'")

    return cast(LinearModel, build_dynare(
        eq_fn,
        variables=GK_VARIABLES,
        shocks=GK_SHOCKS,
        params=full_params,
        steady_state=ss,
        # Only the reference regime must satisfy Blanchard-Kahn; an OccBin
        # alternative regime is solved by backward recursion from the reference
        # rule and is allowed to be indeterminate/unstable on its own.
        strict=(regime == "reference"),
        check_steady_state=check_steady_state,
    ))


# ---------------------------------------------------------------------------
# Result Container
# ---------------------------------------------------------------------------

@dataclass
class GertlerKaradiResult:
    """Result container for Gertler-Karadi (2011) DSGE simulation.

    Attributes
    ----------
    irf : pd.DataFrame
        Impulse response trajectories (deviations from steady state) for all model variables.
    variables : list[str]
        List of endogenous variable names.
    steady_state : dict[str, float]
        Deterministic steady-state values for all model variables.
    solver_method : str
        Solver method used ('klein' or 'occbin').
    shock_type : str, default 'capital_quality'
        Type of shock simulated ('capital_quality', 'tfp', or 'monetary').
    shock_size : float, default -0.05
        Size of the shock innovation.
    binding_periods : int, default 0
        Number of periods the occasionally binding constraint binds (for OccBin).
    regimes : list[int], default factory list
        Sequence of regime indicators across the simulation horizon (0=ref, 1=cons).
    model : LinearModel | None, optional
        Underlying unconstrained reference model.
    occbin_result : OccBinResult | None, optional
        Underlying OccBinResult if solved via OccBin.
    params : dict[str, float], default factory dict
        Model parameters used.
    """

    irf: pd.DataFrame
    variables: list[str]
    steady_state: dict[str, float]
    solver_method: str
    shock_type: str = "capital_quality"
    shock_size: float = -0.05
    binding_periods: int = 0
    regimes: list[int] = field(default_factory=list)
    converged: bool = True
    model: Any | None = None
    occbin_result: Any | None = None
    params: dict[str, float] = field(default_factory=dict)

    def to_frame(self) -> pd.DataFrame:
        """Return simulated trajectory as a pandas DataFrame."""
        return self.irf.copy()

    def __getitem__(self, key: str) -> pd.Series | Any:
        """Allow subscript access to simulated variables."""
        if key in self.irf.columns:
            return self.irf[key]
        if hasattr(self, key):
            return getattr(self, key)
        raise KeyError(f"GertlerKaradiResult has no variable or attribute {key!r}")

    def summary(self) -> str:
        """Render a formatted text summary of calibration and simulation results."""
        horizon = len(self.irf)
        status = f"Piecewise-linear (OccBin), {self.binding_periods} binding period(s)" if self.solver_method == "occbin" else "Linear (Klein QZ)"
        if not self.converged:
            status += " -- WARNING: regime iteration did NOT converge; path is unreliable"
        
        lines = [
            "GERTLER-KARADI (2011) FINANCIAL FRICTIONS DSGE REPORT",
            "=" * 78,
            f"Solver method      : {self.solver_method.upper()} ({status})",
            f"Shock simulated    : {self.shock_type} (size = {self.shock_size:+.4f})",
            f"Simulation horizon : {horizon} quarters",
            "-" * 78,
            "STEADY-STATE CALIBRATION BENCHMARKS",
            "-" * 78,
            f"  Bank Leverage (phi = Q*S/N)  : {self.steady_state.get('phi', 0.0):.4f} (target ~ 4.0)",
            f"  Annualized Credit Spread      : {self.steady_state.get('spread_ann', 0.0):.2f} bps (target ~ 100 bps)",
            f"  Risk-free Gross Return (R)    : {self.steady_state.get('R', 0.0):.6f} (quarterly)",
            f"  Output (Y) / Capital (K)      : {self.steady_state.get('Y', 0.0):.4f} / {self.steady_state.get('K', 0.0):.4f}",
            f"  Bank Net Worth (N)            : {self.steady_state.get('N', 0.0):.4f}",
            "-" * 78,
            "TRAJECTORY SUMMARY STATISTICS (DEVIATIONS FROM STEADY STATE)",
            "-" * 78,
        ]

        key_vars = [v for v in ["Y", "I", "C", "N", "prem", "phi", "Q", "Rn", "Pi", "psi"] if v in self.irf.columns]
        sub_df = self.irf[key_vars]
        stats_df = pd.DataFrame(
            {
                "Impact (t=0)": sub_df.iloc[0],
                "Min": sub_df.min(),
                "Max": sub_df.max(),
                "Mean": sub_df.mean(),
                "Final (t=H)": sub_df.iloc[-1],
            }
        )
        lines.append(stats_df.round(6).to_string())
        lines.append("=" * 78)
        return "\n".join(lines)

    def to_markdown(self, **kwargs) -> str:
        """Export simulated trajectory to Markdown table."""
        from puremacro.reports import _df_to_markdown

        return _df_to_markdown(self.to_frame(), **kwargs)

    def to_latex(self, **kwargs) -> str:
        """Export simulated trajectory to LaTeX tabular."""
        from puremacro.reports import _df_to_latex

        return _df_to_latex(self.to_frame(), **kwargs)

    def to_typst(self, **kwargs) -> str:
        """Export simulated trajectory to Typst table."""
        from puremacro.reports import _df_to_typst

        return _df_to_typst(self.to_frame(), **kwargs)

    def plot(
        self,
        variables: Sequence[str] | None = None,
        style: str = "publication",
        figsize: tuple[float, float] | None = None,
    ):
        """Plot multi-panel impulse responses with highlighted binding regimes.

        Parameters
        ----------
        variables : Sequence[str], optional
            Variables to include in the multi-panel plot. Defaults to
            ``['Y', 'I', 'N', 'prem', 'Rn', 'phi']``.
        style : {'publication', 'default'}, default 'publication'
            Plot style. If 'publication', applies black-and-white publication
            styling from ``puremacro.plotting.bw_style``.
        figsize : tuple of float, optional
            Figure size in inches.

        Returns
        -------
        matplotlib.figure.Figure
            The resulting figure.
        """
        import matplotlib.pyplot as plt

        if variables is None:
            default_vars = ["Y", "I", "N", "prem", "Rn", "phi"]
            variables = [v for v in default_vars if v in self.irf.columns]
        else:
            variables = [v for v in variables if v in self.irf.columns]

        n_vars = len(variables)
        if n_vars == 0:
            raise ValueError(
                "plot: none of the requested variables exist in the simulated path; "
                f"available: {list(self.irf.columns)}"
            )
        n_cols = min(3, n_vars)
        n_rows = (n_vars + n_cols - 1) // n_cols

        if figsize is None:
            figsize = (4.8 * n_cols, 3.2 * n_rows)

        if style == "publication":
            from puremacro.plotting.bw_style import apply_bw_style, bw_colors, bw_linestyles

            apply_bw_style()
            colors: list[str | None] = list(bw_colors(n_vars))
            styles: list[str] = list(bw_linestyles(n_vars))
        else:
            colors = [None] * n_vars
            styles = ["-"] * n_vars

        fig, axes = plt.subplots(n_rows, n_cols, figsize=figsize, squeeze=False)
        axes_flat = axes.flatten()

        horizon = len(self.irf)
        time_grid = np.arange(horizon)

        title_map = {
            "Y": "Output (Y)",
            "I": "Investment (I)",
            "N": "Bank Net Worth (N)",
            "prem": "Credit Spread (Rk - R)",
            "Rn": "Policy Rate (Rn)",
            "phi": "Bank Leverage (phi)",
            "Q": "Tobin's Q",
            "C": "Consumption (C)",
            "Pi": "Inflation (Pi)",
            "psi": "Credit Policy (psi)",
        }

        for i, var in enumerate(variables):
            ax = axes_flat[i]
            c = colors[i] if colors[i] is not None else "black"
            ls = styles[i] if styles[i] is not None else "-"
            ax.plot(time_grid, self.irf[var], label=var, color=c, linestyle=ls, linewidth=1.5)

            # Highlight binding regime periods if solved with OccBin
            if self.binding_periods > 0 and len(self.regimes) >= horizon:
                binding_mask = np.array(self.regimes[:horizon]) == 1
                if np.any(binding_mask):
                    diff = np.diff(np.pad(binding_mask.astype(int), (1, 1), "constant"))
                    starts = np.where(diff == 1)[0]
                    ends = np.where(diff == -1)[0] - 1
                    for s, e in zip(starts, ends):
                        ax.axvspan(s - 0.5, e + 0.5, color="0.85", alpha=0.3, label="Constrained" if i == 0 else None)

            ax.axhline(0.0, color="0.6", linestyle=":", linewidth=0.7)
            ax.set_title(title_map.get(var, var), fontsize=10, fontweight="bold")
            ax.set_xlabel("Quarter", fontsize=8)
            ax.grid(True, linestyle="--", alpha=0.3)

        for j in range(n_vars, len(axes_flat)):
            axes_flat[j].set_visible(False)

        fig.tight_layout()
        return fig


# ---------------------------------------------------------------------------
# Solver Entry Point
# ---------------------------------------------------------------------------

def solve_gertler_karadi(
    params: Mapping[str, float] | None = None,
    shock_type: str = "capital_quality",
    shock_size: float = -0.05,
    horizon: int = 40,
    method: str = "occbin",
    constraint_type: str = "credit_policy",
    threshold: float | None = None,
    max_iter: int = 50,
) -> GertlerKaradiResult:
    """Solve and simulate the Gertler-Karadi (2011) DSGE model.

    Parameters
    ----------
    params : Mapping[str, float], optional
        Model parameters. Defaults to canonical calibration (GK2011_PARAMS).
    shock_type : {'capital_quality', 'tfp', 'monetary'}, default 'capital_quality'
        Type of shock to simulate:
        - 'capital_quality': shock to capital quality xi_t (eps_xi).
        - 'tfp' or 'technology': shock to total factor productivity a_t (eps_a).
        - 'monetary' or 'policy': shock to the Taylor rule (eps_r).
    shock_size : float, default -0.05
        Magnitude of the one-time shock innovation at t=0.
    horizon : int, default 40
        Simulation horizon (number of quarters).
    method : {'occbin', 'klein'}, default 'occbin'
        Solution solver to use:
        - 'occbin': Piecewise-linear backward recursion over regimes.
        - 'klein': Linear first-order rational expectations perturbation via Klein QZ.
    constraint_type : {'credit_policy', 'leverage_cap'}, default 'credit_policy'
        Regime switch specification for OccBin:
        - 'credit_policy': Central bank credit intervention when credit spread
          exceeds threshold (default 100 bps = 0.0025 above steady state).
        - 'leverage_cap': Macroprudential ceiling on bank leverage.
    threshold : float, optional
        Numerical threshold for OccBin constraint. If None:
        - for 'credit_policy': defaults to 0.0025 (100 bps annualized).
        - for 'leverage_cap': defaults to 0.0 (capped at steady-state leverage).
    max_iter : int, default 50
        Maximum backward recursion iterations for OccBin.

    Returns
    -------
    GertlerKaradiResult
        Result container with impulse responses, steady-state dictionary,
        solver diagnostics, and visualization methods.
    """
    if int(horizon) != horizon or horizon < 1:
        raise ValueError(f"horizon must be a positive integer number of quarters, got {horizon!r}")
    horizon = int(horizon)
    p_dict = dict(GK2011_PARAMS)
    if params is not None:
        p_dict.update(params)

    ss = solve_steady_state(p_dict)
    ref_model = build_gertler_karadi_model(p_dict, regime="reference")

    # Map shock_type to shock index
    shock_map = {
        "capital_quality": "eps_xi",
        "xi": "eps_xi",
        "tfp": "eps_a",
        "technology": "eps_a",
        "monetary": "eps_r",
        "policy": "eps_r",
    }
    s_key = shock_type.lower()
    if s_key not in shock_map:
        raise ValueError(f"unknown shock_type {shock_type!r}; expected one of {list(shock_map.keys())}")
    shock_var = shock_map[s_key]
    shock_idx = GK_SHOCKS.index(shock_var)

    shock_seq = np.zeros((horizon, len(GK_SHOCKS)))
    shock_seq[0, shock_idx] = float(shock_size)

    method_clean = method.lower()
    if method_clean == "klein":
        # Linear simulation via Klein decision rules
        dr = ref_model.decision_rules()
        n_vars = len(GK_VARIABLES)
        P_0 = np.zeros((n_vars, n_vars))
        for s in ref_model.states:
            idx_s = GK_VARIABLES.index(s)
            P_0[:, idx_s] = dr.ghx[s].values

        sim_X = np.zeros((horizon, n_vars))
        sim_X[0] = dr.ghu.values @ shock_seq[0]
        for t in range(1, horizon):
            sim_X[t] = P_0 @ sim_X[t - 1]

        irf_df = pd.DataFrame(sim_X, columns=GK_VARIABLES)
        return GertlerKaradiResult(
            irf=irf_df,
            variables=GK_VARIABLES,
            steady_state=ss,
            solver_method="klein",
            shock_type=shock_type,
            shock_size=shock_size,
            binding_periods=0,
            regimes=[0] * horizon,
            model=ref_model,
            occbin_result=None,
            params=p_dict,
        )

    elif method_clean == "occbin":
        # Piecewise-linear simulation via OccBin
        if constraint_type == "credit_policy":
            thresh = 0.0025 if threshold is None else float(threshold)
            constraint = OccBinConstraint(variable="prem", threshold=thresh, operator=">")
            cons_model = build_gertler_karadi_model(
                p_dict,
                regime="constrained",
                constraint_type="credit_policy",
                check_steady_state=False,
            )
        elif constraint_type == "leverage_cap":
            thresh = 0.0 if threshold is None else float(threshold)
            p_cons = dict(p_dict)
            p_cons["phi_max"] = ss["phi_ss"] + thresh
            constraint = OccBinConstraint(variable="phi", threshold=thresh, operator=">")
            cons_model = build_gertler_karadi_model(
                p_cons,
                regime="constrained",
                constraint_type="leverage_cap",
                check_steady_state=False,
            )
        else:
            raise ValueError(f"unknown constraint_type {constraint_type!r}; expected 'credit_policy' or 'leverage_cap'")

        res_occ = solve_occbin(
            ref_model,
            cons_model,
            constraint,
            shock_sequence=shock_seq,
            max_iter=max_iter,
            horizon=horizon,
        )

        occ_converged = bool(getattr(res_occ, "converged", True))
        if not occ_converged:
            warnings.warn(
                f"solve_gertler_karadi: the OccBin regime iteration did not converge "
                f"within max_iter={max_iter}; the returned path is unreliable. "
                "Increase max_iter or shorten the horizon.",
                RuntimeWarning,
                stacklevel=2,
            )
        return GertlerKaradiResult(
            irf=res_occ.simulated_path.copy(),
            variables=GK_VARIABLES,
            steady_state=ss,
            solver_method="occbin",
            shock_type=shock_type,
            shock_size=shock_size,
            binding_periods=res_occ.binding_periods,
            regimes=res_occ.regimes,
            converged=occ_converged,
            model=ref_model,
            occbin_result=res_occ,
            params=p_dict,
        )

    else:
        raise ValueError(f"unknown method {method!r}; expected 'occbin' or 'klein'")
