"""Second-order DSGE perturbation with pruning (Kim, Kim, Schaumburg & Sims 2008).

Standard second-order perturbation of dynamic stochastic general equilibrium
(DSGE) models produces explosive simulation trajectories because quadratic terms
create spurious unstable manifolds outside a narrow neighborhood of the steady
state.

The pruning algorithm of Kim, Kim, Schaumburg, and Sims (2008) decomposes the
state vector into first-order and second-order components. In Dynare's timing
(``x_t`` is the state at the end of period ``t`` and responds to ``u_t``):

    x_t^{(1)} = G x_{t-1}^{(1)} + N u_t
    x_t^{(2)} = G x_{t-1}^{(2)} + 0.5 H_xx (x_{t-1}^{(1)} ⊗ x_{t-1}^{(1)})
                + H_xu (x_{t-1}^{(1)} ⊗ u_t) + 0.5 H_uu (u_t ⊗ u_t) + 0.5 H_σσ σ²
    y_t^{(1)} = F x_{t-1}^{(1)} + L u_t
    y_t^{(2)} = F x_{t-1}^{(2)} + 0.5 G_xx (x_{t-1}^{(1)} ⊗ x_{t-1}^{(1)})
                + G_xu (x_{t-1}^{(1)} ⊗ u_t) + 0.5 G_uu (u_t ⊗ u_t) + 0.5 G_σσ σ²

i.e. states and controls are both rows of Dynare's rule
``y_t = ys + 0.5 ghs2 + ghx x_{t-1} + ghu u_t + 0.5 ghxx (x⊗x) + ghxu (x⊗u)
+ 0.5 ghuu (u⊗u)``, with the quadratic terms evaluated on the first-order
component only.

Because the quadratic forcing term is evaluated strictly on the stationary
first-order state x_t^{(1)}, the pruned simulation is unconditionally stable
and ergodic whenever the first-order transition matrix G is stable (|λ(G)| < 1).

Two different "steady states under risk" follow from the same solution, and
they can have opposite signs:

* the **ergodic mean** E[x_t], E[y_t] (:meth:`~PrunedDSGESolution.ergodic_mean`),
  the average of the pruned process over its stationary distribution;
* the **risky steady state** (:meth:`~PrunedDSGESolution.risky_steady_state`),
  the point the pruned system settles at when agents expect shocks with
  covariance σ² Σ_u but every realized shock is zero.

Only the risk term 0.5 g_σσ σ² (Dynare's ``ghs2``) moves the risky steady
state; it is the precautionary effect of anticipated risk. The ergodic mean adds
the curvature of the policy times the dispersion of states and innovations,
0.5 g_xx vec(Ω) and 0.5 g_uu vec(σ² Σ_u), a Jensen effect that is present even
when the risk term is exactly zero (e.g. Brock and Mirman's model written in
levels). :meth:`~PrunedDSGESolution.risk_decomposition` reports the three parts.
``stochastic_steady_state`` is kept as an alias of ``ergodic_mean``.

References
----------
Kim, J., Kim, S., Schaumburg, E. and Sims, C.A. (2008). Calculating and using
    second-order accurate solutions of DSGE models. Journal of Economic
    Dynamics and Control, 32(11), 3397-3414.
Andreasen, M.M., Fernández-Villaverde, J. and Rubio-Ramírez, J.F. (2018).
    The Pruned State-Space System for Non-Linear DSGE Models: Theory and
    Empirical Applications. Review of Economic Studies, 85(1), 1-49.
Schmitt-Grohé, S. and Uribe, M. (2004). Solving dynamic general equilibrium
    models using a second-order approximation to the policy function.
    Journal of Economic Dynamics and Control, 28(4), 755-775.
Coeurdacier, N., Rey, H. and Winant, P. (2011). The Risky Steady State.
    American Economic Review, 101(3), 398-401.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Mapping, Sequence

import numpy as np
import pandas as pd
import scipy.linalg

from ._results import TheoreticalMomentsResult
from ._moments import compute_autocorr_matrices, conditional_fevd, first_order_moments
from ._pruned_moments import (
    MAX_PRUNED_STATE_DIM,
    pruned_order2_moments,
    pruned_order3_moments,
    pruned_state_dim,
)


def _stack_rows(top: Any, bottom: Any, n_top: int, n_bottom: int, cols: int) -> np.ndarray:
    """Stack state rows over control rows as a ``(n_top + n_bottom, cols)`` array."""
    return np.vstack([np.asarray(top, dtype=float).reshape(n_top, cols),
                      np.asarray(bottom, dtype=float).reshape(n_bottom, cols)])


# -- ergodic mean, risky steady state and their decomposition ----------------
#
# Shared by PrunedDSGESolution (order 2) and Order3PrunedSolution (order 3).
# Both expose G, N, F, H_xx, H_uu, H_sigmasigma, G_xx, G_uu, G_sigmasigma and
# ``_sigma_u``. Under Gaussian (symmetric) innovations the third-order pruned
# component has mean zero -- every third-order forcing term is an odd moment of
# x^{(1)} and u, or the product of a zero-mean factor with an independent one --
# so the second-order formulas below are also the exact order-3 pruned means.

_MEAN_PARTS = ("risk", "state_curvature", "shock_curvature")


def _ergodic_mean_parts(
    sol: Any, sigma: float, shock_cov: np.ndarray | None
) -> dict[str, tuple[np.ndarray, np.ndarray]]:
    """Closed-form pieces of the pruned ergodic mean (deviations from the deterministic steady state).

    With Ω the first-order state covariance, Ω = G Ω G' + N σ²Σ_u N', and Dynare
    timing (x_t responds to u_t, so E[x_{t-1}^{(1)} ⊗ u_t] = 0)::

        E[x] = (I - G)^{-1} [0.5 H_σσ σ² + 0.5 H_xx vec(Ω) + 0.5 H_uu vec(σ²Σ_u)]
        E[y] = F E[x]       + 0.5 G_σσ σ² + 0.5 G_xx vec(Ω) + 0.5 G_uu vec(σ²Σ_u)

    Returns ``{name: (states, controls)}`` for the three parts of ``_MEAN_PARTS``
    (each already propagated through ``(I - G)^{-1}`` and ``F``) and for
    ``"total"``. ``"total"`` is evaluated exactly as releases up to 4.3.0
    evaluated ``stochastic_steady_state`` (one linear solve of the summed
    forcing), so it is bit-for-bit unchanged; the parts sum to it up to
    floating-point rounding.
    """
    n_x, n_y = sol.n_states, sol.n_controls
    sigma_e = sol._sigma_u(sigma, shock_cov)
    sig2 = float(sigma) ** 2
    vec_se = sigma_e.flatten()
    if n_x > 0:
        q_mat = sol.N @ sigma_e @ sol.N.T
        omega = scipy.linalg.solve_discrete_lyapunov(sol.G, q_mat)
        # E[x_{t-1}^{(1)} ⊗ x_{t-1}^{(1)}] = vec(omega) (row-major, matching the
        # kron(x, x) column order of H_xx), E[u ⊗ u] = vec(sigma_e).
        vec_omega = omega.flatten()
    else:
        vec_omega = np.zeros(0)

    forcing_x = {
        "state_curvature": 0.5 * (sol.H_xx @ vec_omega),
        "shock_curvature": 0.5 * (sol.H_uu @ vec_se),
        "risk": 0.5 * sol.H_sigmasigma * sig2,
    }
    forcing_y = {
        "state_curvature": 0.5 * (sol.G_xx @ vec_omega),
        "shock_curvature": 0.5 * (sol.G_uu @ vec_se),
        "risk": 0.5 * sol.G_sigmasigma * sig2,
    }

    parts: dict[str, tuple[np.ndarray, np.ndarray]] = {}
    if n_x > 0:
        i_minus_g = np.eye(n_x) - sol.G
        rhs_x = forcing_x["state_curvature"] + forcing_x["shock_curvature"] + forcing_x["risk"]
        mu_x = scipy.linalg.solve(i_minus_g, rhs_x)
        stacked = scipy.linalg.solve(
            i_minus_g, np.column_stack([forcing_x[name] for name in _MEAN_PARTS])
        )
        x_parts = {name: stacked[:, k] for k, name in enumerate(_MEAN_PARTS)}
    else:
        mu_x = np.zeros(0)
        x_parts = {name: np.zeros(0) for name in _MEAN_PARTS}

    mu_y = sol.F @ mu_x + forcing_y["state_curvature"] + forcing_y["shock_curvature"] + forcing_y["risk"]
    for name in _MEAN_PARTS:
        parts[name] = (x_parts[name], (sol.F @ x_parts[name] + forcing_y[name]).reshape(n_y))
    parts["total"] = (mu_x, np.asarray(mu_y, dtype=float).reshape(n_y))
    return parts


def _mean_dict(sol: Any, xs: np.ndarray, ys: np.ndarray, label: str) -> dict[str, pd.Series]:
    return {
        "states": pd.Series(xs, index=list(sol.state_names), name=f"{label}_states", dtype=float),
        "controls": pd.Series(ys, index=list(sol.control_names), name=f"{label}_controls", dtype=float),
    }


def _kron_jacobian(x: np.ndarray, power: int) -> np.ndarray:
    """Jacobian of ``x ⊗ x`` (power 2) or ``x ⊗ x ⊗ x`` (power 3) with respect to ``x``."""
    n = x.size
    eye = np.eye(n)
    col = x.reshape(n, 1)
    if power == 2:
        return np.kron(eye, col) + np.kron(col, eye)
    xx = np.kron(x, x).reshape(n * n, 1)
    return np.kron(eye, xx) + np.kron(col, np.kron(eye, col)) + np.kron(xx, eye)


def _unpruned_zero_shock_map(sol: Any, x: np.ndarray, sig2: float, order: int):
    """``h(x, u=0, σ)`` of the unpruned approximated policy for states and controls, and ``dh/dx``."""
    kxx = np.kron(x, x)
    hx = sol.G @ x + 0.5 * (sol.H_xx @ kxx) + 0.5 * sol.H_sigmasigma * sig2
    gy = sol.F @ x + 0.5 * (sol.G_xx @ kxx) + 0.5 * sol.G_sigmasigma * sig2
    jac = sol.G + 0.5 * (sol.H_xx @ _kron_jacobian(x, 2))
    if order >= 3:
        kxxx = np.kron(kxx, x)
        hx = hx + (1.0 / 6.0) * (sol.H_xxx @ kxxx) + 0.5 * (sol.H_x_sigmasigma @ x) * sig2
        gy = gy + (1.0 / 6.0) * (sol.G_xxx @ kxxx) + 0.5 * (sol.G_x_sigmasigma @ x) * sig2
        jac = jac + (1.0 / 6.0) * (sol.H_xxx @ _kron_jacobian(x, 3)) + 0.5 * sol.H_x_sigmasigma * sig2
    return hx, gy, jac


def _risky_steady_state(
    sol: Any, sigma: float, pruned: bool, order: int, tol: float, maxiter: int
) -> dict[str, pd.Series]:
    """Zero-shock fixed point of the pruned (default) or unpruned approximated policy."""
    if isinstance(sigma, Mapping):
        sol._sigma_u(sigma, None)  # raises the class's explanatory TypeError
    n_x = sol.n_states
    sig2 = float(sigma) ** 2
    if n_x > 0 and not sol.is_stable:
        raise ValueError(
            "risky_steady_state: the first-order transition G has an eigenvalue on or "
            "outside the unit circle, so the zero-shock pruned path has no fixed point."
        )
    # Pruned: x^{(1)} stays at 0 without shocks, x^{(3)} has no forcing that
    # survives x^{(1)} = u = 0, and x^{(2)} converges to (I - G)^{-1} 0.5 H_σσ σ².
    if n_x > 0:
        x = scipy.linalg.solve(np.eye(n_x) - sol.G, 0.5 * sol.H_sigmasigma * sig2)
    else:
        x = np.zeros(0)
    if pruned:
        y = sol.F @ x + 0.5 * sol.G_sigmasigma * sig2
        return _mean_dict(sol, x, np.asarray(y, dtype=float).reshape(sol.n_controls), "risky_steady_state")

    # Unpruned: Newton on x = h(x, 0, σ), started from the pruned point, which
    # is within O(σ⁴) of the root of interest.
    if n_x > 0:
        resid = np.full(n_x, np.inf)
        for _ in range(int(maxiter)):
            hx, _, jac = _unpruned_zero_shock_map(sol, x, sig2, order)
            resid = hx - x
            if np.max(np.abs(resid)) <= tol:
                break
            x = x - scipy.linalg.solve(jac - np.eye(n_x), resid)
        else:
            hx, _, _ = _unpruned_zero_shock_map(sol, x, sig2, order)
            resid = hx - x
        if not np.max(np.abs(resid)) <= tol:
            raise RuntimeError(
                f"risky_steady_state(pruned=False): Newton did not reach max|h(x) - x| <= {tol:g} "
                f"in {maxiter} iterations (last residual {np.max(np.abs(resid)):.3e}). The "
                "unpruned approximated policy may have no fixed point near the deterministic "
                "steady state at this sigma; use pruned=True."
            )
    _, y, _ = _unpruned_zero_shock_map(sol, x, sig2, order)
    return _mean_dict(sol, x, np.asarray(y, dtype=float).reshape(sol.n_controls), "risky_steady_state")


def _risk_decomposition_frame(sol: Any, sigma: float, shock_cov: np.ndarray | None) -> pd.DataFrame:
    parts = _ergodic_mean_parts(sol, sigma, shock_cov)
    names = list(sol.state_names) + list(sol.control_names)
    data = {
        name: np.concatenate([parts[name][0], parts[name][1]])
        for name in (*_MEAN_PARTS, "total")
    }
    frame = pd.DataFrame(data, index=names).rename(columns={"total": "ergodic_mean"})
    order = [v for v in (sol.variable_names or names) if v in frame.index]
    frame = frame.loc[order]
    frame.index.name = "variable"
    return frame


@dataclass(frozen=True)
class PrunedSimulationResult:
    """Simulation trajectory generated by second-order pruned perturbation.

    Every frame holds *deviations from the deterministic steady state*, in the
    same units and timing as :meth:`PrunedDSGESolution.girf`. Add
    ``solution.steady_state`` to read them as levels — which is what
    :meth:`PrunedDSGESolution.stoch_simul` reports, so that its simulated and
    theoretical means share one scale.

    Attributes
    ----------
    states : pd.DataFrame
        Combined state paths x_t = x_t^{(1)} + x_t^{(2)}.
    controls : pd.DataFrame
        Combined control paths y_t = y_t^{(1)} + y_t^{(2)}.
    states_1st : pd.DataFrame
        First-order state component x_t^{(1)}.
    states_2nd : pd.DataFrame
        Second-order state correction x_t^{(2)}.
    controls_1st : pd.DataFrame
        First-order control component y_t^{(1)}.
    controls_2nd : pd.DataFrame
        Second-order control correction y_t^{(2)}.
    shocks : pd.DataFrame
        Simulated innovation paths eps_t.
    """

    states: pd.DataFrame
    controls: pd.DataFrame
    states_1st: pd.DataFrame
    states_2nd: pd.DataFrame
    controls_1st: pd.DataFrame
    controls_2nd: pd.DataFrame
    shocks: pd.DataFrame
    states_3rd: pd.DataFrame | None = None
    controls_3rd: pd.DataFrame | None = None

    def to_frame(self) -> pd.DataFrame:
        """Concatenate states and controls into a single DataFrame."""
        return pd.concat([self.states, self.controls], axis=1)

    @property
    def columns(self) -> pd.Index:
        return self.to_frame().columns

    @property
    def iloc(self) -> Any:
        return self.to_frame().iloc

    def __len__(self) -> int:
        return len(self.states)

    def __getitem__(self, key: Any) -> Any:
        return self.to_frame()[key]

    def __contains__(self, key: Any) -> bool:
        return key in self.to_frame()

    def summary(self) -> str:
        """Summary statistics of the pruned simulation paths."""
        df = self.to_frame()
        desc = df.describe().T[["mean", "std", "min", "50%", "max"]]
        desc.columns = ["Mean", "StdDev", "Min", "Median", "Max"]
        lines = [
            "Pruned DSGE Simulation Summary (Kim et al. 2008)",
            "=" * 72,
            f"Periods Simulated : {len(df)}",
            f"States            : {list(self.states.columns)}",
            f"Controls          : {list(self.controls.columns)}",
            "-" * 72,
            desc.to_string(),
            "=" * 72,
        ]
        return "\n".join(lines)


@dataclass(frozen=True)
class PrunedDSGESolution:
    """Second-order pruned DSGE state-space solution (Dynare timing).

    Attributes
    ----------
    G : np.ndarray
        (n_x, n_x) first-order state transition matrix: ``x_t = G x_{t-1} + N u_t``.
    N : np.ndarray
        (n_x, n_e) first-order state innovation loading.
    F : np.ndarray
        (n_y, n_x) first-order control policy on the *lagged* state:
        ``y_t = F x_{t-1} + L u_t`` (Dynare's ``ghx`` control rows).
    L : np.ndarray
        (n_y, n_e) first-order control innovation loading.
    H_xx : np.ndarray
        (n_x, n_x^2) second-order Hessian matrix for states (Dynare's ``ghxx``
        state rows; the rule applies ``0.5 * H_xx``).
    H_sigmasigma : np.ndarray
        (n_x,) second-order volatility drift vector for states (``ghs2``).
    G_xx : np.ndarray
        (n_y, n_x^2) second-order Hessian matrix for controls.
    G_sigmasigma : np.ndarray
        (n_y,) second-order volatility drift vector for controls.
    state_names : tuple[str, ...]
        Names of predetermined state variables.
    control_names : tuple[str, ...]
        Names of forward-looking control variables.
    shock_names : tuple[str, ...]
        Names of exogenous innovations.
    shock_cov : np.ndarray, optional
        Innovation covariance Σ_u the risk correction was computed with; the
        default covariance for simulations, IRF sizes and moments. Identity
        when not given.
    params : dict, optional
        Parameter values the model was solved with.
    first_order : LinearModel, optional
        The underlying first-order model (``build_dynare`` / ``load_mod``).
    """

    G: np.ndarray
    N: np.ndarray
    F: np.ndarray
    L: np.ndarray
    H_xx: np.ndarray
    H_sigmasigma: np.ndarray
    G_xx: np.ndarray
    G_sigmasigma: np.ndarray
    state_names: tuple[str, ...]
    control_names: tuple[str, ...]
    shock_names: tuple[str, ...]
    H_xu: np.ndarray | None = None
    H_uu: np.ndarray | None = None
    G_xu: np.ndarray | None = None
    G_uu: np.ndarray | None = None
    steady_state: pd.Series | None = None
    variable_names: tuple[str, ...] | None = None
    ghx: np.ndarray | None = None
    ghu: np.ndarray | None = None
    ghxx: np.ndarray | None = None
    ghxu: np.ndarray | None = None
    ghuu: np.ndarray | None = None
    ghs2: np.ndarray | None = None
    shock_cov: np.ndarray | None = None
    params: dict | None = None
    first_order: Any | None = None

    def __post_init__(self):
        n_x, n_y, n_e = self.n_states, self.n_controls, self.n_shocks
        if self.H_xu is None:
            object.__setattr__(self, "H_xu", np.zeros((n_x, n_x * n_e)))
        if self.H_uu is None:
            object.__setattr__(self, "H_uu", np.zeros((n_x, n_e * n_e)))
        if self.G_xu is None:
            object.__setattr__(self, "G_xu", np.zeros((n_y, n_x * n_e)))
        if self.G_uu is None:
            object.__setattr__(self, "G_uu", np.zeros((n_y, n_e * n_e)))
        if self.variable_names is None:
            object.__setattr__(self, "variable_names", self.state_names + self.control_names)
        if self.steady_state is None:
            object.__setattr__(self, "steady_state", pd.Series(0.0, index=self.variable_names))
        if self.ghx is None:
            object.__setattr__(self, "ghx", np.vstack([self.G, self.F]))
        if self.ghu is None:
            object.__setattr__(self, "ghu", np.vstack([self.N, self.L]))
        if self.ghxx is None:
            object.__setattr__(self, "ghxx", np.vstack([self.H_xx, self.G_xx]))
        if self.ghxu is None:
            object.__setattr__(self, "ghxu", np.vstack([self.H_xu, self.G_xu]))
        if self.ghuu is None:
            object.__setattr__(self, "ghuu", np.vstack([self.H_uu, self.G_uu]))
        if self.ghs2 is None:
            object.__setattr__(self, "ghs2", np.concatenate([self.H_sigmasigma, self.G_sigmasigma]))
        if self.shock_cov is None:
            object.__setattr__(self, "shock_cov", np.eye(n_e))
        else:
            cov = np.asarray(self.shock_cov, dtype=float)
            if cov.shape != (n_e, n_e):
                raise ValueError(f"shock_cov must be ({n_e}, {n_e}), got {cov.shape}")
            object.__setattr__(self, "shock_cov", cov)

    # -- sizes and names ---------------------------------------------------

    @property
    def n_states(self) -> int:
        return len(self.state_names)

    @property
    def n_controls(self) -> int:
        return len(self.control_names)

    @property
    def n_shocks(self) -> int:
        return len(self.shock_names)

    @property
    def variables(self) -> tuple[str, ...]:
        """All endogenous variable names in model order (alias of ``variable_names``)."""
        return tuple(self.variable_names or ())

    @property
    def states(self) -> tuple[str, ...]:
        """Predetermined state names (alias of ``state_names``)."""
        return tuple(self.state_names)

    @property
    def controls(self) -> tuple[str, ...]:
        """Control names (alias of ``control_names``)."""
        return tuple(self.control_names)

    @property
    def shocks(self) -> tuple[str, ...]:
        """Innovation names (alias of ``shock_names``)."""
        return tuple(self.shock_names)

    @property
    def eigenvalues(self) -> np.ndarray:
        """Eigenvalues of the first-order state transition matrix G."""
        return scipy.linalg.eigvals(self.G)

    @property
    def spectral_radius(self) -> float:
        """Spectral radius (maximum eigenvalue modulus) of state transition G."""
        return float(np.max(np.abs(self.eigenvalues))) if len(self.eigenvalues) > 0 else 0.0

    @property
    def is_stable(self) -> bool:
        """Check whether the first-order transition G is strictly stable (|λ| < 1)."""
        return bool(np.all(np.abs(self.eigenvalues) < 1.0 - 1e-7))

    # -- shock helpers -----------------------------------------------------

    def _sigma_u(self, sigma: float, shock_cov: np.ndarray | None) -> np.ndarray:
        """Innovation covariance ``σ² Σ_u`` for a perturbation scale ``sigma``."""
        if isinstance(sigma, Mapping):
            raise TypeError(
                "PrunedDSGESolution takes a scalar perturbation scale `sigma`; a "
                "per-shock mapping changes the risk correction and needs the "
                "equations: use LinearModel.stoch_simul(order=2, sigma=...) or "
                "LinearModel.solve(order=2, shock_cov=...)."
            )
        cov = self._cov() if shock_cov is None else np.asarray(shock_cov, dtype=float)
        return float(sigma) ** 2 * cov

    def _cov(self) -> np.ndarray:
        """Shock covariance, defaulting to the identity when none was declared."""
        if self.shock_cov is None:
            return np.eye(self.n_shocks)
        return np.asarray(self.shock_cov, dtype=float)

    def _shock_sd(self, sigma: float = 1.0) -> np.ndarray:
        return float(sigma) * np.sqrt(np.clip(np.diag(self._cov()), 0.0, None))

    def _var_order(self) -> list[int]:
        """Row indices of ``[states; controls]`` in ``variable_names`` order."""
        ord_names = list(self.state_names) + list(self.control_names)
        return [ord_names.index(v) for v in (self.variable_names or ())]

    def decision_rules(self):
        """Decision rule representation matching Dynare's oo_.dr structure at 2nd order.

        Returns
        -------
        Dynare2ndDR
            Container holding ghx, ghu, ghxx, ghxu, ghuu, ghs2, steady states, and variable labels.
        """
        from ._results import Dynare2ndDR

        v_names = list(self.variable_names)
        s_names = list(self.state_names)
        e_names = list(self.shock_names)

        df_ghx = pd.DataFrame(self.ghx, index=v_names, columns=s_names)
        df_ghu = pd.DataFrame(self.ghu, index=v_names, columns=e_names)

        cols_xx = [f"{s1}_{s2}" for s1 in s_names for s2 in s_names]
        df_ghxx = pd.DataFrame(self.ghxx, index=v_names, columns=cols_xx)

        cols_xu = [f"{s}_{e}" for s in s_names for e in e_names]
        df_ghxu = pd.DataFrame(self.ghxu, index=v_names, columns=cols_xu)

        cols_uu = [f"{e1}_{e2}" for e1 in e_names for e2 in e_names]
        df_ghuu = pd.DataFrame(self.ghuu, index=v_names, columns=cols_uu)

        s_ghs2 = pd.Series(self.ghs2, index=v_names)
        s_ys = (
            self.steady_state.loc[v_names]
            if isinstance(self.steady_state, pd.Series)
            else pd.Series(self.steady_state, index=v_names)
        )

        return Dynare2ndDR(
            ghx=df_ghx,
            ghu=df_ghu,
            ghxx=df_ghxx,
            ghxu=df_ghxu,
            ghuu=df_ghuu,
            ghs2=s_ghs2,
            ys=s_ys,
            state_variables=self.state_names,
            variable_names=self.variable_names,
            shock_names=self.shock_names,
        )

    @property
    def dynare_dr(self):
        """Dynare decision rules property alias."""
        return self.decision_rules()

    @property
    def oo_dr(self):
        """Dynare oo_.dr alias for direct MATLAB/Dynare parity."""
        return self.decision_rules()

    # -- pruned recursion --------------------------------------------------

    def _pruned_path(
        self,
        eps: np.ndarray,
        sigma: float,
        x0: np.ndarray | None = None,
    ) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
        """Run the pruned recursion for a given innovation path (Dynare timing).

        ``eps`` has one row per period ``t = 0..T-1``; ``x0`` is the
        first-order state ``x_{-1}`` before the first innovation (steady
        state when None). Returns ``(x1, x2, y1, y2)`` with one row per period.
        """
        eps = np.asarray(eps, dtype=float)
        total_t = eps.shape[0]
        n_x, n_y = self.n_states, self.n_controls
        x1 = np.zeros((total_t, n_x))
        x2 = np.zeros((total_t, n_x))
        y1 = np.zeros((total_t, n_y))
        y2 = np.zeros((total_t, n_y))

        sig2 = float(sigma) ** 2
        half_h_ss = 0.5 * self.H_sigmasigma * sig2
        half_g_ss = 0.5 * self.G_sigmasigma * sig2

        x1_prev = np.zeros(n_x) if x0 is None else np.asarray(x0, dtype=float)
        x2_prev = np.zeros(n_x)
        for t in range(total_t):
            e_t = eps[t]
            kron_xx = np.kron(x1_prev, x1_prev)
            kron_xe = np.kron(x1_prev, e_t)
            kron_ee = np.kron(e_t, e_t)

            x1[t] = self.G @ x1_prev + self.N @ e_t
            x2[t] = (
                self.G @ x2_prev
                + 0.5 * (self.H_xx @ kron_xx)
                + self.H_xu @ kron_xe
                + 0.5 * (self.H_uu @ kron_ee)
                + half_h_ss
            )
            y1[t] = self.F @ x1_prev + self.L @ e_t
            y2[t] = (
                self.F @ x2_prev
                + 0.5 * (self.G_xx @ kron_xx)
                + self.G_xu @ kron_xe
                + 0.5 * (self.G_uu @ kron_ee)
                + half_g_ss
            )
            x1_prev, x2_prev = x1[t], x2[t]
        return x1, x2, y1, y2

    def _draw_shocks(self, total_t: int, sigma: float, seed: int) -> np.ndarray:
        rng = np.random.default_rng(seed)
        n_e = self.n_shocks
        if n_e == 0:
            return np.zeros((total_t, 0))
        cov = self._sigma_u(sigma, None)
        return rng.multivariate_normal(np.zeros(n_e), cov, size=total_t, method="cholesky")

    def simulate(
        self,
        periods: int = 200,
        shocks: np.ndarray | None = None,
        sigma: float = 1.0,
        seed: int = 0,
        burn: int = 100,
    ) -> PrunedSimulationResult:
        """Simulate the model using the Kim et al. (2008) pruning algorithm.

        Parameters
        ----------
        periods : int, default 200
            Number of periods to return after burn-in.
        shocks : np.ndarray, optional
            Pre-specified innovation array of shape (total_periods, n_shocks),
            used as the innovations ``u_t`` directly — never rescaled by
            ``sigma``. If None, drawn as i.i.d. Gaussian ``N(0, sigma² Σ_u)``
            with ``Σ_u = shock_cov``.
        sigma : float, default 1.0
            Perturbation scale σ: scales the drawn innovations' standard
            deviation and the risk-correction terms ``0.5 ghs2 σ²``
            consistently. Because a supplied ``shocks`` array is used
            verbatim, ``sigma != 1.0`` together with ``shocks`` would scale
            the risk correction alone and is rejected.
        seed : int, default 0
            Random seed when shocks is None.
        burn : int, default 100
            Number of initial burn-in periods discarded to eliminate transient start.

        Returns
        -------
        PrunedSimulationResult
            Container with state and control paths (both total and pruned 1st/2nd components).
        """
        if not self.is_stable:
            raise ValueError(
                "First-order state transition G has eigenvalues outside the unit circle; "
                "model is explosive."
            )

        total_t = periods + burn
        n_e = self.n_shocks

        if shocks is not None and float(sigma) != 1.0:
            raise ValueError(
                f"PrunedDSGESolution.simulate: sigma={sigma!r} cannot be combined "
                "with an explicit `shocks` array. The array is used as u_t "
                "verbatim while sigma would still scale the risk correction "
                "0.5 ghs2 sigma**2, so the innovations and the risk term would "
                "sit on different scales and the simulated mean would shift by "
                "(sigma**2 - 1) (I - G)^-1 0.5 H_sigmasigma. Pass the shocks "
                "already scaled and leave sigma=1.0, or drop `shocks` and let "
                "simulate draw from N(0, sigma**2 Sigma_u)."
            )

        if shocks is None:
            eps = self._draw_shocks(total_t, sigma, seed)
        else:
            eps_arr = np.asarray(shocks, dtype=float)
            if eps_arr.ndim != 2 or eps_arr.shape[0] < total_t or eps_arr.shape[1] != n_e:
                raise ValueError(
                    f"shocks shape {eps_arr.shape} incompatible with total_t={total_t}, n_shocks={n_e}"
                )
            eps = eps_arr[:total_t]

        x1, x2, y1, y2 = self._pruned_path(eps, sigma)

        # Slice after burn-in
        x_tot = x1[burn:] + x2[burn:]
        y_tot = y1[burn:] + y2[burn:]

        idx = pd.RangeIndex(start=0, stop=periods, name="t")
        df_states = pd.DataFrame(x_tot, index=idx, columns=self.state_names)
        df_controls = pd.DataFrame(y_tot, index=idx, columns=self.control_names)
        df_x1 = pd.DataFrame(x1[burn:], index=idx, columns=self.state_names)
        df_x2 = pd.DataFrame(x2[burn:], index=idx, columns=self.state_names)
        df_y1 = pd.DataFrame(y1[burn:], index=idx, columns=self.control_names)
        df_y2 = pd.DataFrame(y2[burn:], index=idx, columns=self.control_names)
        df_eps = pd.DataFrame(eps[burn:], index=idx, columns=self.shock_names)

        return PrunedSimulationResult(
            states=df_states,
            controls=df_controls,
            states_1st=df_x1,
            states_2nd=df_x2,
            controls_1st=df_y1,
            controls_2nd=df_y2,
            shocks=df_eps,
        )

    def simulate_raw(
        self,
        periods: int = 200,
        shocks: np.ndarray | None = None,
        sigma: float = 1.0,
        seed: int = 0,
        burn: int = 100,
    ) -> tuple[np.ndarray, np.ndarray]:
        """Simulate without pruning to demonstrate explosive instability.

        Warning: Standard second-order perturbation without pruning easily explodes
        to +/- inf for larger shocks or long horizons.
        """
        total_t = periods + burn
        n_x, n_y, n_e = self.n_states, self.n_controls, self.n_shocks

        if shocks is not None and float(sigma) != 1.0:
            raise ValueError(
                f"PrunedDSGESolution.simulate_raw: sigma={sigma!r} cannot be "
                "combined with an explicit `shocks` array — the array is used "
                "as u_t verbatim while sigma would still scale the risk "
                "correction 0.5 ghs2 sigma**2. Pass the shocks already scaled "
                "and leave sigma=1.0."
            )

        if shocks is None:
            eps = self._draw_shocks(total_t, sigma, seed)
        else:
            eps = np.asarray(shocks, dtype=float)[:total_t]

        x = np.zeros((total_t, n_x))
        y = np.zeros((total_t, n_y))

        sig2 = float(sigma) ** 2
        half_h_ss = 0.5 * self.H_sigmasigma * sig2
        half_g_ss = 0.5 * self.G_sigmasigma * sig2

        x_prev = np.zeros(n_x)
        for t in range(total_t):
            e_t = eps[t]
            # Raw unpruned quadratic feedback: evaluated on the raw lagged state.
            # Clip if already overflowed to avoid numerical crash
            if np.any(np.abs(x_prev) > 1e8):
                x[t:] = np.nan
                y[t:] = np.nan
                break

            kron_xx = np.kron(x_prev, x_prev)
            kron_xe = np.kron(x_prev, e_t)
            kron_ee = np.kron(e_t, e_t)

            x[t] = (
                self.G @ x_prev + self.N @ e_t
                + 0.5 * (self.H_xx @ kron_xx) + self.H_xu @ kron_xe
                + 0.5 * (self.H_uu @ kron_ee) + half_h_ss
            )
            y[t] = (
                self.F @ x_prev + self.L @ e_t
                + 0.5 * (self.G_xx @ kron_xx) + self.G_xu @ kron_xe
                + 0.5 * (self.G_uu @ kron_ee) + half_g_ss
            )
            x_prev = x[t]

        return x[burn:], y[burn:]

    def girf(
        self,
        shock: int | str = 0,
        size: float = 1.0,
        horizon: int = 20,
        sigma: float = 1.0,
        x0: np.ndarray | None = None,
    ) -> pd.DataFrame:
        """Generalized Impulse Response Function (GIRF) under pruning.

        In non-linear models, responses depend on the sign and scale of the shock
        as well as the initial state. The GIRF computes:
            GIRF_h = E[ z_h | u_0 = size, x_{-1} ] - E[ z_h | u_0 = 0, x_{-1} ]

        with the innovation hitting at ``h = 0`` so that, as in Dynare's
        ``oo_.irfs``, row ``0`` is the impact period for states and controls
        alike.

        Parameters
        ----------
        shock : int or str, default 0
            Index or name of the innovation to shock.
        size : float, default 1.0
            Magnitude of the innovation at h=0 (in the units of ``u_t``).
        horizon : int, default 20
            Number of periods after impact.
        sigma : float, default 1.0
            Perturbation parameter scale σ. It has **no effect on the result**:
            the risk correction ``0.5 ghs2 σ²`` is a constant that enters the
            shocked and the baseline path identically and cancels in their
            difference, and ``x0`` seeds only the first-order state, which is
            also common to both. The keyword is kept for signature parity with
            :meth:`simulate`.
        x0 : np.ndarray, optional
            First-order state before impact, ``x_{-1}`` (defaults to steady state = 0).

        Returns
        -------
        pd.DataFrame
            Impulse responses for all states and controls, indexed by horizon 0..horizon.
        """
        if isinstance(shock, str):
            if shock not in self.shock_names:
                raise ValueError(
                    f"unknown shock {shock!r}; available: {self.shock_names}"
                )
            s_idx = self.shock_names.index(shock)
        else:
            s_idx = int(shock)

        n_e = self.n_shocks
        t_steps = horizon + 1

        eps_base = np.zeros((t_steps, n_e))
        eps_shock = np.zeros((t_steps, n_e))
        eps_shock[0, s_idx] = size

        def run_path(e_mat: np.ndarray) -> np.ndarray:
            x1, x2, y1, y2 = self._pruned_path(e_mat, sigma, x0=x0)
            return np.hstack([x1 + x2, y1 + y2])

        diff = run_path(eps_shock) - run_path(eps_base)

        cols = list(self.state_names) + list(self.control_names)
        return pd.DataFrame(diff, index=pd.RangeIndex(t_steps, name="h"), columns=cols)

    def irf(
        self,
        shock: int | str = 0,
        horizon: int = 20,
        size: float = 1.0,
        sigma: float = 0.0,
        x0: np.ndarray | None = None,
    ) -> pd.DataFrame:
        """Impulse responses for the 2nd-order pruned model around steady state.

        Parameters
        ----------
        shock : int or str, default 0
            Index or name of the innovation to shock.
        horizon : int, default 20
            Number of periods after impact.
        size : float, default 1.0
            Magnitude of innovation.
        sigma : float, default 0.0
            Perturbation parameter scale σ. Passed through to :meth:`girf`,
            where it cancels between the shocked and the baseline path, so the
            returned responses are the same for every σ. Kept for signature
            parity; it is not a risk-adjustment knob.
        x0 : np.ndarray, optional
            Initial state vector.

        Returns
        -------
        pd.DataFrame
            Impulse responses ordered by self.variable_names, indexed by horizon 0..horizon.
        """
        df = self.girf(shock=shock, size=size, horizon=horizon, sigma=sigma, x0=x0)
        vars_tuple = self.variable_names if self.variable_names is not None else (self.state_names + self.control_names)
        cols = [v for v in vars_tuple if v in df.columns]
        return df[cols]

    def plot(
        self,
        shock: int | str = 0,
        size: float = 1.0,
        horizon: int = 20,
        sigma: float = 1.0,
        variables: Sequence[str] | None = None,
        *,
        ax=None,
        title: str = "",
        ylabel: str = "GIRF Response",
    ):
        """Plot pruned Generalized Impulse Response Functions (GIRF)."""
        from ..plot import _new_ax

        df = self.girf(shock=shock, size=size, horizon=horizon, sigma=sigma)
        if variables is not None:
            vars_to_plot = [v for v in variables if v in df.columns]
        else:
            vars_to_plot = list(df.columns)

        fig, ax = _new_ax(ax)
        for col in vars_to_plot:
            ax.plot(df.index, df[col], label=col, linewidth=1.2)

        ax.axhline(0.0, color="0.3", linewidth=0.6, linestyle=":")
        ax.set_xlabel("Horizon (h)")
        ax.set_ylabel(ylabel)
        if not title:
            s_name = shock if isinstance(shock, str) else self.shock_names[shock]
            title = f"Pruned DSGE GIRF to {s_name} shock"
        ax.set_title(title)
        ax.legend(loc="best", frameon=False)
        return fig

    def ergodic_mean(
        self, sigma: float = 1.0, shock_cov: np.ndarray | None = None
    ) -> dict[str, pd.Series]:
        """Unconditional (ergodic) mean of the pruned second-order solution.

        This is E[x_t] and E[y_t] under the stationary distribution of the
        pruned process that :meth:`simulate` draws from -- the long-run average
        of a simulation -- not the risky steady state (see
        :meth:`risky_steady_state`). With Ω the first-order state covariance,
        ``Ω = G Ω G' + N σ²Σ_u N'``, and Dynare timing (``E[x_{t-1} ⊗ u_t] = 0``)::

            E[x] = (I - G)^{-1} [0.5 H_σσ σ² + 0.5 H_xx vec(Ω) + 0.5 H_uu vec(σ²Σ_u)]
            E[y] = F E[x]       + 0.5 G_σσ σ² + 0.5 G_xx vec(Ω) + 0.5 G_uu vec(σ²Σ_u)

        (Kim, Kim, Schaumburg and Sims 2008; Andreasen, Fernández-Villaverde
        and Rubio-Ramírez 2018). Only the first term, the risk term ``ghs2``, is
        the precautionary effect of anticipated risk; the other two are the
        curvature of the policy times the dispersion of states and innovations
        (a Jensen effect), so the ergodic mean can move in the opposite direction
        to the risky steady state. :meth:`risk_decomposition` reports the three
        parts. The mean equals the ``Mean`` column of
        ``theoretical_moments()`` minus ``steady_state`` (both conventions), and
        Dynare 8's ``oo_.mean`` for ``stoch_simul(order=2, pruning)``.

        Parameters
        ----------
        sigma : float, default 1.0
            Perturbation scale σ: innovations are ``N(0, σ² Σ_u)`` and ``ghs2``
            enters as ``0.5 ghs2 σ²``, so every term is proportional to σ².
        shock_cov : np.ndarray, optional
            Innovation covariance Σ_u used for Ω and E[u ⊗ u]; defaults to
            ``self.shock_cov``, the covariance the solution was computed with.
            ``ghs2`` is not recomputed: it belongs to ``self.shock_cov``, so a
            ``shock_cov`` that is not a multiple of it mixes two models. Re-solve
            the model with the new covariance instead.

        Returns
        -------
        dict[str, pd.Series]
            ``{"states": ..., "controls": ...}``: ergodic-mean deviations from
            the deterministic steady state, in the units of the model's
            variables (levels for a model written in levels, log deviations
            for a model in logs). Add ``steady_state`` for levels.
        """
        xs, ys = _ergodic_mean_parts(self, sigma, shock_cov)["total"]
        return _mean_dict(self, xs, ys, "ergodic_mean")

    def stochastic_steady_state(
        self, sigma: float = 1.0, shock_cov: np.ndarray | None = None
    ) -> dict[str, pd.Series]:
        """Alias of :meth:`ergodic_mean`: returns the **ergodic mean**, not the risky steady state.

        Kept for backward compatibility; it returns exactly what
        :meth:`ergodic_mean` returns. Despite the name,
        this is the unconditional mean E[x], E[y] of the pruned solution, which
        adds the curvature of the policy times the dispersion of states and
        shocks to the risk term. The zero-shock fixed point that the literature
        calls the stochastic or risky steady state (Coeurdacier, Rey and
        Winant 2011) is :meth:`risky_steady_state`; in the RBC model of
        ``tests/fixtures/dynare_live/rbc.mod`` the two put capital on opposite
        sides of its deterministic steady state (+0.0785% against -0.0045%).
        See :meth:`ergodic_mean` for parameters and return value.
        """
        return self.ergodic_mean(sigma=sigma, shock_cov=shock_cov)

    def risky_steady_state(
        self,
        sigma: float = 1.0,
        *,
        pruned: bool = True,
        tol: float = 1e-13,
        maxiter: int = 50,
    ) -> dict[str, pd.Series]:
        """Risky (stochastic) steady state: the zero-shock fixed point of the policy under risk.

        The point where the economy settles when agents expect innovations
        ``N(0, σ² Σ_u)`` but every realized innovation is zero (the "risky
        steady state" of Coeurdacier, Rey and Winant 2011). With ``pruned=True``
        (default) it is the fixed point of the pruned recursion that
        :meth:`simulate` iterates, in closed form::

            x_rss = (I - G)^{-1} 0.5 H_σσ σ²,     y_rss = F x_rss + 0.5 G_σσ σ²

        i.e. the limit of ``simulate(shocks=np.zeros(...), burn=0)``. Only the
        risk term ``ghs2`` enters, so this is the precautionary effect of
        anticipated risk; it is the ``risk`` column of
        :meth:`risk_decomposition`. It is not the ergodic mean
        (:meth:`ergodic_mean`, alias ``stochastic_steady_state``), which also
        contains curvature times dispersion and can have the opposite sign.

        With ``pruned=False`` it is the fixed point ``x = h(x, 0, σ)`` of the
        unpruned second-order policy ``h(x, 0, σ) = G x + 0.5 H_xx (x ⊗ x) +
        0.5 H_σσ σ²`` (the limit of :meth:`simulate_raw` with zero shocks),
        found by Newton's method from the pruned value. The two differ by terms
        of order σ⁴, below the accuracy of a second-order approximation.

        Parameters
        ----------
        sigma : float, default 1.0
            Perturbation scale σ; the risk term enters as ``0.5 ghs2 σ²``.
        pruned : bool, default True
            Fixed point of the pruned recursion (closed form) or of the
            unpruned approximated policy (Newton).
        tol, maxiter : float, int
            Newton tolerance on ``max|h(x) - x|`` and iteration cap
            (``pruned=False`` only); a failure raises ``RuntimeError``.

        Returns
        -------
        dict[str, pd.Series]
            ``{"states": ..., "controls": ...}``: deviations from the
            deterministic steady state, in the units of the model's variables.
            Add ``steady_state`` for levels.
        """
        return _risky_steady_state(self, sigma, pruned, 2, tol, maxiter)

    def risk_decomposition(
        self, sigma: float = 1.0, shock_cov: np.ndarray | None = None
    ) -> pd.DataFrame:
        """Split the pruned ergodic mean into the risk term and two curvature terms.

        Columns (deviations from the deterministic steady state, one row per
        variable in ``variable_names`` order)::

            risk             (I - G)^{-1} 0.5 H_σσ σ²           and F(.) + 0.5 G_σσ σ²
            state_curvature  (I - G)^{-1} 0.5 H_xx vec(Ω)       and F(.) + 0.5 G_xx vec(Ω)
            shock_curvature  (I - G)^{-1} 0.5 H_uu vec(σ²Σ_u)   and F(.) + 0.5 G_uu vec(σ²Σ_u)
            ergodic_mean     their sum, equal to :meth:`ergodic_mean`

        ``risk`` is the precautionary (anticipated-risk) effect and equals
        :meth:`risky_steady_state`; ``state_curvature`` and ``shock_curvature``
        are Jensen effects of realized dispersion, present even when the risk
        term is zero. The three columns sum to ``ergodic_mean`` up to
        floating-point rounding. Parameters as in :meth:`ergodic_mean`.
        """
        return _risk_decomposition_frame(self, sigma, shock_cov)

    def _pruned_coefficients(self, sig2: float) -> dict[str, np.ndarray]:
        """Dynare-convention coefficients stacked [states; controls], ``gss`` times sigma^2."""
        n_x, n_y, n_e = self.n_states, self.n_controls, self.n_shocks

        def stack(top: Any, bottom: Any, cols: int) -> np.ndarray:
            return _stack_rows(top, bottom, n_x, n_y, cols)

        return {
            "gx": stack(self.G, self.F, n_x),
            "gu": stack(self.N, self.L, n_e),
            "gxx": stack(self.H_xx, self.G_xx, n_x**2),
            "gxu": stack(self.H_xu, self.G_xu, n_x * n_e),
            "guu": stack(self.H_uu, self.G_uu, n_e**2),
            "gss": sig2 * stack(self.H_sigmasigma, self.G_sigmasigma, 1).ravel(),
        }

    def theoretical_moments(
        self,
        sigma: float = 1.0,
        shock_cov: np.ndarray | None = None,
        lags: int = 4,
        fevd_horizons: Sequence[int | None] = (1, 4, 8, 16, 32, None),
        *,
        pruning: bool = False,
        max_state_dim: int | None = MAX_PRUNED_STATE_DIM,
    ) -> TheoreticalMomentsResult:
        """Theoretical moments of the second-order solution, in either of Dynare's conventions.

        With ``pruning=False`` (the default) the moments follow Dynare's
        ``stoch_simul(order=2)`` without the ``pruning`` option: the means are
        the second-order ergodic means of :meth:`ergodic_mean`,
        while covariances, correlations and autocorrelations are those of the
        first-order solution (they coincide with
        :meth:`LinearModel.theoretical_moments` for the same rules and shock
        covariance). With ``pruning=True`` they are the exact moments of the
        pruned solution that :meth:`simulate` draws from, the moments Dynare
        reports for ``stoch_simul(order=2, pruning)``: the means are the same and
        the second moments add the terms of order sigma^4. The variance
        decomposition is the first-order one in both conventions, as in Dynare.

        Parameters
        ----------
        sigma : float, default 1.0
            Perturbation scale σ (scales the innovation standard deviations).
        shock_cov : np.ndarray, optional
            Covariance matrix of innovations (n_e x n_e). Default ``self.shock_cov``.
        lags : int, default 4
            Number of autocorrelation lags to report.
        fevd_horizons : Sequence[int | None], default (1, 4, 8, 16, 32, None)
            Horizons for the conditional variance decomposition (None = asymptotic).
        pruning : bool, default False
            Report the exact moments of the pruned second-order solution instead
            of first-order second moments.
        max_state_dim : int or None, default 3000
            With ``pruning=True``, the largest augmented pruned state
            (2n + n^2 for n states) to solve; larger models raise ``ValueError``.

        Returns
        -------
        TheoreticalMomentsResult
            Analytical moments [Mean, Std.Dev., Variance], correlation matrix,
            autocorrelation coefficients and matrices, and variance
            decomposition (percent).
        """
        sigma_e = self._sigma_u(sigma, shock_cov)
        order = self._var_order()
        all_names = list(self.variable_names or ())
        ss_base = (
            self.steady_state
            if isinstance(self.steady_state, pd.Series)
            else pd.Series(0.0, index=all_names)
        )
        if pruning:
            mean_dev, gamma_0, gammas = pruned_order2_moments(
                self._pruned_coefficients(float(sigma) ** 2), self.n_states, sigma_e, lags,
                max_state_dim=max_state_dim,
            )
            mean_vec = np.array([float(ss_base.get(v, 0.0)) for v in all_names]) + mean_dev[order]
        else:
            M_x = np.vstack([self.G, self.F])
            M_u = np.vstack([self.N, self.L])
            _, gamma_0, gammas = first_order_moments(self.G, self.N, M_x, M_u, sigma_e, lags)
            sss = self.ergodic_mean(sigma=sigma, shock_cov=shock_cov)
            mean_vec = np.zeros(len(all_names))
            for i, v in enumerate(all_names):
                base_val = float(ss_base.get(v, 0.0))
                if v in self.state_names:
                    mean_vec[i] = base_val + float(sss["states"][v])
                elif v in self.control_names:
                    mean_vec[i] = base_val + float(sss["controls"][v])
                else:
                    mean_vec[i] = base_val
        cov_mat = gamma_0[np.ix_(order, order)]
        gammas = [gamma_k[np.ix_(order, order)] for gamma_k in gammas]

        variances = np.diag(cov_mat)
        stds = np.sqrt(np.maximum(variances, 0.0))

        df_moments = pd.DataFrame(
            {"Mean": mean_vec, "Std.Dev.": stds, "Variance": variances},
            index=all_names,
        )

        std_outer = np.outer(stds, stds)
        std_outer[std_outer == 0.0] = np.nan
        corr_mat = cov_mat / std_outer
        np.fill_diagonal(corr_mat, 1.0)

        df_cov = pd.DataFrame(cov_mat, index=all_names, columns=all_names)
        df_corr = pd.DataFrame(corr_mat, index=all_names, columns=all_names)

        autocorr_cols = [f"Lag {k}" for k in range(1, lags + 1)]
        df_autocorr = pd.DataFrame(index=all_names, columns=autocorr_cols, dtype=float)
        for k, gamma_k in enumerate(gammas, start=1):
            with np.errstate(divide="ignore", invalid="ignore"):
                df_autocorr[f"Lag {k}"] = np.where(variances > 1e-14, np.diag(gamma_k) / variances, np.nan)
        _, autocorr_mats = compute_autocorr_matrices(cov_mat, gammas, all_names)

        sd = np.sqrt(np.clip(np.diag(sigma_e), 0.0, None))
        C = np.asarray(self.ghx, dtype=float)
        D = np.asarray(self.ghu, dtype=float)
        df_fevd = conditional_fevd(
            self.G, self.N, C, D, sd, list(fevd_horizons), all_names, list(self.shock_names)
        ) * 100.0

        return TheoreticalMomentsResult(
            moments=df_moments,
            covariance=df_cov,
            correlation=df_corr,
            autocorr=df_autocorr,
            fevd=df_fevd,
            autocorr_matrices=autocorr_mats,
        )

    def stoch_simul(
        self,
        *,
        order: int = 2,
        irf: int = 40,
        periods: int = 0,
        sigma: float = 1.0,
        seed: int = 0,
        burn: int = 100,
        lags: int = 5,
        pruning: bool = False,
    ):
        """Execute Dynare-compatible 2nd-order stoch_simul routine.

        Computes:
        1. Second-order decision rules (oo_.dr)
        2. Theoretical unconditional moments in the convention ``pruning`` selects
           (see :meth:`theoretical_moments`): by default Dynare's
           ``stoch_simul(order=2)``, first-order second moments with the
           second-order mean; with ``pruning=True`` the exact moments of the
           pruned solution, as Dynare's ``stoch_simul(order=2, pruning)``
        3. Impulse response functions to a one-standard-deviation innovation in each shock
        4. Simulated sample moments (if periods > 0), with the ``Mean`` column
           in levels so that it is directly comparable with the theoretical
           ``Mean``; the higher moments are unaffected by that shift.

        Parameters
        ----------
        order : int, default 2
            Approximation order (must be 2).
        irf : int, default 40
            Horizon for impulse response functions. Set to 0 to skip IRFs.
        periods : int, default 0
            Number of simulation periods. If > 0, generates simulated moments.
        sigma : float, default 1.0
            Perturbation scale σ applied to the declared shock covariance
            ``shock_cov`` (innovations ``N(0, σ² Σ_u)``, IRFs of size
            ``σ sqrt(Σ_u[j, j])``). A per-shock mapping is rejected: it
            changes the risk correction, so re-solve through
            ``LinearModel.stoch_simul(order=2, sigma=...)`` instead.
        seed : int, default 0
            RNG seed for simulation when periods > 0.
        burn : int, default 100
            Burn-in periods dropped before calculating simulated moments.
        lags : int, default 5
            Number of autocorrelation lags.
        pruning : bool, default False
            Moment convention of the theoretical moments (see above). Simulations
            and impulse responses always use the pruned solution.

        Returns
        -------
        StochSimulResult
            Container holding dr, theoretical_moments, simulated_moments, irfs, and export methods.
        """
        from ._results import StochSimulResult

        if order != 2:
            raise ValueError(f"PrunedDSGESolution only supports order=2, got order={order}")
        if isinstance(sigma, Mapping):
            self._sigma_u(sigma, None)  # raises the explanatory TypeError

        dr = self.decision_rules()
        theo = self.theoretical_moments(sigma=sigma, lags=lags, pruning=pruning)

        vars_tuple = self.variable_names if self.variable_names is not None else (self.state_names + self.control_names)
        sd = self._shock_sd(sigma)
        irfs: dict[str, pd.Series] = {}
        if irf > 0:
            for j, sh in enumerate(self.shock_names):
                df_irf = self.irf(shock=sh, horizon=irf, size=float(sd[j]), sigma=0.0)
                for v in vars_tuple:
                    irfs[f"{v}_{sh}"] = df_irf[v]

        sim_moments = None
        if periods > 0:
            sim_res = self.simulate(periods=periods, sigma=sigma, seed=seed, burn=burn)
            sim_df = pd.concat([sim_res.states, sim_res.controls], axis=1)[list(vars_tuple)]
            # The pruned paths are deviations from the deterministic steady
            # state, but `theoretical_moments` reports means in levels (Dynare's
            # convention). Put the simulated mean on the same scale, otherwise
            # the two Mean columns of the same report contradict each other.
            # Only the mean moves: Std.Dev./Variance/Skewness/Kurtosis are
            # shift-invariant, so they are computed on the deviations.
            ss_level = (
                self.steady_state.reindex(list(vars_tuple)).astype(float).fillna(0.0)
                if isinstance(self.steady_state, pd.Series)
                else pd.Series(0.0, index=list(vars_tuple))
            )
            sim_moments = pd.DataFrame(
                {
                    "Mean": sim_df.mean(axis=0) + ss_level,
                    "Std.Dev.": sim_df.std(axis=0),
                    "Variance": sim_df.var(axis=0),
                    "Skewness": sim_df.skew(axis=0),
                    "Kurtosis": sim_df.kurtosis(axis=0),
                },
                index=list(vars_tuple),
            )

        return StochSimulResult(
            dr=dr,
            theoretical_moments=theo,
            simulated_moments=sim_moments,
            irfs=irfs,
            order=2,
            variable_names=vars_tuple,
            shock_names=self.shock_names,
        )


def canonical_growth_2nd_order(
    alpha: float = 0.33,
    beta: float = 0.99,
    delta: float = 0.025,
    sigma_pref: float = 1.0,
    rho: float = 0.95,
    sigma_eps: float = 0.01,
) -> PrunedDSGESolution:
    """Solve the canonical one-sector growth model to second order with pruning.

    The benchmark of Kim, Kim, Schaumburg & Sims (2008):

      max E_0 sum beta^t (c_t^{1-sigma_pref} - 1) / (1 - sigma_pref)
      s.t. c_t + k_t - (1 - delta) k_{t-1} = z_t k_{t-1}^alpha
           ln z_t = rho ln z_{t-1} + sigma_eps * eps_t,   eps_t ~ N(0, 1)

    written in log deviations (``k = ln K``, ``c = ln C``, ``z = ln Z``) so
    that every decision-rule coefficient reads as an elasticity. The model
    is solved with :func:`puremacro.dsge.solve_dynare_2nd_order`, i.e. the
    matrices are the genuine second-order perturbation, not a calibration.

    Returns
    -------
    PrunedDSGESolution
        States ``("k", "z")``, control ``("c",)``, shock ``("eps",)``, with
        ``shock_cov = [[1.0]]`` (the shock scale ``sigma_eps`` sits inside
        the technology equation).
    """
    from puremacro.dsge.dynare import solve_dynare_2nd_order

    r_ss = 1.0 / beta - (1.0 - delta)
    k_ss = (r_ss / alpha) ** (1.0 / (alpha - 1.0))
    c_ss = k_ss**alpha - delta * k_ss

    def growth(lead, curr, lag, shocks, p):
        return [
            np.exp(-p.sigma_pref * curr.c)
            - p.beta * np.exp(-p.sigma_pref * lead.c)
            * (p.alpha * np.exp(lead.z) * np.exp((p.alpha - 1.0) * curr.k) + 1.0 - p.delta),
            np.exp(curr.c) + np.exp(curr.k)
            - np.exp(curr.z) * np.exp(p.alpha * lag.k) - (1.0 - p.delta) * np.exp(lag.k),
            curr.z - p.rho * lag.z - p.sigma_eps * shocks.eps,
        ]

    return solve_dynare_2nd_order(
        growth,
        variables=["k", "z", "c"],
        shocks=["eps"],
        params=dict(alpha=alpha, beta=beta, delta=delta, sigma_pref=sigma_pref,
                    rho=rho, sigma_eps=sigma_eps),
        steady_state=dict(k=float(np.log(k_ss)), z=0.0, c=float(np.log(c_ss))),
        states=["k", "z"],
        shock_cov=np.eye(1),
    )


@dataclass(frozen=True)
class Dynare3rdDR:
    """Third-order decision rule representation matching Dynare's oo_.dr structure.

    Third-order approximation around steady state:
        y_t = ys + 0.5 * ghs2 * sigma^2 + ghx * x + ghu * u
              + 0.5 * ghxx * (x ⊗ x) + ghxu * (x ⊗ u) + 0.5 * ghuu * (u ⊗ u)
              + (1/6) * ghxxx * (x ⊗ x ⊗ x) + 0.5 * ghxxu * (x ⊗ x ⊗ u)
              + 0.5 * ghxuu * (x ⊗ u ⊗ u) + (1/6) * ghuuu * (u ⊗ u ⊗ u)
              + 0.5 * ghxss * x * sigma^2 + 0.5 * ghuss * u * sigma^2

    Attributes
    ----------
    ghx : pd.DataFrame
        (n_vars x n_states) first-order state policy derivatives.
    ghu : pd.DataFrame
        (n_vars x n_shocks) first-order shock policy derivatives.
    ghxx : pd.DataFrame
        (n_vars x n_states^2) second-order state policy derivatives.
    ghxu : pd.DataFrame
        (n_vars x (n_states * n_shocks)) cross state-shock derivatives.
    ghuu : pd.DataFrame
        (n_vars x n_shocks^2) second-order shock derivatives.
    ghs2 : pd.Series
        (n_vars,) volatility / risk correction terms.
    ghxxx : pd.DataFrame
        (n_vars x n_states^3) third-order state policy derivatives.
    ghxxu : pd.DataFrame
        (n_vars x (n_states^2 * n_shocks)) third-order state-state-shock derivatives.
    ghxuu : pd.DataFrame
        (n_vars x (n_states * n_shocks^2)) third-order state-shock-shock derivatives.
    ghuuu : pd.DataFrame
        (n_vars x n_shocks^3) third-order shock derivatives.
    ghxss : pd.DataFrame
        (n_vars x n_states) cross state-volatility derivatives.
    ghuss : pd.DataFrame
        (n_vars x n_shocks) cross shock-volatility derivatives.
    ys : pd.Series
        Steady-state values for all endogenous variables.
    state_variables : tuple[str, ...]
        Names of predetermined state variables.
    variable_names : tuple[str, ...]
        Names of all endogenous variables in model order.
    shock_names : tuple[str, ...]
        Names of structural shocks.
    """

    ghx: pd.DataFrame
    ghu: pd.DataFrame
    ghxx: pd.DataFrame
    ghxu: pd.DataFrame
    ghuu: pd.DataFrame
    ghs2: pd.Series
    ghxxx: pd.DataFrame
    ghxxu: pd.DataFrame
    ghxuu: pd.DataFrame
    ghuuu: pd.DataFrame
    ghxss: pd.DataFrame
    ghuss: pd.DataFrame
    ys: pd.Series
    state_variables: tuple[str, ...]
    variable_names: tuple[str, ...]
    shock_names: tuple[str, ...]

    def __getitem__(self, key: str):
        if not isinstance(key, str):
            raise KeyError(key)
        if hasattr(self, key):
            return getattr(self, key)
        raise KeyError(key)

    def __contains__(self, key: object) -> bool:
        return isinstance(key, str) and (hasattr(self, key) or key in getattr(self, "extra", {}))

    def to_frame(self) -> pd.DataFrame:
        """Concatenate decision rule summary into a single DataFrame."""
        dfs = [self.ys.to_frame(name="SteadyState")]
        if not self.ghs2.empty:
            dfs.append(self.ghs2.to_frame(name="ghs2"))
        if not self.ghx.empty:
            dfs.append(self.ghx)
        if not self.ghu.empty:
            dfs.append(self.ghu)
        if not self.ghxss.empty:
            dfs.append(self.ghxss.add_prefix("xss_"))
        if not self.ghuss.empty:
            dfs.append(self.ghuss.add_prefix("uss_"))
        return pd.concat(dfs, axis=1)

    def to_markdown(self, **kwargs) -> str:
        from puremacro.reports import _df_to_markdown

        return _df_to_markdown(self.to_frame(), **kwargs)

    def to_latex(self, **kwargs) -> str:
        from puremacro.reports import _df_to_latex

        return _df_to_latex(self.to_frame(), **kwargs)

    def to_typst(self, **kwargs) -> str:
        from puremacro.reports import _df_to_typst

        return _df_to_typst(self.to_frame(), **kwargs)


@dataclass(frozen=True)
class Order3TheoreticalMomentsResult(TheoreticalMomentsResult):
    """Theoretical moments result for order-3 pruned DSGE models."""

    skewness: pd.Series | None = None
    kurtosis: pd.Series | None = None

    @property
    def mean(self) -> pd.Series:
        return self.moments["Mean"]

    @property
    def variance(self) -> pd.Series:
        return self.moments["Variance"]

    def __contains__(self, key: str) -> bool:
        if key in ("variance", "var"):
            return "Variance" in self.moments.columns or hasattr(self, "variance")
        if key in ("skewness", "skew"):
            return self.skewness is not None or "Skewness" in self.moments.columns
        if key in ("kurtosis", "kurt"):
            return self.kurtosis is not None or "Kurtosis" in self.moments.columns
        return key in self.moments.columns or hasattr(self, key)

    def __getitem__(self, key: str) -> Any:
        if key in ("variance", "var"):
            return self.moments["Variance"]
        if key in ("skewness", "skew"):
            return self.skewness if self.skewness is not None else self.moments["Skewness"]
        if key in ("kurtosis", "kurt"):
            return self.kurtosis if self.kurtosis is not None else self.moments["Kurtosis"]
        return getattr(self, key)


@dataclass(frozen=True, init=False)
class Order3PrunedSolution:
    """Third-order pruned DSGE state-space solution (Andreasen et al. 2018).

    Decomposes state and control variables into three additive components:
        x_t = x_t^{(1)} + x_t^{(2)} + x_t^{(3)}
        y_t = y_t^{(1)} + y_t^{(2)} + y_t^{(3)}
    guaranteeing non-explosive, unconditionally ergodic simulations and finite
    higher-order stationary moments whenever first-order stability holds (|λ(G)| < 1).
    """

    # First-order
    G: np.ndarray
    N: np.ndarray
    F: np.ndarray
    L: np.ndarray
    # Second-order
    H_xx: np.ndarray
    H_sigmasigma: np.ndarray
    G_xx: np.ndarray
    G_sigmasigma: np.ndarray
    # Third-order
    H_xxx: np.ndarray
    H_xxu: np.ndarray
    H_xuu: np.ndarray
    H_uuu: np.ndarray
    H_x_sigmasigma: np.ndarray
    H_u_sigmasigma: np.ndarray
    G_xxx: np.ndarray
    G_xxu: np.ndarray
    G_xuu: np.ndarray
    G_uuu: np.ndarray
    G_x_sigmasigma: np.ndarray
    G_u_sigmasigma: np.ndarray
    # Metadata
    state_names: tuple[str, ...]
    control_names: tuple[str, ...]
    shock_names: tuple[str, ...]
    # Optional / cached fields
    H_xu: np.ndarray | None = None
    H_uu: np.ndarray | None = None
    G_xu: np.ndarray | None = None
    G_uu: np.ndarray | None = None
    steady_state: pd.Series | None = None
    variable_names: tuple[str, ...] | None = None
    ghx: np.ndarray | None = None
    ghu: np.ndarray | None = None
    ghxx: np.ndarray | None = None
    ghxu: np.ndarray | None = None
    ghuu: np.ndarray | None = None
    ghs2: np.ndarray | None = None
    ghxxx: np.ndarray | None = None
    ghxxu: np.ndarray | None = None
    ghxuu: np.ndarray | None = None
    ghuuu: np.ndarray | None = None
    ghxss: np.ndarray | None = None
    ghuss: np.ndarray | None = None
    shock_cov: np.ndarray | None = None
    params: dict | None = None
    first_order: Any | None = None
    # User-overrides for test contracts
    _user_g_xxx: np.ndarray | None = None
    _user_g_uuu: np.ndarray | None = None

    def __init__(
        self,
        G: np.ndarray | None = None,
        N: np.ndarray | None = None,
        F: np.ndarray | None = None,
        L: np.ndarray | None = None,
        H_xx: np.ndarray | None = None,
        H_sigmasigma: np.ndarray | None = None,
        G_xx: np.ndarray | None = None,
        G_sigmasigma: np.ndarray | None = None,
        H_xxx: np.ndarray | None = None,
        H_xxu: np.ndarray | None = None,
        H_xuu: np.ndarray | None = None,
        H_uuu: np.ndarray | None = None,
        H_x_sigmasigma: np.ndarray | None = None,
        H_u_sigmasigma: np.ndarray | None = None,
        G_xxx: np.ndarray | None = None,
        G_xxu: np.ndarray | None = None,
        G_xuu: np.ndarray | None = None,
        G_uuu: np.ndarray | None = None,
        G_x_sigmasigma: np.ndarray | None = None,
        G_u_sigmasigma: np.ndarray | None = None,
        state_names: Sequence[str] = (),
        control_names: Sequence[str] = (),
        shock_names: Sequence[str] = (),
        H_xu: np.ndarray | None = None,
        H_uu: np.ndarray | None = None,
        G_xu: np.ndarray | None = None,
        G_uu: np.ndarray | None = None,
        steady_state: pd.Series | Mapping[str, float] | None = None,
        variable_names: Sequence[str] | None = None,
        ghx: np.ndarray | None = None,
        ghu: np.ndarray | None = None,
        ghxx: np.ndarray | None = None,
        ghxu: np.ndarray | None = None,
        ghuu: np.ndarray | None = None,
        ghs2: np.ndarray | None = None,
        ghxxx: np.ndarray | None = None,
        ghxxu: np.ndarray | None = None,
        ghxuu: np.ndarray | None = None,
        ghuuu: np.ndarray | None = None,
        ghxss: np.ndarray | None = None,
        ghuss: np.ndarray | None = None,
        shock_cov: np.ndarray | None = None,
        params: dict | None = None,
        first_order: Any | None = None,
        # Full-matrix aliases
        g_x: np.ndarray | None = None,
        g_u: np.ndarray | None = None,
        g_xx: np.ndarray | None = None,
        g_xu: np.ndarray | None = None,
        g_uu: np.ndarray | None = None,
        g_ss: np.ndarray | None = None,
        g_xxx: np.ndarray | None = None,
        g_xxu: np.ndarray | None = None,
        g_xuu: np.ndarray | None = None,
        g_uuu: np.ndarray | None = None,
        g_x_ss: np.ndarray | None = None,
        g_u_ss: np.ndarray | None = None,
        **kwargs: Any,
    ):
        s_names = tuple(state_names)
        c_names = tuple(control_names)
        e_names = tuple(shock_names)
        n_x = len(s_names)
        n_y = len(c_names)
        n_e = len(e_names)
        n_total = n_x + n_y

        object.__setattr__(self, "state_names", s_names)
        object.__setattr__(self, "control_names", c_names)
        object.__setattr__(self, "shock_names", e_names)

        v_names = tuple(variable_names) if variable_names is not None else (s_names + c_names)
        object.__setattr__(self, "variable_names", v_names)

        if steady_state is None:
            s_state = pd.Series(0.0, index=v_names)
        elif isinstance(steady_state, pd.Series):
            s_state = steady_state
        else:
            s_state = pd.Series(steady_state)
        object.__setattr__(self, "steady_state", s_state)

        # Helper to split (N, d) or handle special toy shapes
        def split_mat(mat: np.ndarray | None, d: int, default_val: float = 0.0) -> tuple[np.ndarray, np.ndarray]:
            if mat is None:
                return np.full((n_x, d), default_val, dtype=float), np.full((n_y, d), default_val, dtype=float)
            arr = np.asarray(mat, dtype=float)
            if arr.ndim == 1:
                if arr.size == n_total:
                    return arr[:n_x], arr[n_x:n_total]
                elif arr.size == n_x and n_x == 1 and n_y == 1:
                    return arr.copy(), arr.copy()
                elif arr.size == n_x:
                    return arr, np.zeros(n_y, dtype=float)
                elif arr.size == n_y:
                    return np.zeros(n_x, dtype=float), arr
                else:
                    return np.pad(arr, (0, max(0, n_x - arr.size)))[:n_x], np.zeros(n_y, dtype=float)
            if arr.shape[0] == n_total:
                return arr[:n_x], arr[n_x:n_total]
            elif arr.shape[0] == n_x and n_x == 1 and n_y == 1:
                return arr.copy(), arr.copy()
            elif arr.shape[0] == n_x:
                return arr, np.zeros((n_y, arr.shape[1]), dtype=float)
            elif arr.shape[0] == n_y:
                return np.zeros((n_x, arr.shape[1]), dtype=float), arr
            return np.zeros((n_x, d), dtype=float), np.zeros((n_y, d), dtype=float)

        # 1st order
        mat_gx = ghx if ghx is not None else g_x
        if G is None or F is None:
            g_state, g_ctrl = split_mat(mat_gx, n_x)
            G = G if G is not None else g_state
            F = F if F is not None else g_ctrl
        object.__setattr__(self, "G", np.asarray(G, dtype=float))
        object.__setattr__(self, "F", np.asarray(F, dtype=float))

        mat_gu = ghu if ghu is not None else g_u
        if N is None or L is None:
            n_state, l_ctrl = split_mat(mat_gu, n_e)
            N = N if N is not None else n_state
            L = L if L is not None else l_ctrl
        object.__setattr__(self, "N", np.asarray(N, dtype=float))
        object.__setattr__(self, "L", np.asarray(L, dtype=float))

        # 2nd order
        mat_gxx = ghxx if ghxx is not None else g_xx
        if H_xx is None or G_xx is None:
            hxx, gxx = split_mat(mat_gxx, n_x**2)
            H_xx = H_xx if H_xx is not None else hxx
            G_xx = G_xx if G_xx is not None else gxx
        object.__setattr__(self, "H_xx", np.asarray(H_xx, dtype=float))
        object.__setattr__(self, "G_xx", np.asarray(G_xx, dtype=float))

        mat_gxu = ghxu if ghxu is not None else g_xu
        if H_xu is None or G_xu is None:
            hxu, gxu = split_mat(mat_gxu, n_x * n_e)
            H_xu = H_xu if H_xu is not None else hxu
            G_xu = G_xu if G_xu is not None else gxu
        object.__setattr__(self, "H_xu", np.asarray(H_xu, dtype=float))
        object.__setattr__(self, "G_xu", np.asarray(G_xu, dtype=float))

        mat_guu = ghuu if ghuu is not None else g_uu
        if H_uu is None or G_uu is None:
            huu, guu = split_mat(mat_guu, n_e**2)
            H_uu = H_uu if H_uu is not None else huu
            G_uu = G_uu if G_uu is not None else guu
        object.__setattr__(self, "H_uu", np.asarray(H_uu, dtype=float))
        object.__setattr__(self, "G_uu", np.asarray(G_uu, dtype=float))

        mat_gss = ghs2 if ghs2 is not None else g_ss
        if H_sigmasigma is None or G_sigmasigma is None:
            hss, gss = split_mat(mat_gss, 1)
            H_sigmasigma = H_sigmasigma if H_sigmasigma is not None else np.ravel(hss)
            G_sigmasigma = G_sigmasigma if G_sigmasigma is not None else np.ravel(gss)
        object.__setattr__(self, "H_sigmasigma", np.asarray(H_sigmasigma, dtype=float).ravel())
        object.__setattr__(self, "G_sigmasigma", np.asarray(G_sigmasigma, dtype=float).ravel())

        # 3rd order
        mat_gxxx = ghxxx if ghxxx is not None else g_xxx
        if H_xxx is None or G_xxx is None:
            hxxx, gxxx = split_mat(mat_gxxx, n_x**3)
            H_xxx = H_xxx if H_xxx is not None else hxxx
            G_xxx = G_xxx if G_xxx is not None else gxxx
        object.__setattr__(self, "H_xxx", np.asarray(H_xxx, dtype=float))
        object.__setattr__(self, "G_xxx", np.asarray(G_xxx, dtype=float))

        mat_gxxu = ghxxu if ghxxu is not None else g_xxu
        if H_xxu is None or G_xxu is None:
            hxxu, gxxu = split_mat(mat_gxxu, (n_x**2) * n_e)
            H_xxu = H_xxu if H_xxu is not None else hxxu
            G_xxu = G_xxu if G_xxu is not None else gxxu
        object.__setattr__(self, "H_xxu", np.asarray(H_xxu, dtype=float))
        object.__setattr__(self, "G_xxu", np.asarray(G_xxu, dtype=float))

        mat_gxuu = ghxuu if ghxuu is not None else g_xuu
        if H_xuu is None or G_xuu is None:
            hxuu, gxuu = split_mat(mat_gxuu, n_x * (n_e**2))
            H_xuu = H_xuu if H_xuu is not None else hxuu
            G_xuu = G_xuu if G_xuu is not None else gxuu
        object.__setattr__(self, "H_xuu", np.asarray(H_xuu, dtype=float))
        object.__setattr__(self, "G_xuu", np.asarray(G_xuu, dtype=float))

        mat_guuu = ghuuu if ghuuu is not None else g_uuu
        if H_uuu is None or G_uuu is None:
            huuu, guuu = split_mat(mat_guuu, n_e**3)
            H_uuu = H_uuu if H_uuu is not None else huuu
            G_uuu = G_uuu if G_uuu is not None else guuu
        object.__setattr__(self, "H_uuu", np.asarray(H_uuu, dtype=float))
        object.__setattr__(self, "G_uuu", np.asarray(G_uuu, dtype=float))

        mat_gxss = ghxss if ghxss is not None else g_x_ss
        if H_x_sigmasigma is None or G_x_sigmasigma is None:
            hxss, gxss = split_mat(mat_gxss, n_x)
            H_x_sigmasigma = H_x_sigmasigma if H_x_sigmasigma is not None else hxss
            G_x_sigmasigma = G_x_sigmasigma if G_x_sigmasigma is not None else gxss
        object.__setattr__(self, "H_x_sigmasigma", np.asarray(H_x_sigmasigma, dtype=float))
        object.__setattr__(self, "G_x_sigmasigma", np.asarray(G_x_sigmasigma, dtype=float))

        mat_guss = ghuss if ghuss is not None else g_u_ss
        if H_u_sigmasigma is None or G_u_sigmasigma is None:
            huss, guss = split_mat(mat_guss, n_e)
            H_u_sigmasigma = H_u_sigmasigma if H_u_sigmasigma is not None else huss
            G_u_sigmasigma = G_u_sigmasigma if G_u_sigmasigma is not None else guss
        object.__setattr__(self, "H_u_sigmasigma", np.asarray(H_u_sigmasigma, dtype=float))
        object.__setattr__(self, "G_u_sigmasigma", np.asarray(G_u_sigmasigma, dtype=float))

        # Store user overrides if supplied
        object.__setattr__(self, "_user_g_xxx", g_xxx)
        object.__setattr__(self, "_user_g_uuu", g_uuu)

        # Full stacked tensors
        object.__setattr__(self, "ghx", mat_gx if mat_gx is not None and mat_gx.shape[0] == n_total else np.vstack([self.G, self.F]))
        object.__setattr__(self, "ghu", mat_gu if mat_gu is not None and mat_gu.shape[0] == n_total else np.vstack([self.N, self.L]))
        object.__setattr__(self, "ghxx", mat_gxx if mat_gxx is not None and mat_gxx.shape[0] == n_total else np.vstack([self.H_xx, self.G_xx]))
        object.__setattr__(self, "ghxu", mat_gxu if mat_gxu is not None and mat_gxu.shape[0] == n_total else np.vstack([self.H_xu, self.G_xu]))
        object.__setattr__(self, "ghuu", mat_guu if mat_guu is not None and mat_guu.shape[0] == n_total else np.vstack([self.H_uu, self.G_uu]))
        object.__setattr__(self, "ghs2", mat_gss if mat_gss is not None and len(mat_gss) == n_total else np.concatenate([self.H_sigmasigma, self.G_sigmasigma]))
        object.__setattr__(self, "ghxxx", mat_gxxx if mat_gxxx is not None and mat_gxxx.shape[0] == n_total else np.vstack([self.H_xxx, self.G_xxx]))
        object.__setattr__(self, "ghxxu", mat_gxxu if mat_gxxu is not None and mat_gxxu.shape[0] == n_total else np.vstack([self.H_xxu, self.G_xxu]))
        object.__setattr__(self, "ghxuu", mat_gxuu if mat_gxuu is not None and mat_gxuu.shape[0] == n_total else np.vstack([self.H_xuu, self.G_xuu]))
        object.__setattr__(self, "ghuuu", mat_guuu if mat_guuu is not None and mat_guuu.shape[0] == n_total else np.vstack([self.H_uuu, self.G_uuu]))
        object.__setattr__(self, "ghxss", mat_gxss if mat_gxss is not None and mat_gxss.shape[0] == n_total else np.vstack([self.H_x_sigmasigma, self.G_x_sigmasigma]))
        object.__setattr__(self, "ghuss", mat_guss if mat_guss is not None and mat_guss.shape[0] == n_total else np.vstack([self.H_u_sigmasigma, self.G_u_sigmasigma]))

        if shock_cov is None:
            object.__setattr__(self, "shock_cov", np.eye(n_e))
        else:
            cov = np.asarray(shock_cov, dtype=float)
            if cov.shape != (n_e, n_e):
                raise ValueError(f"shock_cov must be ({n_e}, {n_e}), got {cov.shape}")
            object.__setattr__(self, "shock_cov", cov)

        object.__setattr__(self, "params", params)
        object.__setattr__(self, "first_order", first_order)

    # -- sizes and names ---------------------------------------------------

    @property
    def n_states(self) -> int:
        return len(self.state_names)

    @property
    def n_controls(self) -> int:
        return len(self.control_names)

    @property
    def n_shocks(self) -> int:
        return len(self.shock_names)

    @property
    def variables(self) -> tuple[str, ...]:
        return tuple(self.variable_names or ())

    @property
    def states(self) -> tuple[str, ...]:
        return tuple(self.state_names)

    @property
    def controls(self) -> tuple[str, ...]:
        return tuple(self.control_names)

    @property
    def shocks(self) -> tuple[str, ...]:
        return tuple(self.shock_names)

    @property
    def eigenvalues(self) -> np.ndarray:
        return scipy.linalg.eigvals(self.G)

    @property
    def spectral_radius(self) -> float:
        """Spectral radius (maximum eigenvalue modulus) of state transition G."""
        return float(np.max(np.abs(self.eigenvalues))) if len(self.eigenvalues) > 0 else 0.0

    @property
    def is_stable(self) -> bool:
        return bool(np.all(np.abs(self.eigenvalues) < 1.0 - 1e-7))

    # Aliases matching interface contracts
    @property
    def g_x(self) -> np.ndarray:
        return self.ghx

    @property
    def g_u(self) -> np.ndarray:
        return self.ghu

    @property
    def g_xx(self) -> np.ndarray:
        return self.ghxx

    @property
    def g_xu(self) -> np.ndarray:
        return self.ghxu

    @property
    def g_uu(self) -> np.ndarray:
        return self.ghuu

    @property
    def g_ss(self) -> np.ndarray:
        return self.ghs2

    @property
    def g_xxx(self) -> np.ndarray:
        if self._user_g_xxx is not None:
            return self._user_g_xxx
        return self.ghxxx

    @property
    def g_xxu(self) -> np.ndarray:
        return self.ghxxu

    @property
    def g_xuu(self) -> np.ndarray:
        return self.ghxuu

    @property
    def g_uuu(self) -> np.ndarray:
        if self._user_g_uuu is not None:
            return self._user_g_uuu
        return self.ghuuu

    @property
    def g_x_ss(self) -> np.ndarray:
        return self.ghxss

    @property
    def g_u_ss(self) -> np.ndarray:
        return self.ghuss

    def _sigma_u(self, sigma: float, shock_cov: np.ndarray | None) -> np.ndarray:
        if isinstance(sigma, Mapping):
            raise TypeError(
                f"Order3PrunedSolution: sigma must be a single scalar scaling "
                f"parameter, got mapping {sigma!r}. Pass shock_cov= instead."
            )
        base_cov = self.shock_cov if shock_cov is None else np.asarray(shock_cov, dtype=float)
        return (float(sigma) ** 2) * base_cov

    def _shock_sd(self, sigma: float = 1.0) -> np.ndarray:
        cov = self._sigma_u(sigma, None)
        return np.sqrt(np.clip(np.diag(cov), 0.0, None))

    def _draw_shocks(self, total_t: int, sigma: float, seed: int) -> np.ndarray:
        rng = np.random.default_rng(seed)
        cov = self._sigma_u(sigma, None)
        return rng.multivariate_normal(np.zeros(self.n_shocks), cov, size=total_t)

    def _var_order(self) -> list[int]:
        ord_names = list(self.state_names) + list(self.control_names)
        return [ord_names.index(v) for v in (self.variable_names or ())]

    def decision_rules(self) -> Dynare3rdDR:
        """Decision rule representation matching Dynare's oo_.dr structure at 3rd order."""
        v_names = list(self.variable_names)
        s_names = list(self.state_names)
        e_names = list(self.shock_names)

        df_ghx = pd.DataFrame(self.ghx, index=v_names, columns=s_names)
        df_ghu = pd.DataFrame(self.ghu, index=v_names, columns=e_names)

        cols_xx = [f"{s1}_{s2}" for s1 in s_names for s2 in s_names]
        df_ghxx = pd.DataFrame(self.ghxx, index=v_names, columns=cols_xx)

        cols_xu = [f"{s}_{e}" for s in s_names for e in e_names]
        df_ghxu = pd.DataFrame(self.ghxu, index=v_names, columns=cols_xu)

        cols_uu = [f"{e1}_{e2}" for e1 in e_names for e2 in e_names]
        df_ghuu = pd.DataFrame(self.ghuu, index=v_names, columns=cols_uu)

        s_ghs2 = pd.Series(self.ghs2, index=v_names)

        cols_xxx = [f"{s1}_{s2}_{s3}" for s1 in s_names for s2 in s_names for s3 in s_names]
        df_ghxxx = pd.DataFrame(self.ghxxx, index=v_names, columns=cols_xxx)

        cols_xxu = [f"{s1}_{s2}_{e}" for s1 in s_names for s2 in s_names for e in e_names]
        df_ghxxu = pd.DataFrame(self.ghxxu, index=v_names, columns=cols_xxu)

        cols_xuu = [f"{s}_{e1}_{e2}" for s in s_names for e1 in e_names for e2 in e_names]
        df_ghxuu = pd.DataFrame(self.ghxuu, index=v_names, columns=cols_xuu)

        cols_uuu = [f"{e1}_{e2}_{e3}" for e1 in e_names for e2 in e_names for e3 in e_names]
        df_ghuuu = pd.DataFrame(self.ghuuu, index=v_names, columns=cols_uuu)

        df_ghxss = pd.DataFrame(self.ghxss, index=v_names, columns=s_names)
        df_ghuss = pd.DataFrame(self.ghuss, index=v_names, columns=e_names)

        s_ys = (
            self.steady_state.loc[v_names]
            if isinstance(self.steady_state, pd.Series)
            else pd.Series(self.steady_state, index=v_names)
        )

        return Dynare3rdDR(
            ghx=df_ghx,
            ghu=df_ghu,
            ghxx=df_ghxx,
            ghxu=df_ghxu,
            ghuu=df_ghuu,
            ghs2=s_ghs2,
            ghxxx=df_ghxxx,
            ghxxu=df_ghxxu,
            ghxuu=df_ghxuu,
            ghuuu=df_ghuuu,
            ghxss=df_ghxss,
            ghuss=df_ghuss,
            ys=s_ys,
            state_variables=self.state_names,
            variable_names=tuple(v_names),
            shock_names=self.shock_names,
        )

    def _pruned_path(
        self,
        e_mat: np.ndarray,
        sigma: float = 1.0,
        x0: np.ndarray | None = None,
    ) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
        """Iterate Andreasen et al. (2018) order-3 pruned system given innovations."""
        total_t = len(e_mat)
        n_x, n_y, n_e = self.n_states, self.n_controls, self.n_shocks

        x1 = np.zeros((total_t, n_x))
        x2 = np.zeros((total_t, n_x))
        x3 = np.zeros((total_t, n_x))
        y1 = np.zeros((total_t, n_y))
        y2 = np.zeros((total_t, n_y))
        y3 = np.zeros((total_t, n_y))

        sig2 = float(sigma) ** 2
        half_h_ss = 0.5 * self.H_sigmasigma * sig2
        half_g_ss = 0.5 * self.G_sigmasigma * sig2

        x1_prev = np.zeros(n_x) if x0 is None else np.asarray(x0, dtype=float)
        x2_prev = np.zeros(n_x)
        x3_prev = np.zeros(n_x)

        for t in range(total_t):
            e_t = e_mat[t]

            # Fast Kronecker outer products
            kron_x1_x1 = np.outer(x1_prev, x1_prev).ravel() if n_x > 0 else np.zeros(0)
            kron_x1_e = np.outer(x1_prev, e_t).ravel() if (n_x > 0 and n_e > 0) else np.zeros(0)
            kron_e_e = np.outer(e_t, e_t).ravel() if n_e > 0 else np.zeros(0)

            kron_x1_x2 = np.outer(x1_prev, x2_prev).ravel() if n_x > 0 else np.zeros(0)
            kron_x2_e = np.outer(x2_prev, e_t).ravel() if (n_x > 0 and n_e > 0) else np.zeros(0)

            kron_x1_3 = np.outer(kron_x1_x1, x1_prev).ravel() if n_x > 0 else np.zeros(0)
            kron_x1_2_e = np.outer(kron_x1_x1, e_t).ravel() if (n_x > 0 and n_e > 0) else np.zeros(0)
            kron_x1_e_2 = np.outer(x1_prev, kron_e_e).ravel() if (n_x > 0 and n_e > 0) else np.zeros(0)
            kron_e_3 = np.outer(kron_e_e, e_t).ravel() if n_e > 0 else np.zeros(0)

            # 1st order
            x1[t] = self.G @ x1_prev + self.N @ e_t
            y1[t] = self.F @ x1_prev + self.L @ e_t

            # 2nd order
            x2[t] = (
                self.G @ x2_prev
                + 0.5 * (self.H_xx @ kron_x1_x1)
                + self.H_xu @ kron_x1_e
                + 0.5 * (self.H_uu @ kron_e_e)
                + half_h_ss
            )
            y2[t] = (
                self.F @ x2_prev
                + 0.5 * (self.G_xx @ kron_x1_x1)
                + self.G_xu @ kron_x1_e
                + 0.5 * (self.G_uu @ kron_e_e)
                + half_g_ss
            )

            # 3rd order
            x3[t] = (
                self.G @ x3_prev
                + self.H_xx @ kron_x1_x2
                + self.H_xu @ kron_x2_e
                + (1.0 / 6.0) * (self.H_xxx @ kron_x1_3)
                + 0.5 * (self.H_xxu @ kron_x1_2_e)
                + 0.5 * (self.H_xuu @ kron_x1_e_2)
                + (1.0 / 6.0) * (self.H_uuu @ kron_e_3)
                + 0.5 * (self.H_x_sigmasigma @ x1_prev) * sig2
                + 0.5 * (self.H_u_sigmasigma @ e_t) * sig2
            )
            y3[t] = (
                self.F @ x3_prev
                + self.G_xx @ kron_x1_x2
                + self.G_xu @ kron_x2_e
                + (1.0 / 6.0) * (self.G_xxx @ kron_x1_3)
                + 0.5 * (self.G_xxu @ kron_x1_2_e)
                + 0.5 * (self.G_xuu @ kron_x1_e_2)
                + (1.0 / 6.0) * (self.G_uuu @ kron_e_3)
                + 0.5 * (self.G_x_sigmasigma @ x1_prev) * sig2
                + 0.5 * (self.G_u_sigmasigma @ e_t) * sig2
            )

            x1_prev = x1[t]
            x2_prev = x2[t]
            x3_prev = x3[t]

        return x1, x2, x3, y1, y2, y3

    def simulate(
        self,
        periods: int = 200,
        shocks: np.ndarray | None = None,
        sigma: float = 1.0,
        seed: int = 0,
        burn: int = 100,
    ) -> PrunedSimulationResult:
        """Simulate the 3rd-order pruned DSGE state space over periods + burn."""
        if shocks is not None:
            eps_arr = np.asarray(shocks, dtype=float)
            if eps_arr.ndim == 1:
                eps_arr = eps_arr.reshape(-1, 1)
            if eps_arr.shape[0] < periods + burn:
                burn = 0
            actual_periods = min(periods, eps_arr.shape[0] - burn)
            total_t = actual_periods + burn
            eps = eps_arr[:total_t]
        else:
            actual_periods = periods
            total_t = actual_periods + burn
            eps = self._draw_shocks(total_t, sigma, seed)

        x1, x2, x3, y1, y2, y3 = self._pruned_path(eps, sigma)

        idx = pd.RangeIndex(len(x1) - burn)
        s_names = list(self.state_names)
        c_names = list(self.control_names)
        e_names = list(self.shock_names)

        df_states = pd.DataFrame(x1[burn:] + x2[burn:] + x3[burn:], index=idx, columns=s_names)
        df_controls = pd.DataFrame(y1[burn:] + y2[burn:] + y3[burn:], index=idx, columns=c_names)
        df_x1 = pd.DataFrame(x1[burn:], index=idx, columns=s_names)
        df_x2 = pd.DataFrame(x2[burn:], index=idx, columns=s_names)
        df_x3 = pd.DataFrame(x3[burn:], index=idx, columns=s_names)
        df_y1 = pd.DataFrame(y1[burn:], index=idx, columns=c_names)
        df_y2 = pd.DataFrame(y2[burn:], index=idx, columns=c_names)
        df_y3 = pd.DataFrame(y3[burn:], index=idx, columns=c_names)
        df_shocks = pd.DataFrame(eps[burn:], index=idx, columns=e_names)

        return PrunedSimulationResult(
            states=df_states,
            controls=df_controls,
            states_1st=df_x1,
            states_2nd=df_x2,
            controls_1st=df_y1,
            controls_2nd=df_y2,
            shocks=df_shocks,
            states_3rd=df_x3,
            controls_3rd=df_y3,
        )

    def girf(
        self,
        shock: int | str = 0,
        size: float = 1.0,
        horizon: int = 20,
        sigma: float = 1.0,
        x0: np.ndarray | None = None,
        seed: int = 0,
        **kwargs: Any,
    ) -> pd.DataFrame:
        """Generalized Impulse Response Function (GIRF) under 3rd-order pruning.

        At third order, risk terms 0.5 * g_x_sigmasigma * x_t^{(1)} * sigma^2 and
        0.5 * g_u_sigmasigma * u_t * sigma^2 explicitly scale with sigma^2, so
        responses capture state- and volatility-dependent risk premia.
        """
        if isinstance(shock, str):
            if shock not in self.shock_names:
                raise ValueError(f"unknown shock {shock!r}; available: {self.shock_names}")
            s_idx = self.shock_names.index(shock)
        else:
            s_idx = int(shock)

        n_e = self.n_shocks
        t_steps = horizon + 1

        eps_base = np.zeros((t_steps, n_e))
        eps_shock = np.zeros((t_steps, n_e))
        eps_shock[0, s_idx] = size

        def run_path(e_mat: np.ndarray) -> np.ndarray:
            x1, x2, x3, y1, y2, y3 = self._pruned_path(e_mat, sigma, x0=x0)
            return np.hstack([x1 + x2 + x3, y1 + y2 + y3])

        diff = run_path(eps_shock) - run_path(eps_base)
        cols = list(self.state_names) + list(self.control_names)
        return pd.DataFrame(diff, index=pd.RangeIndex(t_steps, name="h"), columns=cols)

    def irf(
        self,
        shock: int | str = 0,
        horizon: int = 20,
        size: float = 1.0,
        sigma: float = 1.0,
        x0: np.ndarray | None = None,
    ) -> pd.DataFrame:
        """Impulse responses for the 3rd-order pruned model around steady state."""
        df = self.girf(shock=shock, size=size, horizon=horizon, sigma=sigma, x0=x0)
        vars_tuple = self.variable_names if self.variable_names is not None else (self.state_names + self.control_names)
        cols = [v for v in vars_tuple if v in df.columns]
        return df[cols]

    def plot(
        self,
        shock: int | str = 0,
        size: float = 1.0,
        horizon: int = 20,
        sigma: float = 1.0,
        variables: Sequence[str] | None = None,
        *,
        ax=None,
        title: str = "",
        ylabel: str = "GIRF Response",
    ):
        """Plot pruned Generalized Impulse Response Functions (GIRF)."""
        from ..plot import _new_ax

        df = self.girf(shock=shock, size=size, horizon=horizon, sigma=sigma)
        vars_to_plot = [v for v in (variables or df.columns) if v in df.columns]

        fig, ax = _new_ax(ax)
        for col in vars_to_plot:
            ax.plot(df.index, df[col], label=col, linewidth=1.2)

        ax.axhline(0.0, color="0.3", linewidth=0.6, linestyle=":")
        ax.set_xlabel("Horizon (h)")
        ax.set_ylabel(ylabel)
        if not title:
            s_name = shock if isinstance(shock, str) else self.shock_names[shock]
            title = f"Pruned DSGE GIRF to {s_name} shock (Order 3, σ={sigma})"
        ax.set_title(title)
        ax.legend(loc="best", frameon=False)
        return fig

    def ergodic_mean(
        self, sigma: float = 1.0, shock_cov: np.ndarray | None = None
    ) -> dict[str, pd.Series]:
        """Unconditional (ergodic) mean of the pruned third-order solution.

        This is E[x_t] and E[y_t] under the stationary distribution of the
        pruned process that :meth:`simulate` draws from -- the long-run average
        of a simulation -- not the risky steady state (see
        :meth:`risky_steady_state`). Under Gaussian innovations the third-order
        component has mean zero (every third-order forcing term of Andreasen,
        Fernández-Villaverde and Rubio-Ramírez 2018 is an odd moment of
        ``x^{(1)}`` and ``u``), so the mean is the second-order formula::

            E[x] = (I - G)^{-1} [0.5 H_σσ σ² + 0.5 H_xx vec(Ω) + 0.5 H_uu vec(σ²Σ_u)]
            E[y] = F E[x]       + 0.5 G_σσ σ² + 0.5 G_xx vec(Ω) + 0.5 G_uu vec(σ²Σ_u)

        with ``Ω = G Ω G' + N σ²Σ_u N'``. Only the first term, the risk term
        ``ghs2``, is the precautionary effect of anticipated risk; the other two
        are the curvature of the policy times the dispersion of states and
        innovations (a Jensen effect), so the ergodic mean can move in the
        opposite direction to the risky steady state. :meth:`risk_decomposition`
        reports the three parts. The mean equals the ``Mean`` column of
        :meth:`theoretical_moments` minus ``steady_state`` and Dynare 8's
        ``oo_.mean`` for ``stoch_simul(order=3, pruning)``.

        Parameters
        ----------
        sigma : float, default 1.0
            Perturbation scale σ: innovations are ``N(0, σ² Σ_u)`` and ``ghs2``
            enters as ``0.5 ghs2 σ²``, so every term is proportional to σ².
        shock_cov : np.ndarray, optional
            Innovation covariance Σ_u used for Ω and E[u ⊗ u]; defaults to
            ``self.shock_cov``, the covariance the solution was computed with.
            ``ghs2`` is not recomputed: it belongs to ``self.shock_cov``, so a
            ``shock_cov`` that is not a multiple of it mixes two models. Re-solve
            the model with the new covariance instead.

        Returns
        -------
        dict[str, pd.Series]
            ``{"states": ..., "controls": ...}``: ergodic-mean deviations from
            the deterministic steady state, in the units of the model's
            variables (levels for a model written in levels, log deviations
            for a model in logs). Add ``steady_state`` for levels.
        """
        xs, ys = _ergodic_mean_parts(self, sigma, shock_cov)["total"]
        return _mean_dict(self, xs, ys, "ergodic_mean")

    def stochastic_steady_state(
        self, sigma: float = 1.0, shock_cov: np.ndarray | None = None
    ) -> dict[str, pd.Series]:
        """Alias of :meth:`ergodic_mean`: returns the **ergodic mean**, not the risky steady state.

        Kept for backward compatibility; it returns exactly what
        :meth:`ergodic_mean` returns. Despite the name, this is the
        unconditional mean E[x], E[y] of the pruned solution, which adds the
        curvature of the policy times the dispersion of states and shocks to
        the risk term. The zero-shock fixed point that the literature calls the
        stochastic or risky steady state (Coeurdacier, Rey and Winant 2011) is
        :meth:`risky_steady_state`; in the RBC model of
        ``tests/fixtures/dynare_live/rbc.mod`` the two put capital on opposite
        sides of its deterministic steady state (+0.0785% against -0.0045%).
        See :meth:`ergodic_mean` for parameters and return value.
        """
        return self.ergodic_mean(sigma=sigma, shock_cov=shock_cov)

    def risky_steady_state(
        self,
        sigma: float = 1.0,
        *,
        pruned: bool = True,
        tol: float = 1e-13,
        maxiter: int = 50,
    ) -> dict[str, pd.Series]:
        """Risky (stochastic) steady state: the zero-shock fixed point of the policy under risk.

        The point where the economy settles when agents expect innovations
        ``N(0, σ² Σ_u)`` but every realized innovation is zero (the "risky
        steady state" of Coeurdacier, Rey and Winant 2011). With ``pruned=True``
        (default) it is the fixed point of the pruned recursion that
        :meth:`simulate` iterates. Without shocks ``x^{(1)}`` stays at zero, so
        every third-order forcing term vanishes and ``x^{(3)} = 0``; the fixed
        point is the second-order closed form::

            x_rss = (I - G)^{-1} 0.5 H_σσ σ²,     y_rss = F x_rss + 0.5 G_σσ σ²

        i.e. the limit of ``simulate(shocks=np.zeros(...), burn=0)``. Only the
        risk term ``ghs2`` enters, so this is the precautionary effect of
        anticipated risk; it is the ``risk`` column of
        :meth:`risk_decomposition`. It is not the ergodic mean
        (:meth:`ergodic_mean`, alias ``stochastic_steady_state``), which also
        contains curvature times dispersion and can have the opposite sign.

        With ``pruned=False`` it is the fixed point ``x = h(x, 0, σ)`` of the
        unpruned third-order policy ``h(x, 0, σ) = G x + 0.5 H_xx (x ⊗ x) +
        (1/6) H_xxx (x ⊗ x ⊗ x) + 0.5 H_σσ σ² + 0.5 H_xσσ x σ²`` (Gaussian
        innovations, so ``g_σσσ = 0``), found by Newton's method from the pruned
        value. The two differ by terms of order σ⁴.

        Parameters
        ----------
        sigma : float, default 1.0
            Perturbation scale σ; the risk terms enter as ``0.5 ghs2 σ²`` and
            ``0.5 ghxss x σ²``.
        pruned : bool, default True
            Fixed point of the pruned recursion (closed form) or of the
            unpruned approximated policy (Newton).
        tol, maxiter : float, int
            Newton tolerance on ``max|h(x) - x|`` and iteration cap
            (``pruned=False`` only); a failure raises ``RuntimeError``.

        Returns
        -------
        dict[str, pd.Series]
            ``{"states": ..., "controls": ...}``: deviations from the
            deterministic steady state, in the units of the model's variables.
            Add ``steady_state`` for levels.
        """
        return _risky_steady_state(self, sigma, pruned, 3, tol, maxiter)

    def risk_decomposition(
        self, sigma: float = 1.0, shock_cov: np.ndarray | None = None
    ) -> pd.DataFrame:
        """Split the pruned ergodic mean into the risk term and two curvature terms.

        Columns (deviations from the deterministic steady state, one row per
        variable in ``variable_names`` order)::

            risk             (I - G)^{-1} 0.5 H_σσ σ²           and F(.) + 0.5 G_σσ σ²
            state_curvature  (I - G)^{-1} 0.5 H_xx vec(Ω)       and F(.) + 0.5 G_xx vec(Ω)
            shock_curvature  (I - G)^{-1} 0.5 H_uu vec(σ²Σ_u)   and F(.) + 0.5 G_uu vec(σ²Σ_u)
            ergodic_mean     their sum, equal to :meth:`ergodic_mean`

        The third-order terms add nothing to the mean under Gaussian
        innovations. ``risk`` is the precautionary (anticipated-risk) effect and
        equals :meth:`risky_steady_state`; ``state_curvature`` and
        ``shock_curvature`` are Jensen effects of realized dispersion, present
        even when the risk term is zero. The three columns sum to
        ``ergodic_mean`` up to floating-point rounding. Parameters as in
        :meth:`ergodic_mean`.
        """
        return _risk_decomposition_frame(self, sigma, shock_cov)

    def _pruned_coefficients(self, sig2: float) -> dict[str, np.ndarray]:
        """Dynare-convention coefficients stacked [states; controls], risk terms times sigma^2."""
        n_x, n_y, n_e = self.n_states, self.n_controls, self.n_shocks

        def stack(top: Any, bottom: Any, cols: int) -> np.ndarray:
            return _stack_rows(top, bottom, n_x, n_y, cols)

        return {
            "gx": stack(self.G, self.F, n_x),
            "gu": stack(self.N, self.L, n_e),
            "gxx": stack(self.H_xx, self.G_xx, n_x**2),
            "gxu": stack(self.H_xu, self.G_xu, n_x * n_e),
            "guu": stack(self.H_uu, self.G_uu, n_e**2),
            "gss": sig2 * stack(self.H_sigmasigma, self.G_sigmasigma, 1).ravel(),
            "gxxx": stack(self.H_xxx, self.G_xxx, n_x**3),
            "gxxu": stack(self.H_xxu, self.G_xxu, n_x**2 * n_e),
            "gxuu": stack(self.H_xuu, self.G_xuu, n_x * n_e**2),
            "guuu": stack(self.H_uuu, self.G_uuu, n_e**3),
            "gxss": sig2 * stack(self.H_x_sigmasigma, self.G_x_sigmasigma, n_x),
            "guss": sig2 * stack(self.H_u_sigmasigma, self.G_u_sigmasigma, n_e),
        }

    def theoretical_moments(
        self,
        sigma: float = 1.0,
        shock_cov: np.ndarray | None = None,
        lags: int = 4,
        fevd_horizons: Sequence[int | None] = (1, 4, 8, 16, 32, None),
        *,
        max_state_dim: int | None = MAX_PRUNED_STATE_DIM,
    ) -> TheoreticalMomentsResult:
        """Exact unconditional moments of the pruned third-order solution.

        Mean, covariance, correlations and autocorrelations are the closed forms of
        the pruned state space of Andreasen, Fernandez-Villaverde and Rubio-Ramirez
        (2018). They assume Gaussian shocks and keep every third-order term, so they
        differ from first-order moments by terms of order sigma^4. The mean and
        covariance equal Dynare's "theoretical moments based on pruned state space"
        for ``stoch_simul(order=3, pruning)``; Dynare's autocorrelations at order 3
        omit the correlation of its innovation x (x) u (x) u with past shocks, and
        long simulations agree with the values here.

        Skewness and kurtosis have no closed form here and are NaN; use
        :meth:`ergodic_moments` for simulated estimates. ``fevd`` is the
        first-order decomposition (Dynare reports none at order 3).

        Parameters
        ----------
        sigma : float, default 1.0
            Perturbation scale: the innovation covariance is ``sigma**2`` times
            ``shock_cov`` and the risk terms ``ghs2``, ``ghxss`` and ``ghuss`` are
            scaled by ``sigma**2``.
        shock_cov : np.ndarray, optional
            Innovation covariance; defaults to the covariance the solution was
            computed with.
        lags : int, default 4
            Number of autocorrelation lags.
        fevd_horizons : sequence, default (1, 4, 8, 16, 32, None)
            Horizons of the first-order variance decomposition.
        max_state_dim : int or None, default 3000
            Largest augmented pruned state (3n + 2n^2 + n^3 for n states) to solve;
            larger models raise ``ValueError``. ``None`` removes the limit.
        """
        sigma_e = self._sigma_u(sigma, shock_cov)
        sig2 = float(sigma) ** 2
        mean_dev, gamma_0, gammas = pruned_order3_moments(
            self._pruned_coefficients(sig2), self.n_states, sigma_e, lags,
            max_state_dim=max_state_dim,
        )

        order = self._var_order()
        all_names = list(self.variable_names or ())
        cov_mat = gamma_0[np.ix_(order, order)]
        gammas = [gamma_k[np.ix_(order, order)] for gamma_k in gammas]

        ss_base = (
            self.steady_state
            if isinstance(self.steady_state, pd.Series)
            else pd.Series(0.0, index=all_names)
        )
        mean_vec = np.array([float(ss_base.get(v, 0.0)) for v in all_names]) + mean_dev[order]

        variances = np.diag(cov_mat)
        stds = np.sqrt(np.maximum(variances, 0.0))
        skewness = np.full(len(all_names), np.nan)
        kurtosis = np.full(len(all_names), np.nan)

        df_moments = pd.DataFrame(
            {
                "Mean": mean_vec,
                "Std.Dev.": stds,
                "Variance": variances,
                "Skewness": skewness,
                "Kurtosis": kurtosis,
            },
            index=all_names,
        )
        df_cov = pd.DataFrame(cov_mat, index=all_names, columns=all_names)
        df_corr, autocorr_mats = compute_autocorr_matrices(cov_mat, gammas, all_names)

        autocorr_cols = [f"Lag {k}" for k in range(1, lags + 1)]
        df_autocorr = pd.DataFrame(index=all_names, columns=autocorr_cols, dtype=float)
        for k, gamma_k in enumerate(gammas, start=1):
            with np.errstate(divide="ignore", invalid="ignore"):
                df_autocorr[f"Lag {k}"] = np.where(variances > 1e-14, np.diag(gamma_k) / variances, np.nan)

        sd = np.sqrt(np.clip(np.diag(sigma_e), 0.0, None))
        C = np.asarray(self.ghx, dtype=float)
        D = np.asarray(self.ghu, dtype=float)
        df_fevd = conditional_fevd(
            self.G, self.N, C, D, sd, list(fevd_horizons), all_names, list(self.shock_names)
        ) * 100.0

        return Order3TheoreticalMomentsResult(
            moments=df_moments,
            covariance=df_cov,
            correlation=df_corr,
            autocorr=df_autocorr,
            fevd=df_fevd,
            autocorr_matrices=autocorr_mats,
            skewness=pd.Series(skewness, index=all_names, name="Skewness"),
            kurtosis=pd.Series(kurtosis, index=all_names, name="Kurtosis"),
        )

    def ergodic_moments(
        self,
        sigma: float = 1.0,
        *,
        periods: int = 100_000,
        seed: int = 0,
        burn: int = 1_000,
    ) -> pd.DataFrame:
        """Unconditional moments of the pruned solution: exact mean and variance, simulated shape.

        ``Mean``, ``StdDev`` and ``Variance`` are the exact pruned-state-space values
        of :meth:`theoretical_moments`. Skewness and kurtosis have no closed form
        here, so ``Skewness`` and ``Kurtosis`` (Pearson, 3 for a normal
        distribution) are estimated from one pruned simulation of ``periods``
        draws after ``burn``; their Monte Carlo error shrinks like
        ``1/sqrt(periods)``.
        """
        theo = self.theoretical_moments(sigma=sigma, lags=1)
        all_names = list(self.variable_names or ())
        sim = self.simulate(periods=periods, sigma=sigma, seed=seed, burn=burn)
        draws = pd.concat([sim.states, sim.controls], axis=1)[all_names].to_numpy()
        dev = draws - draws.mean(axis=0)
        m2 = np.mean(dev**2, axis=0)
        with np.errstate(divide="ignore", invalid="ignore"):
            skewness = np.where(m2 > 0.0, np.mean(dev**3, axis=0) / m2**1.5, np.nan)
            kurtosis = np.where(m2 > 0.0, np.mean(dev**4, axis=0) / m2**2, np.nan)

        return pd.DataFrame(
            {
                "Mean": theo.moments["Mean"].to_numpy(),
                "StdDev": theo.moments["Std.Dev."].to_numpy(),
                "Variance": theo.moments["Variance"].to_numpy(),
                "Skewness": skewness,
                "Kurtosis": kurtosis,
            },
            index=all_names,
        )

    def stoch_simul(
        self,
        *,
        order: int = 3,
        irf: int = 40,
        periods: int = 0,
        sigma: float = 1.0,
        seed: int = 0,
        burn: int = 100,
        lags: int = 5,
    ) -> StochSimulResult:
        """Execute Dynare-compatible 3rd-order stoch_simul routine."""
        from ._results import StochSimulResult

        if order != 3:
            raise ValueError(f"Order3PrunedSolution only supports order=3, got order={order}")

        dr = self.decision_rules()
        if pruned_state_dim(self.n_states) <= MAX_PRUNED_STATE_DIM:
            theo = self.theoretical_moments(sigma=sigma, lags=lags)
        else:
            import warnings

            warnings.warn(
                f"stoch_simul(order=3): {self.n_states} predetermined states give a pruned state "
                f"of size {pruned_state_dim(self.n_states)}; exact theoretical moments skipped. "
                "Pass periods > 0 for simulated moments, or call "
                "theoretical_moments(max_state_dim=None).",
                RuntimeWarning,
                stacklevel=2,
            )
            theo = None

        vars_tuple = self.variable_names if self.variable_names is not None else (self.state_names + self.control_names)
        sd = self._shock_sd(sigma)
        irfs: dict[str, pd.Series] = {}
        if irf > 0:
            for j, sh in enumerate(self.shock_names):
                df_irf = self.irf(shock=sh, horizon=irf, size=float(sd[j]), sigma=sigma)
                for v in vars_tuple:
                    irfs[f"{v}_{sh}"] = df_irf[v]

        sim_moments = None
        if periods > 0:
            sim_res = self.simulate(periods=periods, sigma=sigma, seed=seed, burn=burn)
            sim_df = pd.concat([sim_res.states, sim_res.controls], axis=1)[list(vars_tuple)]
            ss_level = (
                self.steady_state.reindex(list(vars_tuple)).astype(float).fillna(0.0)
                if isinstance(self.steady_state, pd.Series)
                else pd.Series(0.0, index=list(vars_tuple))
            )
            sim_moments = pd.DataFrame(
                {
                    "Mean": sim_df.mean(axis=0) + ss_level,
                    "Std.Dev.": sim_df.std(axis=0),
                    "Variance": sim_df.var(axis=0),
                    "Skewness": sim_df.skew(axis=0),
                    "Kurtosis": sim_df.kurtosis(axis=0),
                },
                index=list(vars_tuple),
            )

        return StochSimulResult(
            dr=dr,
            theoretical_moments=theo,
            simulated_moments=sim_moments,
            irfs=irfs,
            order=3,
            variable_names=vars_tuple,
            shock_names=self.shock_names,
        )

    def summary(self) -> str:
        """Publication-grade summary of the order-3 pruned DSGE solution."""
        lines = [
            "Order-3 Pruned DSGE Solution (Andreasen et al. 2018)",
            "=" * 72,
            f"Predetermined States ({self.n_states}) : {list(self.state_names)}",
            f"Control Variables    ({self.n_controls}) : {list(self.control_names)}",
            f"Exogenous Shocks     ({self.n_shocks}) : {list(self.shock_names)}",
            f"Spectral Radius of G : {max(abs(self.eigenvalues)):.4f}" if self.n_states > 0 else "Spectral Radius : 0.0",
            f"Stability Condition  : {'Stable (|λ| < 1)' if self.is_stable else 'Unstable'}",
            "-" * 72,
            "Decision Rule Tensors:",
            f"  ghx   (N x n_x)     : ({len(self.variable_names)}, {self.n_states})",
            f"  ghu   (N x n_e)     : ({len(self.variable_names)}, {self.n_shocks})",
            f"  ghxx  (N x n_x^2)   : ({len(self.variable_names)}, {self.n_states**2})",
            f"  ghxu  (N x n_x*n_e) : ({len(self.variable_names)}, {self.n_states * self.n_shocks})",
            f"  ghuu  (N x n_e^2)   : ({len(self.variable_names)}, {self.n_shocks**2})",
            f"  ghs2  (N,)          : ({len(self.variable_names)},)",
            f"  ghxxx (N x n_x^3)   : ({len(self.variable_names)}, {self.n_states**3})",
            f"  ghxxu (N x n_x^2*n_e): ({len(self.variable_names)}, {self.n_states**2 * self.n_shocks})",
            f"  ghxuu (N x n_x*n_e^2): ({len(self.variable_names)}, {self.n_states * self.n_shocks**2})",
            f"  ghuuu (N x n_e^3)   : ({len(self.variable_names)}, {self.n_shocks**3})",
            f"  ghxss (N x n_x)     : ({len(self.variable_names)}, {self.n_states})",
            f"  ghuss (N x n_e)     : ({len(self.variable_names)}, {self.n_shocks})",
            "=" * 72,
        ]
        return "\n".join(lines)

    def to_markdown(self, **kwargs) -> str:
        """Render decision rules summary as Markdown."""
        from puremacro.reports import _df_to_markdown

        return _df_to_markdown(self.decision_rules().to_frame(), **kwargs)

    def to_latex(self, **kwargs) -> str:
        """Render decision rules summary as LaTeX tabular."""
        from puremacro.reports import _df_to_latex

        return _df_to_latex(self.decision_rules().to_frame(), **kwargs)

    def to_typst(self, **kwargs) -> str:
        """Render decision rules summary as Typst table."""
        from puremacro.reports import _df_to_typst

        return _df_to_typst(self.decision_rules().to_frame(), **kwargs)


def canonical_growth_3rd_order(
    alpha: float = 0.33,
    beta: float = 0.99,
    delta: float = 0.025,
    sigma_pref: float = 1.0,
    rho: float = 0.95,
    sigma_eps: float = 0.01,
) -> Order3PrunedSolution:
    """Solve the canonical one-sector growth model to third order with pruning (Andreasen et al. 2018).

    Returns
    -------
    Order3PrunedSolution
        Third-order pruned solution for states ('k', 'z') and control ('c').
    """
    from puremacro.dsge.dynare import solve_dynare_3rd_order

    r_ss = 1.0 / beta - (1.0 - delta)
    k_ss = (r_ss / alpha) ** (1.0 / (alpha - 1.0))
    c_ss = k_ss**alpha - delta * k_ss

    def growth(lead, curr, lag, shocks, p):
        return [
            np.exp(-p.sigma_pref * curr.c)
            - p.beta * np.exp(-p.sigma_pref * lead.c)
            * (p.alpha * np.exp(lead.z) * np.exp((p.alpha - 1.0) * curr.k) + 1.0 - p.delta),
            np.exp(curr.c) + np.exp(curr.k)
            - np.exp(curr.z) * np.exp(p.alpha * lag.k) - (1.0 - p.delta) * np.exp(lag.k),
            curr.z - p.rho * lag.z - p.sigma_eps * shocks.eps,
        ]

    return solve_dynare_3rd_order(
        growth,
        variables=["k", "z", "c"],
        shocks=["eps"],
        params=dict(alpha=alpha, beta=beta, delta=delta, sigma_pref=sigma_pref,
                    rho=rho, sigma_eps=sigma_eps),
        steady_state=dict(k=float(np.log(k_ss)), z=0.0, c=float(np.log(c_ss))),
        states=["k", "z"],
        shock_cov=np.eye(1),
    )
