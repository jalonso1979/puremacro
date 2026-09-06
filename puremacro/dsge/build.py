"""Write a DSGE model as a Python function; get IRFs back.

:func:`puremacro.dsge.klein_solve` already solves linear rational-
expectations models on an iPad with no Dynare and no compiler — but it
takes the matrices ``A``, ``B``, ``C``, and getting those means
differentiating the equilibrium conditions by hand. That derivation is
the part people get wrong, and it is exactly the part a tablet is worst
at: no algebra software, no MATLAB, no patience.

This module removes it. Write the equilibrium conditions as they appear
in the paper, give a steady state (or a guess), and the Jacobians come
out by **complex-step differentiation** — machine-precision derivatives
from one function evaluation per argument, no step-size tuning, no
cancellation error::

    def eqs(xp, x, e, p):
        # xp = t+1, x = t, e = shocks, p = params
        return [
            x.c**-p.sigma - p.beta * xp.c**-p.sigma
                * (p.alpha * xp.z * xp.k**(p.alpha - 1) + 1 - p.delta),
            x.c + xp.k - x.z * x.k**p.alpha - (1 - p.delta) * x.k,
            xp.z - p.rho * x.z - e.eps,
        ]

    m = dsge.build(eqs, variables=["c", "k", "z"], states=["k", "z"],
                      shocks=["eps"], params=dict(alpha=.33, beta=.99,
                      delta=.025, sigma=1.0, rho=.95), guess=dict(c=1, k=10, z=1))
    m.irf("eps", horizon=20)      # DataFrame: horizons x variables

Pure numpy + scipy, so it runs wherever the rest of the estimator core
runs.

Complex-step: the restriction
-----------------------------
The derivative comes from ``Im f(x + ih) / h`` with ``h = 1e-20``, which
is exact to machine precision *provided the residual function is
analytic*. In practice that rules out four things: ``abs``, ``min`` /
``max``, comparisons that branch on the perturbed value, and any
``float()`` / ``np.real`` cast that discards the imaginary part. Models
with occasionally-binding constraints violate this by construction —
pass ``method="central"`` for those and accept ~1e-8 accuracy instead of
~1e-15. :func:`build` checks for the failure mode (an all-zero Jacobian
column where one is not expected) and says so rather than silently
returning a wrong solution.

Timing convention
-----------------
Equations are ``E_t f(z_{t+1}, z_t, u_t) = 0``, matching
:func:`~puremacro.dsge.klein.klein_solve`. An exogenous process written
the usual way, ``z' = rho*z + eps``, has its innovation move the *state*
into the next period, so every state is still at zero in the ``h=0`` row
of an IRF and jumps at ``h=1``.

Forward-looking *controls* are a different matter: they can and usually
do move at ``h=0``, because the innovation is known at ``t`` and they
depend on expectations of ``t+1``. In the three-equation New Keynesian
model, a demand shock leaves the natural rate at zero on impact while
the output gap and inflation jump immediately — agents are reacting to
the higher natural rate they already know is coming. That jump is
Klein's ``L`` loading. Whether a control moves at ``h=0`` is therefore a
property of the model, not of the convention; only the states are
guaranteed to be zero there.

Shocks written directly into a control equation (an i.i.d. policy shock
in a Taylor rule) move that equation's variables at ``h=0`` too.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Callable, Mapping, Sequence

import numpy as np
import pandas as pd
import scipy.linalg
import scipy.optimize

from puremacro.dsge.klein import KleinSolution, klein_solve
from puremacro.dsge._results import DynareDR, TheoreticalMomentsResult, StochSimulResult
from puremacro.dsge._moments import conditional_fevd, first_order_moments

__all__ = [
    "ModelError",
    "SteadyStateError",
    "LinearModel",
    "build",
    "DynareDR",
    "TheoreticalMomentsResult",
    "StochSimulResult",
]

# Complex-step size. Any value small enough that h**2 underflows relative
# to the function value works identically — there is no truncation/
# cancellation trade-off to tune, which is the whole point.
_CSTEP = 1e-20

# Central-difference step, used only by method="central".
_FDSTEP = 1e-6


class ModelError(ValueError):
    """The model as declared is inconsistent (names, shapes, equation count)."""


class SteadyStateError(ValueError):
    """The steady state could not be found, or the one supplied is not one."""


class _Vec:
    """Named access to one time-slice of the state vector.

    Supports ``x.c``, ``x["c"]``, ``x[0]`` and tuple-unpacking, so a
    residual function can be written in whichever style reads best.
    """

    __slots__ = ("_names", "_values", "_index", "_what")

    def __init__(self, names: Sequence[str], values, what: str = "variable"):
        object.__setattr__(self, "_names", tuple(names))
        object.__setattr__(self, "_values", values)
        object.__setattr__(self, "_index", {n: i for i, n in enumerate(names)})
        object.__setattr__(self, "_what", what)

    def __getattr__(self, name):
        try:
            return self._values[self._index[name]]
        except KeyError:
            raise AttributeError(
                f"no {self._what} named {name!r}; declared: "
                f"{list(self._names)}"
            ) from None

    def __getitem__(self, key):
        if isinstance(key, str):
            try:
                return self._values[self._index[key]]
            except KeyError:
                raise KeyError(
                    f"no {self._what} named {key!r}; declared: "
                    f"{list(self._names)}"
                ) from None
        return self._values[key]

    def __iter__(self):
        return iter(self._values)

    def __len__(self):
        return len(self._names)

    def __repr__(self):
        body = ", ".join(f"{n}={v}" for n, v in zip(self._names, self._values))
        return f"<{self._what}s {body}>"


@dataclass(frozen=True)
class LinearModel:
    """A solved first-order approximation of a nonlinear model.

    Attributes
    ----------
    variables : tuple[str, ...]
        Variable names, in the order the residual function sees them.
    states : tuple[str, ...]
        Predetermined variables (Klein's ``x``).
    controls : tuple[str, ...]
        Forward-looking variables (Klein's ``y``).
    shocks : tuple[str, ...]
        Innovation names.
    steady_state : pandas.Series
        Deterministic steady state, indexed by variable name.
    units : dict[str, str]
        Per-variable ``"log"`` or ``"level"``: whether that variable's
        deviations are log deviations (interpretable as fractions of
        steady state) or level deviations. Variables with a non-positive
        steady state cannot be log-linearised and fall back to levels.
        Models built from Dynare lead-lag equations (``build_dynare``,
        ``load_mod``) are approximated in levels, so every entry is
        ``"level"`` there.
    solution : KleinSolution
        The underlying QZ solution: ``G`` (state transition), ``F``
        (policy), ``N`` / ``L`` (shock loadings).
    A, B, C : ndarray
        The Klein-form matrices the Jacobians produced, kept for
        inspection and for callers who want to re-solve by hand.
    method : str
        ``"complex"`` or ``"central"`` — how the Jacobians were taken.
    residual_norm : float
        ``max |f(ss, ss, 0)|``; how exactly the steady state solves the
        model.
    timing : {"klein", "dynare"}
        How the Klein state vector relates to the reported variables.
        ``"klein"`` (``build``): the state vector *is* the reported state
        at ``t`` and moves at ``t+1``, so states are zero in the ``h=0``
        row of an IRF. ``"dynare"`` (``build_dynare`` / ``load_mod``): the
        Klein state is the *lagged* state ``s_{t-1}`` and the reported
        state ``s_t = G s_{t-1} + N u_t`` responds on impact, exactly as
        in Dynare. In both cases ``y_t = F x_t + L u_t`` with ``x_t`` the
        predetermined vector at the start of ``t``.
    """

    variables: tuple
    states: tuple
    controls: tuple
    shocks: tuple
    steady_state: pd.Series
    units: dict
    solution: KleinSolution
    A: np.ndarray
    B: np.ndarray
    C: np.ndarray
    method: str
    residual_norm: float
    _dynare_equations: Callable | None = None
    _params: dict | None = None
    _A_plus: np.ndarray | None = None
    _A_0: np.ndarray | None = None
    _A_minus: np.ndarray | None = None
    _B_u: np.ndarray | None = None
    _shock_cov: np.ndarray | None = None
    timing: str = "klein"

    def __post_init__(self):
        if self.timing not in ("klein", "dynare"):
            raise ValueError(
                f"timing must be 'klein' or 'dynare', got {self.timing!r}"
            )

    # -- inspection ----------------------------------------------------

    @property
    def n_states(self) -> int:
        return len(self.states)

    @property
    def n_controls(self) -> int:
        return len(self.controls)

    @property
    def eigenvalues(self) -> np.ndarray:
        """Generalised eigenvalues, sorted by modulus (Blanchard-Kahn check)."""
        return self.solution.eigenvalues

    @property
    def is_determinate(self) -> bool:
        """True when the QZ solve found a unique stable solution (``eu == (1, 1)``)."""
        return tuple(self.solution.eu) == (1, 1)

    def _require_solution(self, what: str) -> None:
        """Refuse to report decision rules for a model without a unique
        stable solution — zero matrices are not an answer."""
        if not self.is_determinate:
            eu = tuple(self.solution.eu)
            verdict = (
                "no stable solution (Blanchard-Kahn violated)" if eu[0] == 0
                else "indeterminacy (multiple stable solutions)"
            )
            raise ModelError(
                f"{what} needs a unique stable solution, but the Blanchard-Kahn "
                f"check found {verdict}; eu={eu}. Re-solve with strict=True to see "
                "the eigenvalue diagnostics, or fix the model."
            )

    def policy(self) -> pd.DataFrame:
        """The decision rules as a labelled table.

        Rows are variables, columns are states: entry ``(i, j)`` is the
        response of variable ``i`` to a one-unit deviation of state
        ``j``. States map to themselves through ``G``, controls through
        ``F``. Under Dynare timing the column state is the *lagged*
        state ``s_{t-1}`` (this is then the ``ghx`` table).
        """
        block = np.vstack([self.solution.G, self.solution.F])
        frame = pd.DataFrame(
            block, index=list(self.states) + list(self.controls),
            columns=list(self.states),
        )
        return frame.loc[list(self.variables)]

    # -- shock covariance helpers --------------------------------------

    def _shock_sd(self, sigma, *, missing: float = 1.0) -> np.ndarray:
        """Innovation standard deviations implied by a ``sigma`` argument.

        ``None`` means the declared shock covariance (``shocks`` block of a
        .mod file, or ``shock_cov=``) when there is one, else 1.0 per shock.
        """
        n_e = len(self.shocks)
        if sigma is None:
            if self._shock_cov is not None:
                return np.sqrt(np.clip(np.diag(np.asarray(self._shock_cov, dtype=float)), 0.0, None))
            return np.ones(n_e)
        if isinstance(sigma, Mapping):
            return np.array([float(sigma.get(s, missing)) for s in self.shocks])
        return np.full(n_e, float(sigma))

    def _shock_covariance(self, sigma) -> np.ndarray:
        """Innovation covariance matrix implied by ``sigma`` (full declared
        matrix, correlations included, when ``sigma`` is None)."""
        if sigma is None and self._shock_cov is not None:
            cov = np.asarray(self._shock_cov, dtype=float)
            return 0.5 * (cov + cov.T)
        sd = self._shock_sd(sigma)
        return np.diag(sd ** 2)

    def _reported_loadings(self) -> tuple[np.ndarray, np.ndarray]:
        """``(M_x, M_u)`` such that the reported vector ``[states; controls]``
        at ``t`` is ``M_x x_t + M_u u_t`` with ``x_t`` the Klein state."""
        G, F, N, L = (self.solution.G, self.solution.F,
                      self.solution.N, self.solution.L)
        if self.timing == "dynare":
            return np.vstack([G, F]), np.vstack([N, L])
        n_s, n_e = self.n_states, len(self.shocks)
        return np.vstack([np.eye(n_s), F]), np.vstack([np.zeros((n_s, n_e)), L])

    def _dynare_companion(self) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
        """``(A, B, C, D)`` with ``s_t = A s_{t-1} + B u_t`` and
        ``v_t = C s_{t-1} + D u_t`` for the variables in declaration order
        (Dynare's ``ghx`` / ``ghu`` timing)."""
        G, F, N, L = (self.solution.G, self.solution.F,
                      self.solution.N, self.solution.L)
        order_vars = list(self.states) + list(self.controls)
        idx = [order_vars.index(v) for v in self.variables]
        C = np.vstack([G, F])[idx]
        D = np.vstack([N, L])[idx]
        return G, N, C, D

    def decision_rules(self) -> DynareDR:
        """Decision rule representation matching Dynare's oo_.dr structure.

        First-order approximation around steady state, in Dynare's timing:
            y_t = ys + ghx * (x_{t-1} - xs) + ghu * u_t

        where ``x_{t-1}`` are the lagged states. The state rows are
        ``(G, N)`` and the control rows ``(F, L)`` of the Klein solution,
        since ``s_t = G s_{t-1} + N u_t`` and ``c_t = F s_{t-1} + L u_t``.
        For a model built with :func:`build` (Klein timing) the state rows
        describe the end-of-period value ``x_{t+1} = G x_t + N u_t``, which
        is what Dynare calls the time-``t`` state.

        Returns
        -------
        DynareDR
            Container holding ghx, ghu, steady states, and variable labels.
        """
        self._require_solution("decision_rules()")
        G, F, N, L = (self.solution.G, self.solution.F,
                      self.solution.N, self.solution.L)
        ghx_block = np.vstack([G, F])
        ghu_block = np.vstack([N, L])

        order_vars = list(self.states) + list(self.controls)
        df_ghx = pd.DataFrame(
            ghx_block, index=order_vars, columns=list(self.states)
        ).loc[list(self.variables)]
        df_ghu = pd.DataFrame(
            ghu_block, index=order_vars, columns=list(self.shocks)
        ).loc[list(self.variables)]

        return DynareDR(
            ghx=df_ghx,
            ghu=df_ghu,
            ys=self.steady_state.loc[list(self.variables)],
            state_variables=self.states,
            variable_names=self.variables,
            shock_names=self.shocks,
        )

    @property
    def dynare_dr(self) -> DynareDR:
        """Dynare decision rules property alias."""
        return self.decision_rules()

    @property
    def oo_dr(self) -> DynareDR:
        """Dynare oo_.dr alias for direct MATLAB/Dynare parity."""
        return self.decision_rules()

    def theoretical_moments(
        self,
        *,
        sigma: float | Mapping[str, float] | None = None,
        lags: int = 5,
        fevd_horizons: Sequence[int | None] = (1, 4, 8, 16, 32, None),
    ) -> TheoreticalMomentsResult:
        """Calculate analytical theoretical moments matching Dynare's stoch_simul.

        Solves the discrete Lyapunov equation for unconditional stationary moments,
        cross-correlations, autocorrelations, and forecast error variance decomposition.
        The moments are those of the variables as reported by :meth:`irf` and
        :meth:`simulate` (Dynare timing for ``build_dynare`` / ``load_mod``
        models), so a long :meth:`simulate` run reproduces them.

        Parameters
        ----------
        sigma : float | Mapping[str, float], optional
            Shock standard deviations. Default: the covariance declared in the
            .mod ``shocks`` block (or ``shock_cov=``) when present, else 1.0 for
            each shock.
        lags : int, default 5
            Number of autocorrelation lags to evaluate.
        fevd_horizons : Sequence[int | None], default (1, 4, 8, 16, 32, None)
            Forecast horizons for variance decomposition. None represents
            asymptotic infinity (unconditional variance share).

        Returns
        -------
        TheoreticalMomentsResult
            Container with moments, covariance, correlation, autocorrelation, and FEVD.
        """
        self._require_solution("theoretical_moments()")
        g_eigs = np.abs(scipy.linalg.eigvals(self.solution.G))
        if np.any(g_eigs >= 1.0 - 1e-7):
            bad = g_eigs[g_eigs >= 1.0 - 1e-7]
            raise ValueError(
                f"State transition matrix G has non-stationary eigenvalues (|λ| >= 1.0: {bad}); "
                "unconditional stationary moments do not exist."
            )

        sigma_u = self._shock_covariance(sigma)
        G, N = self.solution.G, self.solution.N
        M_x, M_u = self._reported_loadings()
        _, gamma_0, gammas = first_order_moments(G, N, M_x, M_u, sigma_u, lags)

        order_vars = list(self.states) + list(self.controls)
        cov_df = pd.DataFrame(
            gamma_0, index=order_vars, columns=order_vars
        ).loc[list(self.variables), list(self.variables)]

        variances = np.diag(cov_df.to_numpy())
        stds = np.sqrt(np.maximum(variances, 0.0))
        means = np.array([float(self.steady_state[v]) for v in self.variables])

        df_moments = pd.DataFrame(
            {"Mean": means, "Std.Dev.": stds, "Variance": variances},
            index=list(self.variables),
        )

        std_outer = np.outer(stds, stds)
        std_outer[std_outer == 0.0] = np.nan
        corr_mat = cov_df.to_numpy() / std_outer
        np.fill_diagonal(corr_mat, 1.0)
        df_corr = pd.DataFrame(
            corr_mat, index=list(self.variables), columns=list(self.variables)
        )

        # Autocorrelations: diag(Gamma_k) / diag(Gamma_0)
        autocorr_cols = [f"Lag {k}" for k in range(1, lags + 1)]
        df_autocorr = pd.DataFrame(
            index=list(self.variables), columns=autocorr_cols, dtype=float
        )
        for k, gamma_k in enumerate(gammas, start=1):
            df_gamma_k = pd.DataFrame(
                gamma_k, index=order_vars, columns=order_vars
            ).loc[list(self.variables), list(self.variables)]
            diag_gamma = np.diag(df_gamma_k.to_numpy())
            with np.errstate(divide="ignore", invalid="ignore"):
                rho_k = np.where(variances > 1e-14, diag_gamma / variances, np.nan)
            df_autocorr[f"Lag {k}"] = rho_k

        # Variance Decomposition
        df_fevd = self.fevd(horizons=fevd_horizons, sigma=sigma)

        return TheoreticalMomentsResult(
            moments=df_moments,
            covariance=cov_df,
            correlation=df_corr,
            autocorr=df_autocorr,
            fevd=df_fevd,
        )

    def fevd(
        self,
        horizons: Sequence[int | None] = (1, 4, 8, 16, 32, None),
        sigma: float | Mapping[str, float] | None = None,
    ) -> pd.DataFrame:
        """Forecast error variance decomposition (FEVD) shares (in percent).

        Dynare's conditional variance decomposition: the ``h``-step-ahead
        forecast error ``sum_{j=0}^{h-1} Psi_j u_{t+h-j}`` built from the
        ``ghx`` / ``ghu`` decision rules (states in Dynare's end-of-period
        timing). Identical, up to the factor 100, to
        :func:`puremacro.dsge.compute_fevd`.

        Parameters
        ----------
        horizons : Sequence[int | None], default (1, 4, 8, 16, 32, None)
            Evaluation horizons. None represents asymptotic infinity.
        sigma : float | Mapping[str, float], optional
            Shock standard deviations (default: declared shock covariance,
            else 1.0).

        Returns
        -------
        pd.DataFrame
            MultiIndex DataFrame [Variable, Horizon] with percentage shares for
            each shock. Rows with zero forecast-error variance are NaN.
        """
        self._require_solution("fevd()")
        sd = self._shock_sd(sigma)
        A, B, C, D = self._dynare_companion()
        table = conditional_fevd(
            A, B, C, D, sd, list(horizons), list(self.variables), list(self.shocks)
        )
        return table * 100.0

    def fevd_result(
        self,
        horizons: Sequence[int | None] | None = None,
        sigma: float | Mapping[str, float] | None = None,
    ):
        """Compute Forecast Error Variance Decomposition as a FEVDResult."""
        from .decomposition import compute_fevd
        return compute_fevd(self, horizons=horizons, sigma=sigma)

    def shock_decomposition(
        self,
        data: pd.DataFrame,
        initial_state: np.ndarray | None = None,
        sigma: float | Mapping[str, float] | None = None,
    ):
        """Compute Historical Shock Decomposition as a ShockDecompResult."""
        from .decomposition import compute_shock_decomposition
        return compute_shock_decomposition(self, data, initial_state=initial_state, sigma=sigma)

    # -- simulation ----------------------------------------------------

    def _paths(self, horizon: int, impulse: np.ndarray) -> np.ndarray:
        """MA response of the reported ``[states; controls]`` vector to
        ``impulse`` at t=0, one row per date ``0..horizon``.

        Under Klein timing the state rows are zero at ``h=0`` (the
        innovation moves the state into ``t+1``); under Dynare timing the
        states respond on impact and every row is dated the same period.
        """
        G, N = self.solution.G, self.solution.N
        M_x, M_u = self._reported_loadings()
        out = np.zeros((horizon + 1, self.n_states + self.n_controls))
        out[0] = M_u @ impulse
        x = N @ impulse
        for h in range(1, horizon + 1):
            out[h] = M_x @ x
            x = G @ x
        return out

    def irf(self, shock: str, horizon: int = 20, size: float = 1.0) -> pd.DataFrame:
        """Impulse responses to a one-time ``size`` innovation in ``shock``.

        Parameters
        ----------
        shock : str
            Name of the innovation.
        horizon : int, default 20
            Periods after impact.
        size : float, default 1.0
            Innovation size, in the units the model's equations use.

        Returns
        -------
        pandas.DataFrame
            Indexed by horizon ``0..horizon``, one column per variable in
            declaration order. Values are log deviations for variables
            with ``units[name] == "log"`` (multiply by 100 for percent)
            and level deviations otherwise. Row ``h`` holds every
            variable at date ``h``: under Klein timing (:func:`build`) the
            states are zero in the ``h=0`` row for a standard AR(1)
            driving process, while under Dynare timing
            (:func:`build_dynare`, :func:`load_mod`) states and controls
            both respond on impact, as in Dynare's ``oo_.irfs``.
        """
        if shock not in self.shocks:
            raise ModelError(
                f"no shock named {shock!r}; declared: {list(self.shocks)}"
            )
        if horizon < 0:
            raise ValueError(f"horizon must be non-negative, got {horizon}")
        self._require_solution("irf()")
        impulse = np.zeros(len(self.shocks))
        impulse[self.shocks.index(shock)] = size
        paths = self._paths(horizon, impulse)
        frame = pd.DataFrame(
            paths, columns=list(self.states) + list(self.controls),
        )
        frame.index.name = "h"
        return frame[list(self.variables)]

    def plot(
        self,
        shock: str | None = None,
        horizon: int = 20,
        size: float = 1.0,
        variables: Sequence[str] | None = None,
        *,
        ax=None,
        title: str = "",
        ylabel: str = "Response",
    ):
        """Plot impulse responses for the DSGE model."""
        from ..plot import _new_ax

        if shock is None:
            if not self.shocks:
                raise ValueError("Model has no shocks declared to plot.")
            shock = self.shocks[0]

        df = self.irf(shock=shock, horizon=horizon, size=size)
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
            title = f"DSGE IRF to {shock} shock"
        ax.set_title(title)
        ax.legend(loc="best", frameon=False)
        return fig

    def simulate(self, periods: int = 200, *, sigma=None, seed: int = 0,
                 burn: int = 100) -> pd.DataFrame:
        """Simulate the model with i.i.d. Gaussian innovations.

        Parameters
        ----------
        periods : int, default 200
            Periods returned, after ``burn``.
        sigma : float | Mapping[str, float], optional
            Innovation standard deviations. A scalar applies to every
            shock; a mapping sets them by name (missing shocks get 0).
            Default: the declared shock covariance (``shocks`` block /
            ``shock_cov=``, correlations included) when there is one,
            else 1.0 for every shock.
        seed : int, default 0
            Seed for ``numpy.random.default_rng``.
        burn : int, default 100
            Discarded initial periods.

        Returns
        -------
        pandas.DataFrame
            ``periods`` rows, one column per variable, in the same
            deviation units and timing as :meth:`irf`: row ``t`` holds
            every variable at date ``t``.
        """
        self._require_solution("simulate()")
        n_e = len(self.shocks)
        rng = np.random.default_rng(seed)
        total = periods + burn
        if sigma is None and self._shock_cov is not None:
            cov = self._shock_covariance(None)
            shocks = rng.multivariate_normal(np.zeros(n_e), cov, size=total, method="cholesky") \
                if n_e else np.zeros((total, 0))
        else:
            sd = self._shock_sd(sigma, missing=0.0)
            shocks = rng.standard_normal((total, n_e)) * sd

        G, N = self.solution.G, self.solution.N
        M_x, M_u = self._reported_loadings()
        out = np.zeros((total, self.n_states + self.n_controls))
        x = np.zeros(self.n_states)
        for t in range(total):
            out[t] = M_x @ x + M_u @ shocks[t]
            x = G @ x + N @ shocks[t]
        frame = pd.DataFrame(out, columns=list(self.states) + list(self.controls))
        return frame.iloc[burn:].reset_index(drop=True)[list(self.variables)]

    def stoch_simul(
        self,
        *,
        order: int = 1,
        irf: int = 40,
        periods: int = 0,
        sigma: float | Mapping[str, float] | None = None,
        seed: int = 0,
        burn: int = 100,
        lags: int = 5,
    ) -> StochSimulResult:
        """Execute Dynare-compatible stoch_simul routine.

        Computes:
        1. Decision rules (oo_.dr)
        2. Analytical theoretical moments (moments, covariance, correlation, autocorrelations, FEVD)
        3. Impulse response functions to a one-standard-deviation innovation in each shock
        4. Simulated sample moments (if periods > 0)

        Parameters
        ----------
        order : int, default 1
            Approximation order. If order=2, the second-order solution is
            computed with the requested shock covariance and the call is
            delegated to :meth:`PrunedDSGESolution.stoch_simul`.
        irf : int, default 40
            Horizon for impulse response functions. Set to 0 to skip IRFs.
        periods : int, default 0
            Number of simulation periods. If > 0, generates simulated moments.
        sigma : float | Mapping[str, float], optional
            Shock standard deviations. Default: the covariance declared in the
            .mod ``shocks`` block (or ``shock_cov=``) when present, else 1.0
            for each shock — the Dynare convention.
        seed : int, default 0
            RNG seed for simulation when periods > 0.
        burn : int, default 100
            Burn-in periods dropped before calculating simulated moments.
        lags : int, default 5
            Number of autocorrelation lags.

        Returns
        -------
        StochSimulResult
            Container holding dr, theoretical_moments, simulated_moments, irfs, and export methods.
        """
        if order == 2:
            sol2 = self.solve_second_order(shock_cov=self._shock_covariance(sigma))
            return sol2.stoch_simul(
                order=2,
                irf=irf,
                periods=periods,
                sigma=1.0,
                seed=seed,
                burn=burn,
                lags=lags,
            )
        elif order != 1:
            raise ValueError(f"unsupported perturbation order {order}; must be 1 or 2")

        self._require_solution("stoch_simul()")
        sd = self._shock_sd(sigma)
        sigma_full = sigma if sigma is None else dict(zip(self.shocks, sd))

        dr = self.decision_rules()
        theo = self.theoretical_moments(sigma=sigma_full, lags=lags)

        irfs: dict[str, pd.Series] = {}
        if irf > 0:
            for j, sh in enumerate(self.shocks):
                df_irf = self.irf(shock=sh, horizon=irf, size=float(sd[j]))
                for v in self.variables:
                    irfs[f"{v}_{sh}"] = df_irf[v]

        sim_moments = None
        if periods > 0:
            sim_df = self.simulate(periods=periods, sigma=sigma_full, seed=seed, burn=burn)
            sim_moments = pd.DataFrame(
                {
                    "Mean": sim_df.mean(axis=0),
                    "Std.Dev.": sim_df.std(axis=0),
                    "Variance": sim_df.var(axis=0),
                    "Skewness": sim_df.skew(axis=0),
                    "Kurtosis": sim_df.kurtosis(axis=0),
                },
                index=list(self.variables),
            )

        return StochSimulResult(
            dr=dr,
            theoretical_moments=theo,
            simulated_moments=sim_moments,
            irfs=irfs,
            order=1,
            variable_names=self.variables,
            shock_names=self.shocks,
        )

    def summary(self) -> str:
        """One-screen description: sizes, steady state, BK verdict."""
        eu = self.solution.eu
        verdict = {
            (1, 1): "unique stable solution",
            (1, 0): "indeterminate (multiple stable solutions)",
            (0, 0): "no stable solution (Blanchard-Kahn violated)",
        }.get(tuple(eu), f"eu={tuple(eu)}")
        timing = (
            "Dynare (states dated end-of-period, respond on impact)"
            if self.timing == "dynare" else
            "Klein (states predetermined at t, move at t+1)"
        )
        lines = [
            f"LinearModel · {len(self.variables)} variables "
            f"({self.n_states} states, {self.n_controls} controls), "
            f"{len(self.shocks)} shock(s)",
            f"  jacobians    : {self.method}-step",
            f"  timing       : {timing}",
            f"  steady state : residual {self.residual_norm:.2e}",
            f"  Blanchard-Kahn: {verdict}",
            "  steady state values:",
        ]
        for name in self.variables:
            lines.append(
                f"    {name:<10s} {self.steady_state[name]:>12.6g}  "
                f"[{self.units[name]}]"
            )
        return "\n".join(lines)

    def solve(self, order: int = 1, *, shock_cov: np.ndarray | None = None):
        """Return the solution at the requested perturbation order.

        Parameters
        ----------
        order : {1, 2}, default 1
            ``1`` returns this (already solved) first-order model itself;
            ``2`` returns the pruned second-order solution from
            :meth:`solve_second_order`.
        shock_cov : np.ndarray, optional
            Innovation covariance used for the second-order risk correction.
            Defaults to the declared shock covariance, else the identity.

        Returns
        -------
        LinearModel | PrunedDSGESolution
        """
        if order == 1:
            return self
        if order == 2:
            return self.solve_second_order(shock_cov=shock_cov)
        raise ValueError(f"unsupported perturbation order {order}; must be 1 or 2")

    def solve_second_order(
        self,
        shock_cov: np.ndarray | None = None,
    ):
        """Solve second-order perturbation approximation with pruning (SGU 2004, Kim et al. 2008).

        Parameters
        ----------
        shock_cov : np.ndarray, optional
            Covariance matrix of innovations Σ_u. Defaults to the covariance
            declared with the model (``shocks`` block / ``shock_cov=``), else
            the identity matrix.

        Returns
        -------
        PrunedDSGESolution
            Second-order solution equipped with `.simulate()`, `.girf()`, and
            `.stochastic_steady_state()`.
        """
        if self._dynare_equations is None:
            raise ModelError(
                "solve_second_order requires the model to be built with build_dynare "
                "or load_mod (using canonical lead-lag equations)."
            )
        from puremacro.dsge.dynare import solve_dynare_2nd_order

        cov = shock_cov if shock_cov is not None else self._shock_cov
        return solve_dynare_2nd_order(
            self._dynare_equations,
            variables=self.variables,
            shocks=self.shocks,
            params=self._params,
            steady_state=self.steady_state.to_dict(),
            states=self.states,
            shock_cov=cov,
            method=self.method,
        )


# ---------------------------------------------------------------------
# differentiation
# ---------------------------------------------------------------------

def _jacobian(f: Callable, x0: np.ndarray, n_out: int, method: str) -> np.ndarray:
    """Jacobian of ``f`` at ``x0``, shape ``(n_out, len(x0))``."""
    n = len(x0)
    jac = np.zeros((n_out, n))
    if method == "complex":
        base = np.asarray(x0, dtype=complex)
        for j in range(n):
            pert = base.copy()
            pert[j] += 1j * _CSTEP
            out = np.asarray(f(pert), dtype=complex)
            jac[:, j] = out.imag / _CSTEP
        return jac
    base = np.asarray(x0, dtype=float)
    for j in range(n):
        step = _FDSTEP * max(1.0, abs(base[j]))
        up, dn = base.copy(), base.copy()
        up[j] += step
        dn[j] -= step
        jac[:, j] = (np.asarray(f(up), dtype=float)
                     - np.asarray(f(dn), dtype=float)) / (2.0 * step)
    return jac


_NOT_ANALYTIC = (
    "The usual cause is a residual function that is not analytic — abs(), "
    "min()/max(), a comparison that branches on a perturbed value, or a "
    "float()/np.real() cast that throws the imaginary part away. Rewrite "
    "those, or pass method='central' to use finite differences instead "
    "(~1e-8 accuracy rather than ~1e-15)."
)


def _verify_jacobian(label: str, f: Callable, x0: np.ndarray,
                     jac: np.ndarray) -> None:
    """Cross-check one complex-step Jacobian against finite differences.

    Complex-step is exact *if* the residual function is analytic, and
    silently wrong if it is not — ``Im f(x + ih)`` is identically zero
    through an ``abs()``, so the derivative comes back as zero with no
    error anywhere. Finite differences have no such blind spot, and
    disagreeing with them is the signature of the failure.

    One random direction is enough to catch it: a discrepancy in any
    column shows up in the directional derivative almost surely. Costs
    two extra evaluations per block.
    """
    if jac.size == 0:
        return
    # A fixed direction keeps build() deterministic.
    v = np.random.default_rng(0).standard_normal(len(x0))
    step = _FDSTEP * max(1.0, float(np.max(np.abs(x0))))
    fd = (np.asarray(f(x0 + step * v), dtype=float)
          - np.asarray(f(x0 - step * v), dtype=float)) / (2.0 * step)
    cs = jac @ v
    scale = max(1.0, float(np.max(np.abs(fd))), float(np.max(np.abs(cs))))
    gap = float(np.max(np.abs(cs - fd)))
    if gap > 1e-4 * scale:
        rows = np.flatnonzero(np.abs(cs - fd) > 1e-4 * scale)
        raise ModelError(
            f"complex-step and finite-difference derivatives disagree for the "
            f"{label} block (relative gap {gap / scale:.2e}, worst equation "
            f"index {int(rows[0])}). {_NOT_ANALYTIC}"
        )


def _check_analytic(blocks: dict, method: str) -> None:
    """Validate complex-step Jacobians before they become a solved model.

    ``blocks`` maps a label to ``(jacobian, function, base_point)``.
    """
    if method != "complex":
        return
    dead = [label for label, (J, _, _) in blocks.items()
            if J.size and not np.any(J)]
    if dead:
        raise ModelError(
            f"complex-step differentiation produced an all-zero Jacobian for "
            f"the {', '.join(dead)} block. {_NOT_ANALYTIC}"
        )
    for label, (J, f, x0) in blocks.items():
        _verify_jacobian(label, f, x0, J)


# ---------------------------------------------------------------------
# builder
# ---------------------------------------------------------------------

def _validate_names(variables, states, shocks) -> tuple:
    variables = tuple(variables)
    states = tuple(states)
    shocks = tuple(shocks)
    if len(set(variables)) != len(variables):
        raise ModelError(f"duplicate variable names in {list(variables)}")
    if len(set(shocks)) != len(shocks):
        raise ModelError(f"duplicate shock names in {list(shocks)}")
    unknown = [s for s in states if s not in variables]
    if unknown:
        raise ModelError(
            f"states {unknown} are not in variables {list(variables)}"
        )
    if len(set(states)) != len(states):
        raise ModelError(f"duplicate state names in {list(states)}")
    if not states:
        raise ModelError(
            "a model needs at least one predetermined variable; declare the "
            "capital stock / exogenous process(es) in states="
        )
    controls = tuple(v for v in variables if v not in set(states))
    return variables, states, controls, shocks


def _solve_steady_state(f, variables, shocks, params, guess, tol) -> np.ndarray:
    n = len(variables)
    zeros = np.zeros(len(shocks))

    def residual(v):
        vec = _Vec(variables, np.asarray(v, dtype=float))
        return np.asarray(
            f(vec, vec, _Vec(shocks, zeros, "shock"), params), dtype=float,
        )

    x0 = np.array([float(guess[name]) for name in variables])
    out = scipy.optimize.root(residual, x0, method="hybr", tol=tol)
    if not out.success:
        raise SteadyStateError(
            f"steady state did not converge from the supplied guess "
            f"({dict(zip(variables, x0))}): {out.message}. Try a guess closer "
            f"to the solution, or pass steady_state= directly."
        )
    return np.asarray(out.x, dtype=float)


def build(equations: Callable, *, variables: Sequence[str],
          states: Sequence[str], shocks: Sequence[str],
          params: Mapping | None = None,
          steady_state: Mapping | None = None,
          guess: Mapping | None = None,
          linearize: str = "log",
          method: str = "complex",
          verify_derivatives: bool = True,
          strict: bool = True,
          tol: float = 1e-9) -> LinearModel:
    """Linearise and solve a model written as an equilibrium-condition function.

    Parameters
    ----------
    equations : callable
        ``equations(xp, x, e, p)`` returning one residual per equation,
        where ``xp`` / ``x`` are the ``t+1`` / ``t`` variable vectors,
        ``e`` the innovations and ``p`` the parameters. All four support
        attribute access (``x.k``), string indexing (``x["k"]``),
        positional indexing and unpacking. Must return exactly
        ``len(variables)`` residuals.
    variables : sequence of str
        Variable names. Fixes the order everything else is reported in.
    states : sequence of str
        The predetermined subset of ``variables``. Everything else is
        treated as forward-looking.
    shocks : sequence of str
        Innovation names.
    params : mapping, optional
        Parameter values, exposed to ``equations`` as ``p``.
    steady_state : mapping, optional
        A known steady state. Verified against the equations
        (``max|f| <= tol``) rather than trusted.
    guess : mapping, optional
        Starting values for solving the steady state numerically.
        Required unless ``steady_state`` is given.
    linearize : {"log", "level"}, default "log"
        ``"log"`` gives log deviations — the usual choice, and the one
        that makes IRFs read as percentages. Variables whose steady
        state is not strictly positive fall back to level deviations
        automatically; :attr:`LinearModel.units` records which is which.
    method : {"complex", "central"}, default "complex"
        Differentiation scheme. See the module docstring on when
        ``"central"`` is necessary.
    verify_derivatives : bool, default True
        Cross-check the complex-step Jacobians against finite differences
        and raise if they disagree — the only way to catch a residual
        function that is not analytic, since complex-step fails silently
        on those. Costs six extra function evaluations.
    strict : bool, default True
        Raise :class:`~puremacro.dsge.klein.BlanchardKahnError` when the
        model has no unique stable solution, rather than returning zero
        matrices.
    tol : float, default 1e-9
        Steady-state residual tolerance.

    Returns
    -------
    LinearModel

    Raises
    ------
    ModelError
        Names, equation count, or analyticity problems.
    SteadyStateError
        The steady state does not solve the model, or would not converge.
    BlanchardKahnError
        No unique stable solution (when ``strict``).
    """
    if linearize not in ("log", "level"):
        raise ValueError(f"linearize must be 'log' or 'level', got {linearize!r}")
    if method not in ("complex", "central"):
        raise ValueError(f"method must be 'complex' or 'central', got {method!r}")

    variables, states, controls, shocks = _validate_names(variables, states, shocks)
    n = len(variables)
    n_e = len(shocks)
    par = _Vec(tuple(params or {}), list((params or {}).values()), "parameter")

    if steady_state is None:
        if guess is None:
            raise ModelError(
                "pass either steady_state= (a known one) or guess= (starting "
                "values to solve for it)"
            )
        missing = [v for v in variables if v not in guess]
        if missing:
            raise ModelError(f"guess is missing values for {missing}")
        ss = _solve_steady_state(equations, variables, shocks, par, guess, tol)
    else:
        missing = [v for v in variables if v not in steady_state]
        if missing:
            raise ModelError(f"steady_state is missing values for {missing}")
        ss = np.array([float(steady_state[v]) for v in variables])

    # Equation count is only knowable once the function has been called.
    zeros_e = np.zeros(n_e)
    ss_vec = _Vec(variables, ss)
    resid0 = np.asarray(
        equations(ss_vec, ss_vec, _Vec(shocks, zeros_e, "shock"), par), dtype=float,
    )
    if resid0.shape != (n,):
        raise ModelError(
            f"equations() returned {resid0.shape[0] if resid0.ndim else 1} "
            f"residual(s) for {n} variables — a square system needs one "
            f"equation per variable"
        )
    residual_norm = float(np.max(np.abs(resid0)))
    if residual_norm > max(tol, 1e-6):
        raise SteadyStateError(
            f"the supplied steady state does not solve the model: "
            f"max|f(ss, ss, 0)| = {residual_norm:.3e}. Worst equation: index "
            f"{int(np.argmax(np.abs(resid0)))}."
        )

    # Per-variable substitution: x = ss*exp(xhat) where that is defined,
    # x = ss + xhat elsewhere. Mixing is standard practice — a variable
    # with a zero or negative steady state (net exports, a log-level
    # process) has no log deviation to speak of.
    use_log = np.array([
        linearize == "log" and ss[i] > 0.0 for i in range(n)
    ])
    units = {
        name: ("log" if use_log[i] else "level")
        for i, name in enumerate(variables)
    }

    def levels(hat):
        out = np.empty(n, dtype=np.asarray(hat).dtype)
        for i in range(n):
            out[i] = ss[i] * np.exp(hat[i]) if use_log[i] else ss[i] + hat[i]
        return out

    zeros_n = np.zeros(n)

    def f_of_next(hat_p):
        return equations(_Vec(variables, levels(hat_p)),
                         _Vec(variables, levels(zeros_n)),
                         _Vec(shocks, np.zeros(n_e, dtype=np.asarray(hat_p).dtype),
                              "shock"), par)

    def f_of_now(hat_c):
        return equations(_Vec(variables, levels(zeros_n)),
                         _Vec(variables, levels(hat_c)),
                         _Vec(shocks, np.zeros(n_e, dtype=np.asarray(hat_c).dtype),
                              "shock"), par)

    def f_of_shock(eps):
        dtype = np.asarray(eps).dtype
        return equations(_Vec(variables, levels(np.zeros(n, dtype=dtype))),
                         _Vec(variables, levels(np.zeros(n, dtype=dtype))),
                         _Vec(shocks, eps, "shock"), par)

    Fp = _jacobian(f_of_next, zeros_n, n, method)
    Fc = _jacobian(f_of_now, zeros_n, n, method)
    Fu = _jacobian(f_of_shock, zeros_e, n, method) if n_e else np.zeros((n, 0))
    if verify_derivatives:
        _check_analytic({
            "t+1": (Fp, f_of_next, zeros_n),
            "t": (Fc, f_of_now, zeros_n),
            "shock": (Fu, f_of_shock, zeros_e),
        }, method)

    # f = 0 linearised is  Fp z' + Fc z + Fu u = 0, i.e. Klein's
    # A E_t z' = B z + C u with A = Fp, B = -Fc, C = -Fu.
    order = [variables.index(v) for v in (*states, *controls)]
    A = Fp[:, order]
    B = -Fc[:, order]
    C = -Fu

    solution = klein_solve(A, B, len(states), C, strict=strict)

    return LinearModel(
        # tuple(...) is a no-op at run time (_validate_names already
        # returned tuples) and keeps the declared Sequence[str] parameter
        # types from leaking into the frozen dataclass's tuple fields.
        variables=tuple(variables), states=tuple(states), controls=controls,
        shocks=tuple(shocks),
        steady_state=pd.Series(ss, index=list(variables)), units=units,
        solution=solution, A=A, B=B, C=C, method=method,
        residual_norm=residual_norm,
    )
