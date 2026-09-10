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
analytic*. What actually breaks that, measured rather than assumed:

* **Silently wrong** — ``abs`` / ``np.abs`` (returns the modulus, whose
  imaginary part is zero), ``np.linalg.norm``, ``np.real`` / ``.real``
  and ``float()`` casts, and a fractional power of a base that goes
  negative (``(-2 + 1e-20j)**0.33`` is finite and meaningless where the
  real evaluation is ``nan``). These give a derivative with no error
  anywhere, which is what the cross-check below exists for.
* **Loud** — the builtin ``min`` / ``max`` and any explicit comparison,
  which raise ``TypeError: '>' not supported between instances of
  'float' and 'complex'``.
* **Safe** — ``np.maximum`` / ``np.minimum``, which order complex numbers
  lexicographically on the real part and so pick the same branch the
  real evaluation would, away from an exact tie. (At a tie they are as
  wrong as everything else at a kink.)

:func:`build` guards this two ways: a Jacobian block that comes out
identically zero is rejected outright, and every block is cross-checked
against a finite difference along two probe directions, per equation and
relative to that equation's own derivative. A confirmed disagreement is
a :class:`ModelError`; a check the finite difference is too
ill-conditioned to referee is a ``UserWarning`` saying so.

``method="central"`` is the escape hatch where the residual is genuinely
smooth at the steady state and only the *complex arithmetic* is broken —
``abs(x)`` evaluated far from ``x = 0``, say — and costs ~1e-8 accuracy
instead of ~1e-15. It is **not** a fix for a kink: at ``abs(x - x_ss)``
the central difference returns the average of the two one-sided slopes,
which is a different wrong answer rather than a right one. Models with
occasionally-binding constraints belong in
:mod:`puremacro.dsge.occbin`.

Terms outside ``t`` and ``t+1``
-------------------------------
The residual sees exactly two time slices. A ``t-1`` term or a ``t+2``
term is written with an auxiliary variable and an identity, as in a .mod
file: add ``k_lag`` to ``variables`` and to ``states``, add the equation
``xp.k_lag - x.k``, and use ``x.k_lag`` wherever ``k_{t-1}`` appears.

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

import warnings
from dataclasses import dataclass
import math
from typing import Any, Callable, Mapping, Sequence

import numpy as np
import pandas as pd
import scipy.linalg
import scipy.optimize

from puremacro.dsge.klein import KleinSolution, klein_solve
from puremacro.dsge._results import (
    DynareDR,
    TheoreticalMomentsResult,
    StochSimulResult,
    EigenvalueTable,
    ModelDiagnosticsResult,
    IdentificationResult,
    OSRResult,
)
from puremacro.dsge._moments import conditional_fevd, first_order_moments

__all__ = [
    "ModelError",
    "SteadyStateError",
    "LinearModel",
    "build",
    "DynareDR",
    "TheoreticalMomentsResult",
    "StochSimulResult",
    "EigenvalueTable",
    "ModelDiagnosticsResult",
    "IdentificationResult",
    "OSRResult",
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
    _equations: Callable | None = None
    _dynare_equations: Callable | None = None
    _params: dict | None = None
    _A_plus: np.ndarray | None = None
    _A_0: np.ndarray | None = None
    _A_minus: np.ndarray | None = None
    _B_u: np.ndarray | None = None
    _shock_cov: np.ndarray | None = None
    timing: str = "klein"
    # Declarations carried over from a .mod file, when the model came from one.
    # ``None`` on every model built through ``build()``. The dataclass is
    # frozen, so ``load_mod`` attaches these with ``dataclasses.replace``.
    _varobs: tuple | None = None
    _estimated_params: Any | None = None
    _mod_options: dict | None = None
    _equation_tags: tuple | None = None
    anticipated_shocks: dict | None = None

    def __post_init__(self):
        if self.timing not in ("klein", "dynare"):
            raise ValueError(
                f"timing must be 'klein' or 'dynare', got {self.timing!r}"
            )
        if (
            self.anticipated_shocks is None
            and self._mod_options
            and "anticipated_shocks" in self._mod_options
        ):
            object.__setattr__(
                self, "anticipated_shocks", self._mod_options["anticipated_shocks"]
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

    @property
    def A_plus(self) -> np.ndarray | None:
        """Lead variable Jacobian matrix E_t y_{t+1}, or None if not lead-lag solved."""
        return self._A_plus

    @property
    def A_0(self) -> np.ndarray | None:
        """Current variable Jacobian matrix y_t, or None if not lead-lag solved."""
        return self._A_0

    @property
    def A_minus(self) -> np.ndarray | None:
        """Lagged variable Jacobian matrix y_{t-1}, or None if not lead-lag solved."""
        return self._A_minus

    @property
    def B_u(self) -> np.ndarray | None:
        """Shock loading Jacobian matrix u_t, or None if not lead-lag solved."""
        return self._B_u

    def check(self, *, qz_criterium: float = 1.0 + 1e-6) -> EigenvalueTable:
        """Evaluate Blanchard-Kahn determinacy and generalized eigenvalue spectrum.

        Computes generalized eigenvalues, classifies roots into stable, explosive,
        and unit roots, and extracts variable loadings on offending eigenvectors
        if determinacy fails.

        Parameters
        ----------
        qz_criterium : float, default 1.0 + 1e-6
            Threshold modulus above which a root is considered explosive.

        Returns
        -------
        EigenvalueTable
            Structured result table with .summary(), .plot(), and export methods.
        """
        from puremacro.dsge.diagnostics import check as _check

        return _check(self, qz_criterium=qz_criterium)

    def resid(self) -> pd.Series:
        """Compute deterministic steady-state equation residuals.

        Evaluates the model's dynamic equilibrium equations at steady state
        f(ss, ss, ss, 0) and returns residuals sorted by absolute magnitude.

        Returns
        -------
        pandas.Series
            Residuals indexed by equation tags or labels, sorted descending by |resid|.
        """
        from puremacro.dsge.diagnostics import resid as _resid

        return _resid(self)

    def model_diagnostics(
        self,
        *,
        tol: float = 1e-8,
        n_eval_points: int = 5,
    ) -> ModelDiagnosticsResult:
        """Perform comprehensive structural and numerical diagnostics on the DSGE model.

        Performs static Jacobian rank analysis, union numeric incidence evaluation
        across neighbourhood perturbations, dynamic pencil regularity check,
        stochastic singularity check, and unit root detection.

        Parameters
        ----------
        tol : float, default 1e-8
            Tolerance for rank deficiency and residual checks.
        n_eval_points : int, default 5
            Number of points for numeric incidence evaluation.

        Returns
        -------
        ModelDiagnosticsResult
            Structured diagnostics result with .summary(), .plot(), and findings.
        """
        from puremacro.dsge.diagnostics import model_diagnostics as _model_diagnostics

        return _model_diagnostics(self, tol=tol, n_eval_points=n_eval_points)

    def identification(
        self,
        *,
        varobs: Sequence[str] | None = None,
        params: Sequence[str] | Mapping[str, float] | Any | None = None,
        lags: int = 1,
        tol: float = 1e-8,
        prior_mc: int = 0,
        seed: int = 0,
    ) -> IdentificationResult:
        """Perform Iskrev (2010) and Ratto (2011) parameter identification analysis.

        Parameters
        ----------
        varobs : Sequence[str], optional
            Observable variable names. Defaults to model._varobs or all model variables.
        params : Sequence[str] | Mapping[str, float] | EstimatedParams, optional
            Parameters to evaluate. Defaults to model._estimated_params or calibrated parameters.
        lags : int, default 1
            Autocovariance lags included in moment Jacobian J2.
        tol : float, default 1e-8
            Relative SVD tolerance for rank determination.
        prior_mc : int, default 0
            Number of prior Monte Carlo draws to evaluate.
        seed : int, default 0
            Random seed for prior Monte Carlo.

        Returns
        -------
        IdentificationResult
            Frozen dataclass with rank, null spaces, collinearity, and presentation methods.
        """
        from puremacro.dsge.identification import identification as _identification

        return _identification(
            self,
            varobs=varobs,
            params=params,
            lags=lags,
            tol=tol,
            prior_mc=prior_mc,
            seed=seed,
        )

    def osr(
        self,
        rule_params: Sequence[str],
        weights: Mapping[str, float],
        *,
        bounds: Mapping[str, tuple[float, float]] | None = None,
        target_vars: Sequence[str] | None = None,
        optimizer: str = "Nelder-Mead",
        maxiter: int = 1000,
        penalty: float = 1e6,
    ) -> OSRResult:
        """Optimize policy rule parameters against theoretical variance loss.

        Minimizes quadratic loss L(gamma) = sum_i w_i Var(y_i; gamma) subject to
        Blanchard-Kahn determinacy, penalizing determinacy failures with a continuous
        numerical penalty surface.

        Parameters
        ----------
        rule_params : Sequence[str]
            Names of policy rule parameters to optimize (e.g. ["phi_pi", "phi_y"]).
        weights : Mapping[str, float]
            Quadratic loss weights per target variable (e.g. {"pi": 1.0, "y": 0.5}).
        bounds : Mapping[str, tuple[float, float]], optional
            Lower and upper bounds per rule parameter.
        target_vars : Sequence[str], optional
            Target variables (defaults to keys of weights).
        optimizer : str, default "Nelder-Mead"
            Optimization algorithm ("Nelder-Mead" or "Powell").
        maxiter : int, default 1000
            Maximum optimizer iterations.
        penalty : float, default 1e6
            Finite numerical penalty baseline when Blanchard-Kahn condition fails.

        Returns
        -------
        OSRResult
            Frozen dataclass with optimal coefficients, loss comparison, variance table,
            and presentation methods.
        """
        from puremacro.dsge.policy import osr as _osr

        return _osr(
            self,
            rule_params=rule_params,
            weights=weights,
            bounds=bounds,
            target_vars=target_vars,
            optimizer=optimizer,
            maxiter=maxiter,
            penalty=penalty,
        )

    def optimal_policy(
        self,
        loss: Mapping[str, Any] | Sequence[float] | str,
        rule: str = "discretion",
        **kwargs: Any,
    ) -> Any:
        """Solve optimal discretionary or commitment policy for this linear model.

        Parameters
        ----------
        loss : dict, sequence, or str
            Quadratic loss specification (weights, targets, or string expression).
        rule : {"discretion", "commitment"}, default "discretion"
            Policy regime to solve.
        **kwargs : Any
            Additional options passed to optimal_policy (e.g. instruments, target_vars, beta, tol).

        Returns
        -------
        DiscretionaryPolicyResult or PolicyResult
        """
        from puremacro.dsge.policy import optimal_policy as _optimal_policy

        return _optimal_policy(self, loss=loss, rule=rule, **kwargs)

    def dsge_var(
        self,
        data: pd.DataFrame | np.ndarray,
        p: int = 4,
        lamb: float | str | None = None,
        **kwargs: Any,
    ) -> Any:
        """Estimate Del Negro & Schorfheide (2004) DSGE-VAR(lambda).

        Parameters
        ----------
        data : pd.DataFrame or np.ndarray
            Sample observable time series.
        p : int, default 4
            VAR lag order.
        lamb : float, 'optimal', or None, default None
            Prior weight parameter lambda. If None or 'optimal', maximizes
            the marginal data density over lambda in [0.2, 5.0].
        **kwargs : Any
            Additional keyword arguments passed to :func:`estimate_dsge_var`.

        Returns
        -------
        DSGEVARResult
            Estimated hybrid DSGE-VAR container.
        """
        from puremacro.dsge.dsge_var import estimate_dsge_var

        return estimate_dsge_var(self, data, p=p, lamb=lamb, **kwargs)

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
        hp_filter: float | None = None,
        bandpass_filter: tuple[float, float] | Sequence[float] | None = None,
        one_sided_hp_filter: bool | float | None = None,
        contemporaneous_correlation: bool = True,
        ar: int | None = None,
    ) -> TheoreticalMomentsResult:
        """Calculate analytical theoretical moments matching Dynare's stoch_simul.

        Solves the discrete Lyapunov equation or evaluates Gauss-Legendre
        quadrature spectral density integrals for unconditional stationary
        moments, cross-correlations, autocorrelations, and forecast error
        variance decomposition.

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
        hp_filter : float, optional
            Hodrick-Prescott filter smoothing parameter lambda (e.g. 1600.0).
            Theoretical moments are evaluated via Gauss-Legendre quadrature.
        bandpass_filter : tuple of (float, float), optional
            Baxter-King bandpass filter periodicities (low, high), e.g. (6, 32).
        one_sided_hp_filter : bool or float, optional
            One-sided HP filter. Theoretical moments are incompatible and will raise ValueError.
        contemporaneous_correlation : bool, default True
            Whether to compute and populate contemporaneous correlation matrix.
        ar : int, optional
            Number of autocorrelation lags (overrides ``lags`` if specified).

        Returns
        -------
        TheoreticalMomentsResult
            Container with moments, covariance, correlation, autocorrelation, and FEVD.
        """
        if one_sided_hp_filter is not None and one_sided_hp_filter is not False:
            raise ValueError(
                "disp_th_moments:: theoretical moments incompatible with one-sided HP filter. "
                "Use simulated moments instead."
            )

        n_filters = (hp_filter is not None) + (bandpass_filter is not None)
        if n_filters > 1:
            raise ValueError(
                "Only one filter can be specified among hp_filter, one_sided_hp_filter, and bandpass_filter."
            )

        if hp_filter is not None and hp_filter <= 0:
            raise ValueError(f"hp_filter parameter lambda must be positive, got {hp_filter}")

        if bandpass_filter is not None:
            bp = tuple(bandpass_filter)
            if len(bp) != 2 or bp[0] <= 0 or bp[1] <= bp[0]:
                raise ValueError(f"bandpass_filter requires 0 < low < high, got {bandpass_filter}")

        if ar is not None:
            if ar < 0:
                raise ValueError(f"ar must be a non-negative integer, got {ar}")
            lags = int(ar)

        self._require_solution("theoretical_moments()")
        sigma_u = self._shock_covariance(sigma)
        G, N = self.solution.G, self.solution.N
        M_x, M_u = self._reported_loadings()

        from puremacro.dsge._moments import (
            compute_autocorr_matrices,
            first_order_moments,
            spectral_moments,
        )

        if hp_filter is not None or bandpass_filter is not None:
            filter_type = "hp" if hp_filter is not None else "bandpass"
            hp_lambda = float(hp_filter) if hp_filter is not None else 1600.0
            bp_tuple = tuple(bandpass_filter) if bandpass_filter is not None else None
            _, gamma_0, gammas = spectral_moments(
                G,
                N,
                M_x,
                M_u,
                sigma_u,
                lags=lags,
                filter_type=filter_type,
                hp_lambda=hp_lambda,
                bandpass=bp_tuple,
            )
        else:
            g_eigs = np.abs(scipy.linalg.eigvals(self.solution.G))
            if np.any(g_eigs >= 1.0 - 1e-7):
                bad = g_eigs[g_eigs >= 1.0 - 1e-7]
                raise ValueError(
                    f"State transition matrix G has non-stationary eigenvalues (|λ| >= 1.0: {bad}); "
                    "unconditional stationary moments do not exist."
                )
            _, gamma_0, gammas = first_order_moments(G, N, M_x, M_u, sigma_u, lags)

        order_vars = list(self.states) + list(self.controls)
        cov_df = pd.DataFrame(
            gamma_0, index=order_vars, columns=order_vars
        ).loc[list(self.variables), list(self.variables)]

        gammas_reordered = [
            pd.DataFrame(gk, index=order_vars, columns=order_vars).loc[
                list(self.variables), list(self.variables)
            ].to_numpy()
            for gk in gammas
        ]

        df_corr, autocorr_matrices = compute_autocorr_matrices(
            cov_df.to_numpy(), gammas_reordered, list(self.variables)
        )

        variances = np.diag(cov_df.to_numpy())
        stds = np.sqrt(np.maximum(variances, 0.0))
        if hp_filter is not None or bandpass_filter is not None:
            means = np.zeros(len(self.variables))
        else:
            means = np.array([float(self.steady_state[v]) for v in self.variables])

        df_moments = pd.DataFrame(
            {"Mean": means, "Std.Dev.": stds, "Variance": variances},
            index=list(self.variables),
        )

        # Autocorrelations: diag(Gamma_k) / diag(Gamma_0)
        autocorr_cols = [f"Lag {k}" for k in range(1, lags + 1)]
        df_autocorr = pd.DataFrame(
            index=list(self.variables), columns=autocorr_cols, dtype=float
        )
        for k, mat in enumerate(autocorr_matrices, start=1):
            df_autocorr[f"Lag {k}"] = np.diag(mat.to_numpy())

        # Variance Decomposition
        df_fevd = self.fevd(horizons=fevd_horizons, sigma=sigma)

        return TheoreticalMomentsResult(
            moments=df_moments,
            covariance=cov_df,
            correlation=df_corr if contemporaneous_correlation else None,
            autocorr=df_autocorr,
            fevd=df_fevd,
            autocorr_matrices=autocorr_matrices,
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

    # -- estimation ----------------------------------------------------

    def estimate(self, data, *, priors=None, varobs=None, fixed_params=None,
                 measurement_error=None, prefilter=False,
                 observation_trends=None, ridge=0.0, mode_compute="lbfgs",
                 n_draws: int = 10_000, n_chains: int = 2,
                 burn_in: int = 2_000, seed: int = 0, model_name=None,
                 check_identification: bool | str = False,
                 method: str = "kalman", **kwargs):
        """Bayesian estimation of this model on ``data``.

        ``priors`` and ``varobs`` default to the ``estimated_params`` and
        ``varobs`` blocks of the .mod file the model came from, so a Dynare
        file needs no second copy of its own declarations. Delegates to
        :func:`~puremacro.dsge.estimate_dsge`, which is unchanged: this method
        builds the ``observation_eq`` callable that it has always required.

        Parameters
        ----------
        data : pandas.DataFrame
            One column per observable, named as in ``varobs``.
        priors : EstimatedParams or dict, optional
            Defaults to the model's own ``estimated_params`` block.
        varobs : Sequence[str], optional
            Defaults to the model's own ``varobs`` declaration.
        fixed_params : Mapping[str, float], optional
            Parameter overrides held fixed during estimation, applied on top
            of the model's calibration.
        mode_compute : str, default 'lbfgs'
            See :mod:`puremacro.dsge.mode`.
        check_identification : bool | str, default False
            If True or 'warn', runs pre-flight identification check and emits
            a warning if unidentified parameters are found. If 'raise', raises
            ValueError.

        Returns
        -------
        DSGEPosteriorResult

        Notes
        -----
        Only parameters of kind ``"param"`` cause the model to be re-solved; a
        shock standard deviation, a shock correlation and a measurement-error
        standard deviation are written straight into ``Q`` and ``H``. Each
        re-solve warm-starts its steady-state search from the previous one.

        A ``steady_state_model`` block in the .mod file is used for the
        *initial* solve only. Re-solving at a new draw goes through the
        numerical steady-state solver warm-started from the previous draw,
        because the block's analytic formulas are evaluated once at parse time
        and are not re-evaluated per draw.

        There is no ``diffuse_filter`` argument. The Kalman recursion starts
        from the unconditional distribution and falls back to the diffuse
        prior only for a draw that has no unconditional distribution, warning
        when it does; a flag that could not change that would be decoration.
        """
        from .estimate import estimate_dsge
        from ._estimated_params import EstimatedParams

        spec_source = self._estimated_params if priors is None else priors
        if spec_source is None:
            raise ModelError(
                "estimate() needs priors: this model carries no "
                "`estimated_params` block (it was not built from a .mod file "
                "that declares one). Pass priors=... explicitly."
            )
        if isinstance(spec_source, EstimatedParams):
            missing_prior = [s.name for s in spec_source.specs if s.prior is None]
            if missing_prior:
                raise ModelError(
                    f"LinearModel.estimate() performs Bayesian estimation and "
                    f"requires a prior for every parameter in estimated_params; "
                    f"{missing_prior} was declared without a prior shape (Dynare "
                    f"maximum likelihood estimation is not implemented yet). "
                    f"Pass priors=... explicitly or declare a prior shape in the "
                    f".mod file."
                )
            specs = spec_source.specs
            prior_dict = spec_source.priors()
            initial = spec_source.initial_params()
        else:
            from ._estimated_params import EstimatedParamSpec
            from .priors import ensure_prior

            # A plain {name: prior} dict carries no `kind`, and guessing
            # "param" for everything is silently wrong: a name like SE_eps
            # would be passed to the solver as a structural parameter the
            # equations never read, leaving Q at its declared value and the
            # likelihood flat in it. Kinds are taken from the model's own
            # estimated_params block where the name appears there, so
            # overriding a prior keeps its meaning, and anything else must be
            # a declared model parameter.
            declared = {}
            if self._estimated_params is not None:
                declared = {sp.name: sp for sp in self._estimated_params.specs}
            known_params = set(self._params or {})
            prior_dict = {k: ensure_prior(v) for k, v in dict(spec_source).items()}
            built = []
            for k, pr in prior_dict.items():
                if k in declared:
                    d = declared[k]
                    built.append(EstimatedParamSpec(
                        kind=d.kind, target=d.target, name=k, prior=pr,
                        init=d.init, lb=pr.lb, ub=pr.ub, jscale=d.jscale))
                elif k in known_params:
                    built.append(EstimatedParamSpec(
                        kind="param", target=(k,), name=k, prior=pr,
                        init=None, lb=pr.lb, ub=pr.ub))
                else:
                    raise ModelError(
                        f"estimate(): prior {k!r} names neither a declared "
                        f"model parameter nor an entry of this model's "
                        f"estimated_params block, so there is no way to know "
                        f"what it should change. Model parameters: "
                        f"{sorted(known_params)}. Declared estimated "
                        f"parameters: {sorted(declared)}."
                    )
            specs = tuple(built)
            initial = {sp.name: sp.start for sp in specs}

        obs = list(varobs) if varobs is not None else (
            list(self._varobs) if self._varobs else None)
        if not obs:
            raise ModelError(
                "estimate() needs observables: this model carries no `varobs` "
                "declaration. Pass varobs=[...] explicitly."
            )

        if check_identification:
            from .identification import identification as _identification

            ident_res = _identification(self, varobs=obs, params=specs)
            if not ident_res.is_identified:
                unident_count = max(
                    ident_res.j1_n_params - ident_res.j1_rank,
                    ident_res.j2_n_params - ident_res.j2_rank,
                )
                offending = (
                    ident_res.j1_null_combinations
                    or ident_res.j2_null_combinations
                )
                msg = (
                    f"estimate(): DSGE parameter identification pre-flight check failed. "
                    f"Model has {unident_count} structurally unidentified parameter direction(s). "
                    f"J1 rank: {ident_res.j1_rank}/{ident_res.j1_n_params}, "
                    f"J2 rank: {ident_res.j2_rank}/{ident_res.j2_n_params}. "
                    f"Offending parameter combination(s): {list(offending)}"
                )
                if check_identification == "raise":
                    raise ValueError(msg)
                else:
                    warnings.warn(msg, UserWarning)

        observation_eq = _make_observation_eq(
            self, specs, obs, fixed_params=fixed_params,
            measurement_error=measurement_error, prefilter=prefilter,
            ridge=ridge,
        )
        frame = data
        if observation_trends:
            from .smoother import _trend_matrix

            frame = data.copy()
            trend = _trend_matrix(observation_trends, obs, len(data))
            frame[obs] = frame[obs].to_numpy(dtype=float) - trend

        return estimate_dsge(
            frame,
            observation_eq=observation_eq,
            priors=prior_dict,
            observed_vars=obs,
            initial_params=initial,
            model_name=model_name or "mod",
            mode_compute=mode_compute,
            n_draws=n_draws, n_chains=n_chains, burn_in=burn_in, seed=seed,
            method=method,
            model_template=self,
            **kwargs,
        )

    # -- filtering and forecasting -------------------------------------

    def smoother(self, data, **kwargs):
        """Kalman-smooth ``data`` at this model's calibrated parameters.

        Dynare's ``calib_smoother``. ``varobs`` defaults to the ``varobs``
        declaration of the .mod file the model came from. Returns a
        :class:`~puremacro.dsge._results.SmootherResult` carrying the smoothed
        states, the smoothed structural innovations and the fitted observables.
        """
        from .smoother import smooth_model

        return smooth_model(self, data, **kwargs)

    def forecast(self, horizon: int = 8, **kwargs):
        """Forecast the observables ``horizon`` periods ahead.

        With ``data=``, the forecast starts from the terminal filtered state;
        without it, from the steady state. The band reflects shock uncertainty
        only — the parameters are held fixed.
        """
        from .smoother import forecast_model

        return forecast_model(self, horizon, **kwargs)

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

    def news_irf(
        self,
        shock: str,
        lead: int = 0,
        horizon: int = 40,
        size: float = 1.0,
    ) -> Any:
        """Compute impulse response function for surprise or news (anticipated) shock.

        Parameters
        ----------
        shock : str
            Name of the structural shock innovation.
        lead : int, default 0
            Anticipation lead (0 for contemporaneous surprise shock).
            When lead = k > 0, an announcement arrives at t=0 about an innovation
            of magnitude `size` realizing at date t=k.
        horizon : int, default 40
            Simulation horizon in periods.
        size : float, default 1.0
            Magnitude of the shock.

        Returns
        -------
        NewsIRFResult
        """
        from puremacro.dsge.news import news_irf
        return news_irf(self, shock=shock, lead=lead, horizon=horizon, size=size)

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

    def interactive_irf(
        self,
        parameters: Sequence[str] | Mapping[str, tuple[float, ...]] | None = None,
        shocks: Sequence[str] | str | None = None,
        variables: Sequence[str] | None = None,
        horizon: int = 20,
        *,
        param_bounds: Mapping[str, tuple[float, float]] | None = None,
        param_steps: Mapping[str, float] | None = None,
        ncols: int = 2,
        figsize: tuple[float, float] | None = None,
        title: str = "",
        size: float = 1.0,
        show_baseline: bool = True,
        show_reset: bool = True,
        strict: bool = False,
        qz_criterium: float = 1.0 + 1e-8,
        **kwargs,
    ):
        """Spawn an interactive parameter exploration dashboard with Matplotlib Sliders.

        Delegates to :func:`puremacro.dsge.widgets.interactive_irf`.
        """
        from puremacro.dsge.widgets import interactive_irf

        return interactive_irf(
            self,
            parameters=parameters,
            shocks=shocks,
            variables=variables,
            horizon=horizon,
            param_bounds=param_bounds,
            param_steps=param_steps,
            ncols=ncols,
            figsize=figsize,
            title=title,
            size=size,
            show_baseline=show_baseline,
            show_reset=show_reset,
            strict=strict,
            qz_criterium=qz_criterium,
            **kwargs,
        )

    def simulate(
        self,
        periods: int = 200,
        shocks: np.ndarray | None = None,
        *,
        sigma=None,
        seed: int = 0,
        burn: int = 100,
        initial_state: np.ndarray | None = None,
    ) -> pd.DataFrame:
        """Simulate the model with i.i.d. Gaussian innovations or supplied shocks.

        Parameters
        ----------
        periods : int, default 200
            Periods returned, after ``burn``.
        shocks : np.ndarray, optional
            Pre-specified shock innovations of shape ``(periods, n_shocks)``
            or ``(periods + burn, n_shocks)``. If provided, random draws are bypassed.
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
        initial_state : np.ndarray, optional
            Initial state vector of length ``n_states``. Default is zeros.

        Returns
        -------
        pandas.DataFrame
            ``periods`` rows, one column per variable, in the same
            deviation units and timing as :meth:`irf`: row ``t`` holds
            every variable at date ``t``.
        """
        self._require_solution("simulate()")
        n_e = len(self.shocks)
        if shocks is not None:
            shocks_in = np.asarray(shocks, dtype=float)
            if shocks_in.ndim == 1:
                if n_e == 1:
                    shocks_in = shocks_in.reshape(-1, 1)
                elif len(shocks_in) == n_e:
                    shocks_in = shocks_in.reshape(1, n_e)
                else:
                    shocks_in = shocks_in.reshape(-1, 1)
            if n_e > 0 and shocks_in.shape[1] != n_e:
                raise ValueError(
                    f"shocks column dimension ({shocks_in.shape[1]}) does not match "
                    f"model shocks ({n_e})"
                )
            if shocks_in.shape[0] == periods + burn:
                total = periods + burn
                actual_burn = burn
            elif shocks_in.shape[0] == periods:
                total = periods
                actual_burn = 0
            else:
                total = len(shocks_in)
                actual_burn = 0
            actual_shocks = shocks_in
        else:
            rng = np.random.default_rng(seed)
            total = periods + burn
            actual_burn = burn
            if sigma is None and self._shock_cov is not None:
                cov = self._shock_covariance(None)
                actual_shocks = (
                    rng.multivariate_normal(np.zeros(n_e), cov, size=total, method="cholesky")
                    if n_e else np.zeros((total, 0))
                )
            else:
                sd = self._shock_sd(sigma, missing=0.0)
                actual_shocks = rng.standard_normal((total, n_e)) * sd

        G, N = self.solution.G, self.solution.N
        M_x, M_u = self._reported_loadings()
        out = np.zeros((total, self.n_states + self.n_controls))
        if initial_state is not None:
            init_arr = np.asarray(initial_state, dtype=float).ravel()
            if len(init_arr) == self.n_states:
                x = init_arr.copy()
            elif len(init_arr) == len(self.variables):
                x = np.array([init_arr[list(self.variables).index(s)] for s in self.states], dtype=float)
            else:
                raise ValueError(
                    f"initial_state length ({len(init_arr)}) must equal n_states ({self.n_states}) "
                    f"or n_vars ({len(self.variables)})"
                )
        else:
            x = np.zeros(self.n_states)
        for t in range(total):
            out[t] = M_x @ x + M_u @ actual_shocks[t]
            x = G @ x + N @ actual_shocks[t]
        frame = pd.DataFrame(out, columns=list(self.states) + list(self.controls))
        return frame.iloc[actual_burn:].reset_index(drop=True)[list(self.variables)]

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
        hp_filter: float | None = None,
        bandpass_filter: tuple[float, float] | Sequence[float] | None = None,
        one_sided_hp_filter: bool | float | None = None,
        simul_replic: int = 0,
        contemporaneous_correlation: bool = True,
        ar: int | None = None,
        qz_criterium: float = 1.0 + 1e-8,
    ) -> StochSimulResult:
        """Execute Dynare-compatible stoch_simul routine.

        Computes:
        1. Decision rules (oo_.dr)
        2. Analytical theoretical moments (moments, covariance, correlation, autocorrelations, FEVD)
        3. Impulse response functions to a one-standard-deviation innovation in each shock
        4. Simulated sample moments (if periods > 0), with Monte Carlo SEs if simul_replic > 0.

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
        hp_filter : float, optional
            Hodrick-Prescott filter smoothing parameter lambda.
        bandpass_filter : tuple of (float, float), optional
            Baxter-King bandpass filter periodicities (low, high).
        one_sided_hp_filter : bool or float, optional
            One-sided HP filter. When periods > 0, applies recursive forward
            Kalman filter to simulated trajectories.
        simul_replic : int, default 0
            Number of simulation replications. If > 0, runs M independent
            replications and computes Monte Carlo standard errors.
        contemporaneous_correlation : bool, default True
            Whether to compute and expose contemporaneous correlation matrix.
        ar : int, optional
            Number of autocorrelation lags (overrides ``lags`` if specified).
        qz_criterium : float, default 1.0 + 1e-8
            Eigenvalue cutoff for generalized Schur decomposition.

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
                lags=lags if ar is None else int(ar),
            )
        elif order != 1:
            raise ValueError(f"unsupported perturbation order {order}; must be 1 or 2")

        if periods == 0 and (one_sided_hp_filter is not None and one_sided_hp_filter is not False):
            raise ValueError(
                "disp_th_moments:: theoretical moments incompatible with one-sided HP filter. "
                "Use simulated moments instead."
            )

        if simul_replic < 0:
            raise ValueError(f"simul_replic must be non-negative, got {simul_replic}")
        if simul_replic > 0 and periods <= 0:
            raise ValueError(f"simul_replic > 0 requires periods > 0, got periods={periods}")

        n_filters = (
            (hp_filter is not None)
            + (bandpass_filter is not None)
            + bool(one_sided_hp_filter is not None and one_sided_hp_filter is not False)
        )
        if n_filters > 1:
            raise ValueError(
                "Only one filter can be specified among hp_filter, one_sided_hp_filter, and bandpass_filter."
            )

        if hp_filter is not None and hp_filter <= 0:
            raise ValueError(f"hp_filter parameter lambda must be positive, got {hp_filter}")

        if bandpass_filter is not None:
            bp = tuple(bandpass_filter)
            if len(bp) != 2 or bp[0] <= 0 or bp[1] <= bp[0]:
                raise ValueError(f"bandpass_filter requires 0 < low < high, got {bandpass_filter}")

        if ar is not None:
            if ar < 0:
                raise ValueError(f"ar must be a non-negative integer, got {ar}")
            lags = int(ar)

        self._require_solution("stoch_simul()")
        sd = self._shock_sd(sigma)
        sigma_full = sigma if sigma is None else dict(zip(self.shocks, sd))

        dr = self.decision_rules()

        theo = None
        if not (periods > 0 and (one_sided_hp_filter is not None and one_sided_hp_filter is not False)):
            theo = self.theoretical_moments(
                sigma=sigma_full,
                lags=lags,
                hp_filter=hp_filter,
                bandpass_filter=bandpass_filter,
                one_sided_hp_filter=None,
                contemporaneous_correlation=contemporaneous_correlation,
                ar=ar,
            )

        irfs: dict[str, pd.Series] = {}
        if irf > 0:
            for j, sh in enumerate(self.shocks):
                df_irf = self.irf(shock=sh, horizon=irf, size=float(sd[j]))
                for v in self.variables:
                    irfs[f"{v}_{sh}"] = df_irf[v]

        sim_moments = None
        sim_corr = None
        if periods > 0:
            var_names = list(self.variables)
            ss_level = np.array([float(self.steady_state.get(v, 0.0)) for v in var_names])
            is_filtered = (
                (hp_filter is not None)
                or (bandpass_filter is not None)
                or bool(one_sided_hp_filter is not None and one_sided_hp_filter is not False)
            )
            from puremacro.dsge._moments import one_sided_hp_filter as one_sided_hp_func

            def _filter_dataframe(df: pd.DataFrame) -> pd.DataFrame:
                if one_sided_hp_filter is not None and one_sided_hp_filter is not False:
                    lamb = 1600.0 if isinstance(one_sided_hp_filter, bool) else float(one_sided_hp_filter)
                    c_df, _ = one_sided_hp_func(df, lamb=lamb)
                    return c_df
                elif hp_filter is not None:
                    from puremacro.data import hp_filter as two_sided_hp

                    res_df = df.copy()
                    for col in res_df.columns:
                        c_series, _ = two_sided_hp(res_df[col], lamb=float(hp_filter))
                        res_df[col] = c_series.to_numpy()
                    return res_df
                elif bandpass_filter is not None:
                    from puremacro.cycles import baxter_king_filter

                    low, high = bandpass_filter
                    res_df = df.copy()
                    for col in res_df.columns:
                        res_df[col] = baxter_king_filter(res_df[col], low=low, high=high).cycle
                    return res_df
                return df

            if simul_replic == 0:
                sim_df = self.simulate(periods=periods, sigma=sigma_full, seed=seed, burn=burn)
                sim_df = _filter_dataframe(sim_df)
                sim_means = sim_df.mean(axis=0, skipna=True)
                if not is_filtered:
                    sim_means = sim_means + ss_level
                sim_moments = pd.DataFrame(
                    {
                        "Mean": sim_means,
                        "Std.Dev.": sim_df.std(axis=0, skipna=True),
                        "Variance": sim_df.var(axis=0, skipna=True),
                        "Skewness": sim_df.skew(axis=0, skipna=True),
                        "Kurtosis": sim_df.kurtosis(axis=0, skipna=True),
                    },
                    index=var_names,
                )
                sim_corr = sim_df.corr()
            else:
                M = int(simul_replic)
                master_rng = np.random.default_rng(seed)
                n_vars = len(var_names)
                rep_means = np.zeros((M, n_vars))
                rep_stds = np.zeros((M, n_vars))
                rep_vars = np.zeros((M, n_vars))
                rep_skews = np.zeros((M, n_vars))
                rep_kurts = np.zeros((M, n_vars))
                corr_list = []

                from scipy.stats import kurtosis as sp_kurtosis, skew as sp_skew

                for m in range(M):
                    rep_df = self.simulate(periods=periods, sigma=sigma_full, seed=master_rng, burn=burn)
                    rep_df = _filter_dataframe(rep_df)
                    rep_arr = rep_df.to_numpy()

                    m_mean = np.nanmean(rep_arr, axis=0)
                    if not is_filtered:
                        m_mean = m_mean + ss_level
                    rep_means[m] = m_mean

                    rep_stds[m] = np.nanstd(rep_arr, axis=0, ddof=1)
                    rep_vars[m] = np.nanvar(rep_arr, axis=0, ddof=1)
                    rep_skews[m] = sp_skew(rep_arr, axis=0, nan_policy="omit")
                    rep_kurts[m] = sp_kurtosis(rep_arr, axis=0, nan_policy="omit")

                    c_mat = rep_df.corr().to_numpy()
                    corr_list.append(c_mat)

                mean_avg = np.mean(rep_means, axis=0)
                std_avg = np.mean(rep_stds, axis=0)
                var_avg = np.mean(rep_vars, axis=0)
                skew_avg = np.mean(rep_skews, axis=0)
                kurt_avg = np.mean(rep_kurts, axis=0)

                se_mean = np.std(rep_means, axis=0, ddof=1) / np.sqrt(M)
                se_var = np.std(rep_vars, axis=0, ddof=1) / np.sqrt(M)
                se_std = np.std(rep_stds, axis=0, ddof=1) / np.sqrt(M)

                sim_moments = pd.DataFrame(
                    {
                        "Mean": mean_avg,
                        "Std.Dev.": std_avg,
                        "Variance": var_avg,
                        "Skewness": skew_avg,
                        "Kurtosis": kurt_avg,
                        "MC Std.Err.": se_mean,
                    },
                    index=var_names,
                )
                sim_moments.attrs["simul_replic"] = M
                sim_moments.attrs["mc_se_mean"] = pd.Series(se_mean, index=var_names)
                sim_moments.attrs["mc_se_var"] = pd.Series(se_var, index=var_names)
                sim_moments.attrs["mc_se_std"] = pd.Series(se_std, index=var_names)

                avg_corr = np.nanmean(np.stack(corr_list, axis=0), axis=0)
                np.fill_diagonal(avg_corr, 1.0)
                avg_corr = np.clip(avg_corr, -1.0, 1.0)
                sim_corr = pd.DataFrame(avg_corr, index=var_names, columns=var_names)

        return StochSimulResult(
            dr=dr,
            theoretical_moments=theo,
            simulated_moments=sim_moments,
            irfs=irfs,
            order=1,
            variable_names=tuple(self.variables),
            shock_names=tuple(self.shocks),
            _sim_corr=sim_corr,
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

    def solve(
        self,
        order: int = 1,
        *,
        shock_cov: np.ndarray | None = None,
        qz_criterium: float | None = None,
    ):
        """Return the solution at the requested perturbation order.

        Parameters
        ----------
        order : {1, 2}, default 1
            ``1`` returns this (already solved) first-order model itself (or re-solved
            if qz_criterium is specified);
            ``2`` returns the pruned second-order solution from
            :meth:`solve_second_order`.
        shock_cov : np.ndarray, optional
            Innovation covariance used for the second-order risk correction.
            Defaults to the declared shock covariance, else the identity.
        qz_criterium : float, optional
            Stability threshold override for generalised Schur (QZ) root sorting.

        Returns
        -------
        LinearModel | PrunedDSGESolution
        """
        if order == 1:
            if qz_criterium is None and shock_cov is None:
                return self
            if qz_criterium is not None:
                div = float(qz_criterium)
                if self.timing == "dynare":
                    n_s = len(self.states)
                    sol_full = klein_solve(
                        self.A, self.B, n_pre=n_s, C=self.C, strict=True, div=div,
                    )
                    F_full = np.asarray(sol_full.F, dtype=float)
                    L_full = np.asarray(sol_full.L, dtype=float)
                    state_idx = [self.variables.index(v) for v in self.states]
                    ctrl_idx = [self.variables.index(v) for v in self.controls]
                    G = F_full[state_idx]
                    N = L_full[state_idx]
                    F = F_full[ctrl_idx]
                    L = L_full[ctrl_idx]
                    sol = KleinSolution(
                        G=G, F=F, N=N, L=L, eu=tuple(sol_full.eu), eigenvalues=sol_full.eigenvalues,
                    )
                    import dataclasses
                    res = dataclasses.replace(self, solution=sol)
                    object.__setattr__(res, "_qz_criterium", qz_criterium)
                    return res
                else:
                    sol = klein_solve(
                        self.A, self.B, n_pre=len(self.states), C=self.C, strict=True, div=div,
                    )
                    import dataclasses
                    res = dataclasses.replace(self, solution=sol)
                    object.__setattr__(res, "_qz_criterium", qz_criterium)
                    return res
            return self
        if order == 2:
            return self.solve_second_order(shock_cov=shock_cov)
        raise ValueError(f"unsupported perturbation order {order}; must be 1 or 2")

    @property
    def shock_groups(self) -> dict[str, list[str]]:
        """Dictionary of parsed shock groups if defined in .mod file."""
        return getattr(self, "_shock_groups", {}) or (
            getattr(self._dag, "shock_groups", {}) if hasattr(self, "_dag") and self._dag else {}
        )

    def conditional_forecast(
        self,
        conditions: Mapping[str, Sequence[float | None]] | None = None,
        *,
        target_paths: Mapping[str, Sequence[float | None]] | None = None,
        horizon: int | None = None,
        controlled_shocks: Sequence[str] | None = None,
        x0: np.ndarray | Mapping[str, float] | None = None,
        shock_cov: np.ndarray | None = None,
        method: str = "covariance_weighted",
        exact: bool = True,
        ci: float | Sequence[float] | None = None,
        n_sims: int = 0,
        seed: int = 0,
    ):
        """Compute conditional forecast using Waggoner & Zha (1999) shock inversion."""
        from .conditional import conditional_forecast as _cond_fc
        return _cond_fc(
            self,
            conditions=conditions,
            target_paths=target_paths,
            horizon=horizon,
            controlled_shocks=controlled_shocks,
            x0=x0,
            shock_cov=shock_cov,
            method=method,
            exact=exact,
            ci=ci,
            n_sims=n_sims,
            seed=seed,
        )

    def shock_groups_decomposition(
        self,
        data: Any,
        groups: Mapping[str, Sequence[str]] | None = None,
        *,
        initial_state: np.ndarray | None = None,
        sigma: float | Mapping[str, float] | Sequence[float] | None = None,
    ):
        """Compute grouped historical and forecast shock decomposition."""
        from .shock_groups import shock_groups_decomposition as _sg_decomp
        return _sg_decomp(
            self,
            data=data,
            groups=groups,
            initial_state=initial_state,
            sigma=sigma,
        )

    def bayesian_irf(
        self,
        draws: Any = None,
        param_names: Sequence[str] | None = None,
        *,
        horizon: int = 40,
        bands: Sequence[float] = (0.68, 0.90, 0.95),
        quantiles: Sequence[float] | None = None,
        shock: str | None = None,
        variables: Sequence[str] | None = None,
        size: float = 1.0,
        burn_in: int = 0,
        qz_criterium: float = 1.0 + 1e-8,
    ):
        """Compute Bayesian posterior impulse response functions and fan chart bands."""
        from .bayesian import bayesian_irf as _b_irf
        return _b_irf(
            self,
            draws=draws,
            param_names=param_names,
            horizon=horizon,
            bands=bands,
            quantiles=quantiles,
            shock=shock,
            variables=variables,
            size=size,
            burn_in=burn_in,
            qz_criterium=qz_criterium,
        )

    def prior_predictive(
        self,
        priors: Mapping[str, Any] | None = None,
        *,
        n_draws: int = 500,
        draws: int | None = None,
        seed: int = 0,
        moments: bool = True,
        irf: int | None = 40,
        qz_criterium: float = 1.0 + 1e-8,
    ):
        """Perform prior predictive simulation directly from prior distributions."""
        from .bayesian import prior_predictive as _p_pred
        return _p_pred(
            self,
            priors=priors,
            n_draws=n_draws,
            draws=draws,
            seed=seed,
            moments=moments,
            irf=irf,
            qz_criterium=qz_criterium,
        )

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


def _make_observation_eq(model, specs, varobs, *, fixed_params=None,
                         measurement_error=None, prefilter=False, ridge=0.0):
    """``theta -> StateSpaceModel``, re-solving only when the model changes.

    ``EstimatedParamSpec.kind`` decides that: ``"param"`` enters the model's
    equations and needs a re-solve; ``"stderr_shock"``, ``"corr_shock"`` and
    ``"stderr_obs"`` are written straight into ``Q`` or ``H``.

    What that is worth, measured on SW07 rather than assumed: a
    finite-difference sweep over its 36 estimated parameters takes 37
    evaluations and 30 solves, because 7 of the 36 are shock scale parameters
    that cannot change the solved model — 19% avoided. **For the Metropolis
    chain itself the saving is zero**, since a random-walk proposal moves every
    parameter at once and so changes the structural block on every draw. The
    cache pays during the mode search and would pay far more under a blocked
    or single-site sampler.

    The cache holds one entry. A Metropolis chain does not revisit an exact
    structural vector, so depth 1 captures the case that matters (a sweep that
    perturbed only a shock parameter) without growing without bound.

    ``observation_eq.n_solves`` is a documented test hook, not incidental
    state.
    """
    from .observation import make_state_space_from_varobs

    if model._dynare_equations is None:
        raise ModelError(
            "estimate() needs a model built with build_dynare or load_mod: "
            "re-solving at each draw requires the canonical lead-lag equations."
        )

    structural = [s.name for s in specs if s.kind == "param"]
    unknown = [n for n in structural if n not in (model._params or {})]
    if unknown:
        raise ModelError(
            f"_make_observation_eq: {unknown} are marked as structural "
            "parameters but the model has no such parameters, so varying them "
            "could not change anything the solver sees. Model parameters: "
            f"{sorted(model._params or {})}."
        )
    se_shock = {s.name: s.target[0] for s in specs if s.kind == "stderr_shock"}
    corr_shock = {s.name: s.target for s in specs if s.kind == "corr_shock"}
    me_obs = {s.name: s.target[0] for s in specs if s.kind == "stderr_obs"}

    base_params = dict(model._params or {})
    if fixed_params:
        base_params.update({k: float(v) for k, v in fixed_params.items()})
    shocks = list(model.shocks)
    n_e = len(shocks)
    base_cov = (np.asarray(model._shock_cov, dtype=float)
                if model._shock_cov is not None else np.eye(n_e))
    base_me = dict(measurement_error or {})

    cache: dict = {}

    def observation_eq(params):
        key = tuple(float(params[n]) for n in structural)
        solved = cache.get(key)
        if solved is None:
            cache.clear()
            from .dynare import build_dynare

            merged = dict(base_params)
            merged.update({n: float(params[n]) for n in structural})
            solved = build_dynare(
                model._dynare_equations,
                variables=model.variables,
                shocks=model.shocks,
                params=merged,
                # `guess=`, never `steady_state=`: the previous draw's steady
                # state is a starting point, and handing it over as exact would
                # fail the residual check at every new parameter vector.
                guess=observation_eq._last_ss,
                states=model.states,
                order=1,
                method=model.method,
                # The equations are the ones this model was already built and
                # verified with; re-checking the Jacobians on every draw would
                # double the cost to re-learn the same answer.
                verify_derivatives=False,
                strict=True,
            )
            cache[key] = solved
            observation_eq._last_ss = solved.steady_state.to_dict()
            observation_eq.n_solves += 1

        cov = base_cov.copy()
        for name, sh in se_shock.items():
            i = shocks.index(sh)
            cov[i, i] = float(params[name]) ** 2
        for name, (s1, s2) in corr_shock.items():
            i, j = shocks.index(s1), shocks.index(s2)
            c = float(params[name]) * math.sqrt(max(cov[i, i], 0.0) * max(cov[j, j], 0.0))
            cov[i, j] = cov[j, i] = c

        me = dict(base_me)
        for name, v in me_obs.items():
            me[v] = float(params[name])

        return make_state_space_from_varobs(
            solved, varobs, shock_cov=cov,
            measurement_error=me or None, prefilter=prefilter, ridge=ridge,
        )

    observation_eq.n_solves = 0
    observation_eq._last_ss = model.steady_state.to_dict()
    return observation_eq


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
    "The usual cause is a residual function that is not analytic — abs() / "
    "np.abs(), np.linalg.norm(), a np.real()/float() cast that throws the "
    "imaginary part away, or a fractional power of a quantity that goes "
    "negative under perturbation. Rewrite the offending term. Where the "
    "residual is genuinely smooth at the steady state and only the complex "
    "arithmetic is broken (abs(x) evaluated far from x = 0), method='central' "
    "gives the right derivative to ~1e-8 instead of ~1e-15; at an actual kink "
    "neither scheme is right — an occasionally-binding constraint belongs in "
    "puremacro.dsge.occbin."
)

# Cross-check settings.
#
# _VERIFY_SEED seeds the probe directions. It is fixed so that build() is
# bit-for-bit reproducible across runs and machines; nothing about the
# result depends on the particular value, only on the directions having no
# systematically small component (see _probe_directions).
_VERIFY_SEED = 8675309
_VERIFY_PROBES = 2
# Relative tolerance on one entry of the directional derivative.
_VERIFY_RTOL = 1e-4
# How much rounding noise a finite difference is allowed, in multiples of
# eps * |f|: the quotient (f(x+d) - f(x-d))/2 cannot be more accurate than
# the representation error of the two residual values it subtracts.
_FD_NOISE = 64.0
_EPS = float(np.finfo(float).eps)


def _probe_directions(n: int) -> list:
    """Deterministic probe directions with no systematically small entry.

    Drawn from ``np.random.default_rng(_VERIFY_SEED)`` so that two builds
    of the same model agree bit for bit — the package's determinism
    contract — but with the magnitudes forced into ``[0.5, 1.5]`` and only
    the signs left to the draw.

    A plain standard-normal direction (what this used to be) is unusable
    here: whichever component happens to come out near zero is a *permanent*
    blind spot shared by every model, because the seed is fixed. With
    ``default_rng(0)`` that was component 52 at 4.45e-3, 200x smaller than
    its neighbours, so an error in column 52 of any model with 53 or more
    columns entered the comparison at 1/200 weight.
    """
    rng = np.random.default_rng(_VERIFY_SEED)
    return [np.where(rng.standard_normal(n) < 0.0, -1.0, 1.0)
            * (0.5 + rng.random(n)) for _ in range(_VERIFY_PROBES)]


def _fd_directional(f: Callable, x0: np.ndarray, d: np.ndarray):
    """Central difference of ``f`` along the displacement ``d``.

    Returns ``(fd, noise)`` where ``fd ≈ J @ d`` — deliberately *not*
    divided by the step, so that it is compared against ``jac @ d`` on the
    scale the residual values themselves were computed on — and ``noise``
    is the rounding error of that estimate, one entry per equation.
    ``(None, None)`` if either evaluation is not finite: a NaN must never
    be allowed to read as "agrees".
    """
    up = np.asarray(f(x0 + d), dtype=float)
    dn = np.asarray(f(x0 - d), dtype=float)
    if not (np.all(np.isfinite(up)) and np.all(np.isfinite(dn))):
        return None, None
    return 0.5 * (up - dn), _FD_NOISE * _EPS * np.maximum(np.abs(up), np.abs(dn))


def _worst_column(f: Callable, x0: np.ndarray, jac: np.ndarray, row: int) -> int:
    """Which argument the flagged equation's derivative is wrong in.

    Only ever called on the way to raising, so its cost (one central
    difference per column, of one equation) does not matter.
    """
    worst, worst_gap = -1, -1.0
    for j in range(jac.shape[1]):
        h = np.zeros_like(x0)
        h[j] = _FDSTEP * max(1.0, abs(float(x0[j])))
        try:
            up = float(np.asarray(f(x0 + h), dtype=float)[row])
            dn = float(np.asarray(f(x0 - h), dtype=float)[row])
        except Exception:                                  # pragma: no cover
            continue
        if not (np.isfinite(up) and np.isfinite(dn)):
            continue
        gap = abs(jac[row, j] * h[j] - 0.5 * (up - dn))
        if gap > worst_gap:
            worst, worst_gap = j, gap
    return worst


def _verify_jacobian(label: str, f: Callable, x0: np.ndarray,
                     jac: np.ndarray, names: Sequence[str] | None = None) -> None:
    """Cross-check one complex-step Jacobian against finite differences.

    Complex-step is exact *if* the residual function is analytic, and
    silently wrong if it is not — ``Im f(x + ih)`` is identically zero
    through an ``abs()``, so the derivative comes back as zero with no
    error anywhere. Finite differences have no such blind spot, and
    disagreeing with them is the signature of the failure.

    The comparison is made **per equation and relative to that equation's
    own derivative**, with a floor at the finite difference's own rounding
    noise. A single block-wide absolute tolerance (what this used to be)
    checks nothing in a model that mixes scales: one variable in dollars,
    or one equation written 1e-4 times smaller than its neighbours, lifts
    the threshold above every other equation's derivative.

    The finite difference is not treated as an infallible oracle either:
    see :func:`_adjudicate_gap`, which re-probes a flagged equation at
    eight times and an eighth of the step and only believes the
    disagreement if the finite difference reproduces itself better than
    it reproduces the complex step.

    Costs four extra evaluations per block (two probe directions), plus a
    handful more on the way to a diagnosis.
    """
    if jac.size == 0:
        return
    x0 = np.asarray(x0, dtype=float)
    col_scale = np.maximum(1.0, np.abs(x0))
    for v in _probe_directions(jac.shape[1]):
        d = _FDSTEP * col_scale * v
        fd, noise = _fd_directional(f, x0, d)
        shrink = 1.0
        while fd is None and shrink > 1e-8:
            # A model defined only on part of the real line (a fractional
            # power, a log of a quantity whose steady state is 0) can be
            # impossible to probe at the default step. Try closer in
            # before giving up — but never fall through the comparison.
            shrink *= 1e-3
            fd, noise = _fd_directional(f, x0, d * shrink)
        if fd is None:
            warnings.warn(
                f"build: could not cross-check the complex-step Jacobian of "
                f"the {label} block — equations() is not finite at any "
                f"finite-difference probe around the steady state, so a "
                f"non-analytic term in this block would go undetected. "
                f"Check that block by hand, or pass method='central'.",
                UserWarning, stacklevel=3,
            )
            return
        d = d * shrink
        cs = jac @ d
        gap = np.abs(cs - fd)
        tol = np.maximum(_VERIFY_RTOL * np.maximum(np.abs(cs), np.abs(fd)), noise)
        flagged = np.flatnonzero(gap > tol)
        if flagged.size:
            _adjudicate_gap(label, f, x0, jac, d, cs, fd, gap, tol, flagged, names)


def _adjudicate_gap(label, f, x0, jac, d, cs, fd, gap, tol, flagged, names) -> None:
    """Decide whether a flagged disagreement is the model's fault or the FD's.

    The finite difference is not an infallible oracle. Its rounding error
    is fixed in absolute terms, so relative to the derivative it grows as
    the step shrinks; its truncation error falls off as the step squared;
    and a residual with heavy subtractive cancellation
    (``(c + k + 1e9) - 1e9 - y``: analytic, exactly the case complex-step
    exists for, but impossible to difference) can be pure quantisation
    noise at every step. In all three the finite difference disagrees
    with *itself*.

    So the probe is repeated at ``8 d`` and ``d / 8``, the three
    estimates of ``J @ d`` are put on the same scale, and the complex
    step is called wrong only when it sits further from them than they
    sit from each other. Otherwise the check reports itself inconclusive
    and the complex-step Jacobian — the accurate one in exactly these
    cases — is kept.
    """
    big, _ = _fd_directional(f, x0, 8.0 * d)
    fine, _ = _fd_directional(f, x0, d / 8.0)
    confirmed: list = []
    spread = None
    if big is not None and fine is not None:
        estimates = np.stack([big / 8.0, fd, 8.0 * fine])
        spread = estimates.max(axis=0) - estimates.min(axis=0)
        confirmed = [int(i) for i in flagged if gap[i] > 4.0 * spread[i]]
    if not confirmed:
        worst = int(flagged[np.argmax(gap[flagged] / np.maximum(
            np.maximum(np.abs(cs[flagged]), np.abs(fd[flagged])),
            np.finfo(float).tiny))])
        why = ("the residual is not finite at a second step size"
               if spread is None else
               f"the finite difference disagrees with itself by "
               f"{spread[worst]:.2e} across step sizes, against a "
               f"disagreement of {gap[worst]:.2e} with the complex step, so "
               f"it is the finite difference that is ill-conditioned here")
        warnings.warn(
            f"build: the complex-step/finite-difference cross-check on the "
            f"{label} block is inconclusive for equation(s) "
            f"{[int(i) for i in flagged]} — {why}. The complex-step Jacobian "
            f"was accepted unchecked for those equations. A large additive "
            f"constant in the residual is the usual reason; subtracting it "
            f"restores the check.",
            UserWarning, stacklevel=4,
        )
        return
    row = int(max(confirmed, key=lambda i: gap[i] / max(tol[i], np.finfo(float).tiny)))
    col = _worst_column(f, x0, jac, row)
    where = ""
    if col >= 0:
        col_name = (repr(names[col]) if names is not None and col < len(names)
                    else f"index {col}")
        where = (f" The derivative with respect to {col_name} is the largest "
                 f"single contributor.")
    raise ModelError(
        f"complex-step and finite-difference derivatives disagree for the "
        f"{label} block: equation {row} gives {cs[row]:.6e} by complex step "
        f"against {fd[row]:.6e} by finite difference (gap {gap[row]:.2e}, "
        f"tolerance {tol[row]:.2e}), while the finite difference reproduces "
        f"itself to {spread[row]:.2e} across three step sizes."
        f"{where} {_NOT_ANALYTIC}"
    )


def _check_analytic(blocks: dict, method: str) -> None:
    """Validate complex-step Jacobians before they become a solved model.

    ``blocks`` maps a label to ``(jacobian, function, base_point)``, with
    an optional fourth element naming the columns.
    """
    if method != "complex":
        return
    for label, spec in blocks.items():
        J, names = spec[0], (spec[3] if len(spec) > 3 else None)
        if J.size == 0:
            continue
        if not np.any(J):
            if label == "shock" and names is not None:
                raise ModelError(
                    f"none of the declared shocks {list(names)} enters any "
                    f"equation: the Jacobian of the shock block is identically "
                    f"zero, so every IRF and every variance decomposition "
                    f"would be zero. Reference the innovation in an equation "
                    f"(e.g. `xp.z - p.rho * x.z - e.{names[0]}`), or drop it "
                    f"from shocks=. If it is referenced, then the derivative "
                    f"is being lost instead: {_NOT_ANALYTIC}"
                )
            raise ModelError(
                f"complex-step differentiation produced an all-zero Jacobian "
                f"for the {label} block. {_NOT_ANALYTIC}"
            )
        if label == "shock" and names is not None:
            dead = [names[j] for j in np.flatnonzero(~np.any(J, axis=0))]
            if dead:
                warnings.warn(
                    f"build: shock(s) {dead} enter no equation — their IRFs "
                    f"and their share of every variance decomposition are "
                    f"identically zero. Drop them from shocks=, or check for "
                    f"a typo in the equation that should use them.",
                    UserWarning, stacklevel=3,
                )
    for label, spec in blocks.items():
        J, f, x0 = spec[0], spec[1], spec[2]
        _verify_jacobian(label, f, x0, J, spec[3] if len(spec) > 3 else None)


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


# The root finder's own tolerance, deliberately independent of the caller's
# `tol`. `tol` is the *acceptance* tolerance on max|f(ss, ss, 0)|; handing
# it to hybr as well meant a loose tol bought a sloppy steady state that
# then passed its own widened gate, and every downstream derivative
# inherited the error.
_SS_XTOL = 1e-12


def _check_equation_count(resid: np.ndarray, n: int) -> None:
    """One residual per variable, or a message naming what came back."""
    if resid.shape != (n,):
        raise ModelError(
            f"equations() returned {resid.shape[0] if resid.ndim else 1} "
            f"residual(s) for {n} variables — a square system needs one "
            f"equation per variable"
        )


def _solve_steady_state(f, variables, shocks, params, guess, tol,
                        solve_algo: str = "block",
                        homotopy: Mapping[str, tuple[float, float]] | None = None,
                        homotopy_steps: int = 10) -> np.ndarray:
    from .steady import steady
    ss, _ = steady(
        f,
        variables=variables,
        guess=guess,
        params=params,
        shocks=shocks,
        solve_algo=solve_algo,
        homotopy=homotopy,
        homotopy_steps=homotopy_steps,
        tol=tol,
    )
    return ss


def build(equations: Callable, *, variables: Sequence[str],
          states: Sequence[str], shocks: Sequence[str],
          params: Mapping | None = None,
          steady_state: Mapping | None = None,
          guess: Mapping | None = None,
          solve_algo: str = "block",
          homotopy: Mapping[str, tuple[float, float]] | None = None,
          homotopy_steps: int = 10,
          linearize: str = "log",
          method: str = "complex",
          verify_derivatives: bool = True,
          strict: bool = True,
          tol: float = 1e-9,
          qz_criterium: float = 1.0 + 1e-8,
          anticipated_shocks: dict | None = None) -> LinearModel:
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
        (``max|f| <= tol``, and every residual finite) rather than
        trusted.
    guess : mapping, optional
        Starting values for solving the steady state numerically.
        Required unless ``steady_state`` is given. A model with several
        steady states returns whichever root the guess leads to.
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
        on those. The comparison is per equation and relative to that
        equation's own derivative, with a floor at the finite difference's
        rounding noise; a disagreement is confirmed at a second step size
        before it is raised, so an ill-conditioned residual is reported as
        an inconclusive check (``UserWarning``) rather than as the model's
        fault. Costs twelve extra function evaluations.
    strict : bool, default True
        Raise :class:`~puremacro.dsge.klein.BlanchardKahnError` when the
        model has no unique stable solution, rather than returning zero
        matrices.
    tol : float, default 1e-9
        Acceptance tolerance on the steady state: ``build`` raises unless
        ``max|f(ss, ss, 0)| <= tol``, whether ``ss`` was supplied or
        solved for. It is *not* passed to the root finder, which always
        works to ~1e-12 — a loose ``tol`` widens what is accepted, it does
        not buy a sloppier solve.

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
        # Check the equation count at the guess before handing the
        # function to the root finder: scipy reports a wrong count as a
        # TypeError about the shape of '_wrapped_fun', which names
        # nothing the caller wrote.
        guess_vec = _Vec(variables, np.array([float(guess[v]) for v in variables]))
        _check_equation_count(
            np.asarray(equations(guess_vec, guess_vec,
                                 _Vec(shocks, np.zeros(n_e), "shock"), par),
                       dtype=float), n)
        ss = _solve_steady_state(
            equations, variables, shocks, par, guess, tol,
            solve_algo=solve_algo, homotopy=homotopy, homotopy_steps=homotopy_steps,
        )
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
    _check_equation_count(resid0, n)
    # Finiteness first: `nan > tol` is False, so a NaN residual would
    # otherwise walk straight through the gate below and be reported as a
    # solved, determinate model. (Complex-step keeps going where real
    # arithmetic gives up — k**(alpha-1) at k < 0 is nan in float but
    # takes the principal branch in complex — so the Jacobians come out
    # finite and meaningless.)
    if not np.all(np.isfinite(resid0)):
        bad = int(np.argmax(~np.isfinite(resid0)))
        raise SteadyStateError(
            f"equations() does not return a finite residual at the steady "
            f"state: equation index {bad} evaluates to {resid0[bad]}. Steady "
            f"state: {({v: float(x) for v, x in zip(variables, ss)})}. A "
            f"non-finite residual cannot "
            f"be verified, so the model is rejected rather than linearised "
            f"around a point that is not a steady state."
        )
    residual_norm = float(np.max(np.abs(resid0)))
    if residual_norm > tol:
        raise SteadyStateError(
            f"the supplied steady state does not solve the model: "
            f"max|f(ss, ss, 0)| = {residual_norm:.3e} > tol = {tol:.3e}. "
            f"Worst equation: index {int(np.argmax(np.abs(resid0)))}."
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
    for label, J, cols in (("t+1", Fp, variables), ("t", Fc, variables),
                           ("shock", Fu, shocks)):
        if J.size and not np.all(np.isfinite(J)):
            i, j = np.unravel_index(
                int(np.argmax(~np.isfinite(J))), J.shape)
            raise ModelError(
                f"the {label} Jacobian of equations() is not finite: the "
                f"derivative of equation {int(i)} with respect to "
                f"{cols[int(j)]!r} came out {J[i, j]}. Differentiating at a "
                f"point where the residual is not differentiable (a negative "
                f"base under a fractional power, a log of a non-positive "
                f"quantity, a division by zero) gives no usable model."
            )
    if verify_derivatives:
        _check_analytic({
            "t+1": (Fp, f_of_next, zeros_n, variables),
            "t": (Fc, f_of_now, zeros_n, variables),
            "shock": (Fu, f_of_shock, zeros_e, shocks),
        }, method)

    # f = 0 linearised is  Fp z' + Fc z + Fu u = 0, i.e. Klein's
    # A E_t z' = B z + C u with A = Fp, B = -Fc, C = -Fu.
    order = [variables.index(v) for v in (*states, *controls)]
    A = Fp[:, order]
    B = -Fc[:, order]
    C = -Fu

    solution = klein_solve(A, B, len(states), C, strict=strict, div=qz_criterium)

    model = LinearModel(
        # tuple(...) is a no-op at run time (_validate_names already
        # returned tuples) and keeps the declared Sequence[str] parameter
        # types from leaking into the frozen dataclass's tuple fields.
        variables=tuple(variables), states=tuple(states), controls=controls,
        shocks=tuple(shocks),
        steady_state=pd.Series(ss, index=list(variables)), units=units,
        solution=solution, A=A, B=B, C=C, method=method,
        residual_norm=residual_norm,
        _equations=equations,
        _params=dict(params or {}),
        anticipated_shocks=anticipated_shocks,
    )
    object.__setattr__(model, "_qz_criterium", qz_criterium)
    return model
