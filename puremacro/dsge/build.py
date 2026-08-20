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
import scipy.optimize

from puremacro.dsge.klein import KleinSolution, klein_solve

__all__ = [
    "ModelError",
    "SteadyStateError",
    "LinearModel",
    "build",
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

    def policy(self) -> pd.DataFrame:
        """The decision rules as a labelled table.

        Rows are variables, columns are states: entry ``(i, j)`` is the
        response of variable ``i`` to a one-unit deviation of state
        ``j``. States map to themselves through ``G``, controls through
        ``F``.
        """
        block = np.vstack([self.solution.G, self.solution.F])
        frame = pd.DataFrame(
            block, index=list(self.states) + list(self.controls),
            columns=list(self.states),
        )
        return frame.loc[list(self.variables)]

    # -- simulation ----------------------------------------------------

    def _paths(self, horizon: int, impulse: np.ndarray) -> np.ndarray:
        """MA response of [states; controls] to ``impulse`` at t=0."""
        G, F, N, L = (self.solution.G, self.solution.F,
                      self.solution.N, self.solution.L)
        n_s, n_c = self.n_states, self.n_controls
        xs = np.zeros((horizon + 1, n_s))
        ys = np.zeros((horizon + 1, n_c))
        ys[0] = L @ impulse
        for h in range(1, horizon + 1):
            xs[h] = G @ xs[h - 1] + (N @ impulse if h == 1 else 0.0)
            ys[h] = F @ xs[h]
        return np.hstack([xs, ys])

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
            and level deviations otherwise. See the module docstring on
            the timing convention: states are zero in the ``h=0`` row for
            a standard AR(1) driving process, while forward-looking
            controls generally are not.
        """
        if shock not in self.shocks:
            raise ModelError(
                f"no shock named {shock!r}; declared: {list(self.shocks)}"
            )
        if horizon < 0:
            raise ValueError(f"horizon must be non-negative, got {horizon}")
        impulse = np.zeros(len(self.shocks))
        impulse[self.shocks.index(shock)] = size
        paths = self._paths(horizon, impulse)
        frame = pd.DataFrame(
            paths, columns=list(self.states) + list(self.controls),
        )
        frame.index.name = "h"
        return frame[list(self.variables)]

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
            Default 1.0 for every shock.
        seed : int, default 0
            Seed for ``numpy.random.default_rng``.
        burn : int, default 100
            Discarded initial periods.

        Returns
        -------
        pandas.DataFrame
            ``periods`` rows, one column per variable, in the same
            deviation units as :meth:`irf`.
        """
        if sigma is None:
            sd = np.ones(len(self.shocks))
        elif isinstance(sigma, Mapping):
            sd = np.array([float(sigma.get(s, 0.0)) for s in self.shocks])
        else:
            sd = np.full(len(self.shocks), float(sigma))

        rng = np.random.default_rng(seed)
        total = periods + burn
        shocks = rng.standard_normal((total, len(self.shocks))) * sd

        G, F, N, L = (self.solution.G, self.solution.F,
                      self.solution.N, self.solution.L)
        xs = np.zeros((total + 1, self.n_states))
        ys = np.zeros((total, self.n_controls))
        for t in range(total):
            ys[t] = F @ xs[t] + L @ shocks[t]
            xs[t + 1] = G @ xs[t] + N @ shocks[t]
        block = np.hstack([xs[:total], ys])
        frame = pd.DataFrame(block, columns=list(self.states) + list(self.controls))
        return frame.iloc[burn:].reset_index(drop=True)[list(self.variables)]

    def summary(self) -> str:
        """One-screen description: sizes, steady state, BK verdict."""
        eu = self.solution.eu
        verdict = {
            (1, 1): "unique stable solution",
            (1, 0): "indeterminate (multiple stable solutions)",
            (0, 0): "no stable solution (Blanchard-Kahn violated)",
        }.get(tuple(eu), f"eu={tuple(eu)}")
        lines = [
            f"LinearModel · {len(self.variables)} variables "
            f"({self.n_states} states, {self.n_controls} controls), "
            f"{len(self.shocks)} shock(s)",
            f"  jacobians    : {self.method}-step",
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
