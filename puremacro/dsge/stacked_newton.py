r"""Matrix-free stacked-time Newton-Krylov for perfect-foresight systems.

This module solves the same two-point boundary problem as
:func:`puremacro.dsge.solve_perfect_foresight`,

    F_t(Y) = f(y_{t+1}, y_t, y_{t-1}, eps_t) = 0,      t = 1, ..., T,
    y_0 = y_init (fixed),   y_{T+1} = y_end (fixed terminal state),

without ever assembling the stacked Jacobian. It is a model-agnostic port of
the numerical core of the IO research engine ``dynamic_model/native_numerics.py``
(``PathProblem``), ``native_solver.py`` (``_newton``, ``compare_horizons``),
``native_block_preconditioner.py`` (the exact finite time-block inverse) and
``native_krylov_probe.py`` (the right-preconditioned LGMRES adapter). Nothing in
those files could be copied verbatim: every IO preconditioner is written against
the internals of one dynamic MRIO model. What is ported is the structure.

Mathematics
-----------
Stacked Jacobian. With A_t = df_t/dy_{t+1}, B_t = df_t/dy_t and C_t = df_t/dy_{t-1},
the stacked Jacobian J is block tridiagonal in time and its matrix-free action is

    (J v)_t = A_t v_{t+1} + B_t v_t + C_t v_{t-1},   v_0 = v_{T+1} = 0.

:class:`StackedProblem` evaluates the stacked residual and this product from an
analytic per-date ``jvp_fn``, from central differences of the whole stack
(two residual evaluations per product, step h = fd_step * max(1, |Y|_inf) / |v|_inf)
or from a complex step (one complex evaluation, h = 1e-20, no cancellation).

Inexact Newton. At an iterate Y with residual F, find a direction d with

    |J d + F|_2 <= eta |F|_2,    eta = max(eta_min, min(eta_max, sqrt(|F|_inf / s))),

by LGMRES (SciPy, ``inner_m`` = 40, ``outer_k`` = 6, at most ``krylov_maxiter`` = 35
outer iterations) or, when ``krylov_method="gmres"`` is requested explicitly, by
restarted GMRES (there is no automatic GMRES fallback). A direction returned at the
Krylov cap is recorded (``krylov_status > 0`` with its true linear residual) and
still line-searched, as in the IO engine. ``s`` is the residual scale
(``jacobian_scale``, by default the largest entry of the steady-state blocks,
floored at one) that also enters the convergence test below; the IO engine uses
s = 1. The step is bounded in path units,
``step = min(1, max_log_move * max(1, |Y|_inf) / |d|_inf)`` with ``max_log_move`` = 0.75
(the IO engine bounds the absolute move of its log variables by 0.75; the two rules
coincide whenever |Y|_inf <= 1, which holds on the IO fixture, and the relative form
keeps a model written in levels of order 1e3 or 1e4 converging in the same number of
Newton steps; ``None`` disables the bound), and accepted by an Armijo test on the
squared residual, ``|F(Y + step d)|_2^2 <= |F(Y)|_2^2 (1 - 1e-4 step)``, halving the
step at most ``max_backtracks`` = 23 times (24 trial evaluations, the IO ``range(24)``
rule). SciPy's LGMRES fires its callback before each
outer residual check, so a direction returned after the iteration cap was never
seen by the callback: the true linear residual ``|J d + F|_2`` of the RETURNED
direction is always recomputed in original coordinates and recorded.

Convergence. A solve is converged when ``|F|_inf <= tol * s`` and the next Newton
direction satisfies ``|d|_inf <= step_tol * max(1, |Y|_inf)``, the scale-aware
rule of :func:`solve_perfect_foresight` (a rescaled system must return the same
path and the untouched initial guess is never returned). ``step_tol=None`` drops
the step test and accepts on the residual alone; together with
``jacobian_scale=1.0`` this is the acceptance rule of the IO ``native_solver._newton``.
When the residual meets ``tol * s`` but the step test cannot be met (a residual at
its noise floor, for instance behind an inner iterative solve), the solve raises
with ``converged_by="step_test_failure"``. Anything else that fails raises
:class:`StackedNewtonError` with the last state attached; ``converged=True`` is
never returned by a failed solve.

Preconditioners. With constant steady-state blocks (A, B, C) the block-Thomas
recursion

    D_1 = B,  U_1 = D_1^{-1} A,  D_t = B - C U_{t-1},  U_t = D_t^{-1} A,
    z_1 = D_1^{-1} r_1,  z_t = D_t^{-1} (r_t - C z_{t-1}),
    x_T = z_T,  x_t = z_t - U_t x_{t+1},

is the exact inverse of J0 = kron(I, B) + kron(S, A) + kron(S', C), S the T x T
upper shift, in O(T n^3) setup and O(T n^2) memory and apply. It is the
Laffargue-Boucekkine-Juillard relaxation written in matrix form. With
``time_block = H`` the same recursion is factored once for H dates and applied to
every block independently (block Jacobi in time; the links between blocks stay
in the outer Jacobian), exactly as the IO ``FiniteBlockCapitalPreconditioner``.
``method="splu"`` factors the same sparse matrix with SuperLU instead. The
block-Thomas recursion needs every leading pivot D_t nonsingular, which a
nonsingular stacked matrix does not guarantee (a rank-deficient B is enough to
break it). In floating point such a pivot is rarely exactly zero, so a pivot is
rejected when LAPACK's reciprocal condition estimate falls below ``n * eps``, and
the finished factorization is checked once on a fixed random right-hand side: a
normwise backward error ``|J0 x - r|_inf / (|J0|_inf |x|_inf + |r|_inf)`` above
1e-10 rejects it as well. The named preconditioners of the solver then fall back to
SuperLU, record the fallback in the preconditioner name, and raise
:class:`StackedNewtonError` only when the stacked matrix itself cannot be factored.
:class:`StructuredBlockTridiagonalPreconditioner` is the protocol a model-specific
elimination can implement (per-date diagonal solves and off-diagonal actions, or a
global structured solve). The IO capital and national-Schur eliminations have the
structure of its second form; they are not ported, and no puremacro module plugs
a preconditioner into this protocol in this release.

Horizon acceptance. ``compare_horizons`` reports max_{t <= periods} |y^T_t - y^2T_t|
against a tolerance (default 1e-5), the IO ``state_gap`` criterion.

Limitations quoted from the IO documentation
--------------------------------------------
NATIVE_NUMERICS.md: "Preconditioner speed and memory must be established by
executions; block structure alone does not establish native-scale performance."
and "These linear identities do not guarantee numerical accuracy or nonlinear
convergence." README.md: "No GTAP horizon is accepted yet." No native-scale MRIO
transition is solved by this module; the exact block-tridiagonal inverse costs
O(T n^2) memory, which is trivial for DSGE sizes (n up to a few hundred, T in
the hundreds) and infeasible for native MRIO systems with n of 7,000 to 9,500
unknowns per date, where a structured elimination is mandatory. The seven IO
preconditioner variants beyond the exact time-block inverse (capital, Schur,
periodic, swept, Galerkin, deflated, Jacobi-deflated) are not ported.
"""
from __future__ import annotations

import inspect
import math
import time
import warnings
from dataclasses import dataclass, field
from typing import Any, Callable, Mapping, Sequence

import numpy as np
import pandas as pd
import scipy.linalg
import scipy.optimize
import scipy.sparse as sp
import scipy.sparse.linalg as spla

from puremacro.reports import df_to_latex, df_to_markdown, df_to_typst

__all__ = [
    "StackedProblem",
    "BlockTridiagonalPreconditioner",
    "StructuredBlockTridiagonalPreconditioner",
    "preconditioned_lgmres",
    "solve_stacked_newton_krylov",
    "compare_horizons",
    "compare_stacked_horizons",
    "StackedNewtonResult",
    "HorizonComparison",
    "StackedNewtonError",
]

# Relative tolerance on the Newton STEP (same constant as perfect_foresight.py).
_STEP_TOL = 1e-10
# Armijo sufficient-decrease constant (IO native_solver._newton).
_ARMIJO_C = 1e-4
# Complex-step size (exact to machine precision, no cancellation).
_COMPLEX_STEP = 1e-20
# Errors a trial residual evaluation may raise inside the line search.
_TRIAL_ERRORS = (ValueError, FloatingPointError, ZeroDivisionError, OverflowError,
                 RuntimeError, np.linalg.LinAlgError)
_PRECONDITIONER_NAMES = ("steady_block_tridiagonal", "steady_splu", "time_block_jacobi",
                         "current_block_tridiagonal", "none")
# Errors a linear solve (Jacobian products, preconditioner applications) may raise
# mid-solve; they end the solve with the last state attached. TypeError is not
# included: it signals a programming error (for instance a complex-unsafe model).
_LINEAR_ERRORS = (ValueError, ArithmeticError, np.linalg.LinAlgError, RuntimeError)
# Normwise backward error above which a finished block-Thomas factorization is rejected.
_THOMAS_BACKWARD_TOL = 1e-10


class _SingularPivotError(ValueError):
    """A leading block-Thomas pivot is singular (the stacked matrix may still be nonsingular)."""


class StackedNewtonError(RuntimeError):
    """A stacked Newton-Krylov solve did not reach the requested tolerance.

    When raised by :func:`solve_stacked_newton_krylov`, ``result`` always carries
    the last state as a :class:`StackedNewtonResult` with ``converged=False`` and
    ``metadata["converged_by"]`` naming the cause (``"iteration_limit"``,
    ``"line_search_failure"``, ``"step_test_failure"``, ``"linear_solve_failure"``,
    ``"nonfinite_direction"`` or ``"preconditioner_failure"``). ``result`` is
    ``None`` only when the error comes from a building block called directly, for
    instance :meth:`StackedProblem.residual` on a nonfinite residual or a
    preconditioner's ``apply`` producing a nonfinite action, or from
    :func:`compare_horizons` on unaccepted inputs.
    """

    def __init__(self, message: str, result: "StackedNewtonResult | None" = None):
        super().__init__(message)
        self.result = result


def _norm_inf(x: np.ndarray) -> float:
    return float(np.max(np.abs(x), initial=0.0))


class _FrozenRecord(dict):
    """A ``dict`` whose contents cannot change after construction (one linear-history record).

    It stays a real ``dict`` (JSON-serializable, picklable, equal to a plain dict
    with the same items); ``dict(record)`` gives a mutable copy.
    """

    __slots__ = ()

    def _refuse(self, *args: Any, **kwargs: Any):
        raise TypeError("linear_history records are read-only; copy one with dict(record)")

    __setitem__ = __delitem__ = __ior__ = _refuse
    clear = pop = popitem = setdefault = update = _refuse

    def copy(self) -> dict:
        return dict(self)

    def __reduce__(self):
        return (type(self), (dict(self),))


def _readonly(a: np.ndarray, dtype=float) -> np.ndarray:
    out = np.array(a, dtype=dtype, copy=True)
    out.flags.writeable = False
    return out


def _check_positive_int(value: Any, name: str, minimum: int = 1) -> int:
    if isinstance(value, bool) or not isinstance(value, (int, np.integer)) or int(value) < minimum:
        raise ValueError(f"{name} must be an integer >= {minimum}, got {value!r}")
    return int(value)


def _check_positive_float(value: Any, name: str) -> float:
    try:
        out = float(value)
    except (TypeError, ValueError) as exc:
        raise ValueError(f"{name} must be a finite positive number, got {value!r}") from exc
    if not np.isfinite(out) or out <= 0.0:
        raise ValueError(f"{name} must be a finite positive number, got {value!r}")
    return out


# ---------------------------------------------------------------------------
# Stacked residual and Jacobian-vector products
# ---------------------------------------------------------------------------
class StackedProblem:
    """Stacked residual and matrix-free Jacobian action of a perfect-foresight system.

    Parameters
    ----------
    equations_fn : callable
        ``equations_fn(y_next, y_curr, y_prev, eps_t) -> array_like (n_vars,)``,
        the same signature as the callable path of :func:`solve_perfect_foresight`.
    y_init, y_end : array_like (n_vars,)
        Fixed predetermined state y_0 and fixed terminal state y_{T+1}.
    exogenous_path : array_like, optional
        Exogenous values by date, shape ``(T,)`` or ``(T, n_eps)``. When omitted,
        ``n_periods`` is required and every date receives ``0.0``.
    n_periods : int, optional
        Horizon T. When omitted, every row of ``exogenous_path`` is an interior
        date (T = its length). When given, a path of exactly ``T + 2`` rows (the
        :func:`solve_perfect_foresight` layout that includes both boundary rows) is
        trimmed to rows ``1..T`` and a longer one is truncated to its first T rows;
        pass ``n_periods`` whenever the path carries boundary rows, otherwise
        every shock lands one date later.
    jvp_fn : callable, optional
        Analytic per-date Jacobian-vector product
        ``jvp_fn(y_next, y_curr, y_prev, eps_t, v_next, v_curr, v_prev) -> (n_vars,)``.
    blocks_fn : callable, optional
        Analytic per-date blocks ``blocks_fn(y_next, y_curr, y_prev, eps_t) -> (A, B, C)``
        with A = df/dy_next, B = df/dy_curr, C = df/dy_prev.
    jvp_method : {"central", "complex"}
        Finite-difference rule for the stacked product when ``jvp_fn`` and
        ``stacked_jvp_factory`` are absent, and for ``date_blocks`` when
        ``blocks_fn`` is absent. ``"complex"`` requires a complex-safe ``equations_fn``.
    stacked_jvp_factory : callable, optional
        ``stacked_jvp_factory(Y_flat) -> Callable[[v], J v]`` overriding the whole
        stacked product (used by models with batched per-policy derivatives).
    variable_names : sequence of str, optional
    fd_step : float, default 1e-7
        Relative central-difference step.
    steady_exogenous : array_like, optional
        Exogenous values used by :meth:`steady_blocks`; default the last row of
        the exogenous path (the terminal policy is held, as in the IO engine).

    Notes
    -----
    Stacking conventions match :func:`solve_perfect_foresight`: date t of the
    flat vector occupies ``Y[t*n_vars:(t+1)*n_vars]``, y_prev at t = 0 is
    ``y_init`` and y_next at t = T-1 is ``y_end``, so the product uses
    v_prev = 0 at the first date and v_next = 0 at the last date.

    A model whose date-t residual also needs next date's exogenous values (the IO
    ``NativeEconomy.equations(..., policy_t, policy_next)`` pattern) packs both into
    each row, ``exogenous_path[t] = (eps_t, eps_{t+1})`` with the last row repeating
    ``eps_T``. Packing the values themselves, rather than a date index, keeps the
    held-exogenous rule of :func:`compare_horizons` meaningful.
    """

    def __init__(self, equations_fn: Callable, y_init: Any, y_end: Any,
                 exogenous_path: Any = None, *, n_periods: int | None = None,
                 jvp_fn: Callable | None = None, blocks_fn: Callable | None = None,
                 jvp_method: str = "central", stacked_jvp_factory: Callable | None = None,
                 variable_names: Sequence[str] | None = None, fd_step: float = 1e-7,
                 steady_exogenous: Any = None):
        if not callable(equations_fn):
            raise TypeError("equations_fn must be callable")
        for name, value in (("jvp_fn", jvp_fn), ("blocks_fn", blocks_fn),
                            ("stacked_jvp_factory", stacked_jvp_factory)):
            if value is not None and not callable(value):
                raise TypeError(f"{name} must be callable or None")
        y0 = np.asarray(y_init, dtype=float).ravel()
        yT = np.asarray(y_end, dtype=float).ravel()
        if y0.size == 0:
            raise ValueError("y_init must contain at least one variable")
        if yT.shape != y0.shape:
            raise ValueError(f"Dimension mismatch: len(y_end)={yT.size} != len(y_init)={y0.size}")
        if not (np.isfinite(y0).all() and np.isfinite(yT).all()):
            raise ValueError("y_init and y_end must be finite")
        if exogenous_path is None:
            if n_periods is None:
                raise ValueError("n_periods is required when exogenous_path is None")
            T = _check_positive_int(n_periods, "n_periods")
            exo = np.zeros(T, dtype=float)
        else:
            exo = np.asarray(exogenous_path, dtype=float)
            if exo.ndim == 0 or exo.shape[0] == 0:
                raise ValueError("exogenous_path must have at least one date")
            if n_periods is None:
                T = int(exo.shape[0])
            else:
                T = _check_positive_int(n_periods, "n_periods")
                if exo.shape[0] == T + 2:
                    exo = exo[1:T + 1]
                elif exo.shape[0] >= T:
                    exo = exo[:T]
                else:
                    raise ValueError(
                        f"exogenous_path length ({exo.shape[0]}) must be at least n_periods ({T})")
            if not np.isfinite(exo).all():
                raise ValueError("exogenous_path must be finite")
        if jvp_method not in ("central", "complex"):
            raise ValueError(f"unknown jvp_method {jvp_method!r}; expected 'central' or 'complex'")
        self._fd_step = _check_positive_float(fd_step, "fd_step")
        n = int(y0.size)
        if variable_names is None:
            names = tuple(f"y_{i}" for i in range(n))
        else:
            names = tuple(str(v) for v in variable_names)
            if len(names) != n:
                raise ValueError(f"variable_names has {len(names)} entries for {n} variables")
        if steady_exogenous is None:
            steady_eps = np.array(exo[-1], dtype=float, copy=True)
        else:
            steady_eps = np.asarray(steady_exogenous, dtype=float)
            if steady_eps.shape != np.shape(exo[-1]):
                raise ValueError("steady_exogenous must have the shape of one exogenous row")
        self._equations = equations_fn
        self._jvp_fn = jvp_fn
        self._blocks_fn = blocks_fn
        self._factory = stacked_jvp_factory
        self._fd_rule = jvp_method
        self._y_init = _readonly(y0)
        self._y_end = _readonly(yT)
        self._exo = _readonly(exo)
        self._steady_eps = _readonly(steady_eps)
        self._names = names
        self._n_vars = n
        self._T = T
        self._cache_x: np.ndarray | None = None
        self._cache_f: np.ndarray | None = None
        self._steady_blocks: tuple[np.ndarray, np.ndarray, np.ndarray] | None = None

    # -- geometry ---------------------------------------------------------
    @property
    def n_vars(self) -> int:
        """Number of endogenous variables per date."""
        return self._n_vars

    @property
    def n_periods(self) -> int:
        """Horizon T."""
        return self._T

    @property
    def n(self) -> int:
        """Number of stacked unknowns, T * n_vars."""
        return self._T * self._n_vars

    @property
    def y_init(self) -> np.ndarray:
        """Fixed predetermined state y_0 (read-only)."""
        return self._y_init

    @property
    def y_end(self) -> np.ndarray:
        """Fixed terminal state y_{T+1} (read-only)."""
        return self._y_end

    @property
    def exogenous_path(self) -> np.ndarray:
        """Exogenous values by interior date (read-only)."""
        return self._exo

    @property
    def steady_exogenous(self) -> np.ndarray:
        """Exogenous values used for the steady-state blocks (read-only)."""
        return self._steady_eps

    @property
    def variable_names(self) -> tuple[str, ...]:
        """Names of the endogenous variables, in stacking order."""
        return self._names

    @property
    def jvp_method(self) -> str:
        """Effective Jacobian-vector rule: 'stacked_factory', 'analytic', 'central' or 'complex'."""
        if self._factory is not None:
            return "stacked_factory"
        if self._jvp_fn is not None:
            return "analytic"
        return self._fd_rule

    def reshape(self, Y_flat: Any) -> np.ndarray:
        """Return ``Y`` as a ``(T, n_vars)`` array, validating its size."""
        Y = np.asarray(Y_flat)
        if Y.size != self.n:
            raise ValueError(f"path has {Y.size} entries, expected {self.n} = {self._T} x {self._n_vars}")
        return Y.reshape(self._T, self._n_vars)

    def linear_initial_path(self) -> np.ndarray:
        """Linear interpolation from y_init to y_end, the default initial guess."""
        w = (np.arange(1, self._T + 1, dtype=float) / (self._T + 1.0))[:, None]
        return (1.0 - w) * self._y_init[None, :] + w * self._y_end[None, :]

    # -- residual ---------------------------------------------------------
    def _date_args(self, Y: np.ndarray, t: int, y0: np.ndarray, yT: np.ndarray):
        yl = y0 if t == 0 else Y[t - 1]
        yp = yT if t == self._T - 1 else Y[t + 1]
        return yp, Y[t], yl, self._exo[t]

    def _evaluate(self, yp, yc, yl, eps, t: int, dtype=float) -> np.ndarray:
        raw = self._equations(yp, yc, yl, eps)
        if np.dtype(dtype).kind == "c" and not np.iscomplexobj(raw):
            raise TypeError(
                "jvp_method='complex' requires a complex-safe equations_fn: the residual returned "
                "for complex inputs is real, so the function discards the imaginary part (np.real, "
                "float() or a float dtype) and the complex-step derivative would be identically zero; "
                "use jvp_method='central' or supply jvp_fn")
        out = np.asarray(raw, dtype=dtype).ravel()
        if out.size != self._n_vars:
            raise ValueError(
                f"equations_fn returned {out.size} equations at t={t + 1}, expected {self._n_vars}")
        return out

    def _stack(self, Y: np.ndarray, dtype=float) -> np.ndarray:
        y0 = self._y_init.astype(dtype)
        yT = self._y_end.astype(dtype)
        R = np.empty((self._T, self._n_vars), dtype=dtype)
        for t in range(self._T):
            yp, yc, yl, eps = self._date_args(Y, t, y0, yT)
            R[t] = self._evaluate(yp, yc, yl, eps, t, dtype)
        return R.ravel()

    def residual(self, Y_flat: Any) -> np.ndarray:
        """Stacked residual F(Y), shape ``(T * n_vars,)``.

        Raises :class:`StackedNewtonError` when the residual is not finite, so a
        line search can reject the trial point (the IO ``_evaluate`` contract).
        """
        Y = self.reshape(np.asarray(Y_flat, dtype=float))
        flat = Y.ravel()
        if self._cache_x is not None and np.array_equal(flat, self._cache_x):
            return self._cache_f.copy()
        F = self._stack(Y)
        if not np.isfinite(F).all():
            raise StackedNewtonError("Nonfinite stacked residual")
        self._cache_x = flat.copy()
        self._cache_f = F.copy()
        return F

    def residual_matrix(self, Y_flat: Any) -> np.ndarray:
        """Stacked residual reshaped to ``(T, n_vars)``."""
        return self.residual(Y_flat).reshape(self._T, self._n_vars)

    # -- Jacobian-vector product ------------------------------------------
    def jvp(self, Y_flat: Any) -> Callable[[np.ndarray], np.ndarray]:
        """Return ``matvec(v) = J(Y) v`` for the stacked Jacobian at ``Y``."""
        Y = np.array(self.reshape(np.asarray(Y_flat, dtype=float)), copy=True)
        flat = Y.ravel()
        if self._factory is not None:
            inner = self._factory(flat.copy())
            if not callable(inner):
                raise TypeError("stacked_jvp_factory must return a callable")

            def matvec_factory(v):
                v = np.asarray(v, dtype=float).ravel()
                out = np.asarray(inner(v), dtype=float).ravel()
                if out.size != self.n:
                    raise ValueError("stacked_jvp_factory product has the wrong size")
                return out
            return matvec_factory
        if self._jvp_fn is not None:
            return self._analytic_matvec(Y)
        if self._fd_rule == "complex":
            return self._complex_matvec(Y)
        return self._central_matvec(Y)

    def _analytic_matvec(self, Y: np.ndarray) -> Callable[[np.ndarray], np.ndarray]:
        T, n = self._T, self._n_vars
        zero = np.zeros(n)
        y0, yT = self._y_init, self._y_end

        def matvec(v):
            V = self.reshape(np.asarray(v, dtype=float))
            out = np.empty((T, n))
            for t in range(T):
                yp, yc, yl, eps = self._date_args(Y, t, y0, yT)
                vl = zero if t == 0 else V[t - 1]
                vp = zero if t == T - 1 else V[t + 1]
                row = np.asarray(self._jvp_fn(yp, yc, yl, eps, vp, V[t], vl), dtype=float).ravel()
                if row.size != n:
                    raise ValueError(f"jvp_fn returned {row.size} entries at t={t + 1}, expected {n}")
                out[t] = row
            return out.ravel()
        return matvec

    def _central_matvec(self, Y: np.ndarray) -> Callable[[np.ndarray], np.ndarray]:
        scale = max(1.0, _norm_inf(Y))

        def matvec(v):
            V = self.reshape(np.asarray(v, dtype=float))
            vmax = _norm_inf(V)
            if vmax == 0.0:
                return np.zeros(self.n)
            h = self._fd_step * scale / vmax
            plus = self._stack(Y + h * V)
            minus = self._stack(Y - h * V)
            return (plus - minus) / (2.0 * h)
        return matvec

    def _complex_matvec(self, Y: np.ndarray) -> Callable[[np.ndarray], np.ndarray]:
        def matvec(v):
            V = self.reshape(np.asarray(v, dtype=float))
            if not np.any(V):
                return np.zeros(self.n)
            Z = Y.astype(complex) + 1j * _COMPLEX_STEP * V
            try:
                out = self._stack(Z, dtype=complex)
            except TypeError as exc:
                raise self._complex_type_error(exc) from exc
            return out.imag / _COMPLEX_STEP
        return matvec

    @staticmethod
    def _complex_type_error(exc: TypeError) -> TypeError:
        """Normalise TypeErrors raised on complex inputs into one 'complex-safe' message."""
        if "complex-safe" in str(exc):
            return exc
        return TypeError("jvp_method='complex' requires a complex-safe equations_fn (equations_fn "
                         f"raised TypeError on complex inputs: {exc}); use jvp_method='central' or "
                         "supply jvp_fn")

    # -- per-date blocks --------------------------------------------------
    def blocks_at(self, yp: Any, yc: Any, yl: Any, eps: Any) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        """Dense per-date blocks ``(A, B, C) = (df/dy_next, df/dy_curr, df/dy_prev)``.

        Uses ``blocks_fn`` when supplied; otherwise the columns of ``jvp_fn`` on
        unit directions when an analytic product is available; otherwise
        column-wise central differences with step ``fd_step * max(1, |y_j|)`` (the
        rule of :func:`solve_perfect_foresight`) or the complex step when
        ``jvp_method="complex"``.
        """
        n = self._n_vars
        yp = np.asarray(yp, dtype=float).ravel()
        yc = np.asarray(yc, dtype=float).ravel()
        yl = np.asarray(yl, dtype=float).ravel()
        if self._blocks_fn is None and self._jvp_fn is not None:
            zero = np.zeros(n)
            eye = np.eye(n)
            A = np.empty((n, n))
            B = np.empty((n, n))
            C = np.empty((n, n))
            for j in range(n):
                e = eye[j]
                A[:, j] = np.asarray(self._jvp_fn(yp, yc, yl, eps, e, zero, zero), dtype=float).ravel()
                B[:, j] = np.asarray(self._jvp_fn(yp, yc, yl, eps, zero, e, zero), dtype=float).ravel()
                C[:, j] = np.asarray(self._jvp_fn(yp, yc, yl, eps, zero, zero, e), dtype=float).ravel()
            return A, B, C
        if self._blocks_fn is not None:
            blocks = self._blocks_fn(yp, yc, yl, eps)
            if len(blocks) != 3:
                raise ValueError("blocks_fn must return (A, B, C)")
            out = []
            for name, block in zip("ABC", blocks):
                M = np.asarray(block, dtype=float)
                if M.shape != (n, n):
                    raise ValueError(f"blocks_fn block {name} has shape {M.shape}, expected {(n, n)}")
                out.append(M)
            return out[0], out[1], out[2]
        args = [yp, yc, yl]
        result = []
        for which in range(3):
            M = np.empty((n, n))
            base = args[which]
            for j in range(n):
                if self._fd_rule == "complex":
                    cargs = [a.astype(complex) for a in args]
                    pert = cargs[which].copy()
                    pert[j] += 1j * _COMPLEX_STEP
                    cargs[which] = pert
                    try:
                        out = self._evaluate(cargs[0], cargs[1], cargs[2], eps, 0, complex)
                    except TypeError as exc:
                        raise self._complex_type_error(exc) from exc
                    M[:, j] = out.imag / _COMPLEX_STEP
                else:
                    h = self._fd_step * max(1.0, abs(base[j]))
                    up = [a for a in args]
                    dn = [a for a in args]
                    pu = base.copy()
                    pu[j] += h
                    pd_ = base.copy()
                    pd_[j] -= h
                    up[which] = pu
                    dn[which] = pd_
                    fp = self._evaluate(up[0], up[1], up[2], eps, 0)
                    fm = self._evaluate(dn[0], dn[1], dn[2], eps, 0)
                    M[:, j] = (fp - fm) / (2.0 * h)
            result.append(M)
        return result[0], result[1], result[2]

    def date_blocks(self, t: int, Y_flat: Any) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        """Blocks ``(A_t, B_t, C_t)`` at date ``t`` (0-based) of the path ``Y``.

        All three derivatives are returned; the stacked Jacobian ignores ``A_{T-1}``
        and ``C_0`` because the boundary states are fixed.
        """
        Y = self.reshape(np.asarray(Y_flat, dtype=float))
        if not 0 <= int(t) < self._T:
            raise ValueError(f"date index {t} outside 0..{self._T - 1}")
        yp, yc, yl, eps = self._date_args(Y, int(t), self._y_init, self._y_end)
        return self.blocks_at(yp, yc, yl, eps)

    def steady_blocks(self) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        """Constant blocks at ``(y_end, y_end, y_end, steady_exogenous)`` (cached)."""
        if self._steady_blocks is None:
            A, B, C = self.blocks_at(self._y_end, self._y_end, self._y_end, self._steady_eps)
            if not (np.isfinite(A).all() and np.isfinite(B).all() and np.isfinite(C).all()):
                raise ValueError("steady-state blocks are not finite; check y_end and steady_exogenous")
            self._steady_blocks = (_readonly(A), _readonly(B), _readonly(C))
        return self._steady_blocks

    def stacked_jacobian(self, Y_flat: Any) -> sp.csc_matrix:
        """Sparse block-tridiagonal Jacobian assembled from :meth:`date_blocks`.

        This is an oracle for tests and the source of the
        ``"current_block_tridiagonal"`` preconditioner; the solver itself never
        needs it.
        """
        T, n = self._T, self._n_vars
        Y = self.reshape(np.asarray(Y_flat, dtype=float))
        A = np.empty((T, n, n))
        B = np.empty((T, n, n))
        C = np.empty((T, n, n))
        for t in range(T):
            A[t], B[t], C[t] = self.date_blocks(t, Y)
        return _assemble_sparse(A, B, C, T, n, None)

    def current_blocks(self, Y_flat: Any) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        """Per-date blocks along the whole path as ``(T, n, n)`` arrays."""
        T, n = self._T, self._n_vars
        Y = self.reshape(np.asarray(Y_flat, dtype=float))
        A = np.empty((T, n, n))
        B = np.empty((T, n, n))
        C = np.empty((T, n, n))
        for t in range(T):
            A[t], B[t], C[t] = self.date_blocks(t, Y)
        return A, B, C


# ---------------------------------------------------------------------------
# Block-tridiagonal-in-time preconditioners
# ---------------------------------------------------------------------------
def _expand_blocks(A: Any, B: Any, C: Any, T: int) -> tuple[np.ndarray, np.ndarray, np.ndarray, bool]:
    """Return read-only copies of ``(A, B, C)`` as ``(T, n, n)`` arrays and whether they are constant.

    Copies decouple the operator from the caller's arrays: mutating the inputs after
    construction changes neither :meth:`BlockTridiagonalPreconditioner.apply` nor
    :meth:`BlockTridiagonalPreconditioner.to_sparse`.
    """
    B = np.asarray(B, dtype=float)
    if B.ndim == 2:
        n = B.shape[0]
        if B.shape != (n, n):
            raise ValueError(f"B must be square, got shape {B.shape}")
        A = np.asarray(A, dtype=float)
        C = np.asarray(C, dtype=float)
        if A.shape != (n, n) or C.shape != (n, n):
            raise ValueError("A, B and C must share the same (n, n) shape")
        if not (np.isfinite(A).all() and np.isfinite(B).all() and np.isfinite(C).all()):
            raise ValueError("preconditioner blocks must be finite")
        A, B, C = _readonly(A), _readonly(B), _readonly(C)
        return (np.broadcast_to(A, (T, n, n)), np.broadcast_to(B, (T, n, n)),
                np.broadcast_to(C, (T, n, n)), True)
    if B.ndim == 3:
        if B.shape[0] != T or B.shape[1] != B.shape[2]:
            raise ValueError(f"per-date B must have shape (T, n, n) with T={T}, got {B.shape}")
        A = np.asarray(A, dtype=float)
        C = np.asarray(C, dtype=float)
        if A.shape != B.shape or C.shape != B.shape:
            raise ValueError("per-date A, B and C must share the same (T, n, n) shape")
        if not (np.isfinite(A).all() and np.isfinite(B).all() and np.isfinite(C).all()):
            raise ValueError("preconditioner blocks must be finite")
        return _readonly(A), _readonly(B), _readonly(C), False
    raise ValueError("blocks must be (n, n) constant matrices or (T, n, n) per-date arrays")


def _assemble_sparse(A: np.ndarray, B: np.ndarray, C: np.ndarray, T: int, n: int,
                     time_block: int | None) -> sp.csc_matrix:
    """kron-style assembly of the stacked matrix, masking links across time blocks."""
    rows = []
    cols = []
    vals = []
    ii, jj = np.meshgrid(np.arange(n), np.arange(n), indexing="ij")
    ii = ii.ravel()
    jj = jj.ravel()
    for t in range(T):
        rows.append(t * n + ii)
        cols.append(t * n + jj)
        vals.append(B[t].ravel())
        same_next = t + 1 < T and (time_block is None or (t // time_block) == ((t + 1) // time_block))
        if same_next:
            rows.append(t * n + ii)
            cols.append((t + 1) * n + jj)
            vals.append(A[t].ravel())
            rows.append((t + 1) * n + ii)
            cols.append(t * n + jj)
            vals.append(C[t + 1].ravel())
    return sp.csc_matrix((np.concatenate(vals), (np.concatenate(rows), np.concatenate(cols))),
                         shape=(T * n, T * n))


class _ThomasFactor:
    """Block-LU factors of one block-tridiagonal stretch of ``L`` dates."""

    def __init__(self, A: np.ndarray, B: np.ndarray, C: np.ndarray):
        L = B.shape[0]
        self.length = L
        self.C = C
        self.lu = []
        self.U = []
        D = np.array(B[0], copy=True)
        self.lu.append(self._factor(D, 0))
        for t in range(1, L):
            U_prev = scipy.linalg.lu_solve(self.lu[t - 1], A[t - 1], check_finite=False)
            self.U.append(U_prev)
            D = B[t] - C[t] @ U_prev
            self.lu.append(self._factor(D, t))

    @staticmethod
    def _factor(D: np.ndarray, t: int):
        """LU of the pivot ``D``; reject it when singular or numerically singular.

        A pivot of a rank-deficient block is rarely exactly zero in floating point
        (rounding leaves entries of order 1e-17), so the test is LAPACK's reciprocal
        1-norm condition estimate ``rcond < n * eps`` in addition to an exact zero.
        SciPy's exact-zero ``LinAlgWarning`` is converted into this exception.
        """
        n = D.shape[0]
        anorm = float(np.linalg.norm(D, 1))
        with warnings.catch_warnings():
            warnings.simplefilter("ignore", scipy.linalg.LinAlgWarning)
            lu, piv = scipy.linalg.lu_factor(D, check_finite=False)
        rcond = 0.0
        if np.isfinite(lu).all() and not np.any(np.diag(lu) == 0.0) and anorm > 0.0:
            gecon, = scipy.linalg.lapack.get_lapack_funcs(("gecon",), (lu,))
            rcond, info = gecon(lu, anorm, norm="1")
            rcond = float(rcond) if info == 0 and np.isfinite(rcond) else 0.0
        if rcond < n * np.finfo(float).eps:
            raise _SingularPivotError(
                f"block-Thomas pivot D_{t + 1} is singular (reciprocal condition estimate {rcond:.2e} "
                f"below n * eps = {n * np.finfo(float).eps:.2e}); the block-Thomas elimination needs every "
                "leading pivot nonsingular, which a nonsingular stacked matrix does not guarantee; "
                "use method='splu'")
        return lu, piv

    @property
    def memory_bytes(self) -> int:
        total = sum(lu.nbytes + piv.nbytes for lu, piv in self.lu)
        total += sum(U.nbytes for U in self.U)
        return int(total)

    def solve(self, R: np.ndarray) -> np.ndarray:
        """Solve for right-hand sides ``R`` of shape ``(L, n)`` or ``(L, n, batch)``."""
        L = self.length
        z = np.empty_like(R, dtype=float)
        z[0] = scipy.linalg.lu_solve(self.lu[0], R[0], check_finite=False)
        for t in range(1, L):
            rhs = R[t] - np.tensordot(self.C[t], z[t - 1], axes=([1], [0]))
            z[t] = scipy.linalg.lu_solve(self.lu[t], rhs, check_finite=False)
        for t in range(L - 2, -1, -1):
            z[t] -= np.tensordot(self.U[t], z[t + 1], axes=([1], [0]))
        return z


class BlockTridiagonalPreconditioner:
    """Exact inverse of a block-tridiagonal-in-time stacked matrix.

    ``J0 = kron(I_T, B) + kron(S, A) + kron(S', C)`` for constant blocks (``S`` the
    upper shift), or the matrix with per-date blocks when ``A, B, C`` have shape
    ``(T, n, n)``. ``method="thomas"`` stores the block-Thomas factors (T dense
    LU factors of size n plus T-1 coupling matrices, O(T n^2) memory);
    ``method="splu"`` uses SuperLU on the sparse assembly. ``time_block=H`` drops
    the links between consecutive H-date blocks (block Jacobi in time): every
    full block reuses one factorization and a shorter remainder block gets its
    own, as in the IO ``FiniteBlockCapitalPreconditioner``. With ``H = T`` (the
    default) the operator is the exact inverse of ``J0``. The blocks are copied
    (read-only), so later changes to the caller's arrays do not reach the operator.

    ``method="thomas"`` raises ``ValueError`` when a leading pivot ``D_t`` is
    singular or numerically singular (LAPACK reciprocal condition estimate below
    ``n * eps``) even if the stacked matrix is not, and when the finished
    factorization fails a one-vector accuracy check: for a fixed random ``r`` the
    normwise backward error ``|J0 x - r|_inf / (|J0|_inf |x|_inf + |r|_inf)`` of
    ``x = apply(r)`` must not exceed 1e-10 (a backward-stable solve gives about
    1e-16 whatever the conditioning of ``J0``). ``method="splu"`` handles these
    cases; the named preconditioners of :func:`solve_stacked_newton_krylov` fall
    back to it automatically. ``method="splu"`` raises ``ValueError`` only when
    SuperLU cannot factor the stacked matrix.

    Parameters
    ----------
    A, B, C : array_like
        ``df/dy_next``, ``df/dy_curr``, ``df/dy_prev``: constant ``(n, n)`` blocks
        (typically :meth:`StackedProblem.steady_blocks`) or ``(T, n, n)`` per-date arrays.
    n_periods : int
        Horizon T.
    method : {"thomas", "splu"}
    time_block : int, optional
        Block length H in dates, ``1 <= H <= T``.
    """

    def __init__(self, A: Any, B: Any, C: Any, n_periods: int, *, method: str = "thomas",
                 time_block: int | None = None):
        T = _check_positive_int(n_periods, "n_periods")
        if method not in ("thomas", "splu"):
            raise ValueError(f"unknown method {method!r}; expected 'thomas' or 'splu'")
        A3, B3, C3, constant = _expand_blocks(A, B, C, T)
        n = B3.shape[1]
        if time_block is not None:
            H = _check_positive_int(time_block, "time_block")
            if H > T:
                raise ValueError(f"time_block ({H}) cannot exceed n_periods ({T})")
        else:
            H = T
        self._T, self._n, self._H = T, n, H
        self._method = method
        self._constant = constant
        self._A, self._B, self._C = A3, B3, C3
        self.shape = (T * n, T * n)
        self._factors: list[tuple[int, int, Any]] = []   # (start, length, factor)
        self._main_batched: tuple[Any, int] | None = None
        self._splu = None
        self._lu_bytes = 0
        if method == "thomas":
            self._build_thomas()
        else:
            try:
                self._splu = spla.splu(self.to_sparse())
            except RuntimeError as exc:
                raise ValueError(f"SuperLU factorization of the stacked matrix is singular: {exc}") from exc
            self._lu_bytes = int((self._splu.L.nnz + self._splu.U.nnz) * 12)
            probe = self._splu.solve(np.ones(T * n))
            if not np.isfinite(probe).all():
                raise ValueError("SuperLU factorization of the stacked matrix is singular (nonfinite solve)")

    def _build_thomas(self) -> None:
        T, H = self._T, self._H
        count, rem = divmod(T, H)
        if self._constant:
            main = _ThomasFactor(self._A[:H], self._B[:H], self._C[:H])
            self._main_batched = (main, count)
            self._factors = [(k * H, H, main) for k in range(count)]
            if rem:
                tail = _ThomasFactor(self._A[:rem], self._B[:rem], self._C[:rem])
                self._factors.append((count * H, rem, tail))
        else:
            self._main_batched = None
            start = 0
            while start < T:
                length = min(H, T - start)
                block = _ThomasFactor(self._A[start:start + length], self._B[start:start + length],
                                      self._C[start:start + length])
                self._factors.append((start, length, block))
                start += length
        seen = set()
        total = 0
        for _, _, factor in self._factors:
            if id(factor) not in seen:
                seen.add(id(factor))
                total += factor.memory_bytes
        self._lu_bytes = int(total)
        self._check_thomas_accuracy()

    def _check_thomas_accuracy(self) -> None:
        """Reject a factorization whose normwise backward error on one fixed vector exceeds 1e-10.

        The pivot test catches rounding-level pivots; this catches the remaining
        ill-conditioned pivots that make the unpivoted block elimination inaccurate
        although every factor is finite. Cost: one application and one block product.
        """
        T, n = self._T, self._n
        r = np.random.default_rng(0).standard_normal((T, n))
        with np.errstate(all="ignore"):
            x = self._thomas_solve(r)
            residual = self._stacked_matvec(x) - r
        if self._constant:
            rows = np.abs(self._A[0]).sum(axis=1) + np.abs(self._B[0]).sum(axis=1) + np.abs(self._C[0]).sum(axis=1)
        else:
            rows = np.abs(self._A).sum(axis=2) + np.abs(self._B).sum(axis=2) + np.abs(self._C).sum(axis=2)
        denom = float(np.max(rows)) * _norm_inf(x) + _norm_inf(r)
        backward = _norm_inf(residual) / denom if np.isfinite(x).all() and np.isfinite(denom) else float("inf")
        if not np.isfinite(backward) or backward > _THOMAS_BACKWARD_TOL:
            raise _SingularPivotError(
                f"block-Thomas pivots are too ill-conditioned: normwise backward error {backward:.2e} on a "
                f"random right-hand side exceeds {_THOMAS_BACKWARD_TOL:.0e}; the unpivoted block "
                "elimination is inaccurate for this matrix; use method='splu'")

    def _stacked_matvec(self, X: np.ndarray) -> np.ndarray:
        """``J0 X`` for a ``(T, n)`` array, with the links across time blocks dropped."""
        T, H = self._T, self._H
        if self._constant:
            A0, B0, C0 = self._A[0], self._B[0], self._C[0]
            out = X @ B0.T
            upper = X[1:] @ A0.T
            lower = X[:-1] @ C0.T
        else:
            out = np.einsum("tij,tj->ti", self._B, X)
            upper = np.einsum("tij,tj->ti", self._A[:-1], X[1:])
            lower = np.einsum("tij,tj->ti", self._C[1:], X[:-1])
        if T > 1:
            link = ((np.arange(T - 1) // H) == (np.arange(1, T) // H))[:, None]
            out[:-1] += np.where(link, upper, 0.0)
            out[1:] += np.where(link, lower, 0.0)
        return out

    # -- properties -------------------------------------------------------
    @property
    def n_periods(self) -> int:
        """Horizon T."""
        return self._T

    @property
    def n_vars(self) -> int:
        """Variables per date (block size n)."""
        return self._n

    @property
    def time_block(self) -> int:
        """Block length in dates (``n_periods`` for the exact inverse)."""
        return self._H

    @property
    def method(self) -> str:
        """Factorization method, ``"thomas"`` or ``"splu"``."""
        return self._method

    @property
    def memory_bytes(self) -> int:
        """Bytes held by the factorization (LU factors and coupling matrices)."""
        return self._lu_bytes

    @property
    def name(self) -> str:
        """Label recorded by the solver, e.g. ``block_tridiagonal[thomas, exact]``."""
        kind = "exact" if self._H == self._T else f"time_block={self._H}"
        return f"block_tridiagonal[{self._method}, {kind}]"

    def to_sparse(self) -> sp.csc_matrix:
        """The (block-masked) stacked matrix this operator inverts."""
        return _assemble_sparse(self._A, self._B, self._C, self._T, self._n,
                                None if self._H == self._T else self._H)

    # -- application ------------------------------------------------------
    def _thomas_solve(self, R: np.ndarray) -> np.ndarray:
        """Block-Thomas solve for a ``(T, n)`` right-hand side (no finiteness check)."""
        out = np.empty_like(R, dtype=float)
        if self._main_batched is not None:
            main, count = self._main_batched
            H = self._H
            if count:
                block = R[:count * H].reshape(count, H, self._n).transpose(1, 2, 0)
                out[:count * H] = main.solve(block).transpose(2, 0, 1).reshape(count * H, self._n)
            if count * H < self._T:
                start, length, tail = self._factors[-1]
                out[start:] = tail.solve(R[start:])
        else:
            for start, length, factor in self._factors:
                out[start:start + length] = factor.solve(R[start:start + length])
        return out

    def apply(self, r: Any) -> np.ndarray:
        """Solve ``J0 x = r`` for a flat or ``(T, n)`` right-hand side; returns flat ``x``."""
        R = np.asarray(r, dtype=float)
        if R.size != self.shape[0]:
            raise ValueError(f"right-hand side has {R.size} entries, expected {self.shape[0]}")
        if self._method == "splu":
            out = self._splu.solve(R.ravel())
        else:
            out = self._thomas_solve(R.reshape(self._T, self._n)).ravel()
        if not np.isfinite(out).all():
            raise StackedNewtonError("Nonfinite block-tridiagonal preconditioner action")
        return out

    def as_linear_operator(self) -> spla.LinearOperator:
        """SciPy ``LinearOperator`` wrapping :meth:`apply`."""
        return spla.LinearOperator(self.shape, matvec=self.apply, dtype=float)


class StructuredBlockTridiagonalPreconditioner:
    """Protocol for model-specific approximate inverses of a stacked Jacobian.

    Two ways to use it:

    1. Supply per-date callables and a sweep. ``diag_solve(t, r_t)`` returns
       ``D_t^{-1} r_t``, ``lower_apply(t, x)`` returns ``C_t x`` (the action of the
       block below the diagonal on the previous date) and ``upper_apply(t, x)``
       returns ``A_t x`` (the block above, acting on the next date). The sweeps are
       ``"jacobi"`` (``x_t = D_t^{-1} r_t``), ``"forward"`` (block Gauss-Seidel,
       ``x = (D + L)^{-1} r``), ``"backward"`` (``x = (D + U)^{-1} r``) and
       ``"symmetric"`` (``x = (D + U)^{-1} D (D + L)^{-1} r``, the classical block SGS
       formula that the IO swept block driver uses with exact finite time blocks).
    2. Subclass and override :meth:`apply` with a global structured elimination.
       The IO ``CapitalPreconditioner`` (N scalar time-tridiagonal solves for the
       capital stocks followed by static national labor and budget LU solves) and
       ``SchurCapitalPreconditioner`` (national Schur complement kept block
       tridiagonal in time) have the structure of this second form, so a
       model-specific transition solver could supply such an elimination through
       it. Neither is ported here, and in this release no puremacro module uses the
       protocol: ``puremacro.trade.dynamic`` ships its own capital preconditioner
       and Newton loop.

    Either way, :meth:`as_linear_operator` returns the object SciPy's Krylov
    solvers accept, and :meth:`apply` raises :class:`StackedNewtonError` on a
    nonfinite action, as the IO Schur preconditioner does (inside
    :func:`solve_stacked_newton_krylov` that error ends the solve with the last
    state attached).
    """

    def __init__(self, n_periods: int, n_vars: int, *, diag_solve: Callable | None = None,
                 lower_apply: Callable | None = None, upper_apply: Callable | None = None,
                 sweep: str = "symmetric", name: str | None = None):
        T = _check_positive_int(n_periods, "n_periods")
        n = _check_positive_int(n_vars, "n_vars")
        if sweep not in ("jacobi", "forward", "backward", "symmetric"):
            raise ValueError(f"unknown sweep {sweep!r}; expected jacobi, forward, backward or symmetric")
        overridden = type(self).apply is not StructuredBlockTridiagonalPreconditioner.apply
        if not overridden:
            if diag_solve is None or not callable(diag_solve):
                raise ValueError("supply a callable diag_solve or override apply() in a subclass")
            if sweep in ("forward", "symmetric") and not callable(lower_apply):
                raise ValueError(f"sweep={sweep!r} requires a callable lower_apply")
            if sweep in ("backward", "symmetric") and not callable(upper_apply):
                raise ValueError(f"sweep={sweep!r} requires a callable upper_apply")
        self._T, self._n = T, n
        self.shape = (T * n, T * n)
        self._diag_solve = diag_solve
        self._lower = lower_apply
        self._upper = upper_apply
        self._sweep = sweep
        self._name = name

    @property
    def n_periods(self) -> int:
        """Horizon T."""
        return self._T

    @property
    def n_vars(self) -> int:
        """Variables per date."""
        return self._n

    @property
    def sweep(self) -> str:
        """Sweep of the per-date form: jacobi, forward, backward or symmetric."""
        return self._sweep

    @property
    def name(self) -> str:
        """Label recorded by the solver (the ``name`` argument or ``structured[<sweep>]``)."""
        return self._name or f"structured[{self._sweep}]"

    def _reshape(self, r: Any) -> np.ndarray:
        R = np.asarray(r, dtype=float)
        if R.size != self.shape[0]:
            raise ValueError(f"right-hand side has {R.size} entries, expected {self.shape[0]}")
        return R.reshape(self._T, self._n)

    def _solve(self, t: int, rhs: np.ndarray) -> np.ndarray:
        out = np.asarray(self._diag_solve(t, rhs), dtype=float).ravel()
        if out.size != self._n:
            raise ValueError(f"diag_solve returned {out.size} entries at date {t}, expected {self._n}")
        return out

    def _act(self, fn: Callable, t: int, x: np.ndarray) -> np.ndarray:
        out = np.asarray(fn(t, x), dtype=float).ravel()
        if out.size != self._n:
            raise ValueError(f"off-diagonal action returned {out.size} entries at date {t}, expected {self._n}")
        return out

    def apply(self, r: Any) -> np.ndarray:
        """Apply the sweep to a flat or ``(T, n)`` right-hand side; returns flat ``x``."""
        R = self._reshape(r)
        T = self._T
        x = np.empty_like(R)
        if self._sweep == "jacobi":
            for t in range(T):
                x[t] = self._solve(t, R[t])
        elif self._sweep == "forward":
            for t in range(T):
                rhs = R[t] if t == 0 else R[t] - self._act(self._lower, t, x[t - 1])
                x[t] = self._solve(t, rhs)
        elif self._sweep == "backward":
            for t in range(T - 1, -1, -1):
                rhs = R[t] if t == T - 1 else R[t] - self._act(self._upper, t, x[t + 1])
                x[t] = self._solve(t, rhs)
        else:
            z = np.empty_like(R)
            for t in range(T):
                rhs = R[t] if t == 0 else R[t] - self._act(self._lower, t, z[t - 1])
                z[t] = self._solve(t, rhs)
            x[T - 1] = z[T - 1]
            for t in range(T - 2, -1, -1):
                x[t] = z[t] - self._solve(t, self._act(self._upper, t, x[t + 1]))
        out = x.ravel()
        if not np.isfinite(out).all():
            raise StackedNewtonError("Nonfinite structured preconditioner action")
        return out

    def as_linear_operator(self) -> spla.LinearOperator:
        """SciPy ``LinearOperator`` wrapping :meth:`apply`."""
        return spla.LinearOperator(self.shape, matvec=self.apply, dtype=float)


def _as_operator(obj: Any, shape: tuple[int, int] | None = None) -> spla.LinearOperator | None:
    """Coerce ``None``, arrays, sparse matrices, LinearOperators or preconditioner objects."""
    if obj is None:
        return None
    if hasattr(obj, "as_linear_operator"):
        obj = obj.as_linear_operator()
    op = spla.aslinearoperator(obj)
    if shape is not None and tuple(op.shape) != tuple(shape):
        raise ValueError(f"operator has shape {op.shape}, expected {shape}")
    return op


# ---------------------------------------------------------------------------
# Krylov adapter with true-residual reporting
# ---------------------------------------------------------------------------
def _relative_tolerance_keyword(solver: Callable, rtol: float) -> dict[str, float]:
    """``{"rtol": rtol}`` for SciPy >= 1.12, ``{"tol": rtol}`` for the older signature.

    ``pyproject.toml`` admits SciPy 1.10, whose ``lgmres``/``gmres`` take ``tol``
    (``rtol`` was added in 1.12 and ``tol`` later removed); the signature decides.
    """
    try:
        params = inspect.signature(solver).parameters
    except (TypeError, ValueError):
        return {"rtol": rtol}
    if "rtol" in params or "tol" not in params:
        return {"rtol": rtol}
    return {"tol": rtol}


def preconditioned_lgmres(operator: Any, rhs: Any, M: Any = None, *, side: str = "right",
                          method: str = "lgmres", rtol: float = 1e-8, atol: float = 0.0,
                          inner_m: int = 40, outer_k: int = 6, maxiter: int = 35,
                          callback: Callable | None = None, x0: Any = None
                          ) -> tuple[np.ndarray, int, float]:
    """Solve ``operator @ d = rhs`` by LGMRES (or restarted GMRES) with an explicit side.

    Returns ``(direction, info, true_residual)`` where ``info`` is SciPy's status
    (0 converged, > 0 iteration cap reached, < 0 breakdown) and ``true_residual``
    is ``|operator @ direction - rhs|_2`` recomputed in original coordinates for
    the direction actually returned. SciPy's callbacks fire before each outer
    residual check, so when the cap is exhausted the last callback never saw the
    returned direction (the IO finding behind ``native_krylov_probe.make_lgmres``
    and its recorded drivers); the recomputed value is therefore the only
    trustworthy record of the linear error.

    ``side="right"`` solves ``(A M) y = rhs`` and returns ``d = M y`` (the Krylov
    residual is then the true residual; the callback receives ``y``);
    ``side="left"`` passes ``M`` to SciPy, which builds the Krylov space on ``M A``
    but still checks the true residual at outer iterations. Right preconditioning
    does not accept ``x0``. ``method="gmres"`` uses ``restart=inner_m`` cycles; it
    is an explicit option, not an automatic fallback when LGMRES stalls (the caller
    sees ``info`` and the true residual and decides). The relative tolerance is
    passed as ``rtol`` (SciPy >= 1.12) or ``tol`` (older SciPy), whichever the
    installed signature accepts; the callback timing described above was verified
    on SciPy 1.18.
    """
    if side not in ("left", "right"):
        raise ValueError("side must be 'left' or 'right'")
    if method not in ("lgmres", "gmres"):
        raise ValueError("method must be 'lgmres' or 'gmres'")
    inner_m = _check_positive_int(inner_m, "inner_m")
    outer_k = _check_positive_int(outer_k, "outer_k", minimum=0)
    maxiter = _check_positive_int(maxiter, "maxiter")
    rtol = float(rtol)
    atol = float(atol)
    if not np.isfinite(rtol) or rtol < 0.0 or not np.isfinite(atol) or atol < 0.0:
        raise ValueError("rtol and atol must be finite and nonnegative")
    A = _as_operator(operator)
    b = np.asarray(rhs, dtype=float).ravel()
    if b.size != A.shape[0]:
        raise ValueError(f"rhs has {b.size} entries, operator expects {A.shape[0]}")
    Mop = _as_operator(M, A.shape)
    if side == "right" and Mop is not None and x0 is not None:
        raise ValueError("right preconditioning expects the solver's zero Krylov guess (x0=None)")

    def run(op, precond):
        if method == "lgmres":
            solver = spla.lgmres
            return solver(op, b, x0=x0, M=precond, atol=atol, maxiter=maxiter, inner_m=inner_m,
                          outer_k=outer_k, callback=callback, **_relative_tolerance_keyword(solver, rtol))
        solver = spla.gmres
        return solver(op, b, x0=x0, M=precond, atol=atol, restart=inner_m, maxiter=maxiter,
                      callback=callback, callback_type="x" if callback is not None else None,
                      **_relative_tolerance_keyword(solver, rtol))

    if side == "left" or Mop is None:
        direction, info = run(A, Mop)
    else:
        right = spla.LinearOperator(A.shape, matvec=lambda v: A @ (Mop @ v), dtype=float)
        transformed, info = run(right, None)
        direction = Mop @ np.asarray(transformed, dtype=float)
    direction = np.asarray(direction, dtype=float).ravel()
    if np.isfinite(direction).all():
        true_residual = float(np.linalg.norm(A @ direction - b))
    else:
        true_residual = float("inf")
    return direction, int(info), true_residual


# ---------------------------------------------------------------------------
# Result containers
# ---------------------------------------------------------------------------
@dataclass(frozen=True)
class StackedNewtonResult:
    """Outcome of :func:`solve_stacked_newton_krylov`.

    Attributes
    ----------
    path : pd.DataFrame, shape (T, n_vars)
        Solved trajectory for t = 1..T (index name ``t``). The solver builds it on a
        read-only array, so in-place edits (``path.iloc[0, 0] = ...``) raise; treat it
        as read-only and use :meth:`to_dataframe` or :meth:`to_numpy` for a copy.
    converged : bool
        True only when the scale-aware residual and step tests both passed.
    iterations : int
        Accepted Newton steps (a MINPACK pre-solve, if any, counts as none).
    residual_norm : float
        Max-norm of the stacked residual at the returned path.
    initial_residual_norm : float
        Max-norm of the residual at the initial guess.
    terminal_error : float
        ``max |y_T - y_end|``, as in :class:`PerfectForesightResult`.
    linear_history : tuple of read-only dicts
        One record per linear solve (``dict(record)`` gives a mutable copy): ``newton_iteration``, ``residual_before``,
        ``krylov_outer_iterations`` (LGMRES outer iterations entered, IO convention),
        ``krylov_status``, ``true_linear_residual`` (2-norm of ``J d + F`` for the
        returned direction), ``requested_tolerance`` (``eta |F|_2``),
        ``rhs_norm`` (``|F|_2``), ``direction_norm``, ``step``, ``backtracks``,
        ``residual`` (after the step), ``seconds_linear`` and ``seconds`` (elapsed at the
        accepted step); a final ``convergence_check`` record has ``step = 0``; a record
        that ends the solve carries ``line_search_failed = True`` (``step`` is then the
        last trial step) or ``linear_solve_failed = True`` with the ``error`` text; a
        MINPACK pre-solve (``direct_below``) prepends
        ``{"method": "MINPACK", "evaluations", "residual"}``. With ``step_tol=None``
        no convergence-check solve is made.
    preconditioner : str
        Name of the preconditioner used.
    variable_names : tuple of str
    metadata : dict
        Solver settings, ``y_init``, ``y_end``, ``exogenous_path``, ``jvp_method``,
        ``tolerance_scale`` (the residual scale ``s``), ``seconds``, ``linear_solves``
        and ``converged_by`` (``"residual_and_step"``, ``"residual"`` when
        ``step_tol=None``, or the failure cause listed in :class:`StackedNewtonError`).
    """

    path: pd.DataFrame
    converged: bool
    iterations: int
    residual_norm: float
    initial_residual_norm: float
    terminal_error: float
    linear_history: tuple[Mapping[str, Any], ...] = ()
    preconditioner: str = "none"
    variable_names: tuple[str, ...] = ()
    metadata: dict[str, Any] = field(default_factory=dict)

    @property
    def n_periods(self) -> int:
        """Horizon T (rows of ``path``)."""
        return int(len(self.path))

    @property
    def n_vars(self) -> int:
        """Variables per date (columns of ``path``)."""
        return int(self.path.shape[1])

    def to_numpy(self) -> np.ndarray:
        """The trajectory as a ``(T, n_vars)`` array copy."""
        return self.path.to_numpy(dtype=float, copy=True)

    def to_dataframe(self) -> pd.DataFrame:
        """The trajectory as a DataFrame copy."""
        return self.path.copy()

    def to_frame(self) -> pd.DataFrame:
        """Alias of :meth:`to_dataframe` (matches :class:`PerfectForesightResult`)."""
        return self.path.copy()

    def __getitem__(self, key: str) -> pd.Series:
        """A trajectory column by variable name, or a result attribute by name."""
        if isinstance(key, str):
            if hasattr(self, key):
                return getattr(self, key)
            if key in self.path.columns:
                return self.path[key]
        raise KeyError(f"Variable or attribute {key!r} not found in StackedNewtonResult.")

    def linear_history_frame(self) -> pd.DataFrame:
        """The linear history as a DataFrame (one row per linear solve)."""
        if not self.linear_history:
            return pd.DataFrame()
        return pd.DataFrame([dict(h) for h in self.linear_history])

    def krylov_outer_iterations(self) -> tuple[int, ...]:
        """LGMRES outer iterations entered at each Newton step (IO convention)."""
        return tuple(int(h.get("krylov_outer_iterations", 0)) for h in self.linear_history
                     if "krylov_outer_iterations" in h)

    def summary(self) -> str:
        """Plain-text convergence summary."""
        outer = self.krylov_outer_iterations()
        seconds = self.metadata.get("seconds", float("nan"))
        lines = [
            "STACKED NEWTON-KRYLOV RESULT",
            "=" * 72,
            f"Convergence status  : {'CONVERGED' if self.converged else 'FAILED'}",
            f"Newton steps        : {self.iterations}",
            f"Linear solves       : {len(outer)} (outer iterations {list(outer)})",
            f"Residual norm       : {self.residual_norm:.4e}  (initial guess: {self.initial_residual_norm:.4e})",
            f"Terminal error      : {self.terminal_error:.4e}",
            f"Preconditioner      : {self.preconditioner}",
            f"Horizon / variables : {self.n_periods} periods, {self.n_vars} variables",
            f"Seconds             : {seconds:.3f}" if np.isfinite(seconds) else "Seconds             : n/a",
            "=" * 72,
        ]
        return "\n".join(lines)

    def to_markdown(self, *, head: int | None = None, index: bool = True, **kwargs: Any) -> str:
        """Render the trajectory (or its first ``head`` rows) as a Markdown table."""
        df = self.path.head(head) if head is not None else self.path
        return df_to_markdown(df, index=index, **kwargs)

    def to_latex(self, *, head: int | None = None, index: bool = True, **kwargs: Any) -> str:
        """Render the trajectory as a LaTeX tabular."""
        df = self.path.head(head) if head is not None else self.path
        return df_to_latex(df, index=index, **kwargs)

    def to_typst(self, *, head: int | None = None, index: bool = True, **kwargs: Any) -> str:
        """Render the trajectory as a Typst table."""
        df = self.path.head(head) if head is not None else self.path
        return df_to_typst(df, index=index, **kwargs)

    def to_perfect_foresight_result(self):
        """Convert to :class:`puremacro.dsge.PerfectForesightResult` (same field semantics)."""
        from puremacro.dsge.perfect_foresight import PerfectForesightResult

        return PerfectForesightResult(
            path=self.path.copy(), converged=bool(self.converged), iterations=int(self.iterations),
            residual_norm=float(self.residual_norm), terminal_error=float(self.terminal_error),
            variable_names=tuple(self.variable_names),
            initial_residual_norm=float(self.initial_residual_norm))

    def plot(self, variables: Sequence[str] | None = None, *, ax=None, title: str | None = None,
             xlabel: str = "Period (t)", ylabel: str = "Level", **kwargs: Any):
        """Plot trajectories in the package's black-and-white publication style."""
        from puremacro.plot import _new_ax
        from puremacro.plotting.bw_style import bw_colors, bw_linestyles

        if variables is not None:
            cols = [v for v in variables if v in self.path.columns]
            if not cols:
                raise ValueError(f"None of requested variables {list(variables)} found in {list(self.path.columns)}")
        else:
            cols = list(self.path.columns)
        fig, ax = _new_ax(ax)
        colors = bw_colors(len(cols))
        styles = bw_linestyles(len(cols))
        for i, col in enumerate(cols):
            ax.plot(self.path.index, self.path[col], label=str(col), color=colors[i],
                    linestyle=styles[i], linewidth=1.3, **kwargs)
        ax.spines["top"].set_visible(False)
        ax.spines["right"].set_visible(False)
        ax.grid(True, linestyle=":", linewidth=0.5, color="0.7", alpha=0.7)
        ax.set_xlabel(xlabel)
        ax.set_ylabel(ylabel)
        ax.set_title(title if title is not None else "Stacked Newton-Krylov transition")
        ax.legend(loc="best", frameon=False)
        return fig


@dataclass(frozen=True)
class HorizonComparison:
    """Horizon-doubling acceptance record from :func:`compare_horizons`.

    ``max_abs_difference`` is ``max_{t <= periods} |y^short_t - y^long_t|`` over all
    variables; ``passed`` is ``max_abs_difference <= tolerance``. ``variable_differences``
    holds the same maximum per variable, ``worst_variable``/``worst_period`` locate it
    (period is 1-based).
    """

    periods: int
    short_horizon: int
    long_horizon: int
    max_abs_difference: float
    tolerance: float
    passed: bool
    variable_names: tuple[str, ...] = ()
    variable_differences: tuple[float, ...] = ()
    worst_variable: str = ""
    worst_period: int = 0
    metadata: dict[str, Any] = field(default_factory=dict)

    def to_dataframe(self) -> pd.DataFrame:
        """Per-variable maximum absolute difference over the comparison window."""
        names = list(self.variable_names) or [f"y_{i}" for i in range(len(self.variable_differences))]
        df = pd.DataFrame({"variable": names, "max_abs_difference": list(self.variable_differences)})
        df["passed"] = df["max_abs_difference"] <= self.tolerance
        return df.set_index("variable")

    def summary(self) -> str:
        """One-line verdict with the statistic, tolerance and worst variable and date."""
        status = "PASSED" if self.passed else "FAILED"
        return (f"Horizon comparison {status}: max |y^{self.short_horizon} - y^{self.long_horizon}| "
                f"over the first {self.periods} periods = {self.max_abs_difference:.4e} "
                f"(tolerance {self.tolerance:.1e}; worst {self.worst_variable!r} at t={self.worst_period}).")

    def to_markdown(self, **kwargs: Any) -> str:
        """Per-variable table of :meth:`to_dataframe` as Markdown."""
        return df_to_markdown(self.to_dataframe(), **kwargs)

    def to_latex(self, **kwargs: Any) -> str:
        """Per-variable table of :meth:`to_dataframe` as a LaTeX tabular."""
        return df_to_latex(self.to_dataframe(), **kwargs)

    def to_typst(self, **kwargs: Any) -> str:
        """Per-variable table of :meth:`to_dataframe` as a Typst table."""
        return df_to_typst(self.to_dataframe(), **kwargs)


# ---------------------------------------------------------------------------
# Newton-Krylov harness
# ---------------------------------------------------------------------------
def _build_block_preconditioner(A: Any, B: Any, C: Any, T: int, method: str, time_block: int | None
                                ) -> tuple[BlockTridiagonalPreconditioner, bool]:
    """Build a block preconditioner; on a singular block-Thomas pivot fall back to SuperLU.

    Returns ``(operator, fell_back)``. Raises :class:`StackedNewtonError` when the
    stacked matrix itself cannot be factored (both methods fail).
    """
    if method == "thomas":
        try:
            return BlockTridiagonalPreconditioner(A, B, C, T, method="thomas", time_block=time_block), False
        except _SingularPivotError:
            pass
    try:
        return BlockTridiagonalPreconditioner(A, B, C, T, method="splu", time_block=time_block), method == "thomas"
    except ValueError as exc:
        raise StackedNewtonError(
            f"the block-tridiagonal preconditioner cannot be built: {exc}; the (block-masked) stacked "
            "matrix is singular at the expansion point, so supply a structured preconditioner or "
            "preconditioner=None") from exc


def _resolve_preconditioner(problem: StackedProblem, preconditioner: Any, time_block: int | None
                            ) -> tuple[str, Callable[[np.ndarray], spla.LinearOperator | None]]:
    """Return ``(name, factory)`` with ``factory(x) -> LinearOperator | None``.

    Bad option values raise ``ValueError``/``TypeError``; a steady-state matrix that
    cannot be factored raises :class:`StackedNewtonError` (the caller attaches the state).
    """
    T = problem.n_periods
    if preconditioner is None or (isinstance(preconditioner, str) and preconditioner == "none"):
        return "none", lambda x: None
    if isinstance(preconditioner, str):
        if preconditioner not in _PRECONDITIONER_NAMES:
            raise ValueError(f"unknown preconditioner {preconditioner!r}; expected one of "
                             f"{_PRECONDITIONER_NAMES}, a LinearOperator, a preconditioner object "
                             "or a factory callable")
        if preconditioner == "current_block_tridiagonal":
            def factory_current(x):
                A, B, C = problem.current_blocks(x)
                pre, _ = _build_block_preconditioner(A, B, C, T, "thomas", time_block)
                return pre.as_linear_operator()
            return "current_block_tridiagonal", factory_current
        A, B, C = problem.steady_blocks()
        if preconditioner == "steady_block_tridiagonal":
            pre, fell_back = _build_block_preconditioner(A, B, C, T, "thomas", time_block)
        elif preconditioner == "steady_splu":
            pre, fell_back = _build_block_preconditioner(A, B, C, T, "splu", time_block)
        else:  # time_block_jacobi
            pre, fell_back = _build_block_preconditioner(A, B, C, T, "thomas",
                                                         1 if time_block is None else time_block)
        name = pre.name[:-1] + ", thomas_pivot_fallback]" if fell_back else pre.name
        op = pre.as_linear_operator()
        return name, lambda x: op
    if hasattr(preconditioner, "as_linear_operator") or isinstance(preconditioner, spla.LinearOperator) \
            or sp.issparse(preconditioner) or isinstance(preconditioner, np.ndarray):
        op = _as_operator(preconditioner, (problem.n, problem.n))
        name = getattr(preconditioner, "name", None)
        if not name:
            name = "LinearOperator" if isinstance(preconditioner, spla.LinearOperator) else type(preconditioner).__name__
        return str(name), lambda x: op
    if callable(preconditioner):
        def factory_user(x):
            return _as_operator(preconditioner(x), (problem.n, problem.n))
        return getattr(preconditioner, "__name__", "factory"), factory_user
    raise TypeError("preconditioner must be a name, None, a LinearOperator, a preconditioner "
                    "object with as_linear_operator() or a factory callable")


def _step_test_message(res: float, res_bound: float, step_norm: float, step_bound: float, reason: str) -> str:
    return (f"Newton step test failed: the residual {res:.3g} meets tol * s = {res_bound:.3g}, but the "
            f"Newton step {step_norm:.3g} exceeds step_tol * max(1, |Y|_inf) = {step_bound:.3g} and {reason}. "
            "The residual may be at its noise floor (an inner iterative solve or finite-difference noise); "
            "pass step_tol=None for the residual-only acceptance rule of the IO engine, or a larger step_tol.")


def solve_stacked_newton_krylov(
    problem: StackedProblem,
    Y0: Any = None,
    *,
    preconditioner: Any = "steady_block_tridiagonal",
    tol: float = 1e-8,
    max_iter: int = 50,
    forcing: tuple[float, float] = (1e-5, 0.1),
    krylov_method: str = "lgmres",
    inner_m: int = 40,
    outer_k: int = 6,
    krylov_maxiter: int = 35,
    side: str = "right",
    max_log_move: float | None = 0.75,
    line_search: bool = True,
    max_backtracks: int = 23,
    step_tol: float | None = _STEP_TOL,
    jacobian_scale: float | None = None,
    direct_below: int = 0,
    time_block: int | None = None,
    progress: Callable | None = None,
    checkpoint: Callable | None = None,
) -> StackedNewtonResult:
    """Solve a :class:`StackedProblem` by inexact Newton with preconditioned LGMRES.

    Parameters
    ----------
    problem : StackedProblem
    Y0 : array_like, optional
        Initial path, shape ``(T, n_vars)`` or flat. Default: linear interpolation
        from ``y_init`` to ``y_end`` (the :func:`solve_perfect_foresight` guess).
    preconditioner : str, None, LinearOperator, preconditioner object or callable
        ``"steady_block_tridiagonal"`` (default; exact block-Thomas inverse of the
        steady-state stacked Jacobian, falling back to SuperLU when a leading
        block pivot is singular, in which case the recorded name ends in
        ``thomas_pivot_fallback``), ``"steady_splu"`` (same, SuperLU),
        ``"time_block_jacobi"`` (steady blocks, ``time_block`` dates per block,
        default 1), ``"current_block_tridiagonal"`` (refactored from the current
        date blocks at every Newton step: an exact Newton direction in one Krylov
        cycle at O(3 n) equation calls per date), ``"none"`` or ``None``; or a
        ready operator (anything with ``as_linear_operator()`` or accepted by
        ``scipy.sparse.linalg.aslinearoperator``); or a factory ``f(Y_flat) ->
        operator`` evaluated at every Newton iterate (the IO ``preconditioner_factory``).
    tol : float
        Residual tolerance; convergence requires ``|F|_inf <= tol * jacobian_scale``
        and, unless ``step_tol=None``, a Newton step below ``step_tol * max(1, |Y|_inf)``.
    max_iter : int
        Maximum accepted Newton steps.
    forcing : (eta_min, eta_max)
        Inexact-Newton forcing bounds: ``eta = max(eta_min, min(eta_max, sqrt(|F|_inf / s)))``.
    krylov_method : {"lgmres", "gmres"}
        Krylov method for every linear solve; ``"gmres"`` is an explicit choice,
        not an automatic fallback.
    inner_m, outer_k, krylov_maxiter : int
        LGMRES inner dimension, augmentation vectors and outer iteration cap
        (IO defaults 40, 6, 35); for GMRES ``inner_m`` is the restart length. A
        direction returned at the cap is not an error: its status and true linear
        residual are recorded and it is line-searched like any other direction (the
        IO rule); the solve fails only if that ends in a failed line search or the
        iteration limit.
    side : {"right", "left"}
    max_log_move : float or None
        Bound on the move per step in path units,
        ``step = min(1, max_log_move * max(1, |Y|_inf) / |d|_inf)``. The IO engine
        bounds the absolute move of its log variables by 0.75; the relative form is
        identical whenever ``|Y|_inf <= 1`` and keeps a model written in levels of
        order 1e3 or 1e4 converging in the same number of Newton steps. ``None``
        disables the bound.
    line_search : bool
        Armijo backtracking on ``|F|_2^2`` (``False`` takes the bounded full step
        as long as the residual is finite).
    max_backtracks : int
        Halvings of the step in the line search (default 23, i.e. 24 trial
        evaluations, the IO ``range(24)`` rule).
    step_tol : float or None
        Relative Newton-step tolerance of the convergence test (default 1e-10, the
        :func:`solve_perfect_foresight` constant). ``None`` (or ``inf``) drops the
        step test and accepts as soon as ``|F|_inf <= tol * s``, without a final
        linear solve; the untouched initial guess is then returned when it already
        meets the tolerance.
    jacobian_scale : float, optional
        Residual scale ``s``; default ``max(1, max |steady blocks|)``. Pass ``1.0``
        for the IO absolute residual criterion or to skip the steady-block
        evaluation; ``jacobian_scale=1.0`` together with ``step_tol=None`` is the
        acceptance rule of the IO ``native_solver._newton``.
    direct_below : int
        Try MINPACK ``hybr`` first when the stacked system has at most this many
        unknowns and keep it when it lowers the residual (the IO engine uses 240;
        the puremacro default 0 keeps every solve on the Krylov path).
    time_block : int, optional
        Block length for the named block-Jacobi preconditioners.
    progress : callable, optional
        Receives the per-step record and, every ten seconds inside a linear solve,
        a ``{"phase": "linear_solve", ...}`` record.
    checkpoint : callable, optional
        ``checkpoint(Y_flat_copy, record)`` after every accepted step.

    Returns
    -------
    StackedNewtonResult

    Raises
    ------
    StackedNewtonError
        Failed line search, iteration limit, a residual that meets the tolerance
        while the step test cannot be met (``step_test_failure``), an error or a
        nonfinite value inside a linear solve (Jacobian product or preconditioner
        application: ``linear_solve_failure``), a nonfinite direction, or a
        preconditioner that cannot be built (including a steady-state stacked matrix
        that cannot be factored). ``.result`` always holds the last state with
        ``converged=False`` and ``metadata["converged_by"]`` naming the cause.
    ValueError, TypeError
        Invalid options or a nonfinite residual at the initial path. A
        ``TypeError`` raised by user callables (for instance a complex-unsafe
        ``equations_fn`` under ``jvp_method="complex"``) propagates unchanged.
    """
    if not isinstance(problem, StackedProblem):
        raise TypeError("problem must be a StackedProblem")
    tol = _check_positive_float(tol, "tol")
    if step_tol is None or (isinstance(step_tol, (int, float, np.floating)) and not isinstance(step_tol, bool)
                            and float(step_tol) == float("inf")):
        step_tol = None      # residual-only acceptance (IO native_solver._newton)
    else:
        step_tol = _check_positive_float(step_tol, "step_tol")
    max_iter = _check_positive_int(max_iter, "max_iter")
    max_backtracks = _check_positive_int(max_backtracks, "max_backtracks", minimum=0)
    direct_below = _check_positive_int(direct_below, "direct_below", minimum=0)
    inner_m = _check_positive_int(inner_m, "inner_m")
    outer_k = _check_positive_int(outer_k, "outer_k", minimum=0)
    krylov_maxiter = _check_positive_int(krylov_maxiter, "krylov_maxiter")
    if krylov_method not in ("lgmres", "gmres"):
        raise ValueError("krylov_method must be 'lgmres' or 'gmres'")
    if side not in ("left", "right"):
        raise ValueError("side must be 'left' or 'right'")
    try:
        eta_min, eta_max = (float(forcing[0]), float(forcing[1]))
    except (TypeError, ValueError, IndexError) as exc:
        raise ValueError("forcing must be a pair (eta_min, eta_max)") from exc
    if not (0.0 < eta_min <= eta_max < 1.0):
        raise ValueError("forcing must satisfy 0 < eta_min <= eta_max < 1")
    if max_log_move is not None:
        max_log_move = _check_positive_float(max_log_move, "max_log_move")
    if time_block is not None:
        time_block = _check_positive_int(time_block, "time_block")
        if time_block > problem.n_periods:
            raise ValueError("time_block cannot exceed n_periods")
    if progress is not None and not callable(progress):
        raise TypeError("progress must be callable")
    if checkpoint is not None and not callable(checkpoint):
        raise TypeError("checkpoint must be callable")

    T, n, N = problem.n_periods, problem.n_vars, problem.n
    if Y0 is None:
        x = problem.linear_initial_path().ravel()
    else:
        x = np.array(problem.reshape(np.asarray(Y0, dtype=float)), dtype=float, copy=True).ravel()
        if not np.isfinite(x).all():
            raise ValueError("Y0 must be finite")
    tic = time.perf_counter()
    if jacobian_scale is None:
        A, B, C = problem.steady_blocks()
        jac_scale = max(1.0, _norm_inf(A), _norm_inf(B), _norm_inf(C))
    else:
        jac_scale = _check_positive_float(jacobian_scale, "jacobian_scale")
    try:
        f = problem.residual(x)
    except StackedNewtonError as exc:
        raise ValueError("the initial path has a nonfinite residual") from exc
    res = _norm_inf(f)
    initial_res = res
    history: list[dict[str, Any]] = []
    prec_name = preconditioner if isinstance(preconditioner, str) else "unbuilt"

    def build_result(x_now: np.ndarray, f_now: np.ndarray, converged: bool, steps: int,
                     converged_by: str) -> StackedNewtonResult:
        Y = _readonly(x_now.reshape(T, n))
        path = pd.DataFrame(Y, index=pd.RangeIndex(1, T + 1, name="t"),
                            columns=list(problem.variable_names), copy=False)
        meta = {
            "tol": tol, "tolerance_scale": jac_scale, "step_tol": step_tol,
            "forcing": (eta_min, eta_max), "krylov_method": krylov_method, "inner_m": inner_m,
            "outer_k": outer_k, "krylov_maxiter": krylov_maxiter, "side": side,
            "max_log_move": max_log_move, "line_search": bool(line_search),
            "max_backtracks": max_backtracks, "direct_below": direct_below,
            "jvp_method": problem.jvp_method, "n_periods": T, "n_vars": n,
            "y_init": _readonly(problem.y_init), "y_end": _readonly(problem.y_end),
            "exogenous_path": _readonly(problem.exogenous_path),
            "seconds": time.perf_counter() - tic,
            "linear_solves": sum(1 for h in history if "krylov_status" in h),
            "converged_by": converged_by,
        }
        return StackedNewtonResult(
            path=path, converged=bool(converged), iterations=int(steps),
            residual_norm=_norm_inf(f_now), initial_residual_norm=float(initial_res),
            terminal_error=float(_norm_inf(Y[-1] - problem.y_end)),
            linear_history=tuple(_FrozenRecord(h) for h in history), preconditioner=prec_name,
            variable_names=tuple(problem.variable_names), metadata=meta)

    try:
        prec_name, prec_factory = _resolve_preconditioner(problem, preconditioner, time_block)
    except StackedNewtonError as exc:
        raise StackedNewtonError(
            f"Preconditioner {preconditioner!r} failed before the first Newton iteration: {exc}",
            build_result(x, f, False, 0, "preconditioner_failure")) from exc

    # Optional MINPACK direct solve for tiny stacked systems (IO native_solver._newton).
    if direct_below and N <= direct_below and res > tol * jac_scale:
        fs = None
        try:
            small = scipy.optimize.root(problem.residual, x, method="hybr",
                                        options={"xtol": min(tol, 1e-10), "maxfev": max_iter * (N + 1)})
            fs = problem.residual(small.x)
        except (StackedNewtonError,) + _TRIAL_ERRORS:
            fs = None
        if fs is not None and np.isfinite(fs).all() and _norm_inf(fs) < res:
            x, f = np.array(small.x, dtype=float, copy=True), fs
            res = _norm_inf(f)
            history.append({"method": "MINPACK", "evaluations": int(small.nfev), "residual": res})

    converged = False
    converged_by = "iteration_limit"
    steps = 0
    fc = f
    small_res = False
    step_norm = float("nan")
    y_scale = max(1.0, _norm_inf(x))
    for it in range(max_iter + 1):
        res = _norm_inf(f)
        y_scale = max(1.0, _norm_inf(x))
        small_res = res <= tol * jac_scale
        if small_res and step_tol is None:
            converged, converged_by = True, "residual"
            break
        if it == max_iter and not small_res:
            break
        eta = max(eta_min, min(eta_max, math.sqrt(res / jac_scale)))
        rhs_norm = float(np.linalg.norm(f))
        try:
            M = prec_factory(x)
        except (StackedNewtonError, ValueError, np.linalg.LinAlgError) as exc:
            raise StackedNewtonError(f"Preconditioner failed at Newton iteration {it + 1}: {exc}",
                                     build_result(x, f, False, steps, "preconditioner_failure")) from exc
        count = [0]
        linear_start = time.perf_counter()
        last_update = [linear_start]

        def callback(_):
            count[0] += 1
            now = time.perf_counter()
            if progress is not None and now - last_update[0] >= 10.0:
                progress({"phase": "linear_solve", "newton_iteration": it + 1,
                          "krylov_outer_iteration": count[0], "seconds": now - linear_start,
                          "nonlinear_residual": res})
                last_update[0] = now

        try:
            product = problem.jvp(x)

            def matvec(v, product=product):
                out = product(v)
                if not np.isfinite(out).all():
                    raise StackedNewtonError("Nonfinite Jacobian-vector product")
                return out

            operator = spla.LinearOperator((N, N), matvec=matvec, dtype=float)
            d, info, true_res = preconditioned_lgmres(
                operator, -f, M, side=side, method=krylov_method, rtol=eta, atol=0.0,
                inner_m=inner_m, outer_k=outer_k, maxiter=krylov_maxiter, callback=callback)
        except _LINEAR_ERRORS as exc:
            history.append({"newton_iteration": it + 1, "residual_before": res,
                            "krylov_outer_iterations": int(count[0]), "linear_solve_failed": True,
                            "error": f"{type(exc).__name__}: {exc}",
                            "seconds_linear": time.perf_counter() - linear_start})
            raise StackedNewtonError(
                f"Linear solve failed at Newton iteration {it + 1} (residual {res:.3g}): "
                f"{type(exc).__name__}: {exc}",
                build_result(x, f, False, steps, "linear_solve_failure")) from exc
        record: dict[str, Any] = {
            "newton_iteration": it + 1, "residual_before": res,
            "krylov_outer_iterations": int(count[0]), "krylov_status": int(info),
            "true_linear_residual": true_res, "requested_tolerance": eta * rhs_norm,
            "rhs_norm": rhs_norm, "direction_norm": _norm_inf(d) if np.isfinite(d).all() else float("inf"),
            "seconds_linear": time.perf_counter() - linear_start,
        }
        if not np.isfinite(d).all():
            history.append(record)
            raise StackedNewtonError(
                f"Nonfinite Newton direction at residual {res:.3g}; Krylov status {info}.",
                build_result(x, f, False, steps, "nonfinite_direction"))
        step_norm = _norm_inf(d)
        if small_res and step_norm <= step_tol * y_scale:
            record.update({"step": 0.0, "backtracks": 0, "residual": res, "convergence_check": True})
            history.append(record)
            converged, converged_by = True, "residual_and_step"
            break
        if it == max_iter:
            history.append(record)
            break
        # Bound the move in path units (the IO rule scaled by max(1, |Y|_inf)); never clip the path.
        if max_log_move is None:
            step = 1.0
        else:
            bound = max_log_move * y_scale
            step = min(1.0, bound / max(bound, step_norm))
        merit = float(f @ f)
        accepted = False
        backtracks = 0
        for backtracks in range(max_backtracks + 1):
            if backtracks:
                step *= 0.5          # halve only when another trial follows (IO range(24) sequence)
            candidate = x + step * d
            try:
                fc = problem.residual(candidate)
                if line_search:
                    accepted = float(fc @ fc) <= merit * (1.0 - _ARMIJO_C * step)
                else:
                    accepted = True
            except (StackedNewtonError,) + _TRIAL_ERRORS:
                accepted = False
            if accepted or not line_search:
                break
        if not accepted:
            record.update({"step": step, "backtracks": backtracks, "line_search_failed": True})
            history.append(record)
            if small_res:
                raise StackedNewtonError(
                    _step_test_message(res, tol * jac_scale, step_norm, step_tol * y_scale,
                                       "the line search along it failed"),
                    build_result(x, f, False, steps, "step_test_failure"))
            raise StackedNewtonError(
                f"Newton line search failed at residual {res:.3g}; Krylov status {info} "
                f"(true linear residual {true_res:.3g}, requested {eta * rhs_norm:.3g}).",
                build_result(x, f, False, steps, "line_search_failure"))
        record.update({"step": step, "backtracks": backtracks, "residual": _norm_inf(fc),
                       "seconds": time.perf_counter() - tic})
        history.append(record)
        if progress is not None:
            progress(dict(record))
        x, f = candidate, fc
        steps = it + 1
        if checkpoint is not None:
            checkpoint(x.copy(), dict(record))

    if not converged:
        if small_res:
            raise StackedNewtonError(
                _step_test_message(_norm_inf(f), tol * jac_scale, step_norm, step_tol * y_scale,
                                   f"the iteration limit ({max_iter}) was reached"),
                build_result(x, f, False, steps, "step_test_failure"))
        raise StackedNewtonError(
            f"Newton iteration limit: residual {_norm_inf(f):.6g}, tolerance {tol:.3g} "
            f"(scaled by {jac_scale:.3g}) after {steps} accepted steps.",
            build_result(x, f, False, steps, "iteration_limit"))
    return build_result(x, f, True, steps, converged_by)


# ---------------------------------------------------------------------------
# Horizon comparison
# ---------------------------------------------------------------------------
def _trajectory(obj: Any, label: str) -> dict[str, Any]:
    """Extract path, names, convergence and boundary information from a result-like object."""
    info: dict[str, Any] = {"converged": None, "y_init": None, "y_end": None, "exo": None}
    if isinstance(obj, StackedNewtonResult):
        info["path"] = obj.to_numpy()
        info["names"] = tuple(obj.variable_names) or tuple(map(str, obj.path.columns))
        info["converged"] = bool(obj.converged)
        info["y_init"] = obj.metadata.get("y_init")
        info["y_end"] = obj.metadata.get("y_end")
        info["exo"] = obj.metadata.get("exogenous_path")
    elif isinstance(obj, pd.DataFrame):
        info["path"] = obj.to_numpy(dtype=float)
        info["names"] = tuple(map(str, obj.columns))
    elif hasattr(obj, "path") and isinstance(getattr(obj, "path"), pd.DataFrame):
        info["path"] = obj.path.to_numpy(dtype=float)
        info["names"] = tuple(getattr(obj, "variable_names", ()) or map(str, obj.path.columns))
        if hasattr(obj, "converged"):
            info["converged"] = bool(obj.converged)
    else:
        arr = np.asarray(obj, dtype=float)
        if arr.ndim != 2:
            raise ValueError(f"{label} trajectory must be a (T, n_vars) array, DataFrame or result")
        info["path"] = arr
        info["names"] = tuple(f"y_{i}" for i in range(arr.shape[1]))
    path = np.asarray(info["path"], dtype=float)
    if path.ndim != 2 or path.shape[0] < 1:
        raise ValueError(f"{label} trajectory must be a (T, n_vars) array")
    if not np.isfinite(path).all():
        raise ValueError(f"{label} trajectory contains nonfinite values")
    info["path"] = path
    return info


def compare_horizons(short: Any, long: Any, *, periods: int = 20, tolerance: float = 1e-5
                     ) -> HorizonComparison:
    """Horizon-doubling acceptance: compare the first ``periods`` dates of two solves.

    ``short`` and ``long`` are :class:`StackedNewtonResult` objects (preferred),
    :class:`PerfectForesightResult` objects, DataFrames or ``(T, n_vars)`` arrays.
    The long trajectory must be strictly longer, both must have the same number
    of variables, and results that carry a ``converged`` flag must be converged
    (the IO rule "Cannot compare unaccepted trajectories" raises
    :class:`StackedNewtonError`). When both results record their boundaries and
    exogenous paths, the boundaries must agree to 1e-9 and the long exogenous path
    must equal the short one on the shared dates and hold its last row afterwards
    (the IO "same announced policy, held beyond the short horizon" rule).

    The statistic is ``max_{t <= periods} |y^short_t - y^long_t|`` (the IO
    ``state_gap`` criterion, default tolerance 1e-5). Passing it at one horizon
    pair is evidence of horizon stability of the retained variables over the
    window only; it is not a certificate for the infinite-horizon problem.
    """
    tolerance = float(tolerance)
    if not np.isfinite(tolerance) or tolerance <= 0.0:
        raise ValueError("Horizon tolerance must be finite and positive.")
    a = _trajectory(short, "short")
    b = _trajectory(long, "long")
    Ts, Tl = a["path"].shape[0], b["path"].shape[0]
    periods = int(periods)
    if periods < 1 or periods > min(Ts, Tl):
        raise ValueError("Comparison periods must lie within both trajectories.")
    if Tl <= Ts:
        raise ValueError("Second solution must have a longer horizon.")
    if a["path"].shape[1] != b["path"].shape[1]:
        raise ValueError("Horizon trajectories must have the same state dimensions.")
    if a["converged"] is False or b["converged"] is False:
        raise StackedNewtonError("Cannot compare unaccepted trajectories.")
    for name in ("y_init", "y_end"):
        first, second = a[name], b[name]
        if first is not None and second is not None:
            if np.shape(first) != np.shape(second) or not np.allclose(first, second, rtol=0.0, atol=1e-9):
                raise ValueError(f"Horizon comparisons require identical {name} boundaries.")
    if a["exo"] is not None and b["exo"] is not None:
        es = np.asarray(a["exo"], dtype=float).reshape(Ts, -1)
        el = np.asarray(b["exo"], dtype=float).reshape(Tl, -1)
        held = np.broadcast_to(es[-1], (Tl - Ts, es.shape[1]))
        if (es.shape[1] != el.shape[1] or not np.allclose(el[:Ts], es, rtol=0.0, atol=1e-12)
                or not np.allclose(el[Ts:], held, rtol=0.0, atol=1e-12)):
            raise ValueError("Horizon comparisons require the same exogenous path, held beyond the short horizon.")
    diff = np.abs(a["path"][:periods] - b["path"][:periods])
    per_variable = diff.max(axis=0)
    error = float(per_variable.max(initial=0.0))
    names = tuple(a["names"]) if len(a["names"]) == diff.shape[1] else tuple(f"y_{i}" for i in range(diff.shape[1]))
    t_idx, v_idx = np.unravel_index(int(np.argmax(diff)), diff.shape)
    return HorizonComparison(
        periods=periods, short_horizon=int(Ts), long_horizon=int(Tl), max_abs_difference=error,
        tolerance=tolerance, passed=bool(error <= tolerance), variable_names=names,
        variable_differences=tuple(float(v) for v in per_variable),
        worst_variable=names[v_idx], worst_period=int(t_idx) + 1,
        metadata={"criterion": "state_gap", "norm": "max over variables and the first `periods` dates"})


compare_stacked_horizons = compare_horizons
