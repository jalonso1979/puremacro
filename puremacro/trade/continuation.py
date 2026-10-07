"""Parameter continuation and multistart helpers for audited policy equilibria.

Two small, generic tools on top of :func:`puremacro.trade.solve_policy_equilibrium`:

* :func:`continue_parameter` follows a model parameter (typically the
  intermediate sourcing elasticity ``sigma``) from an audited equilibrium to a
  target value at a fixed, complete tariff policy, with adaptive
  natural-parameter stepping. :func:`sigma_path` is the convenience wrapper for
  the elasticity path.
* :func:`try_starts` tries named warm starts at one exact target and returns
  the first audited root, recording every attempt.

Both helpers are ports of the research helpers ``continue_origin`` and
``try_direct_target`` in ``headlinePaper/retaliation_2026-09-22/continuation.py``
and of ``continue_ces`` in ``headlinePaper/rebuild/ces_continuation.py`` (IO
workspace), re-expressed on puremacro's consistent-accounting equilibrium
objects. The step law follows ``continue_origin``: halve the step after a
rejected trial, stop at ``min_step`` while retaining the last audited state,
and grow the step by ``growth`` (capped at ``max_step``) after a trial that
converged in at most ``fast_iterations`` iterations (``continue_ces`` used the
cap 0.4, five fast iterations and ``min_step`` 1/128; here all four are
parameters with the ``continue_origin`` values as defaults). One deliberate
deviation: a trial within 1e-12 of fraction one is taken as one, which avoids
the duplicated terminal stage the IO loop produces from float accumulation.
The IO helpers also capped every trial at 25 Newton iterations
(``continue_ces``: 12); here the cap is the solver's ``max_iter`` (100 in
``solve_policy_equilibrium``), so pass ``max_iter=25`` to reproduce that
rejection rule. Every accepted stage is re-audited with the
consistent-accounting evaluator (the same audit ``solve_policy_equilibrium``
applies), and the recorded tariff schedules and technology of every accepted
state must equal the requested ones.

As in the IO helpers, every trial is by default a Newton solve warm-started
from the previous accepted state (in :func:`try_starts`, from the named
start), and any other outcome halves the step. With ``allow_fallback=False``
the calibration adapter calls ``solve_policy_equilibrium`` with
``method="newton"`` and refuses ``"auto"``, ``"hybr"`` and ``"keller_pac"``,
whose seeds ignore the warm start; a returned state whose solver record shows
the fallback ladder, or a seed other than the supplied start, is a rejected
trial. ``allow_fallback=True`` opts into the ladder (hybrid from the
calibrated state, Keller continuation from the zero-tariff schedule); such a
stage is an audited root at the requested parameters but was not reached by
following the path. The start of a path is audited, not followed, so
:func:`sigma_path` solves a missing start with the full audited ladder.

These are audited path tools, not recovery guarantees. In the IO production
runs the production-origin homotopy and the direct-target multistart did not
recover any of the eleven low-elasticity GTAP cases they were written for (all
eleven ``recovery.json`` records are ``unresolved``). A qualification adapted
from the IO text to parameter paths is attached to every result; the IO
wording, quoted in ``docs/trade_continuation.md``, is::

    An admissible exact-target equilibrium found from an alternative start or
    production-elasticity path is not established as belonging to the forward
    low-elasticity tariff branch. Local stability is reported separately;
    neither convergence nor unsuccessful searches prove global uniqueness or
    nonexistence.

See ``docs/trade_continuation.md``.
"""
from __future__ import annotations

import copy
import hashlib
import json
import time
from dataclasses import dataclass, field
from numbers import Integral, Real
from typing import Any, Callable, Mapping, Sequence

import numpy as np
import pandas as pd

from puremacro.reports import df_to_latex, df_to_markdown, df_to_typst
from ._results import TradeCalibrationResult, TradeEquilibriumResult
from .policy_solver import PolicyEquilibriumError, solve_policy_equilibrium
from .solver import _resolve_tariffs, build_initial_guess
from .welfare import _checked_state

__all__ = [
    "DirectTargetResult",
    "ParameterContinuationFailure",
    "ParameterContinuationResult",
    "continue_parameter",
    "sigma_path",
    "try_starts",
]

_QUALIFICATION = (
    "An admissible exact-target equilibrium found from an alternative start or "
    "parameter path is not established as belonging to the forward tariff branch "
    "of the benchmark. Local stability is not assessed here; neither convergence "
    "nor unsuccessful searches prove global uniqueness or nonexistence."
)
# Numerical failures of a solve. ValueError/TypeError raised by the solver
# signal invalid inputs and propagate unchanged; a returned state that fails
# the audit is a numerical failure; a state carrying the wrong parameters is a
# broken contract (_ContractError) and propagates.
_SOLVER_FAILURES = (RuntimeError, ArithmeticError, np.linalg.LinAlgError)
_DEFAULT_TOLERANCE = 1e-8  # the default of solve_policy_equilibrium
# rtol = atol used to compare the reported fields (p, y, w, r, T, P, c,
# tariffs, gdp) with the recomputed equilibrium, as in solve_policy_equilibrium.
_FIELD_TOLERANCE = 1e-8
_AUDIT = "consistent-accounting re-evaluation at the requested tolerance"
_PATH_KEYS = ("method", "solver", "step_law", "allow_fallback", "continued", "invariant",
              "n_countries", "n_sectors", "unit", "solver_calls", "seconds", "audit", "field_tolerance")
_SIGMA_KEYS = ("sigma_start", "sigma_target", "tariff_schedules", "start_source", "start_requested_method")
_DIRECT_KEYS = ("method", "solver", "max_iter", "allow_fallback", "n_starts", "seconds",
                "n_countries", "n_sectors", "audit", "field_tolerance")


class _ContractError(ValueError):
    """The solver returned a state that does not carry the requested parameters."""


class ParameterContinuationFailure(RuntimeError):
    """A parameter path that could not be completed.

    ``state`` is the last audited :class:`TradeEquilibriumResult` (never the
    rejected trial) and ``result`` the partial :class:`ParameterContinuationResult`
    with ``status="unresolved"``, its accepted stages and every rejected trial.
    The message ends with the last solver attempt of the final rejected trial
    when the solver recorded its attempts.
    """

    def __init__(self, message: str, state: TradeEquilibriumResult,
                 result: "ParameterContinuationResult"):
        super().__init__(message)
        self.state = state
        self.result = result

    def __reduce__(self):
        return type(self), (str(self), self.state, self.result)


# ---------------------------------------------------------------------------
# Result containers
# ---------------------------------------------------------------------------
@dataclass(frozen=True)
class ParameterContinuationResult:
    """Accepted stages, rejected trials and the final state of a parameter path.

    ``fractions`` are the accepted path fractions in order (the first entry is
    the audited start). ``parameter_values`` holds the continued parameter at
    each accepted stage: a float when exactly one scalar parameter is continued,
    otherwise a dict of JSON-able fingerprints. ``stages`` and ``failures`` are
    JSON-serializable records (no state vectors) owned by the result: callbacks
    received independent copies. ``final`` is the last accepted state, an
    ordinary :class:`TradeEquilibriumResult` that welfare and post-processing
    functions accept unchanged. ``states`` retains every accepted state only
    when ``keep_states=True`` was requested. ``status`` is ``"accepted"`` when
    the path reached fraction one and ``"unresolved"`` for the partial result
    carried by :class:`ParameterContinuationFailure`.
    """

    fractions: np.ndarray
    parameter_values: tuple[Any, ...]
    parameter: str
    stages: tuple[dict[str, Any], ...]
    failures: tuple[dict[str, Any], ...]
    final: TradeEquilibriumResult
    states: tuple[TradeEquilibriumResult, ...]
    status: str
    start_fraction: float
    last_fraction: float
    invariant_id: str
    qualification: str
    metadata: dict[str, Any] = field(default_factory=dict)

    @property
    def n_stages(self) -> int:
        """Number of accepted stages, including the audited start."""
        return len(self.stages)

    def to_dataframe(self) -> pd.DataFrame:
        """One row per accepted stage: fraction, parameter, solver and audit diagnostics."""
        rows = []
        for k, stage in enumerate(self.stages):
            row: dict[str, Any] = {"stage": k, "fraction": stage["fraction"]}
            for name, value in stage["parameters"].items():
                row[name] = value if isinstance(value, Real) else json.dumps(value, sort_keys=True)
            row.update(step=stage["step"], method=stage["method"], seed=stage.get("seed"),
                       fallback_used=stage.get("fallback_used"), iterations=stage["iterations"],
                       solver_max_residual=stage["solver_max_residual"],
                       audited_max_residual=stage["audited_max_residual"],
                       seconds=stage["seconds"], minimum_price=stage["minimum_price"],
                       minimum_output_ratio=stage["minimum_output_ratio"],
                       minimum_wage=stage["minimum_wage"])
            rows.append(row)
        return pd.DataFrame(rows).set_index("stage")

    def summary(self) -> str:
        """One-line status, stage and failure counts, and the qualification."""
        audited = max((s["audited_max_residual"] for s in self.stages), default=float("nan"))
        return (f"Parameter continuation [{self.parameter or 'identity'}]: status={self.status}, "
                f"{len(self.stages)} accepted stages, {len(self.failures)} rejected trials, "
                f"last fraction {self.last_fraction:.6g}, max audited residual {audited:.3g}. "
                f"{self.qualification}")

    def to_markdown(self, **kwargs: Any) -> str:
        return df_to_markdown(self.to_dataframe(), **kwargs)

    def to_latex(self, **kwargs: Any) -> str:
        return df_to_latex(self.to_dataframe(), **kwargs)

    def to_typst(self, **kwargs: Any) -> str:
        return df_to_typst(self.to_dataframe(), **kwargs)

    def plot(self, values: Any = None, *, ax: Any = None, label: str | None = None) -> Any:
        """Draw a scalar per accepted stage against the continued parameter.

        ``values`` is ``None`` (the audited residual of each stage on a log
        axis), the name of a stage record entry, a sequence with one number per
        accepted stage, or a callable applied to each retained state (requires
        ``keep_states=True``), for example a country's Hicksian EV. The x axis is
        the parameter when a single scalar parameter is continued and the path
        fraction otherwise. Never calls ``plt.show()``.
        """
        import matplotlib.pyplot as plt

        scalar = all(isinstance(v, Real) for v in self.parameter_values) and self.parameter
        x = np.asarray(self.parameter_values, dtype=float) if scalar else self.fractions
        if values is None:
            y = [s["audited_max_residual"] for s in self.stages]
            name, log = "audited max residual", True
        elif isinstance(values, str):
            y = [s[values] for s in self.stages]
            name, log = values, False
        elif callable(values):
            if len(self.states) != len(self.stages):
                raise ValueError("A callable needs the accepted states; rerun with keep_states=True")
            y = [float(values(state)) for state in self.states]
            name, log = getattr(values, "__name__", "value"), False
        else:
            y = np.asarray(values, dtype=float).ravel()
            if len(y) != len(self.stages):
                raise ValueError("values must supply one number per accepted stage")
            name, log = "value", False
        if ax is None:
            _, ax = plt.subplots()
        ax.plot(x, y, marker="o", label=label)
        ax.set_xlabel(self.parameter if scalar else "path fraction")
        ax.set_ylabel(name)
        if log:
            ax.set_yscale("log")
        if label is not None:
            ax.legend()
        return ax


@dataclass(frozen=True)
class DirectTargetResult:
    """Every attempted start at one exact target and the first audited root.

    ``attempts`` records each start in order with ``status`` ``"accepted"`` or
    ``"unresolved"``; a rejected start is never substituted for a root.
    ``final`` is the accepted :class:`TradeEquilibriumResult` or ``None``, and
    ``accepted_start`` its name. ``target`` holds JSON-able fingerprints of the
    requested solver keyword arguments.
    """

    attempts: tuple[dict[str, Any], ...]
    accepted_start: str | None
    final: TradeEquilibriumResult | None
    status: str
    target: dict[str, Any]
    qualification: str
    metadata: dict[str, Any] = field(default_factory=dict)

    def to_dataframe(self) -> pd.DataFrame:
        """One row per attempted start."""
        rows = []
        for attempt in self.attempts:
            rows.append({"start": attempt["name"], "status": attempt["status"],
                         "method": attempt.get("method"), "seed": attempt.get("seed"),
                         "iterations": attempt.get("iterations"),
                         "audited_max_residual": attempt.get("audited_max_residual"),
                         "seconds": attempt.get("seconds"), "error": attempt.get("error")})
        return pd.DataFrame(rows, columns=["start", "status", "method", "seed", "iterations",
                                           "audited_max_residual", "seconds", "error"]).set_index("start")

    def summary(self) -> str:
        """One-line status, attempt count, accepted start and the qualification."""
        return (f"Direct target multistart: status={self.status}, {len(self.attempts)} attempts, "
                f"accepted start {self.accepted_start!r}. {self.qualification}")

    def to_markdown(self, **kwargs: Any) -> str:
        return df_to_markdown(self.to_dataframe(), **kwargs)

    def to_latex(self, **kwargs: Any) -> str:
        return df_to_latex(self.to_dataframe(), **kwargs)

    def to_typst(self, **kwargs: Any) -> str:
        return df_to_typst(self.to_dataframe(), **kwargs)


# ---------------------------------------------------------------------------
# Private helpers
# ---------------------------------------------------------------------------
def _array_digest(value: np.ndarray) -> str:
    arr = np.ascontiguousarray(np.asarray(value, dtype=float))
    return hashlib.sha256(arr.tobytes()).hexdigest()


def _jsonable(value: Any) -> Any:
    """JSON-able record of a value: scalars and lists as themselves, arrays as shape and digest."""
    if isinstance(value, TradeEquilibriumResult):
        return {"type": "TradeEquilibriumResult", "sha256": _array_digest(value.x_sol)}
    if isinstance(value, TradeCalibrationResult):
        return {"type": "TradeCalibrationResult",
                "sha256": _array_digest(np.r_[value.a.ravel(), value.afd.ravel(), value.ytot.ravel()])}
    if isinstance(value, Mapping):
        return {str(k): _jsonable(v) for k, v in value.items()}
    if isinstance(value, np.ndarray):
        if value.ndim == 0:
            return value.item()
        return {"shape": [int(n) for n in value.shape], "sha256": _array_digest(value)}
    if isinstance(value, np.generic):
        return value.item()
    if isinstance(value, (list, tuple)):
        return [_jsonable(v) for v in value]
    if value is None or isinstance(value, (bool, int, float, str)):
        return value
    return repr(value)


def _fingerprint(value: Any) -> str:
    return json.dumps(_jsonable(value), sort_keys=True, default=repr)


def _nonnegative_integer(value: Any, name: str) -> int:
    if isinstance(value, bool) or not isinstance(value, Integral) or value < 0:
        raise ValueError(f"{name} must be a nonnegative integer")
    return int(value)


def _check_steps(initial_step, min_step, max_step, growth, fast_iterations, start_fraction):
    values = (initial_step, min_step, max_step, growth, start_fraction)
    if not all(isinstance(v, Real) and not isinstance(v, bool) and np.isfinite(v) for v in values):
        raise ValueError("Continuation settings must be finite real numbers")
    if not 0 < min_step <= initial_step <= max_step <= 1:
        raise ValueError("Steps must satisfy 0 < min_step <= initial_step <= max_step <= 1")
    if min_step < 1e-12:
        raise ValueError("min_step must be at least 1e-12 so that every trial advances the fraction")
    if growth < 1:
        raise ValueError("growth must be at least one")
    if not 0 <= start_fraction <= 1:
        raise ValueError("start_fraction must be finite and in [0, 1]")
    _nonnegative_integer(fast_iterations, "fast_iterations")


def _user_metadata(metadata: Any, reserved: Sequence[str]) -> dict[str, Any]:
    """User metadata as a dict; keys that the result records itself are refused."""
    if metadata is None:
        return {}
    if not isinstance(metadata, Mapping):
        raise TypeError("metadata must be a mapping")
    clash = sorted(str(k) for k in metadata if str(k) in reserved)
    if clash:
        raise ValueError(f"metadata keys {clash} are reserved for the continuation record")
    return dict(metadata)


def _adapter_kwargs(kwargs: Mapping[str, Any], allow_fallback: bool) -> dict[str, Any]:
    """Solver keyword arguments for ``solve_policy_equilibrium``: Newton unless the ladder is allowed."""
    kw = dict(kwargs)
    method = kw.setdefault("method", "auto" if allow_fallback else "newton")
    if not allow_fallback and method != "newton":
        raise ValueError(
            f'method={method!r} does not start from the warm start ("auto" runs the fallback ladder, '
            '"hybr" starts from the calibrated state and "keller_pac" from the zero-tariff schedule); '
            'pass allow_fallback=True to accept such stages or use method="newton"')
    return kw


def _resolve_solver(solve: Any, calib: TradeCalibrationResult | None, allow_fallback: bool):
    """Return ``(solve(kwargs, x0), calib, name, check(kwargs))`` for a calibration or a callable."""
    if not isinstance(allow_fallback, bool):
        raise TypeError("allow_fallback must be a bool")
    if isinstance(solve, TradeCalibrationResult):
        if calib is not None and calib is not solve:
            raise ValueError("calib must be the calibration supplied as solve, or omitted")

        def adapter(kwargs: Mapping[str, Any], x0: np.ndarray | None) -> TradeEquilibriumResult:
            return solve_policy_equilibrium(solve, x0=x0, **_adapter_kwargs(kwargs, allow_fallback))

        def check(kwargs: Mapping[str, Any]) -> None:
            _adapter_kwargs(kwargs, allow_fallback)

        return adapter, solve, "solve_policy_equilibrium", check
    if callable(solve):
        if not isinstance(calib, TradeCalibrationResult):
            raise TypeError("A TradeCalibrationResult is required (as solve or as calib) for the stage audit")
        return solve, calib, getattr(solve, "__name__", type(solve).__name__), lambda kwargs: None
    raise TypeError("solve must be a TradeCalibrationResult or a callable solve(kwargs, x0)")


def _requested_tolerance(kwargs: Mapping[str, Any]) -> float:
    tol = kwargs.get("tol", _DEFAULT_TOLERANCE)
    if not isinstance(tol, Real) or isinstance(tol, bool) or not np.isfinite(tol) or tol <= 0:
        raise ValueError("Solver kwargs must carry a finite positive tol")
    return float(tol)


def _check_kwargs(kwargs: Any, where: str) -> dict[str, Any]:
    if not isinstance(kwargs, Mapping) or any(not isinstance(k, str) for k in kwargs):
        raise TypeError(f"{where} must return a mapping of solver keyword arguments")
    if "x0" in kwargs:
        raise ValueError(f"{where} must not set x0; the continuation supplies the warm start")
    return dict(kwargs)


def _check_parameters(result: Any, calib: TradeCalibrationResult, kwargs: Mapping[str, Any]) -> None:
    """The returned state must carry the requested closure, technology and tariff schedules."""
    if not isinstance(result, TradeEquilibriumResult):
        raise TypeError("solve must return a TradeEquilibriumResult")
    meta = result.metadata
    if meta.get("accounting") != "consistent":
        raise _ContractError('Continuation requires accounting="consistent" equilibria')
    if meta.get("fiscal_closure") != "lump_sum":
        raise _ContractError("Continuation requires lump-sum fiscal transfers, the closure that "
                             "solve_policy_equilibrium and its audit support")
    sigma = kwargs.get("sigma", 0.)
    if not isinstance(sigma, Real) or isinstance(sigma, bool) or not np.isfinite(sigma) or sigma < 0:
        raise ValueError("sigma must be finite and nonnegative")
    if meta.get("sigma") is None or float(meta["sigma"]) != float(sigma):
        raise _ContractError("Returned technology differs from the requested sigma")
    ta, tf, _, _ = _resolve_tariffs(calib, kwargs.get("tau"), kwargs.get("tau_fd"))
    for name, expected in (("intermediate_tariff_multipliers", ta), ("final_tariff_multipliers", tf)):
        actual = meta.get(name)
        if actual is None or np.shape(actual) != expected.shape or not np.array_equal(actual, expected):
            raise _ContractError("Returned tariff schedules differ from the requested policy")


def _last_attempt(result: TradeEquilibriumResult) -> Mapping[str, Any] | None:
    attempts = result.metadata.get("policy_solver_attempts") or ()
    if attempts and isinstance(attempts[-1], Mapping):
        return attempts[-1]
    return None


def _audit(result: TradeEquilibriumResult, calib: TradeCalibrationResult, kwargs: Mapping[str, Any],
           *, allow_fallback: bool, warm_started: bool) -> tuple[float, dict[str, Any]]:
    """Re-evaluate every equilibrium and accounting equation at the returned state.

    Raises ``ValueError`` when the state does not declare convergence, when
    ``allow_fallback`` is false and the solver record shows the fallback
    ladder or (for a warm-started solve) a seed other than the supplied start,
    when the state fails its recorded tolerance, when the reported fields
    disagree with the recalculated equilibrium (rtol = atol = 1e-8), or when
    the audited residual exceeds the requested ``tol``.
    """
    tol = _requested_tolerance(kwargs)
    if not result.converged:
        raise ValueError("Solver did not return a converged equilibrium")
    if not allow_fallback:
        if result.metadata.get("policy_solver_fallback_used"):
            raise ValueError("State was reached through the solver fallback ladder (hybrid from the "
                             "calibrated state or Keller continuation from the zero-tariff schedule), "
                             "not from the warm start; pass allow_fallback=True to accept such stages")
        last = _last_attempt(result)
        seed = None if last is None else last.get("seed")
        if warm_started and seed is not None and seed != "warm":
            raise ValueError(f"State was solved from the {seed} seed, not from the supplied start; "
                             "pass allow_fallback=True to accept such stages")
    try:
        blocks, _ = _checked_state(result, calib, _FIELD_TOLERANCE)
    except NotImplementedError as exc:
        raise _ContractError(f"Continuation cannot audit this state: {exc}") from exc
    residual = np.r_[blocks["residuals"], blocks["physical_residuals"]]
    audited = float(np.max(np.abs(residual)))
    if not np.isfinite(audited) or audited > tol:
        raise ValueError(f"Audited residual {audited:.3g} exceeds the requested tolerance {tol:.3g}")
    return audited, blocks


def _solver_attempts(exc: BaseException) -> list[dict[str, Any]] | None:
    """JSON-able summary of ``PolicyEquilibriumError.attempts`` (or ``None``)."""
    attempts = getattr(exc, "attempts", None)
    if not isinstance(attempts, (list, tuple)):
        return None
    keys = ("method", "seed", "solver_converged", "solver_max_residual", "audited_max_residual", "error")
    return [{k: _jsonable(a[k]) for k in keys if k in a} for a in attempts if isinstance(a, Mapping)]


def _describe(exc: BaseException) -> str:
    """``str(exc)`` followed by the last recorded solver attempt, when there is one."""
    text = str(exc)
    attempts = _solver_attempts(exc)
    if attempts:
        last = attempts[-1]
        residual = last.get("solver_max_residual")
        residual_text = "unavailable" if not isinstance(residual, Real) else f"{residual:.3g}"
        text += (f" (last attempt: {last.get('method')} from the {last.get('seed')} seed, "
                 f"converged={last.get('solver_converged')}, max residual {residual_text}: "
                 f"{last.get('error', 'no error recorded')})")
    return text


def _diagnostics(result: TradeEquilibriumResult, calib: TradeCalibrationResult,
                 blocks: Mapping[str, Any]) -> dict[str, float]:
    y0 = np.asarray(calib.ytot, dtype=float).ravel()
    y = np.asarray(result.y_sol, dtype=float).ravel()
    return dict(minimum_price=float(np.min(result.p_sol)),
                minimum_output_ratio=float(np.min(y/y0)),
                minimum_wage=float(np.min(result.w_sol)),
                minimum_final_expenditure=float(np.min(blocks["expenditure"])))


def _solver_record(result: TradeEquilibriumResult) -> dict[str, Any]:
    meta = result.metadata
    last = _last_attempt(result)
    seed = None if last is None else last.get("seed")
    return dict(method=str(meta.get("policy_solver_method", meta.get("method", "unknown"))),
                seed=None if seed is None else str(seed),
                iterations=int(result.iterations),
                solver_max_residual=float(result.max_residual) if np.isfinite(result.max_residual) else None,
                fallback_used=bool(meta.get("policy_solver_fallback_used", False)))


def _continued_keys(path: Callable[[float], Mapping[str, Any]], continued: Any) -> tuple[str, ...]:
    if continued is not None:
        keys = (continued,) if isinstance(continued, str) else tuple(continued)
        if not keys or any(not isinstance(k, str) for k in keys):
            raise ValueError("continued must name one or more solver keyword arguments")
        return keys
    first, last = _check_kwargs(path(0.), "path"), _check_kwargs(path(1.), "path")
    if set(first) != set(last):
        raise ValueError("path must return the same keyword arguments at every fraction")
    return tuple(k for k in first if _fingerprint(first[k]) != _fingerprint(last[k]))


def _invariant_of(kwargs: Mapping[str, Any], continued: Sequence[str], invariant: Any) -> str:
    if invariant is not None:
        return _fingerprint(invariant(kwargs))
    return _fingerprint({k: _fingerprint(v) for k, v in kwargs.items() if k not in continued})


def _parameter_record(kwargs: Mapping[str, Any], continued: Sequence[str]) -> dict[str, Any]:
    return {k: _jsonable(kwargs[k]) for k in continued if k in kwargs}


def _parameter_value(record: Mapping[str, Any]) -> Any:
    if len(record) == 1:
        value = next(iter(record.values()))
        if isinstance(value, Real) and not isinstance(value, bool):
            return float(value)
    return dict(record)


def _seed(x0: Any) -> np.ndarray | None:
    if x0 is None:
        return None
    if isinstance(x0, TradeEquilibriumResult):
        x0 = x0.x_sol
    seed = np.asarray(x0, dtype=float).ravel()
    if seed.size == 0 or not np.isfinite(seed).all():
        raise ValueError("Start vector must be a finite nonempty equilibrium vector")
    return seed.copy()


def _readonly(values: Sequence[float]) -> np.ndarray:
    arr = np.array(values, dtype=float, copy=True)
    arr.flags.writeable = False
    return arr


# ---------------------------------------------------------------------------
# Public functions
# ---------------------------------------------------------------------------
def continue_parameter(
    solve: Any,
    path: Callable[[float], Mapping[str, Any]],
    start: TradeEquilibriumResult,
    *,
    calib: TradeCalibrationResult | None = None,
    initial_step: float = .25,
    min_step: float = 1/1024,
    max_step: float = .5,
    growth: float = 1.5,
    fast_iterations: int = 6,
    start_fraction: float = 0.,
    invariant: Callable[[Mapping[str, Any]], Any] | None = None,
    continued: str | Sequence[str] | None = None,
    stage_callback: Callable[[TradeEquilibriumResult, dict[str, Any]], Any] | None = None,
    progress: Callable[[dict[str, Any]], Any] | None = None,
    keep_states: bool = False,
    allow_fallback: bool = False,
    metadata: Mapping[str, Any] | None = None,
) -> ParameterContinuationResult:
    """Follow a model parameter from an audited state with adaptive natural-parameter steps.

    ``path(fraction)`` returns the solver keyword arguments at path fraction
    ``f`` in ``[0, 1]`` (for example ``{"tau": ta, "tau_fd": tf, "sigma": 2*f,
    "tol": 1e-9}``); it must never set ``x0``. ``solve`` is either a
    :class:`TradeCalibrationResult`, in which case every trial calls
    :func:`solve_policy_equilibrium` with those arguments and the previous
    accepted state as warm start, or a callable ``solve(kwargs, x0)`` returning
    a consistent-accounting :class:`TradeEquilibriumResult` (then ``calib`` is
    required for the audit). A solver signals numerical failure by raising
    ``RuntimeError`` (``PolicyEquilibriumError``), ``ArithmeticError`` or
    ``LinAlgError``; ``ValueError``/``TypeError`` mean invalid inputs and
    propagate. ``start`` must already solve the equations at
    ``path(start_fraction)``: it is audited, never re-solved, so a resumed path
    is checked at its actual fraction.

    Step law (from the IO ``continue_origin``; ``continue_ces`` differs only in
    its constants): a trial at ``min(1, f + step)`` that fails halves ``step``;
    when ``step`` falls below ``min_step`` the path stops and
    :class:`ParameterContinuationFailure` carries the last audited state. An
    accepted trial whose solver reported at most ``fast_iterations``
    iterations multiplies ``step`` by ``growth``, capped at ``max_step``. A
    trial within 1e-12 of fraction one is taken as one (the IO loop can record
    a duplicate terminal stage from float accumulation). Settings must satisfy
    ``1e-12 <= min_step <= initial_step <= max_step <= 1`` and ``growth >= 1``,
    which guarantees that every accepted trial advances the fraction.
    ``iterations`` is the solver's own count: Newton iterations, SciPy ``hybr``
    function evaluations, or Keller continuation steps. The per-trial
    iteration cap is the solver's ``max_iter`` (100 by default in
    ``solve_policy_equilibrium``); the IO helpers rejected trials needing more
    than 25 Newton iterations (``continue_ces``: 12), which ``path`` reproduces
    by returning ``max_iter=25``.

    Acceptance: a trial is accepted only when the returned state (i) declares
    convergence, (ii) carries consistent accounting, lump-sum transfers and
    exactly the requested ``sigma`` and tariff schedules, (iii) passes the
    consistent-accounting re-evaluation of all equilibrium and accounting
    equations (including the omitted goods market and foreign balances) at its
    recorded tolerance, with every reported field (prices, outputs, factor
    prices, transfers, price indices, final demand, tariff revenue and GDP)
    equal to the recomputed equilibrium to ``rtol = atol = 1e-8``, (iv) has an
    audited maximum absolute residual at most the requested ``tol`` (default
    1e-8, the default of ``solve_policy_equilibrium``; ``tol`` is absolute in
    the calibration's units), and (v) unless ``allow_fallback=True``, was
    neither reached through the policy solver's fallback ladder
    (``metadata["policy_solver_fallback_used"]``) nor, according to its
    recorded last attempt, solved from a seed other than the warm start. A
    state violating (ii) is a broken contract and raises ``ValueError``; the
    other failures are rejected trials. The audit is the model's own
    evaluator, as in ``solve_policy_equilibrium``, not a separately derived
    oracle.

    Warm starts: with ``allow_fallback=False`` (default, the IO semantics) the
    calibration adapter calls ``solve_policy_equilibrium`` with
    ``method="newton"`` and refuses any other method (``"auto"`` runs the
    fallback ladder, ``"hybr"`` starts from the calibrated state and
    ``"keller_pac"`` from the zero-tariff schedule, so none of them follows the
    path). With ``allow_fallback=True`` the adapter defaults to
    ``method="auto"`` and an accepted stage may be an audited root reached by
    ``hybr`` from the calibrated state or by Keller tariff continuation from
    the zero-tariff schedule rather than by following the path; every stage
    records ``seed`` (``"warm"`` or ``"calibrated"``) and ``fallback_used``.
    The supplied ``start`` is audited but not subject to this restriction.

    Invariant: by default every keyword argument that ``path`` leaves unchanged
    between fractions 0 and 1 (tariff schedules by their bytes, scalars by
    value, other states by their digest) must be identical at every trial;
    ``continued`` names the arguments allowed to change explicitly and
    ``invariant(kwargs)`` replaces the default fingerprint. A violation raises
    ``ValueError`` before the trial is solved. An identity path (nothing
    continued) still audits the supplied state and reports fraction one.

    ``stage_callback(state, stage)`` receives every accepted state with an
    independent copy of its record (the caller persists states; the result
    keeps them only with ``keep_states=True``), and ``progress(event)``
    receives trial, accepted and rejected events. Rejected-trial records carry
    ``solver_attempts`` (method, seed, convergence flag, residuals and error of
    every policy-solver attempt) when the solver raised
    ``PolicyEquilibriumError``. ``metadata`` is merged into the result's
    metadata; keys the result records itself (``method``, ``solver``,
    ``audit``, ...) raise ``ValueError``.

    This is an audited path tool: the accepted endpoint is not established as
    the continuation of any other branch, and an unresolved path proves neither
    a fold nor nonexistence.
    """
    started = time.perf_counter()
    solver, calib, solver_name, check_kwargs = _resolve_solver(solve, calib, allow_fallback)
    user_metadata = _user_metadata(metadata, _PATH_KEYS)
    if not callable(path):
        raise TypeError("path must be a callable path(fraction) -> solver keyword arguments")
    if not isinstance(start, TradeEquilibriumResult):
        raise TypeError("start must be a TradeEquilibriumResult")
    _check_steps(initial_step, min_step, max_step, growth, fast_iterations, start_fraction)
    keys = _continued_keys(path, continued)
    fraction, step = float(start_fraction), float(initial_step)
    kwargs = _check_kwargs(path(fraction), "path")
    check_kwargs(kwargs)
    if any(k not in kwargs for k in keys):
        raise ValueError("continued names keyword arguments that path does not return")
    invariant_value = _invariant_of(kwargs, keys, invariant)
    invariant_id = hashlib.sha256(invariant_value.encode()).hexdigest()
    try:
        _check_parameters(start, calib, kwargs)
        audited, blocks = _audit(start, calib, kwargs, allow_fallback=True, warm_started=False)
    except ValueError as exc:
        raise ValueError(f"Start state is not an audited equilibrium at start_fraction "
                         f"{fraction:.9g}: {exc}") from exc
    stages: list[dict[str, Any]] = []
    failures: list[dict[str, Any]] = []
    states: list[TradeEquilibriumResult] = []
    calls = 0
    settings = dict(initial_step=float(initial_step), min_step=float(min_step),
                    max_step=float(max_step), growth=float(growth), fast_iterations=int(fast_iterations))

    def accept(state, kw, audited_value, block_values, seconds, previous, used_step):
        stage = dict(fraction=fraction, previous_fraction=previous, step=used_step,
                     parameters=_parameter_record(kw, keys), requested_tolerance=_requested_tolerance(kw),
                     audited_max_residual=audited_value, seconds=seconds,
                     **_solver_record(state), **_diagnostics(state, calib, block_values))
        stages.append(stage)
        if keep_states:
            states.append(state)
        if stage_callback is not None:
            stage_callback(state, copy.deepcopy(stage))
        if progress is not None:
            progress(dict(event="accepted", **copy.deepcopy(stage)))
        return state

    def build(status, state, last):
        return ParameterContinuationResult(
            fractions=_readonly([s["fraction"] for s in stages]),
            parameter_values=tuple(_parameter_value(s["parameters"]) for s in stages),
            parameter=",".join(keys), stages=tuple(copy.deepcopy(s) for s in stages),
            failures=tuple(copy.deepcopy(f) for f in failures), final=state, states=tuple(states),
            status=status, start_fraction=float(start_fraction), last_fraction=float(last),
            invariant_id=invariant_id, qualification=_QUALIFICATION,
            metadata={**user_metadata,
                      "method": "adaptive natural-parameter continuation", "solver": solver_name,
                      "step_law": dict(settings), "allow_fallback": allow_fallback,
                      "continued": list(keys), "invariant": invariant_value,
                      "n_countries": int(calib.nc), "n_sectors": int(calib.ns),
                      "unit": calib.metadata.get("unit", "calibration value units"),
                      "solver_calls": calls, "seconds": time.perf_counter()-started,
                      "audit": _AUDIT, "field_tolerance": _FIELD_TOLERANCE})

    def reject(exc, trial, used_step, record, tic):
        """Record a rejected trial; halve the step or stop with the last audited state."""
        failure = dict(fraction=trial, previous_fraction=fraction, step=used_step, parameters=record,
                       error=str(exc), exception_type=type(exc).__name__,
                       solver_attempts=_solver_attempts(exc), seconds=time.perf_counter()-tic)
        failures.append(failure)
        if progress is not None:
            progress(dict(event="rejected", **copy.deepcopy(failure)))
        halved = used_step*.5
        if halved < min_step:
            raise ParameterContinuationFailure(
                f"Continuation unresolved after fraction {fraction:.9g}: {_describe(exc)}",
                state, build("unresolved", state, fraction)) from exc
        return halved

    state = accept(start, kwargs, audited, blocks, 0., fraction, 0.)
    if not keys:
        return build("accepted", state, 1.)
    while fraction < 1:
        trial = min(1., fraction + step)
        if 1. - trial < 1e-12:  # accumulated rounding must not add a duplicate final stage
            trial = 1.
        kwargs = _check_kwargs(path(trial), "path")
        if _invariant_of(kwargs, keys, invariant) != invariant_value:
            raise ValueError("Continuation changed its policy or other model assumptions along the path")
        record = _parameter_record(kwargs, keys)
        if progress is not None:
            progress(dict(event="trial", fraction=trial, previous_fraction=fraction,
                          step=step, parameters=copy.deepcopy(record)))
        tic = time.perf_counter()
        calls += 1
        try:
            candidate = solver(kwargs, state.x_sol.copy())
        except _SOLVER_FAILURES as exc:
            step = reject(exc, trial, step, record, tic)
            continue
        _check_parameters(candidate, calib, kwargs)
        try:
            audited, blocks = _audit(candidate, calib, kwargs, allow_fallback=allow_fallback,
                                     warm_started=True)
        except ValueError as exc:
            step = reject(exc, trial, step, record, tic)
            continue
        previous, fraction = fraction, trial
        state = accept(candidate, kwargs, audited, blocks, time.perf_counter()-tic, previous, step)
        if stages[-1]["iterations"] <= fast_iterations:
            step = min(max_step, step*growth)
    return build("accepted", state, fraction)


def sigma_path(
    calib: TradeCalibrationResult,
    tau: np.ndarray | None,
    tau_fd: np.ndarray | None,
    sigma_target: float,
    *,
    sigma_start: float = 0.,
    x0: np.ndarray | None = None,
    start: TradeEquilibriumResult | None = None,
    start_fraction: float = 0.,
    initial_step: float = .25,
    min_step: float = 1/1024,
    max_step: float = .5,
    growth: float = 1.5,
    fast_iterations: int = 6,
    stage_callback: Callable[[TradeEquilibriumResult, dict[str, Any]], Any] | None = None,
    progress: Callable[[dict[str, Any]], Any] | None = None,
    keep_states: bool = False,
    allow_fallback: bool = False,
    metadata: Mapping[str, Any] | None = None,
    **solver_options: Any,
) -> ParameterContinuationResult:
    """Continue the intermediate sourcing elasticity at a fixed, complete tariff policy.

    The path is ``sigma(f) = sigma_start + f*(sigma_target - sigma_start)`` with
    the exact endpoints at ``f = 0`` and ``f = 1``; ``f`` never means a tariff
    fraction. ``tau`` and ``tau_fd`` are the complete intermediate and final
    tariff multipliers (``None`` is free trade) and stay fixed along the path.
    ``solver_options`` (``tol``, ``max_iter``, ``max_steps``, ``method``,
    ``base_result``) are passed to :func:`solve_policy_equilibrium` at every
    trial; ``method`` defaults to ``"newton"``, and any other method requires
    ``allow_fallback=True`` (see :func:`continue_parameter`).

    Start: a supplied ``start`` is audited at ``sigma(start_fraction)`` without
    re-solving, which is how a saved stage is resumed (``x0`` is then
    refused). When ``start`` is omitted it is solved at ``sigma(start_fraction)``
    from ``x0`` with the full audited ladder of ``solve_policy_equilibrium``
    (``method="auto"`` unless ``solver_options`` names a method): the start is
    audited like a supplied one, not followed, so the Newton-only rule applies
    to the trials only. The first stage records the method, seed and fallback
    use of that solve, and ``metadata["start_source"]`` is ``"solved"`` or
    ``"supplied"``. A failed start solve raises ``PolicyEquilibriumError``
    naming that fraction before any stage exists. An identity path
    (``sigma_start == sigma_target``) audits the start without further solves
    (``metadata["solver_calls"]`` counts trials only). Everything else is
    :func:`continue_parameter` with ``continued="sigma"``.

    The endpoint is an audited equilibrium at ``sigma_target`` under the
    requested schedules; it may be compared with a direct
    ``solve_policy_equilibrium(calib, tau, tau_fd, sigma=sigma_target)``.
    Neither agreement nor disagreement between the two proves uniqueness.
    """
    if not isinstance(calib, TradeCalibrationResult):
        raise TypeError("calib must be a TradeCalibrationResult")
    if not isinstance(allow_fallback, bool):
        raise TypeError("allow_fallback must be a bool")
    user_metadata = _user_metadata(metadata, _PATH_KEYS + _SIGMA_KEYS)
    for name, value in (("sigma_start", sigma_start), ("sigma_target", sigma_target)):
        if (not isinstance(value, Real) or isinstance(value, bool)
                or not np.isfinite(value) or value < 0):
            raise ValueError(f"{name} must be finite and nonnegative")
    _check_steps(initial_step, min_step, max_step, growth, fast_iterations, start_fraction)
    if "sigma" in solver_options or "x0" in solver_options:
        raise ValueError("sigma and x0 are set by the path; pass sigma_start/sigma_target and x0 explicitly")
    if start is not None and x0 is not None:
        raise ValueError("Pass either start (an audited state to resume) or x0 (a seed for the start solve), not both")
    ta, tf, _, _ = _resolve_tariffs(calib, tau, tau_fd)
    options = _adapter_kwargs(solver_options, allow_fallback)

    def sigma_at(f: float) -> float:
        if f >= 1:
            return float(sigma_target)
        if f <= 0:
            return float(sigma_start)
        return float(sigma_start + f*(sigma_target - sigma_start))

    def path(f: float) -> dict[str, Any]:
        return dict(tau=ta, tau_fd=tf, sigma=sigma_at(f), **options)

    record: dict[str, Any] = {"sigma_start": float(sigma_start), "sigma_target": float(sigma_target),
                              "tariff_schedules": "fixed and complete along the path",
                              "start_source": "supplied", "start_requested_method": None}
    if start is None:
        start_options = dict(options, method=solver_options.get("method", "auto"))
        record.update(start_source="solved", start_requested_method=start_options["method"])
        try:
            start = solve_policy_equilibrium(calib, ta, tf, x0=_seed(x0), sigma=sigma_at(start_fraction),
                                             **start_options)
        except PolicyEquilibriumError as exc:
            raise PolicyEquilibriumError(
                f"Start solve at sigma={sigma_at(start_fraction):.9g} (fraction {float(start_fraction):.9g}) "
                f"failed before any stage: {_describe(exc)}", exc.attempts) from exc
    identity = float(sigma_start) == float(sigma_target)
    return continue_parameter(
        calib, path, start, initial_step=initial_step, min_step=min_step, max_step=max_step,
        growth=growth, fast_iterations=fast_iterations, start_fraction=start_fraction,
        continued=None if identity else "sigma", stage_callback=stage_callback, progress=progress,
        keep_states=keep_states, allow_fallback=allow_fallback, metadata={**user_metadata, **record})


def try_starts(
    solve: Any,
    target: Mapping[str, Any],
    starts: Sequence[Mapping[str, Any]],
    *,
    calib: TradeCalibrationResult | None = None,
    max_iter: int = 35,
    progress: Callable[[dict[str, Any]], Any] | None = None,
    allow_fallback: bool = False,
    metadata: Mapping[str, Any] | None = None,
) -> DirectTargetResult:
    """Try named warm starts at one exact target and return the first audited root.

    ``target`` holds the solver keyword arguments of the unchanged target
    (schedules, ``sigma``, ``tol``, ...); ``max_iter`` (default 35, as in the IO
    ``try_direct_target``) is added to them for every attempt. Each entry of
    ``starts`` is a mapping with a nonempty ``name``, an ``x0`` (an equilibrium
    vector, a :class:`TradeEquilibriumResult`, or ``None`` for the calibrated
    initial state) and an optional JSON-able ``provenance``, stored verbatim
    (arrays as shape and digest). ``solve``, ``calib`` and ``allow_fallback``
    are as in :func:`continue_parameter`: by default each start is a single
    Newton solve from that start (the IO ``try_direct_target`` semantics), and
    a state reached through the fallback ladder or, for a start with a vector,
    from a seed other than that vector is recorded as unresolved. With
    ``allow_fallback=True`` a root may come from a seed that ignores the named
    start; each attempt records ``seed`` and ``fallback_used``. A start whose
    vector is not finite or does not have the calibration's state length is
    recorded as unresolved without a solve, as is a start whose solve raises
    a numerical failure (with ``solver_attempts`` when the solver recorded
    them). Attempts stop at the first state that passes the parameter check
    and the audit of :func:`continue_parameter`; a rejected start is never
    substituted for a root, and an unresolved search returns ``final=None``.
    """
    started = time.perf_counter()
    solver, calib, solver_name, check_kwargs = _resolve_solver(solve, calib, allow_fallback)
    user_metadata = _user_metadata(metadata, _DIRECT_KEYS)
    max_iter = _nonnegative_integer(max_iter, "max_iter")
    if max_iter < 1:
        raise ValueError("max_iter must be a positive integer")
    kwargs = _check_kwargs(target, "target")
    check_kwargs(kwargs)
    kwargs["max_iter"] = max_iter
    entries = list(starts)
    for entry in entries:
        if not isinstance(entry, Mapping) or not entry.get("name") or "x0" not in entry:
            raise ValueError("Every start requires a name and x0")
    expected_size = int(build_initial_guess(calib).size)
    attempts: list[dict[str, Any]] = []
    accepted: TradeEquilibriumResult | None = None
    accepted_name = None

    def unresolved(attempt, exc, tic):
        attempt.update(status="unresolved", error=str(exc), exception_type=type(exc).__name__,
                       solver_attempts=_solver_attempts(exc), seconds=time.perf_counter()-tic)
        attempts.append(attempt)

    for entry in entries:
        attempt: dict[str, Any] = dict(name=str(entry["name"]), provenance=_jsonable(entry.get("provenance", {})),
                                       max_iter=max_iter)
        tic = time.perf_counter()
        if progress is not None:
            progress(dict(event="start", name=attempt["name"], parameters=_jsonable(kwargs)))
        try:
            seed = _seed(entry["x0"])
            if seed is not None and seed.size != expected_size:
                raise ValueError(f"Start vector has {seed.size} entries; the calibration's equilibrium "
                                 f"vector has {expected_size}")
        except ValueError as exc:
            unresolved(attempt, exc, tic)
            continue
        try:
            result = solver(kwargs, seed)
        except _SOLVER_FAILURES as exc:
            unresolved(attempt, exc, tic)
            continue
        _check_parameters(result, calib, kwargs)
        try:
            audited, blocks = _audit(result, calib, kwargs, allow_fallback=allow_fallback,
                                     warm_started=seed is not None)
        except ValueError as exc:
            unresolved(attempt, exc, tic)
            continue
        attempt.update(status="accepted", audited_max_residual=audited,
                       requested_tolerance=_requested_tolerance(kwargs),
                       seconds=time.perf_counter()-tic, **_solver_record(result),
                       **_diagnostics(result, calib, blocks))
        attempts.append(attempt)
        accepted, accepted_name = result, attempt["name"]
        break
    return DirectTargetResult(
        attempts=tuple(attempts), accepted_start=accepted_name, final=accepted,
        status="accepted" if accepted is not None else "unresolved",
        target=_jsonable(kwargs), qualification=_QUALIFICATION,
        metadata={**user_metadata,
                  "method": "direct_exact_target", "solver": solver_name, "max_iter": max_iter,
                  "allow_fallback": allow_fallback, "n_starts": len(entries),
                  "seconds": time.perf_counter()-started,
                  "n_countries": int(calib.nc), "n_sectors": int(calib.ns),
                  "audit": _AUDIT, "field_tolerance": _FIELD_TOLERANCE})
