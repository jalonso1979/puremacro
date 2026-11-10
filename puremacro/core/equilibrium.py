"""Dense local linearizations of explicitly supplied equilibrium equations.

This interface differentiates a finite residual system, not a numerical solver's
iterations. It never replaces a singular derivative with a regularized answer.
Local stencil checks do not establish global smoothness or branch uniqueness.
"""
from __future__ import annotations

from collections.abc import Callable, Hashable, Sequence
from dataclasses import dataclass, field
import hashlib
from typing import Any

import numpy as np
import pandas as pd
from scipy.linalg import lu_factor, lu_solve

from puremacro.reports import df_to_latex, df_to_markdown, df_to_typst
from puremacro._numeric_contracts import real_scalar

__all__ = [
    "EquilibriumSystem", "EquilibriumLinearization", "EquilibriumSensitivityResult",
    "linearize_equilibrium",
]

_Equation = Callable[[np.ndarray, np.ndarray], np.ndarray]


def _array(value: Any, shape: tuple[int, ...], name: str) -> np.ndarray:
    raw = np.asarray(value)
    if np.iscomplexobj(raw):
        raise ValueError(f"{name} must be real")
    out = np.asarray(raw, dtype=float)
    if out.shape != shape or not np.isfinite(out).all():
        raise ValueError(f"{name} must be finite with shape {shape}; got {out.shape}")
    return out.copy()


def _freeze(value: Any) -> np.ndarray:
    a = np.asarray(value, dtype=float)
    return np.frombuffer(a.tobytes(), dtype=float).reshape(a.shape)


def _names(value: Sequence[str], name: str) -> tuple[str, ...]:
    if isinstance(value, (str, bytes)):
        raise ValueError(f"{name} must be a sequence of names, not a string")
    out = tuple(value)
    if not out or any(not isinstance(s, str) or not s.strip() for s in out) or len(set(out)) != len(out):
        raise ValueError(f"{name} must contain unique nonempty strings")
    return out


def _positive(value: float, name: str) -> float:
    value = real_scalar(value, name)
    if value <= 0:
        raise ValueError(f"{name} must be positive and finite")
    return value


@dataclass(frozen=True)
class EquilibriumSystem:
    """Square residual ``F(state, parameters)`` with named, scaled coordinates.

    All callbacks have the exact two-argument signature above and must be pure.
    Generic callbacks remain caller-owned; the built-in model adapters snapshot
    their numerical inputs. Scales are positive coordinate/equation units, not
    data-dependent row equilibration. Supply corresponding scales when changing
    units. A regime callback returns a hashable active-set/interpolation label.
    The initial implementation uses dense matrices and a state-count guard.
    """

    residual: _Equation
    state_names: tuple[str, ...]
    parameter_names: tuple[str, ...]
    state_scale: Any = None
    parameter_scale: Any = None
    residual_scale: Any = None
    state_jacobian: _Equation | None = None
    parameter_jacobian: _Equation | None = None
    regime: Callable[[np.ndarray, np.ndarray], Hashable] | None = None
    equation_names: tuple[str, ...] | None = None
    label: str = "equilibrium"

    def __post_init__(self) -> None:
        for name in ("state_names", "parameter_names"):
            object.__setattr__(self, name, _names(getattr(self, name), name))
        n, p = len(self.state_names), len(self.parameter_names)
        equations = self.state_names if self.equation_names is None else _names(self.equation_names, "equation_names")
        if len(equations) != n:
            raise ValueError("equation_names must match the square state dimension")
        object.__setattr__(self, "equation_names", equations)
        for name, size in (("state_scale", n), ("parameter_scale", p), ("residual_scale", n)):
            raw = getattr(self, name)
            arr = np.ones(size) if raw is None else _array(raw, (size,), name)
            if np.any(arr <= 0):
                raise ValueError(f"{name} must be strictly positive")
            object.__setattr__(self, name, _freeze(arr))
        for name in ("residual", "state_jacobian", "parameter_jacobian", "regime"):
            callback = getattr(self, name)
            if not callable(callback) and (name == "residual" or callback is not None):
                raise TypeError(f"{name} must be callable")

    def evaluate(self, state: Any, parameters: Any) -> np.ndarray:
        """Evaluate the declared equations; callback exceptions propagate."""
        n, p = len(self.state_names), len(self.parameter_names)
        x, theta = _array(state, (n,), "state"), _array(parameters, (p,), "parameters")
        return _array(self.residual(x, theta), (n,), "residual")


@dataclass(frozen=True)
class EquilibriumSensitivityResult:
    """Local state derivatives, or NaNs with explicit ineligibility reasons.

    ``valid`` concerns the retained discretized equations and tested stencils.
    It is not a statistical identification or economic-validity assertion.
    """

    state_names: tuple[str, ...]
    parameter_names: tuple[str, ...]
    jacobian: np.ndarray
    status: str
    reasons: tuple[str, ...]
    equilibrium_residual: float
    linearized_residual: float
    condition_number: float
    branch_id: str | None
    snapshot_id: str

    def __post_init__(self) -> None:
        object.__setattr__(self, "jacobian", _freeze(self.jacobian))

    @property
    def valid(self) -> bool:
        return self.status == "regular"

    def to_frame(self) -> pd.DataFrame:
        """Derivatives in physical state units per physical parameter unit."""
        return pd.DataFrame(self.jacobian.copy(), index=self.state_names, columns=self.parameter_names)

    def to_dataframe(self) -> pd.DataFrame:
        return self.to_frame()

    def summary(self) -> pd.DataFrame:
        return pd.DataFrame({"status": [self.status], "reasons": [", ".join(self.reasons)],
                             "equilibrium_residual": [self.equilibrium_residual],
                             "linearized_residual": [self.linearized_residual],
                             "condition_number": [self.condition_number]})

    def to_markdown(self, **kwargs: Any) -> str:
        return df_to_markdown(self.to_frame().rename_axis("state").reset_index(), index=False, **kwargs)

    def to_latex(self, **kwargs: Any) -> str:
        return df_to_latex(self.to_frame().rename_axis("state").reset_index(), index=False, **kwargs)

    def to_typst(self, **kwargs: Any) -> str:
        return df_to_typst(self.to_frame().rename_axis("state").reset_index(), index=False, **kwargs)

    def plot(self, *, states: Sequence[str] | None = None, ax: Any = None) -> Any:
        """Plot selected derivative rows; never calls ``show``."""
        import matplotlib.pyplot as plt

        frame = self.to_frame()
        if states is not None:
            frame = frame.loc[list(states)]
        if len(frame) > 30:
            raise ValueError("select at most 30 states for a readable derivative plot")
        if ax is None:
            _, ax = plt.subplots()
        frame.plot.bar(ax=ax)
        ax.set_title(f"Local equilibrium sensitivities ({self.status})")
        return ax


@dataclass(frozen=True)
class EquilibriumLinearization:
    """Owned numerical snapshot and a reference to its residual system.

    ``snapshot_id`` hashes the retained point/matrices/scales, not Python code.
    ``branch_id`` is a caller label and does not certify branch discovery.
    """

    system: EquilibriumSystem
    state: np.ndarray
    parameters: np.ndarray
    residual: np.ndarray
    state_jacobian: np.ndarray
    parameter_jacobian: np.ndarray
    state_steps: np.ndarray
    parameter_steps: np.ndarray
    state_derivative_source: str
    parameter_derivative_source: str
    reasons: tuple[str, ...]
    condition_number: float
    smallest_scaled_singular_value: float
    equilibrium_residual: float
    branch_id: str | None = None
    _factor: Any = field(default=None, init=False, repr=False, compare=False)

    def __post_init__(self) -> None:
        for name in ("state", "parameters", "residual", "state_jacobian", "parameter_jacobian",
                     "state_steps", "parameter_steps"):
            object.__setattr__(self, name, _freeze(getattr(self, name)))

    @property
    def status(self) -> str:
        return self.reasons[0] if self.reasons else "regular"

    @property
    def snapshot_id(self) -> str:
        digest = hashlib.sha256()
        digest.update(repr((self.system.state_names, self.system.parameter_names,
                            self.system.equation_names, self.branch_id)).encode("utf-8"))
        for value in (self.state, self.parameters, self.residual, self.state_jacobian,
                      self.parameter_jacobian, self.system.state_scale,
                      self.system.parameter_scale, self.system.residual_scale):
            digest.update(np.asarray(value, dtype="<f8").tobytes())
        return digest.hexdigest()

    def jvp(self, direction: Any) -> np.ndarray:
        return self.state_jacobian @ _array(direction, self.state.shape, "direction")

    def vjp(self, cotangent: Any) -> np.ndarray:
        return self.state_jacobian.T @ _array(cotangent, self.residual.shape, "cotangent")

    def solve(self, rhs: Any, *, transpose: bool = False) -> np.ndarray:
        """Solve the retained state Jacobian; refuse ineligible derivatives."""
        if not isinstance(transpose, (bool, np.bool_)):
            raise ValueError("transpose must be boolean")
        if self.reasons:
            raise ValueError("equilibrium linearization is ineligible: " + ", ".join(self.reasons))
        raw = np.asarray(rhs)
        n = len(self.state)
        if raw.ndim not in (1, 2) or raw.shape[0] != n:
            raise ValueError("rhs must have the state dimension as its first axis")
        values = _array(raw, raw.shape, "rhs")
        vector = values.ndim == 1
        values = values[:, None] if vector else values
        xs = self.system.state_scale[:, None]
        rs = self.system.residual_scale[:, None]
        if self._factor is None:
            scaled = self.state_jacobian * xs.T / rs
            object.__setattr__(self, "_factor", lu_factor(scaled))
        if transpose:
            answer = lu_solve(self._factor, xs * values, trans=1) / rs
        else:
            answer = xs * lu_solve(self._factor, values / rs)
        return answer[:, 0] if vector else answer

    def sensitivity(self) -> EquilibriumSensitivityResult:
        """Compute ``dx/dtheta`` with one shared factorization and no regularization."""
        reasons = self.reasons
        answer = np.full(self.parameter_jacobian.shape, np.nan)
        error = float("nan")
        if not reasons:
            answer = self.solve(-self.parameter_jacobian)
            remainder = self.state_jacobian @ answer + self.parameter_jacobian
            scaled = remainder * self.system.parameter_scale[None, :] / self.system.residual_scale[:, None]
            error = float(np.max(np.abs(scaled)))
            if not np.isfinite(answer).all() or not np.isfinite(error) or error > 1e-7:
                reasons = ("linear_solve_failed",)
                answer = np.full_like(answer, np.nan)
        return EquilibriumSensitivityResult(
            self.system.state_names, self.system.parameter_names, answer,
            reasons[0] if reasons else "regular", reasons, self.equilibrium_residual,
            error, self.condition_number, self.branch_id, self.snapshot_id,
        )


def _signature(system: EquilibriumSystem, x: np.ndarray, theta: np.ndarray) -> Hashable:
    if system.regime is None:
        return None
    signature = system.regime(x.copy(), theta.copy())
    try:
        hash(signature)
    except TypeError as exc:
        raise TypeError("regime must return a hashable signature, such as a tuple or bytes") from exc
    return signature


def _differentiate(system: EquilibriumSystem, x: np.ndarray, theta: np.ndarray,
                   base: np.ndarray, *, parameters: bool, step: float,
                   tolerance: float) -> tuple[np.ndarray, np.ndarray, list[str]]:
    point = theta if parameters else x
    scale = system.parameter_scale if parameters else system.state_scale
    columns = np.empty((len(x), len(point)))
    used = np.empty(len(point))
    reasons: list[str] = []
    signature = _signature(system, x, theta)
    for column in range(len(point)):
        h = step * scale[column]
        same_regime, consistent = True, True
        for _ in range(9):
            plus, minus = point.copy(), point.copy()
            plus[column] += h
            minus[column] -= h
            hp, hm = plus[column] - point[column], point[column] - minus[column]
            if min(hp, hm) <= 0:
                raise ValueError("finite-difference step is not representable; revise coordinate scales")
            xp, pp = (x, plus) if parameters else (plus, theta)
            xm, pm = (x, minus) if parameters else (minus, theta)
            fp, fm = system.evaluate(xp, pp), system.evaluate(xm, pm)
            same_regime = (_signature(system, xp, pp) == signature == _signature(system, xm, pm))
            derivative = (fp - fm) / (hp + hm)
            change = ((fp - base) / hp - (base - fm) / hm) * scale[column] / system.residual_scale
            magnitude = np.abs(derivative * scale[column] / system.residual_scale)
            consistent = bool(np.max(np.abs(change) / (1 + magnitude)) <= tolerance)
            if same_regime and consistent:
                break
            h *= 0.5
        columns[:, column] = derivative
        used[column] = 0.5 * (hp + hm)
        if not same_regime:
            reasons.append("regime_change")
        if not consistent:
            reasons.append("inconsistent_stencil")
    return columns, used, reasons


def linearize_equilibrium(
    system: EquilibriumSystem, state: Any, parameters: Any, *,
    step: float = 1e-5, residual_tol: float = 1e-7, derivative_tol: float = 1e-3,
    singular_tol: float = 1e-10, condition_limit: float = 1e10,
    max_states: int = 1500, branch_id: str | None = None,
) -> EquilibriumLinearization:
    """Linearize a square system at the supplied point without re-solving it.

    Numerical derivatives use centered stencils in declared coordinate units,
    shrinking up to eight times for regime changes or one-sided disagreement.
    Supplied Jacobians are checked against these same independent stencils.
    Singular-value diagnostics use ``diag(1/residual_scale) F_x diag(state_scale)``.
    A non-equilibrium or failed diagnostic remains inspectable, with no ordinary
    sensitivity returned. No statistical inference or grid-convergence claim is made.
    """
    if not isinstance(system, EquilibriumSystem):
        raise TypeError("system must be an EquilibriumSystem")
    if isinstance(max_states, bool) or not isinstance(max_states, (int, np.integer)) or max_states < 1:
        raise ValueError("max_states must be a positive integer")
    n, p = len(system.state_names), len(system.parameter_names)
    if n > max_states:
        raise ValueError(f"dense linearization has {n} states, exceeding max_states={max_states}")
    step = _positive(step, "step")
    residual_tol = _positive(residual_tol, "residual_tol")
    derivative_tol = _positive(derivative_tol, "derivative_tol")
    singular_tol = _positive(singular_tol, "singular_tol")
    condition_limit = _positive(condition_limit, "condition_limit")
    if branch_id is not None and (not isinstance(branch_id, str) or not branch_id):
        raise ValueError("branch_id must be a nonempty label or None")
    x, theta = _array(state, (n,), "state"), _array(parameters, (p,), "parameters")
    residual = system.evaluate(x, theta)
    residual_norm = float(np.max(np.abs(residual / system.residual_scale)))
    reasons = ["not_equilibrium"] if residual_norm > residual_tol else []
    numerical_x, sx, flags_x = _differentiate(system, x, theta, residual, parameters=False,
                                             step=step, tolerance=derivative_tol)
    numerical_p, sp, flags_p = _differentiate(system, x, theta, residual, parameters=True,
                                             step=step, tolerance=derivative_tol)
    reasons.extend(flags_x + flags_p)
    matrices, sources = [], []
    for callback, numerical, scales in (
        (system.state_jacobian, numerical_x, system.state_scale),
        (system.parameter_jacobian, numerical_p, system.parameter_scale),
    ):
        if callback is None:
            matrices.append(numerical)
            sources.append("central_residual_differences")
        else:
            supplied = _array(callback(x.copy(), theta.copy()), numerical.shape, "supplied Jacobian")
            reference = numerical * scales[None, :] / system.residual_scale[:, None]
            gap = (supplied - numerical) * scales[None, :] / system.residual_scale[:, None]
            if float(np.max(np.abs(gap) / (1 + np.abs(reference)))) > derivative_tol:
                reasons.append("derivative_mismatch")
            matrices.append(supplied)
            sources.append("supplied")
    jx, jp = matrices
    scaled = jx * system.state_scale[None, :] / system.residual_scale[:, None]
    if not (np.isfinite(jx).all() and np.isfinite(jp).all() and np.isfinite(scaled).all()):
        raise ValueError("nonfinite derivative matrix; revise equations, coordinate scales or step")
    singular_values = np.linalg.svd(scaled, compute_uv=False)
    largest, smallest = float(singular_values[0]), float(singular_values[-1])
    condition = largest / smallest if smallest > 0 else float("inf")
    if smallest <= singular_tol * max(1.0, largest):
        reasons.append("singular")
    elif condition > condition_limit:
        reasons.append("ill_conditioned")
    return EquilibriumLinearization(
        system, x, theta, residual, jx, jp, sx, sp, sources[0], sources[1],
        tuple(dict.fromkeys(reasons)), condition, smallest, residual_norm, branch_id,
    )
