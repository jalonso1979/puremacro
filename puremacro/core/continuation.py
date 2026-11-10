"""Dense single-parameter pseudo-arclength continuation of equilibrium equations.

The path is local, with explicit numerical termination and fold diagnostics.
It does not enumerate disconnected branches or infer economic stability.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import numpy as np
import pandas as pd

from puremacro.core.equilibrium import (
    EquilibriumLinearization, EquilibriumSystem, _array, _freeze, _positive,
    _signature, linearize_equilibrium,
)
from puremacro.reports import df_to_latex, df_to_markdown, df_to_typst

__all__ = [
    "ContinuationAttempt", "EquilibriumFold", "EquilibriumBranchResult",
    "continue_equilibrium",
]


@dataclass(frozen=True)
class ContinuationAttempt:
    """One attempted predictor/corrector step, including rejected steps."""

    from_index: int
    step: float
    accepted: bool
    iterations: int
    residual_norm: float
    arclength_residual: float
    reason: str


@dataclass(frozen=True)
class EquilibriumFold:
    """Numerical turning-point evidence; classification is tolerance-dependent."""

    left_index: int
    right_index: int
    state: np.ndarray
    parameters: np.ndarray
    tangent: np.ndarray
    equilibrium_residual: float
    state_corank: int
    augmented_condition_number: float
    transversality: float
    curvature: float
    curvature_error: float
    status: str
    reason: str
    iterations: int

    def __post_init__(self) -> None:
        for name in ("state", "parameters", "tangent"):
            object.__setattr__(self, name, _freeze(getattr(self, name)))


@dataclass(frozen=True)
class EquilibriumBranchResult:
    """Accepted equilibria and explicit termination of a local branch trace.

    Tangents are unit vectors in scaled state/selected-parameter coordinates.
    arclength is accumulated chord length in those coordinates. step_sizes are
    the requested hyperplane distances, with zero at the supplied initial point.
    All other parameter coordinates are held fixed. No completeness claim is
    attached to max_steps, parameter_bound, or any other termination status.
    """

    system: EquilibriumSystem
    parameter_name: str
    states: np.ndarray
    parameters: np.ndarray
    tangents: np.ndarray
    arclength: np.ndarray
    step_sizes: np.ndarray
    residual_norms: np.ndarray
    newton_iterations: np.ndarray
    augmented_condition_numbers: np.ndarray
    folds: tuple[EquilibriumFold, ...]
    attempts: tuple[ContinuationAttempt, ...]
    status: str
    message: str
    branch_id: str | None = None

    def __post_init__(self) -> None:
        for name in ("states", "parameters", "tangents", "arclength", "step_sizes",
                     "residual_norms", "newton_iterations", "augmented_condition_numbers"):
            object.__setattr__(self, name, _freeze(getattr(self, name)))
        object.__setattr__(self, "folds", tuple(self.folds))
        object.__setattr__(self, "attempts", tuple(self.attempts))

    def to_frame(self) -> pd.DataFrame:
        frame = pd.DataFrame(self.states.copy(), columns=["state:" + s for s in self.system.state_names])
        for j, name in enumerate(self.system.parameter_names):
            frame["parameter:" + name] = self.parameters[:, j]
        frame["arclength"] = self.arclength
        frame["step_size"] = self.step_sizes
        frame["residual_norm"] = self.residual_norms
        frame["newton_iterations"] = self.newton_iterations.astype(int)
        frame["augmented_condition_number"] = self.augmented_condition_numbers
        frame["tangent_parameter"] = self.tangents[:, -1]
        return frame

    def to_dataframe(self) -> pd.DataFrame:
        return self.to_frame()

    def summary(self) -> pd.DataFrame:
        return pd.DataFrame({
            "status": [self.status], "message": [self.message], "points": [len(self.states)],
            "simple_folds": [sum(f.status == "simple_fold" for f in self.folds)],
            "other_fold_candidates": [sum(f.status != "simple_fold" for f in self.folds)],
            "rejected_steps": [sum(not a.accepted for a in self.attempts)],
            "max_residual": [float(np.max(self.residual_norms))],
        })

    def attempts_frame(self) -> pd.DataFrame:
        return pd.DataFrame([vars(a) for a in self.attempts],
                            columns=list(ContinuationAttempt.__dataclass_fields__))

    def to_markdown(self, **kwargs: Any) -> str:
        return df_to_markdown(self.to_frame(), index=False, **kwargs)

    def to_latex(self, **kwargs: Any) -> str:
        return df_to_latex(self.to_frame(), index=False, **kwargs)

    def to_typst(self, **kwargs: Any) -> str:
        return df_to_typst(self.to_frame(), index=False, **kwargs)

    def plot(self, state: str | None = None, *, ax: Any = None) -> Any:
        """Plot a state against the continued parameter; never calls show."""
        import matplotlib.pyplot as plt

        state = self.system.state_names[0] if state is None else state
        i = self.system.state_names.index(state)
        j = self.system.parameter_names.index(self.parameter_name)
        if ax is None:
            _, ax = plt.subplots()
        ax.plot(self.parameters[:, j], self.states[:, i], label=self.branch_id or "traced equilibria")
        for status, marker in (("simple_fold", "o"), ("degenerate_or_unresolved", "x"), ("unresolved", "x")):
            selected = [f for f in self.folds if f.status == status]
            if selected:
                ax.scatter([f.parameters[j] for f in selected], [f.state[i] for f in selected],
                           marker=marker, label=status.replace("_", " "))
        ax.set(xlabel=self.parameter_name, ylabel=state)
        ax.legend()
        return ax


@dataclass(frozen=True)
class _Controls:
    residual_tol: float
    arclength_tol: float
    derivative_step: float
    derivative_tol: float
    rank_tol: float
    condition_limit: float
    max_states: int
    max_newton: int
    fold_tol: float
    fold_max_iter: int
    curvature_step: float
    degeneracy_tol: float


@dataclass
class _Point:
    y: np.ndarray
    linearization: EquilibriumLinearization
    jacobian: np.ndarray
    tangent: np.ndarray
    condition: float


class _StepFailure(Exception):
    def __init__(self, reason: str, iterations: int = 0, residual: float = float("nan"),
                 arc: float = float("nan")):
        super().__init__(reason)
        self.reason, self.iterations, self.residual, self.arc = reason, iterations, residual, arc


def _integer(value: int, name: str, minimum: int = 1) -> int:
    if isinstance(value, bool) or not isinstance(value, (int, np.integer)) or value < minimum:
        raise ValueError(f"{name} must be an integer >= {minimum}")
    return int(value)


class _Stepper:
    def __init__(self, system: EquilibriumSystem, parameters: np.ndarray, index: int,
                 controls: _Controls, bounds: tuple[float, float] | None,
                 signature: Any):
        self.system, self.fixed, self.index = system, parameters.copy(), index
        self.controls, self.bounds, self.signature = controls, bounds, signature
        self.scales = np.r_[system.state_scale, system.parameter_scale[index]]
        n, p = len(system.state_names), len(system.parameter_names)

        def theta(q: np.ndarray) -> np.ndarray:
            values = self.fixed.copy()
            values[index] = q[0]
            return values

        def state_derivative(x: np.ndarray, q: np.ndarray) -> np.ndarray:
            callback = system.state_jacobian
            assert callback is not None
            return callback(x, theta(q))

        def parameter_derivative(x: np.ndarray, q: np.ndarray) -> np.ndarray:
            callback = system.parameter_jacobian
            assert callback is not None
            return _array(callback(x, theta(q)), (n, p), "parameter_jacobian")[:, index:index+1]

        # Check only the selected parameter direction. Nonsmoothness in a
        # different, fixed parameter must not invalidate this local trace.
        self.reduced = EquilibriumSystem(
            lambda x, q: system.evaluate(x, theta(q)), system.state_names,
            (system.parameter_names[index],), state_scale=system.state_scale,
            parameter_scale=[system.parameter_scale[index]], residual_scale=system.residual_scale,
            state_jacobian=None if system.state_jacobian is None else state_derivative,
            parameter_jacobian=None if system.parameter_jacobian is None else parameter_derivative,
            regime=(None if system.regime is None else lambda x, q: _signature(system, x, theta(q))),
            equation_names=system.equation_names, label=system.label,
        )

    def unpack(self, y: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        physical = y * self.scales
        theta = self.fixed.copy()
        theta[self.index] = physical[-1]
        return physical[:-1], theta

    def check(self, y: np.ndarray) -> None:
        if not np.isfinite(y).all():
            raise _StepFailure("nonfinite_step")
        x, theta = self.unpack(y)
        if self.bounds is not None and not self.bounds[0] <= theta[self.index] <= self.bounds[1]:
            raise _StepFailure("parameter_bound")
        if _signature(self.system, x, theta) != self.signature:
            raise _StepFailure("regime_change")

    def residual(self, y: np.ndarray) -> np.ndarray:
        self.check(y)
        x, theta = self.unpack(y)
        return self.system.evaluate(x, theta) / self.system.residual_scale

    def point(self, y: np.ndarray, reference: np.ndarray | None = None) -> _Point:
        self.check(y)
        x, theta = self.unpack(y)
        c = self.controls
        lin = linearize_equilibrium(
            self.reduced, x, [theta[self.index]], step=c.derivative_step,
            residual_tol=c.residual_tol, derivative_tol=c.derivative_tol,
            singular_tol=c.rank_tol, condition_limit=c.condition_limit, max_states=c.max_states,
        )
        forbidden = [reason for reason in lin.reasons
                     if reason not in ("not_equilibrium", "singular", "ill_conditioned")]
        if forbidden:
            raise _StepFailure(forbidden[0], residual=lin.equilibrium_residual)
        jac = np.column_stack((lin.state_jacobian, lin.parameter_jacobian))
        jac = jac * self.scales[None, :] / self.system.residual_scale[:, None]
        _, values, vh = np.linalg.svd(jac, full_matrices=True)
        largest, smallest = float(values[0]), float(values[-1])
        if smallest <= c.rank_tol * max(1., largest):
            raise _StepFailure("rank_deficient", residual=lin.equilibrium_residual)
        if smallest <= np.sqrt(c.rank_tol) * max(1., largest):
            # A centered derivative of x**3 at zero is h**2, not zero.
            # Near rank loss, halve the stencil and require the smallest
            # singular value to exceed the observed derivative discrepancy.
            finer = linearize_equilibrium(
                self.reduced, x, [theta[self.index]], step=c.derivative_step / 2,
                residual_tol=c.residual_tol, derivative_tol=c.derivative_tol,
                singular_tol=c.rank_tol, condition_limit=c.condition_limit, max_states=c.max_states,
            )
            bad = [r for r in finer.reasons if r not in ("not_equilibrium", "singular", "ill_conditioned")]
            if bad:
                raise _StepFailure(bad[0], residual=finer.equilibrium_residual)
            refined = np.column_stack((finer.state_jacobian, finer.parameter_jacobian))
            refined = refined * self.scales[None, :] / self.system.residual_scale[:, None]
            uncertainty = float(np.linalg.norm(refined - jac))
            lin, jac = finer, refined
            _, values, vh = np.linalg.svd(jac, full_matrices=True)
            largest, smallest = float(values[0]), float(values[-1])
            if smallest <= c.rank_tol * max(1., largest):
                raise _StepFailure("rank_deficient", residual=lin.equilibrium_residual)
            if smallest <= uncertainty:
                raise _StepFailure("rank_unresolved", residual=lin.equilibrium_residual)
        condition = largest / smallest
        if condition > c.condition_limit:
            raise _StepFailure("ill_conditioned", residual=lin.equilibrium_residual)
        tangent = vh[-1].copy()
        if reference is not None and float(tangent @ reference) < 0:
            tangent *= -1
        return _Point(y.copy(), lin, jac, tangent, condition)

    def correct(self, predictor: np.ndarray, tangent: np.ndarray, radius: float) -> tuple[_Point, int, float]:
        c, y = self.controls, predictor.copy()
        arc_tol = min(c.arclength_tol, .01*radius)
        residual_norm, arc = float("nan"), float("nan")
        for iteration in range(c.max_newton + 1):
            try:
                point = self.point(y, tangent)
            except _StepFailure as exc:
                raise _StepFailure(exc.reason, iteration, exc.residual, arc) from exc
            f = point.linearization.residual / self.system.residual_scale
            residual_norm = float(np.max(np.abs(f)))
            arc = float(tangent @ (y - predictor))
            if residual_norm <= c.residual_tol and abs(arc) <= arc_tol:
                if float(point.tangent @ tangent) < .5:
                    raise _StepFailure("tangent_turn", iteration, residual_norm, arc)
                return point, iteration, arc
            if iteration == c.max_newton:
                raise _StepFailure("newton_failed", iteration, residual_norm, arc)
            bordered = np.vstack((point.jacobian, tangent))
            sv = np.linalg.svd(bordered, compute_uv=False)
            if sv[-1] <= c.rank_tol * max(1., sv[0]) or sv[0] / sv[-1] > c.condition_limit:
                raise _StepFailure("singular_corrector", iteration, residual_norm, arc)
            try:
                correction = np.linalg.solve(bordered, -np.r_[f, arc])
            except np.linalg.LinAlgError as exc:
                raise _StepFailure("linear_solve_failed", iteration, residual_norm, arc) from exc
            merit = max(residual_norm / c.residual_tol, abs(arc) / arc_tol)
            reason = "line_search_failed"
            for power in range(12):
                damping = 0.5**power
                trial = y + damping * correction
                if np.linalg.norm(trial - predictor) > radius:
                    reason = "large_correction"
                    continue
                try:
                    trial_f = self.residual(trial)
                except _StepFailure as exc:
                    reason = exc.reason
                    continue
                trial_arc = float(tangent @ (trial - predictor))
                trial_merit = max(float(np.max(np.abs(trial_f))) / c.residual_tol,
                                  abs(trial_arc) / arc_tol)
                if trial_merit <= 1 or trial_merit < merit * (1 - 1e-4*damping):
                    y = trial
                    break
            else:
                raise _StepFailure(reason, iteration + 1, residual_norm, arc)
        raise AssertionError("unreachable Newton exit")

    def fold(self, left: _Point, right: _Point, left_index: int, right_index: int) -> EquilibriumFold:
        c = self.controls
        best = left if abs(left.tangent[-1]) < abs(right.tangent[-1]) else right
        refined, reason, iterations = False, "refinement_limit", 0
        for iterations in range(c.fold_max_iter + 1):
            width = float(np.linalg.norm(right.y - left.y))
            if width <= c.fold_tol or best.tangent[-1] == 0:
                refined, reason = True, ""
                break
            if iterations == c.fold_max_iter:
                break
            secant = (right.y - left.y) / width
            try:
                middle, _, _ = self.correct((left.y + right.y) / 2, secant, .75*width)
            except _StepFailure as exc:
                reason = exc.reason
                break
            if middle.tangent @ left.tangent < 0:
                middle.tangent *= -1
            if abs(middle.tangent[-1]) < abs(best.tangent[-1]):
                best = middle
            if left.tangent[-1] * middle.tangent[-1] <= 0:
                right = middle
            else:
                left = middle
        a, b = best.jacobian[:, :-1], best.jacobian[:, -1]
        u, sv, vh = np.linalg.svd(a)
        cutoff = max(10*c.fold_tol, c.rank_tol) * max(1., float(sv[0]))
        corank = int(np.sum(sv <= cutoff))
        normal, direction = u[:, -1], vh[-1]
        transversality = float(normal @ b)
        curvature, curvature_error = float("nan"), float("nan")
        try:
            base = self.residual(best.y)
            estimates = []
            for h in (c.curvature_step, c.curvature_step / 2):
                displacement = np.r_[h * direction, 0.]
                second = (self.residual(best.y + displacement) - 2*base
                          + self.residual(best.y - displacement)) / h**2
                estimates.append(float(normal @ second))
            curvature = estimates[1]
            curvature_error = abs(estimates[0] - estimates[1]) / (1 + abs(curvature))
        except _StepFailure as exc:
            refined, reason = False, exc.reason
        simple = (refined and corank == 1 and abs(best.tangent[-1]) <= 10*c.fold_tol
                  and abs(transversality) > c.degeneracy_tol and abs(curvature) > c.degeneracy_tol
                  and curvature_error <= c.derivative_tol)
        status = "simple_fold" if simple else ("degenerate_or_unresolved" if refined else "unresolved")
        if refined and not simple:
            reason = "nondegeneracy_checks"
        x, theta = self.unpack(best.y)
        return EquilibriumFold(left_index, right_index, x, theta, best.tangent,
                               best.linearization.equilibrium_residual, corank, best.condition,
                               transversality, curvature, curvature_error, status, reason, iterations)


def continue_equilibrium(
    system: EquilibriumSystem, state: Any, parameters: Any, *,
    parameter: str | None = None, direction: int = 1, initial_tangent: Any = None,
    step: float = .05, min_step: float = 1e-5, max_step: float = .2,
    max_steps: int = 100, max_newton: int = 12, max_retries: int = 8,
    residual_tol: float = 1e-9, arclength_tol: float = 1e-9,
    derivative_step: float = 1e-5, derivative_tol: float = 1e-3,
    rank_tol: float = 1e-10, condition_limit: float = 1e10, max_states: int = 1500,
    fold_tol: float = 1e-7, fold_max_iter: int = 40,
    curvature_step: float = 1e-4, degeneracy_tol: float = 1e-5,
    parameter_bounds: tuple[float, float] | None = None,
    branch_id: str | None = None,
) -> EquilibriumBranchResult:
    """Trace a local equilibrium branch, allowing simple parameter folds.

    Coordinates are scaled by EquilibriumSystem's units. direction specifies
    the initial parameter direction unless initial_tangent is supplied; that
    tangent must be in scaled (state, selected parameter) coordinates. The
    latter is required when starting exactly at a fold.

    Bounds limit tracing and may terminate short of the boundary; derivative
    probes still require a two-sided mathematical neighborhood. A declared
    regime change is not crossed. Callback/domain errors propagate; numerical
    failures return only accepted points and retained attempt records. Fold
    classifications are local numerical evidence, not global certificates.
    """
    if not isinstance(system, EquilibriumSystem):
        raise TypeError("system must be an EquilibriumSystem")
    max_states = _integer(max_states, "max_states")
    n, p = len(system.state_names), len(system.parameter_names)
    if n > max_states:
        raise ValueError(f"dense continuation has {n} states, exceeding max_states={max_states}")
    max_steps = _integer(max_steps, "max_steps")
    max_newton = _integer(max_newton, "max_newton")
    max_retries = _integer(max_retries, "max_retries", 0)
    fold_max_iter = _integer(fold_max_iter, "fold_max_iter")
    step, min_step, max_step = (_positive(value, name) for value, name in
                                ((step, "step"), (min_step, "min_step"), (max_step, "max_step")))
    if not min_step <= step <= max_step:
        raise ValueError("require min_step <= step <= max_step")
    if isinstance(direction, (bool, np.bool_)) or not isinstance(direction, (int, np.integer)) or direction not in (-1, 1):
        raise ValueError("direction must be -1 or 1")
    if parameter is None:
        if p != 1:
            raise ValueError("select parameter by name when the system has multiple parameters")
        parameter = system.parameter_names[0]
    if parameter not in system.parameter_names:
        raise ValueError(f"unknown continuation parameter {parameter!r}")
    if branch_id is not None and (not isinstance(branch_id, str) or not branch_id):
        raise ValueError("branch_id must be a nonempty label or None")
    index = system.parameter_names.index(parameter)
    x, theta = _array(state, (n,), "state"), _array(parameters, (p,), "parameters")
    bounds = None
    if parameter_bounds is not None:
        raw = _array(parameter_bounds, (2,), "parameter_bounds")
        if raw[0] >= raw[1] or not raw[0] <= theta[index] <= raw[1]:
            raise ValueError("parameter_bounds must be increasing and contain the seed parameter")
        bounds = (float(raw[0]), float(raw[1]))
    c = _Controls(
        _positive(residual_tol, "residual_tol"), _positive(arclength_tol, "arclength_tol"),
        _positive(derivative_step, "derivative_step"), _positive(derivative_tol, "derivative_tol"),
        _positive(rank_tol, "rank_tol"), _positive(condition_limit, "condition_limit"),
        max_states, max_newton, _positive(fold_tol, "fold_tol"), fold_max_iter,
        _positive(curvature_step, "curvature_step"), _positive(degeneracy_tol, "degeneracy_tol"),
    )
    engine = _Stepper(system, theta, index, c, bounds, _signature(system, x, theta))
    y = np.r_[x, theta[index]] / engine.scales
    try:
        current = engine.point(y)
    except _StepFailure as exc:
        raise ValueError("ineligible continuation seed: " + exc.reason) from exc
    if current.linearization.equilibrium_residual > c.residual_tol:
        raise ValueError("ineligible continuation seed: not_equilibrium")
    if initial_tangent is None:
        if abs(current.tangent[-1]) <= c.rank_tol:
            raise ValueError("initial_tangent is required at a parameter turning point")
        current.tangent *= direction * np.sign(current.tangent[-1])
    else:
        reference = _array(initial_tangent, (n+1,), "initial_tangent")
        norm = float(np.linalg.norm(reference))
        if not np.isfinite(norm) or norm == 0:
            raise ValueError("initial_tangent must have a finite nonzero norm")
        reference /= norm
        if np.max(np.abs(current.jacobian @ reference)) > c.derivative_tol * (1 + np.linalg.norm(current.jacobian)):
            raise ValueError("initial_tangent must lie in the scaled augmented nullspace")
        if current.tangent @ reference < 0:
            current.tangent *= -1
    # Retain O(points * states) path data, not every dense Jacobian snapshot.
    path, tangents = [current.y.copy()], [current.tangent.copy()]
    residuals, conditions = [current.linearization.equilibrium_residual], [current.condition]
    sizes, iterations, lengths = [0.], [0], [0.]
    folds: list[EquilibriumFold] = []
    attempts: list[ContinuationAttempt] = []
    if abs(current.tangent[-1]) <= c.rank_tol:
        folds.append(engine.fold(current, current, 0, 0))
    status, message = "max_steps", "Requested continuation step budget reached."
    for _ in range(max_steps):
        accepted = False
        last_reason = "step_too_small"
        for _retry in range(max_retries + 1):
            if step < min_step:
                break
            try:
                predictor = current.y + step*current.tangent
                progress = float((predictor-current.y) @ current.tangent)
                if abs(progress/step - 1) > 1e-4:
                    raise _StepFailure("step_not_representable")
                candidate, count, arc = engine.correct(predictor,
                                                        current.tangent, .75*step)
                # An odd determinant reversal without a parameter-tangent
                # reversal warns of a possible branching singularity.
                old_sign = np.linalg.slogdet(current.jacobian[:, :-1])[0]
                new_sign = np.linalg.slogdet(candidate.jacobian[:, :-1])[0]
                if old_sign * new_sign < 0 and current.tangent[-1] * candidate.tangent[-1] > 0:
                    raise _StepFailure("possible_branch_point", count,
                                       candidate.linearization.equilibrium_residual, arc)
            except _StepFailure as exc:
                last_reason = exc.reason
                attempts.append(ContinuationAttempt(len(path)-1, step, False, exc.iterations,
                                                     exc.residual, exc.arc, exc.reason))
                step *= .5
                continue
            attempts.append(ContinuationAttempt(len(path)-1, step, True, count,
                                                 candidate.linearization.equilibrium_residual, arc, "accepted"))
            left_index = len(path)-1
            path.append(candidate.y.copy())
            tangents.append(candidate.tangent.copy())
            residuals.append(candidate.linearization.equilibrium_residual)
            conditions.append(candidate.condition)
            sizes.append(step)
            iterations.append(count)
            lengths.append(lengths[-1] + float(np.linalg.norm(candidate.y - current.y)))
            # Avoid duplicate events when an accepted point lands on the fold.
            if abs(current.tangent[-1]) > c.rank_tol:
                if current.tangent[-1] * candidate.tangent[-1] < 0 or abs(candidate.tangent[-1]) <= c.rank_tol:
                    folds.append(engine.fold(current, candidate, left_index, left_index+1))
            alignment = float(current.tangent @ candidate.tangent)
            if count <= 3 and alignment > .95:
                step = min(max_step, step * 1.3)
            elif count >= 7 or alignment < .8:
                step = max(min_step, step * .7)
            current, accepted = candidate, True
            break
        if not accepted:
            status, message = last_reason, "No acceptable next step within the step-size/retry limits."
            break
    physical = [engine.unpack(y) for y in path]
    return EquilibriumBranchResult(
        system, parameter, np.array([item[0] for item in physical]), np.array([item[1] for item in physical]),
        np.array(tangents), np.array(lengths), np.array(sizes), np.array(residuals),
        np.array(iterations), np.array(conditions),
        tuple(folds), tuple(attempts), status, message, branch_id,
    )
