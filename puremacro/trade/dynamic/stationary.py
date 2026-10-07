"""Exact national-price condensation of the sector-capital steady state.

Ported from the IO engine's ``dynamic_model/native_stationary.py``. This
module eliminates sector quantities algebraically at stationarity; it does not
aggregate any industry or replace its capital intensity. All original
equations and resource accounts certify the reconstructed equilibrium.

At stationarity ``R_j = PI_c(j) (1/beta - 1 + delta)``. Conditional on the
``2C`` unknowns ``u = (log w, log PI/PI0)`` the factor demands per unit of
output are known (NATIVE_NUMERICS.md):

.. code-block:: text

    v_j  = w_c**(1-alpha_j) (PI_c/PI0_c)**alpha_j
    hK_j = (K0_j/y0_j) (w_c/(PI_c/PI0_c))**(1-alpha_j)
    hL_j = (L0_j/y0_j) ((PI_c/PI0_c)/w_c)**alpha_j
    [ LC        LI          ] [ C     ]   [ L0 ]
    [ -delta KC  I-delta KI ] [ I_nat ] = [ 0  ]   (solved in deviations)

with ``KC, KI, LC, LI`` the national sums of ``hK``/``hL`` times the cached
Leontief responses. The nonlinear residual is ``(log PI_implied/PI_guess,
national budgets with one numeraire)`` with an exact dense ``2C x 2C``
Jacobian, solved by damped Newton with Armijo backtracking. The full
``N + 2C`` residual and the 17-entry certificate are re-evaluated before a
result is returned. The condensation is valid only when the conditional
quantity matrix is invertible.
"""
from __future__ import annotations

import time
from typing import Any, Callable

import numpy as np
from scipy import linalg

from ._results import DynamicSolveError, DynamicSteadyStateResult, EconomicDomainError, _freeze_state

# Internal module: users reach the condensation through
# solve_dynamic_steady_state(method="condensed"); tests import the names directly.
__all__: list[str] = []


def _national_columns(values: np.ndarray, countries: np.ndarray, count: int) -> np.ndarray:
    result = np.zeros((count, values.shape[1]), dtype=float)
    np.add.at(result, countries, values)
    return result


def _require_certificate(certificate: dict[str, float], tol: float) -> None:
    values = np.asarray(list(certificate.values()), dtype=float)
    if not np.isfinite(values).all() or float(np.max(np.abs(values), initial=0.0)) > max(1e-8, tol):
        worst = max(certificate, key=certificate.get)
        raise DynamicSolveError(
            f"Independent economic certificate failed: {worst}={certificate[worst]:.3g} "
            f"exceeds {max(1e-8, tol):.3g}", certificate=certificate)


class StationaryProblem:
    """The ``2C`` condensed stationary problem with exact batched derivatives."""

    def __init__(self, model: Any, policy: Any):
        self.model, self.policy = model, policy
        self.d = model.p
        self.network = model._network(policy)
        self.nc = model.nc
        self._u = self._residual = self._jacobian = None
        self.state = self.z = None

    def evaluate(self, u: np.ndarray) -> np.ndarray:
        """Condensed residual ``(log PI_implied/PI_guess, budgets with numeraire)`` at ``u`` (cached)."""
        u = np.asarray(u, dtype=float)
        if self._u is not None and np.array_equal(u, self._u):
            return self._residual
        d, m, nc = self.d, self.model, self.nc
        if u.shape != (2 * nc,) or not np.all(np.isfinite(u)) or np.max(np.abs(u)) > 200:
            raise EconomicDomainError("Invalid condensed stationary price vector")
        w = np.exp(u[:nc])
        PIguess = d.PI0 * np.exp(u[nc:])
        va = np.exp((1. - d.alpha) * u[:nc][d.country] + d.alpha * u[nc:][d.country])
        R = (1. / d.beta - 1. + d.delta) * PIguess[d.country]
        capital_log_scale = (1. - d.alpha) * (u[:nc][d.country] - u[nc:][d.country])
        labor_log_scale = d.alpha * (u[nc:][d.country] - u[:nc][d.country])
        hK = d.K0 / d.y0 * np.exp(capital_log_scale)
        hL = d.L0sector / d.y0 * np.exp(labor_log_scale)
        KC = _national_columns(hK[:, None] * m.YC, d.country, nc)
        KI = _national_columns(hK[:, None] * m.YI, d.country, nc)
        LC = _national_columns(hL[:, None] * m.YC, d.country, nc)
        LI = _national_columns(hL[:, None] * m.YI, d.country, nc)
        quantity_matrix = np.block([[LC, LI], [-d.delta * KC, np.eye(nc) - d.delta * KI]])
        quantity_factor = linalg.lu_factor(quantity_matrix, check_finite=False)
        labor_change = -np.bincount(d.country, weights=d.L0sector * np.expm1(labor_log_scale), minlength=nc)
        investment_change = np.bincount(d.country, weights=d.I0 * np.expm1(capital_log_scale), minlength=nc)
        changes = linalg.lu_solve(quantity_factor, np.r_[labor_change, investment_change], check_finite=False)
        C, investment = d.C0 + changes[:nc], m.Inational0 + changes[nc:]
        output_change = m.YC @ changes[:nc] + m.YI @ changes[nc:]
        y = d.y0 + output_change
        capital_log_change = np.log1p(output_change / d.y0) + capital_log_scale
        K = d.K0 * np.exp(capital_log_change)
        if (not np.all(np.isfinite(changes)) or np.any(C <= 0) or np.any(investment <= 0)
                or np.any(y <= 0) or np.any(K <= 0)):
            raise EconomicDomainError("Condensed stationary quantities leave the positive interior")
        p = self.network.solve(d.b * va)
        z = np.r_[u[:nc], np.log1p(changes[:nc] / d.C0), capital_log_change]
        state = m.state(z, z, self.policy, _price=p)
        budget = state["budget"] / d.budget_scale
        budget[m.omitted_budget_index] = u[m.numeraire_index]
        residual = np.r_[np.log(state["PI"] / PIguess), budget]
        if not np.all(np.isfinite(residual)):
            raise EconomicDomainError("Nonfinite condensed stationary residual")
        self._u, self._residual, self._jacobian = u.copy(), residual, None
        self.state, self.z = state, z
        self.w, self.PIguess, self.va, self.R = w, PIguess, va, R
        self.hK, self.hL, self.quantity_factor = hK, hL, quantity_factor
        return residual

    def jacobian(self, u: np.ndarray) -> np.ndarray:
        """Exact dense ``2C x 2C`` Jacobian of :meth:`evaluate` at ``u`` (cached)."""
        self.evaluate(u)
        if self._jacobian is not None:
            return self._jacobian
        d, m, s, nc = self.d, self.model, self.state, self.nc
        eye = np.eye(nc)
        wlog = np.c_[eye, np.zeros_like(eye)]
        ilog = np.c_[np.zeros_like(eye), eye]
        vlog = (1. - d.alpha[:, None]) * wlog[d.country] + d.alpha[:, None] * ilog[d.country]
        dhK = self.hK[:, None] * (vlog - ilog[d.country])
        dhL = self.hL[:, None] * (vlog - wlog[d.country])
        quantity_rhs = np.vstack((
            -_national_columns(dhL * s["y"][:, None], d.country, nc),
            d.delta * _national_columns(dhK * s["y"][:, None], d.country, nc)))
        dquantities = linalg.lu_solve(self.quantity_factor, quantity_rhs, check_finite=False)
        dC, dI = dquantities[:nc], dquantities[nc:]
        dy = m.YC @ dC + m.YI @ dI
        dK = dhK * s["y"][:, None] + self.hK[:, None] * dy
        dp = self.network.solve(d.b[:, None] * self.va[:, None] * vlog)
        iweights = d.omegaI * (1. + self.policy.investment_rates)
        dPI = iweights.T @ dp / (1. - d.tI)[:, None]
        price_derivative = dPI / s["PI"][:, None] - ilog
        # Final-use tariffs and final-use taxes are paid and rebated within a
        # country; they cancel exactly from the consolidated national budget.
        # Production taxes and intermediate tariffs remain in this derivative.
        basicC, basicI = s["p"] @ d.omegaC, s["p"] @ d.omegaI
        final_spending = ((d.omegaC.T @ dp) * s["C"][:, None] + basicC[:, None] * dC
                          + (d.omegaI.T @ dp) * s["Inational"][:, None] + basicI[:, None] * dI
                          + d.qOther.T @ dp)
        dw = self.w[:, None] * wlog
        dR = self.R[:, None] * ilog[d.country]
        factor_income = dw * d.L0[:, None] + _national_columns(
            dR * s["K"][:, None] + self.R[:, None] * dK, d.country, nc)
        production_taxes = _national_columns(
            d.tax[:, None] * (dp * s["y"][:, None] + s["p"][:, None] * dy), d.country, nc)
        tariff_unit = self.network.tariff_unit_cost(s["p"])
        dtariff_unit = self.network.tariff_unit_cost(dp)
        tariffs = _national_columns(dy * tariff_unit[:, None] + s["y"][:, None] * dtariff_unit,
                                    d.country, nc)
        transfer = d.XN0[:, None] * dw[-1][None, :]
        budget_derivative = (final_spending + transfer - factor_income - production_taxes - tariffs) / d.budget_scale[:, None]
        budget_derivative[m.omitted_budget_index] = wlog[m.numeraire_index]
        self._jacobian = np.vstack((price_derivative, budget_derivative))
        if not np.all(np.isfinite(self._jacobian)):
            raise EconomicDomainError("Nonfinite condensed stationary Jacobian")
        return self._jacobian


def _condensed_location(model: Any, residual: np.ndarray) -> dict[str, Any]:
    """Name the condensed equation with the largest residual (country and block)."""
    nc, countries = model.nc, model.country_codes
    k = int(np.argmax(np.abs(residual)))
    info: dict[str, Any] = {"method": "condensed", "residual": float(residual[k])}
    if k < nc:
        info.update(block="investment_price", country=countries[k],
                    equation=f"stationary investment-price consistency of {countries[k]}")
    elif k - nc == model.omitted_budget_index:
        info.update(block="numeraire", country=countries[k - nc], equation="wage numeraire")
    else:
        info.update(block="budget", country=countries[k - nc],
                    equation=f"household budget of {countries[k - nc]}")
    info["message"] = f"Largest condensed residual: {info['equation']} ({info['residual']:.3g})."
    return info


def solve_condensed_steady(model: Any, policy: Any = None, *, start: np.ndarray | None = None,
                           tol: float = 1e-9, max_iter: int = 40,
                           progress: Callable | None = None) -> DynamicSteadyStateResult:
    """Solve and certify the full steady state through exact elimination (internal).

    ``start`` may be a full retained vector (``n_vars``) or a condensed
    ``2C`` vector. Acceptance requires the condensed residual, the
    reconstructed full residual and every certificate entry at or below
    ``tol`` (certificate floor ``1e-8``); otherwise :class:`DynamicSolveError`
    (a line-search stall or the iteration limit carries ``location`` naming
    the condensed equation with the largest residual).
    """
    tic = time.perf_counter()
    policy = model.zero_policy() if policy is None else policy
    problem = StationaryProblem(model, policy)
    if start is None:
        u = np.zeros(2 * model.nc)
    else:
        start = np.asarray(start, dtype=float)
        if start.shape == (model.n_vars,):
            initial_state = model.state(start, start, policy)
            u = np.r_[start[model.w_slice], np.log(initial_state["PI"] / model.p.PI0)]
        elif start.shape == (2 * model.nc,):
            u = start.copy()
        else:
            raise ValueError("Condensed steady-state start has incompatible dimensions")
    if not np.isfinite(tol) or tol <= 0 or not isinstance(max_iter, (int, np.integer)) or max_iter < 1:
        raise ValueError("A finite positive tolerance and positive integer iteration limit are required")
    history: list[dict[str, Any]] = []
    trial = None
    for iteration in range(max_iter + 1):
        residual = problem.evaluate(u)
        error = float(np.max(np.abs(residual)))
        if error <= tol:
            break
        if iteration == max_iter:
            location = _condensed_location(model, residual)
            raise DynamicSolveError(f"Condensed stationary iteration limit: residual {error:.6g}. "
                                    f"{location['message']}", residual=error, iterate=u,
                                    location=location, history=history)
        jacobian = problem.jacobian(u)
        try:
            direction = linalg.solve(jacobian, -residual, check_finite=False)
        except linalg.LinAlgError as exc:
            raise DynamicSolveError("Singular condensed stationary Jacobian", residual=error,
                                    iterate=u, history=history) from exc
        step = min(1., .75 / max(.75, float(np.max(np.abs(direction)))))
        merit = float(residual @ residual)
        domain_rejections, first_domain_error = 0, None
        for _ in range(24):
            candidate = u + step * direction
            try:
                trial = problem.evaluate(candidate)
                accepted = float(trial @ trial) < merit * (1. - 1e-4 * step)
            except (EconomicDomainError, FloatingPointError, RuntimeError, linalg.LinAlgError) as exc:
                accepted = False
                domain_rejections += 1
                first_domain_error = str(exc) if first_domain_error is None else first_domain_error
            if accepted:
                break
            step *= .5
        else:
            # Re-evaluate at the accepted iterate so the location refers to it.
            residual = problem.evaluate(u)
            location = _condensed_location(model, residual)
            location.update(rejected_trials_domain=domain_rejections,
                            rejected_trials_merit=24 - domain_rejections,
                            first_domain_error=first_domain_error)
            raise DynamicSolveError(f"Condensed stationary line search failed at residual {error:.3g}. "
                                    f"{location['message']}", residual=error, iterate=u,
                                    location=location, history=history)
        item = dict(method="exact_stationary_condensation", iteration=iteration + 1,
                    residual_before=error, residual=float(np.max(np.abs(trial))), step=step,
                    nonlinear_unknowns=2 * model.nc, native_capital_stocks=model.n)
        history.append(item)
        if progress is not None:
            progress(item)
        u = candidate
    problem.evaluate(u)
    z, state = problem.z, problem.state
    full_residual = model.residual_from_states(state, state)
    full_error = float(np.max(np.abs(full_residual)))
    if not np.isfinite(full_error) or full_error > tol:
        raise DynamicSolveError(f"Reconstructed original stationary equations fail: {full_error:.6g}",
                                residual=full_error, iterate=z, history=history)
    certificate = model.certificate(z, z, z, policy, policy, _state=state, _following=state)
    _require_certificate(certificate, tol)
    z = np.array(z, dtype=float, copy=True)
    z.flags.writeable = False
    return DynamicSteadyStateResult(
        z=z, converged=True, max_residual=full_error, iterations=iteration,
        seconds=time.perf_counter() - tic, method="condensed", history=tuple(history),
        certificate=certificate, state=_freeze_state(state), policy=policy, country_codes=model.country_codes,
        cell_labels=model.cell_labels, benchmark=model.benchmark(), tol=float(tol),
        metadata={"nonlinear_unknowns": 2 * model.nc, "capital_stocks": model.n,
                  "network_backend": problem.network.backend})
