"""Published one-asset HANK equilibrium of Auclert et al. (2021).

Endogenous hours, government bonds, Rotemberg price adjustment, and a Taylor
rule. The asset, labor, and Phillips-curve equations determine inflation,
wages, and output. Goods clearing is checked independently. This is distinct
from the reduced, fixed-labor model in :mod:`hank_sequence_space`.

Arrays use (asset, skill) order. ``r[t]`` is the realized return: unexpected
inflation at date zero revalues the predetermined nominal government debt.
The terminal household value and aggregate leads equal their steady states.
The implementation uses only NumPy/SciPy; the authors' code is used solely to
generate independent validation fixtures.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from math import comb
from typing import Any

import numpy as np
import pandas as pd
from scipy import sparse
from scipy.linalg import lu_factor, lu_solve
from scipy.optimize import root
from scipy.sparse.linalg import spsolve

from puremacro.reports import _df_to_latex, _df_to_markdown, _df_to_typst
from puremacro.vfi.discretize import rouwenhorst


@dataclass
class OneAssetHANKSteadyState:
    """Calibrated stationary equilibrium, including household policies."""

    calibration: dict[str, float]
    asset_grid: np.ndarray
    e_grid: np.ndarray
    transition: np.ndarray
    policy_a: np.ndarray
    policy_c: np.ndarray
    policy_n: np.ndarray
    marginal_value: np.ndarray
    distribution: np.ndarray
    aggregates: dict[str, float]
    diagnostics: dict[str, Any]
    converged: bool
    _jacobians: dict[int, dict[str, dict[str, np.ndarray]]] = field(default_factory=dict, repr=False)

    def summary(self) -> str:
        """Return a concise equilibrium and calibration report."""
        return (f"One-asset HANK: converged={self.converged}, "
                f"beta={self.calibration['beta']:.8f}, vphi={self.calibration['vphi']:.8f}\n"
                f"A={self.aggregates['A']:.8f}, C={self.aggregates['C']:.8f}, "
                f"NE={self.aggregates['NE']:.8f}")

    def to_dataframe(self) -> pd.DataFrame:
        """Return aggregate levels and calibrated preference parameters."""
        return pd.DataFrame({'value': dict(self.aggregates, beta=self.calibration['beta'],
                                          vphi=self.calibration['vphi'])})

    def to_latex(self, **kwargs: Any) -> str:
        return _df_to_latex(self.to_dataframe(), **kwargs)

    def to_markdown(self, **kwargs: Any) -> str:
        return _df_to_markdown(self.to_dataframe(), **kwargs)

    def to_typst(self, **kwargs: Any) -> str:
        return _df_to_typst(self.to_dataframe(), **kwargs)


@dataclass
class OneAssetHANKResult:
    """Equilibrium paths in level deviations, and explicit numerical checks.

    Inflation and interest rates are decimal rates, not percentages. Linear
    results clear the linearized equations; nonlinear results are checked
    against the original equilibrium equations and household accounting.
    """

    paths: pd.DataFrame
    steady_state: OneAssetHANKSteadyState
    nonlinear: bool
    converged: bool
    diagnostics: dict[str, Any]

    def summary(self) -> str:
        return (f"One-asset HANK ({'nonlinear' if self.nonlinear else 'linear'}): "
                f"converged={self.converged}, horizon={len(self.paths)}, "
                f"max market residual={self.diagnostics['market_residual_max_abs']:.3g}")

    def to_dataframe(self) -> pd.DataFrame:
        return self.paths.copy()

    def to_latex(self, **kwargs: Any) -> str:
        return _df_to_latex(self.paths, **kwargs)

    def to_markdown(self, **kwargs: Any) -> str:
        return _df_to_markdown(self.paths, **kwargs)

    def to_typst(self, **kwargs: Any) -> str:
        return _df_to_typst(self.paths, **kwargs)


def _interpolate(x: np.ndarray, xp: np.ndarray, yp: np.ndarray) -> np.ndarray:
    # Linear extrapolation matches EGM's endogenous grid convention. The
    # borrowing constraint is imposed by solving the constrained household.
    j = np.clip(np.searchsorted(xp, x, side='right') - 1, 0, len(xp) - 2)
    w = (x - xp[j]) / (xp[j + 1] - xp[j])
    return yp[j] + w * (yp[j + 1] - yp[j])


def _backward(value: np.ndarray, a: np.ndarray, e: np.ndarray, transition: np.ndarray,
              p: dict[str, float], r: float, w: float, transfer: float
              ) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    uc = p['beta'] * (value @ transition.T)
    if np.min(uc) <= 0 or r <= -1 or w <= 0:
        raise ValueError('household prices or marginal values are infeasible')
    we = w * e[None, :]
    ce = uc ** -p['eis']
    ne = (we * uc / p['vphi']) ** p['frisch']
    endog = (ce - we * ne + a[:, None] - transfer * e[None, :]) / (1 + r)
    if np.any(np.diff(endog, axis=0) <= 0):
        raise ValueError('endogenous asset grid is not increasing')
    c, n = np.empty_like(ce), np.empty_like(ne)
    for j in range(len(e)):
        c[:, j] = _interpolate(a, endog[:, j], ce[:, j])
        n[:, j] = _interpolate(a, endog[:, j], ne[:, j])
    cash = (1 + r) * a[:, None] + transfer * e[None, :]
    saving = cash + we * n - c
    constrained = saving < a[0]
    if np.any(constrained):
        wages = np.broadcast_to(we, saving.shape)[constrained]
        resources = cash[constrained] - a[0]
        # Monotone Newton solve in log marginal utility for the binding budget
        # and intratemporal FOC. It also works when EIS differs from Frisch.
        log_uc = np.log(uc[constrained])
        for _ in range(50):
            cc = np.exp(-p['eis'] * log_uc)
            nn = (wages / p['vphi']) ** p['frisch'] * np.exp(p['frisch'] * log_uc)
            error = cc - wages * nn - resources
            if np.max(np.abs(error)) < 1e-12:
                break
            log_uc += error / (p['eis'] * cc + p['frisch'] * wages * nn)
        else:
            raise RuntimeError('constrained household did not converge')
        c[constrained], n[constrained], saving[constrained] = cc, nn, a[0]
    if np.min(c) <= 0 or np.min(n) <= 0:
        raise ValueError('nonpositive household consumption or hours')
    return (1 + r) * c ** (-1 / p['eis']), saving, c, n


def _lottery(policy: np.ndarray, grid: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    lo = np.clip(np.searchsorted(grid, policy, side='right') - 1, 0, len(grid) - 2)
    hi_weight = (policy - grid[lo]) / (grid[lo + 1] - grid[lo])
    return lo, hi_weight


def _forward(d: np.ndarray, policy: np.ndarray, grid: np.ndarray, transition: np.ndarray) -> np.ndarray:
    lo, weight = _lottery(policy, grid)
    out = np.zeros_like(d)
    for j in range(d.shape[1]):
        out[:, j] = (np.bincount(lo[:, j], weights=(d * (1 - weight))[:, j], minlength=len(grid))
                     + np.bincount(lo[:, j] + 1, weights=(d * weight)[:, j], minlength=len(grid)))
    return out @ transition


def _transition(policy: np.ndarray, grid: np.ndarray, transition: np.ndarray) -> sparse.csr_matrix:
    lo, weight = _lottery(policy, grid)
    na, ne = policy.shape
    rows = np.broadcast_to(np.arange(na * ne).reshape(na, ne, 1), (na, ne, ne))
    cols = lo[:, :, None] * ne + np.arange(ne)[None, None, :]
    prob = np.broadcast_to(transition[None, :, :], (na, ne, ne))
    return sparse.coo_matrix((np.concatenate(((prob * (1 - weight[:, :, None])).ravel(),
                                              (prob * weight[:, :, None]).ravel())),
                              (np.tile(rows.ravel(), 2), np.concatenate((cols.ravel(), (cols + ne).ravel())))),
                             shape=(na * ne, na * ne)).tocsr()


def _household_steady(a: np.ndarray, e: np.ndarray, transition: np.ndarray,
                       p: dict[str, float], transfer: float):
    value = (1 + p['r']) * (.1 * ((1 + p['r']) * a[:, None]
                                  + (p['w'] + transfer) * e[None, :])) ** (-1 / p['eis'])
    old_a = np.full_like(value, np.inf)
    for iteration in range(20000):
        value, policy, c, n = _backward(value, a, e, transition, p, p['r'], p['w'], transfer)
        error = float(np.max(np.abs(policy - old_a)))
        if error < 1e-12:
            break
        old_a = policy
    else:
        raise RuntimeError('stationary household policies did not converge')
    mat = _transition(policy, a, transition)
    # Pin the mass at the borrowing-constrained low-skill state, then
    # normalize. A dense normalization row causes excessive sparse fill-in
    # when the income grid grows. Summing these equations gives d[0] = 1;
    # the remaining equations are exactly invariant-distribution equations.
    system = (sparse.eye(mat.shape[0]) - mat.T).tolil()
    system[0, 0] += 1
    rhs = np.zeros(mat.shape[0])
    rhs[0] = 1
    d = np.asarray(spsolve(system.tocsr(), rhs)).reshape(policy.shape)
    if np.all(np.isfinite(d)) and d.sum() > 0:
        d /= d.sum()
    if not np.all(np.isfinite(d)) or d.sum() <= 0 or np.min(d) < -1e-12:
        raise RuntimeError('invalid invariant distribution; increase the asset grid')
    # Remove only solver roundoff; preserve total mass, and check invariance.
    d = np.maximum(d, 0)
    d /= d.sum()
    distribution_error = float(np.max(np.abs(mat.T @ d.ravel() - d.ravel())))
    if distribution_error > 1e-11:
        raise RuntimeError('invariant distribution did not converge')
    return value, policy, c, n, d, dict(egm_iterations=iteration + 1,
                                      policy_fixed_point_error=error,
                                      distribution_error=distribution_error)


def calibrate_one_asset_hank(*, n_a: int = 500, n_e: int = 7, a_max: float = 150.,
                            r: float = .005, B: float = 5.6, eis: float = .5,
                            frisch: float = .5, rho_e: float = .966,
                            sigma_e: float = .5, mu: float = 1.2,
                            kappa: float = .1, phi: float = 1.5) -> OneAssetHANKSteadyState:
    """Fit beta and labor disutility to bonds ``B`` and effective labor one.

    Defaults reproduce the authors' ``notebooks/hank.ipynb`` calibration.
    ``sigma_e`` is the unconditional standard deviation of log skill. Output
    and productivity are one; skill is normalized to stationary mean one.
    Dividends and bond-interest taxes are proportional to skill. A numerical
    failure raises rather than returning an apparently calibrated equilibrium.
    """
    if isinstance(n_a, bool) or int(n_a) != n_a or n_a < 10:
        raise ValueError('n_a must be an integer >= 10')
    if isinstance(n_e, bool) or int(n_e) != n_e or n_e < 2:
        raise ValueError('n_e must be an integer >= 2')
    values = [a_max, r, B, eis, frisch, rho_e, sigma_e, mu, kappa, phi]
    if not np.all(np.isfinite(values)):
        raise ValueError('calibration parameters must be finite')
    if not (a_max > B > 0 and r > -1 and eis > 0 and frisch > 0
            and abs(rho_e) < 1 and sigma_e > 0 and mu > 1 and kappa > 0 and phi > 1):
        raise ValueError('invalid HANK calibration or non-active Taylor rule')
    a = np.geomspace(.25, a_max + .25, n_a) - .25
    log_e, transition = rouwenhorst(n_e, rho_e, sigma_e * np.sqrt(1 - rho_e ** 2))
    pi_e = np.array([comb(n_e - 1, j) for j in range(n_e)], dtype=float) / 2 ** (n_e - 1)
    e = np.exp(log_e)
    e /= pi_e @ e
    p = dict(r=r, rstar=r, B=B, eis=eis, frisch=frisch, rho_e=rho_e,
             sigma_e=sigma_e, mu=mu, kappa=kappa, phi=phi, Y=1., Z=1., w=1 / mu,
             beta=.986, vphi=.8)
    div, tax = 1 - p['w'], r * B

    def evaluate(x: np.ndarray):
        if not (0 < x[0] < 1 / (1 + r) and x[1] > 0):
            raise ValueError('calibration left the stationary preference domain')
        pp = dict(p, beta=float(x[0]), vphi=float(x[1]))
        h = _household_steady(a, e, transition, pp, div - tax)
        return np.array([np.sum(h[4] * h[1]) - B, np.sum(h[4] * h[3] * e[None, :]) - 1])

    fit = root(evaluate, [min(.98, .99 / (1 + r)), .8], tol=1e-10)
    p.update(beta=float(fit.x[0]), vphi=float(fit.x[1]))
    value, policy, c, n, d, diagnostics = _household_steady(a, e, transition, p, div - tax)
    aggregates = dict(A=float(np.sum(d * policy)), C=float(np.sum(d * c)),
                      NE=float(np.sum(d * n * e[None, :])), Div=div, Tax=tax, Y=1., pi=0.)
    residual = max(abs(aggregates['A'] - B), abs(aggregates['NE'] - 1), abs(aggregates['C'] - 1))
    diagnostics.update(market_residual_max_abs=residual, upper_grid_mass=float(d[-1].sum()),
                       calibration_evaluations=int(fit.nfev))
    if residual > 1e-8:
        raise RuntimeError(f'HANK calibration residual {residual:.3g}: {fit.message}')
    return OneAssetHANKSteadyState(p, a, e, transition, policy, c, n, value, d,
                                  aggregates, diagnostics, True)


def _household_jacobians(ss: OneAssetHANKSteadyState, horizon: int):
    if horizon in ss._jacobians:
        return ss._jacobians[horizon]
    p, a, e, pi, d = ss.calibration, ss.asset_grid, ss.e_grid, ss.transition, ss.distribution
    policies = dict(A=ss.policy_a, C=ss.policy_c, NE=ss.policy_n * e[None, :])
    mat = _transition(ss.policy_a, a, pi)
    expectations = {}
    for key, policy in policies.items():
        expected = np.empty((horizon - 1, d.size))
        v = policy.ravel().copy()
        for t in range(horizon - 1):
            v -= v.mean()
            expected[t] = v
            v = mat @ v
        expectations[key] = expected
    lo, _ = _lottery(ss.policy_a, a)
    span = a[lo + 1] - a[lo]
    base = dict(r=p['r'], w=p['w'], transfer=ss.aggregates['Div'] - ss.aggregates['Tax'])
    jac: dict[str, dict[str, np.ndarray]] = {k: {} for k in policies}
    h = 1e-5
    for inp in base:
        direct = {k: np.empty(horizon) for k in policies}
        dd = np.empty((horizon, d.size))
        dv = np.zeros_like(ss.marginal_value)
        for t in range(horizon):
            outputs = []
            for sign in (1, -1):
                prices = base.copy()
                if t == 0:
                    prices[inp] += sign * h
                v = ss.marginal_value if t == 0 else ss.marginal_value + sign * h * dv
                outputs.append(_backward(v, a, e, pi, p, **prices))
            dv, da, dc, dn = [(up - down) / (2 * h) for up, down in zip(*outputs)]
            for key, derivative in zip(policies, (da, dc, dn * e[None, :])):
                direct[key][t] = np.sum(d * derivative)
            # Differentiate lottery weights at fixed steady-state brackets.
            # Finite-differencing the entire distribution would spuriously
            # switch brackets near a grid node.
            amount = d * da / span
            moved = np.zeros_like(d)
            for j in range(len(e)):
                moved[:, j] = (np.bincount(lo[:, j] + 1, weights=amount[:, j], minlength=len(a))
                                - np.bincount(lo[:, j], weights=amount[:, j], minlength=len(a)))
            dd[t] = (moved @ pi).ravel()
        for key in policies:
            news = np.empty((horizon, horizon))
            news[0] = direct[key]
            news[1:] = expectations[key] @ dd.T
            for t in range(1, horizon):
                news[1:, t] += news[:-1, t - 1]
            jac[key][inp] = news
    ss._jacobians[horizon] = jac
    return jac


def _linear_system(ss: OneAssetHANKSteadyState, horizon: int):
    p = ss.calibration
    eye = np.eye(horizon)
    lag = np.eye(horizon, k=-1)
    lead = lag.T
    zero = np.zeros_like(eye)
    rr = np.hstack((p['phi'] * lag - (1 + p['r']) * eye, zero, zero))
    ww = np.hstack((zero, eye, zero))
    yy = np.hstack((zero, zero, eye))
    div = (1 - p['w']) * yy - ww
    transfer = div - p['B'] * rr
    j = _household_jacobians(ss, horizon)
    outcomes = {k: j[k]['r'] @ rr + j[k]['w'] @ ww + j[k]['transfer'] @ transfer for k in j}
    external = {k: (j[k]['r'] - p['B'] * j[k]['transfer']) @ lag for k in j}
    nkpc = np.hstack((-eye + lead / (1 + p['r']), p['kappa'] * eye, zero))
    system = np.vstack((nkpc, outcomes['A'], outcomes['NE'] - yy))
    forcing = np.vstack((zero, external['A'], external['NE']))
    return system, forcing, outcomes, external, rr, div


def _nonlinear_paths(ss: OneAssetHANKSteadyState, unknowns: np.ndarray, shock: np.ndarray):
    p, a, e = ss.calibration, ss.asset_grid, ss.e_grid
    horizon = len(shock)
    pi, w, y = unknowns.reshape(3, horizon) + np.array([0., p['w'], 1.])[:, None]
    if np.min(pi) <= -1 or np.min(w) <= 0 or np.min(y) <= 0:
        raise ValueError('infeasible aggregate path')
    r = (1 + np.r_[p['r'], p['r'] + shock[:-1]] + p['phi'] * np.r_[0., pi[:-1]]) / (1 + pi) - 1
    adjustment = p['mu'] / (p['mu'] - 1) / (2 * p['kappa']) * np.log1p(pi) ** 2 * y
    div = y - w * y - adjustment
    tax = p['B'] * r
    nkpc = (p['kappa'] * (w - 1 / p['mu'])
            + np.r_[y[1:], 1.] / y * np.log1p(np.r_[pi[1:], 0.]) / (1 + np.r_[r[1:], p['r']])
            - np.log1p(pi))
    pol_a, pol_c, pol_n = [np.empty((horizon,) + ss.policy_a.shape) for _ in range(3)]
    value = ss.marginal_value
    for t in range(horizon - 1, -1, -1):
        value, pol_a[t], pol_c[t], pol_n[t] = _backward(value, a, e, ss.transition, p, r[t], w[t], div[t] - tax[t])
    d = ss.distribution.copy()
    aa, cc, nn = np.empty(horizon), np.empty(horizon), np.empty(horizon)
    mass_error, negative_mass, top_mass, budget_error = 0., 0., 0., 0.
    for t in range(horizon):
        aa[t], cc[t], nn[t] = (np.sum(d * pol_a[t]), np.sum(d * pol_c[t]), np.sum(d * pol_n[t] * e[None, :]))
        mass_error = max(mass_error, abs(float(d.sum()) - 1))
        negative_mass = min(negative_mass, float(d.min()))
        top_mass = max(top_mass, float(d[-1].sum()))
        budget = (pol_c[t] + pol_a[t] - (1 + r[t]) * a[:, None]
                  - w[t] * e[None, :] * pol_n[t] - (div[t] - tax[t]) * e[None, :])
        budget_error = max(budget_error, float(np.max(np.abs(budget))))
        d = _forward(d, pol_a[t], a, ss.transition)
        mass_error = max(mass_error, abs(float(d.sum()) - 1))
        negative_mass = min(negative_mass, float(d.min()))
        top_mass = max(top_mass, float(d[-1].sum()))
    paths = dict(pi=pi, w=w - p['w'], Y=y - 1, C=cc - ss.aggregates['C'],
                 A=aa - ss.aggregates['A'], NE=nn - ss.aggregates['NE'], r=r - p['r'],
                 Div=div - ss.aggregates['Div'], Tax=tax - ss.aggregates['Tax'],
                 goods_mkt=y - cc - adjustment, asset_mkt=aa - p['B'], labor_mkt=nn - y, nkpc_res=nkpc)
    residual = np.concatenate((nkpc, aa - p['B'], nn - y))
    diagnostics = dict(mass_error=mass_error, minimum_mass=negative_mass, upper_grid_mass=top_mass,
                       household_budget_error=budget_error, minimum_consumption=float(pol_c.min()),
                       minimum_saving=float(pol_a.min()))
    return paths, residual, diagnostics


def solve_one_asset_hank(steady_state: OneAssetHANKSteadyState | None = None, *,
                         horizon: int = 300, magnitude: float = -.0025, rho: float = .61,
                         nonlinear: bool = False, tolerance: float = 1e-9,
                         max_iterations: int = 50) -> OneAssetHANKResult:
    """Solve a monetary-shock equilibrium with all three market equations.

    The Taylor-rule intercept changes by ``magnitude * rho**t``. Linear paths
    use household sequence Jacobians. Nonlinear paths use backward household
    policies, forward distributions, and a damped Newton iteration with the
    stationary Jacobian. Failed convergence is reported explicitly in the
    result. Horizon and grid convergence must be assessed for the use case.
    """
    if isinstance(horizon, bool) or int(horizon) != horizon or horizon < 2:
        raise ValueError('horizon must be an integer >= 2')
    if not np.all(np.isfinite([magnitude, rho, tolerance])) or not (abs(rho) < 1 and 0 < tolerance <= 1e-6):
        raise ValueError('finite shock, |rho| < 1, and 0 < tolerance <= 1e-6 required')
    if isinstance(max_iterations, bool) or int(max_iterations) != max_iterations or max_iterations < 1:
        raise ValueError('max_iterations must be a positive integer')
    ss = calibrate_one_asset_hank() if steady_state is None else steady_state
    if not ss.converged:
        raise ValueError('a converged steady state is required')
    shock = magnitude * rho ** np.arange(horizon)
    system, forcing, outcomes, external, rr, div = _linear_system(ss, horizon)
    factors = lu_factor(system)
    x = lu_solve(factors, -forcing @ shock)
    diagnostics: dict[str, Any] = {}
    iterations = 0
    if nonlinear:
        paths, residual, diagnostics = _nonlinear_paths(ss, x, shock)
        for iterations in range(max_iterations):
            if np.max(np.abs(residual)) <= tolerance:
                break
            dx = lu_solve(factors, -residual)
            current_error = float(np.max(np.abs(residual)))
            for step in 2. ** -np.arange(16):
                try:
                    trial, trial_residual, trial_diagnostics = _nonlinear_paths(ss, x + step * dx, shock)
                except (ValueError, RuntimeError, FloatingPointError):
                    continue
                if np.max(np.abs(trial_residual)) < current_error:
                    x += step * dx
                    paths, residual, diagnostics = trial, trial_residual, trial_diagnostics
                    break
            else:
                break
    else:
        pi, w, y = x.reshape(3, horizon)
        dr = rr @ x + np.r_[0., shock[:-1]]
        paths = {k: outcomes[k] @ x + external[k] @ shock for k in outcomes}
        paths.update(pi=pi, w=w, Y=y, r=dr, Div=div @ x, Tax=ss.calibration['B'] * dr)
        residual = system @ x + forcing @ shock
        paths.update(nkpc_res=residual[:horizon], asset_mkt=paths['A'],
                     labor_mkt=paths['NE'] - y, goods_mkt=y - paths['C'])
    market_error = max(float(np.max(np.abs(paths[k])))
                       for k in ('nkpc_res', 'asset_mkt', 'labor_mkt', 'goods_mkt'))
    diagnostics.update(market_residual_max_abs=market_error, iterations=iterations,
                       tolerance=tolerance, residual_kind='nonlinear' if nonlinear else 'linearized')
    feasible = (diagnostics.get('minimum_mass', 0) >= -1e-12
                and diagnostics.get('mass_error', 0) < 1e-10
                and diagnostics.get('household_budget_error', 0) < 1e-9)
    converged = bool(market_error <= max(1e-8, tolerance) and feasible)
    frame = pd.DataFrame(paths)
    frame.index.name = 'period'
    return OneAssetHANKResult(frame, ss, nonlinear, converged, diagnostics)
