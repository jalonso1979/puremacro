"""Experimental joint sourcing, capital and sector-labor trade transitions.

Two countries and two sectors, competitive production, national financial
autarky and pooled household income. All adjustment costs use real final goods.
The finite perfect-foresight boundary problem has an explicit approximate
terminal value; a solved path does not establish infinite-horizon accuracy.
"""
from __future__ import annotations

from dataclasses import dataclass, field
import hashlib
import json
from typing import Any, Callable, Mapping

import numpy as np
import pandas as pd

from puremacro._numeric_contracts import freeze as _freeze_metadata, thaw as _thaw, readonly_array, real_scalar
from scipy import sparse
from scipy.linalg import ordqz, solve
from scipy.optimize import least_squares

from puremacro.reports import _df_to_latex, _df_to_markdown, _df_to_typst
from .dynamic.welfare import consumption_equivalent_welfare

__all__ = ["ReallocationEconomy", "ReallocationSteadyState", "ReallocationTransitionResult",
           "ReallocationHorizonResult", "ReallocationSolveError", "audit_reallocation",
           "solve_reallocation_steady_state", "solve_reallocation_transition",
           "run_reallocation_horizon_ladder", "reallocation_stability"]

_FOREIGN = (np.arange(2)[:, None] != np.arange(2)[None, :])[:, None, :]
_SIZE = 16
_BOUND = 30.


def _array(value: Any, name: str, shape: tuple[int, ...] | None = None) -> np.ndarray:
    if np.iscomplexobj(value):
        raise ValueError(f"{name} must be real")
    try:
        out = np.array(value, dtype=float, copy=True)
    except (TypeError, ValueError) as exc:
        raise ValueError(f"{name} must be a finite numeric array") from exc
    if not np.isfinite(out).all() or shape is not None and out.shape != shape:
        raise ValueError(f"{name} must be finite with shape {shape}")
    return readonly_array(out)


def _number(value: Any, name: str) -> float:
    return real_scalar(value, name)


def _integer(value: Any, name: str, minimum: int = 1) -> int:
    if isinstance(value, (bool, np.bool_)) or not isinstance(value, (int, np.integer)) or value < minimum:
        raise ValueError(f"{name} must be an integer >= {minimum}")
    return int(value)


@dataclass(frozen=True)
class ReallocationEconomy:
    """Illustrative two-country/two-sector primitives, with owned array copies.

    Array axes are (country, sector), except labor_supply (country,).
    Home shares are positive CES weights, not fixed realized purchase shares.
    Labor and sourcing costs are in national final goods per worker; capital
    costs are in final goods per installed capital. Investment is reversible.
    All parameters are assumptions; the defaults are not empirical estimates.
    """
    countries: tuple[str, str] = ("AAA", "BBB")
    sectors: tuple[str, str] = ("S01", "S02")
    beta: float = .96
    delta: float = .08
    capital_share: float = .3
    risk_aversion: float = 2.
    trade_elasticity: float = 4.
    capital_cost: float = 4.
    labor_cost: float = 5.
    sourcing_cost: float = .1
    productivity: np.ndarray = field(default_factory=lambda: np.ones((2, 2)))
    sector_weights: np.ndarray = field(default_factory=lambda: np.full((2, 2), .5))
    home_shares: np.ndarray = field(default_factory=lambda: np.full((2, 2), .75))
    labor_supply: np.ndarray = field(default_factory=lambda: np.ones(2))

    def __post_init__(self) -> None:
        for name in ("countries", "sectors"):
            labels = getattr(self, name)
            if (not isinstance(labels, (tuple, list)) or len(labels) != 2
                    or any(not isinstance(v, str) or not v.strip() for v in labels)
                    or len(set(labels)) != 2):
                raise ValueError(f"{name} must contain two distinct nonempty labels")
            object.__setattr__(self, name, tuple(labels))
        for name in ("beta", "delta", "capital_share", "risk_aversion", "trade_elasticity",
                     "capital_cost", "labor_cost", "sourcing_cost"):
            value = _number(getattr(self, name), name)
            if (name in ("beta", "delta", "capital_share") and not 0 < value < 1
                    or name in ("risk_aversion", "trade_elasticity") and value <= 0
                    or name.endswith("_cost") and value < 0):
                raise ValueError(f"invalid {name}")
            object.__setattr__(self, name, value)
        for name in ("productivity", "sector_weights", "home_shares", "labor_supply"):
            array = _array(getattr(self, name), name, (2,) if name == "labor_supply" else (2, 2))
            if np.min(array) <= 0:
                raise ValueError(f"{name} must be positive")
            if name == "sector_weights" and not np.allclose(array.sum(axis=1), 1., atol=1e-12, rtol=0):
                raise ValueError("sector_weights must sum to one in each country")
            if name == "home_shares" and np.max(array) >= 1:
                raise ValueError("home_shares must be strictly between zero and one")
            object.__setattr__(self, name, array)

    @property
    def fingerprint(self) -> str:
        data = {name: value.tolist() if isinstance(value, np.ndarray) else value
                for name, value in vars(self).items()}
        return hashlib.sha256(json.dumps(data, sort_keys=True, allow_nan=False).encode()).hexdigest()


class ReallocationSolveError(RuntimeError):
    """An unaccepted numerical attempt, with diagnostics and final coordinates."""
    def __init__(self, message: str, diagnostics: Mapping[str, Any], coordinates: np.ndarray | None = None):
        super().__init__(message)
        self.diagnostics = _freeze_metadata(diagnostics)
        self.coordinates = None if coordinates is None else _array(coordinates, "failed coordinates")


class _Report:
    def to_frame(self) -> pd.DataFrame:
        raise NotImplementedError

    def to_markdown(self) -> str:
        return _df_to_markdown(self.to_frame())

    def to_latex(self) -> str:
        return _df_to_latex(self.to_frame())

    def to_typst(self) -> str:
        return _df_to_typst(self.to_frame())


def _freeze(values: Mapping[str, np.ndarray]) -> Mapping[str, np.ndarray]:
    return _freeze_metadata({name: _array(value, name) for name, value in values.items()})


@dataclass(frozen=True)
class ReallocationSteadyState(_Report):
    economy: ReallocationEconomy
    tariffs: np.ndarray
    coordinates: np.ndarray
    values: Mapping[str, np.ndarray]
    diagnostics: Mapping[str, Any]

    def __post_init__(self) -> None:
        object.__setattr__(self, "tariffs", _array(self.tariffs, "tariffs", (2, 2)))
        object.__setattr__(self, "coordinates", _array(self.coordinates, "coordinates", (_SIZE,)))
        object.__setattr__(self, "values", _freeze(self.values))
        object.__setattr__(self, "diagnostics", _freeze_metadata(self.diagnostics))

    @property
    def converged(self) -> bool:
        return bool(self.diagnostics.get("accepted", False))

    def to_frame(self) -> pd.DataFrame:
        return _country_frame(self.economy, {k: v[None] for k, v in self.values.items()}).droplevel("date")


@dataclass(frozen=True)
class ReallocationTransitionResult(_Report):
    economy: ReallocationEconomy
    initial: ReallocationSteadyState
    terminal: ReallocationSteadyState
    tariffs: np.ndarray
    coordinates: np.ndarray
    paths: Mapping[str, np.ndarray]
    diagnostics: Mapping[str, Any]

    def __post_init__(self) -> None:
        object.__setattr__(self, "tariffs", _array(self.tariffs, "tariffs"))
        object.__setattr__(self, "coordinates", _array(self.coordinates, "coordinates"))
        object.__setattr__(self, "paths", _freeze(self.paths))
        object.__setattr__(self, "diagnostics", _freeze_metadata(self.diagnostics))

    @property
    def converged(self) -> bool:
        """Finite equations and accounting only; inspect the horizon ladder."""
        return bool(self.diagnostics.get("accepted", False))

    @property
    def capital_path(self) -> np.ndarray:
        return _array(np.concatenate([self.initial.values["capital"][None], self.paths["capital_next"]]), "capital path")

    @property
    def welfare(self) -> pd.DataFrame:
        return _welfare(self)

    def to_frame(self) -> pd.DataFrame:
        return _country_frame(self.economy, self.paths)

    def sector_frame(self) -> pd.DataFrame:
        T = len(self.tariffs)
        index = pd.MultiIndex.from_product([range(T), self.economy.countries, self.economy.sectors],
                                          names=["date", "country", "sector"])
        names = ("producer_prices", "output", "capital", "capital_next", "investment", "labor", "wages",
                 "rentals", "supplier_log_ratio", "capital_costs", "sourcing_costs", "import_shares")
        frame = pd.DataFrame({name: self.paths[name].ravel() for name in names}, index=index)
        frame["real_wage"] = (self.paths["wages"]/self.paths["price_index"][..., None]).ravel()
        frame["labor_income"] = (self.paths["wages"]*self.paths["labor"]).ravel()
        return frame


def _country_frame(model: ReallocationEconomy, paths: Mapping[str, np.ndarray]) -> pd.DataFrame:
    index = pd.MultiIndex.from_product([range(len(paths["consumption"])), model.countries], names=["date", "country"])
    frame = pd.DataFrame({name: paths[name].ravel() for name in
                          ("consumption", "price_index", "gross_absorption", "tariff_revenue", "rebates", "labor_costs")}, index=index)
    for name in ("investment", "capital_costs", "sourcing_costs"):
        frame[name] = paths[name].sum(axis=-1).ravel()
    frame["real_gdp"] = (np.sum(paths["producer_prices"]*paths["output"], axis=-1)/paths["price_index"]).ravel()
    return frame


def _recipes(model: ReallocationEconomy, r: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    alpha = np.where(_FOREIGN, 1-model.home_shares[..., None], model.home_shares[..., None])
    rho = (model.trade_elasticity-1)/model.trade_elasticity
    if rho == 0:
        normalizer = (1-model.home_shares)*r
        foreign_weight = np.broadcast_to(1-model.home_shares, r.shape)
    else:
        if abs(rho) < 1e-5:
            log_sum = np.log1p((1-model.home_shares)*np.expm1(rho*r))
        else:
            terms = np.log(alpha)+rho*r[..., None]*_FOREIGN
            shift = terms.real.max(axis=-1)
            log_sum = shift+np.log(np.exp(terms-shift[..., None]).sum(axis=-1))
        normalizer = log_sum/rho
        foreign_weight = np.exp(np.log(1-model.home_shares)+rho*r-log_sum)
    q = alpha*np.exp(r[..., None]*_FOREIGN-normalizer[..., None])
    return q, q*(_FOREIGN-foreign_weight[..., None])


def _state(model: ReallocationEconomy, z: np.ndarray, tariffs: np.ndarray,
           initial: Mapping[str, np.ndarray] | None = None) -> dict[str, np.ndarray]:
    T = len(z)
    p = np.exp(z[:, :4]).reshape(T, 2, 2)
    C, Knext = np.exp(z[:, 4:6]), np.exp(z[:, 6:10]).reshape(T, 2, 2)
    share = 1/(1+np.exp(-z[:, 10:12]))
    L = model.labor_supply[None, :, None]*np.stack([share, 1-share], axis=-1)
    r = z[:, 12:16].reshape(T, 2, 2)
    K = Knext if initial is None else np.concatenate([initial["capital"][None], Knext[:-1]])
    Lprev = L if initial is None else np.concatenate([initial["labor"][None], L[:-1]])
    rprev = r if initial is None else np.concatenate([initial["supplier_log_ratio"][None], r[:-1]])
    g = Knext/K-1
    I = Knext-(1-model.delta)*K
    costK = model.capital_cost/2*K*g**2
    costL = model.labor_cost/2*np.sum((L-Lprev)**2, axis=-1)/model.labor_supply
    costQ = model.sourcing_cost/2*model.labor_supply[:, None]*model.sector_weights*(r-rprev)**2
    A = C+np.sum(I+costK+costQ, axis=-1)+costL
    Y = model.productivity*K**model.capital_share*L**(1-model.capital_share)
    wage, rental = (1-model.capital_share)*p*Y/L, model.capital_share*p*Y/K
    q, dq = _recipes(model, r)
    origin_prices = p.transpose(0, 2, 1)[:, None, :, :]
    delivered = origin_prices*(1+tariffs[..., None]*_FOREIGN)
    Ps = np.sum(delivered*q, axis=-1)
    Pc = np.exp(np.sum(model.sector_weights*np.log(Ps), axis=-1))
    X = model.sector_weights*Pc[..., None]*A[..., None]/Ps
    bilateral = X[..., None]*q
    revenue = np.sum(tariffs[..., None]*_FOREIGN*origin_prices*bilateral, axis=(2, 3))
    imports = np.sum(delivered*q*_FOREIGN, axis=-1)/Ps
    return dict(producer_prices=p, consumption=C, capital=K, capital_next=Knext, labor=L,
                supplier_log_ratio=r, output=Y, wages=wage, rentals=rental, investment=I,
                capital_costs=costK, labor_costs=costL, sourcing_costs=costQ, gross_absorption=A,
                recipes=q, sector_absorption=X, bilateral_quantities=bilateral,
                sector_prices=Ps, price_index=Pc, tariff_revenue=revenue, rebates=revenue.copy(),
                import_shares=imports, capital_q=1+model.capital_cost*g,
                _price_derivative=np.sum(delivered*dq, axis=-1)/Ps)


def _scales(model: ReallocationEconomy, v: Mapping[str, np.ndarray]) -> dict[str, np.ndarray]:
    return dict(goods=v["output"], budget=v["price_index"]*v["gross_absorption"],
                labor=np.mean(v["wages"], axis=-1)/v["price_index"], source=v["gross_absorption"]/model.labor_supply)


def _equations(model: ReallocationEconomy, flat: np.ndarray, tariffs: np.ndarray,
               scales: Mapping[str, np.ndarray], initial: Mapping[str, np.ndarray] | None = None,
               terminal: Mapping[str, np.ndarray] | None = None) -> np.ndarray:
    z = flat.reshape(-1, _SIZE)
    v = _state(model, z, tariffs, initial)
    C, Pc, L, r = (v[k] for k in ("consumption", "price_index", "labor", "supplier_log_ratio"))
    labor = (v["wages"][:, :, 0]-v["wages"][:, :, 1])/Pc
    source = v["gross_absorption"][..., None]/model.labor_supply[:, None]*v["_price_derivative"]
    if initial is None:
        capital = 1-model.beta*(v["rentals"]/Pc[..., None]+1-model.delta)
    else:
        assert terminal is not None
        m = model.beta*(C[1:]/C[:-1])**(-model.risk_aversion)
        g = v["capital_next"]/v["capital"]-1
        capital = np.empty_like(g)
        capital[:-1] = v["capital_q"][:-1]-m[..., None]*(v["rentals"][1:]/Pc[1:, :, None]
            +1-model.delta+model.capital_cost*(g[1:]+g[1:]**2/2))
        capital[-1] = v["capital_q"][-1]-(terminal["consumption"]/C[-1])[:, None]**(-model.risk_aversion)
        dl = (L[:, :, 0]-np.concatenate([initial["labor"][None, :, 0], L[:-1, :, 0]]))/model.labor_supply
        dr = r-np.concatenate([initial["supplier_log_ratio"][None], r[:-1]])
        labor -= 2*model.labor_cost*dl
        labor[:-1] += m*2*model.labor_cost*dl[1:]
        source += model.sourcing_cost*dr
        source[:-1] -= m[..., None]*model.sourcing_cost*dr[1:]
    goods = v["output"]-v["bilateral_quantities"].sum(axis=1).transpose(0, 2, 1)
    budgets = Pc*v["gross_absorption"]-np.sum(v["wages"]*L+v["rentals"]*v["capital"], axis=-1)-v["rebates"]
    residual = np.empty_like(z)
    residual[:, :4] = (goods/scales["goods"]).reshape(-1, 4)
    residual[:, 4:6] = budgets/scales["budget"]
    residual[:, 5] = z[:, 3]  # Redundant BBB budget, independently audited.
    residual[:, 6:10] = capital.reshape(-1, 4)
    residual[:, 10:12] = labor/scales["labor"]
    residual[:, 12:16] = (source/scales["source"][:, None]).reshape(-1, 4)
    return residual.ravel()


def _jacobian(fun: Callable, horizon: int) -> Callable:
    """48 disjoint complex-step colors for three adjacent 16-column blocks."""
    bands = sparse.diags([np.ones(horizon-1), np.ones(horizon), np.ones(horizon-1)], [-1, 0, 1],
                         shape=(horizon, horizon), format="csr")
    pattern = sparse.kron(bands, np.ones((_SIZE, _SIZE)), format="csr")
    rows = np.repeat(np.arange(horizon*_SIZE), np.diff(pattern.indptr))
    groups = []
    for phase in range(min(horizon, 3)):
        for j in range(_SIZE):
            columns = np.arange(phase, horizon, 3)*_SIZE+j
            locations = np.flatnonzero((pattern.indices % _SIZE == j) & ((pattern.indices//_SIZE) % 3 == phase))
            groups.append((columns, locations, rows[locations]))

    def jac(x):
        matrix = pattern.copy()
        for columns, locations, affected_rows in groups:
            direction = np.array(x, dtype=complex)
            direction[columns] += 1e-25j
            matrix.data[locations] = np.imag(fun(direction))[affected_rows]/1e-25
        return matrix
    return jac


def _guess(model: ReallocationEconomy) -> np.ndarray:
    a, rental = model.capital_share, 1/model.beta-1+model.delta
    productivity = np.exp(np.mean(np.log(model.productivity)))
    wage = (1-a)*(productivity*(a/rental)**a)**(1/(1-a))
    output = wage*model.labor_supply/(1-a)
    capital = a*output[:, None]*model.sector_weights/rental
    consumption = output-model.delta*capital.sum(axis=-1)
    return np.r_[np.zeros(4), np.log(consumption), np.log(capital).ravel(),
                 np.log(model.sector_weights[:, 0]/model.sector_weights[:, 1]), np.zeros(4)]


def _options(tolerance: float, max_evaluations: int) -> tuple[float, int]:
    tol = _number(tolerance, "tolerance")
    if not 0 < tol <= 1e-5:
        raise ValueError("tolerance must lie in (0, 1e-5]")
    return tol, _integer(max_evaluations, "max_evaluations")


def _optimize(fun: Callable, start: np.ndarray, tolerance: float, max_evaluations: int) -> tuple[np.ndarray, dict]:
    if not np.isfinite(start).all() or np.max(abs(start)) >= _BOUND:
        raise ValueError("initial coordinates exceed the numerical domain (-30, 30)")
    T = start.size//_SIZE
    try:
        with np.errstate(over="raise", invalid="raise", divide="raise"):
            solution = least_squares(fun, start.ravel(), jac=_jacobian(fun, T), bounds=(-_BOUND, _BOUND),
                x_scale="jac", tr_solver="lsmr", tr_options=dict(atol=1e-12, btol=1e-12),
                ftol=1e-13, xtol=1e-13, gtol=min(1e-11, tolerance*.01), max_nfev=max_evaluations)
    except (ValueError, FloatingPointError, OverflowError, np.linalg.LinAlgError) as exc:
        raise ReallocationSolveError(f"trade solver numerical failure: {exc}",
            dict(accepted=False, stage="numerical_solve", error_type=type(exc).__name__,
                 error=str(exc), max_evaluations=max_evaluations), start) from exc
    norm = float(np.max(abs(fun(solution.x))))
    near_bound = bool(np.max(abs(solution.x)) > _BOUND-.01)
    report = dict(accepted=bool(solution.success and np.isfinite(norm) and norm <= tolerance and not near_bound),
                  solver_success=bool(solution.success), solver_message=str(solution.message),
                  solver_status=int(solution.status), evaluations=int(solution.nfev),
                  residual_max_abs=norm, tolerance=tolerance, numerical_bound_active=near_bound,
                  jacobian="block-tridiagonal complex-step", max_evaluations=max_evaluations)
    if not report["accepted"]:
        raise ReallocationSolveError(f"trade reallocation solve not accepted: {solution.message}; residual={norm:.3g}", report, solution.x)
    return solution.x.reshape(T, _SIZE), report


def _tariffs(value: Any, *, path: bool) -> np.ndarray:
    out = _array(value, "tariffs")
    if (out.ndim != (3 if path else 2) or out.shape[-2:] != (2, 2)
            or path and not 2 <= len(out) <= 2000 or np.any(out < 0)):
        raise ValueError("tariffs require nonnegative (time, country, sector) or (country, sector) rates; horizon 2..2000")
    return out


def audit_reallocation(result: ReallocationSteadyState | ReallocationTransitionResult,
                       *, tolerance: float = 1e-8) -> dict[str, Any]:
    """Reconstruct quantities, factor/budget/fiscal accounts and each FOC.

    This uses the retained economic arrays, not the solver's residual vector.
    The omitted national budget and both external balances are included.
    """
    from ._reallocation_audit import audit
    return audit(result, tolerance=_options(tolerance, 1)[0])


def solve_reallocation_steady_state(economy: ReallocationEconomy, tariffs: Any = None, *,
        tolerance: float = 1e-8, max_evaluations: int = 200,
        start: np.ndarray | None = None) -> ReallocationSteadyState:
    """Solve a stationary competitive equilibrium and independently audit it."""
    tol, budget = _options(tolerance, max_evaluations)
    rates = _tariffs(np.zeros((2, 2)) if tariffs is None else tariffs, path=False)
    z0 = _guess(economy) if start is None else _array(start, "start", (_SIZE,))
    reference = {k: v[0] for k, v in _state(economy, _guess(economy)[None], np.zeros((1, 2, 2))).items()}
    scales = _scales(economy, reference)
    fun = lambda z: _equations(economy, z, rates[None], scales)
    coordinates, report = _optimize(fun, z0, tol, budget)
    values = {k: v[0] for k, v in _state(economy, coordinates, rates[None]).items() if not k.startswith("_")}
    provisional = ReallocationSteadyState(economy, rates, coordinates[0], values, report)
    certificate = audit_reallocation(provisional, tolerance=tol)
    report.update(audit_max_abs=certificate["max_error"], audit_checks=certificate["checks"],
                  accepted=certificate["accepted"], scope="local stationary equations and accounting")
    if not certificate["accepted"]:
        raise ReallocationSolveError("stationary trade accounting or domain audit failed", report, coordinates[0])
    return ReallocationSteadyState(economy, rates, coordinates[0], values, report)


def _boundary(economy: ReallocationEconomy, value: ReallocationSteadyState, name: str, tol: float) -> None:
    if not isinstance(value, ReallocationSteadyState) or value.economy.fingerprint != economy.fingerprint:
        raise ValueError(f"{name} must use the same economy")
    if not value.converged or not audit_reallocation(value, tolerance=tol)["accepted"]:
        raise ValueError(f"{name} must pass the stationary audit")


def reallocation_stability(steady: ReallocationSteadyState) -> dict[str, Any]:
    """Local stable graph for the stationary second-order equation pencil.

    A dz[t-1]+B dz[t]+C dz[t+1]=0 gives a 32-dimensional companion
    pencil. A valid graph requires 16 stable roots, including algebraic zero
    modes for variables without lags. Inspect the lag rank for the number of
    economically active inherited directions. This is a local test, not a
    global equilibrium uniqueness result or a nonlinear tail certificate.
    """
    _boundary(steady.economy, steady, "steady state", 1e-8)
    model, n = steady.economy, _SIZE
    tariffs = np.broadcast_to(steady.tariffs, (3, 2, 2))
    fun = lambda z: _equations(model, z, tariffs, _scales(model, steady.values), steady.values, steady.values)
    J = _jacobian(fun, 3)(np.tile(steady.coordinates, 3)).toarray()
    A, B, C = np.split(J[n:2*n], 3, axis=1)
    left = np.block([[-B, -A], [np.eye(n), np.zeros((n, n))]])
    right = np.block([[C, np.zeros((n, n))], [np.zeros((n, n)), np.eye(n)]])
    _, _, alpha, beta, _, Z = ordqz(left, right, sort="iuc", output="real")
    with np.errstate(divide="ignore", invalid="ignore"):
        modulus = abs(alpha/beta)
    stable = int(np.sum(modulus < 1-1e-8))
    near_unit = int(np.sum(abs(modulus-1) < 1e-8))
    singular = bool(np.isnan(modulus).any())
    report: dict[str, Any] = dict(stable_graph=False, stable_roots=stable, required_stable_roots=n,
        near_unit_roots=near_unit, singular_pencil=singular,
        lag_rank=int(np.linalg.matrix_rank(A)), lead_rank=int(np.linalg.matrix_rank(C)),
        finite_root_moduli=sorted(modulus[np.isfinite(modulus)].tolist()),
        infinite_roots=int(np.isinf(modulus).sum()), projection_condition=None,
        quadratic_residual_max_abs=None, scope="local stationary stable graph only")
    if stable == n and not near_unit and not singular:
        lower = Z[n:, :n]
        condition = float(np.linalg.cond(lower))
        report["projection_condition"] = condition if np.isfinite(condition) else None
        if np.isfinite(condition) and condition < 1e12:
            H = solve(lower.T, Z[:n, :n].T).T
            error = float(np.max(abs(A+B@H+C@H@H)))
            report["quadratic_residual_max_abs"] = error
            report["stable_graph"] = bool(np.isfinite(error) and error < 1e-8)
    return report


def solve_reallocation_transition(economy: ReallocationEconomy, tariffs: Any, *,
        initial: ReallocationSteadyState | None = None, terminal: ReallocationSteadyState | None = None,
        tolerance: float = 1e-8, max_evaluations: int = 200,
        start: np.ndarray | None = None, require_stable_terminal: bool = True) -> ReallocationTransitionResult:
    """Joint perfect-foresight path with a linearized stationary terminal value.

    Rates[date, importing_country, sector] apply to foreign goods only. All
    policy dates are known at date zero. Initial capital, employment and
    sourcing come from an audited steady state. Last-date capital satisfies
    qK=(C_terminal/C_last)^(-gamma); forward labor/sourcing cost terms vanish.
    Numerical log/logit coordinates are bounded by +/-30 and bound-hitting
    solutions are rejected. Convergence covers the finite boundary problem;
    use the horizon ladder before interpreting the approximate-tail welfare.
    A local stable terminal graph is required by default. Explicitly setting
    require_stable_terminal=False permits a finite boundary experiment only.
    """
    tol, budget = _options(tolerance, max_evaluations)
    rates = _tariffs(tariffs, path=True)
    first = solve_reallocation_steady_state(economy, tolerance=tol) if initial is None else initial
    last = solve_reallocation_steady_state(economy, rates[-1], tolerance=tol) if terminal is None else terminal
    _boundary(economy, first, "initial", tol)
    _boundary(economy, last, "terminal", tol)
    if not np.array_equal(last.tariffs, rates[-1]):
        raise ValueError("terminal tariffs must equal the final path policy")
    if not isinstance(require_stable_terminal, (bool, np.bool_)):
        raise ValueError("require_stable_terminal must be boolean")
    stability = reallocation_stability(last)
    if require_stable_terminal and not stability["stable_graph"]:
        raise ReallocationSolveError("terminal equilibrium does not identify a local stable graph",
            dict(accepted=False, stage="terminal_stability", terminal_stability=stability), last.coordinates)
    T = len(rates)
    z0 = np.broadcast_to(last.coordinates, (T, _SIZE)).copy() if start is None else _array(start, "start", (T, _SIZE))
    scales = _scales(economy, first.values)
    fun = lambda z: _equations(economy, z, rates, scales, first.values, last.values)
    coordinates, report = _optimize(fun, z0, tol, budget)
    paths = {k: v for k, v in _state(economy, coordinates, rates, first.values).items() if not k.startswith("_")}
    report.update(terminal_gap=float(np.max(abs(coordinates[-1]-last.coordinates))),
                  scope="local finite perfect-foresight equations and accounting; horizon accuracy separate",
                  terminal_value="first-order stationary continuation in capital; zero lagged labor/sourcing derivatives",
                  initial_fingerprint=first.economy.fingerprint, terminal_stability=stability,
                  stable_terminal_required=bool(require_stable_terminal))
    provisional = ReallocationTransitionResult(economy, first, last, rates, coordinates, paths, report)
    certificate = audit_reallocation(provisional, tolerance=tol)
    report.update(audit_max_abs=certificate["max_error"], audit_checks=certificate["checks"], accepted=certificate["accepted"])
    if not certificate["accepted"]:
        raise ReallocationSolveError("trade transition accounting or domain audit failed", report, coordinates)
    return ReallocationTransitionResult(economy, first, last, rates, coordinates, paths, report)


def _welfare(run: ReallocationTransitionResult) -> pd.DataFrame:
    model, C0, CT = run.economy, run.initial.values["consumption"], run.terminal.values["consumption"]
    T = len(run.tariffs)
    constant_tail = consumption_equivalent_welfare(run.paths["consumption"], baseline_consumption=C0,
        beta=model.beta, risk_aversion=model.risk_aversion, terminal_consumption=CT)
    correction = model.beta**(T-1)*(CT/C0)**(-model.risk_aversion)*np.sum(
        run.paths["capital_next"][-1]-run.terminal.values["capital"], axis=-1)/C0
    utility = constant_tail.normalized_lifetime_utility+correction
    average = (1-model.beta)*utility
    power = 1-model.risk_aversion
    if power == 0:
        log_equivalent = average
    else:
        if np.any(1+power*average <= 0):
            raise ValueError("linear terminal value leaves the CRRA welfare domain")
        log_equivalent = np.log1p(power*average)/power
    with np.errstate(over="raise", invalid="raise"):
        equivalent = 100*np.expm1(log_equivalent)
    return pd.DataFrame(dict(consumption_equivalent_pct=equivalent,
        normalized_lifetime_utility=utility, terminal_capital_utility_correction=correction,
        tail_is_approximation=True), index=pd.Index(model.countries, name="country"))


@dataclass(frozen=True)
class ReallocationHorizonResult(_Report):
    runs: tuple[ReallocationTransitionResult, ...]
    attempts: tuple[Mapping[str, Any], ...]
    status: str

    def __post_init__(self) -> None:
        object.__setattr__(self, "runs", tuple(self.runs))
        object.__setattr__(self, "attempts", tuple(_freeze_metadata(a) for a in self.attempts))

    @property
    def converged(self) -> bool:
        return self.status == "verified"

    @property
    def accepted(self) -> ReallocationTransitionResult | None:
        return self.runs[-1] if self.converged else None

    def to_frame(self) -> pd.DataFrame:
        return pd.DataFrame([_thaw(a) for a in self.attempts])


def run_reallocation_horizon_ladder(economy: ReallocationEconomy, policy_factory: Callable[[int], np.ndarray], *,
        horizons: tuple[int, ...] = (40, 80, 160), initial: ReallocationSteadyState | None = None,
        terminal: ReallocationSteadyState | None = None, window: int = 20,
        path_tolerance: float = 1e-6, welfare_tolerance: float = 1e-4, terminal_tolerance: float = 1e-5,
        tolerance: float = 1e-8, max_evaluations: int = 200) -> ReallocationHorizonResult:
    """Retain failed solves/refinements; verify a common policy and endpoint.

    Path gaps use log/logit coordinates over the first window dates, welfare
    gaps use consumption-equivalent percentage points. Endpoint accuracy is
    the largest terminal coordinate gap. This is an observed refinement test,
    not a global proof of uniqueness or an infinite-tail error bound.
    """
    grid = tuple(_integer(T, "horizon", 2) for T in horizons)
    window = _integer(window, "window")
    if len(grid) < 2 or any(b <= a for a, b in zip(grid, grid[1:])) or window > grid[0]:
        raise ValueError("require increasing horizons and window <= first horizon")
    limits = [_number(v, n) for v, n in ((path_tolerance, "path_tolerance"),
              (welfare_tolerance, "welfare_tolerance"), (terminal_tolerance, "terminal_tolerance"))]
    if min(limits) <= 0:
        raise ValueError("refinement tolerances must be positive")
    # Validate the whole family before accepting an earlier pair.
    policies = [_tariffs(policy_factory(T), path=True) for T in grid]
    for T, rates in zip(grid, policies):
        if len(rates) != T:
            raise ValueError("policy_factory returned an incorrect horizon")
    if any(not np.array_equal(a, b[:len(a)]) or not np.array_equal(a[-1], b[-1])
           for a, b in zip(policies, policies[1:])):
        raise ValueError("horizon policies must share identical prefixes and terminal policies")
    if any(not np.array_equal(b[len(a):], np.broadcast_to(a[-1], b[len(a):].shape))
           for a, b in zip(policies, policies[1:])):
        raise ValueError("horizon extensions must keep the terminal policy; later policy events change the problem")
    first = solve_reallocation_steady_state(economy, tolerance=tolerance) if initial is None else initial
    last = solve_reallocation_steady_state(economy, policies[-1][-1], tolerance=tolerance) if terminal is None else terminal
    runs: list[ReallocationTransitionResult] = []
    attempts: list[Mapping[str, Any]] = []
    previous = None
    for T, rates in zip(grid, policies):
        try:
            run = solve_reallocation_transition(economy, rates, initial=first, terminal=last,
                tolerance=tolerance, max_evaluations=max_evaluations)
        except ReallocationSolveError as exc:
            attempts.append(dict(horizon=T, solved=False, verified=False, error=str(exc), diagnostics=exc.diagnostics))
            previous = None
            continue
        runs.append(run)
        row = dict(horizon=T, solved=True, verified=False, terminal_gap=run.diagnostics["terminal_gap"])
        try:
            welfare = run.welfare.consumption_equivalent_pct.to_numpy()
            if not np.isfinite(welfare).all():
                raise ValueError("nonfinite consumption equivalent")
        except (ValueError, FloatingPointError, OverflowError) as exc:
            row.update(error=str(exc), stage="terminal_welfare")
            attempts.append(row)
            previous = None
            continue
        if previous is not None:
            path_gap = float(np.max(abs(run.coordinates[:window]-previous.coordinates[:window])))
            welfare_gap = float(np.max(abs(welfare-previous.welfare.consumption_equivalent_pct.to_numpy())))
            row.update(path_gap=path_gap, welfare_gap_pp=welfare_gap)
            row["verified"] = bool(path_gap <= path_tolerance and welfare_gap <= welfare_tolerance
                                   and row["terminal_gap"] <= terminal_tolerance)
        attempts.append(row)
        if row["verified"]:
            return ReallocationHorizonResult(tuple(runs), tuple(attempts), "verified")
        previous = run
    return ReallocationHorizonResult(tuple(runs), tuple(attempts), "unresolved")
