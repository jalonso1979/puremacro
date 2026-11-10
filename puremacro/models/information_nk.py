"""Finite present-value NK forecasts under maintained Gaussian information sets.

Demand and pricing use present values, not a representative Euler equation
with an average expectation substituted into it. This is a linear forecast
system with exogenous information processing constraints, not a solution of
endogenous market learning or rational inattention. The continuation after
the declared horizon is exactly zero. Welfare is not identified here.
"""
from __future__ import annotations

from dataclasses import dataclass
import hashlib
import json
from typing import Any, Mapping

import numpy as np
import pandas as pd

from puremacro._numeric_contracts import freeze as _freeze_metadata

from puremacro.expectations import GaussianInformation, audit_gaussian_information
from puremacro.expectations.gaussian import _array, _number, _weights
from puremacro.reports import _df_to_latex, _df_to_markdown, _df_to_typst

__all__ = ["InformationNKParameters", "InformationNKSolution", "InformationNKPath",
           "solve_information_nk", "audit_information_nk"]


@dataclass(frozen=True)
class InformationNKParameters:
    """Quarterly illustrative parameters; the intertemporal elasticity is EIS."""
    beta: float = .99
    intertemporal_elasticity: float = 1.
    price_stickiness: float = .75
    inverse_frisch: float = 1.

    def __post_init__(self) -> None:
        for name in vars(self):
            value = _number(getattr(self, name), name)
            if (name in ("beta", "price_stickiness") and not 0 < value < 1
                    or name == "intertemporal_elasticity" and value <= 0
                    or name == "inverse_frisch" and value < 0):
                raise ValueError(f"invalid {name}")
            object.__setattr__(self, name, value)
        if not np.isfinite(self.kappa):
            raise ValueError("nonfinite implied Phillips slope")

    @property
    def kappa(self) -> float:
        theta = self.price_stickiness
        return (1-theta)*(1-self.beta*theta)/theta*(self.inverse_frisch+1/self.intertemporal_elasticity)


class _Report:
    def to_frame(self) -> pd.DataFrame:
        raise NotImplementedError

    def to_markdown(self) -> str:
        return _df_to_markdown(self.to_frame())

    def to_latex(self) -> str:
        return _df_to_latex(self.to_frame())

    def to_typst(self) -> str:
        return _df_to_typst(self.to_frame())


@dataclass(frozen=True)
class InformationNKPath(_Report):
    """One primitive realization; rates and inflation are per-period deviations."""
    solution: InformationNKSolution
    realization: np.ndarray
    output: np.ndarray
    inflation: np.ndarray
    nominal_rate: np.ndarray
    next_output_forecasts: np.ndarray
    next_inflation_forecasts: np.ndarray
    next_rate_forecasts: np.ndarray

    def __post_init__(self) -> None:
        for name in ("realization", "output", "inflation", "nominal_rate", "next_output_forecasts",
                     "next_inflation_forecasts", "next_rate_forecasts"):
            object.__setattr__(self, name, _array(getattr(self, name), name))

    def to_frame(self) -> pd.DataFrame:
        info = self.solution.information
        expected_y = self.next_output_forecasts@info.weights
        expected_pi = self.next_inflation_forecasts@self.solution.firm_weights
        return pd.DataFrame(dict(output_gap=self.output, inflation=self.inflation, nominal_rate=self.nominal_rate,
            consumer_next_output=expected_y, firm_next_inflation=expected_pi,
            consumer_output_forecast_error=np.r_[self.output[1:], 0.]-expected_y,
            firm_inflation_forecast_error=np.r_[self.inflation[1:], 0.]-expected_pi),
            index=pd.RangeIndex(info.horizon, name="date"))

    def forecast_frame(self) -> pd.DataFrame:
        info = self.solution.information
        index = pd.MultiIndex.from_product([range(info.horizon), info.group_names], names=["date", "group"])
        return pd.DataFrame(dict(next_output=self.next_output_forecasts.ravel(),
            next_inflation=self.next_inflation_forecasts.ravel(), next_nominal_rate=self.next_rate_forecasts.ravel()), index=index)


@dataclass(frozen=True)
class InformationNKSolution(_Report):
    information: GaussianInformation
    parameters: InformationNKParameters
    firm_weights: np.ndarray
    rate_loadings: np.ndarray
    output_loadings: np.ndarray
    inflation_loadings: np.ndarray
    diagnostics: Mapping[str, Any]

    def __post_init__(self) -> None:
        for name in ("firm_weights", "rate_loadings", "output_loadings", "inflation_loadings"):
            object.__setattr__(self, name, _array(getattr(self, name), name))
        object.__setattr__(self, "diagnostics", _freeze_metadata(self.diagnostics))

    @property
    def converged(self) -> bool:
        return bool(self.diagnostics.get("accepted", False))

    @property
    def fingerprint(self) -> str:
        specification = dict(information=self.information.fingerprint, parameters=vars(self.parameters),
                             firm_weights=self.firm_weights.tolist(), rates=self.rate_loadings.tolist())
        return hashlib.sha256(json.dumps(specification, sort_keys=True, allow_nan=False).encode()).hexdigest()

    def to_frame(self) -> pd.DataFrame:
        index = pd.MultiIndex.from_product([range(self.information.horizon), self.information.state_names],
                                          names=["date", "state"])
        return pd.DataFrame(dict(output=self.output_loadings.ravel(), inflation=self.inflation_loadings.ravel(),
                                 nominal_rate=self.rate_loadings.ravel()), index=index)

    def evaluate(self, realization: Any) -> InformationNKPath:
        """Evaluate paths and one-date-ahead group forecasts; terminal outcomes are zero."""
        info = self.information
        z = _array(realization, "realization", (len(info.covariance),))
        means = info.beliefs(z)
        forecasts = []
        for coefficients in (self.output_loadings, self.inflation_loadings, self.rate_loadings):
            following = np.vstack([coefficients[1:], np.zeros((1, len(z)))])
            forecasts.append(np.einsum("ti,tgi->tg", following, means))
        return InformationNKPath(self, z, self.output_loadings@z, self.inflation_loadings@z,
                                 self.rate_loadings@z, *forecasts)


def solve_information_nk(information: GaussianInformation, nominal_rate_loadings: Any, *,
        parameters: InformationNKParameters | None = None, firm_weights: Any = None,
        tolerance: float = 1e-9) -> InformationNKSolution:
    """Solve finite present-value demand and pricing by backward recursion.

    The primitive state has the common zero-mean Gaussian prior in information.
    R[t] @ z is the exogenous per-period nominal-rate deviation; the natural-rate
    deviation is zero. Consumer weights come from information.weights; firms
    use the same weights unless supplied explicitly. The full future signal
    schedule is known. No unlisted endogenous market signal is processed.

    All rates, output and inflation at/after the horizon are exactly zero, so
    include the full policy support in R. This finite boundary solution does
    not establish infinite-horizon Taylor-rule determinacy or welfare.
    Acceptance checks the entire coefficient system, not one realized path.
    """
    tol = _number(tolerance, "tolerance")
    if not 0 < tol <= 1e-6 or not isinstance(information, GaussianInformation):
        raise ValueError("require GaussianInformation and tolerance in (0,1e-6]")
    certificate = audit_gaussian_information(information, tolerance=tol)
    if not certificate["accepted"]:
        raise ValueError("information set audit failed")
    model = InformationNKParameters() if parameters is None else parameters
    if not isinstance(model, InformationNKParameters):
        raise ValueError("parameters must be InformationNKParameters")
    T, n = information.horizon, len(information.covariance)
    R = _array(nominal_rate_loadings, "nominal rate loadings", (T, n))
    weights = _weights(information.weights if firm_weights is None else firm_weights, len(information.weights))
    consumers = information.average_projectors
    firms = np.einsum("g,tgij->tij", weights, information.projectors)
    Y, Pi = np.zeros((T+1, n)), np.zeros((T+1, n))
    rates_pv, income_pv, pricing_pv = np.zeros(n), np.zeros(n), np.zeros(n)
    beta, theta, eis = model.beta, model.price_stickiness, model.intertemporal_elasticity
    try:
        with np.errstate(over="raise", invalid="raise", divide="raise"):
            for t in range(T-1, -1, -1):
                rates_pv = R[t]-Pi[t+1]+beta*rates_pv
                income_pv = Y[t+1]+beta*income_pv
                Y[t] = (-eis*rates_pv+(1-beta)*income_pv)@consumers[t]
                pricing_pv = beta*theta*(model.kappa*Y[t+1]+(1-theta)/theta*Pi[t+1]+pricing_pv)
                Pi[t] = model.kappa*Y[t]+pricing_pv@firms[t]
    except FloatingPointError as exc:
        raise ValueError("nonfinite NK recursion; no clipped solution returned") from exc
    result = InformationNKSolution(information, model, weights, R, Y[:-1], Pi[:-1], {})
    audit = audit_information_nk(result, tolerance=tol)
    if not audit["accepted"]:
        raise ValueError(f"present-value NK coefficient audit failed: {audit}")
    report = dict(accepted=True, max_error=audit["max_error"], checks=audit["checks"], tolerance=tol,
                  information_fingerprint=information.fingerprint,
                  continuation="exact zero after the declared finite policy horizon",
                  information_scope="only listed signals are processed; endogenous market learning excluded",
                  model_scope="linear present-value NK forecast system; no welfare claim")
    return InformationNKSolution(information, model, weights, R, Y[:-1], Pi[:-1], report)


def audit_information_nk(solution: InformationNKSolution, *, tolerance: float = 1e-9) -> dict[str, Any]:
    """Independent explicit present-value sums, for every primitive direction."""
    tol = _number(tolerance, "tolerance")
    if not 0 < tol <= 1e-6 or not isinstance(solution, InformationNKSolution):
        raise ValueError("require InformationNKSolution and tolerance in (0,1e-6]")
    info, model = solution.information, solution.parameters
    if not isinstance(info, GaussianInformation) or not isinstance(model, InformationNKParameters):
        return dict(accepted=False, max_error=None, reason="invalid retained information or parameters")
    information_check = audit_gaussian_information(info, tolerance=tol)
    if not information_check["accepted"]:
        return dict(accepted=False, max_error=None, reason="information audit failed",
                    information=information_check)
    T, n = info.horizon, len(info.covariance)
    Y, Pi, R = solution.output_loadings, solution.inflation_loadings, solution.rate_loadings
    if any(x.shape != (T, n) or not np.isfinite(x).all() for x in (Y, Pi, R)):
        return dict(accepted=False, max_error=None, reason="invalid coefficient shapes or values")
    try:
        _weights(solution.firm_weights, len(info.weights))
    except ValueError:
        return dict(accepted=False, max_error=None, reason="invalid firm weights")
    demand_errors, pricing_errors = [], []
    for t in range(T):
        demand, pricing = np.zeros(n), model.kappa*Y[t].copy()
        for g, P in enumerate(info.projectors[t]):
            for j in range(t, T):
                next_pi = Pi[j+1] if j+1 < T else np.zeros(n)
                demand -= info.weights[g]*model.intertemporal_elasticity*model.beta**(j-t)*(R[j]-next_pi)@P
                if j > t:
                    demand += info.weights[g]*(1-model.beta)*model.beta**(j-t-1)*Y[j]@P
                    pricing += solution.firm_weights[g]*(model.beta*model.price_stickiness)**(j-t)*(
                        model.kappa*Y[j]+(1-model.price_stickiness)/model.price_stickiness*Pi[j])@P
        demand_errors.append(float(np.max(abs(Y[t]-demand))/(1+np.max(abs(Y[t]))+np.max(abs(demand)))))
        pricing_errors.append(float(np.max(abs(Pi[t]-pricing))/(1+np.max(abs(Pi[t]))+np.max(abs(pricing)))))
    checks = dict(demand_present_value=max(demand_errors), pricing_present_value=max(pricing_errors),
                  information=information_check["max_error"])
    maximum = max(v for v in checks.values() if v is not None)
    return dict(accepted=bool(information_check["accepted"] and np.isfinite(maximum) and maximum <= tol),
                max_error=maximum, checks=checks, tolerance=tol,
                period_errors=np.column_stack([demand_errors, pricing_errors]),
                scope="coefficient-space equations, valid for all primitive realizations")
