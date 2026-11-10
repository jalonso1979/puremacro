"""Three-date sovereign default with mandated domestic bank exposure.

Finite-state pure price/default equilibria are enumerated, including multiple
and zero-price equilibria. This is a contract economy with restricted household
asset access, not a recursive Arellano model or an endogenous portfolio model.
"""
from __future__ import annotations

from dataclasses import asdict, dataclass
import hashlib
import json
from typing import Any, Mapping

import numpy as np
import pandas as pd

from puremacro._numeric_contracts import freeze as _freeze_metadata, readonly_array, real_scalar, real_vector

from puremacro.reports import _df_to_latex, _df_to_markdown, _df_to_typst

__all__ = ["SovereignBankingParameters", "SovereignBankingEquilibrium",
           "SovereignBankingResult", "sovereign_bank_accounts",
           "solve_sovereign_banking", "audit_sovereign_banking"]

_COLUMNS = ("income", "productivity", "default", "repayment", "default_resource_loss",
            "bank_equity", "credit", "loan_rate", "foreign_funding", "safe_assets",
            "project_output", "firm_profit", "bank_dividend", "consumption1",
            "consumption2", "continuation_value")
_STATUSES = ("inconsistent_default", "equilibrium", "infeasible_portfolio")


def _number(value: Any, name: str) -> float:
    return real_scalar(value, name)


def _vector(value: Any, name: str) -> np.ndarray:
    return real_vector(value, name)


def _owned(value: Any) -> np.ndarray:
    return readonly_array(value)


@dataclass(frozen=True)
class SovereignBankingParameters:
    """All stocks/flows in the same good; R is a gross per-date safe return.

    ``domestic_share`` mandates the fraction of sovereign face held by banks.
    Foreign banks fund private credit at R after the sovereign decision; loans
    cannot exceed ``leverage`` times post-settlement equity. ``default_loss``
    is a separate fraction of date-1 income destroyed in default.
    """
    beta: float = .95
    gross_rate: float = 1.02
    initial_income: float = 1.
    terminal_income: float = .3
    initial_equity: float = .15
    domestic_share: float = .4
    leverage: float = 4.
    credit_scale: float = .1
    default_loss: float = .1

    def __post_init__(self) -> None:
        for name in vars(self):
            object.__setattr__(self, name, _number(getattr(self, name), name))
        if (not 0 < self.beta <= 1 or self.gross_rate <= 0 or self.terminal_income <= 0
                or not 0 < self.initial_equity < self.initial_income
                or not 0 <= self.domestic_share < 1 or self.leverage < 1
                or self.credit_scale <= 0 or not 0 <= self.default_loss < 1):
            raise ValueError("invalid sovereign-bank parameters")


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
class SovereignBankingEquilibrium(_Report):
    """One consistent price/default mask for one debt grid point."""
    debt_index: int
    mask: int
    price: float
    initial_consumption: float
    initial_reserves: float
    value: float
    default: np.ndarray
    accounts: np.ndarray

    def __post_init__(self) -> None:
        object.__setattr__(self, "default", _owned(self.default))
        object.__setattr__(self, "accounts", _owned(self.accounts))

    def to_frame(self) -> pd.DataFrame:
        return pd.DataFrame(self.accounts.copy(), columns=_COLUMNS,
                            index=pd.RangeIndex(len(self.accounts), name="state"))


@dataclass(frozen=True)
class SovereignBankingResult(_Report):
    parameters: SovereignBankingParameters
    income: np.ndarray
    productivity: np.ndarray
    probabilities: np.ndarray
    debt_grid: np.ndarray
    price_selection: str
    decision_tolerance: float
    equilibria: tuple[SovereignBankingEquilibrium, ...]
    candidates: np.ndarray
    selected_indices: np.ndarray
    selected_index: int
    diagnostics: Mapping[str, Any]

    def __post_init__(self) -> None:
        for name in ("income", "productivity", "probabilities", "debt_grid", "candidates", "selected_indices"):
            object.__setattr__(self, name, _owned(getattr(self, name)))
        object.__setattr__(self, "equilibria", tuple(self.equilibria))
        object.__setattr__(self, "diagnostics", _freeze_metadata(self.diagnostics))

    @property
    def selected(self) -> SovereignBankingEquilibrium:
        return self.equilibria[self.selected_index]

    @property
    def fingerprint(self) -> str:
        record = dict(parameters=asdict(self.parameters), income=self.income.tolist(),
                      productivity=self.productivity.tolist(), probabilities=self.probabilities.tolist(),
                      debt_grid=self.debt_grid.tolist(), price_selection=self.price_selection,
                      decision_tolerance=self.decision_tolerance)
        return hashlib.sha256(json.dumps(record, sort_keys=True, allow_nan=False).encode()).hexdigest()

    def to_frame(self) -> pd.DataFrame:
        """Selected equilibrium's realized accounts, one row per state."""
        return self.selected.to_frame()

    def equilibria_frame(self) -> pd.DataFrame:
        return pd.DataFrame([dict(equilibrium=i, debt_index=e.debt_index, debt=self.debt_grid[e.debt_index],
            mask=e.mask, price=e.price, value=e.value, default_probability=float(self.probabilities@e.default),
            expected_credit=float(self.probabilities@e.accounts[:, 6]), initial_consumption=e.initial_consumption,
            initial_reserves=e.initial_reserves, selected_at_debt=i == self.selected_indices[e.debt_index],
            selected=i == self.selected_index) for i, e in enumerate(self.equilibria)])

    def debt_frame(self) -> pd.DataFrame:
        counts = np.bincount([e.debt_index for e in self.equilibria], minlength=len(self.debt_grid))
        return pd.DataFrame(dict(debt=self.debt_grid.copy(), pure_equilibria=counts,
            selected_equilibrium=self.selected_indices.copy(), multiple=counts > 1,
            status=np.where(counts > 0, "available", "no_pure_equilibrium")))

    def candidates_frame(self) -> pd.DataFrame:
        masks = _masks(len(self.income))
        prices = _prices(masks, self.probabilities, self.parameters.gross_rate)
        return pd.DataFrame(dict(debt_index=np.repeat(np.arange(len(self.debt_grid)), len(masks)),
            debt=np.repeat(self.debt_grid, len(masks)), mask=np.tile(np.arange(len(masks)), len(self.debt_grid)),
            price=np.tile(prices, len(self.debt_grid)), status=np.array(_STATUSES)[self.candidates.ravel()]))

    def counterfactual_frame(self, equilibrium: SovereignBankingEquilibrium | None = None) -> pd.DataFrame:
        """Forced repayment/default at the SAME contracted price, not new equilibria.

        Infeasible repayment consumption is retained; its undefined log utility
        is missing, with feasible=False. It never enters a selected allocation.
        """
        e = self.selected if equilibrium is None else equilibrium
        frames = [sovereign_bank_accounts(self.parameters, income=self.income, productivity=self.productivity,
            debt=self.debt_grid[e.debt_index], price=e.price, default=np.full(len(self.income), d, dtype=bool))
            for d in (False, True)]
        return pd.concat(frames).reset_index()


def _masks(n: int) -> np.ndarray:
    return (np.arange(2**n)[:, None] >> np.arange(n)[None, :]) & 1


def _prices(masks: np.ndarray, probabilities: np.ndarray, rate: float) -> np.ndarray:
    prices = (1-masks@probabilities)/rate
    prices[0], prices[-1] = 1/rate, 0.
    return prices


def _allocate(p: SovereignBankingParameters, y: np.ndarray, a: np.ndarray, debt: float,
              prices: np.ndarray, default: np.ndarray) -> np.ndarray:
    """Vectorized candidate x state x account; infeasible utility alone is NaN."""
    q = np.asarray(prices)[:, None]
    d = np.broadcast_to(default, (len(prices), len(y)))
    Y, A = np.broadcast_to(y, d.shape), np.broadcast_to(a, d.shape)
    repayment = (1-d)*debt
    loss = p.default_loss*d*Y
    equity = p.gross_rate*(p.initial_equity-q*p.domestic_share*debt)+p.domestic_share*repayment
    credit = np.minimum(np.maximum(A/p.gross_rate-p.credit_scale, 0.), p.leverage*equity)
    rate = np.maximum(p.gross_rate, A/(p.credit_scale+credit))
    funding, safe = np.maximum(credit-equity, 0.), np.maximum(equity-credit, 0.)
    output = A*np.log1p(credit/p.credit_scale)
    profit = output-rate*credit
    dividend = rate*credit+p.gross_rate*(safe-funding)
    c1, c2 = Y-loss-repayment, p.terminal_income+profit+dividend
    value = np.full(d.shape, np.nan)
    feasible = (c1 > 0) & (c2 > 0)
    value[feasible] = np.log(c1[feasible])+p.beta*np.log(c2[feasible])
    return np.stack([Y, A, d, repayment, loss, equity, credit, rate, funding, safe,
                     output, profit, dividend, c1, c2, value], axis=-1)


def sovereign_bank_accounts(parameters: SovereignBankingParameters, *, income: Any,
        productivity: Any, debt: float, price: float, default: Any) -> pd.DataFrame:
    """Both actual and forced state allocations for one already contracted bond.

    This function does not impose lender pricing or sovereign optimality. Use
    solve_sovereign_banking for equilibrium. Losses occur before private credit.
    """
    if not isinstance(parameters, SovereignBankingParameters):
        raise ValueError("parameters must be SovereignBankingParameters")
    y, a = _vector(income, "income"), _vector(productivity, "productivity")
    d = np.asarray(default)
    b, q = _number(debt, "debt"), _number(price, "price")
    if (a.shape != y.shape or d.shape != y.shape or d.dtype.kind != "b" or np.any(y <= 0)
            or np.any(a <= 0) or b < 0 or not 0 <= q <= 1/parameters.gross_rate
            or parameters.initial_equity-q*parameters.domestic_share*b < 0):
        raise ValueError("invalid states, decisions, price or initial bank portfolio")
    try:
        with np.errstate(over="raise", divide="raise", invalid="raise"):
            accounts = _allocate(parameters, y, a, b, np.array([q]), d)[0]
    except FloatingPointError as exc:
        raise ValueError("nonfinite bank accounts") from exc
    if not np.isfinite(accounts[:, :-1]).all():
        raise ValueError("nonfinite bank accounts")
    frame = pd.DataFrame(accounts, columns=_COLUMNS, index=pd.RangeIndex(len(y), name="state"))
    frame["feasible"] = np.isfinite(accounts[:, -1])
    return frame


def _specification(income: Any, productivity: Any, probabilities: Any, debt_grid: Any,
                   price_selection: str, decision_tolerance: float) -> tuple[np.ndarray, ...]:
    y, a, p, b = (_vector(x, name) for x, name in zip(
        (income, productivity, probabilities, debt_grid), ("income", "productivity", "probabilities", "debt_grid")))
    if (a.shape != y.shape or p.shape != y.shape or np.any(y <= 0) or np.any(a <= 0)
            or np.any(p <= 0) or abs(float(p.sum())-1) > 1e-12
            or b[0] != 0 or np.any(np.diff(b) <= 0)):
        raise ValueError("positive joint states/probabilities summing to one and increasing debt starting at zero required")
    if price_selection not in ("highest", "lowest") or not 0 <= decision_tolerance <= 1e-8:
        raise ValueError("declare highest/lowest price selection and decision_tolerance in [0,1e-8]")
    if len(y) > 12 or len(b)*2**len(y) > 200_000:
        raise ValueError("enumeration limit: at most 12 states and 200000 debt/mask candidates")
    p /= p.sum()
    return y, a, p, b


def solve_sovereign_banking(*, income: Any, productivity: Any, probabilities: Any,
        debt_grid: Any, price_selection: str, parameters: SovereignBankingParameters | None = None,
        decision_tolerance: float = 1e-12) -> SovereignBankingResult:
    """Enumerate pure equilibria, apply the declared price selection, choose debt.

    The finite joint state distribution has positive probabilities. Probabilities
    within 1e-12 of sum one are normalized. Every default mask is considered,
    including all-default (q=0). Infeasible initial bank portfolios are retained
    as rejected candidates; missing pure equilibria are not interpolated.
    Repayment wins utility gains <= decision_tolerance. The smallest debt wins
    exact date-0 utility ties. The routine audits all candidates independently.
    """
    model = SovereignBankingParameters() if parameters is None else parameters
    if not isinstance(model, SovereignBankingParameters):
        raise ValueError("parameters must be SovereignBankingParameters")
    tol = _number(decision_tolerance, "decision_tolerance")
    y, a, p, grid = _specification(income, productivity, probabilities, debt_grid, price_selection, tol)
    masks = _masks(len(y)); prices = _prices(masks, p, model.gross_rate)
    candidates = np.zeros((len(grid), len(masks)), dtype=np.int8)
    equilibria: list[SovereignBankingEquilibrium] = []
    chosen = np.full(len(grid), -1, dtype=int)
    try:
        with np.errstate(over="raise", divide="raise", invalid="raise"):
            for i, b in enumerate(grid):
                reserves = model.initial_equity-prices*model.domestic_share*b
                valid = reserves >= 0
                candidates[i, ~valid] = 2
                ids = np.flatnonzero(valid)
                rep = _allocate(model, y, a, b, prices[valid], np.zeros(len(y)))
                default = _allocate(model, y, a, b, prices[valid], np.ones(len(y)))
                decisions = (~np.isfinite(rep[:, :, -1])) | (default[:, :, -1] > rep[:, :, -1]+tol)
                consistent = np.all(decisions == masks[valid], axis=1)
                ids = ids[consistent]
                candidates[i, ids] = 1
                at_debt = []
                for local, mask in zip(np.flatnonzero(consistent), ids):
                    accounts = np.where(decisions[local, :, None], default[local], rep[local])
                    c0 = model.initial_income-model.initial_equity+prices[mask]*b
                    value = np.log(c0)+model.beta*(p@accounts[:, -1])
                    if not np.isfinite(accounts).all() or not np.isfinite(value):
                        raise ValueError("nonfinite feasible equilibrium")
                    at_debt.append(len(equilibria))
                    equilibria.append(SovereignBankingEquilibrium(i, int(mask), float(prices[mask]), float(c0),
                        float(reserves[mask]), float(value), decisions[local], accounts))
                if at_debt:
                    selection = max if price_selection == "highest" else min
                    chosen[i] = selection(at_debt, key=lambda j: equilibria[j].price)
    except FloatingPointError as exc:
        raise ValueError("nonfinite sovereign-bank enumeration; no clipped solution returned") from exc
    selected = max((int(j) for j in chosen if j >= 0), key=lambda j: (equilibria[j].value, -equilibria[j].debt_index))
    result = SovereignBankingResult(model, y, a, p, grid, price_selection, tol, tuple(equilibria), candidates,
                                    chosen, selected, {})
    audit = audit_sovereign_banking(result)
    if not audit["accepted"]:
        raise ValueError(f"sovereign-bank audit failed: {audit}")
    diagnostics = dict(audit, price_selection=price_selection, pure_equilibria=len(equilibria),
        enumerated_candidates=candidates.size, debt_points_without_pure_equilibrium=int(np.sum(chosen < 0)),
        multiple_equilibrium_debt_points=int(np.sum(np.sum(candidates == 1, axis=1) > 1)),
        selected_upper_debt_boundary=equilibria[selected].debt_index == len(grid)-1 and len(grid) > 1,
        scope="three-date contract economy; imposed domestic exposure; finite-grid pure equilibria")
    return SovereignBankingResult(model, y, a, p, grid, price_selection, tol, tuple(equilibria), candidates,
                                   chosen, selected, diagnostics)


def audit_sovereign_banking(result: SovereignBankingResult, *, tolerance: float = 1e-9) -> dict[str, Any]:
    """Scalar reconstruction of every mask, state, account and selected decision."""
    from puremacro.sovereign._banking_audit import audit
    tol = _number(tolerance, "tolerance")
    if not 0 < tol <= 1e-6 or not isinstance(result, SovereignBankingResult):
        raise ValueError("require SovereignBankingResult and tolerance in (0,1e-6]")
    return audit(result, tol)
