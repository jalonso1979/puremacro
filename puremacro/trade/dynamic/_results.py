"""Frozen result containers and exceptions for :mod:`puremacro.trade.dynamic`.

Every container is a ``@dataclass(frozen=True)`` built by a validating function
(never by a raising ``__post_init__``) and exposes ``to_dataframe()``,
``to_markdown()``, ``to_latex()`` and ``to_typst()`` through
:mod:`puremacro.reports`. Arrays stored here are read-only copies.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Sequence

import numpy as np
import pandas as pd

from puremacro.reports import df_to_latex, df_to_markdown, df_to_typst

__all__ = [
    "EconomicDomainError",
    "DynamicSolveError",
    "DynamicDeterminacyError",
    "DynamicCalibration",
    "DynamicSteadyStateResult",
    "DynamicTransitionResult",
    "HorizonComparisonResult",
    "HorizonLadderResult",
    "ConsumptionEquivalentResult",
    "DynamicStabilityResult",
]


# ---------------------------------------------------------------------------
# Exceptions
# ---------------------------------------------------------------------------
class EconomicDomainError(ValueError):
    """A trial point is outside the specified interior equilibrium domain.

    Raised by :class:`~puremacro.trade.dynamic.DynamicEconomy` when a state
    would need negative investment, a nonpositive capital stock, output,
    price or continuation payoff, or an installation rate off the increasing
    branch of the adjustment-cost function. The solvers treat it as a rejected
    line-search trial, never as a solution.
    """


class DynamicSolveError(RuntimeError):
    """A dynamic solve did not meet its tolerance or an economic certificate failed.

    Attributes
    ----------
    residual : float or None
        Maximum absolute retained-equation residual at the last iterate.
    iterate : np.ndarray or None
        The last accepted iterate: a stacked path ``(T * n_vars,)`` or
        ``(T, n_vars)`` from the transition solver, a retained steady-state
        vector ``(n_vars,)`` from the full solver, or the condensed ``2C``
        vector ``(log w, log PI/PI0)`` from ``method="condensed"``. It is a
        diagnostic, never a solution.
    location : dict or None
        Where the failure occurred: date, equation block, cell or country and
        installation-domain diagnostics when a line search stalled.
    history : tuple of dict
        Newton history up to the failure.
    certificate : dict or None
        The failing independent certificate, when one was evaluated.
    """

    def __init__(self, message: str, *, residual: float | None = None,
                 iterate: np.ndarray | None = None, location: dict | None = None,
                 history: Sequence[dict] | None = None,
                 certificate: dict | None = None) -> None:
        super().__init__(message)
        self.residual = residual
        self.iterate = None if iterate is None else np.array(iterate, dtype=float, copy=True)
        self.location = dict(location) if location else None
        self.history = tuple(history or ())
        self.certificate = dict(certificate) if certificate else None


class DynamicDeterminacyError(DynamicSolveError):
    """The linearised dynamics are not saddle-path determinate at the terminal state.

    Attributes
    ----------
    report : DynamicStabilityResult
        The Blanchard-Kahn count that triggered the error (``None`` only for
        an instance rebuilt without one, e.g. after unpickling in a worker
        process, where the pickled attribute is restored afterwards).
    """

    def __init__(self, message: str, report: "DynamicStabilityResult | None" = None, **kwargs: Any) -> None:
        super().__init__(message, **kwargs)
        self.report = report


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------
def _readonly(value: Any, dtype: Any = float) -> np.ndarray:
    array = np.array(value, dtype=dtype, copy=True)
    array.flags.writeable = False
    return array


def _frozen_dicts(items: Sequence[dict] | None) -> tuple[dict, ...]:
    return tuple(dict(item) for item in (items or ()))


def _freeze_state(state: dict[str, Any]) -> dict[str, Any]:
    """Mark every ndarray value of a date-state dict read-only (in place) and return it."""
    for value in state.values():
        if isinstance(value, np.ndarray):
            value.flags.writeable = False
    return state


def _ce_listing(codes: Sequence[str], pct: np.ndarray, *, limit: int = 5, hint: str = "to_dataframe()") -> str:
    """``code=+x.xxxx`` entries of the ``limit`` largest |CE| values, then a count of the rest."""
    pct = np.asarray(pct, dtype=float)
    labels = tuple(codes) if len(codes) == len(pct) else tuple(str(i) for i in range(len(pct)))
    order = np.argsort(-np.abs(pct), kind="stable")[:limit]
    text = ", ".join(f"{labels[i]}={pct[i]:+.4f}" for i in order)
    rest = len(pct) - len(order)
    if rest > 0:
        text += f" and {rest} more (largest |CE| shown; see {hint})"
    return text


class _ReportMixin:
    """Markdown / LaTeX / Typst rendering of ``to_dataframe()``."""

    def to_dataframe(self, *args: Any, **kwargs: Any) -> pd.DataFrame:  # pragma: no cover - overridden
        raise NotImplementedError

    def to_markdown(self, *args: Any, **kwargs: Any) -> str:
        """``to_dataframe(*args, **kwargs)`` rendered as a Markdown table."""
        return df_to_markdown(self.to_dataframe(*args, **kwargs))

    def to_latex(self, *args: Any, **kwargs: Any) -> str:
        """``to_dataframe(*args, **kwargs)`` rendered as a LaTeX table."""
        return df_to_latex(self.to_dataframe(*args, **kwargs))

    def to_typst(self, *args: Any, **kwargs: Any) -> str:
        """``to_dataframe(*args, **kwargs)`` rendered as a Typst table."""
        return df_to_typst(self.to_dataframe(*args, **kwargs))


# ---------------------------------------------------------------------------
# Calibration
# ---------------------------------------------------------------------------
@dataclass(frozen=True)
class DynamicCalibration(_ReportMixin):
    """An exactly stationary sector-capital economy reconciled with one IO year.

    Built by :func:`puremacro.trade.dynamic.calibrate_dynamic`. Cells are the
    *active* source cells (positive gross output) in country-major order;
    ``active_indices`` maps them back to the full table.

    Attributes
    ----------
    countries, sectors : tuple of str
        Registries of the source table.
    active_indices : np.ndarray, shape (N,)
        Full-table cell indices of the N active cells.
    country : np.ndarray, shape (N,)
        Country index of every active cell.
    A : scipy.sparse.csr_matrix, shape (N, N)
        Input coefficients ``A = Z diag(1/y0)`` (rows sellers, columns buyers).
    Z : scipy.sparse.csr_matrix, shape (N, N)
        Basic-price intermediate deliveries among active cells.
    y0, VA0, b, tax, alpha : np.ndarray, shape (N,)
        Benchmark output, factor income, factor cost share ``b = VA0/y0``,
        output-tax wedge ``tax`` (production and product taxes over output)
        and capital share in value added.
    omegaC, omegaCtax, omegaI, qOther : np.ndarray, shape (N, C)
        Fixed consumption basket (including exempt purchases abroad), its
        taxable part, the investment basket, and a zero placeholder basket.
    TV : np.ndarray, shape (C,)
        Zero placeholder for other wage-indexed receipts.
    tC, tI : np.ndarray, shape (C,)
        Final-use tax shares of purchaser value, ``PC0 = 1/(1-tC)``.
    C0, I0, K0, R0, L0sector, L0, PC0, PI0, Y0, XN0 : np.ndarray
        Benchmark consumption (C,), replacement investment by cell (N,),
        capital stocks (N,), rentals (N,), labour by cell (N,) and country (C,),
        purchaser prices (C,), GDP (C,) and balanced net-export transfers (C,).
    beta, delta : float
        Discount factor and depreciation rate.
    merchandise_mask : np.ndarray or None, shape (S,)
        Goods sectors, when the accounts carried one.
    report : dict
        The complete adjustment ledger (see ``to_dataframe``).
    """

    countries: tuple[str, ...]
    sectors: tuple[str, ...]
    active_indices: np.ndarray
    country: np.ndarray
    A: Any
    Z: Any
    y0: np.ndarray
    VA0: np.ndarray
    b: np.ndarray
    tax: np.ndarray
    alpha: np.ndarray
    omegaC: np.ndarray
    omegaCtax: np.ndarray
    omegaI: np.ndarray
    qOther: np.ndarray
    TV: np.ndarray
    tC: np.ndarray
    tI: np.ndarray
    C0: np.ndarray
    I0: np.ndarray
    K0: np.ndarray
    R0: np.ndarray
    L0sector: np.ndarray
    L0: np.ndarray
    PC0: np.ndarray
    PI0: np.ndarray
    Y0: np.ndarray
    XN0: np.ndarray
    beta: float
    delta: float
    merchandise_mask: np.ndarray | None = None
    report: dict[str, Any] = field(default_factory=dict)
    metadata: dict[str, Any] = field(default_factory=dict)

    @property
    def n_cells(self) -> int:
        """Number of active cells N."""
        return int(len(self.y0))

    @property
    def n_countries(self) -> int:
        """Number of countries C."""
        return len(self.countries)

    @property
    def n_sectors(self) -> int:
        """Number of sectors S in the source registry."""
        return len(self.sectors)

    @property
    def Inational0(self) -> np.ndarray:
        """Benchmark replacement investment by country, ``sum_j in c I0_j`` (read-only)."""
        out = np.bincount(self.country, weights=self.I0, minlength=self.n_countries)
        out.flags.writeable = False
        return out

    @property
    def budget_scale(self) -> np.ndarray:
        """Numerical normalization of national budgets, explicitly separate from measured GDP."""
        return np.maximum(np.abs(self.Y0), self.L0 + np.bincount(
            self.country, weights=self.alpha * self.VA0, minlength=self.n_countries))

    @property
    def cell_labels(self) -> tuple[str, ...]:
        """``country:sector`` labels of the active cells."""
        count = len(self.sectors)
        return tuple(f"{self.countries[int(i) // count]}:{self.sectors[int(i) % count]}"
                     for i in self.active_indices)

    @property
    def cell_sector(self) -> np.ndarray:
        """Sector index of every active cell."""
        return np.asarray(self.active_indices) % len(self.sectors)

    def expand_cells(self, values: np.ndarray, fill: float = 0.0) -> np.ndarray:
        """Restore inactive source cells (with ``fill``) without pretending they have capital."""
        values = np.asarray(values)
        shape = values.shape[:-1] + (len(self.countries) * len(self.sectors),)
        result = np.full(shape, fill, dtype=float)
        result[..., self.active_indices] = values
        return result

    def to_dataframe(self, which: str = "countries") -> pd.DataFrame:
        """Adjustment ledger tables.

        ``which="countries"`` (default): one row per country with observed and
        stationary investment, the investment flow reclassified to consumption,
        inventories and valuables reclassified, whether the investment basket
        was rebasketed, and the benchmark aggregates. ``"factors"``: the
        factor-reclassification records (one row per altered cell).
        ``"investment"``: the investment-basket reallocation records.
        """
        rep = self.report
        if which == "countries":
            rebasketed = set(rep.get("investment_basket_reallocated_countries", ()))
            basket_change = {r["country"]: r["source_basket_absolute_change"]
                             for r in rep.get("investment_basket_reallocations", ())}
            frame = pd.DataFrame({
                "observed_investment": rep.get("observed_investment", self.Inational0),
                "stationary_investment": rep.get("stationary_investment", self.Inational0),
                "investment_to_consumption": rep.get("investment_to_consumption", np.zeros(self.n_countries)),
                "inventory_to_consumption": rep.get("inventory_flow_reclassified_to_consumption", np.zeros(self.n_countries)),
                "valuables_to_consumption": rep.get("valuables_reclassified_to_consumption", np.zeros(self.n_countries)),
                "rebasketed": [c in rebasketed for c in self.countries],
                "basket_absolute_change": [basket_change.get(c, 0.0) for c in self.countries],
                "C0": self.C0, "K0": np.bincount(self.country, weights=self.K0, minlength=self.n_countries),
                "L0": self.L0, "PC0": self.PC0, "PI0": self.PI0, "Y0": self.Y0, "XN0": self.XN0,
            }, index=pd.Index(list(self.countries), name="country"))
            return frame
        if which == "factors":
            columns = ["cell", "labor_observed", "capital_observed", "factor_income_calibrated",
                       "capital_share_calibrated", "production_tax_change"]
            records = rep.get("factor_adjustments", ())
            return pd.DataFrame(list(records), columns=columns)
        if which == "investment":
            columns = ["country", "stationary_investment", "source_basket_absolute_change",
                       "capacity_bound_cells", "new_source_cells", "new_source_investment"]
            records = rep.get("investment_basket_reallocations", ())
            return pd.DataFrame(list(records), columns=columns)
        raise ValueError("which must be 'countries', 'factors' or 'investment'")

    def summary(self) -> str:
        """One-line description of the calibration and the policies applied."""
        rep = self.report
        return (f"DynamicCalibration: {self.n_countries} countries x {self.n_sectors} sectors, "
                f"{self.n_cells} active cells ({rep.get('inactive_cells', 0)} inactive); "
                f"beta={self.beta}, delta={self.delta}; factor policy {rep.get('factor_policy')} "
                f"({rep.get('factor_adjusted_cells', 0)} cells altered); investment policy "
                f"{rep.get('investment_policy')} ({len(rep.get('investment_basket_reallocated_countries', ()))} "
                f"countries rebasketed); accounting policy {rep.get('accounting_policy')}.")


# ---------------------------------------------------------------------------
# Welfare
# ---------------------------------------------------------------------------
@dataclass(frozen=True)
class ConsumptionEquivalentResult(_ReportMixin):
    """CRRA consumption equivalents by country relative to a stationary base.

    ``consumption_equivalent_ratio`` is the permanent proportional change in
    benchmark consumption whose discounted CRRA utility equals that of the
    computed path (plus its stationary tail when ``tail_included``).
    Percentages are ``100 * (ratio - 1)``. Monetary fields are ``None`` when no
    benchmark price was supplied. There is no cross-country aggregation.
    """

    consumption_equivalent_ratio: np.ndarray
    consumption_equivalent_pct: np.ndarray
    normalized_lifetime_utility: np.ndarray
    equivalent_annual_expenditure: np.ndarray | None
    equivalent_present_value_expenditure: np.ndarray | None
    ev_pct_gdp: np.ndarray | None
    beta: float
    risk_aversion: float
    horizon: int
    tail_included: bool
    tail_is_approximation: bool
    terminal_discount_weight: float
    total_discount_weight: float
    interpretation: str
    country_codes: tuple[str, ...] = ()
    metadata: dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        """JSON-compatible record with explicit units and tail timing."""
        out: dict[str, Any] = {}
        for name in ("consumption_equivalent_ratio", "consumption_equivalent_pct",
                     "normalized_lifetime_utility", "equivalent_annual_expenditure",
                     "equivalent_present_value_expenditure", "ev_pct_gdp"):
            value = getattr(self, name)
            out[name] = None if value is None else np.asarray(value).tolist()
        for name in ("beta", "risk_aversion", "horizon", "tail_included", "tail_is_approximation",
                     "terminal_discount_weight", "total_discount_weight", "interpretation"):
            out[name] = getattr(self, name)
        out["country_codes"] = list(self.country_codes)
        return out

    def to_dataframe(self) -> pd.DataFrame:
        """One row per country: ratio, percent, utility and monetary equivalents."""
        n = len(self.consumption_equivalent_ratio)
        index = pd.Index(list(self.country_codes) if len(self.country_codes) == n
                         else [str(i) for i in range(n)], name="country")
        data = {"consumption_equivalent_ratio": self.consumption_equivalent_ratio,
                "consumption_equivalent_pct": self.consumption_equivalent_pct,
                "normalized_lifetime_utility": self.normalized_lifetime_utility}
        if self.equivalent_annual_expenditure is not None:
            data["equivalent_annual_expenditure"] = self.equivalent_annual_expenditure
            data["equivalent_present_value_expenditure"] = self.equivalent_present_value_expenditure
        if self.ev_pct_gdp is not None:
            data["ev_pct_gdp"] = self.ev_pct_gdp
        return pd.DataFrame(data, index=index)

    def summary(self) -> str:
        """One line with the five largest |CE| percentages (all countries are in ``to_dataframe()``)."""
        tail = "with stationary tail" if self.tail_included else "finite horizon only"
        pct = _ce_listing(self.country_codes, self.consumption_equivalent_pct)
        return (f"Consumption-equivalent welfare ({tail}, T={self.horizon}, beta={self.beta}, "
                f"sigma={self.risk_aversion}) in percent of benchmark consumption: {pct}.")


# ---------------------------------------------------------------------------
# Stability
# ---------------------------------------------------------------------------
@dataclass(frozen=True)
class DynamicStabilityResult(_ReportMixin):
    """Blanchard-Kahn count of the linearised sector-capital dynamics.

    Generalized eigenvalues of the first-order pencil built from the model's
    own Jacobian-vector products at a stationary state. ``determinate`` means
    the number of stable roots (modulus below ``1 - tolerance``) equals the
    number of predetermined capital stocks and no root sits on the unit circle.
    ``loading_share`` gives the squared-norm share of the unstable eigenvector
    with the smallest modulus above one by country; ``top_loadings`` lists its
    largest entries (normalized to a maximum of one) as ``(label, value)``.
    """

    n_variables: int
    n_predetermined: int
    n_stable: int
    n_unit: int
    n_unstable: int
    n_infinite: int
    determinate: bool
    slowest_stable_root: float
    smallest_unstable_root: float
    eigenvalues: np.ndarray
    unstable_eigenvalue: complex | None
    loading_countries: tuple[str, ...]
    loading_share: np.ndarray
    top_loadings: tuple[tuple[str, float], ...]
    tolerance: float
    seconds: float
    metadata: dict[str, Any] = field(default_factory=dict)

    def to_dataframe(self, which: str = "countries") -> pd.DataFrame:
        """``"countries"``: unstable-eigenvector mass share by country (descending);
        ``"loadings"``: the largest normalized entries of that eigenvector."""
        if which == "countries":
            frame = pd.DataFrame({"loading_share": self.loading_share},
                                 index=pd.Index(list(self.loading_countries), name="country"))
            return frame.sort_values("loading_share", ascending=False)
        if which == "loadings":
            return pd.DataFrame(list(self.top_loadings), columns=["variable", "loading"])
        raise ValueError("which must be 'countries' or 'loadings'")

    def summary(self) -> str:
        """One-line Blanchard-Kahn verdict with the counts and the extreme roots."""
        verdict = "determinate" if self.determinate else "NOT determinate"
        extra = ""
        if self.unstable_eigenvalue is not None and len(self.loading_share):
            top = int(np.argmax(self.loading_share))
            extra = (f"; smallest unstable root {self.smallest_unstable_root:.6g} loads "
                     f"{100 * self.loading_share[top]:.1f}% on {self.loading_countries[top]}")
        return (f"Blanchard-Kahn: {self.n_stable} stable roots for {self.n_predetermined} "
                f"predetermined stocks ({self.n_unit} unit, {self.n_unstable} unstable, "
                f"{self.n_infinite} infinite): {verdict}; slowest stable root "
                f"{self.slowest_stable_root:.6g}{extra}.")


# ---------------------------------------------------------------------------
# Steady state
# ---------------------------------------------------------------------------
_COUNTRY_FIELDS = ("w", "C", "Inational", "PC", "PI", "TR", "PT", "transfer", "budget", "income", "EV")
_CELL_FIELDS = ("K", "Knext", "I", "x", "y", "va", "R", "labor_j", "p", "q")


def _country_frame(state: dict, benchmark: dict, countries: Sequence[str]) -> pd.DataFrame:
    return pd.DataFrame({
        "wage": state["w"], "consumption": state["C"],
        "consumption_change_pct": 100.0 * (state["C"] / benchmark["C0"] - 1.0),
        "consumption_price": state["PC"], "investment_price": state["PI"],
        "investment": state["Inational"],
        "investment_change_pct": 100.0 * (state["Inational"] / benchmark["Inational0"] - 1.0),
        "tariff_receipts": state["TR"], "tax_receipts": state["PT"],
        "transfer": state["transfer"], "budget_residual": state["budget"],
    }, index=pd.Index(list(countries), name="country"))


def _cell_frame(state: dict, benchmark: dict, labels: Sequence[str]) -> pd.DataFrame:
    return pd.DataFrame({
        "output": state["y"], "output_change_pct": 100.0 * (state["y"] / benchmark["y0"] - 1.0),
        "capital": state["K"], "capital_change_pct": 100.0 * (state["K"] / benchmark["K0"] - 1.0),
        "investment": state["I"], "investment_rate": state["x"], "rental": state["R"],
        "price": state["p"], "tobin_q": state["q"], "labor": state["labor_j"],
    }, index=pd.Index(list(labels), name="cell"))


@dataclass(frozen=True)
class DynamicSteadyStateResult(_ReportMixin):
    """A certified stationary equilibrium of :class:`DynamicEconomy`.

    ``z`` is the retained vector ``[log w, log C/C0, log K/K0]``; ``state`` is
    the full reconstructed date state (prices, quantities, receipts, budgets).
    ``certificate`` holds the 17 independently rebuilt accounting checks, each
    at or below ``max(1e-8, tol)``. A result exists only if it converged.
    """

    z: np.ndarray
    converged: bool
    max_residual: float
    iterations: int
    seconds: float
    method: str
    history: tuple[dict, ...]
    certificate: dict[str, float]
    state: dict[str, np.ndarray]
    policy: Any
    country_codes: tuple[str, ...]
    cell_labels: tuple[str, ...]
    benchmark: dict[str, np.ndarray]
    tol: float
    metadata: dict[str, Any] = field(default_factory=dict)

    @property
    def n_vars(self) -> int:
        """Retained unknowns ``N + 2C``."""
        return int(len(self.z))

    def to_dataframe(self, level: str = "country") -> pd.DataFrame:
        """Country table (wages, consumption, prices, receipts, budgets) or cell table."""
        if level == "country":
            return _country_frame(self.state, self.benchmark, self.country_codes)
        if level == "cell":
            return _cell_frame(self.state, self.benchmark, self.cell_labels)
        raise ValueError("level must be 'country' or 'cell'")

    def summary(self) -> str:
        """One line with method, iterations, residual and the worst certificate entry."""
        worst = max(self.certificate, key=self.certificate.get) if self.certificate else "none"
        return (f"Dynamic steady state [{getattr(self.policy, 'label', 'policy')}] via {self.method}: "
                f"{self.iterations} iterations, residual {self.max_residual:.3g}, certificate max "
                f"{max(self.certificate.values()) if self.certificate else float('nan'):.3g} ({worst}), "
                f"{self.seconds:.2f} s.")


# ---------------------------------------------------------------------------
# Transition
# ---------------------------------------------------------------------------
@dataclass(frozen=True)
class DynamicTransitionResult(_ReportMixin):
    """A certified finite-horizon perfect-foresight transition.

    ``path[t]`` is the retained vector at date ``t = 0..T-1`` with ``initial``
    the fixed inherited state and ``terminal`` the separately solved stationary
    state under the last policy. ``states[t]`` are the full date states and
    ``boundary_state`` the terminal date evaluated with the capital actually
    inherited from ``path[-1]``. ``certificate`` holds the maximum over dates of
    every independent check. ``terminal_max_log_gap`` and
    ``discounted_endpoint_capital_value`` are the two horizon diagnostics used
    by :func:`compare_horizons`; a solved finite stack does not by itself
    establish an adequate horizon.
    """

    path: np.ndarray
    policies: tuple[Any, ...]
    initial: np.ndarray
    terminal: np.ndarray
    states: tuple[dict, ...]
    boundary_state: dict[str, np.ndarray]
    converged: bool
    max_residual: float
    iterations: int
    seconds: float
    history: tuple[dict, ...]
    certificate: dict[str, float]
    welfare: ConsumptionEquivalentResult | None
    terminal_max_log_gap: float
    discounted_endpoint_capital_value: np.ndarray
    stability: DynamicStabilityResult | None
    country_codes: tuple[str, ...]
    cell_labels: tuple[str, ...]
    benchmark: dict[str, np.ndarray]
    tol: float
    metadata: dict[str, Any] = field(default_factory=dict)

    @property
    def horizon(self) -> int:
        """Number of solved dates T."""
        return int(self.path.shape[0])

    @property
    def n_vars(self) -> int:
        """Retained unknowns per date ``N + 2C``."""
        return int(self.path.shape[1])

    @property
    def z(self) -> np.ndarray:
        """Alias of ``path`` (the IO engine's name)."""
        return self.path

    def consumption_ratio(self) -> np.ndarray:
        """``C[t, c] / C0[c]`` for every date and country."""
        return np.asarray([s["C"] for s in self.states]) / self.benchmark["C0"][None, :]

    def to_dataframe(self, level: str = "country", variables: Sequence[str] | None = None) -> pd.DataFrame:
        """Wide date table, columns ``variable:country`` or ``variable:cell``.

        Default country variables: C, w, Inational, PC, PI. Default cell
        variables: K, I, y, p, q. Any field of the date state is accepted.
        """
        if level == "country":
            names = tuple(variables) if variables is not None else ("C", "w", "Inational", "PC", "PI")
            labels = self.country_codes
            allowed = _COUNTRY_FIELDS
        elif level == "cell":
            names = tuple(variables) if variables is not None else ("K", "I", "y", "p", "q")
            labels = self.cell_labels
            allowed = _CELL_FIELDS
        else:
            raise ValueError("level must be 'country' or 'cell'")
        columns: dict[str, np.ndarray] = {}
        for name in names:
            if name not in allowed:
                raise ValueError(f"Unknown {level} variable {name!r}; choose from {allowed}")
            block = np.asarray([s[name] for s in self.states])
            for k, label in enumerate(labels):
                columns[f"{name}:{label}"] = block[:, k]
        return pd.DataFrame(columns, index=pd.RangeIndex(self.horizon, name="date"))

    def summary(self) -> str:
        """One line with iterations, residual, certificate, terminal gap and the five largest |CE|."""
        worst = max(self.certificate, key=self.certificate.get) if self.certificate else "none"
        welfare = ""
        if self.welfare is not None:
            welfare = " CE pct: " + _ce_listing(self.country_codes, self.welfare.consumption_equivalent_pct,
                                                hint=".welfare.to_dataframe()")
        return (f"Dynamic transition T={self.horizon} ({self.horizon * self.n_vars:,} unknowns): "
                f"{self.iterations} Newton iterations, residual {self.max_residual:.3g}, certificate max "
                f"{max(self.certificate.values()) if self.certificate else float('nan'):.3g} ({worst}), "
                f"terminal log gap {self.terminal_max_log_gap:.3g}, {self.seconds:.1f} s.{welfare}")

    def plot(self, ax: Any = None, *, countries: Sequence[str] | None = None,
             variable: str = "C", show: bool = False) -> Any:
        """Percent deviation of a country variable from its benchmark by date.

        ``variable`` is one of C, w, Inational, PC, PI. Draws into ``ax`` when
        given (returns it) or creates a figure (returns it). Never calls
        ``plt.show()`` unless ``show=True``.
        """
        import matplotlib.pyplot as plt

        base = {"C": "C0", "Inational": "Inational0", "PC": "PC0", "PI": "PI0", "w": None}
        if variable not in base:
            raise ValueError("variable must be one of C, w, Inational, PC, PI")
        values = np.asarray([s[variable] for s in self.states])
        if base[variable] is not None:
            values = 100.0 * (values / self.benchmark[base[variable]][None, :] - 1.0)
        else:
            values = 100.0 * (values - 1.0)
        chosen = list(countries) if countries is not None else list(self.country_codes)
        fig = None
        if ax is None:
            fig, ax = plt.subplots(figsize=(7, 4))
        for code in chosen:
            if code not in self.country_codes:
                raise ValueError(f"Unknown country {code!r}")
            ax.plot(np.arange(self.horizon), values[:, self.country_codes.index(code)], label=code)
        ax.axhline(0.0, color="black", linewidth=0.6)
        ax.set_xlabel("date")
        ax.set_ylabel(f"{variable}: percent deviation from benchmark")
        ax.legend(loc="best", fontsize=8)
        if show:
            plt.show()
        return fig if fig is not None else ax


# ---------------------------------------------------------------------------
# Horizon acceptance
# ---------------------------------------------------------------------------
@dataclass(frozen=True)
class HorizonComparisonResult(_ReportMixin):
    """Early-window, welfare and terminal comparison of two horizons.

    ``passed`` requires all three checks. ``status`` is ``"verified"`` under
    the default ``state_gap`` terminal check, ``"verified_window_and_welfare"``
    under the explicitly weaker ``discounted_wealth`` check, and
    ``"not_verified"`` otherwise. See :func:`compare_horizons`.
    """

    periods: int
    short_horizon: int
    long_horizon: int
    max_log_difference: float
    tolerance: float
    window_passed: bool
    welfare_max_difference_pp: float
    welfare_tolerance: float
    welfare_passed: bool
    terminal_check: str
    terminal_max_log_gap: float
    terminal_tolerance: float
    max_discounted_wealth: float
    discounted_wealth_tolerance: float
    terminal_passed: bool
    passed: bool
    status: str
    validation_scope: str
    metadata: dict[str, Any] = field(default_factory=dict)

    def to_dataframe(self) -> pd.DataFrame:
        """One row per check (value, tolerance, passed); ``attrs['terminal_check']`` names the terminal mode."""
        rows = [
            ("early_window_max_log_difference", self.max_log_difference, self.tolerance, self.window_passed),
            ("welfare_max_difference_pp", self.welfare_max_difference_pp, self.welfare_tolerance, self.welfare_passed),
            ("terminal_max_log_gap", self.terminal_max_log_gap, self.terminal_tolerance,
             self.terminal_max_log_gap <= self.terminal_tolerance),
            ("max_discounted_wealth", self.max_discounted_wealth, self.discounted_wealth_tolerance,
             self.max_discounted_wealth <= self.discounted_wealth_tolerance),
        ]
        frame = pd.DataFrame(rows, columns=["check", "value", "tolerance", "passed"]).set_index("check")
        frame.attrs["terminal_check"] = self.terminal_check
        return frame

    def summary(self) -> str:
        """One line with the three checks, their values and the status."""
        return (f"Horizons {self.short_horizon} -> {self.long_horizon}: window {self.max_log_difference:.3g} "
                f"(tol {self.tolerance:g}), welfare {self.welfare_max_difference_pp:.3g} pp "
                f"(tol {self.welfare_tolerance:g}), terminal check {self.terminal_check}: "
                f"state gap {self.terminal_max_log_gap:.3g}, discounted wealth "
                f"{self.max_discounted_wealth:.3g}; status {self.status}.")


@dataclass(frozen=True)
class HorizonLadderResult(_ReportMixin):
    """Solutions and comparisons produced by :func:`run_horizon_ladder`.

    ``accepted_horizon`` is the first horizon whose comparison with its
    predecessor passed, or ``None``. ``status`` mirrors the IO driver:
    ``verified``, ``verified_window_and_welfare`` or
    ``solved_horizon_not_validated``.
    """

    horizons: tuple[int, ...]
    solutions: tuple[DynamicTransitionResult, ...]
    comparisons: tuple[HorizonComparisonResult, ...]
    accepted_horizon: int | None
    status: str
    validation_scope: str
    metadata: dict[str, Any] = field(default_factory=dict)

    @property
    def accepted(self) -> DynamicTransitionResult | None:
        """The accepted transition (the longest solved one when accepted)."""
        if self.accepted_horizon is None:
            return None
        return self.solutions[self.horizons.index(self.accepted_horizon)]

    @property
    def last(self) -> DynamicTransitionResult:
        """The longest solved transition, accepted or not."""
        return self.solutions[-1]

    def to_dataframe(self) -> pd.DataFrame:
        """One row per horizon comparison with the three check values and the status."""
        rows = [(c.short_horizon, c.long_horizon, c.max_log_difference, c.welfare_max_difference_pp,
                 c.terminal_max_log_gap, c.max_discounted_wealth, c.passed, c.status)
                for c in self.comparisons]
        return pd.DataFrame(rows, columns=["short", "long", "max_log_difference", "welfare_difference_pp",
                                           "terminal_max_log_gap", "max_discounted_wealth", "passed", "status"])

    def summary(self) -> str:
        """One line with the horizons, the status and the accepted horizon."""
        return (f"Horizon ladder {self.horizons}: status {self.status}; accepted horizon "
                f"{self.accepted_horizon}; {len(self.comparisons)} comparisons.")
