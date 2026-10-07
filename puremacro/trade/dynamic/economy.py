"""Accounted, sector-capital dynamics at the source table's native resolution.

Ported from the IO engine's ``dynamic_model/native_economy.py``. The
production and final-use nests are fixed coefficient (Leontief). Value added
is a benchmark-normalized Cobb-Douglas aggregate of sector-specific installed
capital and nationally mobile, inelastically supplied labour. Countries are
financially autarkic; benchmark net exports are financed by explicitly balanced
lump-sum transfers indexed to the last country's wage. No world bond or
endogenous inventory is implicit in these equations.

Model (cell ``j`` in country ``c(j)``, date ``t``; NATIVE_MODEL.md):

.. code-block:: text

    v[j,t] = w[c(j),t]**(1-alpha[j]) * (R[j,t]/R0[j])**alpha[j]
    (1-tax[j]) p[j,t] = sum_i A[i,j] (1+tau[i,c(j),t]) p[i,t] + b[j] v[j,t]
    L[j,t] = (1-alpha[j]) b[j] v[j,t] y[j,t] / w[c(j),t]
    K[j,t] = alpha[j] b[j] v[j,t] y[j,t] / R[j,t]
    y = A y + omegaC C + omegaI I_national
    PC[c] = sum_i (1+tauC[i,c]) omegaCtax[i,c] p[i] / (1-tC[c])  (+ exempt X part)
    PI[c] = sum_i (1+tauI[i,c]) omegaI[i,c] p[i] / (1-tI[c])
    K[j,t+1] = (1-delta) K[j,t] + K[j,t] Phi(x[j,t]),  x = I/K
    Phi(x) = x - phi/2 (x-delta)**2
    q[j,t] = PI[c(j),t] / Phi'(x[j,t])
    q[j,t] = m[c,t+1] (R[j,t+1] + q[j,t+1] G[j,t+1]),  G = 1-delta+Phi(x)-x Phi'(x)
    m[c,t+1] = beta (C[c,t+1]/C[c,t])**(-sigma) PC[c,t]/PC[c,t+1]
    PC C + PI I_nat + XN0 w_last = w L0 + sum_j R K + taxes + tariff receipts

The retained state per date is ``z_t = [log w (C), log C/C0 (C), log K_{t+1}/K0 (N)]``.
Goods clearing, capital-market clearing and installation are eliminated
exactly (NATIVE_NUMERICS.md). The residual is: C labour-clearing equations,
C national budgets with the one belonging to the largest budget scale replaced
by the wage numeraire ``log w_last = 0``, and N capital Euler equations. The
omitted budget is checked independently by :meth:`DynamicEconomy.certificate`.

Tariffs are ad-valorem rates ``r > -1`` on the seller's basic-price value by
source cell and destination country (``tau = 1 + r``; ``r = 0`` is free trade
and negative values are import subsidies, as in the IO engine's
``NativePolicy``); domestic transactions
carry ``tau = 1`` and residents' purchases abroad are exempt. Separate
intermediate, consumption and investment schedules are allowed. Only N x C
rate tables are stored, never an N x N tariff tensor (destination-sector
specific duties are not representable).

Limitation quoted from NATIVE_MODEL.md: "This is an interior investment model;
it does not claim to solve an investment-irreversibility complementarity problem."
"""
from __future__ import annotations

from dataclasses import dataclass, field
import hashlib
from numbers import Integral
from typing import Any, Sequence

import numpy as np
import pandas as pd
from scipy import linalg, sparse
from scipy.sparse import linalg as spla

from puremacro.reports import df_to_latex, df_to_markdown, df_to_typst
from ._results import DynamicCalibration, DynamicSolveError, EconomicDomainError

__all__ = ["DynamicTariff", "tariff_path", "DynamicEconomy", "EconomicDomainError"]

_USES = ("intermediate", "consumption", "investment")


def _national(x: np.ndarray, country: np.ndarray, count: int) -> np.ndarray:
    return np.bincount(country, weights=x, minlength=count)


def _maxabs(x: np.ndarray) -> float:
    return float(np.max(np.abs(x), initial=0.0))


def _rate_table(value: Any, shape: tuple[int, int], name: str) -> np.ndarray:
    table = np.array(value, dtype=float, copy=True)
    if table.shape != shape:
        raise ValueError(f"{name} must have shape {shape}, got {table.shape}")
    if not np.all(np.isfinite(table)) or np.any(table <= -1):
        raise ValueError(f"{name} must be finite and exceed -1")
    table.flags.writeable = False
    return table


_TARIFF_TABLES = ("rates", "consumption_rates", "investment_rates")


def _content_fingerprint(tables: Sequence[np.ndarray]) -> str:
    """sha256 of the three rate tables (the IO ``NativePolicy`` digest)."""
    digest = hashlib.sha256()
    for table in tables:
        digest.update(table.tobytes())
    return digest.hexdigest()


# ---------------------------------------------------------------------------
# Tariff policy
# ---------------------------------------------------------------------------
@dataclass(frozen=True)
class DynamicTariff:
    """Additional ad-valorem tariff rates by source cell and destination country.

    ``rates`` applies to intermediate deliveries; ``consumption_rates`` and
    ``investment_rates`` to the two final uses (they default to ``rates`` in
    :meth:`build`, and a ``None`` passed to the constructor also means
    ``rates``). Rates must exceed ``-1`` (``0`` is free trade, negative values
    are import subsidies). Purchases abroad are exempt. Build instances with
    the validating classmethods :meth:`build`, :meth:`zero`, :meth:`uniform`
    or :meth:`from_rates`.

    ``fingerprint`` is not an argument: it is always derived from the three
    stored tables (sha256 of their bytes, as in the IO engine's
    ``NativePolicy``), so ``dataclasses.replace(policy, rates=...)`` yields a
    new fingerprint and can never be solved as the earlier policy. The
    economy shares one price-network factorisation among policies with equal
    fingerprints. The derivation never raises (library rule): tables that
    are not a finite, two-dimensional, equally shaped set of rates above
    ``-1`` get an empty fingerprint, and :class:`DynamicEconomy` then refuses
    the policy with a ``ValueError`` (use :meth:`build` to see why).
    """

    rates: np.ndarray
    consumption_rates: np.ndarray
    investment_rates: np.ndarray
    label: str = "policy"
    fingerprint: str = field(init=False, repr=False, compare=False, default="")
    metadata: dict[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        # Never raises: validation lives in build(). This only stores
        # read-only float copies and derives the fingerprint from them, so a
        # replaced or directly constructed policy cannot inherit a stale key.
        fingerprint = ""
        try:
            common = np.array(self.rates, dtype=float, copy=True)
            tables = [common]
            for name in _TARIFF_TABLES[1:]:
                value = getattr(self, name)
                tables.append(common.copy() if value is None else np.array(value, dtype=float, copy=True))
            for name, table in zip(_TARIFF_TABLES, tables):
                table.flags.writeable = False
                object.__setattr__(self, name, table)
            if (common.ndim == 2 and all(t.shape == common.shape for t in tables)
                    and all(np.all(np.isfinite(t)) and not np.any(t <= -1) for t in tables)):
                fingerprint = _content_fingerprint(tables)
        except Exception:  # noqa: BLE001 - never raise here; an empty key is refused on use
            fingerprint = ""
        object.__setattr__(self, "fingerprint", fingerprint)

    @classmethod
    def build(cls, rates: Any, *, consumption_rates: Any | None = None,
              investment_rates: Any | None = None, label: str = "policy",
              metadata: dict[str, Any] | None = None) -> "DynamicTariff":
        """Validate (finite, two-dimensional, > -1, equal shapes); the fingerprint follows."""
        common = np.array(rates, dtype=float, copy=True)
        if common.ndim != 2:
            raise ValueError("Tariff rates must be a two-dimensional (cells x countries) table")
        shape = common.shape
        tables = []
        for name, value in (("rates", common), ("consumption_rates", consumption_rates),
                            ("investment_rates", investment_rates)):
            tables.append(_rate_table(common if value is None else value, shape, name))
        return cls(rates=tables[0], consumption_rates=tables[1], investment_rates=tables[2],
                   label=str(label), metadata=dict(metadata or {}))

    @classmethod
    def zero(cls, calibration: DynamicCalibration) -> "DynamicTariff":
        """The baseline: no additional duties."""
        return cls.build(np.zeros((calibration.n_cells, calibration.n_countries)), label="baseline")

    @classmethod
    def uniform(cls, calibration: DynamicCalibration, importer: str | int, rate: float, *,
                exporters: Sequence[str] | None = None,
                sectors: Sequence[str] | str | None = None,
                uses: Sequence[str] = _USES, label: str | None = None) -> "DynamicTariff":
        """A uniform duty ``rate`` levied by ``importer`` on foreign source cells.

        ``exporters`` restricts the taxed source countries (default: all
        foreign); ``sectors`` restricts the taxed source sectors (a sequence of
        sector codes, ``"merchandise"`` for the calibration's goods mask, or
        ``None`` for all); ``uses`` selects which of ``intermediate``,
        ``consumption`` and ``investment`` deliveries pay the duty (the others
        stay at zero). Domestic deliveries never pay. A selection with no
        foreign source cell (for example ``exporters=(importer,)`` or an
        all-false merchandise mask) raises ``ValueError``, as in the IO driver,
        instead of returning a zero policy under a tariff label.
        """
        if isinstance(importer, str):
            if importer not in calibration.countries:
                raise ValueError(f"Unknown importer {importer!r}")
            dest = calibration.countries.index(importer)
        elif isinstance(importer, Integral) and not isinstance(importer, bool):
            dest = int(importer)
            if not 0 <= dest < calibration.n_countries:
                raise ValueError("importer index is out of range")
        else:
            raise TypeError("importer must be a country code or index")
        rate = float(rate)
        if not np.isfinite(rate) or rate <= -1:
            raise ValueError("rate must be finite and exceed -1")
        chosen_uses = tuple(uses)
        if not chosen_uses or any(u not in _USES for u in chosen_uses):
            raise ValueError(f"uses must be a nonempty subset of {_USES}")
        rows = calibration.country != dest
        if exporters is not None:
            codes = tuple(exporters)
            unknown = set(codes) - set(calibration.countries)
            if unknown:
                raise ValueError(f"Unknown exporters {sorted(unknown)}")
            wanted = np.array([calibration.countries.index(c) for c in codes], dtype=int)
            rows &= np.isin(calibration.country, wanted)
        if sectors is not None:
            if isinstance(sectors, str):
                if sectors != "merchandise":
                    raise ValueError("sectors must be a sequence of sector codes, 'merchandise' or None")
                if calibration.merchandise_mask is None:
                    raise ValueError("The calibration carries no merchandise mask")
                sector_mask = np.asarray(calibration.merchandise_mask, dtype=bool)
            else:
                codes = tuple(sectors)
                unknown = set(codes) - set(calibration.sectors)
                if unknown:
                    raise ValueError(f"Unknown sectors {sorted(unknown)}")
                sector_mask = np.array([s in codes for s in calibration.sectors], dtype=bool)
            rows &= sector_mask[calibration.cell_sector]
        if not rows.any():
            raise ValueError(f"The tariff selection for {calibration.countries[dest]} contains no foreign "
                             f"source cells (exporters={exporters}, sectors={sectors})")
        table = np.zeros((calibration.n_cells, calibration.n_countries))
        table[rows, dest] = rate
        zero = np.zeros_like(table)
        parts = {u: (table if u in chosen_uses else zero) for u in _USES}
        name = label if label is not None else f"{calibration.countries[dest]} uniform {100 * rate:g}%"
        return cls.build(parts["intermediate"], consumption_rates=parts["consumption"],
                         investment_rates=parts["investment"], label=name,
                         metadata={"importer": calibration.countries[dest], "rate": rate,
                                   "exporters": None if exporters is None else list(exporters),
                                   "sectors": sectors if isinstance(sectors, str) or sectors is None else list(sectors),
                                   "uses": list(chosen_uses)})

    @classmethod
    def from_rates(cls, calibration: DynamicCalibration, rates: Any, *,
                   consumption_rates: Any | None = None, investment_rates: Any | None = None,
                   label: str = "policy") -> "DynamicTariff":
        """Explicit rate tables of shape ``(N_active, C)`` or full-table ``(M, C)``.

        Full-table rows are subset to the calibration's active cells. Domestic
        entries (source country equal to destination) must be zero.
        """
        tables = []
        for name, value in (("rates", rates), ("consumption_rates", consumption_rates),
                            ("investment_rates", investment_rates)):
            if value is None:
                tables.append(None)
                continue
            table = np.array(value, dtype=float, copy=True)
            full = calibration.n_countries * calibration.n_sectors
            if table.shape == (full, calibration.n_countries) and full != calibration.n_cells:
                table = table[calibration.active_indices]
            if table.shape != (calibration.n_cells, calibration.n_countries):
                raise ValueError(f"{name} must have shape {(calibration.n_cells, calibration.n_countries)} "
                                 f"or {(full, calibration.n_countries)}")
            domestic = table[np.arange(calibration.n_cells), calibration.country]
            if np.any(domestic != 0):
                raise ValueError(f"{name} has nonzero domestic entries; domestic transactions never pay duties")
            tables.append(table)
        return cls.build(tables[0], consumption_rates=tables[1], investment_rates=tables[2], label=label)

    @property
    def n_cells(self) -> int:
        """Source cells (rows of the rate tables)."""
        return int(self.rates.shape[0])

    @property
    def n_countries(self) -> int:
        """Destination countries (columns of the rate tables)."""
        return int(self.rates.shape[1])

    def is_zero(self) -> bool:
        """True when all three rate tables are identically zero (free trade)."""
        return not (np.any(self.rates) or np.any(self.consumption_rates) or np.any(self.investment_rates))

    def to_dataframe(self, calibration: DynamicCalibration | None = None) -> pd.DataFrame:
        """Long table of the nonzero entries (source cell, destination, three rates)."""
        mask = (self.rates != 0) | (self.consumption_rates != 0) | (self.investment_rates != 0)
        rows, cols = np.nonzero(mask)
        if calibration is not None and calibration.n_cells == self.n_cells:
            labels, countries = calibration.cell_labels, calibration.countries
            source = [labels[i] for i in rows]
            dest = [countries[j] for j in cols]
        else:
            source, dest = rows.tolist(), cols.tolist()
        return pd.DataFrame({"source": source, "destination": dest,
                             "intermediate": self.rates[rows, cols],
                             "consumption": self.consumption_rates[rows, cols],
                             "investment": self.investment_rates[rows, cols]})

    def to_markdown(self, calibration: DynamicCalibration | None = None) -> str:
        """``to_dataframe(calibration)`` rendered as a Markdown table."""
        return df_to_markdown(self.to_dataframe(calibration))

    def to_latex(self, calibration: DynamicCalibration | None = None) -> str:
        """``to_dataframe(calibration)`` rendered as a LaTeX table."""
        return df_to_latex(self.to_dataframe(calibration))

    def to_typst(self, calibration: DynamicCalibration | None = None) -> str:
        """``to_dataframe(calibration)`` rendered as a Typst table."""
        return df_to_typst(self.to_dataframe(calibration))


def tariff_path(baseline: DynamicTariff, shock: DynamicTariff, *, horizon: int,
                announcement: int = 0, duration: int | None = None) -> tuple[DynamicTariff, ...]:
    """Date-by-date policy schedule for :func:`solve_dynamic_transition`.

    ``baseline`` applies at dates ``t < announcement``; ``shock`` from
    ``announcement`` for ``duration`` dates (permanently when ``None``);
    ``baseline`` again afterwards. Announced at date 0, so a positive
    ``announcement`` is an anticipated policy. As in the IO driver, the
    horizon must extend past the whole announced schedule
    (``announcement < horizon`` and ``announcement + duration < horizon``).
    """
    for name, policy in (("baseline", baseline), ("shock", shock)):
        if not isinstance(policy, DynamicTariff):
            raise TypeError(f"{name} must be a DynamicTariff")
    if baseline.rates.shape != shock.rates.shape:
        raise ValueError("baseline and shock must have the same dimensions")
    if not isinstance(horizon, Integral) or isinstance(horizon, bool) or horizon < 1:
        raise ValueError("horizon must be a positive integer")
    if not isinstance(announcement, Integral) or isinstance(announcement, bool) or announcement < 0:
        raise ValueError("announcement must be a nonnegative integer")
    if announcement >= horizon:
        raise ValueError("The horizon must extend past the announcement date")
    if duration is not None:
        if not isinstance(duration, Integral) or isinstance(duration, bool) or duration < 1:
            raise ValueError("duration must be a positive integer or None")
        if announcement + duration >= horizon:
            raise ValueError("The horizon must extend past the full announced policy schedule")
    return tuple(shock if t >= announcement and (duration is None or t < announcement + duration)
                 else baseline for t in range(int(horizon)))


# ---------------------------------------------------------------------------
# Price network
# ---------------------------------------------------------------------------
class _PriceNetwork:
    """Tariff-adjusted unit-cost system ``[diag(1-tax) - (A .* (1+tau_dest))^T] p = b v``.

    One factorisation per distinct policy: dense LU when ``n > direct_threshold``
    and the input matrix has density above .25, sparse LU when
    ``n <= direct_threshold``, otherwise preconditioned BiCGSTAB with a GMRES
    fallback at ``tolerance``. An iterative solve whose relative residual
    exceeds ``30 * tolerance`` raises :class:`DynamicSolveError` with
    ``location={"block": "price_network", "backend": ..., ...}``.
    """

    def __init__(self, cal: DynamicCalibration, policy: DynamicTariff,
                 tolerance: float, direct_threshold: int):
        self.cal, self.policy, self.tolerance = cal, policy, tolerance
        # Other policy objects verified to carry identical tables (strong
        # references keep their ids from being reused).
        self.aliases: dict[int, DynamicTariff] = {}
        self.denominator = 1. - cal.tax
        # These baskets depend on policy, not on dates or Newton directions.
        self.tariffCweights = policy.consumption_rates * cal.omegaCtax
        self.tariffIweights = policy.investment_rates * cal.omegaI
        self.tariffOtherweights = policy.consumption_rates * cal.qOther
        self.cweights = cal.omegaC + self.tariffCweights
        self.iweights = cal.omegaI + self.tariffIweights
        self.otherweights = cal.qOther + self.tariffOtherweights
        self._final_weights = {
            "consumption": self.cweights,
            "investment": self.iweights,
            "consumption_tariff": self.tariffCweights,
            "investment_tariff": self.tariffIweights,
            "other": self.otherweights,
            "other_tariff": self.tariffOtherweights,
        }
        self._nonzero_final_weights = {
            name: weights for name, weights in self._final_weights.items() if np.any(weights)
        }
        self.AT = cal.A.T.tocsr()
        self.parts = []
        for c in range(cal.n_countries):
            if np.any(policy.rates[:, c]):
                cols = np.flatnonzero(cal.country == c)
                self.parts.append((c, cols, cal.A[:, cols].T.tocsr(), policy.rates[:, c]))
        n = cal.n_cells
        self.op = spla.LinearOperator((n, n), matvec=self.matvec, dtype=float)
        self.factor = None
        self.dense_factor = None
        self.backend = "iterative"
        density = cal.A.nnz / (n * n)
        if n > direct_threshold and density > .25:
            matrix = np.asarray(cal.A.T.toarray(), order="F")
            for c in range(cal.n_countries):
                rows = np.flatnonzero(cal.country == c)
                matrix[rows, :] *= 1. + policy.rates[:, c][None, :]
            matrix *= -1.
            matrix[np.diag_indices(n)] += self.denominator
            self.dense_factor = linalg.lu_factor(matrix, overwrite_a=True, check_finite=False)
            self.backend = "dense_lu"
        elif n <= direct_threshold:
            coo = cal.A.tocoo()
            adjusted = sparse.coo_matrix((coo.data * (1. + policy.rates[coo.row, cal.country[coo.col]]),
                                          (coo.row, coo.col)), shape=coo.shape).tocsc()
            self.factor = spla.splu(sparse.diags(self.denominator).tocsc() - adjusted.T.tocsc())
            self.backend = "sparse_lu"
        diagonal = self.denominator - cal.A.diagonal()
        self.preconditioner = spla.LinearOperator((n, n), matvec=lambda x: x / diagonal, dtype=float)

    def tariff_unit_cost(self, prices: np.ndarray) -> np.ndarray:
        result = np.zeros_like(prices, dtype=float)
        for _, cols, block, rates in self.parts:
            result[cols] = block @ (rates * prices if prices.ndim == 1 else rates[:, None] * prices)
        return result

    def final_costs(self, prices: np.ndarray) -> dict[str, np.ndarray]:
        """Contract policy baskets, accepting source cells by date prices."""
        prices = np.asarray(prices)
        left = prices if prices.ndim == 1 else prices.T
        shape = (self.cal.n_countries,) if prices.ndim == 1 else (prices.shape[1], self.cal.n_countries)
        return {
            name: left @ weights if name in self._nonzero_final_weights else np.zeros(shape)
            for name, weights in self._final_weights.items()
        }

    def matvec(self, prices: np.ndarray) -> np.ndarray:
        return self.denominator * prices - self.AT @ prices - self.tariff_unit_cost(prices)

    def solve(self, rhs: np.ndarray) -> np.ndarray:
        if self.dense_factor is not None:
            return linalg.lu_solve(self.dense_factor, np.asarray(rhs), check_finite=False)
        if self.factor is not None:
            return np.asarray(self.factor.solve(np.asarray(rhs)))
        rhs = np.asarray(rhs, dtype=float)
        if rhs.ndim == 2:
            return np.column_stack([self.solve(rhs[:, j]) for j in range(rhs.shape[1])])
        if not np.any(rhs):
            return np.zeros_like(rhs)
        solution, status = spla.bicgstab(self.op, rhs, M=self.preconditioner,
                                         rtol=self.tolerance, atol=0., maxiter=1200)
        error = _maxabs(self.matvec(solution) - rhs) / max(_maxabs(rhs), 1e-300)
        if status != 0 or not np.isfinite(error) or error > 30 * self.tolerance:
            solution, status = spla.gmres(self.op, rhs, M=self.preconditioner,
                                          rtol=self.tolerance, atol=0., restart=60, maxiter=100)
            error = _maxabs(self.matvec(solution) - rhs) / max(_maxabs(rhs), 1e-300)
        if status != 0 or not np.isfinite(error) or error > 30 * self.tolerance:
            raise DynamicSolveError(f"Price-network solve failed: status={status}, residual={error:.3g}",
                                    residual=float(error),
                                    location={"block": "price_network", "backend": self.backend,
                                              "krylov_status": int(status), "relative_residual": float(error),
                                              "tolerance": float(self.tolerance)})
        return solution


# ---------------------------------------------------------------------------
# Economy
# ---------------------------------------------------------------------------
class DynamicEconomy:
    """Sector-specific capital economy with exact static elimination and analytic JVPs.

    Parameters
    ----------
    calibration : DynamicCalibration
    adjustment_cost : float
        ``phi >= 0`` in ``Phi(x) = x - phi/2 (x - delta)**2``; zero gives
        frictionless capital with ``q = PI``.
    risk_aversion : float
        CRRA ``sigma > 0`` (reciprocal of the intertemporal elasticity).
    network_tolerance : float
        Relative tolerance of the iterative price and goods solves used only
        when the table is both large and sparse.
    direct_threshold : int
        Cells up to this count use sparse LU factorisations.

    Raises
    ------
    TypeError, ValueError
        Invalid calibration or parameters.
    DynamicSolveError
        The cached Leontief goods responses do not converge (iterative
        backend) or do not reproduce benchmark output to ``1e-7``. The price
        network raises the same error (``location["block"] ==
        "price_network"``) when an iterative price solve misses its tolerance.

    Attributes
    ----------
    n, nc, n_vars : int
        Active cells, countries and retained unknowns per date ``N + 2C``.
    w_slice, c_slice, k_slice : slice
        Positions of ``log w``, ``log C/C0`` and ``log K_next/K0`` in ``z``.
    numeraire_index : int
        The last country's wage is the numeraire.
    omitted_budget_index : int
        The budget replaced by the numeraire equation (largest budget scale;
        Walras' law makes it redundant, the certificate checks it anyway).
    YC, YI : np.ndarray, shape (N, C)
        Cached Leontief responses ``(I - A)^-1 omegaC`` and ``(I - A)^-1 omegaI``.

    Notes
    -----
    The last country of the registry is both the wage numeraire and the
    anchor of the benchmark net-export transfers (``XN0 * w_last``).
    Reordering countries therefore changes the closure and the real
    allocation whenever relative wages move (by about the change in the
    anchor's relative wage times ``XN0``), exactly as in the IO engine;
    reordering sectors is a pure relabelling.

    Three caches live on the instance and are keyed by policy fingerprint
    only: ``_networks`` (one price-network factorisation per policy),
    ``_stability_cache`` (the determinacy report that gates
    :func:`solve_dynamic_transition`, one per terminal policy) and the
    capital preconditioner. After mutating ``omitted_budget_index`` or the
    adjustment parameters, call :meth:`clear_caches`.
    """

    def __init__(self, calibration: DynamicCalibration, *, adjustment_cost: float = 2.,
                 risk_aversion: float = 2., network_tolerance: float = 2e-12,
                 direct_threshold: int = 600):
        if not isinstance(calibration, DynamicCalibration):
            raise TypeError("calibration must be a DynamicCalibration")
        if not np.isfinite(adjustment_cost) or adjustment_cost < 0 or not np.isfinite(risk_aversion) or risk_aversion <= 0:
            raise ValueError("Adjustment cost must be nonnegative; risk aversion positive")
        self.calibration = self.p = calibration
        self.adjustment_cost, self.risk_aversion = float(adjustment_cost), float(risk_aversion)
        self.network_tolerance, self.direct_threshold = float(network_tolerance), int(direct_threshold)
        self.n, self.nc = calibration.n_cells, calibration.n_countries
        self.numeraire_index = self.nc - 1
        # Walras' law permits any one national budget to be omitted.  Omitting
        # the largest avoids magnifying world rounding errors by the GDP of a
        # tiny residual region.  The wage numeraire and transfer anchor remain
        # unchanged, so this selection does not change the economic closure.
        self.omitted_budget_index = int(np.argmax(calibration.budget_scale))
        self.n_vars = self.n + 2 * self.nc
        self.w_slice = slice(0, self.nc)
        self.c_slice = slice(self.nc, 2 * self.nc)
        self.k_slice = slice(2 * self.nc, self.n_vars)
        self.baseline = np.zeros(self.n_vars)
        self.baseline_policy = DynamicTariff.zero(calibration)
        self.Inational0 = calibration.Inational0
        self._networks: dict[str, _PriceNetwork] = {}
        self._stability_cache: dict[str, Any] = {}
        self.YC, self.YI, self.YOther = self._goods_responses()
        base_y = self.YC @ calibration.C0 + self.YI @ calibration.Inational0 + self.YOther
        output_error = _maxabs((base_y - calibration.y0) / np.maximum(calibration.y0, 1e-300))
        if output_error > 1e-7:
            raise DynamicSolveError("Leontief response calibration does not reproduce benchmark output",
                                    residual=output_error,
                                    location={"block": "goods_response", "backend": self.goods_backend,
                                              "relative_output_error": output_error})

    # -- setup --------------------------------------------------------------
    def _goods_responses(self) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        d = self.p
        rhs = np.column_stack((d.omegaC, d.omegaI, d.qOther.sum(axis=1)))
        density = d.A.nnz / (self.n * self.n)
        if self.n > self.direct_threshold and density > .25:
            matrix = np.asarray(-d.A.toarray(), order="F")
            matrix[np.diag_indices(self.n)] += 1.
            factor = linalg.lu_factor(matrix, overwrite_a=True, check_finite=False)
            response = linalg.lu_solve(factor, rhs, check_finite=False)
            self.goods_backend = "dense_lu_batch"
        elif self.n <= self.direct_threshold:
            lu = spla.splu(sparse.eye(self.n, format="csc") - d.A.tocsc())
            response = lu.solve(rhs)
            self.goods_backend = "sparse_lu_batch"
        else:
            # All demand-basket right-hand sides are solved together and reused
            # at every date and Newton iteration.
            response = rhs.copy()
            for _ in range(1600):
                update = rhs + d.A @ response
                error = np.max(np.abs(update - response) / np.maximum(np.abs(update), 1e-20))
                response = update
                if error <= self.network_tolerance:
                    break
            else:
                raise DynamicSolveError("Leontief goods response iteration did not converge",
                                        residual=float(error),
                                        location={"block": "goods_response", "backend": "sparse_iteration_batch",
                                                  "iterations": 1600, "relative_change": float(error)})
            residual = response - d.A @ response - rhs
            if _maxabs(residual) > 100 * self.network_tolerance * max(_maxabs(rhs), 1e-300):
                raise DynamicSolveError("Leontief goods response residual is too large",
                                        residual=_maxabs(residual),
                                        location={"block": "goods_response", "backend": "sparse_iteration_batch",
                                                  "residual": _maxabs(residual)})
            self.goods_backend = "sparse_iteration_batch"
        return response[:, :self.nc], response[:, self.nc:2 * self.nc], response[:, -1]

    def _network(self, policy: DynamicTariff | None) -> _PriceNetwork:
        policy = self.baseline_policy if policy is None else policy
        if not isinstance(policy, DynamicTariff):
            raise TypeError("policy must be a DynamicTariff (use DynamicTariff.build/zero/uniform/from_rates)")
        key = policy.fingerprint
        if not key:
            raise ValueError("DynamicTariff has no fingerprint: its rate tables are not a finite, "
                             "two-dimensional, equally shaped set of rates above -1 (build it with "
                             "DynamicTariff.build to see which check fails)")
        if policy.rates.shape != (self.n, self.nc):
            raise ValueError(f"Policy rates need shape {(self.n, self.nc)}")
        network = self._networks.get(key)
        if network is None:
            network = self._networks[key] = _PriceNetwork(self.p, policy, self.network_tolerance,
                                                          self.direct_threshold)
        elif policy is not network.policy and id(policy) not in network.aliases:
            # Defence in depth: the fingerprint is derived from the tables, but
            # a forged or mutated one must never select another policy's prices.
            if not all(np.array_equal(getattr(policy, name), getattr(network.policy, name))
                       for name in _TARIFF_TABLES):
                raise ValueError(f"DynamicTariff {policy.label!r} carries the fingerprint of a different "
                                 "policy; its rate tables were altered after construction")
            network.aliases[id(policy)] = policy
        return network

    def zero_policy(self) -> DynamicTariff:
        """The baseline policy (no additional duties)."""
        return self.baseline_policy

    def clear_caches(self) -> None:
        """Drop the cached price networks, determinacy reports and capital preconditioner."""
        self._networks.clear()
        self._stability_cache.clear()
        if hasattr(self, "_capital_preconditioner"):
            del self._capital_preconditioner

    # -- one date -----------------------------------------------------------
    def _install(self, K: np.ndarray, Knext: np.ndarray,
                 log_growth: np.ndarray | None = None) -> tuple[np.ndarray, np.ndarray]:
        growth = Knext / K - 1. if log_growth is None else np.expm1(log_growth)
        phi = self.adjustment_cost
        if phi == 0:
            x, derivative = self.p.delta + growth, np.ones(self.n)
        else:
            discriminant = 1. - 2. * phi * growth
            if np.any(discriminant <= 0):
                raise EconomicDomainError("Capital growth exceeds the increasing installation branch")
            derivative = np.sqrt(discriminant)
            x = self.p.delta + 2. * growth / (1. + derivative)
        if np.any(x <= 0) or not np.all(np.isfinite(x)):
            raise EconomicDomainError("The interior investment model requires strictly positive investment")
        return x, derivative

    def state(self, z: np.ndarray, zlag: np.ndarray,
              policy: DynamicTariff | None = None, *,
              _price: np.ndarray | None = None,
              _final_costs: dict[str, np.ndarray] | None = None,
              _tariff_unit: np.ndarray | None = None) -> dict[str, np.ndarray]:
        """Full date state from ``z_t`` and ``z_{t-1}`` (which supplies ``K_t``).

        Returns a dict with cell arrays (``K, Knext, I, x, y, va, R, labor_j,
        p, q``) and country arrays (``w, C, Inational, PC, PI, lam, TR, PT,
        EV, transfer, income, budget`` and their components).
        """
        d = self.p
        z, zlag = np.asarray(z, dtype=float), np.asarray(zlag, dtype=float)
        if z.shape != (self.n_vars,) or zlag.shape != z.shape or not np.all(np.isfinite(z + zlag)):
            raise EconomicDomainError("Invalid date-state vector")
        if _maxabs(z) > 200 or _maxabs(zlag) > 200:
            raise EconomicDomainError("State log levels exceed the numerical domain")
        w = np.exp(z[self.w_slice])
        consumption_change = d.C0 * np.expm1(z[self.c_slice])
        C = d.C0 + consumption_change
        K, Knext = d.K0 * np.exp(zlag[self.k_slice]), d.K0 * np.exp(z[self.k_slice])
        x, installation_derivative = self._install(K, Knext, z[self.k_slice] - zlag[self.k_slice])
        I = K * x
        investment_change = d.I0 * np.expm1(zlag[self.k_slice]) + K * (x - d.delta)
        national_investment_change = _national(investment_change, d.country, self.nc)
        Inational = self.Inational0 + national_investment_change
        output_change = self.YC @ consumption_change + self.YI @ national_investment_change
        # Centering is the same Leontief system in deviations.  It reproduces
        # observed benchmark quantities exactly and avoids magnifying a large
        # nominal-level matrix-summation error in almost capital-only sectors.
        y = d.y0 + output_change
        if np.any(y <= 0) or not np.all(np.isfinite(y)):
            raise EconomicDomainError("Nonpositive gross output")
        log_output = np.log1p(output_change / d.y0)
        log_ratio = log_output - zlag[self.k_slice]
        log_va = z[self.w_slice][d.country] + d.alpha / (1. - d.alpha) * log_ratio
        log_rental = z[self.w_slice][d.country] + log_ratio / (1. - d.alpha)
        if _maxabs(log_va) > 600 or _maxabs(log_rental) > 600:
            raise EconomicDomainError("Factor-price overflow outside admissible trial region")
        va = np.exp(log_va)
        R = d.R0 * np.exp(log_rental)
        labor_j = d.L0sector * np.exp(log_output + d.alpha / (1. - d.alpha) * log_ratio)
        Ldem = _national(labor_j, d.country, self.nc)
        network = self._network(policy)
        policy = network.policy
        p = network.solve(d.b * va) if _price is None else _price
        if np.any(p <= 0) or not np.all(np.isfinite(p)):
            raise EconomicDomainError("Nonpositive producer prices: the tariff-adjusted unit-cost system is "
                                      "not productive at this state (Hawkins-Simon condition fails under "
                                      "this policy)")
        final_costs = network.final_costs(p) if _final_costs is None else _final_costs
        PC = final_costs["consumption"] / (1. - d.tC)
        PI = final_costs["investment"] / (1. - d.tI)
        if np.any(PC <= 0) or np.any(PI <= 0):
            raise EconomicDomainError("Nonpositive final purchaser prices")
        q = PI[d.country] / installation_derivative
        # Multiplying each country's utility by C0**sigma leaves its choices
        # unchanged and keeps marginal utility well scaled across monetary units.
        lam = np.exp(-self.risk_aversion * z[self.c_slice]) / PC
        tariff_unit = network.tariff_unit_cost(p) if _tariff_unit is None else _tariff_unit
        TR_intermediate = _national(y * tariff_unit, d.country, self.nc)
        TR_consumption = final_costs["consumption_tariff"] * C
        TR_investment = final_costs["investment_tariff"] * Inational
        TR_other = final_costs["other_tariff"]
        TR = TR_intermediate + TR_consumption + TR_investment + TR_other
        PT_production = _national(d.tax * p * y, d.country, self.nc)
        PT_consumption, PT_investment = d.tC * PC * C, d.tI * PI * Inational
        PT_other = d.TV * w[-1]
        PT = PT_production + PT_consumption + PT_investment + PT_other
        EV = final_costs["other"] + PT_other
        transfer = d.XN0 * w[-1]
        income = w * d.L0 + _national(R * K, d.country, self.nc) + PT + TR
        budget = PC * C + PI * Inational + EV + transfer - income
        continuation = 1. - d.delta + .5 * self.adjustment_cost * (x * x - d.delta * d.delta)
        return dict(w=w, C=C, K=K, Knext=Knext, I=I, Inational=Inational, x=x,
                    installation_derivative=installation_derivative, continuation=continuation,
                    y=y, va=va, R=R, labor_j=labor_j, Ldem=Ldem, p=p, PC=PC, PI=PI,
                    q=q, lam=lam, TR=TR, TR_intermediate=TR_intermediate,
                    TR_consumption=TR_consumption, TR_investment=TR_investment,
                    TR_other=TR_other, PT=PT, PT_production=PT_production,
                    PT_consumption=PT_consumption, PT_investment=PT_investment,
                    PT_other=PT_other, EV=EV, transfer=transfer, income=income,
                    budget=budget, tariff_unit=tariff_unit,
                    tariff_consumption_unit=final_costs["consumption_tariff"],
                    tariff_investment_unit=final_costs["investment_tariff"])

    def states(self, zs: np.ndarray, zlags: np.ndarray,
               policies: Sequence[DynamicTariff]) -> list[dict]:
        """Evaluate dates together using one batched price solve per distinct policy."""
        zs, zlags = np.asarray(zs), np.asarray(zlags)
        if zs.ndim != 2 or zs.shape != zlags.shape or len(zs) != len(policies):
            raise ValueError("Batched states need matching date arrays and policies")
        d = self.p
        rhs = np.empty((self.n, len(zs)))
        groups: dict[str, list[int]] = {}
        for t, (z, lag, policy) in enumerate(zip(zs, zlags, policies)):
            if not np.all(np.isfinite(z + lag)) or _maxabs(z) > 200 or _maxabs(lag) > 200:
                raise EconomicDomainError("Invalid batched state log levels")
            consumption_change = d.C0 * np.expm1(z[self.c_slice])
            K, Kn = d.K0 * np.exp(lag[self.k_slice]), d.K0 * np.exp(z[self.k_slice])
            x, _ = self._install(K, Kn, z[self.k_slice] - lag[self.k_slice])
            investment_change = d.I0 * np.expm1(lag[self.k_slice]) + K * (x - d.delta)
            ni_change = _national(investment_change, d.country, self.nc)
            output_change = self.YC @ consumption_change + self.YI @ ni_change
            y = d.y0 + output_change
            if np.any(y <= 0):
                raise EconomicDomainError("Nonpositive batched output")
            log_va = (z[self.w_slice][d.country] + d.alpha / (1. - d.alpha)
                      * (np.log1p(output_change / d.y0) - lag[self.k_slice]))
            if _maxabs(log_va) > 600:
                raise EconomicDomainError("Batched factor-price overflow")
            rhs[:, t] = d.b * np.exp(log_va)
            groups.setdefault(self._network(policy).policy.fingerprint, []).append(t)
        prices = np.empty_like(rhs)
        tariff_units = np.empty_like(rhs)
        final_costs = {name: np.empty((len(zs), self.nc)) for name in
                       ("consumption", "investment", "consumption_tariff",
                        "investment_tariff", "other", "other_tariff")}
        for indices in groups.values():
            network = self._network(policies[indices[0]])
            policy_prices = network.solve(rhs[:, indices])
            prices[:, indices] = policy_prices
            tariff_units[:, indices] = network.tariff_unit_cost(policy_prices)
            for name, cost in network.final_costs(policy_prices).items():
                final_costs[name][indices] = cost
        return [self.state(z, lag, policy, _price=prices[:, t],
                           _final_costs={name: cost[t] for name, cost in final_costs.items()},
                           _tariff_unit=tariff_units[:, t])
                for t, (z, lag, policy) in enumerate(zip(zs, zlags, policies))]

    def residual_from_states(self, state: dict, following: dict) -> np.ndarray:
        """Retained equations: labour clearing, budgets (one numeraire), Euler."""
        d = self.p
        budget = state["budget"] / d.budget_scale
        budget = budget.copy()
        budget[self.omitted_budget_index] = np.log(state["w"][self.numeraire_index])
        payoff = following["R"] + following["q"] * following["continuation"]
        if np.any(payoff <= 0):
            raise EconomicDomainError("Nonpositive capital continuation payoff")
        euler = (np.log(state["q"]) - np.log(d.beta)
                 - np.log(following["lam"][d.country]) + np.log(state["lam"][d.country])
                 - np.log(payoff))
        return np.r_[(state["Ldem"] - d.L0) / d.L0, budget, euler]

    def equations(self, zplus: np.ndarray, z: np.ndarray, zlag: np.ndarray,
                  policy: DynamicTariff | None = None,
                  policy_next: DynamicTariff | None = None) -> np.ndarray:
        """Date-``t`` residual ``F(z_{t+1}, z_t, z_{t-1}; policy_t, policy_{t+1})``."""
        s = self.state(z, zlag, policy)
        sn = self.state(zplus, z, policy if policy_next is None else policy_next)
        return self.residual_from_states(s, sn)

    # -- derivatives --------------------------------------------------------
    def state_jvp(self, z: np.ndarray, zlag: np.ndarray, direction: np.ndarray,
                  lag_direction: np.ndarray, policy: DynamicTariff | None = None,
                  state: dict | None = None, *,
                  _price_derivative: np.ndarray | None = None,
                  _final_cost_derivatives: dict[str, np.ndarray] | None = None,
                  _tariff_unit_derivative: np.ndarray | None = None) -> dict[str, np.ndarray]:
        """Exact directional derivative of :meth:`state`, including implicit network prices."""
        s = self.state(z, zlag, policy) if state is None else state
        d = self.p
        direction, lag_direction = np.asarray(direction), np.asarray(lag_direction)
        dwlog, dClog = direction[self.w_slice], direction[self.c_slice]
        dKlog, dKnlog = lag_direction[self.k_slice], direction[self.k_slice]
        dw, dC = s["w"] * dwlog, s["C"] * dClog
        dK, dKn = s["K"] * dKlog, s["Knext"] * dKnlog
        dx = (s["Knext"] / s["K"]) * (dKnlog - dKlog) / s["installation_derivative"]
        dI = s["I"] * dKlog + s["K"] * dx
        dInational = _national(dI, d.country, self.nc)
        dy = self.YC @ dC + self.YI @ dInational
        ratio = dy / s["y"] - dKlog
        dva = s["va"] * (dwlog[d.country] + d.alpha / (1. - d.alpha) * ratio)
        dR = s["R"] * (dwlog[d.country] + ratio / (1. - d.alpha))
        dLj = s["labor_j"] * (dy / s["y"] + d.alpha / (1. - d.alpha) * ratio)
        dLdem = _national(dLj, d.country, self.nc)
        network = self._network(policy)
        policy = network.policy
        dp = network.solve(d.b * dva) if _price_derivative is None else _price_derivative
        final_changes = (network.final_costs(dp) if _final_cost_derivatives is None
                         else _final_cost_derivatives)
        dPC = final_changes["consumption"] / (1. - d.tC)
        dPI = final_changes["investment"] / (1. - d.tI)
        dq = s["q"] * (dPI[d.country] / s["PI"][d.country]
                       + self.adjustment_cost * dx / s["installation_derivative"])
        dlam = s["lam"] * (-self.risk_aversion * dClog - dPC / s["PC"])
        dtariff_unit = (network.tariff_unit_cost(dp) if _tariff_unit_derivative is None
                        else _tariff_unit_derivative)
        dTR_intermediate = _national(dy * s["tariff_unit"] + s["y"] * dtariff_unit, d.country, self.nc)
        dTR_consumption = (final_changes["consumption_tariff"] * s["C"]
                           + s["tariff_consumption_unit"] * dC)
        dTR_investment = (final_changes["investment_tariff"] * s["Inational"]
                          + s["tariff_investment_unit"] * dInational)
        dTR_other = final_changes["other_tariff"]
        dTR = dTR_intermediate + dTR_consumption + dTR_investment + dTR_other
        dPT_production = _national(d.tax * (dp * s["y"] + s["p"] * dy), d.country, self.nc)
        dPT_consumption = d.tC * (dPC * s["C"] + s["PC"] * dC)
        dPT_investment = d.tI * (dPI * s["Inational"] + s["PI"] * dInational)
        dPT_other = d.TV * dw[-1]
        dPT = dPT_production + dPT_consumption + dPT_investment + dPT_other
        dEV = final_changes["other"] + dPT_other
        dtransfer = d.XN0 * dw[-1]
        dincome = dw * d.L0 + _national(dR * s["K"] + s["R"] * dK, d.country, self.nc) + dPT + dTR
        dbudget = (dPC * s["C"] + s["PC"] * dC + dPI * s["Inational"]
                   + s["PI"] * dInational + dEV + dtransfer - dincome)
        dcontinuation = self.adjustment_cost * s["x"] * dx
        return dict(w=dw, C=dC, K=dK, Knext=dKn, I=dI, Inational=dInational,
                    x=dx, continuation=dcontinuation, y=dy, va=dva, R=dR,
                    labor_j=dLj, Ldem=dLdem, p=dp, PC=dPC, PI=dPI, q=dq,
                    lam=dlam, TR=dTR, PT=dPT, EV=dEV, transfer=dtransfer,
                    income=dincome, budget=dbudget)

    def states_jvp(self, zs: np.ndarray, zlags: np.ndarray, directions: np.ndarray,
                   lag_directions: np.ndarray, policies: Sequence[DynamicTariff],
                   states: Sequence[dict] | None = None) -> list[dict]:
        """Exact date derivatives with batched implicit price differentiation."""
        states = self.states(zs, zlags, policies) if states is None else list(states)
        if len(states) != len(directions) or len(states) != len(lag_directions):
            raise ValueError("Batched derivative dimensions disagree")
        d = self.p
        rhs = np.empty((self.n, len(states)))
        groups: dict[str, list[int]] = {}
        for t, (s, v, vlag, policy) in enumerate(zip(states, directions, lag_directions, policies)):
            dwlog, dClog = v[self.w_slice], v[self.c_slice]
            dKlog, dKnlog = vlag[self.k_slice], v[self.k_slice]
            dx = (s["Knext"] / s["K"]) * (dKnlog - dKlog) / s["installation_derivative"]
            dI = s["I"] * dKlog + s["K"] * dx
            dInational = _national(dI, d.country, self.nc)
            dy = self.YC @ (s["C"] * dClog) + self.YI @ dInational
            dva = s["va"] * (dwlog[d.country] + d.alpha / (1. - d.alpha)
                             * (dy / s["y"] - dKlog))
            rhs[:, t] = d.b * dva
            groups.setdefault(self._network(policy).policy.fingerprint, []).append(t)
        dprices = np.empty_like(rhs)
        dtariff_units = np.empty_like(rhs)
        final_changes = {name: np.empty((len(states), self.nc)) for name in
                         ("consumption", "investment", "consumption_tariff",
                          "investment_tariff", "other", "other_tariff")}
        for indices in groups.values():
            network = self._network(policies[indices[0]])
            policy_dprices = network.solve(rhs[:, indices])
            dprices[:, indices] = policy_dprices
            dtariff_units[:, indices] = network.tariff_unit_cost(policy_dprices)
            for name, cost in network.final_costs(policy_dprices).items():
                final_changes[name][indices] = cost
        return [self.state_jvp(z, lag, v, vlag, policy, s, _price_derivative=dprices[:, t],
                               _final_cost_derivatives={name: cost[t] for name, cost in final_changes.items()},
                               _tariff_unit_derivative=dtariff_units[:, t])
                for t, (z, lag, v, vlag, policy, s) in enumerate(
                    zip(zs, zlags, directions, lag_directions, policies, states))]

    def jvp_from_states(self, s: dict, sn: dict, ds: dict, dsn: dict) -> np.ndarray:
        """Residual derivative from date and following-date state derivatives."""
        d = self.p
        dbudget = ds["budget"] / d.budget_scale
        dbudget = dbudget.copy()
        dbudget[self.omitted_budget_index] = ds["w"][self.numeraire_index] / s["w"][self.numeraire_index]
        payoff = sn["R"] + sn["q"] * sn["continuation"]
        dpayoff = dsn["R"] + dsn["q"] * sn["continuation"] + sn["q"] * dsn["continuation"]
        deuler = (ds["q"] / s["q"] - dsn["lam"][d.country] / sn["lam"][d.country]
                  + ds["lam"][d.country] / s["lam"][d.country] - dpayoff / payoff)
        return np.r_[ds["Ldem"] / d.L0, dbudget, deuler]

    def equations_jvp(self, zplus: np.ndarray, z: np.ndarray, zlag: np.ndarray,
                      plus_direction: np.ndarray, direction: np.ndarray,
                      lag_direction: np.ndarray, policy: DynamicTariff | None = None,
                      policy_next: DynamicTariff | None = None) -> np.ndarray:
        """Directional derivative of :meth:`equations` in ``(zplus, z, zlag)``."""
        next_policy = policy if policy_next is None else policy_next
        s, sn = self.state(z, zlag, policy), self.state(zplus, z, next_policy)
        ds = self.state_jvp(z, zlag, direction, lag_direction, policy, s)
        dsn = self.state_jvp(zplus, z, plus_direction, direction, next_policy, sn)
        return self.jvp_from_states(s, sn, ds, dsn)

    # -- certificate --------------------------------------------------------
    def certificate(self, zplus: np.ndarray, z: np.ndarray, zlag: np.ndarray,
                    policy: DynamicTariff | None = None,
                    policy_next: DynamicTariff | None = None, *,
                    _state: dict | None = None,
                    _following: dict | None = None) -> dict[str, float]:
        """Independently reconstruct accounts, including the omitted budget.

        Seventeen relative checks (each a maximum absolute value): root,
        goods, zero_profit, labor, capital_accumulation, factor_payments,
        capital_market, sector_labor_demand, production_technology,
        investment_foc, capital_euler, tariff_revenue, government_budget,
        budget_all (every national budget, including the omitted one),
        current_account (actual net exports equal the transfer),
        world_transfer_balance and investment_goods_value.
        """
        d = self.p
        s = self.state(z, zlag, policy) if _state is None else _state
        sn = (self.state(zplus, z, policy if policy_next is None else policy_next)
              if _following is None else _following)
        network = self._network(policy)
        policy = network.policy
        intermediate = d.A @ s["y"]
        final = d.omegaC * s["C"] + d.omegaI * s["Inational"] + d.qOther
        goods = s["y"] - intermediate - final.sum(axis=1)
        # Actual national net exports: sales minus domestic intermediate and
        # final basic-price absorption.  Domestic trade cancels without needing
        # a full NxN value matrix.
        sales = _national(s["p"] * s["y"], d.country, self.nc)
        basic_inputs = _national(s["y"] * (d.A.T @ s["p"]), d.country, self.nc)
        actual_exports = sales - basic_inputs - np.sum(s["p"][:, None] * final, axis=0)
        cost = ((1. - d.tax) * s["p"] - d.A.T @ s["p"]
                - network.tariff_unit_cost(s["p"]) - d.b * s["va"])
        phi_value = s["x"] - .5 * self.adjustment_cost * (s["x"] - d.delta) ** 2
        capital = s["Knext"] - (1. - d.delta + phi_value) * s["K"]
        factor = s["w"][d.country] * s["labor_j"] + s["R"] * s["K"] - d.b * s["va"] * s["y"]
        capital_demand = s["R"] * s["K"] - d.alpha * d.b * s["va"] * s["y"]
        labor_demand = s["w"][d.country] * s["labor_j"] - (1. - d.alpha) * d.b * s["va"] * s["y"]
        technology = (np.log(s["y"] / d.y0) - d.alpha * np.log(s["K"] / d.K0)
                      - (1. - d.alpha) * np.log(s["labor_j"] / d.L0sector))
        investment_foc = s["PI"][d.country] - s["q"] * (1. - self.adjustment_cost * (s["x"] - d.delta))
        tariff_units = network.tariff_unit_cost(s["p"])
        tariff_receipts = _national(s["y"] * tariff_units, d.country, self.nc)
        tariff_receipts += np.sum(policy.consumption_rates * s["p"][:, None]
                                  * (d.omegaCtax * s["C"] + d.qOther), axis=0)
        tariff_receipts += np.sum(policy.investment_rates * s["p"][:, None]
                                  * d.omegaI * s["Inational"], axis=0)
        tax_receipts = (_national(d.tax * s["p"] * s["y"], d.country, self.nc)
                        + d.tC * s["PC"] * s["C"] + d.tI * s["PI"] * s["Inational"]
                        + d.TV * s["w"][-1])
        factor_income = s["w"] * d.L0 + _national(s["R"] * s["K"], d.country, self.nc)
        residual = self.residual_from_states(s, sn)
        return dict(root=_maxabs(residual), goods=_maxabs(goods / d.y0),
                    zero_profit=_maxabs(cost / s["p"]),
                    labor=_maxabs((s["Ldem"] - d.L0) / d.L0),
                    capital_accumulation=_maxabs(capital / d.K0),
                    factor_payments=_maxabs(factor / np.maximum(d.VA0, 1e-300)),
                    capital_market=_maxabs(capital_demand / d.VA0),
                    sector_labor_demand=_maxabs(labor_demand / d.VA0),
                    production_technology=_maxabs(technology),
                    investment_foc=_maxabs(investment_foc / s["PI"][d.country]),
                    capital_euler=_maxabs(residual[2 * self.nc:]),
                    tariff_revenue=_maxabs((tariff_receipts - s["TR"]) / d.budget_scale),
                    government_budget=_maxabs((s["income"] - factor_income - tariff_receipts - tax_receipts) / d.budget_scale),
                    budget_all=_maxabs(s["budget"] / d.budget_scale),
                    current_account=_maxabs((actual_exports - s["transfer"]) / d.budget_scale),
                    world_transfer_balance=abs(float(s["transfer"].sum())) / max(float(d.Y0.sum()), 1.),
                    investment_goods_value=_maxabs((np.sum(s["p"][:, None]
                        * (1. + policy.investment_rates) * d.omegaI * s["Inational"], axis=0)
                        / (1. - d.tI) - s["PI"] * s["Inational"]) / d.budget_scale))

    # -- labels ---------------------------------------------------------------
    @property
    def country_codes(self) -> tuple[str, ...]:
        """Country registry of the calibration."""
        return self.p.countries

    @property
    def cell_labels(self) -> tuple[str, ...]:
        """``country:sector`` labels of the active cells."""
        return self.p.cell_labels

    def variable_labels(self) -> tuple[str, ...]:
        """Labels of the retained unknowns ``w:C, C:C, K:cell``."""
        return (tuple(f"w:{c}" for c in self.p.countries) + tuple(f"C:{c}" for c in self.p.countries)
                + tuple(f"K:{label}" for label in self.p.cell_labels))

    def benchmark(self) -> dict[str, np.ndarray]:
        """Benchmark aggregates used by result tables."""
        d = self.p
        return {"C0": d.C0, "Inational0": self.Inational0, "PC0": d.PC0, "PI0": d.PI0,
                "K0": d.K0, "y0": d.y0, "I0": d.I0, "Y0": d.Y0, "L0": d.L0}
