"""Benchmark calibration of the condensed one-factor Leontief tariff model.

At the benchmark every price and wage equals one and every tariff multiplier
equals one. From a :class:`~puremacro.trade.condensed.table.BalancedIOTable`
with ``y0_j = sum_i Z_ij + TLS_j + VA_j``::

    a_ij = Z_ij / y0_j          b_j = VA_j / y0_j          t_j = TLS_j / y0_j
    B^f_k = sum_i F^f_ik        omega^f_ik = F^f_ik / B^f_k          (f in C, G, X)
    A0^f_k = B^f_k + TFD^f_k    t^f_k = TFD^f_k / A0^f_k
    theta^f_k = A0^f_k / sum_f' A0^f'_k
    qV_ik = F^V_ik              TV_k = TFD^V_k
    Fbar_k = sum_{j in k} VA_j
    Y0_k = Fbar_k + sum_{j in k} TLS_j + sum_f TFD^f_k + TFD^V_k
    XN0_k = fob exports minus imports over all uses
    EV0_k = sum_i qV_ik + TV_k      Atil0_k = Y0_k - XN0_k - EV0_k
    s_k = XN0_k / sum_l Fbar_l  (also over world GDP and own GDP)

Every gate raises :class:`~puremacro.trade.condensed.errors.CalibrationError`.
The productivity gate ``rho(a) < 1`` uses the Collatz-Wielandt bounds of
``puremacro.trade.regularize.compute_spectral_radius`` (strongly connected
blocks, shifted iteration, resolvent test vector) wrapped in
:func:`certify_productivity`, which keeps the IO engine's three outcomes:
accepted, rejected (certified lower bound at least one) and inconclusive.
"""
from __future__ import annotations

import warnings
from dataclasses import dataclass, field, replace
from typing import Any, Callable

import numpy as np
import pandas as pd
import scipy.linalg as la

from puremacro.reports import df_to_latex, df_to_markdown, df_to_typst
from puremacro.trade.regularize import compute_spectral_radius

from .errors import CalibrationError, ExistenceViolation, ProductivityUncertified
from .table import ENDOGENOUS_FD, FD_CODES, BalancedIOTable

CELL_BALANCE_TOL = 1e-10
NATIONAL_ACCOUNTS_TOL = 1e-12
THETA_SUM_TOL = 1e-12
EXISTENCE_MARGIN = 1e-3
BASELINE_TARIFF_TOL = 1e-11


def _readonly(a: np.ndarray, dtype: Any = float) -> np.ndarray:
    a = np.array(a, dtype=dtype, copy=True)
    a.flags.writeable = False
    return a


def collatz_wielandt_bound(matvec: Callable[[np.ndarray], np.ndarray], n: int,
                           iters: int = 60, eps: float = 1e-3) -> tuple[float, float]:
    """Upper and lower bounds on the spectral radius of a nonnegative matrix from matrix-vector products.

    Iterates ``x <- (G + eps I) x / ||.||_inf`` from ``x = 1`` and returns
    ``(max_i (G x)_i / x_i, min_i (G x)_i / x_i)``. For any ``x > 0`` these bracket
    ``rho(G)`` (Collatz-Wielandt). The shift keeps ``x`` strictly positive.

    Parameters
    ----------
    matvec : callable
        ``x -> G @ x`` for a nonnegative ``G``.
    n : int
        Dimension.
    iters, eps : int, float
        Iterations and the shift.

    Returns
    -------
    (upper, lower) : tuple of float
        ``(inf, nan)`` when the iteration leaves the positive orthant.
    """
    x = np.ones(n)
    for _ in range(iters):
        x = matvec(x) + eps * x
        top = float(np.max(np.abs(x)))
        if not np.isfinite(top) or top <= 0:
            return float("inf"), float("nan")
        x = x / top
    if not np.all(np.isfinite(x)) or np.any(x <= 0):
        return float("inf"), float("nan")
    gx = matvec(x)
    ratio = gx / x
    return float(np.max(ratio)), float(np.min(ratio))


@dataclass(frozen=True)
class ProductivityBounds:
    """Outcome of :func:`certify_productivity`.

    Attributes
    ----------
    lower, upper : float
        Certified Collatz-Wielandt bounds on the spectral radius.
    rho : float
        Numerical spectral estimate (Rayleigh quotient or eigenvalue); not a bound.
    method : str
        ``"collatz_wielandt"`` or ``"positive_resolvent_witness"``.
    margin : float
        The gate: the upper bound must be below ``1 - margin``.
    """

    lower: float
    upper: float
    rho: float
    method: str
    margin: float
    metadata: dict[str, Any] = field(default_factory=dict)

    @property
    def passed(self) -> bool:
        """True when the upper bound certifies ``rho < 1 - margin``."""
        return bool(np.isfinite(self.upper) and self.upper < 1.0 - self.margin)

    def to_dict(self) -> dict[str, Any]:
        """Bounds, estimate, method, margin and the gate outcome as a plain dictionary."""
        return {"lower": self.lower, "upper": self.upper, "rho": self.rho,
                "method": self.method, "margin": self.margin, "passed": self.passed}

    def to_dataframe(self) -> pd.DataFrame:
        """Two-column table (item, value) of :meth:`to_dict`."""
        return pd.DataFrame(list(self.to_dict().items()), columns=["Item", "Value"]).set_index("Item")

    def summary(self) -> str:
        """One line with the certified bracket, the gate and the outcome."""
        return (f"ProductivityBounds: rho in [{self.lower:.6g}, {self.upper:.6g}] ({self.method}); "
                f"gate rho < {1.0 - self.margin:.6g}: {'passed' if self.passed else 'not certified'}")

    def to_markdown(self, **kwargs: Any) -> str:
        """The table of :meth:`to_dataframe` as Markdown."""
        return df_to_markdown(self.to_dataframe(), **kwargs)

    def to_latex(self, **kwargs: Any) -> str:
        """The table of :meth:`to_dataframe` as a LaTeX tabular."""
        return df_to_latex(self.to_dataframe(), **kwargs)

    def to_typst(self, **kwargs: Any) -> str:
        """The table of :meth:`to_dataframe` as a Typst table."""
        return df_to_typst(self.to_dataframe(), **kwargs)


def certify_productivity(B: np.ndarray, *, margin: float = EXISTENCE_MARGIN) -> ProductivityBounds:
    """Certify ``rho(B) < 1 - margin`` for a nonnegative matrix, or fail closed.

    The bounds come from ``puremacro.trade.regularize.compute_spectral_radius``.
    If the upper bound is below ``1 - margin`` the matrix is productive. If the
    lower bound or a diagonal entry is at least one, :class:`ExistenceViolation`
    is raised. Otherwise a positive resolvent witness ``x = (I - B)^{-1} 1`` is
    tried (its Collatz ratio ``max (Bx)_i / x_i`` is a valid bound whenever
    ``x > 0``); if it does not certify either, :class:`ProductivityUncertified`
    is raised with the best bounds attached and one of two messages:
    "productivity margin not met" when the bounds do certify ``rho < 1`` but not
    ``rho < 1 - margin`` (the gate is deliberately fail-closed: lower ``margin``
    knowingly or accept the rejection), or "productivity is inconclusive" when
    they certify nothing. An upper bound above one is never treated as
    nonexistence.

    Parameters
    ----------
    B : np.ndarray, shape (M, M)
        Finite nonnegative matrix.
    margin : float, default 1e-3
        Required gap below one, the design margin of the IO engine's
        specification (not distributed); the IO productivity certificate
        used ``1e-12`` in code.
    """
    B = np.asarray(B, dtype=float)
    if B.ndim != 2 or B.shape[0] != B.shape[1] or not np.all(np.isfinite(B)) or np.any(B < 0):
        raise ValueError("certify_productivity requires a finite square nonnegative matrix")
    if not np.isfinite(margin) or margin < 0 or margin >= 1:
        raise ValueError("margin must lie in [0, 1)")
    n = B.shape[0]
    if n == 0:
        raise ValueError("empty productivity problem")
    rho, lower, upper = compute_spectral_radius(B)
    diag_max = float(np.max(np.diag(B)))
    if np.isfinite(upper) and upper < 1.0 - margin:
        return ProductivityBounds(float(lower), float(upper), float(rho), "collatz_wielandt", float(margin))
    if lower >= 1.0 or diag_max >= 1.0:
        raise ExistenceViolation(
            f"nonproductive matrix: certified lower bound {max(lower, diag_max):.12g} >= 1"
        )
    witness_error = float("inf")
    try:
        with warnings.catch_warnings():
            warnings.simplefilter("error", la.LinAlgWarning)
            witness = la.solve(np.eye(n) - B, np.ones(n), check_finite=False)
        witness_error = float(np.max(np.abs(witness - B @ witness - 1.0)) / max(1.0, float(np.max(np.abs(witness)))))
        if np.all(np.isfinite(witness)) and np.all(witness > 0):
            ratio = (B @ witness) / witness
            up_w, lo_w = float(np.max(ratio)), float(np.min(ratio))
            if up_w < 1.0 - margin:
                return ProductivityBounds(max(float(lower), lo_w), up_w, float(rho), "positive_resolvent_witness",
                                          float(margin), metadata={"witness_residual": witness_error})
    except (la.LinAlgError, la.LinAlgWarning, ValueError):
        pass
    best_upper = float(upper)
    if np.isfinite(witness_error) and np.all(np.isfinite(witness)) and np.all(witness > 0):
        best_upper = min(best_upper, float(np.max((B @ witness) / witness)))
    if np.isfinite(best_upper) and best_upper < 1.0:
        raise ProductivityUncertified(
            f"productivity margin not met: rho is certified in [{lower:.12g}, {best_upper:.12g}], below one "
            f"but not below 1 - margin = {1.0 - margin:.12g} (margin={margin:g}); lower the margin knowingly "
            "or accept the rejection",
            lower=lower, upper=best_upper, margin=margin,
        )
    raise ProductivityUncertified(
        f"productivity is inconclusive: lower={lower:.12g}, upper={upper:.12g}, margin={margin:g}; "
        f"the positive-witness solve did not certify rho < 1 (residual {witness_error:.3g})",
        lower=lower, upper=best_upper, margin=margin,
    )


@dataclass(frozen=True)
class BaselineTariffAccounts:
    """Explicit benchmark duties separated from the source's net product taxes.

    Attributes
    ----------
    wedges : TariffWedges
        Gross benchmark duty levels (``tau0 >= 1``, domestic 1).
    P0 : np.ndarray, shape (3, N)
        Benchmark purchaser prices of the endogenous baskets, ``sum_i omega^f_ik tau0^f_ik``.
    TR0 : np.ndarray, shape (N,)
        Benchmark customs revenue by country, already included in source GDP.
    TLS_other, TFD_other, TV_other : np.ndarray
        Residual product taxes on production, endogenous final demand and inventories.
    country_codes : tuple of str
        Country registry (storage order) used by :meth:`to_dataframe`.
    metadata : dict
    """

    wedges: Any
    P0: np.ndarray
    TR0: np.ndarray
    TLS_other: np.ndarray
    TFD_other: np.ndarray
    TV_other: np.ndarray
    country_codes: tuple[str, ...] = ()
    metadata: dict[str, Any] = field(default_factory=dict)

    def to_dataframe(self) -> pd.DataFrame:
        """One row per country: benchmark duty, basket purchaser prices and residual final-demand taxes."""
        n = int(self.TR0.shape[0])
        index = pd.Index(list(self.country_codes) if len(self.country_codes) == n else range(n), name="country")
        return pd.DataFrame({
            "TR0": self.TR0, "P0_C": self.P0[0], "P0_G": self.P0[1], "P0_X": self.P0[2],
            "TFD_other_C": self.TFD_other[:, 0], "TFD_other_G": self.TFD_other[:, 1],
            "TFD_other_X": self.TFD_other[:, 2], "TV_other": self.TV_other,
        }, index=index)

    def summary(self) -> str:
        """One line: total benchmark duty and the range of basket purchaser prices."""
        return (f"BaselineTariffAccounts: benchmark duty {float(self.TR0.sum()):.6g} over {self.TR0.shape[0]} "
                f"economies; P0 in [{float(self.P0.min()):.6g}, {float(self.P0.max()):.6g}]")

    def to_markdown(self, **kwargs: Any) -> str:
        """The country table of :meth:`to_dataframe` as Markdown."""
        return df_to_markdown(self.to_dataframe(), **kwargs)

    def to_latex(self, **kwargs: Any) -> str:
        """The country table of :meth:`to_dataframe` as a LaTeX tabular."""
        return df_to_latex(self.to_dataframe(), **kwargs)

    def to_typst(self, **kwargs: Any) -> str:
        """The country table of :meth:`to_dataframe` as a Typst table."""
        return df_to_typst(self.to_dataframe(), **kwargs)


@dataclass(frozen=True, eq=False)
class CondensedCalibration:
    """Benchmark parameters of the condensed model.

    Attributes
    ----------
    a : np.ndarray, shape (M, M)
        Input coefficients ``Z_ij / y0_j``.
    b, t : np.ndarray, shape (M,)
        Value-added share and net product-tax rate on output value.
    omega : np.ndarray, shape (M, N, 3)
        Basket coefficients of the endogenous final-demand categories (C, G, X).
    tfd : np.ndarray, shape (N, 3)
        Final-demand product-tax rates on purchaser value.
    A0, B0 : np.ndarray, shape (N, 3)
        Benchmark purchaser and basic-price values of the endogenous categories.
    theta : np.ndarray, shape (N, 3)
        Cobb-Douglas spending shares over absorption net of inventories.
    qV : np.ndarray, shape (M, N)
        Fixed inventory quantities (base prices; may be negative).
    TV : np.ndarray, shape (N,)
        Inventory product taxes (a real amount valued at the investment price).
    Fbar, Y0, XN0, EV0, Atil0, T0 : np.ndarray, shape (N,)
        Factor income, GDP, fob trade balance, inventory spending, absorption net
        of inventories and net government revenue at the benchmark.
    s_factor, s_gdp, s_own : np.ndarray, shape (N,)
        Deficit shares of world factor income, of world GDP and of own GDP.
    y0 : np.ndarray, shape (M,)
        Benchmark gross output.
    active : np.ndarray of bool, shape (M,)
        Cells with genuine output (phantoms are False).
    goods : np.ndarray of bool, shape (S,)
        Merchandise sectors (ISIC A-C), the default tariff coverage.
    table : BalancedIOTable
        The raw flows, kept for the independent certificate.
    gates : dict
        Achieved value of every calibration gate.
    baseline_tariffs : BaselineTariffAccounts or None
        Set by :func:`separate_baseline_tariffs`.
    """

    a: np.ndarray
    b: np.ndarray
    t: np.ndarray
    omega: np.ndarray
    tfd: np.ndarray
    A0: np.ndarray
    B0: np.ndarray
    theta: np.ndarray
    qV: np.ndarray
    TV: np.ndarray
    Fbar: np.ndarray
    Y0: np.ndarray
    XN0: np.ndarray
    EV0: np.ndarray
    Atil0: np.ndarray
    T0: np.ndarray
    s_factor: np.ndarray
    s_gdp: np.ndarray
    s_own: np.ndarray
    y0: np.ndarray
    active: np.ndarray
    goods: np.ndarray
    table: BalancedIOTable
    gates: dict[str, Any] = field(default_factory=dict)
    baseline_tariffs: BaselineTariffAccounts | None = None
    metadata: dict[str, Any] = field(default_factory=dict)

    @property
    def P0(self) -> np.ndarray:
        """Benchmark purchaser prices of the endogenous baskets before residual product taxes (3, N)."""
        if self.baseline_tariffs is None:
            return np.ones((3, self.n_countries))
        return self.baseline_tariffs.P0

    @property
    def TR0(self) -> np.ndarray:
        """Explicit benchmark customs revenue (N,); zero unless baseline duties were separated."""
        if self.baseline_tariffs is None:
            return np.zeros(self.n_countries)
        return self.baseline_tariffs.TR0

    @property
    def country_codes(self) -> tuple[str, ...]:
        """Country registry in storage order."""
        return self.table.country_codes

    @property
    def sector_codes(self) -> tuple[str, ...]:
        """Sector registry in storage order."""
        return self.table.sector_codes

    @property
    def n_countries(self) -> int:
        """Number of economies ``N``."""
        return self.table.n_countries

    @property
    def n_sectors(self) -> int:
        """Number of industries per economy ``S``."""
        return self.table.n_sectors

    @property
    def n_cells(self) -> int:
        """Number of country-industry cells ``M = N S``."""
        return self.table.n_cells

    @property
    def country_of_cell(self) -> np.ndarray:
        """Country index of every cell (country-major layout)."""
        return np.repeat(np.arange(self.n_countries), self.n_sectors)

    def index(self, code: str) -> int:
        """Storage position of a country code."""
        return self.table.country_index(code)

    @property
    def reference(self) -> int:
        """Storage position of the reference country (dropped income equation, Walras check)."""
        return self.table.reference

    def summary(self) -> str:
        """One line with the table size and the achieved values of the main gates."""
        g = self.gates
        return (
            f"CondensedCalibration: {self.n_countries} economies x {self.n_sectors} industries; "
            f"cell balance {g.get('cell_balance_rel', float('nan')):.1e}, national accounts "
            f"{g.get('national_accounts_rel', float('nan')):.1e}, rho(a) <= "
            f"{g.get('rho_a_upper', float('nan')):.4f}, min theta {g.get('theta_min', float('nan')):.4f}"
            + ("; explicit baseline duties" if self.baseline_tariffs is not None else "")
        )

    def to_dataframe(self) -> pd.DataFrame:
        """Country benchmark aggregates: factor income, GDP, net exports, inventories, absorption."""
        return pd.DataFrame({
            "factor_income": self.Fbar, "gdp": self.Y0, "net_exports": self.XN0,
            "inventory_spending": self.EV0, "absorption_net_of_inventories": self.Atil0,
            "net_revenue": self.T0, "deficit_share_factor": self.s_factor,
        }, index=pd.Index(self.country_codes, name="country"))

    def to_markdown(self, **kwargs: Any) -> str:
        """The country aggregates of :meth:`to_dataframe` as Markdown."""
        return df_to_markdown(self.to_dataframe(), **kwargs)

    def to_latex(self, **kwargs: Any) -> str:
        """The country aggregates of :meth:`to_dataframe` as a LaTeX tabular."""
        return df_to_latex(self.to_dataframe(), **kwargs)

    def to_typst(self, **kwargs: Any) -> str:
        """The country aggregates of :meth:`to_dataframe` as a Typst table."""
        return df_to_typst(self.to_dataframe(), **kwargs)


def calibrate_condensed(
    table: BalancedIOTable,
    *,
    merchandise_mask: np.ndarray | None = None,
    allow_empty_purchases_abroad: bool = False,
    existence_margin: float = EXISTENCE_MARGIN,
) -> CondensedCalibration:
    """Calibrate the condensed model and enforce every benchmark gate.

    Parameters
    ----------
    table : BalancedIOTable
        A table from :func:`~puremacro.trade.condensed.table.table_from_arrays`
        (phantom cells are required).
    merchandise_mask : array of bool, shape (S,), optional
        Goods sectors; defaults to ``table.merchandise_mask``.
    allow_empty_purchases_abroad : bool, default False
        Accept countries with no residents' purchases abroad (X). Their X
        basket copies the C basket as an auxiliary positive price; expenditure
        and quantity are identically zero. Required for tables built from
        layouts without a purchases-abroad column (FIGARO, EXIOBASE, WIOD,
        Eora, or ``("C", "G", "V")``).
    existence_margin : float, default 1e-3
        Gap required below one for ``rho(a)`` (see :func:`certify_productivity`).

    Returns
    -------
    CondensedCalibration

    Raises
    ------
    CalibrationError
        Non-finite values, a broken cell balance (``1e-10`` relative) or
        national-accounts identity (``1e-12``), invalid spending shares,
        negative basket coefficients, tax rates of 100% or more, non-positive
        value added or absorption, deficit shares not summing to zero.
    ExistenceViolation, ProductivityUncertified
        ``rho(a) >= 1`` certified, or inconclusive bounds.
    """
    if not isinstance(table, BalancedIOTable):
        raise TypeError("table must be a BalancedIOTable")
    n, s = table.n_countries, table.n_sectors
    Z, F, VA, TLS, TFD = table.Z, table.F, table.VA, table.TLS, table.TFD
    for name, arr in (("Z", Z), ("F", F), ("VA", VA), ("TLS", TLS), ("TFD", TFD)):
        if not np.all(np.isfinite(arr)):
            raise CalibrationError(f"{name} contains NaN or Inf")
    if np.any(Z < 0):
        raise CalibrationError("negative intermediate flows: the existence bound needs a nonnegative input matrix")
    country = np.repeat(np.arange(n), s)
    endo = [FD_CODES.index(c) for c in ENDOGENOUS_FD]
    iv = FD_CODES.index("V")

    y0 = Z.sum(axis=0) + TLS + VA
    rows = Z.sum(axis=1) + F.sum(axis=(1, 2))
    if np.any(y0 <= 0):
        raise CalibrationError("a cell has non-positive output; empty cells need phantoms")
    cell_balance = float(np.max(np.abs(y0 - rows) / y0))
    if cell_balance > CELL_BALANCE_TOL:
        raise CalibrationError(f"cell balance row = column fails: {cell_balance:.2e} > {CELL_BALANCE_TOL:.0e}")

    a = Z / y0[None, :]
    b = VA / y0
    t = TLS / y0
    Fe = F[:, :, endo]
    B0 = Fe.sum(axis=0)
    TFDe = TFD[:, endo]
    A0 = B0 + TFDe
    empty_x = (B0[:, 2] == 0) & (TFDe[:, 2] == 0)
    allowed_empty = np.zeros_like(B0, dtype=bool)
    if allow_empty_purchases_abroad:
        allowed_empty[:, 2] = empty_x
    if np.any((B0 <= 0) & ~allowed_empty):
        bad = np.argwhere((B0 <= 0) & ~allowed_empty)
        hint = ""
        if np.all(bad[:, 1] == 2):
            hint = ("; the table has no purchases abroad (X) for these economies: pass "
                    "allow_empty_purchases_abroad=True (layouts without an X column always need it)")
        raise CalibrationError(
            "final-demand category with zero basic value (country, category): "
            f"{[(table.country_codes[i], ENDOGENOUS_FD[f]) for i, f in bad[:5]]}{hint}"
        )
    omega = np.divide(Fe, B0[None, :, :], out=np.zeros_like(Fe), where=B0[None, :, :] > 0)
    if allow_empty_purchases_abroad:
        omega[:, empty_x, 2] = omega[:, empty_x, 0]
    tfd = np.divide(TFDe, A0, out=np.zeros_like(TFDe), where=A0 > 0)
    theta = A0 / A0.sum(axis=1, keepdims=True)
    qV = F[:, :, iv].copy()
    TV = TFD[:, iv].copy()
    Fbar = np.bincount(country, weights=VA, minlength=n)
    PT0 = np.bincount(country, weights=TLS, minlength=n)
    Y0 = Fbar + PT0 + TFD.sum(axis=1)
    T0 = Y0 - Fbar

    XN0 = table.net_exports()
    ref = table.reference
    EV0 = qV.sum(axis=0) + TV
    Atil0 = Y0 - XN0 - EV0
    s_factor = XN0 / Fbar.sum()
    s_gdp = XN0 / Y0.sum()
    s_own = XN0 / Y0
    for shares in (s_factor, s_gdp):
        shares[ref] = -(shares.sum() - shares[ref])

    nat = Y0 - A0.sum(axis=1) - EV0 - XN0
    national_accounts = float(np.max(np.abs(nat) / Y0))
    theta_sum = float(np.max(np.abs(theta.sum(axis=1) - 1.0)))
    one_minus_t = float(np.min(1.0 - t))
    one_minus_tfd = float(np.min(1.0 - tfd))
    active = np.asarray(table.active, dtype=bool)
    gates: dict[str, Any] = {
        "cell_balance_rel": cell_balance,
        "national_accounts_rel": national_accounts,
        "theta_sum_abs": theta_sum,
        "theta_min": float(theta.min()),
        "omega_min": float(omega.min()),
        "one_minus_t_min": one_minus_t,
        "one_minus_tfd_min": one_minus_tfd,
        "b_min_active": float(b[active].min()) if np.any(active) else float(b.min()),
        "atil0_min": float(Atil0.min()),
        "deficit_share_sum": float(abs(s_factor.sum())),
        "max_input_column_sum": float(a.sum(axis=0).max()),
    }
    if national_accounts > NATIONAL_ACCOUNTS_TOL:
        raise CalibrationError(f"national accounts identity fails: {national_accounts:.2e}")
    if theta_sum > THETA_SUM_TOL or np.any((theta <= 0) & ~allowed_empty):
        raise CalibrationError(f"spending shares invalid: |sum-1|={theta_sum:.2e}, min={theta.min():.3e}")
    if np.any(omega < 0):
        bad = np.argwhere(omega < 0)
        i, k, f = bad[0]
        raise CalibrationError(
            f"negative basket coefficient in C, G or X ({len(bad)} cells, for example "
            f"{table.labels[i]}->{table.country_codes[k]} in {ENDOGENOUS_FD[f]}); use "
            "negative_investment='to_inventory' or a table with signed inventories in V"
        )
    if one_minus_t <= 0 or one_minus_tfd <= 0:
        raise CalibrationError("a product-tax rate is 100% or more")
    if np.any(b <= 0):
        raise CalibrationError("non-positive value-added share")
    if np.any(Atil0 <= 0):
        raise CalibrationError("non-positive absorption net of inventories")
    productive = certify_productivity(a, margin=existence_margin)
    gates["rho_a_upper"], gates["rho_a_lower"] = productive.upper, productive.lower
    gates["rho_a_estimate"], gates["rho_a_method"] = productive.rho, productive.method
    if abs(s_factor.sum()) > 0.0 and abs(s_factor.sum()) > 1e-15 * np.abs(s_factor).sum():
        raise CalibrationError(f"deficit shares do not sum to zero: {s_factor.sum():.3e}")

    goods = np.asarray(table.merchandise_mask if merchandise_mask is None else merchandise_mask)
    if goods.dtype != bool or goods.shape != (s,):
        raise CalibrationError("merchandise_mask must be boolean with one entry per sector")
    return CondensedCalibration(
        a=_readonly(a), b=_readonly(b), t=_readonly(t), omega=_readonly(np.ascontiguousarray(omega)),
        tfd=_readonly(tfd), A0=_readonly(A0), B0=_readonly(B0), theta=_readonly(theta), qV=_readonly(qV),
        TV=_readonly(TV), Fbar=_readonly(Fbar), Y0=_readonly(Y0), XN0=_readonly(XN0), EV0=_readonly(EV0),
        Atil0=_readonly(Atil0), T0=_readonly(T0), s_factor=_readonly(s_factor), s_gdp=_readonly(s_gdp),
        s_own=_readonly(s_own), y0=_readonly(y0), active=_readonly(active, bool), goods=_readonly(goods, bool),
        table=table, gates=gates,
        metadata={"model": "condensed_leontief_one_factor", "benchmark": "p = w = 1, tau = 1",
                  "productivity_gate": "puremacro.trade.regularize.compute_spectral_radius"},
    )


def separate_baseline_tariffs(calib: CondensedCalibration, wedges: Any) -> CondensedCalibration:
    """Separate identified customs duties from the source's net product taxes.

    ``wedges`` must hold gross benchmark LEVELS ``tau0 >= 1`` (domestic
    transactions and purchases abroad equal one), not policy increments. The
    observed physical flows, GDP, absorption and factor endowments are
    preserved: ``TLS_other = TLS - sum_i (tau0_ij - 1) Z_ij`` and likewise for
    the final-demand and inventory taxes; ``P0^f_k = sum_i omega^f_ik tau0^f_ik``;
    ``TR0_k`` is the total benchmark duty; ``t = TLS_other / y0``,
    ``t^f = TFD_other / A0``, ``T0 <- T0 - TR0`` and ``TV <- TV_other / P0^G``.
    Two gates at ``1e-11``: the cost normalization
    ``|b + sum_i a_ij tau0_ij + t - 1|`` and the quantity normalization
    ``|(1 - t^f) A0^f / P0^f - B^f| / max(1, B^f)``. Residual product taxes may
    be negative, as may the source's net taxes. This accounting separation is
    not evidence that an unobserved tariff is zero.
    """
    if calib.baseline_tariffs is not None:
        raise CalibrationError("baseline tariffs have already been separated")
    c = calib
    M, N = c.n_cells, c.n_countries
    for name, shape in (("tau", (M, M)), ("tau_fd", (M, N, 3)), ("tau_V", (M, N))):
        value = np.asarray(getattr(wedges, name))
        if value.shape != shape or not np.isfinite(value).all() or np.any(value < 1):
            raise CalibrationError("invalid gross benchmark tariff levels: " + name)
    country = c.country_of_cell
    for k in range(N):
        own = country == k
        if (np.any(wedges.tau[np.ix_(own, own)] != 1) or np.any(wedges.tau_fd[own, k] != 1)
                or np.any(wedges.tau_V[own, k] != 1)):
            raise CalibrationError("domestic benchmark transactions cannot bear import duties")
    if np.any(wedges.tau_fd[:, :, 2] != 1):
        raise CalibrationError("purchases abroad are not customs imports")
    table = c.table
    Fe = table.F[:, :, [FD_CODES.index(f) for f in ENDOGENOUS_FD]]
    tr_i = ((wedges.tau - 1) * table.Z).sum(axis=0)
    tr_f = ((wedges.tau_fd - 1) * Fe).sum(axis=0)
    tr_v = ((wedges.tau_V - 1) * c.qV).sum(axis=0)
    tls = table.TLS - tr_i
    tfd_raw = table.TFD[:, :3] - tr_f
    tv = table.TFD[:, 3] - tr_v
    P0 = np.einsum("ikf->fk", c.omega * wedges.tau_fd)
    TR0 = tr_i.reshape(N, c.n_sectors).sum(axis=1) + tr_f.sum(axis=1) + tr_v
    t = tls / c.y0
    tfd = np.divide(tfd_raw, c.A0, out=np.zeros_like(tfd_raw), where=c.A0 > 0)
    if np.any(1 - t <= 0) or np.any(1 - tfd <= 0):
        raise CalibrationError("invalid residual product-tax calibration")
    cost_error = float(np.max(np.abs(c.b + (c.a * wedges.tau).sum(axis=0) + t - 1)))
    quantity_error = float(np.max(np.abs((1 - tfd) * c.A0 / P0.T - c.B0) / np.maximum(1, c.B0)))
    if max(cost_error, quantity_error) > BASELINE_TARIFF_TOL:
        raise CalibrationError(
            f"explicit tariff benchmark fails cost ({cost_error:.2e}) or final-demand ({quantity_error:.2e}) normalization"
        )
    accounts = BaselineTariffAccounts(
        wedges=wedges, P0=_readonly(P0), TR0=_readonly(TR0), TLS_other=_readonly(tls),
        TFD_other=_readonly(tfd_raw), TV_other=_readonly(tv), country_codes=tuple(c.country_codes),
        metadata={"cost_error": cost_error, "quantity_error": quantity_error, "tol": BASELINE_TARIFF_TOL},
    )
    gates = {**c.gates, "explicit_tariff_cost_error": cost_error,
             "explicit_tariff_quantity_error": quantity_error,
             "explicit_tariff_revenue": TR0.tolist(),
             "one_minus_t_min": float(np.min(1 - t)),
             "one_minus_tfd_min": float(np.min(1 - tfd))}
    return replace(c, t=_readonly(t), tfd=_readonly(tfd), TV=_readonly(tv / P0[1]), T0=_readonly(c.T0 - TR0),
                   baseline_tariffs=accounts, gates=gates)


__all__ = [
    "BaselineTariffAccounts",
    "CondensedCalibration",
    "ProductivityBounds",
    "calibrate_condensed",
    "certify_productivity",
    "collatz_wielandt_bound",
    "separate_baseline_tariffs",
]
