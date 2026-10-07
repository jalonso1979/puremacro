"""Reduced local-stability diagnostic for converged trade equilibria (experimental).

The diagnostic classifies a converged :class:`TradeEquilibriumResult` under one
explicit, closure-dependent adjustment process: relative factor prices respond
to their own excess demand while producer prices, gross outputs and national
fiscal budgets clear at every factor-price vector. It is evidence about that
process only. It is never a uniqueness theorem, a basin-of-attraction
statement or an estimated adjustment law, and the IO qualification text
travels with every result (``StabilityResult.qualification``).

Mathematics
-----------
Write the model's physical residual vector as ``r(x)`` with the puremacro
unknown layout ``x = [log p; log y; log r; log w; T; XN]`` (``n = ns*nc``
cells, ``nc`` countries). Partition the unknowns into the slow block
``z = (log r, log w)`` (``2 nc`` entries) and the fast block
``u = (log p, log y, T)`` (``2 n + nc`` entries), with the foreign balances
``XN`` held fixed. Partition the rows into the factor rows
``f = (labour gap, capital gap)`` and the fast rows
``g = (goods, zero profit, fiscal budgets)``; the current-account rows are not
imposed on the fast block (they are a certificate of the converged state only;
keeping them in the fast block together with ``XN`` gives a structurally
rank-deficient reduced matrix, which the diagnostic reports as
``'degenerate'``).

The fast-cleared excess-demand ratios are ``e(z) = -f(u(z), z) / endowment``
where ``u(z)`` solves ``g(u, z) = 0``. Their Jacobian is the Schur complement

    J_F = f_z - f_u g_u^{-1} g_z,          A = -J_F / endowment,

with rows ordered ``(capital, labour)`` so that they match the columns
``(log r, log w)``. Foreign balances are re-denominated so that the map is
homogeneous of degree zero in all factor prices (``A 1 = 0``); by default they
are fixed in units of world factor income ``sum_k (w_k L_k + r_k K_k)``, the
numeraire of the IO reduced-stability code. Under the consistent accounting
the value-weighted world identity

    p.goods + (1 - tax) y.(pp - p) + w.labour_gap + r.capital_gap + 1.budget = 0

holds at every ``x``, so on the fast-cleared manifold the value-weighted excess
factor demand ``W.e`` vanishes identically (``W = (r K, w L)``) and its
derivative gives ``W A + W*e = 0``, i.e. ``(r K, w L) A = 0`` at an
equilibrium (Walras). The IO code restores one dropped budget row from the
same identity; puremacro evaluates every budget row directly and uses the
identity as a check instead.

The relative law ``d log(f_i / f_ref) / dt = e_i - e_ref`` for every factor
price ``f_i`` other than the reference gives the ``(2 nc - 1)``-dimensional
matrix ``R = A[keep, keep] - A[ref, keep]``. With ``A 1 = 0`` every reference
yields a similar ``R``, so the spectrum does not depend on the reference.
Optional country-specific adjustment speeds ``D`` replace ``A`` by ``D A``.

Classification: ``'degenerate'`` when ``R`` is numerically rank deficient
(singular values below ``rank_tol * max(1, largest)``: a structural null
space or a fold); otherwise ``'unstable'`` when ``max Re eig(R) > tol``,
``'stable'`` when ``< -tol``, ``'nonhyperbolic'`` in between.

Derivatives are central finite differences of
:func:`puremacro.trade._accounting.evaluate` (consistent results) or
:func:`puremacro.trade.equilibrium.evaluate_equilibrium_residuals` (legacy
results); complex step is unavailable because the state unpacker casts to
float. The steps on the level unknowns (``T``, and ``XN`` in the naive
partition) are scaled by national income because those unknowns enter the
lump-sum model linearly in calibration units. ``dense=True`` recomputes ``A``
independently by re-solving the fast block (chord iteration) at perturbed
factor prices and differencing the excess-demand ratios. This departs from the
IO code, which eliminates the fast unknowns from a complex-step Jacobian of the
full system: the re-solve is a nonlinear check of the fast-cleared map itself,
not a second linear elimination of the same Jacobian.

Under the consistent accounting ``A 1 = 0`` (world-factor-income or
reference-factor-price balances, or the naive partition) and the Walras and
world identities are exact, so a violation beyond ``homogeneity_tol`` (world
identity: 1e-10) can only be numerical error and raises
:class:`StabilityError`. Under the legacy accounting, or with balances fixed
in numeraire units, homogeneity is not exact: a violation issues a
``RuntimeWarning``, sets ``metadata['homogeneous'] = False`` and is repeated in
``summary()`` and ``closure``.

Limitations quoted from the IO documentation: "A local stability result is
conditional on the specified relative-wage adjustment process, not a
uniqueness theorem" (README) and "Numerical verification does not establish
empirical identification, global uniqueness, or stability under every possible
adjustment process" (VALIDATION.md). Both synthetic puremacro fixtures used in
the tests classify as ``'unstable'`` under this partition with fixed-coefficient
intermediate sourcing (``sigma = 0``) and as ``'stable'`` once the intermediate
substitution elasticity reaches one; the outcome is closure dependent. See
``docs/trade_stability.md``.
"""
from __future__ import annotations

import warnings
from dataclasses import dataclass, field
from numbers import Integral
from typing import Any, Callable

import numpy as np
import pandas as pd
from scipy.linalg import eig, lu_factor, lu_solve

from puremacro.reports import df_to_latex, df_to_markdown, df_to_typst
from ._results import TradeCalibrationResult, TradeEquilibriumResult

__all__ = ["StabilityError", "StabilityResult", "reduced_stability"]

QUALIFICATION = (
    "Local diagnostic under the specified relative-factor-price adjustment law with "
    "producer prices, goods and national fiscal budgets clearing at each factor-price "
    "vector; the classification is closure- and partition-dependent evidence, not a "
    "verdict. IO README: 'A local stability result is conditional on the specified "
    "relative-wage adjustment process, not a uniqueness theorem.' IO VALIDATION.md: "
    "'Numerical verification does not establish empirical identification, global "
    "uniqueness, or stability under every possible adjustment process.'"
)

_NON_HOMOGENEOUS_CAVEAT = ("map not homogeneous of degree zero; relative reduction conditional on "
                           "the reference price")
_WORLD_IDENTITY_TOL = 1e-10
_MAX_LOG_STEP = 1e-2
_BALANCE_UNITS = ("world_factor_income", "reference_factor_price", "numeraire")
_CURRENT_ACCOUNT = ("certificate", "fast")
_CLASSES = ("stable", "unstable", "nonhyperbolic", "degenerate")


class StabilityError(RuntimeError):
    """Numerical failure of the reduced-stability computation.

    Raised when the fast block is singular, when the state does not satisfy
    its own equilibrium equations at the requested tolerance, when an identity
    that is exact for the evaluated map (homogeneity, Walras or the world value
    identity under the consistent accounting) fails beyond its tolerance, or
    when the independent dense re-solve check disagrees with the Schur
    complement.
    """


@dataclass(frozen=True)
class StabilityResult:
    """Local stability of a relative-factor-price tatonnement at one equilibrium.

    ``eigenvalues`` are those of ``relative_matrix`` sorted by decreasing real
    part; ``reduced_jacobian`` is ``A`` (rows and columns ordered
    ``r[0..nc-1], w[0..nc-1]``, see ``factor_labels``); ``rank`` is the
    numerical rank of ``relative_matrix``. ``homogeneity_error`` and
    ``walras_error`` are relative; ``equilibrium_residual`` (max absolute
    physical residual) and ``dense_check_error`` (max absolute entrywise
    disagreement with ``A``) are absolute; all are documented in
    :func:`reduced_stability`. ``dense_check_error`` is ``None`` when the
    independent check was not run. Arrays, including those in ``metadata``,
    are read-only.
    """

    classification: str
    maximum_real_eigenvalue: float
    eigenvalues: np.ndarray
    reduced_jacobian: np.ndarray
    relative_matrix: np.ndarray
    rank: int
    homogeneity_error: float
    walras_error: float
    fast_block_condition: float
    equilibrium_residual: float
    dense_check_error: float | None
    adjustment: str
    closure: str
    qualification: str
    reference: str
    factor_labels: tuple[str, ...]
    metadata: dict[str, Any] = field(default_factory=dict)

    def to_dataframe(self) -> pd.DataFrame:
        """Eigenvalues of the relative matrix: real part, imaginary part, modulus."""
        values = np.asarray(self.eigenvalues)
        return pd.DataFrame({"real": values.real, "imag": values.imag, "modulus": np.abs(values)},
                            index=pd.Index(np.arange(len(values)), name="mode"))

    def mode_loadings(self, mode: int = 0) -> pd.DataFrame:
        """Right eigenvector of ``relative_matrix`` for ``mode`` on the factor-price labels.

        The reference factor price carries loading zero by construction (relative
        coordinates); the other entries are the real and imaginary parts of the
        eigenvector normalised to unit Euclidean norm.
        """
        vectors = np.asarray(self.metadata["eigenvectors"])
        if not 0 <= mode < vectors.shape[1]:
            raise ValueError("mode index out of range")
        keep = np.asarray(self.metadata["keep_indices"])
        full = np.zeros(len(self.factor_labels), dtype=complex)
        full[keep] = vectors[:, mode]
        return pd.DataFrame({"real": full.real, "imag": full.imag},
                            index=pd.Index(list(self.factor_labels), name="factor_price"))

    def summary(self) -> str:
        """One-paragraph verdict with its closure, checks and qualification.

        The closure text carries the non-homogeneity caveat when the map is not
        homogeneous of degree zero, so the verdict line always shows it.
        """
        return (f"Reduced local stability [{self.closure}]: {self.classification}, "
                f"max Re eigenvalue {self.maximum_real_eigenvalue:+.6g} over "
                f"{len(self.eigenvalues)} relative modes (rank {self.rank}); homogeneity "
                f"{self.homogeneity_error:.1e}, Walras {self.walras_error:.1e}, fast-block "
                f"condition {self.fast_block_condition:.3g}. Conditional on the stated "
                f"adjustment process; not a uniqueness result.")

    def to_markdown(self, **kwargs: Any) -> str:
        return df_to_markdown(self.to_dataframe(), **kwargs)

    def to_latex(self, **kwargs: Any) -> str:
        return df_to_latex(self.to_dataframe(), **kwargs)

    def to_typst(self, **kwargs: Any) -> str:
        return df_to_typst(self.to_dataframe(), **kwargs)


def _country_index(calib: TradeCalibrationResult, country: int | str) -> int:
    if isinstance(country, str):
        if country not in calib.country_codes:
            raise ValueError(f"Unknown country {country!r}")
        return calib.country_codes.index(country)
    if isinstance(country, Integral) and not isinstance(country, bool):
        index = int(country)
        if not 0 <= index < calib.nc:
            raise ValueError("reference country index is out of range")
        return index
    raise TypeError("reference must be a country code or an integer index")


def _resolved_schedules(calib, result, tau, tau_fd, tauf, tauf_fd, accounting):
    """Tariff schedules for the residual evaluation, taken from or checked against metadata."""
    from .solver import _resolve_tariffs
    meta = result.metadata
    explicit = any(v is not None for v in (tau, tau_fd, tauf, tauf_fd))
    if accounting == "consistent":
        if any(k not in meta for k in ("intermediate_tariff_multipliers", "final_tariff_multipliers")):
            raise ValueError("Consistent result lacks its recorded tariff schedules; solve it again")
        ta = np.asarray(meta["intermediate_tariff_multipliers"], dtype=float)
        tf = np.asarray(meta["final_tariff_multipliers"], dtype=float)
        if explicit:
            ta_x, tf_x, _, _ = _resolve_tariffs(calib, tau, tau_fd, tauf, tauf_fd)
            if ta_x.shape != ta.shape or tf_x.shape != tf.shape or not (
                    np.allclose(ta_x, ta, rtol=0, atol=1e-12) and np.allclose(tf_x, tf, rtol=0, atol=1e-12)):
                raise ValueError("Explicit tariffs disagree with the schedules recorded on the consistent result")
        return ta, tf, np.zeros(calib.nc), np.zeros(calib.nc)
    ta, tf, tauf_vec, tauf_fd_vec = _resolve_tariffs(calib, tau, tau_fd, tauf, tauf_fd)
    n, ns, nc, nfd = calib.nc*calib.ns, calib.ns, calib.nc, calib.n_final_demand
    if ta.shape != (n, ns, nc) or tf.shape != (n, nfd, nc):
        raise ValueError("Tariff multipliers must resolve to shapes (ns*nc, ns, nc) and (ns*nc, nfd, nc)")
    return ta, tf, tauf_vec, tauf_fd_vec


def _residual_function(calib, result, accounting, schedules) -> Callable[[np.ndarray], np.ndarray]:
    meta = result.metadata
    sigma = float(meta.get("sigma", 0.))
    ta, tf, tauf_vec, tauf_fd_vec = schedules
    if accounting == "consistent":
        from ._accounting import evaluate

        def physical(x):
            return evaluate(x, calib, ta, tf, None, None, sigma=sigma)["physical_residuals"]
        return physical
    from .equilibrium import evaluate_equilibrium_residuals
    options = dict(tauf=tauf_vec, tauf_fd=tauf_fd_vec,
                   replicate_matlab_precedence=bool(meta.get("replicate_matlab_precedence", True)),
                   sigma=sigma, tariff_revenue_mode=meta.get("tariff_revenue_mode", "legacy_national"),
                   accounting="legacy", fiscal_closure=meta.get("fiscal_closure", "lump_sum"))

    def legacy(x):
        return evaluate_equilibrium_residuals(x, calib, tau=ta, tau_fd=tf, **options)
    return legacy


def _rows(n, nc, accounting):
    """Row index sets of the physical residual vector for each accounting."""
    n_ca = nc if accounting == "consistent" else nc-1
    start = 0
    goods = np.arange(start, start+n); start += n
    prices = np.arange(start, start+n); start += n
    labour = np.arange(start, start+nc); start += nc
    capital = np.arange(start, start+nc); start += nc
    current = np.arange(start, start+n_ca); start += n_ca
    budget = np.arange(start, start+nc); start += nc
    return dict(goods=goods, prices=prices, labour=labour, capital=capital, current=current,
                budget=budget, total=start)


def _jacobian(f, x, columns, steps, order):
    """Central finite differences of ``f`` at ``x`` along ``columns`` with per-column steps."""
    r0 = f(x)
    J = np.empty((len(r0), len(columns)))
    for j, (col, h) in enumerate(zip(columns, steps)):
        xp, xm = x.copy(), x.copy()
        xp[col] += h; xm[col] -= h
        if order == 2:
            J[:, j] = (f(xp)-f(xm))/(2*h)
        else:
            xpp, xmm = x.copy(), x.copy()
            xpp[col] += 2*h; xmm[col] -= 2*h
            J[:, j] = (-f(xpp)+8*f(xp)-8*f(xm)+f(xmm))/(12*h)
    if not np.isfinite(J).all():
        raise StabilityError("Finite-difference Jacobian is not finite")
    return r0, J


def _classify(eigenvalues, rank, full_rank, tol):
    if rank < full_rank:
        return "degenerate"
    maximum = float(np.max(eigenvalues.real))
    if maximum > tol:
        return "unstable"
    if maximum < -tol:
        return "stable"
    return "nonhyperbolic"


def reduced_stability(
    calib: TradeCalibrationResult,
    result: TradeEquilibriumResult,
    *,
    adjustment: str = "relative_factor_prices",
    reference: int | str = 0,
    reference_factor: str = "w",
    balance_units: str = "world_factor_income",
    current_account: str = "certificate",
    tau: np.ndarray | None = None,
    tau_fd: np.ndarray | None = None,
    tauf: np.ndarray | None = None,
    tauf_fd: np.ndarray | None = None,
    speeds: np.ndarray | None = None,
    step: float = 1e-6,
    fd_order: int = 2,
    dense: bool = True,
    dense_step: float = 1e-4,
    dense_tol: float = 1e-6,
    tol: float = 1e-7,
    rank_tol: float = 1e-7,
    homogeneity_tol: float = 1e-6,
    equilibrium_tol: float | None = None,
) -> StabilityResult:
    """Classify the local stability of a relative-factor-price tatonnement at ``result``.

    The reduced Jacobian ``A`` of the fast-cleared excess factor-demand ratios is
    built as the Schur complement ``-(f_z - f_u g_u^{-1} g_z) / endowment`` of the
    central finite-difference Jacobian of the physical residuals (module
    docstring). ``result`` must be converged and, for the residual function
    selected by ``result.metadata['accounting']``, satisfy ``max |r(x)| <=
    equilibrium_tol`` (default: the solver tolerance recorded in the metadata,
    else 1e-8). Only the lump-sum fiscal closure without capacity margins or
    recycling is supported.

    Parameters
    ----------
    calib, result
        The calibration and a converged equilibrium solved on it. Consistent
        results carry their tariff schedules in ``metadata``; legacy results do
        not, so pass the same ``tau``/``tau_fd`` (or ``tauf``/``tauf_fd``) that
        produced them (free trade when omitted).
    adjustment
        Only ``'relative_factor_prices'``: ``d log(f_i/f_ref)/dt = e_i - e_ref``.
    reference, reference_factor
        Country (index or code) and factor (``'w'`` wage or ``'r'`` rental)
        whose price is the reference of the relative law.
    balance_units
        Denomination in which the exogenous foreign balances are held fixed
        while factor prices move: ``'world_factor_income'`` (default, the IO
        numeraire; ``A`` is homogeneous of degree zero and its relative
        spectrum is reference-invariant), ``'reference_factor_price'`` (the
        assessment prototype's convention; homogeneous, but the spectrum then
        depends on the reference when balances are nonzero) or ``'numeraire'``
        (the model's own closure: balances fixed in numeraire units; ``A`` is
        not homogeneous and the reduction is reference-dependent; reported, not
        recommended).
    current_account
        ``'certificate'`` (default) excludes the current-account rows from the
        fast block and holds ``XN`` fixed; ``'fast'`` keeps ``nc-1`` of them
        with ``XN`` as fast unknowns, the naive partition that yields a
        structurally degenerate reduced matrix. ``balance_units`` does not
        apply in that mode (``metadata['balance_units']`` is None).
    speeds
        Optional positive country-factor adjustment speeds (length ``2 nc`` in
        the ``factor_labels`` order); ``A`` is replaced by ``diag(speeds) A``
        in the relative law only, so ``eig(R)`` equals ``eig(diag(speeds) A)``
        without its homogeneity zero.
    step, fd_order
        Finite-difference step for log unknowns, in ``(0, 1e-2]`` (the steps
        for the level unknowns ``T`` and, in the naive partition, ``XN`` are
        scaled by national income because they enter the lump-sum model
        linearly) and stencil order (2 or 4).
    dense, dense_step, dense_tol
        Independent check: re-solve the fast block at ``+-dense_step``
        perturbations of each factor price (``dense_step`` in ``(0, 1e-2]``)
        by chord iteration on ``g`` with the factorised ``g_u``, at most 30
        iterations, stopping when every fast-row block is below its own
        tolerance (goods rows ``1e-12 * max(1, max y)``, zero-profit rows
        ``1e-12 * max(1, max p)``, budget and current-account rows
        ``1e-12 * max(1, max|income|)``), and difference the excess-demand
        ratios; ``StabilityError`` when the re-solve does not converge or the
        maximum absolute disagreement with ``A`` exceeds
        ``dense_tol * max(1, max|A|)``.
    tol
        Classification band on ``max Re eig(R)``: ``'unstable'`` above
        ``tol``, ``'stable'`` below ``-tol``, ``'nonhyperbolic'`` in between.
    rank_tol
        Relative singular-value threshold for the rank of ``R``: singular
        values at or below ``rank_tol * max(1, largest)`` count as zero and a
        rank below ``2 nc - 1`` gives ``'degenerate'``. Separate from ``tol``
        so that a conservative sign band does not turn ordinary equilibria
        into degenerate ones.
    homogeneity_tol
        Relative threshold on ``homogeneity_error`` and ``walras_error``.
        Where the identity is exact for the evaluated map (homogeneity: the
        consistent accounting with ``balance_units`` other than
        ``'numeraire'`` or the naive partition; Walras: the consistent
        accounting) exceeding it raises ``StabilityError``. Otherwise
        (legacy accounting with production taxes or MATLAB precedence, or
        ``balance_units='numeraire'``) a homogeneity violation issues a
        ``RuntimeWarning``, ``metadata['homogeneous']`` is False and the
        relative reduction is conditional on the reference price.
    equilibrium_tol
        Acceptance threshold on ``max|r(x)|`` at the state (absolute, in the
        residuals' own units).

    Returns
    -------
    StabilityResult
        ``homogeneity_error = max|A 1| / max|A|``;
        ``walras_error = max_j |sum_i W_i A_ij + W_j e_j| / max_ij |W_i A_ij|``
        with ``W = (r K, w L)`` and ``e`` the excess demand at the state (the
        derivative of the identity ``W.e = 0`` on the fast-cleared manifold;
        at an equilibrium ``e = 0`` and it reduces to ``(r K, w L) A = 0``;
        exact only under the consistent accounting);
        ``fast_block_condition`` = the 2-norm condition number of ``g_u``
        after row and column max-abs equilibration (the raw block mixes log
        unknowns with levels in calibration units);
        ``equilibrium_residual`` and ``dense_check_error`` (absolute) and the
        closure/adjustment/qualification texts. ``metadata`` records the
        partition, the eigenvectors and singular values of ``R``, the
        absolute spectrum of ``diag(speeds) A``, the excess demand at the
        state, the world-identity check (relative, exact under the consistent
        accounting, gated at 1e-10) and every tolerance.
    """
    if not isinstance(calib, TradeCalibrationResult):
        raise TypeError("calib must be a TradeCalibrationResult")
    if not isinstance(result, TradeEquilibriumResult):
        raise TypeError("result must be a TradeEquilibriumResult")
    if adjustment != "relative_factor_prices":
        raise ValueError("adjustment must be 'relative_factor_prices'")
    if reference_factor not in ("w", "r"):
        raise ValueError("reference_factor must be 'w' or 'r'")
    if balance_units not in _BALANCE_UNITS:
        raise ValueError(f"balance_units must be one of {_BALANCE_UNITS}")
    if current_account not in _CURRENT_ACCOUNT:
        raise ValueError(f"current_account must be one of {_CURRENT_ACCOUNT}")
    if fd_order not in (2, 4):
        raise ValueError("fd_order must be 2 or 4")
    for name, value in (("step", step), ("dense_step", dense_step), ("dense_tol", dense_tol), ("tol", tol),
                        ("rank_tol", rank_tol), ("homogeneity_tol", homogeneity_tol)):
        if not np.isfinite(value) or value <= 0:
            raise ValueError(f"{name} must be finite and positive")
    for name, value in (("step", step), ("dense_step", dense_step)):
        if value > _MAX_LOG_STEP:
            raise ValueError(f"{name} is a step in log factor prices and must not exceed {_MAX_LOG_STEP:g}")
    if not result.converged:
        raise ValueError("reduced_stability requires a converged equilibrium")
    meta = result.metadata
    accounting = meta.get("accounting", "legacy")
    if accounting not in ("legacy", "consistent"):
        raise ValueError("result.metadata['accounting'] must be 'legacy' or 'consistent'")
    if meta.get("fiscal_closure", "lump_sum") not in ("lump_sum", "baseline", ""):
        raise NotImplementedError("reduced_stability supports the lump-sum fiscal closure only")
    if meta.get("capacity_margins") is not None or meta.get("recycling_params"):
        raise NotImplementedError("reduced_stability does not support capacity margins or revenue recycling")
    nc, ns, nfd = calib.nc, calib.ns, calib.n_final_demand
    n = nc*ns
    if nc < 2:
        raise ValueError("At least two countries are required for a relative adjustment law")
    x_star = np.asarray(result.x_sol, dtype=float).ravel().copy()
    if len(x_star) != 2*n+4*nc-1 or not np.isfinite(x_star).all():
        raise ValueError("result.x_sol must be a finite state vector of length 2*ns*nc + 4*nc - 1")
    country = _country_index(calib, reference)
    ref = country if reference_factor == "r" else nc+country
    if speeds is not None:
        speeds = np.asarray(speeds, dtype=float).ravel()
        if speeds.shape != (2*nc,) or not np.isfinite(speeds).all() or np.any(speeds <= 0):
            raise ValueError("speeds must be 2*nc finite positive adjustment speeds")

    schedules = _resolved_schedules(calib, result, tau, tau_fd, tauf, tauf_fd, accounting)
    raw = _residual_function(calib, result, accounting, schedules)
    rows = _rows(n, nc, accounting)

    # Index sets of the unknown vector.
    i_p = np.arange(0, n); i_y = np.arange(n, 2*n)
    i_r = np.arange(2*n, 2*n+nc); i_w = np.arange(2*n+nc, 2*n+2*nc)
    i_T = np.arange(2*n+2*nc, 2*n+3*nc); i_XN = np.arange(2*n+3*nc, 2*n+4*nc-1)
    slow = np.r_[i_r, i_w]
    fast = np.r_[i_p, i_y, i_T]
    fast_rows = np.r_[rows["goods"], rows["prices"], rows["budget"]]
    if current_account == "fast":
        fast = np.r_[fast, i_XN]
        fast_rows = np.r_[fast_rows, rows["current"][:nc-1]]
    factor_rows = np.r_[rows["labour"], rows["capital"]]
    L = np.asarray(calib.l_endow, dtype=float).ravel()
    K = np.asarray(calib.k_endow, dtype=float).ravel()
    xn_star = x_star[i_XN].copy()
    omega_star = float(np.exp(x_star[i_w])@L + np.exp(x_star[i_r])@K)

    def scale(xx):
        if current_account == "fast" or balance_units == "numeraire":
            return 1.
        if balance_units == "reference_factor_price":
            return float(np.exp(xx[2*n+ref]-x_star[2*n+ref]))
        return float(np.exp(xx[i_w])@L + np.exp(xx[i_r])@K)/omega_star

    def closed(xx):
        xx = xx.copy()
        if current_account != "fast":
            xx[i_XN] = xn_star*scale(xx)
        return raw(xx)

    r_star = closed(x_star)
    if len(r_star) != rows["total"]:
        raise StabilityError("Unexpected physical residual layout")
    residual = float(np.max(np.abs(r_star)))
    if equilibrium_tol is None:
        equilibrium_tol = float(meta.get("tol", 1e-8))
    if not np.isfinite(equilibrium_tol) or equilibrium_tol <= 0:
        raise ValueError("equilibrium_tol must be finite and positive")
    if not np.isfinite(residual) or residual > equilibrium_tol:
        raise StabilityError(f"State violates its equilibrium equations: max residual {residual:.3g} > {equilibrium_tol:.3g}")

    income = np.exp(x_star[i_w])*L + np.exp(x_star[i_r])*K + x_star[i_T]
    columns = np.r_[fast, slow]
    steps = np.full(len(columns), float(step))
    is_T = (columns >= i_T[0]) & (columns < i_T[-1]+1)
    steps[is_T] = step*np.maximum(1., np.maximum(np.abs(x_star[columns[is_T]]), np.abs(income[columns[is_T]-i_T[0]])))
    # XN (fast only in the naive partition) are levels too: an unscaled log-size step
    # drowns their linear effect in roundoff and hides the structural null space.
    is_XN = (columns >= i_XN[0]) & (columns < i_XN[-1]+1)
    steps[is_XN] = step*max(1., float(np.max(np.abs(income))))
    _, J = _jacobian(closed, x_star, columns, steps, fd_order)
    nf = len(fast)
    G = J[np.ix_(fast_rows, np.arange(nf))]
    J_fz = J[np.ix_(factor_rows, np.arange(nf, nf+2*nc))]
    J_fu = J[np.ix_(factor_rows, np.arange(nf))]
    J_gz = J[np.ix_(fast_rows, np.arange(nf, nf+2*nc))]
    try:
        lu = lu_factor(G, check_finite=False)
    except (ValueError, np.linalg.LinAlgError) as error:
        raise StabilityError("Fast block Jacobian could not be factorised") from error
    if not np.isfinite(lu[0]).all() or np.any(np.diag(lu[0]) == 0):
        raise StabilityError("Fast block Jacobian is singular: the partition does not define a fast-cleared map")
    # Condition number after row/column equilibration: the raw block mixes log
    # unknowns with level transfers in calibration units, so cond(G) itself
    # measures units rather than near-singularity.
    col_scale = np.max(np.abs(G), axis=0); col_scale[col_scale == 0] = 1.
    G_eq = G/col_scale[None, :]
    row_scale = np.max(np.abs(G_eq), axis=1); row_scale[row_scale == 0] = 1.
    condition = float(np.linalg.cond(G_eq/row_scale[:, None]))
    JF = J_fz - J_fu @ lu_solve(lu, J_gz, check_finite=False)
    endowment = np.r_[L, K]
    order = np.r_[np.arange(nc, 2*nc), np.arange(0, nc)]   # (labour, capital) rows -> (capital, labour)
    A = (-JF/endowment[:, None])[order]
    scale_A = float(np.max(np.abs(A)))
    if not np.isfinite(A).all() or scale_A == 0:
        raise StabilityError("Reduced Jacobian is not finite or vanishes")
    excess_star = (-r_star[factor_rows]/endowment)[order]
    weights = np.r_[np.exp(x_star[i_r])*K, np.exp(x_star[i_w])*L]
    homogeneity = float(np.max(np.abs(A@np.ones(2*nc)))/scale_A)
    # Derivative of W.e = 0 on the fast-cleared manifold: W A + W*e = 0 (W A = 0 at e = 0).
    walras = float(np.max(np.abs(weights@A + weights*excess_star))/np.max(np.abs(weights[:, None]*A)))

    # World value identity of the physical residuals (exact under consistent accounting).
    rng = np.random.default_rng(0)
    probe = x_star.copy()
    probe[np.r_[i_p, i_y, i_r, i_w]] += rng.uniform(-.05, .05, 2*n+2*nc)
    probe[i_T] += rng.uniform(-.05, .05, nc)*np.maximum(1., income)
    rp = closed(probe)
    pv = np.exp(probe[i_p]); yv = np.exp(probe[i_y]); rv = np.exp(probe[i_r]); wv = np.exp(probe[i_w])
    tax = np.asarray(calib.tax, dtype=float).ravel(order="F")
    identity_terms = np.r_[pv*rp[rows["goods"]], (1-tax)*yv*rp[rows["prices"]], wv*rp[rows["labour"]],
                           rv*rp[rows["capital"]], rp[rows["budget"]]]
    world_identity = float(abs(identity_terms.sum())/max(1., np.max(np.abs(identity_terms))))

    # Identities that are exact for the evaluated map are acceptance checks: a violation
    # is finite-difference or evaluator error, never a modelling caveat.
    homogeneity_exact = accounting == "consistent" and (current_account == "fast" or balance_units != "numeraire")
    walras_exact = accounting == "consistent"
    if homogeneity_exact and not homogeneity <= homogeneity_tol:
        raise StabilityError(
            f"Reduced map violates exact homogeneity of degree zero: max|A 1|/max|A| = {homogeneity:.3g} > "
            f"{homogeneity_tol:.3g} (finite-difference error; step={step:g}, fd_order={fd_order})")
    if walras_exact and not walras <= homogeneity_tol:
        raise StabilityError(
            f"Reduced map violates the exact value-weighted Walras identity: {walras:.3g} > {homogeneity_tol:.3g} "
            f"(finite-difference error; step={step:g}, fd_order={fd_order})")
    if walras_exact and not world_identity <= _WORLD_IDENTITY_TOL:
        raise StabilityError(
            f"Consistent residuals violate the world value identity: {world_identity:.3g} > {_WORLD_IDENTITY_TOL:g}")

    homogeneous = bool(homogeneity <= homogeneity_tol)
    if not homogeneous:
        warnings.warn(
            f"Reduced excess-demand map is not homogeneous of degree zero (relative error "
            f"{homogeneity:.2e}); the relative-price reduction is then conditional on the reference "
            f"price and metadata['absolute_eigenvalues'] holds the spectrum of the full map. Under "
            f"the legacy accounting this happens with production taxes or MATLAB operator precedence.",
            RuntimeWarning, stacklevel=2)

    A_law = A if speeds is None else speeds[:, None]*A
    keep = np.flatnonzero(np.arange(2*nc) != ref)
    R = A_law[np.ix_(keep, keep)] - A_law[ref, keep][None, :]
    values, vectors = eig(R)
    order_eig = np.argsort(-values.real, kind="stable")
    values, vectors = values[order_eig], vectors[:, order_eig]
    singular = np.linalg.svd(R, compute_uv=False)
    rank = int(np.sum(singular > rank_tol*max(1., float(singular[0]))))
    classification = _classify(values, rank, 2*nc-1, tol)
    absolute = np.linalg.eigvals(A_law)
    absolute = absolute[np.argsort(-absolute.real, kind="stable")]

    dense_error = None
    if dense:
        y_level = float(np.max(np.exp(x_star[i_y])))
        p_level = float(np.max(np.exp(x_star[i_p])))
        income_level = float(np.max(np.abs(income)))
        row_tolerance = np.r_[np.full(n, 1e-12*max(1., y_level)), np.full(n, 1e-12*max(1., p_level)),
                              np.full(len(fast_rows)-2*n, 1e-12*max(1., income_level))]
        dense_error = _dense_check(closed, x_star, fast, slow, fast_rows, factor_rows, lu, endowment, order,
                                   A, dense_step, row_tolerance)
        if dense_error > dense_tol*max(1., scale_A):
            raise StabilityError(f"Independent fast-block re-solve (dense_step={dense_step:g}) disagrees with "
                                 f"the Schur complement: {dense_error:.3g} > {dense_tol*max(1., scale_A):.3g}; "
                                 f"the check itself carries an O(dense_step^2) error")

    codes = tuple(calib.country_codes) if len(calib.country_codes) == nc else tuple(f"C{k:02d}" for k in range(nc))
    labels = tuple(f"r[{c}]" for c in codes) + tuple(f"w[{c}]" for c in codes)
    if current_account == "fast":
        closure = (f"accounting={accounting}; fast block = goods, zero profit, fiscal budgets, current accounts; "
                   f"foreign balances XN are fast unknowns (balance_units not applicable)")
    else:
        closure = (f"accounting={accounting}; fast block = goods, zero profit, fiscal budgets; foreign balances "
                   f"fixed in {balance_units.replace('_', ' ')} units; current accounts not imposed "
                   f"(certificate only)")
    if not homogeneous:
        closure += f"; {_NON_HOMOGENEOUS_CAVEAT}"
    adjustment_text = (f"d log(f_i/{labels[ref]})/dt = e_i - e_ref with e = factor demand/endowment - 1 "
                       f"for every factor price f_i other than {labels[ref]}"
                       + ("; country-factor speeds applied" if speeds is not None else "")
                       + "; producer prices, gross outputs and national fiscal budgets clear at each factor-price vector.")
    speeds_copy = None if speeds is None else speeds.copy()
    metadata = dict(
        accounting=accounting, sigma=float(meta.get("sigma", 0.)),
        balance_units=None if current_account == "fast" else balance_units,
        current_account=current_account, reference_index=int(ref), keep_indices=keep,
        eigenvectors=vectors, singular_values=singular, excess_demand_at_state=excess_star,
        absolute_eigenvalues=absolute, homogeneous=homogeneous, homogeneity_tol=float(homogeneity_tol),
        homogeneity_exact=bool(homogeneity_exact),
        world_identity_error=world_identity, world_identity_exact=bool(walras_exact),
        world_identity_tol=_WORLD_IDENTITY_TOL, walras_identity_exact=bool(walras_exact), step=float(step),
        fd_order=int(fd_order), dense=bool(dense), dense_step=float(dense_step),
        dense_tol=float(dense_tol), tol=float(tol), rank_tol=float(rank_tol),
        equilibrium_tol=float(equilibrium_tol), speeds=speeds_copy, fast_block_dimension=int(nf),
        jacobian_evaluations=int(len(columns)*fd_order+1), tariff_revenue_mode=meta.get("tariff_revenue_mode"),
        reduced_jacobian_order="rows and columns follow factor_labels: r[0..nc-1] then w[0..nc-1]",
        excess_demand_sign="positive = excess demand (factor demand above endowment)",
        experimental=True)
    for array in (A, R, values, keep, vectors, singular, excess_star, absolute, speeds_copy):
        if array is not None:
            array.flags.writeable = False
    return StabilityResult(
        classification=classification, maximum_real_eigenvalue=float(np.max(values.real)),
        eigenvalues=values, reduced_jacobian=A, relative_matrix=R, rank=rank,
        homogeneity_error=homogeneity, walras_error=walras, fast_block_condition=condition,
        equilibrium_residual=residual, dense_check_error=dense_error, adjustment=adjustment_text,
        closure=closure, qualification=QUALIFICATION, reference=labels[ref], factor_labels=labels,
        metadata=metadata)


def _dense_check(closed, x_star, fast, slow, fast_rows, factor_rows, lu, endowment, order, A, h, row_tolerance):
    """Recompute ``A`` by re-solving the fast block at perturbed factor prices.

    Chord iteration ``u <- u - g_u^{-1} g(u, z)`` with the factorised fast block,
    at most 30 iterations, until ``|g_i| <= row_tolerance_i`` for every fast row.
    """

    def excess(z):
        xx = x_star.copy()
        xx[slow] = z
        u = xx[fast].copy()
        for _ in range(30):
            xx[fast] = u
            g = closed(xx)[fast_rows]
            if np.all(np.abs(g) <= row_tolerance):
                break
            u = u - lu_solve(lu, g, check_finite=False)
        else:
            raise StabilityError(f"Fast block re-solve did not converge in the dense check "
                                 f"(dense_step={h:g}, 30 chord iterations)")
        r = closed(xx)
        return (-r[factor_rows]/endowment)[order]

    z0 = x_star[slow].copy()
    direct = np.empty_like(A)
    for j in range(len(z0)):
        zp, zm = z0.copy(), z0.copy()
        zp[j] += h; zm[j] -= h
        direct[:, j] = (excess(zp)-excess(zm))/(2*h)
    return float(np.max(np.abs(direct-A)))
