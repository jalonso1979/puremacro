"""Exact block Newton for nested CES technology on consistent trade accounting.

The equations are those of ``accounting="consistent"`` in
:mod:`puremacro.trade._accounting` (producer/purchaser valuation, homogeneous
factor costs, output-revenue taxes, tax-inclusive final expenditure shares, fixed
baseline foreign saving in numeraire units, first producer price equal to one),
with the Leontief/flat-CES intermediate technology replaced by a benchmark
normalized three-tier nested CES: value added versus materials
(``sigma_va_materials``), material sectors (``sigma_sectors``), origins within a
sector (``sigma_origins``), plus puremacro's capital/labour nest (``rho_va``).
Preferences, fiscal closure, foreign balance and numeraire are unchanged, so at
flat technology (``sigma_va_materials = 0``, ``sigma_sectors = sigma_origins``,
``rho_va = 1``) the residual equals ``_accounting.evaluate(sigma=...)`` row by
row and :func:`puremacro.trade.compute_hicksian_welfare` accepts the result.

Technology (cell ``j`` buys origin cell ``i`` at ``r_ij = p_i tau_ij``)::

    h_sj = CES_so({r_ij}_{i in s}; u0_ij)          u0_ij = a0_ij / sum_{k in s} a0_kj
    m_j  = CES_ss({h_sj}_s; eta0_sj)               eta0_sj = sum_{i in s} a0_ij / A0_j
    v_j  = CES_rho({w_c, r_c}; (1-alpha_j, alpha_j))
    c_j  = CES_sv({v_j, m_j}; (v0_j, A0_j)/(1-t_j))    (unit cost, one at the benchmark)
    a_ij = a0_ij (c_j/m_j)^sv (m_j/h_sj)^ss (h_sj/r_ij)^so
    b_Lj = b0_Lj (c_j/v_j)^sv (v_j/w_c)^rho,   b_Kj = b0_Kj (c_j/v_j)^sv (v_j/r_c)^rho
    (1 - t_j) c_j = sum_i a_ij r_ij + b_Lj w_c + b_Kj r_c      (Euler identity)

with ``CES_sigma(x; s) = (sum_i s_i x_i^(1-sigma))^(1/(1-sigma))``, the
geometric mean at ``sigma = 1`` and one on an empty nest.

The Newton step solves the exact Jacobian system of the level residual through
two ``M x M`` LU factorizations (price block ``diag(p) - diag(c) alpha^T`` and
goods block ``I - a``) and a small macro Schur complement of size ``3N + 1``
(log r, log w, T and the value of the omitted goods equation; the ``N - 1``
foreign-balance rows are an identity block and are eliminated first). The
derivation is in ``docs/trade_ces_newton.md``. This is a local linear-algebra
identity conditional on the three factorizations being nonsingular; it proves
neither existence nor uniqueness of an equilibrium nor global convergence of
the damped iteration.

Acceptance: a state is returned only when (i) the imposed residual rows divided
by their benchmark scales are at most ``tol``, (ii) every level row that
postprocessing and :func:`puremacro.trade.compute_hicksian_welfare` audit,
including the omitted goods equation and the realized foreign balances implied
by Walras' law, is at most ``tol * max(scale)`` with no negative final
expenditure (the Newton loop keeps iterating until both hold), and (iii)
:func:`certify_ces_equilibrium`, an independent flow reconstruction, is at most
``certificate_tol``. Everything else raises :class:`CESNewtonError`.

Ported from the IO research workspace (under ``headlinePaper/rebuild/``:
``ces_newton.py``, ``ces_continuation.py``, ``vendor/puremacro/trade/corrected/ces.py``
and the proof ``empirical_2019/CES_NEWTON_PROOF.md``), whose limitations apply verbatim:
"The proof is a conditional local identity, not a convergence or
global-equilibrium theorem" (README.md); "The discrete checks do not certify a
continuous branch between stages. ... failed procedures are not nonexistence
certificates" (README.md); "all CES elasticities are explicitly stylized"
(CHANGES.md).
"""
from __future__ import annotations

from dataclasses import dataclass, field, replace
import time
from typing import Any, Callable

import numpy as np
import scipy.linalg as la
from scipy.sparse.linalg import LinearOperator

from puremacro.reports import df_to_latex, df_to_markdown, df_to_typst
from ._accounting import _parameters, postprocess as _postprocess_consistent
from ._results import TradeCalibrationResult, TradeEquilibriumResult
from .solver import _resolve_tariffs, build_initial_guess

__all__ = [
    "NestedCESTechnology",
    "CESBlockJacobian",
    "CESBlockNewtonResult",
    "CESNewtonError",
    "solve_ces_block_newton",
    "continue_tariff_homotopy",
    "certify_ces_equilibrium",
]


class CESNewtonError(RuntimeError):
    """Raised when the block Newton or the homotopy cannot certify a state.

    Attributes carry what was learned: ``history`` (per-iteration records),
    ``macro_condition`` (last macro Schur condition number), ``certificate``
    (last certificate, if one was computed), ``stages`` and ``failed_stages``
    (homotopy records) and ``last_result`` (the last certified stage, if any).
    """

    def __init__(self, message: str, *, history=(), macro_condition=float("nan"), certificate=None,
                 stages=(), failed_stages=(), last_result=None):
        super().__init__(message)
        self.history = tuple(history)
        self.macro_condition = float(macro_condition)
        self.certificate = certificate
        self.stages = tuple(stages)
        self.failed_stages = tuple(failed_stages)
        self.last_result = last_result


# ---------------------------------------------------------------------------
# Technology description
# ---------------------------------------------------------------------------

@dataclass(frozen=True)
class NestedCESTechnology:
    """Elasticities of the three-tier nested CES technology plus the K/L nest.

    ``sigma_va_materials`` (IO ``sv``): value added versus the material
    composite. ``sigma_sectors`` (IO ``ss``): across material sectors; ``None``
    means equal to ``sigma_origins``. ``sigma_origins`` (IO ``so``): across
    origins within a sector. ``rho_va``: capital versus labour (one is
    Cobb-Douglas, puremacro's default). All elasticities are stylized inputs,
    not estimates. Construct through :meth:`validated` to check the domain;
    the plain constructor never raises.
    """

    sigma_va_materials: float = 0.0
    sigma_sectors: float | None = None
    sigma_origins: float = 0.0
    rho_va: float = 1.0

    @classmethod
    def validated(cls, **kw: Any) -> "NestedCESTechnology":
        """Build and validate: finite, ``>= 0`` elasticities and ``rho_va > 0``."""
        tech = cls(**kw)
        _check_technology(tech)
        return tech

    @classmethod
    def from_flexible(cls, config: Any) -> "NestedCESTechnology":
        """Map a ``FlexibleTechnologyConfig`` (``sigma_y``, ``sigma_inter``, ``rho_va``).

        ``sigma_y`` becomes ``sigma_va_materials``; ``sigma_inter`` becomes both
        ``sigma_sectors`` and ``sigma_origins`` (the flexible module has no sector
        tier); ``rho_va`` is kept. Capacity penalties, LES and markups are not
        part of this technology and are ignored.
        """
        return cls.validated(sigma_va_materials=float(getattr(config, "sigma_y", 0.0)),
                             sigma_sectors=None,
                             sigma_origins=float(getattr(config, "sigma_inter", 0.0)),
                             rho_va=float(getattr(config, "rho_va", 1.0)))

    @property
    def elasticities(self) -> tuple[float, float, float, float]:
        """Resolved ``(sv, ss, so, rho)`` with ``ss`` defaulting to ``so``."""
        so = float(self.sigma_origins)
        ss = so if self.sigma_sectors is None else float(self.sigma_sectors)
        return float(self.sigma_va_materials), ss, so, float(self.rho_va)

    @property
    def is_flat(self) -> bool:
        """True when the technology equals ``_accounting.evaluate(sigma=sigma_origins)``."""
        sv, ss, so, rho = self.elasticities
        return sv == 0.0 and ss == so and rho == 1.0

    def to_dict(self) -> dict[str, float]:
        """Resolved elasticities as a plain dict (``sigma_sectors`` defaulted to ``sigma_origins``)."""
        sv, ss, so, rho = self.elasticities
        return {"sigma_va_materials": sv, "sigma_sectors": ss, "sigma_origins": so, "rho_va": rho}


def _check_technology(tech: NestedCESTechnology) -> tuple[float, float, float, float]:
    if not isinstance(tech, NestedCESTechnology):
        raise TypeError("technology must be a NestedCESTechnology")
    values = tech.elasticities
    if not np.all(np.isfinite(values)):
        raise ValueError("Technology elasticities must be finite")
    if min(values[:3]) < 0:
        raise ValueError("CES elasticities must be nonnegative")
    if values[3] <= 0:
        raise ValueError("rho_va must be strictly positive")
    return values


def _ces_index(prices, shares, sigma, axis=0):
    """Benchmark-normalized CES index; one on empty nests. Complex-step safe."""
    mass = np.sum(shares, axis=axis)
    if abs(sigma - 1.0) < 1e-8:
        value = np.exp(np.sum(shares * np.log(prices), axis=axis))
    else:
        total = np.sum(shares * prices ** (1.0 - sigma), axis=axis)
        base = np.where(mass > 0, total, 1.0)
        value = base ** (1.0 / (1.0 - sigma))
    return np.where(mass > 0, value, 1.0)


# ---------------------------------------------------------------------------
# Private model: calibration-derived constants and the complex-safe residual
# ---------------------------------------------------------------------------

class _Model:
    """Static data of one (calibration, tariff schedule, technology) problem."""

    def __init__(self, calib, ta, tf, tauf_vec, tauf_fd_vec, technology):
        if not isinstance(calib, TradeCalibrationResult):
            raise TypeError("calib must be a TradeCalibrationResult")
        self.calib = calib
        self.technology = technology
        self.elasticities = _check_technology(technology)
        nc, ns, nfd = calib.nc, calib.ns, calib.n_final_demand
        n = nc * ns
        self.nc, self.ns, self.nfd, self.n = nc, ns, nfd, n
        self.dim = 2 * n + 4 * nc - 1
        fd_tax, basic0, inv = _parameters(calib)
        self.fd_tax = np.asarray(fd_tax, dtype=float).reshape(nfd, nc)
        self.basic0 = np.asarray(basic0, dtype=float).reshape(nfd, nc)
        self.active = self.basic0 > 0
        self.inv = inv
        ta = np.asarray(ta, dtype=float)
        tf = np.asarray(tf, dtype=float)
        for schedule, shape in ((ta, (n, ns, nc)), (tf, (n, nfd, nc))):
            if schedule.shape != shape or not np.isfinite(schedule).all() or np.any(schedule <= 0):
                raise ValueError("Tariff multipliers must have the expected shape and be finite and positive")
            for c in range(nc):
                if not np.allclose(schedule[c * ns:(c + 1) * ns, :, c], 1.0, rtol=0, atol=1e-14):
                    raise ValueError("Import tariffs must be one on domestic transactions")
        self.ta, self.tf = ta, tf
        self.tauf_vec, self.tauf_fd_vec = tauf_vec, tauf_fd_vec
        self.tau = ta.reshape(n, n, order="F")                     # [origin i, buyer j]
        self.a0 = np.asarray(calib.a, dtype=float).reshape(n, n, order="F")
        self.afd = np.asarray(calib.afd, dtype=float)              # (n, nfd, nc)
        self.t = np.asarray(calib.tax, dtype=float).ravel(order="F")
        self.alpha = np.asarray(calib.alpha, dtype=float).ravel(order="F")
        beta = np.asarray(calib.beta, dtype=float).ravel(order="F")
        self.v0 = (1.0 / self.alpha) ** self.alpha * (1.0 / (1.0 - self.alpha)) ** (1.0 - self.alpha) / beta
        self.b0L = (1.0 - self.alpha) * self.v0
        self.b0K = self.alpha * self.v0
        self.A0 = self.a0.sum(axis=0)
        a03 = self.a0.reshape(nc, ns, n)                            # [origin country, origin sector, buyer]
        sector_mass = a03.sum(axis=0)                               # (ns, n)
        self.u0 = np.divide(a03, sector_mass[None], out=np.zeros_like(a03), where=sector_mass[None] > 0)
        self.eta0 = np.divide(sector_mass, self.A0[None], out=np.zeros_like(sector_mass), where=self.A0[None] > 0)
        self.top_v = self.v0 / (1.0 - self.t)
        self.top_m = self.A0 / (1.0 - self.t)
        if np.max(np.abs(self.top_v + self.top_m - 1.0)) > 1e-9:
            raise ValueError("Production benchmark shares do not sum to one; the calibration table is not balanced")
        self.theta = np.asarray(calib.theta, dtype=float).reshape(nfd, nc)
        self.L = np.asarray(calib.l_endow, dtype=float).ravel()
        self.K = np.asarray(calib.k_endow, dtype=float).ravel()
        self.B0 = np.asarray(calib.invforT, dtype=float).ravel()
        self.y0 = np.asarray(calib.ytot, dtype=float).ravel(order="F")
        if calib.T is not None:
            T0 = np.asarray(calib.T, dtype=float).ravel()
        elif calib.TT is not None and calib.TTfd is not None:
            T0 = (np.asarray(calib.TT, dtype=float) + np.asarray(calib.TTfd, dtype=float)).ravel()
        else:
            T0 = np.zeros(nc)
        self.T0 = T0
        self.gdp0 = self.L + self.K + T0
        self.country = np.repeat(np.arange(nc), ns)
        goods_scale = self.y0.copy()
        goods_scale[0] = 1.0
        self.scale = np.concatenate([goods_scale, np.ones(n), self.L, self.K,
                                     self.gdp0[:nc - 1], self.gdp0])
        if not np.all(np.isfinite(self.scale)) or np.any(self.scale <= 0):
            raise ValueError("Residual scales require positive benchmark outputs, endowments and income")

    # -- unknowns ----------------------------------------------------------
    def unpack(self, x):
        n, nc = self.n, self.nc
        x = np.asarray(x)
        if x.shape != (self.dim,):
            raise ValueError(f"State vector must have length {self.dim}, got {x.shape}")
        p = np.exp(x[:n]); y = np.exp(x[n:2 * n])
        r = np.exp(x[2 * n:2 * n + nc]); w = np.exp(x[2 * n + nc:2 * n + 2 * nc])
        T = x[2 * n + 2 * nc:2 * n + 3 * nc]; XN = x[2 * n + 3 * nc:]
        B = np.concatenate([XN, [-np.sum(XN)]])
        return p, y, r, w, T, XN, B

    # -- technology ----------------------------------------------------------
    def technology_at(self, p, r, w):
        """Unit cost, input coefficients and factor requirements at prices ``(p, r, w)``."""
        sv, ss, so, rho = self.elasticities
        nc, ns, n = self.nc, self.ns, self.n
        rr3 = (p[:, None] * self.tau).reshape(nc, ns, n)
        h = _ces_index(rr3, self.u0, so, axis=0)                        # (ns, n)
        m = _ces_index(h, self.eta0, ss, axis=0)                        # (n,)
        wc, rc = w[self.country], r[self.country]
        vhat = _ces_index(np.stack([wc, rc]), np.stack([1.0 - self.alpha, self.alpha]), rho, axis=0)
        c = _ces_index(np.stack([vhat, m]), np.stack([self.top_v, self.top_m]), sv, axis=0)
        a3 = self.u0 * self.eta0[None] * self.A0[None, None]
        a = a3 * ((c / m) ** sv)[None, None, :] * ((m[None, :] / h) ** ss)[None, :, :] * (h[None] / rr3) ** so
        bL = self.b0L * (c / vhat) ** sv * (vhat / wc) ** rho
        bK = self.b0K * (c / vhat) ** sv * (vhat / rc) ** rho
        return dict(c=c, a=a.reshape(n, n), bL=bL, bK=bK, h=h, m=m, vhat=vhat, rr=rr3.reshape(n, n))

    # -- residual blocks (same keys as _accounting.evaluate) ------------------
    def blocks(self, x):
        nc, ns, nfd, n = self.nc, self.ns, self.nfd, self.n
        p, y, r, w, T, XN, B = self.unpack(x)
        tech = self.technology_at(p, r, w)
        c, a, bL, bK = tech["c"], tech["a"], tech["bL"], tech["bK"]
        Z = a * y[None, :]
        labor, capital = bL * y, bK * y
        Q = np.sum(self.afd * p[:, None, None] * self.tf, axis=0)      # (nfd, nc)
        Q = np.where(self.active, Q, 1.0)
        P = Q / (1.0 - self.fd_tax)
        income = w * self.L + r * self.K + T
        expenditure = self.theta * income[None]
        expenditure = expenditure.astype(np.result_type(expenditure, B), copy=True)
        expenditure[self.inv] = expenditure[self.inv] - B
        quantity = expenditure / P
        F = self.afd * quantity[None]                                   # (n, nfd, nc)
        final_tax = self.fd_tax * expenditure
        production_tax = self.t * p * y
        duties_i = (self.tau - 1.0) * p[:, None] * Z                     # (n, n)
        duties_f = (self.tf - 1.0) * p[:, None, None] * F
        tariffs = (duties_i.sum(axis=0).reshape(nc, ns).sum(axis=1)
                   + duties_f.sum(axis=(0, 1)))
        government = production_tax.reshape(nc, ns).sum(axis=1) + final_tax.sum(axis=0) + tariffs
        zvals = Z * p[:, None]
        fm = F.reshape(n, nfd * nc, order="F")
        fvals = fm * p[:, None]
        trade = zvals.reshape(nc, ns, nc, ns).sum(axis=(1, 3)) + fvals.reshape(nc, ns, nc, nfd).sum(axis=(1, 3))
        np.fill_diagonal(trade, 0.0)
        net_exports = trade.sum(axis=1) - trade.sum(axis=0)
        goods = y - Z.sum(axis=1) - fm.sum(axis=1)
        prices = c - p
        labor_gap = self.L - labor.reshape(nc, ns).sum(axis=1)
        capital_gap = self.K - capital.reshape(nc, ns).sum(axis=1)
        budget_gap = T - government
        balances = XN - self.B0[:nc - 1]
        solver_goods = goods.copy()
        solver_goods[0] = p[0] - 1.0
        residuals = np.concatenate([solver_goods, prices, labor_gap, capital_gap, balances, budget_gap])
        physical = np.concatenate([goods, prices, labor_gap, capital_gap, net_exports - B, budget_gap])
        return dict(p=p.reshape(1, ns, nc, order="F"), pp=c.reshape(1, ns, nc, order="F"),
                    y=y.reshape(1, ns, nc, order="F"), r=r.reshape(1, 1, nc), w=w.reshape(1, 1, nc),
                    T=T.reshape(1, 1, nc), B=B.reshape(1, nc), XN=XN, Z=Z, F=fm,
                    Q=Q[None], P=P[None], c=quantity[None], income=income.reshape(1, 1, nc),
                    expenditure=expenditure[None], final_tax=final_tax[None],
                    production_tax=production_tax.reshape(1, ns, nc, order="F"),
                    labor=labor.reshape(1, ns, nc, order="F"), capital=capital.reshape(1, ns, nc, order="F"),
                    duties_i=duties_i.reshape(n, ns, nc, order="F"), duties_f=duties_f, tariffs=tariffs,
                    trade=trade, producer_values=np.hstack([zvals, fvals]), ta=self.ta, tf=self.tf,
                    basic0=self.basic0[None], fd_tax=self.fd_tax[None], residuals=residuals,
                    physical_residuals=physical, goods_gap=goods, net_exports_gap=net_exports - B,
                    government=government,
                    # extra fields used by the Jacobian and the certificate
                    pv=p, yv=y, rv=r, wv=w, Tv=T, tech=tech, Fq=F, quantity=quantity, Pmat=P,
                    Qmat=Q, exp_mat=expenditure, income_v=income)

    def residual(self, x):
        return self.blocks(x)["residuals"]

    def _label(self, kind, index):
        codes = tuple(self.calib.country_codes or ())
        sectors = tuple(self.calib.sector_codes or ())
        if kind == "cell":
            c, s = divmod(int(index), self.ns)
            country = codes[c] if c < len(codes) else f"country {c}"
            sector = sectors[s] if s < len(sectors) else f"sector {s}"
            return f"{country}/{sector}"
        return codes[int(index)] if int(index) < len(codes) else f"country {int(index)}"

    def inadmissibility(self, b) -> str | None:
        """First violated admissibility condition of a block evaluation, or ``None`` if admissible.

        Positive finite producer prices, outputs, rental rates, wages and
        household incomes; finite transfers and residuals; positive purchaser
        indices and final quantities on active categories and exactly zero
        quantities on structurally absent ones.
        """
        for key, label, kind in (("pv", "producer price", "cell"), ("yv", "output", "cell"),
                                 ("rv", "rental rate", "country"), ("wv", "wage", "country"),
                                 ("income_v", "household income", "country")):
            v = b[key]
            bad = ~np.isfinite(v) | (np.real(v) <= 0)
            if np.any(bad):
                k = int(np.argmax(bad))
                return f"{label} of {self._label(kind, k)} is {np.real(v)[k]:.4g}"
        if not np.isfinite(b["Tv"]).all() or not np.isfinite(b["residuals"]).all():
            return "nonfinite transfers or residuals"
        q, P = np.real(b["quantity"]), np.real(b["Pmat"])
        if not np.isfinite(q).all():
            return "nonfinite final quantities"
        bad = self.active & (P <= 0)
        if np.any(bad):
            k, c = (int(i) for i in np.argwhere(bad)[0])
            return f"purchaser index of final-use category {k} in {self._label('country', c)} is {P[k, c]:.4g}"
        bad = self.active & ~(q > 0)
        if np.any(bad):
            k, c = (int(i) for i in np.argwhere(bad)[0])
            spend = float(np.real(b["exp_mat"])[k, c])
            what = " (investment, which carries the fixed foreign saving)" if k == self.inv else ""
            return (f"final quantity of active category {k}{what} in {self._label('country', c)} is "
                    f"{q[k, c]:.4g} (expenditure {spend:.4g})")
        bad = ~self.active & (q != 0)
        if np.any(bad):
            k, c = (int(i) for i in np.argwhere(bad)[0])
            return f"structurally absent category {k} in {self._label('country', c)} has quantity {q[k, c]:.4g}"
        return None

    def admissible(self, b) -> bool:
        """True when :meth:`inadmissibility` finds no violation."""
        return self.inadmissibility(b) is None

    def level_audit(self, b) -> tuple[float, bool]:
        """The absolute audit postprocessing applies: ``max`` over the level residuals and the
        physical rows (every goods equation including the omitted one, realized foreign
        balances) and demand feasibility (no negative final expenditure)."""
        rows = np.concatenate([np.real(b["residuals"]), np.real(b["physical_residuals"])])
        worst = float(np.max(np.abs(rows))) if np.isfinite(rows).all() else float("inf")
        return worst, bool(np.all(np.real(b["exp_mat"]) >= 0))

    def absolute_tol(self, tol) -> float:
        """Absolute level bound ``tol * max(scale)`` recorded as ``metadata['tol']``."""
        return float(tol * np.max(self.scale))


def _build_model(calib, tau, tau_fd, tauf, tauf_fd, technology):
    ta, tf, tauf_vec, tauf_fd_vec = _resolve_tariffs(calib, tau, tau_fd, tauf, tauf_fd)
    return _Model(calib, ta, tf, tauf_vec, tauf_fd_vec, technology)


# ---------------------------------------------------------------------------
# Exact Jacobian action
# ---------------------------------------------------------------------------

class CESBlockJacobian:
    """Exact Jacobian of the consistent-accounting CES residual at one point.

    Layout: unknowns ``[log p (M); log y (M); log r (N); log w (N); T (N); XN (N-1)]``,
    rows ``[goods (M, row 0 = numeraire); prices (M); labour (N); capital (N);
    balances (N-1); budgets (N)]``, exactly as ``_accounting.evaluate``.

    ``solve(r)`` returns ``J^{-1} r`` by block elimination: two dense ``M x M`` LU
    factorizations (``diag(p) - diag(c) alpha^T`` and ``I - a``) and one LU of the
    ``(3N+1) x (3N+1)`` macro Schur complement in ``(log r, log w, T, mu)`` where
    ``mu`` is the value of the omitted goods equation. ``apply(v)`` returns
    ``J v`` from the same analytic differentials. ``macro_condition`` is the
    condition number of the row- and column-scaled Schur block. ``n`` is the
    number of cells ``M`` (not the system dimension); ``dim == shape[0] ==
    2M + 4N - 1`` is the number of unknowns.

    Memory (tracemalloc on a synthetic 77 by 11 table, ``M = 847``, one
    ``M x M`` double array = 5.7 MB): the problem data retain about ``3.3 M^2``
    doubles, the flows of the linearization point ``5.9 M^2`` and the operator
    itself ``6.2 M^2`` (two LU factors, the cost shares, nest shares and duty
    flows), about ``15 M^2`` together (88 MB); a full damped Newton solve peaks
    at about ``22 M^2`` (127 MB), because the line search evaluates a trial
    state while the current one is alive. ``max_cells`` (default 4000, a solve
    peak near 2.8 GB) refuses larger tables before allocating.
    """

    def __init__(self, calib, x, *, tau=None, tau_fd=None, tauf=None, tauf_fd=None,
                 technology: NestedCESTechnology = NestedCESTechnology(), block_size: int = 16,
                 max_cells: int = 4000, _model=None, _blocks=None):
        model = _model if _model is not None else _build_model(calib, tau, tau_fd, tauf, tauf_fd, technology)
        if model.n > max_cells:
            raise ValueError(f"{model.n} cells exceed max_cells={max_cells}; the operator with its data needs "
                             f"about {15 * model.n ** 2 * 8 / 1e9:.2f} GB and a Newton solve peaks near "
                             f"{22 * model.n ** 2 * 8 / 1e9:.2f} GB")
        if not isinstance(block_size, (int, np.integer)) or block_size < 1:
            raise ValueError("block_size must be a positive integer")
        self.model = model
        self.n, self.nc, self.ns, self.nfd = model.n, model.nc, model.ns, model.nfd
        self.dim = model.dim
        self.shape = (model.dim, model.dim)
        self.technology = model.technology
        x = np.asarray(x, dtype=float).ravel()
        b = _blocks if _blocks is not None else model.blocks(x)
        if not np.isfinite(b["residuals"]).all():
            raise ValueError("The residual is not finite at the linearization point")
        self.x = x
        self.blocks = b
        self.residual = b["residuals"]
        self.scale = model.scale
        self._factor()
        self._assemble_macro(block_size)

    # -- current shares and factorizations ----------------------------------
    def _factor(self):
        m, b = self.model, self.blocks
        n, nc, ns = self.n, self.nc, self.ns
        tech = b["tech"]
        p, y, r, w = b["pv"], b["yv"], b["rv"], b["wv"]
        c, a, bL, bK, rr = tech["c"], tech["a"], tech["bL"], tech["bK"], tech["rr"]
        net = (1.0 - m.t) * c
        self.p, self.y, self.r, self.w, self.c = p, y, r, w, c
        self.a, self.bL, self.bK = a, bL, bK
        self.alpha = a * rr / net[None, :]                                # cost shares (n, n)
        self.betaL = bL * w[m.country] / net
        self.betaK = bK * r[m.country] / net
        alpha3 = self.alpha.reshape(nc, ns, n)
        sector = alpha3.sum(axis=0)                                       # (ns, n)
        mass = sector.sum(axis=0)
        self.u = np.divide(alpha3, sector[None], out=np.zeros_like(alpha3), where=sector[None] > 0)
        self.eta = np.divide(sector, mass[None], out=np.zeros_like(sector), where=mass[None] > 0)
        va = self.betaL + self.betaK
        self.thetaL = np.divide(self.betaL, va, out=np.zeros_like(va), where=va > 0)
        self.thetaK = 1.0 - self.thetaL
        self.Z = b["Z"]                                                    # a * y, already in the blocks
        self.Zrow = self.Z.sum(axis=1)
        self.D = (m.tau - 1.0) * p[:, None] * self.Z                       # duty flows (n, n)
        self.D_sector = self.D.reshape(nc, ns, n).sum(axis=0)              # (ns, n)
        self.duty = self.D.sum(axis=0)                                     # per buyer cell
        self.q, self.P, self.Q = b["quantity"], b["Pmat"], b["Qmat"]
        self.Qt = np.sum(m.afd * p[:, None, None], axis=0)
        self.Wq = m.afd * m.tf * p[:, None, None]                          # dQ weights
        self.Wt = m.afd * p[:, None, None]                                 # dQtilde weights
        self.wL, self.rK = w * m.L, r * m.K
        self.tpy = m.t * p * y
        self.price_lu = la.lu_factor(np.diag(p) - c[:, None] * self.alpha.T, check_finite=False)
        self.goods_lu = la.lu_factor(np.eye(n) - a, check_finite=False)
        C = np.zeros((n, nc)); C[np.arange(n), m.country] = 1.0
        self.L_r = la.lu_solve(self.price_lu, (c * self.betaK)[:, None] * C, check_finite=False)
        self.L_w = la.lu_solve(self.price_lu, (c * self.betaL)[:, None] * C, check_finite=False)

    def _differentials(self, l, lr, lw, dT, dXN):
        """Analytic differentials for K simultaneous directions (columns)."""
        m = self.model
        sv, ss, so, rho = m.elasticities
        n, nc, ns, nfd = self.n, self.nc, self.ns, self.nfd
        K = l.shape[1]
        lw_cell, lr_cell = lw[m.country], lr[m.country]
        dlogc = self.alpha.T @ l + self.betaL[:, None] * lw_cell + self.betaK[:, None] * lr_cell
        l3 = l.reshape(nc, ns, K)
        Z3, D3 = self.Z.reshape(nc, ns, n), self.D.reshape(nc, ns, n)
        dlogm = np.zeros((n, K)); sector_goods = np.zeros((nc, ns, K)); sector_duty = np.zeros((n, K))
        for s in range(ns):
            dlogh = self.u[:, s, :].T @ l3[:, s, :]                       # (n, K) for buyers
            dlogm += self.eta[s][:, None] * dlogh
            sector_goods[:, s, :] = Z3[:, s, :] @ dlogh
            sector_duty += self.D_sector[s][:, None] * dlogh
        dlogv = self.thetaL[:, None] * lw_cell + self.thetaK[:, None] * lr_cell
        common = sv * dlogc + (ss - sv) * dlogm
        da_y = self.Z @ common + (so - ss) * sector_goods.reshape(n, K) - so * self.Zrow[:, None] * l
        dbL = self.bL[:, None] * (sv * dlogc + (rho - sv) * dlogv - rho * lw_cell)
        dbK = self.bK[:, None] * (sv * dlogc + (rho - sv) * dlogv - rho * lr_cell)
        dduty = self.duty[:, None] * common + (so - ss) * sector_duty + (1.0 - so) * (self.D.T @ l)
        dQ = np.tensordot(self.Wq, l, axes=(0, 0))                         # (nfd, nc, K)
        dQt = np.tensordot(self.Wt, l, axes=(0, 0))
        dP = dQ / (1.0 - m.fd_tax)[:, :, None]
        dincome = self.wL[:, None] * lw + self.rK[:, None] * lr + dT       # (nc, K)
        dB = np.vstack([dXN, -dXN.sum(axis=0, keepdims=True)])
        dexp = m.theta[:, :, None] * dincome[None]
        dexp[m.inv] -= dB
        dq = (dexp - self.q[:, :, None] * dP) / self.P[:, :, None]
        dF = np.tensordot(m.afd, dq, axes=([1, 2], [0, 1]))                # (n, K)
        dfinal_tax = (m.fd_tax[:, :, None] * dexp).sum(axis=0)
        dduties_f = ((self.Q - self.Qt)[:, :, None] * dq + (dQ - dQt) * self.q[:, :, None]).sum(axis=0)
        return dict(dlogc=dlogc, da_y=da_y, dbL=dbL, dbK=dbK, dduty=dduty, dF=dF,
                    dfinal_tax=dfinal_tax, dduties_f=dduties_f)

    def _macro_rows(self, d, l, dly, dT):
        """Labour, capital, budget and numeraire rows for given directions."""
        m = self.model
        nc, ns = self.nc, self.ns
        K = l.shape[1]
        labor = -((self.bL * self.y)[:, None] * dly + self.y[:, None] * d["dbL"]).reshape(nc, ns, K).sum(axis=1)
        capital = -((self.bK * self.y)[:, None] * dly + self.y[:, None] * d["dbK"]).reshape(nc, ns, K).sum(axis=1)
        dprod = (self.tpy[:, None] * (l + dly)).reshape(nc, ns, K).sum(axis=1)
        dduty_i = (self.duty[:, None] * dly + d["dduty"]).reshape(nc, ns, K).sum(axis=1)
        budget = dT - (dprod + d["dfinal_tax"] + dduty_i + d["dduties_f"])
        numeraire = self.p[0] * l[0:1, :]
        return np.vstack([labor, capital, budget, numeraire])

    def response(self, l, lr, lw, dT, dXN, goods_rhs=None):
        """Output response and macro-row action for K directions.

        Given log-price directions ``l`` (M x K), macro directions ``lr``, ``lw``,
        ``dT`` (N x K), ``dXN`` ((N-1) x K) and an optional goods right-hand side,
        solve the goods block ``(I - a) diag(y) dly = (da) y + afd.dq + goods_rhs``
        and return ``(dly, macro)`` where ``macro`` stacks the labour, capital,
        budget and numeraire rows (3N+1 x K).
        """
        l = np.atleast_2d(np.asarray(l, dtype=float)).reshape(self.n, -1)
        K = l.shape[1]
        lr = np.asarray(lr, dtype=float).reshape(self.nc, K)
        lw = np.asarray(lw, dtype=float).reshape(self.nc, K)
        dT = np.asarray(dT, dtype=float).reshape(self.nc, K)
        dXN = np.asarray(dXN, dtype=float).reshape(self.nc - 1, K)
        d = self._differentials(l, lr, lw, dT, dXN)
        rhs = d["da_y"] + d["dF"]
        if goods_rhs is not None:
            rhs = rhs + np.asarray(goods_rhs, dtype=float).reshape(self.n, K)
        dly = la.lu_solve(self.goods_lu, rhs, check_finite=False) / self.y[:, None]
        return dly, self._macro_rows(d, l, dly, dT)

    def _assemble_macro(self, block_size):
        n, nc = self.n, self.nc
        size = 3 * nc + 1
        self.macro_size = size
        self.output_macro = np.empty((n, size))
        S = np.empty((size, size))
        for first in range(0, size, block_size):
            cols = range(first, min(first + block_size, size))
            K = len(cols)
            l = np.zeros((n, K)); lr = np.zeros((nc, K)); lw = np.zeros((nc, K))
            dT = np.zeros((nc, K)); dXN = np.zeros((nc - 1, K)); rhs = np.zeros((n, K))
            for k, col in enumerate(cols):
                if col < nc:
                    lr[col, k] = 1.0; l[:, k] = self.L_r[:, col]
                elif col < 2 * nc:
                    lw[col - nc, k] = 1.0; l[:, k] = self.L_w[:, col - nc]
                elif col < 3 * nc:
                    dT[col - 2 * nc, k] = 1.0
                else:
                    rhs[0, k] = 1.0
            dly, macro = self.response(l, lr, lw, dT, dXN, rhs)
            self.output_macro[:, first:first + K] = dly
            S[:, first:first + K] = macro
        row_scale = np.concatenate([self.model.L, self.model.K, self.model.gdp0, [1.0]])
        col_scale = np.concatenate([np.ones(2 * nc), self.model.gdp0, [self.model.y0[0]]])
        scaled = S / row_scale[:, None] * col_scale[None, :]
        if not np.isfinite(scaled).all():
            raise CESNewtonError("Nonfinite CES macro Newton block (a singular price or goods factorization at the "
                                 "linearization point)", macro_condition=float("inf"))
        try:
            self.macro_condition = float(np.linalg.cond(scaled))
        except np.linalg.LinAlgError as exc:
            raise CESNewtonError(f"CES macro Newton block condition number unavailable: {exc}",
                                 macro_condition=float("inf")) from exc
        if not np.isfinite(self.macro_condition):
            raise CESNewtonError("Singular CES macro Newton block", macro_condition=self.macro_condition)
        self.macro = S
        self.macro_lu = la.lu_factor(S, check_finite=False)

    # -- public actions ----------------------------------------------------
    def solve(self, r: np.ndarray) -> np.ndarray:
        """Return ``J^{-1} r`` (exact block elimination)."""
        n, nc = self.n, self.nc
        r = np.asarray(r, dtype=float).ravel()
        if r.shape != (self.shape[0],):
            raise ValueError(f"Right-hand side must have length {self.shape[0]}")
        r_g, r_p = r[:n], r[n:2 * n]
        r_L, r_K = r[2 * n:2 * n + nc], r[2 * n + nc:2 * n + 2 * nc]
        r_bal, r_bud = r[2 * n + 2 * nc:2 * n + 3 * nc - 1], r[2 * n + 3 * nc - 1:]
        dXN = r_bal
        l0 = -la.lu_solve(self.price_lu, r_p, check_finite=False)
        rhs = r_g.copy(); rhs[0] = 0.0
        zero = np.zeros((nc, 1))
        dly0, force = self.response(l0[:, None], zero, zero, zero, dXN[:, None], rhs[:, None])
        target = np.concatenate([r_L, r_K, r_bud, [r_g[0]]])
        macro = la.lu_solve(self.macro_lu, target - force[:, 0], check_finite=False)
        lr, lw, dT = macro[:nc], macro[nc:2 * nc], macro[2 * nc:3 * nc]
        l = l0 + self.L_r @ lr + self.L_w @ lw
        dly = dly0[:, 0] + self.output_macro @ macro
        return np.concatenate([l, dly, lr, lw, dT, dXN])

    def apply(self, v: np.ndarray) -> np.ndarray:
        """Return ``J v`` (forward action from the analytic differentials)."""
        n, nc = self.n, self.nc
        v = np.asarray(v, dtype=float).ravel()
        if v.shape != (self.shape[0],):
            raise ValueError(f"Direction must have length {self.shape[0]}")
        l, dly = v[:n, None], v[n:2 * n, None]
        lr, lw = v[2 * n:2 * n + nc, None], v[2 * n + nc:2 * n + 2 * nc, None]
        dT, dXN = v[2 * n + 2 * nc:2 * n + 3 * nc, None], v[2 * n + 3 * nc:, None]
        d = self._differentials(l, lr, lw, dT, dXN)
        ydly = self.y[:, None] * dly
        goods = ydly - self.a @ ydly - d["da_y"] - d["dF"]
        goods[0, 0] = self.p[0] * l[0, 0]
        prices = self.c[:, None] * d["dlogc"] - self.p[:, None] * l
        macro = self._macro_rows(d, l, dly, dT)
        labor, capital, budget = macro[:nc], macro[nc:2 * nc], macro[2 * nc:3 * nc]
        return np.concatenate([goods[:, 0], prices[:, 0], labor[:, 0], capital[:, 0], dXN[:, 0], budget[:, 0]])

    def as_preconditioner(self) -> LinearOperator:
        """``LinearOperator`` whose matvec is :meth:`solve` (for Krylov ``inner_M``)."""
        return LinearOperator(self.shape, matvec=self.solve, dtype=float)

    def as_operator(self) -> LinearOperator:
        """``LinearOperator`` whose matvec is :meth:`apply` (the forward Jacobian)."""
        return LinearOperator(self.shape, matvec=self.apply, dtype=float)


# ---------------------------------------------------------------------------
# Certificate
# ---------------------------------------------------------------------------

def _certificate(model: _Model, b: dict[str, Any]) -> dict[str, float]:
    """Independent raw-flow reconstruction from the solved technology and flows."""
    m = model
    nc, ns, n = m.nc, m.ns, m.n
    p, y, r, w, T = (np.real(b[k]) for k in ("pv", "yv", "rv", "wv", "Tv"))
    tech = b["tech"]
    a, bL, bK, c = np.real(tech["a"]), np.real(tech["bL"]), np.real(tech["bK"]), np.real(tech["c"])
    q = np.real(b["quantity"]); F = m.afd * q[None]
    Z = a * y[None, :]
    country = m.country
    purchaser = np.sum(a * m.tau * p[:, None], axis=0) + bL * w[country] + bK * r[country]
    zero = (1.0 - m.t) * p - purchaser
    receipts = p * (Z.sum(axis=1) + F.sum(axis=(1, 2)))
    exports = np.zeros(nc); imports = np.zeros(nc); duty = np.zeros(nc)
    labor = np.zeros(nc); capital = np.zeros(nc); spending = np.zeros(nc); taxes = np.zeros(nc)
    for k in range(nc):
        own = country == k
        others = np.arange(nc) != k
        exports[k] = np.sum(p[own, None] * Z[np.ix_(own, ~own)]) + np.sum(p[own, None, None] * F[own][:, :, others])
        imports[k] = np.sum(p[~own, None] * Z[np.ix_(~own, own)]) + np.sum(p[~own, None] * F[~own, :, k])
        duty[k] = np.sum((m.tau[np.ix_(~own, own)] - 1.0) * p[~own, None] * Z[np.ix_(~own, own)])
        duty[k] += np.sum((m.tf[~own, :, k] - 1.0) * p[~own, None] * F[~own, :, k])
        labor[k] = np.sum(bL[own] * y[own]); capital[k] = np.sum(bK[own] * y[own])
        taxes[k] = np.sum(m.t[own] * p[own] * y[own]) + np.sum(m.fd_tax[:, k] * np.real(b["exp_mat"])[:, k])
        spending[k] = np.sum(np.real(b["exp_mat"])[:, k])
    income = w * m.L + r * m.K + T
    gdp = income
    expenditure_gdp = spending + exports - imports
    gdp0 = m.gdp0
    goods0 = y[0] - Z[0].sum() - F[0].sum()
    scaled = b["residuals"] / m.scale
    return {"zero_profit": float(np.max(np.abs(zero) / np.maximum(1.0, p))),
            "cost_adding_up": float(np.max(np.abs(purchaser / (1.0 - m.t) - c) / np.maximum(1.0, c))),
            "goods_value": float(np.max(np.abs(receipts - p * y)[1:] / (p * m.y0)[1:])),
            "dropped_equation": float(abs(goods0) / m.y0[0]),
            "current_account": float(np.max(np.abs(exports - imports - m.B0) / gdp0)),
            "tariff_revenue": float(np.max(np.abs(duty - np.real(b["tariffs"])) / gdp0)),
            "factor_clearing": float(max(np.max(np.abs(labor - m.L) / m.L), np.max(np.abs(capital - m.K) / m.K))),
            "household_budget": float(np.max(np.abs(spending + m.B0 - income) / gdp0)),
            "government_budget": float(np.max(np.abs(T - taxes - duty) / gdp0)),
            "gdp_identity": float(np.max(np.abs(gdp - expenditure_gdp) / gdp0)),
            "root_residual": float(np.max(np.abs(scaled)))}


def certify_ces_equilibrium(calib, x, *, tau=None, tau_fd=None, tauf=None, tauf_fd=None,
                            technology: NestedCESTechnology = NestedCESTechnology()) -> dict[str, float]:
    """Independent flow certificate of a state ``x`` (all entries dimensionless).

    Keys: ``zero_profit`` (net price minus reconstructed unit outlay, relative to
    ``max(1, p)``), ``cost_adding_up`` (Euler identity of the nested cost),
    ``goods_value`` (producer receipts minus ``p y`` over ``p y0``, omitting cell 0),
    ``dropped_equation`` (the omitted goods equation over ``y0``), ``current_account``
    (exports minus imports minus baseline saving over baseline income),
    ``tariff_revenue`` (bilaterally reconstructed duties versus recorded receipts),
    ``factor_clearing``, ``household_budget``, ``government_budget``,
    ``gdp_identity`` (income equals absorption plus net exports) and
    ``root_residual`` (max scaled residual). Intermediate and factor flows are
    rebuilt from the technology at ``(p, r, w)`` with explicit per-country loops;
    final quantities, final expenditure and recorded duty receipts come from the
    demand assembly of the residual and are checked through the goods-value,
    dropped-equation, budget, current-account, tariff-revenue and GDP keys. The
    solver accepts a state only when all keys are at most ``certificate_tol``
    (default ``1e-8``).
    """
    model = _build_model(calib, tau, tau_fd, tauf, tauf_fd, technology)
    return _certificate(model, model.blocks(np.asarray(x, dtype=float).ravel()))


# ---------------------------------------------------------------------------
# Result container
# ---------------------------------------------------------------------------

@dataclass(frozen=True)
class CESBlockNewtonResult:
    """Certified nested-CES equilibrium with solver diagnostics.

    ``equilibrium`` is a :class:`TradeEquilibriumResult` with
    ``metadata['accounting'] == 'consistent'`` and
    ``metadata['effective_method'] == 'ces_block_newton'``. ``residual_max`` is
    the maximum scaled residual (what ``tol`` bounds); the absolute level
    maximum is ``equilibrium.max_residual``. ``certificate`` is the independent
    flow audit, ``macro_condition`` the last Schur-block condition number,
    ``history`` the per-iteration records and ``stages``/``failed_stages`` the
    homotopy records (empty for a direct solve). On a homotopy result
    ``iterations``, ``history``, ``residual_max``, ``certificate`` and
    ``macro_condition`` describe the final certified stage (the target
    schedule) while ``calls`` and ``seconds`` are totals over the accepted
    stages including the start; ``stages_frame()`` has the per-stage detail,
    and the start record carries ``macro_condition`` NaN with zero iterations
    when ``x0`` already solved the start schedule (no Newton step was needed).
    """

    equilibrium: TradeEquilibriumResult
    technology: NestedCESTechnology
    converged: bool
    iterations: int
    calls: int
    seconds: float
    residual_max: float
    certificate: dict[str, float]
    macro_condition: float
    history: tuple[dict[str, Any], ...] = ()
    stages: tuple[dict[str, Any], ...] = ()
    failed_stages: tuple[dict[str, Any], ...] = ()
    country_codes: tuple[str, ...] = field(default_factory=tuple)
    sector_codes: tuple[str, ...] = field(default_factory=tuple)
    metadata: dict[str, Any] = field(default_factory=dict)

    def to_dataframe(self):
        """Newton history: iteration, residual before/after, step, macro condition."""
        import pandas as pd
        cols = ["iteration", "residual_before", "step", "macro_condition", "residual_after"]
        rows = [{k: h.get(k) for k in cols} for h in self.history]
        return pd.DataFrame(rows, columns=cols)

    def stages_frame(self):
        """Homotopy stages: fraction, iterations, calls, seconds, certificate maximum."""
        import pandas as pd
        cols = ["fraction", "iterations", "calls", "seconds", "residual_max", "certificate_max", "macro_condition"]
        rows = [{k: s.get(k) for k in cols} for s in self.stages]
        return pd.DataFrame(rows, columns=cols)

    def certificate_frame(self):
        """The independent flow certificate as a one-column frame indexed by check."""
        import pandas as pd
        return pd.DataFrame({"check": list(self.certificate), "value": list(self.certificate.values())}).set_index("check")

    def summary(self) -> str:
        """One-line report; on a homotopy result the iterations are the final stage's, calls and seconds totals."""
        sv, ss, so, rho = self.technology.elasticities
        stage = " (final stage)" if self.stages else ""
        totals = " over all accepted stages" if self.stages else ""
        return (f"CES block Newton: technology (sv={sv:g}, ss={ss:g}, so={so:g}, rho_va={rho:g}), "
                f"{'converged' if self.converged else 'FAILED'} in {self.iterations} iterations{stage} "
                f"({self.calls} residual calls{totals}, {self.seconds:.3f} s); scaled residual {self.residual_max:.3g}, "
                f"certificate max {max(self.certificate.values()):.3g}, macro condition {self.macro_condition:.3g}"
                + (f"; {len(self.stages)} homotopy stages, {len(self.failed_stages)} failed attempts" if self.stages else ""))

    def to_markdown(self, **kwargs: Any) -> str:
        """Newton history as a Markdown table (``puremacro.reports.df_to_markdown``)."""
        return df_to_markdown(self.to_dataframe(), **kwargs)

    def to_latex(self, **kwargs: Any) -> str:
        """Newton history as a LaTeX table (``puremacro.reports.df_to_latex``)."""
        return df_to_latex(self.to_dataframe(), **kwargs)

    def to_typst(self, **kwargs: Any) -> str:
        """Newton history as a Typst table (``puremacro.reports.df_to_typst``)."""
        return df_to_typst(self.to_dataframe(), **kwargs)


def _equilibrium_result(model: _Model, b: dict[str, Any], *, converged, iterations, tol_abs, max_iter,
                        seconds, certificate, macro_condition, extra_meta) -> TradeEquilibriumResult:
    """Package blocks like ``postprocess_trade_equilibrium`` does for consistent accounting."""
    calib = model.calib
    flows = _postprocess_consistent(b, calib, None)
    sv, ss, so, rho = model.elasticities
    meta = {"method": "ces_block_newton", "effective_method": "ces_block_newton", "accounting": "consistent",
            "backend": "numpy", "tol": tol_abs, "max_iter": max_iter, "replicate_matlab_precedence": False,
            "solve_duration_seconds": seconds, "sigma": so, "fiscal_closure": "lump_sum",
            "recycling_params": None, "capacity_margins": None, "capacity_target_country": None,
            "penalty_scale": None, "penalty_exponent": None,
            "technology": model.technology.to_dict(), "flat_technology": model.technology.is_flat,
            "hicksian_welfare_supported": model.technology.is_flat,
            "certificate": dict(certificate), "macro_condition": macro_condition,
            "tariff_revenue_mode": "schedule", "fiscal_tariffs": flows["fiscal_tariffs"], "matlab_compat": False,
            "bilateral_trade": flows["bilateral_trade"], "tariffs_interm": flows["tariffs_interm"],
            "tariffs_fd": flows["tariffs_fd"], "data_model_vf": flows["data_model_vf"],
            "data_tariff_vf": flows["data_tariff_vf"]}
    meta.update(flows.get("accounting_metadata", {}))
    meta.update(extra_meta)
    physical = meta["physical_residuals"]
    converged = bool(converged and meta["demand_feasible"] and np.isfinite(physical).all()
                     and np.max(np.abs(physical)) <= tol_abs
                     and np.isfinite(flows["max_residual"]) and flows["max_residual"] <= tol_abs)
    return TradeEquilibriumResult(
        x_sol=np.asarray(b["x"], dtype=float).ravel(), p_sol=flows["p_sol"], y_sol=flows["y_sol"],
        r_sol=flows["r_sol"], w_sol=flows["w_sol"], T_sol=flows["T_sol"], XN_sol=flows["XN_sol"],
        intermediate_flows=flows["intermediate_flows"], final_demand_flows=flows["final_demand_flows"],
        p_fd=flows["p_fd"], c_fd=flows["c_fd"], gdp=flows["gdp"], gdp_fc=flows["gdp_fc"],
        imports=flows["imports"], exports=flows["exports"], tariffs=flows["tariffs"], cpi=flows["cpi"],
        terms_of_trade=flows["terms_of_trade"], c_sol=flows["c_fd"], pfd_sol=flows["p_fd"],
        Pfd_final=flows["P_fd"], qxX0_sol=flows["qxX0"], qxFD0_sol=flows["qxFD0"],
        data_model_vf=flows["data_model_vf"], data_tariff_vf=flows["data_tariff_vf"],
        converged=converged, iterations=iterations, diff=flows["diff"], residual_norm=flows["residual_norm"],
        max_residual=flows["max_residual"], residuals=flows["residuals"],
        country_codes=calib.country_codes, sector_codes=calib.sector_codes, metadata=meta)


# ---------------------------------------------------------------------------
# Damped exact Newton
# ---------------------------------------------------------------------------

def _newton(model: _Model, x0, *, tol, max_iter, certificate_tol, max_backtracks, block_size, max_cells,
            progress) -> tuple[np.ndarray, dict[str, Any], dict[str, Any]]:
    """Run the damped Newton; return ``(x, blocks, info)`` or raise ``CESNewtonError``."""
    if not (np.isfinite(tol) and tol > 0 and np.isfinite(certificate_tol) and certificate_tol > 0):
        raise ValueError("tol and certificate_tol must be finite and positive")
    if not isinstance(max_iter, (int, np.integer)) or max_iter < 0:
        raise ValueError("max_iter must be a nonnegative integer")
    if not isinstance(max_backtracks, (int, np.integer)) or max_backtracks < 1:
        raise ValueError("max_backtracks must be a positive integer")
    x = np.array(x0, dtype=float).ravel()
    if x.shape != (model.dim,):
        raise ValueError(f"x0 must have length {model.dim}")
    n, nc = model.n, model.nc
    n_log = 2 * n + 2 * nc
    tic = time.perf_counter()
    history: list[dict[str, Any]] = []
    calls = 0
    condition = float("nan")
    scale = model.scale
    tol_abs = model.absolute_tol(tol)
    for iteration in range(max_iter + 1):
        b = model.blocks(x); calls += 1
        reason = model.inadmissibility(b)
        if reason is not None:
            raise CESNewtonError(f"CES Newton iterate is inadmissible: {reason}", history=history,
                                 macro_condition=condition)
        scaled = b["residuals"] / scale
        res = float(np.max(np.abs(scaled)))
        # One acceptance test, identical to the postprocessing audit: the imposed rows in
        # scaled units AND every level row, including the rows implied by Walras' law.
        audit, feasible = model.level_audit(b)
        if res <= tol and audit <= tol_abs:
            if feasible:
                break
            # Unreachable under the admissibility filter (positive active quantities); kept so
            # that no state with negative final expenditure can ever pass.
            raise CESNewtonError("CES Newton root has negative final expenditure (demand infeasible)",
                                 history=history, macro_condition=condition)
        if iteration == max_iter:
            pressed = ""
            if history and history[-1]["rejected_inadmissible"] > 0:
                pressed = (f"; the last step was cut by the admissibility filter "
                           f"({history[-1]['rejected_inadmissible']} inadmissible trials): the iterate is pressed "
                           f"against the admissible boundary")
            raise CESNewtonError(f"CES Newton iteration limit ({max_iter}); scaled residual={res:.3e} (tol {tol:g}), "
                                 f"level audit {audit:.3e} (bound tol*max(scale)={tol_abs:.3e}), "
                                 f"macro condition {condition:.3g}{pressed}", history=history, macro_condition=condition)
        polishing = res <= tol       # imposed rows pass; only the implied rows exceed the level bound
        jac = CESBlockJacobian(model.calib, x, _model=model, _blocks=b, block_size=block_size, max_cells=max_cells)
        dx = -jac.solve(b["residuals"])
        condition = jac.macro_condition
        del jac
        if not np.isfinite(dx).all():
            raise CESNewtonError("Nonfinite CES Newton direction", history=history, macro_condition=condition)
        step = min(1.0, 1.0 / max(1.0, float(np.max(np.abs(dx[:n_log])))))
        merit = float(scaled @ scaled)
        accepted = None
        inadmissible = armijo = 0
        binding = None
        for _ in range(max_backtracks):
            with np.errstate(over="ignore", invalid="ignore", divide="ignore"):
                candidate = model.blocks(x + step * dx); calls += 1
            why = model.inadmissibility(candidate)
            if why is None:
                cs = candidate["residuals"] / scale
                if float(cs @ cs) <= (1.0 - 1e-4 * step) * merit:
                    accepted = candidate
                    break
                armijo += 1
            else:
                inadmissible += 1
                binding = (step, why)
            step *= 0.5
        if accepted is None:
            counts = f"{armijo} Armijo rejections, {inadmissible} inadmissible trials"
            if polishing:
                message = (f"CES Newton line search failed at the level audit: the scaled residual {res:.3e} meets "
                           f"tol but the level rows reach {audit:.3e} > tol*max(scale)={tol_abs:.3e} (the omitted "
                           f"goods equation and realized foreign balances are implied by Walras' law), and no "
                           f"step reduces the residual further ({counts}). Either the level bound is below the "
                           f"rounding floor of the current flows (loosen tol) or the implied rows are structurally "
                           f"nonzero (Walras' law fails). Macro condition {condition:.3g}")
            elif armijo == 0:
                message = (f"CES Newton line search failed: all {inadmissible} trial steps left the admissible "
                           f"domain (at the smallest trial step {binding[0]:.2e}: {binding[1]}); scaled "
                           f"residual={res:.3e}, macro condition "
                           f"{condition:.3g}. The direction drives the state out of the admissible set; when an "
                           f"active final quantity or expenditure is the binding constraint the nearby root may "
                           f"be inadmissible under this closure, which a tariff homotopy cannot change")
            else:
                message = (f"CES Newton line search failed ({counts}); scaled residual={res:.3e} (tol {tol:g}), "
                           f"level audit {audit:.3e} (bound {tol_abs:.3e}), macro condition {condition:.3g} (a "
                           f"growing condition number signals a nearly singular Jacobian: try "
                           f"continue_tariff_homotopy or a different start; a residual already near the rounding "
                           f"floor of the current flows needs a looser tol)")
            raise CESNewtonError(message, history=history, macro_condition=condition)
        x = x + step * dx
        history.append(dict(iteration=iteration, residual_before=res, step=step, macro_condition=condition,
                            residual_after=float(np.max(np.abs(accepted["residuals"] / scale))),
                            level_audit_before=audit, rejected_armijo=armijo, rejected_inadmissible=inadmissible))
        if progress is not None:
            progress(dict(history[-1]))
    cert = _certificate(model, b)
    if not np.isfinite(list(cert.values())).all() or max(cert.values()) > certificate_tol:
        raise CESNewtonError(f"CES flow certificate failed (max {max(cert.values()):.3e} > {certificate_tol:g}): {cert}",
                             history=history, macro_condition=condition, certificate=cert)
    b["x"] = x
    info = dict(iterations=iteration, calls=calls, seconds=time.perf_counter() - tic, residual_max=res,
                certificate=cert, macro_condition=condition, history=tuple(history))
    return x, b, info


def _package(model: _Model, b, info, *, tol, max_iter, stages=(), failed_stages=(), extra_meta=None):
    """Wrap an accepted state; re-run the level audit of postprocessing (the Newton loop already applied it)."""
    tol_abs = model.absolute_tol(tol)
    seconds = float(info["seconds"])
    meta = {"scaled_tol": tol, "residual_scale_max": float(np.max(model.scale)),
            "residual_max_scaled": info["residual_max"],
            "newton_history": info["history"], "homotopy_stages": tuple(stages),
            "homotopy_failed_stages": tuple(failed_stages)}
    if extra_meta:
        meta.update(extra_meta)
    equilibrium = _equilibrium_result(model, b, converged=True, iterations=info["iterations"], tol_abs=tol_abs,
                                      max_iter=max_iter, seconds=seconds, certificate=info["certificate"],
                                      macro_condition=info["macro_condition"], extra_meta=meta)
    if not equilibrium.converged:          # defensive: the Newton loop applies the same audit before accepting
        raise CESNewtonError(f"Postprocessed state fails the absolute physical/accounting audit "
                             f"(max level row {model.level_audit(b)[0]:.3e}, bound {tol_abs:.3e})",
                             history=info["history"], macro_condition=info["macro_condition"],
                             certificate=info["certificate"])
    return CESBlockNewtonResult(
        equilibrium=equilibrium, technology=model.technology, converged=True, iterations=int(info["iterations"]),
        calls=int(info["calls"]), seconds=seconds, residual_max=float(info["residual_max"]),
        certificate=dict(info["certificate"]), macro_condition=float(info["macro_condition"]),
        history=tuple(info["history"]), stages=tuple(stages), failed_stages=tuple(failed_stages),
        country_codes=model.calib.country_codes, sector_codes=model.calib.sector_codes,
        metadata={"solver": "exact CES block Newton", "dimension": model.dim, "macro_size": 3 * model.nc + 1,
                  "tol": tol, "tol_absolute": tol_abs, "accounting": "consistent",
                  "technology": model.technology.to_dict(),
                  "message": "Residual and independent flow certificate accepted"})


def solve_ces_block_newton(calib, tau=None, tau_fd=None, *, tauf=None, tauf_fd=None,
                           technology: NestedCESTechnology = NestedCESTechnology(), x0=None,
                           tol: float = 2e-11, max_iter: int = 20, certificate_tol: float = 1e-8,
                           max_backtracks: int = 25, max_cells: int = 4000, block_size: int = 16,
                           progress: Callable[[dict[str, Any]], None] | None = None) -> CESBlockNewtonResult:
    """Damped exact-Jacobian Newton for the nested-CES consistent-accounting equilibrium.

    Parameters follow ``solve_trade_equilibrium``: ``tau``/``tau_fd`` are gross
    multipliers (``(M, S, N)`` and ``(M, K, N)``, rate vectors of length ``N`` or
    scalars are expanded by ``_resolve_tariffs``), ``tauf``/``tauf_fd`` national
    rates, ``x0`` a start (default the calibrated benchmark). ``tol`` bounds the
    maximum residual after dividing each row by its benchmark scale (output,
    endowment or income); ``certificate_tol`` bounds every key of
    :func:`certify_ces_equilibrium`. The step is ``min(1, 1/max|dx_log|)`` with
    Armijo backtracking on the squared scaled residual and an admissibility
    filter (positive prices, outputs, factor prices, incomes and active final
    quantities, exactly zero quantities on absent categories). The loop stops only
    when the scaled test passes AND the level audit of postprocessing holds: every
    level residual and physical row (including the omitted goods equation and the
    realized foreign balances, which no Newton row imposes) at most
    ``tol * max(scale)`` (recorded as ``equilibrium.metadata['tol']``) and no
    negative final expenditure; when only the implied rows fail it takes further
    (polishing) Newton steps. ``max_iter = 0`` only certifies ``x0``.
    ``max_cells`` (default 4000) refuses tables with more cells before the dense
    operator is allocated (a solve peaks near ``22 M^2`` doubles);
    ``block_size`` (default 16) is the number of unit macro directions pushed
    through the goods block per LU solve when the Schur complement is
    assembled (results are independent of it). ``progress``, if given, receives
    each accepted iteration's history record (``iteration``,
    ``residual_before``, ``step``, ``macro_condition``, ``residual_after``,
    ``level_audit_before``, ``rejected_armijo``, ``rejected_inadmissible``).
    Any failure raises :class:`CESNewtonError` with the history; line-search
    failures say whether the trial steps were rejected by the Armijo test or by
    the admissibility filter, naming the binding quantity. No unconverged or
    uncertified state is ever returned.
    """
    model = _build_model(calib, tau, tau_fd, tauf, tauf_fd, technology)
    x_init = build_initial_guess(calib) if x0 is None else np.asarray(x0, dtype=float).ravel()
    x, b, info = _newton(model, x_init, tol=tol, max_iter=max_iter, certificate_tol=certificate_tol,
                         max_backtracks=max_backtracks, block_size=block_size, max_cells=max_cells,
                         progress=progress)
    return _package(model, b, info, tol=tol, max_iter=max_iter)


# ---------------------------------------------------------------------------
# Adaptive tariff homotopy
# ---------------------------------------------------------------------------

def continue_tariff_homotopy(calib, tau, tau_fd, *, tauf=None, tauf_fd=None,
                             technology: NestedCESTechnology = NestedCESTechnology(), x0=None,
                             tau_start=None, tau_fd_start=None, first_step: float = 0.25,
                             max_step: float = 0.4, min_step: float = 1.0 / 128, growth: float = 1.5,
                             fast_iterations: int = 5, stage_callback=None, progress=None,
                             **newton_options: Any) -> CESBlockNewtonResult:
    """Adaptive natural-parameter homotopy in the tariff schedules, certified at every stage.

    The schedules move linearly, ``tau(lambda) = tau_start + lambda (tau_target -
    tau_start)`` on both intermediate and final multipliers (``tau_start`` defaults
    to free trade, the calibrated benchmark). Stage 0 solves at ``tau_start`` from
    ``x0``. The step starts at ``first_step``, grows by ``growth`` (capped at
    ``max_step``) after a stage needing at most ``fast_iterations`` Newton
    iterations, halves after a failed stage and aborts below ``min_step``. Every
    accepted stage passes the residual, level-audit and certificate tests of
    :func:`solve_ces_block_newton` and is packaged as a
    :class:`CESBlockNewtonResult` when it is accepted (a packaging failure counts
    as a failed attempt); ``stage_callback(result, record)`` is called after
    each. Failed attempts are recorded in ``failed_stages``. Every abort raises
    :class:`CESNewtonError` carrying ``stages``, ``failed_stages`` and
    ``last_result`` (the last accepted stage, ``None`` when the start schedule
    itself is unresolved).

    ``progress`` receives two record shapes: ``{"fraction", "step"}`` before each
    trial stage, and the Newton history records of every stage (``iteration``,
    ``residual_before``, ``step``, ``macro_condition``, ``residual_after``,
    ``level_audit_before``, ``rejected_armijo``, ``rejected_inadmissible``).
    ``newton_options`` are forwarded to the Newton (``tol`` default ``2e-11``,
    ``max_iter`` default 12, ``certificate_tol`` default ``1e-8``,
    ``max_backtracks`` default 25, ``block_size`` default 16, ``max_cells``
    default 4000). The discrete stages do not certify a continuous branch between
    them, and an abort is not evidence that no equilibrium exists at the target.
    """
    if not (0 < first_step <= 1 and 0 < max_step <= 1 and 0 < min_step <= first_step and growth >= 1):
        raise ValueError("Step parameters must satisfy 0 < min_step <= first_step <= 1, 0 < max_step <= 1, growth >= 1")
    options = dict(tol=2e-11, max_iter=12, certificate_tol=1e-8, max_backtracks=25, block_size=16, max_cells=4000)
    unknown = set(newton_options) - set(options)
    if unknown:
        raise TypeError(f"Unknown Newton options: {sorted(unknown)}")
    options.update(newton_options)
    ta1, tf1, _, _ = _resolve_tariffs(calib, tau, tau_fd, tauf, tauf_fd)
    ta0, tf0, _, _ = _resolve_tariffs(calib, tau_start, tau_fd_start)
    if ta0.shape != ta1.shape or tf0.shape != tf1.shape:
        raise ValueError("Start and target schedules must have identical shapes")
    _check_technology(technology)
    extra = {"path": "adaptive tariff homotopy from the start schedule",
             "calls_scope": "accepted stages including the start; rejected attempts excluded",
             "continuity_claim": "none: discrete certified stages only"}

    def model_at(fraction):
        return _Model(calib, ta0 + fraction * (ta1 - ta0), tf0 + fraction * (tf1 - tf0),
                      np.zeros(calib.nc), np.zeros(calib.nc), technology)

    x = build_initial_guess(calib) if x0 is None else np.asarray(x0, dtype=float).ravel()
    stages: list[dict[str, Any]] = []
    failures: list[dict[str, Any]] = []
    last: CESBlockNewtonResult | None = None

    def attempt(fraction, start):
        """Solve, certify and package one stage; any failure raises ``CESNewtonError``."""
        model = model_at(fraction)
        xs, b, info = _newton(model, start, tol=options["tol"], max_iter=options["max_iter"],
                              certificate_tol=options["certificate_tol"], max_backtracks=options["max_backtracks"],
                              block_size=options["block_size"], max_cells=options["max_cells"], progress=progress)
        record = dict(fraction=float(fraction), iterations=int(info["iterations"]), calls=int(info["calls"]),
                      seconds=float(info["seconds"]), residual_max=float(info["residual_max"]),
                      certificate_max=float(max(info["certificate"].values())),
                      macro_condition=float(info["macro_condition"]))
        result = _package(model, b, info, tol=options["tol"], max_iter=options["max_iter"],
                          stages=stages + [record], failed_stages=failures, extra_meta=extra)
        return model, xs, b, info, record, result

    def abort(message, exc):
        last_result = None if last is None else replace(last, failed_stages=tuple(failures))
        return CESNewtonError(message, history=exc.history, macro_condition=exc.macro_condition,
                              certificate=exc.certificate, stages=stages, failed_stages=failures,
                              last_result=last_result)

    try:
        model, x, b, info, record, last = attempt(0.0, x)
    except CESNewtonError as exc:
        raise abort(f"Tariff homotopy start schedule unresolved: {exc}", exc) from exc
    stages.append(record)
    if stage_callback is not None:
        stage_callback(last, dict(record))
    fraction, step = 0.0, float(first_step)
    while fraction < 1.0:
        trial = min(1.0, fraction + step)
        if progress is not None:
            progress(dict(fraction=trial, step=step))
        try:
            model, x_new, b, info, record, result = attempt(trial, x)
        except CESNewtonError as exc:
            failures.append(dict(fraction=float(trial), start_fraction=float(fraction), step=float(step),
                                 error=str(exc), history=exc.history, macro_condition=exc.macro_condition))
            step *= 0.5
            if step < min_step:
                raise abort(f"Tariff path unresolved beyond fraction {fraction:.8f}: {exc}", exc) from exc
            continue
        x, fraction = x_new, trial
        stages.append(record)
        last = result
        if stage_callback is not None:
            stage_callback(result, dict(record))
        if record["iterations"] <= fast_iterations:
            step = min(max_step, step * growth)
    total = dict(info, calls=sum(s["calls"] for s in stages), seconds=sum(s["seconds"] for s in stages))
    try:
        return _package(model, b, total, tol=options["tol"], max_iter=options["max_iter"], stages=stages,
                        failed_stages=failures, extra_meta=extra)
    except CESNewtonError as exc:       # defensive: the same state was packaged when it was accepted
        raise abort(f"Tariff homotopy endpoint failed the post-audit: {exc}", exc) from exc
