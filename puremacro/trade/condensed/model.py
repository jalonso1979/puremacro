"""The condensed one-factor Leontief tariff model in ``2N`` unknowns.

Unknowns are ``z = (omega, Y_hat)`` with ``w_k = exp(omega_k)`` (the factor
price of country k) and ``Y_k = Y_hat_k * Y0_k`` (nominal national income).
Prices and outputs are eliminated exactly with two LU factorizations:

* zero profit (E1): ``(1 - t_j) p_j = b_j w_c(j) + sum_i a_ij tau_ij p_i``, so
  ``p = (I - B_tau^T)^{-1} [(b / (1 - t)) o w_c(.)]`` with
  ``B_tau,ij = a_ij tau_ij / (1 - t_j)`` (factorized once per wedge set);
* goods clearing (E2): ``y = (I - a)^{-1} d`` with the delivery vector
  ``d_i = sum_k (sum_f omega^f_ik q^f_k + qV_ik)`` (factorized once per table).

Given ``z`` one evaluation computes, in this order: ``w``; the trade balance
``XN`` from the closure; ``p``; the basket purchaser prices
``P^f_k = sum_i omega^f_ik tau^f_ik p_i`` and basic prices ``Ptilde^f_k``;
inventory spending ``E^V_k = sum_i tau^V_ik p_i qV_ik + TV_k P^G_k``; absorption
net of inventories ``Atil_k = Y_k - XN_k - E^V_k``; Cobb-Douglas spending
``E^f_k = theta^f_k Atil_k`` (or the fixed-real-investment rule); basket
quantities ``q^f_k = (1 - t^f_k) E^f_k / P^f_k``; outputs ``y``; product taxes
``PT_k = sum_{j in k} t_j p_j y_j + sum_f t^f_k E^f_k + TV_k P^G_k``; tariff
revenue ``TR_k = sum_{j in k} sum_i (tau_ij - 1) a_ij p_i y_j + sum_f (P^f_k -
Ptilde^f_k) q^f_k + sum_i (tau^V_ik - 1) p_i qV_ik``.

Residuals are scaled by base quantities only::

    r^F_k = (sum_{j in k} b_j y_j - Fbar_k) / Fbar_k             (factor markets, all k)
    r^Y_k = (Y_k - w_k Fbar_k - PT_k - TR_k) / Y0_k              (income, k != reference)
    r^N   = numeraire residual (replaces r^Y of the reference country)

The dropped ``r^Y_reference`` is the Walras check reported with every state.

Closures (Dekle-Eaton-Kortum): ``"factor"`` fixes ``XN_k = s_k sum_l w_l Fbar_l``
(shares of world factor income, the headline); ``"gdp"`` uses shares of world
GDP ``sum_l Y_l``; ``"own"`` uses ``s_k Y_k`` with the reference country
residual. Numeraires: ``"factor"`` (``sum_l w_l Fbar_l = sum_l Fbar_l``) or a
country code (its wage equals one). The IO engine's ``"w_USA"``/``"w_ROW"``
spellings are accepted as aliases.

This is NOT the two-factor model of ``solve_trade_equilibrium`` nor its
``accounting="consistent"`` mode: value added is one composite factor, final
demand has three Cobb-Douglas baskets plus exogenous inventories, trade
balances are shares of world factor income, and the numeraire is world factor
income. See docs/trade_condensed.md for the comparison table.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

import numpy as np
import pandas as pd
import scipy.linalg as la

from puremacro.reports import df_to_latex, df_to_markdown, df_to_typst

from .calibration import CondensedCalibration
from .errors import CalibrationError, UnsupportedExtension
from .table import ENDOGENOUS_FD
from .tariffs import TariffWedges

CLOSURES: tuple[str, ...] = ("factor", "gdp", "own")
FD_RULES: tuple[str, ...] = ("absorption", "fixed_investment")
INVENTORY_RULES: tuple[str, ...] = ("tariffed", "untariffed")


def _ro(a: np.ndarray) -> np.ndarray:
    a.flags.writeable = False
    return a


@dataclass(frozen=True)
class CondensedState:
    """Complete economic state of the condensed model at an unknown vector.

    Attributes
    ----------
    z : np.ndarray, shape (2N,)
        Unknowns ``(omega, Y_hat)``.
    w : np.ndarray, shape (N,)
        Factor prices ``exp(omega)``.
    Y, Y_hat, T : np.ndarray, shape (N,)
        Nominal income, its ratio to the benchmark, and net transfers ``Y - w Fbar``.
    p, y : np.ndarray, shape (M,)
        Basic seller prices and gross outputs.
    P, P_tilde : np.ndarray, shape (3, N)
        Purchaser and basic prices of the C, G, X baskets.
    E_V, A_tilde : np.ndarray, shape (N,)
        Inventory spending and absorption net of inventories.
    E, q : np.ndarray, shape (3, N)
        Purchaser expenditure and composite quantities of the C, G, X baskets.
    XN, PT, TR : np.ndarray, shape (N,)
        Fob trade balance from the closure, net product taxes and tariff revenue.
    r_F, r_Y : np.ndarray, shape (N,)
        Scaled factor-market and income residuals (all countries).
    r_N : float
        Scaled numeraire residual.
    residual : np.ndarray, shape (2N,)
        The vector solved by Newton (``r_Y`` of the reference country replaced by ``r_N``).
    walras_residual : float
        The dropped ``r_Y`` of the reference country.
    """

    z: np.ndarray
    w: np.ndarray
    Y: np.ndarray
    Y_hat: np.ndarray
    T: np.ndarray
    p: np.ndarray
    y: np.ndarray
    P: np.ndarray
    P_tilde: np.ndarray
    E_V: np.ndarray
    A_tilde: np.ndarray
    E: np.ndarray
    q: np.ndarray
    XN: np.ndarray
    PT: np.ndarray
    TR: np.ndarray
    r_F: np.ndarray
    r_Y: np.ndarray
    r_N: float
    residual: np.ndarray
    walras_residual: float
    metadata: dict[str, Any] = field(default_factory=dict)

    @property
    def max_scaled_residual(self) -> float:
        """``max |residual|`` over the ``2N`` equations."""
        return float(np.max(np.abs(self.residual)))

    def to_dataframe(self, country_codes: Any = None) -> pd.DataFrame:
        """Per-country levels in numeraire units (never report these as results)."""
        n = self.w.shape[0]
        index = pd.Index(list(country_codes) if country_codes is not None else range(n), name="country")
        return pd.DataFrame({
            "w": self.w, "Y_hat": self.Y_hat, "Y": self.Y, "T": self.T, "XN": self.XN,
            "PT": self.PT, "TR": self.TR, "P_C": self.P[0], "P_G": self.P[1], "P_X": self.P[2],
            "r_F": self.r_F, "r_Y": self.r_Y,
        }, index=index)

    def to_markdown(self, country_codes: Any = None, **kwargs: Any) -> str:
        """The per-country levels of :meth:`to_dataframe` as Markdown (``country_codes`` labels the rows)."""
        return df_to_markdown(self.to_dataframe(country_codes), **kwargs)

    def to_latex(self, country_codes: Any = None, **kwargs: Any) -> str:
        """The per-country levels of :meth:`to_dataframe` as a LaTeX tabular (``country_codes`` labels the rows)."""
        return df_to_latex(self.to_dataframe(country_codes), **kwargs)

    def to_typst(self, country_codes: Any = None, **kwargs: Any) -> str:
        """The per-country levels of :meth:`to_dataframe` as a Typst table (``country_codes`` labels the rows)."""
        return df_to_typst(self.to_dataframe(country_codes), **kwargs)


def _resolve_numeraire(calib: CondensedCalibration, numeraire: str) -> tuple[str, int | None]:
    """Normalize a numeraire option to ``("factor", None)`` or ``("wage", country_index)``."""
    if numeraire == "factor":
        return "factor", None
    code = numeraire[2:] if numeraire.startswith("w_") and numeraire[2:] in calib.country_codes else numeraire
    if code in calib.country_codes:
        return "wage", calib.index(code)
    raise UnsupportedExtension(
        f"numeraire must be 'factor' or a country code, got {numeraire!r}"
    )


class CondensedLeontiefModel:
    """Residual system of the condensed model for one calibration and one wedge set.

    Parameters
    ----------
    calib : CondensedCalibration
    wedges : TariffWedges
    closure : {"factor", "gdp", "own"}, default "factor"
        Trade-balance closure: shares of world factor income (headline), of
        world GDP at market prices, or of own GDP with the reference country residual.
    numeraire : str, default "factor"
        ``"factor"`` (world factor income at its base value) or a country code
        whose wage is fixed at one.
    fd_rule : {"absorption", "fixed_investment"}, default "absorption"
        ``"absorption"``: ``E^f = theta^f Atil``. ``"fixed_investment"``:
        investment keeps its benchmark real value, ``E^G = A0^G P^G / P0^G``,
        and the remainder is split between C and X by renormalized shares.
    inventories : {"tariffed", "untariffed"}, default "tariffed"
        Whether ``wedges.tau_V`` applies. (The IO ``"merged"`` variant is a
        table-level choice: build the table with ``fd_codes=("C", "G", "X")``.)
    lu_output : optional
        A cached ``scipy.linalg.lu_factor`` of ``I - a`` from another model on
        the same calibration (continuation reuses it).
    """

    def __init__(
        self,
        calib: CondensedCalibration,
        wedges: TariffWedges,
        *,
        closure: str = "factor",
        numeraire: str = "factor",
        fd_rule: str = "absorption",
        inventories: str = "tariffed",
        lu_output: Any = None,
    ) -> None:
        if not isinstance(calib, CondensedCalibration):
            raise TypeError("calib must be a CondensedCalibration")
        if not isinstance(wedges, TariffWedges):
            raise TypeError("wedges must be a TariffWedges")
        if closure not in CLOSURES:
            raise UnsupportedExtension(f"closure must be one of {CLOSURES}, got {closure!r}")
        if fd_rule not in FD_RULES:
            raise UnsupportedExtension(f"fd_rule must be one of {FD_RULES}, got {fd_rule!r}")
        if inventories == "merged":
            raise UnsupportedExtension(
                "inventories='merged' is a table-level choice: build the table with fd_codes=('C', 'G', 'X')"
            )
        if inventories not in INVENTORY_RULES:
            raise UnsupportedExtension(f"inventories must be one of {INVENTORY_RULES}, got {inventories!r}")
        N, S, M = calib.n_countries, calib.n_sectors, calib.n_cells
        if wedges.tau.shape != (M, M) or wedges.tau_fd.shape != (M, N, 3) or wedges.tau_V.shape != (M, N):
            raise CalibrationError("wedge shapes do not match the calibration")
        self.calib = calib
        self.wedges = wedges
        self.closure = closure
        self.numeraire = numeraire
        self.numeraire_kind, self.numeraire_country = _resolve_numeraire(calib, numeraire)
        self.fd_rule = fd_rule
        self.inventories = inventories
        self.n_countries, self.n_sectors, self.n_cells = N, S, M
        self.reference = calib.reference
        self.country_of_cell = calib.country_of_cell

        one_minus_t = 1.0 - calib.t
        if np.any(one_minus_t <= 0):
            raise CalibrationError("1 - t_j <= 0 detected in the cost matrix denominator")
        B_tau = (calib.a * wedges.tau) / one_minus_t[None, :]
        self.B_tau = B_tau
        self.b_scaled = calib.b / one_minus_t
        self.LU_P = la.lu_factor(np.eye(M, dtype=float) - B_tau.T)
        self.LU_Y = la.lu_factor(np.eye(M, dtype=float) - calib.a) if lu_output is None else lu_output
        self.Atar_T = np.ascontiguousarray(((wedges.tau - 1.0) * calib.a).T)
        self.omtau = np.ascontiguousarray(calib.omega * wedges.tau_fd)
        self.omega = calib.omega
        self.qV = calib.qV
        self.TV = calib.TV
        self.tau_V = wedges.tau_V if inventories == "tariffed" else np.ones((M, N), dtype=float)
        self.Fbar = calib.Fbar
        self.Fbar_sum = float(calib.Fbar.sum())
        self.Y0 = calib.Y0
        self.A0 = calib.A0
        self.theta = calib.theta
        self.tfd = calib.tfd
        self.one_minus_tfd = 1.0 - calib.tfd
        self.idx_C = ENDOGENOUS_FD.index("C")
        self.idx_G = ENDOGENOUS_FD.index("G")
        self.idx_X = ENDOGENOUS_FD.index("X")
        self.z0 = np.concatenate([np.zeros(N, dtype=float), np.ones(N, dtype=float)])

    @property
    def n_vars(self) -> int:
        """Number of unknowns, ``2N``."""
        return 2 * self.n_countries

    @property
    def options(self) -> dict[str, str]:
        """Closure, numeraire, final-demand rule and inventory treatment of this model."""
        return {"closure": self.closure, "numeraire": self.numeraire,
                "fd_rule": self.fd_rule, "inventories": self.inventories}

    def with_wedges(self, wedges: TariffWedges) -> "CondensedLeontiefModel":
        """Same calibration and options with other wedges (reuses the output factorization)."""
        return CondensedLeontiefModel(self.calib, wedges, closure=self.closure, numeraire=self.numeraire,
                                      fd_rule=self.fd_rule, inventories=self.inventories, lu_output=self.LU_Y)

    def prices(self, w: np.ndarray) -> np.ndarray:
        """Exact basic seller prices ``p(w) = (I - B_tau^T)^{-1} [(b / (1 - t)) o w_c(.)]``."""
        rhs = self.b_scaled * w[self.country_of_cell]
        return la.lu_solve(self.LU_P, rhs)

    def trade_balance(self, w: np.ndarray, Y: np.ndarray) -> np.ndarray:
        """Fob trade balances ``XN`` from the closure (E5)."""
        if self.closure == "factor":
            return self.calib.s_factor * float(w @ self.Fbar)
        if self.closure == "gdp":
            return self.calib.s_gdp * float(Y.sum())
        xn = self.calib.s_own * Y
        xn[self.reference] = -(xn.sum() - xn[self.reference])
        return xn

    def state(self, z: np.ndarray, p: np.ndarray | None = None) -> CondensedState:
        """Evaluate the full state at ``z = (omega, Y_hat)``.

        Parameters
        ----------
        z : np.ndarray, shape (2N,)
        p : np.ndarray, shape (M,), optional
            Precomputed prices ``p(w)``; reused across the ``Y_hat`` columns of
            the finite-difference Jacobian, where ``w`` does not change.
        """
        N = self.n_countries
        z = np.asarray(z, dtype=float)
        if z.shape != (2 * N,):
            raise ValueError(f"z must have shape {(2 * N,)}")
        omega = z[:N]
        Y_hat = z[N:]
        w = np.exp(omega)
        Y = Y_hat * self.Y0
        if p is None:
            p = self.prices(w)
        else:
            p = np.asarray(p, dtype=float)
            if p.shape != (self.n_cells,):
                raise ValueError("p must have one entry per cell")
            if p.flags.writeable:
                p = p.copy()
        P = np.einsum("ikf,i->fk", self.omtau, p)
        P_tilde = np.einsum("ikf,i->fk", self.omega, p)
        XN = self.trade_balance(w, Y)
        PG = P[self.idx_G, :]
        inv_goods = np.einsum("id,i->d", self.tau_V * self.qV, p)
        E_V = inv_goods + self.TV * PG
        A_tilde = Y - XN - E_V
        E = self.theta.T * A_tilde[None, :]
        if self.fd_rule == "fixed_investment":
            E[self.idx_G] = self.A0[:, self.idx_G] * P[self.idx_G] / self.calib.P0[self.idx_G]
            remaining = A_tilde - E[self.idx_G]
            shares = self.theta[:, [self.idx_C, self.idx_X]] / (1 - self.theta[:, self.idx_G, None])
            E[[self.idx_C, self.idx_X]] = shares.T * remaining
        q = (self.one_minus_tfd.T * E) / P
        FD_tax = np.sum(self.tfd.T * E, axis=0)
        endo_demand = np.einsum("ikf,fk->i", self.omega, q)
        d = endo_demand + self.qV.sum(axis=1)
        y = la.lu_solve(self.LU_Y, d)
        F_dem = np.bincount(self.country_of_cell, weights=self.calib.b * y, minlength=N)
        r_F = (F_dem - self.Fbar) / self.Fbar
        ind_tax = np.bincount(self.country_of_cell, weights=self.calib.t * p * y, minlength=N)
        PT = ind_tax + FD_tax + self.TV * PG
        interm_tar = np.bincount(self.country_of_cell, weights=y * (self.Atar_T @ p), minlength=N)
        fd_tar = np.sum((P - P_tilde) * q, axis=0)
        inv_tar = np.einsum("id,i->d", (self.tau_V - 1.0) * self.qV, p)
        TR = interm_tar + fd_tar + inv_tar
        T = Y - w * self.Fbar
        r_Y = (Y - w * self.Fbar - PT - TR) / self.Y0
        if self.numeraire_kind == "factor":
            r_N = float((w @ self.Fbar - self.Fbar_sum) / self.Fbar_sum)
        else:
            r_N = float(omega[self.numeraire_country])
        residual = np.empty(2 * N, dtype=float)
        residual[:N] = r_F
        residual[N:] = r_Y
        walras_residual = float(r_Y[self.reference])
        residual[N + self.reference] = r_N
        return CondensedState(
            z=_ro(z.copy()), w=_ro(w), Y=_ro(Y), Y_hat=_ro(Y_hat.copy()), T=_ro(T), p=_ro(p), y=_ro(y),
            P=_ro(P), P_tilde=_ro(P_tilde), E_V=_ro(E_V), A_tilde=_ro(A_tilde), E=_ro(E), q=_ro(q),
            XN=_ro(np.asarray(XN, dtype=float)), PT=_ro(PT), TR=_ro(TR), r_F=_ro(r_F), r_Y=_ro(r_Y),
            r_N=r_N, residual=_ro(residual), walras_residual=walras_residual,
            metadata={"closure": self.closure, "numeraire": self.numeraire, "fd_rule": self.fd_rule,
                      "inventories": self.inventories},
        )

    def residual(self, z: np.ndarray, p: np.ndarray | None = None) -> np.ndarray:
        """The ``2N`` scaled residual vector ``r(z)``."""
        return self.state(z, p=p).residual

    def dropped_walras_residual(self, z: np.ndarray, p: np.ndarray | None = None) -> float:
        """The dropped income residual of the reference country (Walras' law check)."""
        return self.state(z, p=p).walras_residual

    def jacobian(self, z: np.ndarray, r0: np.ndarray | None = None, p0: np.ndarray | None = None,
                 h: float = 1e-7, *, scheme: str = "forward") -> np.ndarray:
        """Finite-difference Jacobian ``(2N, 2N)`` with prices cached across the ``Y_hat`` columns.

        Parameters
        ----------
        z : np.ndarray, shape (2N,)
        r0, p0 : optional
            Residual and prices at ``z`` (forward scheme only).
        h : float, default 1e-7
            Relative step ``h * max(1, |z_j|)``.
        scheme : {"forward", "central"}
            The production scheme is forward differences, as in the IO
            engine; central differences are provided for tests.
        """
        N = self.n_countries
        dim = 2 * N
        z = np.asarray(z, dtype=float)
        if scheme not in ("forward", "central"):
            raise ValueError("scheme must be 'forward' or 'central'")
        J = np.empty((dim, dim), dtype=float)
        zp = z.copy()
        if scheme == "central":
            zm = z.copy()
            p_base = self.prices(np.exp(z[:N]))
            for j in range(dim):
                step = h * max(1.0, abs(z[j]))
                zp[j] += step
                zm[j] -= step
                p_arg = None if j < N else p_base
                J[:, j] = (self.residual(zp, p=p_arg) - self.residual(zm, p=p_arg)) / (2 * step)
                zp[j] = z[j]
                zm[j] = z[j]
            return J
        if r0 is None or p0 is None:
            st0 = self.state(z)
            r0 = st0.residual
            p0 = st0.p
        for j in range(dim):
            step = h * max(1.0, abs(z[j]))
            zp[j] += step
            p_arg = None if j < N else p0
            rj = self.residual(zp, p=p_arg)
            J[:, j] = (rj - r0) / step
            zp[j] = z[j]
        return J


__all__ = ["CondensedLeontiefModel", "CondensedState"]
