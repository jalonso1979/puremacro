"""Numeraire-free price, welfare and trade measures of the condensed model (M1-M10).

All measures are invariant to the numeraire (world factor income, ``w_k = 1``
for any k): re-solving under another numeraire changes them by rounding only.
Base prices equal one and ``P^U_k = prod_f (P^f_k / P0^f_k)^theta^f_k`` is the
Cobb-Douglas cost-of-living index over the C, G, X baskets.

* M1 ``consumer_price_relative_to_wages``: ``Pi^C_k = P^C_k / (P0^C_k w_k) - 1``,
  the inverse real consumption wage. It is not a CPI inflation rate.
* M2 ``ev_split``: ``EV_k = Atil^1_k / P^U_k - Atil^0_k`` in base-price units,
  reported as percent of base GDP, with the exact additive split
  ``EV_k / Y0_k = (Fbar_k / Y0_k)(w_k / P^U_k - 1) + (T_k / P^U_k - T0_k) / Y0_k
  - ((XN_k + E^V_k) / P^U_k - XN0_k - EV0_k) / Y0_k``; the revenue term splits
  into tariff revenue ``TR_k / (P^U_k Y0_k) - TR0_k / Y0_k`` and other taxes.
  Static EV includes investment in the absorption composite (IO CHANGES.md).
* M3 ``real_national_income``: ``(Y_k / P^U_k) / Y0_k - 1``.
* M4 ``tot_fisher``: Fisher fob ex-duty terms of trade
  ``sqrt(L^X_k P^X_k / (L^M_k P^M_k)) - 1`` with Laspeyres/Paasche unit values of
  foreign sales and foreign purchases at base quantities.
* M5 ``tariff_revenue_and_effective_rate``: ``TR_k / Y_k`` and the effective
  rate on dutiable imports ``TR_k / sum_dutiable p_i * flow``.
* M6 ``constant_price_rgdp``: ``sum_{j in k} (VA_j + TLS_j) yhat_j + sum_f TFD^f_k
  qhat^f_k + TFD^V_k`` relative to ``Y0_k``; a composition effect only.
* M8 ``pe_unit_cost_bound``: the Leontief unit cost at base factor prices
  ``p^PE = (I - B_tau^T)^{-1} (b / (1 - t))`` and the implied
  ``Pi^{C,PE}``; ``pe_first_order_decomposition``: direct ``omega^C . delta^C``
  plus the intermediate cascade ``(omega^C o tau^C_0)^T (I - Gamma^T)^{-1} g``.
* M9 ``aggregation_gaps``: fine minus coarse gaps of any measure dictionary.
* M10 ``sector_incidence``: ``ln(p_j / w_k)``, ``ln(y_j / y0_j)`` and the
  output-weighted within-group price dispersion for an explicit group vector.

``compute_measures`` gathers them into a :class:`CondensedMeasuresResult`.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Mapping, Sequence

import numpy as np
import pandas as pd
import scipy.linalg as la

from puremacro.reports import df_to_latex, df_to_markdown, df_to_typst

from .calibration import CondensedCalibration
from .model import CondensedState
from .table import ENDOGENOUS_FD, FD_CODES
from .tariffs import TariffWedges


def _country(calib: CondensedCalibration, country: str | int) -> int:
    if isinstance(country, (int, np.integer)):
        k = int(country)
        if not 0 <= k < calib.n_countries:
            raise ValueError("country index out of range")
        return k
    return calib.index(str(country))


def _price_index(calib: CondensedCalibration, state: CondensedState, k: int) -> float:
    return float(np.prod((state.P[:, k] / calib.P0[:, k]) ** calib.theta[k, :]))


def _ev_all(calib: CondensedCalibration, state: CondensedState) -> np.ndarray:
    """Equivalent variation of every country in base-price units."""
    out = np.empty(calib.n_countries)
    for k in range(calib.n_countries):
        out[k] = state.A_tilde[k] / _price_index(calib, state, k) - calib.Atil0[k]
    return out


def _y_hat(calib: CondensedCalibration, state: CondensedState) -> np.ndarray:
    """``y / y0`` on every cell (phantoms included, as in the certificate; ``y0 > 0`` by calibration)."""
    return state.y / calib.y0


def _q_hat(calib: CondensedCalibration, state: CondensedState) -> np.ndarray:
    endo = [FD_CODES.index(c) for c in ENDOGENOUS_FD]
    B0 = calib.table.F[:, :, endo].sum(axis=0)
    return np.where(B0.T > 0, state.q / np.where(B0.T > 0, B0.T, 1.0), 1.0)


@dataclass(frozen=True)
class EVSplit:
    """Exact three-way additive equivalent-variation decomposition (pp of base GDP).

    ``ev_pct_base_gdp = factor_income_pp + revenue_pp + deficit_inventory_pp`` up
    to ``residual_pp`` (rounding); ``revenue_pp = tariff_revenue_pp + other_taxes_pp``.
    """

    ev_pct_base_gdp: float
    factor_income_pp: float
    revenue_pp: float
    tariff_revenue_pp: float
    other_taxes_pp: float
    deficit_inventory_pp: float
    residual_pp: float
    country: str = ""
    metadata: dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict[str, float]:
        """The seven split components by name (percent and percentage points of base GDP)."""
        return {"ev_pct_base_gdp": self.ev_pct_base_gdp, "factor_income_pp": self.factor_income_pp,
                "revenue_pp": self.revenue_pp, "tariff_revenue_pp": self.tariff_revenue_pp,
                "other_taxes_pp": self.other_taxes_pp, "deficit_inventory_pp": self.deficit_inventory_pp,
                "residual_pp": self.residual_pp}

    def to_dataframe(self) -> pd.DataFrame:
        """One row per component (EV, the three terms, the revenue sub-split and the residual)."""
        rows = [("Equivalent variation (% of base GDP)", self.ev_pct_base_gdp),
                ("(i) real factor income (pp)", self.factor_income_pp),
                ("(ii) real net revenue (pp)", self.revenue_pp),
                ("(ii-a) tariff revenue (pp)", self.tariff_revenue_pp),
                ("(ii-b) other net taxes (pp)", self.other_taxes_pp),
                ("(iii) deficit and inventory transfer (pp)", self.deficit_inventory_pp),
                ("residual (pp)", self.residual_pp)]
        return pd.DataFrame(rows, columns=["Component", "Value"]).set_index("Component")

    def summary(self) -> str:
        """One line: EV equals factor income plus net revenue plus the transfer term, with the residual."""
        return (f"EV split [{self.country}]: EV {self.ev_pct_base_gdp:.6f}% of base GDP = factor income "
                f"{self.factor_income_pp:.6f} + net revenue {self.revenue_pp:.6f} + transfer "
                f"{self.deficit_inventory_pp:.6f} (residual {self.residual_pp:.1e})")

    def to_markdown(self, **kwargs: Any) -> str:
        """The component table of :meth:`to_dataframe` as Markdown."""
        return df_to_markdown(self.to_dataframe(), **kwargs)

    def to_latex(self, **kwargs: Any) -> str:
        """The component table of :meth:`to_dataframe` as a LaTeX tabular."""
        return df_to_latex(self.to_dataframe(), **kwargs)

    def to_typst(self, **kwargs: Any) -> str:
        """The component table of :meth:`to_dataframe` as a Typst table."""
        return df_to_typst(self.to_dataframe(), **kwargs)


def consumer_price_relative_to_wages(calib: CondensedCalibration, state: CondensedState, *,
                                     country: str | int) -> float:
    """M1: ``Pi^C_k = P^C_k / (P0^C_k w_k) - 1`` (fraction), the inverse real consumption wage."""
    k = _country(calib, country)
    idx_C = ENDOGENOUS_FD.index("C")
    return float(state.P[idx_C, k] / calib.P0[idx_C, k] / state.w[k] - 1.0)


def ev_split(calib: CondensedCalibration, state: CondensedState, *, country: str | int) -> EVSplit:
    """M2: equivalent variation as percent of base GDP with its exact additive split."""
    k = _country(calib, country)
    PU = _price_index(calib, state, k)
    Y0 = float(calib.Y0[k])
    ev_units = float(state.A_tilde[k]) / PU - float(calib.Atil0[k])
    ev_pct = 100.0 * ev_units / Y0
    wk, Fbar = float(state.w[k]), float(calib.Fbar[k])
    T1, T0 = float(state.T[k]), float(calib.T0[k] + calib.TR0[k])
    XN1, XN0 = float(state.XN[k]), float(calib.XN0[k])
    EV1, EV0 = float(state.E_V[k]), float(calib.EV0[k])
    TR1 = float(state.TR[k])
    split_factor = 100.0 * (Fbar / Y0) * (wk / PU - 1.0)
    split_revenue = 100.0 * (T1 / PU - T0) / Y0
    split_tr = 100.0 * (TR1 / PU - float(calib.TR0[k])) / Y0
    split_other = 100.0 * ((T1 - TR1) / PU - float(calib.T0[k])) / Y0
    split_transfer = 100.0 * ((XN1 + EV1) / PU - (XN0 + EV0)) / Y0
    residual = split_factor + split_revenue - split_transfer - ev_pct
    return EVSplit(ev_pct_base_gdp=ev_pct, factor_income_pp=split_factor, revenue_pp=split_revenue,
                   tariff_revenue_pp=float(split_tr), other_taxes_pp=float(split_other),
                   deficit_inventory_pp=-split_transfer, residual_pp=residual,
                   country=calib.country_codes[k], metadata={"price_index": PU, "ev_base_units": ev_units})


def real_national_income(calib: CondensedCalibration, state: CondensedState, *, country: str | int) -> float:
    """M3: ``(Y_k / P^U_k) / Y0_k - 1`` (fraction)."""
    k = _country(calib, country)
    return float((state.Y[k] / _price_index(calib, state, k)) / calib.Y0[k] - 1.0)


def tot_fisher(calib: CondensedCalibration, state: CondensedState, *, country: str | int) -> float:
    """M4: Fisher fob ex-duty terms of trade (fraction).

    ``ToT_k = sqrt(L^X_k P^X_k / (L^M_k P^M_k)) - 1`` where ``X0, X1`` are foreign
    sales of goods produced in k and ``M0, M1`` are k's purchases of foreign
    goods, at base and counterfactual quantities (``Z yhat`` and ``F qhat``),
    all valued at the seller prices ``p``.
    """
    n, s, m = calib.n_countries, calib.n_sectors, calib.n_cells
    k = _country(calib, country)
    table = calib.table
    Z = table.Z
    F = table.F[:, :, [FD_CODES.index(c) for c in ENDOGENOUS_FD]]
    y_hat = _y_hat(calib, state)
    q_hat = _q_hat(calib, state)
    p = state.p
    Zq = Z * y_hat[None, :]
    Fq = np.einsum("ikf,fk->ik", F, q_hat)
    F0 = F.sum(axis=2)
    rows_k = slice(k * s, (k + 1) * s)
    cols_k = slice(k * s, (k + 1) * s)
    foreign_buyers = np.ones(m, dtype=bool)
    foreign_buyers[cols_k] = False
    foreign_dest = np.ones(n, dtype=bool)
    foreign_dest[k] = False
    X0_i = Z[rows_k, :][:, foreign_buyers].sum(axis=1) + F0[rows_k, :][:, foreign_dest].sum(axis=1)
    X1_i = Zq[rows_k, :][:, foreign_buyers].sum(axis=1) + Fq[rows_k, :][:, foreign_dest].sum(axis=1)
    pk = p[rows_k]
    sum_X0, sum_X1 = float(X0_i.sum()), float(X1_i.sum())
    if sum_X0 > 0 and sum_X1 > 0:
        LX = float(X0_i @ pk) / sum_X0
        PX = float(X1_i @ pk) / sum_X1
    else:
        LX = PX = 1.0
    foreign_origins = np.ones(m, dtype=bool)
    foreign_origins[rows_k] = False
    M0_i = Z[foreign_origins, cols_k].sum(axis=1) + F0[foreign_origins, k]
    M1_i = Zq[foreign_origins, cols_k].sum(axis=1) + Fq[foreign_origins, k]
    p_foreign = p[foreign_origins]
    sum_M0, sum_M1 = float(M0_i.sum()), float(M1_i.sum())
    if sum_M0 > 0 and sum_M1 > 0:
        LM = float(M0_i @ p_foreign) / sum_M0
        PM = float(M1_i @ p_foreign) / sum_M1
    else:
        LM = PM = 1.0
    return float(np.sqrt((LX * PX) / (LM * PM)) - 1.0)


def constant_price_rgdp(calib: CondensedCalibration, state: CondensedState, *, country: str | int) -> float:
    """M6: constant-price real GDP relative to the benchmark (fraction)."""
    k = _country(calib, country)
    table = calib.table
    rows = slice(k * calib.n_sectors, (k + 1) * calib.n_sectors)
    yh_k = state.y[rows] / calib.y0[rows]
    prod_term = float(np.sum((table.VA + table.TLS)[rows] * yh_k))
    endo = [FD_CODES.index(c) for c in ENDOGENOUS_FD]
    iv = FD_CODES.index("V")
    B0_k = table.F[:, k, endo].sum(axis=0)
    qh_k = np.where(B0_k > 0, state.q[:, k] / np.where(B0_k > 0, B0_k, 1.0), 1.0)
    tfd_endo = float(np.sum(table.TFD[k, endo] * qh_k))
    tfd_v = float(table.TFD[k, iv])
    return float((prod_term + tfd_endo + tfd_v) / calib.Y0[k] - 1.0)


def _inventory_treatment(state: CondensedState, inventories: str | None) -> str:
    """The inventory treatment to value dutiable inventories with: explicit, else the state's own."""
    treatment = state.metadata.get("inventories", "tariffed") if inventories is None else inventories
    if treatment not in ("tariffed", "untariffed"):
        raise ValueError(f"inventories must be 'tariffed' or 'untariffed', got {treatment!r}")
    return str(treatment)


def tariff_revenue_and_effective_rate(calib: CondensedCalibration, wedges: TariffWedges, state: CondensedState,
                                      *, country: str | int, inventories: str | None = None) -> tuple[float, float]:
    """M5: ``(100 TR_k / Y_k, 100 TR_k / dutiable imports)`` for the importing country ``k``.

    Dutiable imports are the counterfactual producer values ``p_i * flow`` of
    every purchase whose multiplier exceeds one (intermediate, final and
    inventory). Inventory purchases count as dutiable only when the state was
    solved with ``inventories="tariffed"``: under ``"untariffed"`` the model
    collects no inventory duty, whatever ``wedges.tau_V`` holds, so ``tau_V`` is
    treated as one. The treatment is read from ``state.metadata["inventories"]``
    (set by :class:`CondensedLeontiefModel`) unless ``inventories`` is given. A
    uniform rate ``r`` on all dutiable purchases gives exactly ``100 r``. The
    effective rate is zero when nothing is dutiable. Both numbers refer to the
    requested ``country`` (the IO engine reported the effective rate for the
    United States regardless of the country asked for, and always counted
    inventories with ``tau_V > 1`` as dutiable; both are documented deviations).
    """
    treatment = _inventory_treatment(state, inventories)
    k = _country(calib, country)
    tr_share = 100.0 * float(state.TR[k] / state.Y[k])
    s = calib.n_sectors
    table = calib.table
    yh = _y_hat(calib, state)
    qh = _q_hat(calib, state)
    p = state.p
    Zq = (table.Z * yh[None, :]) * p[:, None]
    cols = slice(k * s, (k + 1) * s)
    tau_k = wedges.tau[:, cols]
    dutiable_z = Zq[:, cols][tau_k > 1.0].sum()
    endo = [FD_CODES.index(c) for c in ENDOGENOUS_FD]
    Fq = (table.F[:, k, endo] * qh.T[k, :]) * p[:, None]
    dutiable_f = Fq[wedges.tau_fd[:, k, :] > 1.0].sum()
    iv = FD_CODES.index("V")
    Vq = table.F[:, k, iv] * p
    dutiable_v = Vq[wedges.tau_V[:, k] > 1.0].sum() if treatment == "tariffed" else 0.0
    total = float(dutiable_z + dutiable_f + dutiable_v)
    effective = 100.0 * float(state.TR[k] / total) if total > 1e-12 else 0.0
    return tr_share, effective


def pe_unit_cost_bound(calib: CondensedCalibration, wedges: TariffWedges, *,
                       country: str | int) -> tuple[np.ndarray, float]:
    """M8: the partial-equilibrium unit-cost prices ``p^PE`` and ``Pi^{C,PE}_k`` (fraction).

    ``p^PE = (I - B_tau^T)^{-1} (b / (1 - t))`` is zero profit at base factor
    prices; ``Pi^{C,PE}_k = sum_i omega^C_ik tau^C_ik p^PE_i / P0^C_k - 1``.
    """
    k = _country(calib, country)
    one_minus_t = 1.0 - calib.t
    B_tau = (calib.a * wedges.tau) / one_minus_t[None, :]
    p_PE = la.solve(np.eye(calib.n_cells, dtype=float) - B_tau.T, calib.b / one_minus_t)
    idx_C = ENDOGENOUS_FD.index("C")
    omtau_C = calib.omega[:, k, idx_C] * wedges.tau_fd[:, k, idx_C]
    return p_PE, float(omtau_C @ p_PE / calib.P0[idx_C, k] - 1.0)


def pe_first_order_decomposition(calib: CondensedCalibration, wedges: TariffWedges, *,
                                 country: str | int) -> tuple[float, float, float]:
    """M8 first order: ``(total, direct, cascade)`` of the consumer-price change (fractions).

    ``direct = omega^C . delta^C`` and ``cascade = (omega^C o tau^C_0)^T (I - Gamma^T)^{-1} g``
    with ``Gamma = a tau_0 / (1 - t)`` and ``g_j = sum_i a_ij delta_ij / (1 - t_j)``,
    where ``tau_0`` are the separated baseline duties (one when none).
    """
    k = _country(calib, country)
    idx_C = ENDOGENOUS_FD.index("C")
    omega_C = calib.omega[:, k, idx_C] / calib.P0[idx_C, k]
    base = calib.baseline_tariffs
    tau0 = 1.0 if base is None else base.wedges.tau
    tauC0 = 1.0 if base is None else base.wedges.tau_fd[:, k, idx_C]
    delta_C = wedges.tau_fd[:, k, idx_C] - tauC0
    direct = float(omega_C @ delta_C)
    one_minus_t = 1.0 - calib.t
    Gamma = calib.a * tau0 / one_minus_t[None, :]
    g = np.sum(calib.a * (wedges.tau - tau0) / one_minus_t[None, :], axis=0)
    cascade = float((omega_C * tauC0) @ la.solve(np.eye(calib.n_cells, dtype=float) - Gamma.T, g))
    return direct + cascade, direct, cascade


def sector_incidence(calib: CondensedCalibration, state: CondensedState, *, country: str | int,
                     groups: Sequence[int] | None = None) -> tuple[np.ndarray, np.ndarray, dict[int, float]]:
    """M10: ``ln(p_j / w_k)``, ``ln(y_j / y0_j)`` and output-weighted within-group price dispersion.

    Parameters
    ----------
    groups : sequence of int, optional
        Group id of every sector (any integers). The dispersion of group G is
        ``sum_{s in G} (y0_s / y0_G) (ln p_s - mean_G)^2`` with the
        output-weighted mean; an empty dictionary when ``groups`` is None.
    """
    k = _country(calib, country)
    s = calib.n_sectors
    rows = slice(k * s, (k + 1) * s)
    pk = state.p[rows]
    wk = float(state.w[k])
    yk = state.y[rows]
    y0k = calib.y0[rows]
    dln_p_minus_dln_w = np.log(pk) - np.log(wk)
    dln_y = np.log(np.maximum(yk / np.maximum(y0k, 1e-300), 1e-300))
    dispersion: dict[int, float] = {}
    if groups is not None:
        grp = np.asarray(groups, dtype=int)
        if grp.shape != (s,):
            raise ValueError(f"groups must have one entry per sector ({s})")
        for g in np.unique(grp):
            mem = grp == g
            tot = float(y0k[mem].sum())
            if tot > 0:
                weights = y0k[mem] / tot
                mean = float(np.sum(weights * np.log(pk[mem])))
                dispersion[int(g)] = float(np.sum(weights * (np.log(pk[mem]) - mean) ** 2))
            else:
                dispersion[int(g)] = 0.0
    return dln_p_minus_dln_w, dln_y, dispersion


def aggregation_gaps(fine: Mapping[str, float], coarse: Mapping[str, float]) -> dict[str, dict[str, float]]:
    """M9: absolute gap ``fine - coarse`` and relative gap ``gap / |coarse|`` for shared keys.

    The numerics are those of the IO engine (the relative gap is NaN when
    ``|coarse| <= 1e-12``). The names are generic because any fine/coarse pair
    of measure dictionaries qualifies: the IO arguments ``measures_45`` and
    ``measures_11`` became ``fine`` and ``coarse``, and the IO output keys
    ``val_45``, ``val_11``, ``absolute_gap_pp`` became ``fine``, ``coarse``,
    ``absolute_gap`` (``relative_gap`` is unchanged); a documented rename.
    """
    gaps: dict[str, dict[str, float]] = {}
    for key in fine:
        if key in coarse:
            f, c = float(fine[key]), float(coarse[key])
            gap = f - c
            gaps[key] = {"fine": f, "coarse": c, "absolute_gap": gap,
                         "relative_gap": gap / abs(c) if abs(c) > 1e-12 else float("nan")}
    return gaps


@dataclass(frozen=True)
class CondensedMeasuresResult:
    """Headline measures of one equilibrium for one country (all in percent).

    Attributes
    ----------
    country : str
    pc_over_w_pct : float
        M1, consumer prices relative to wages.
    ev_pct_base_gdp : float
        M2, equivalent variation as percent of base GDP.
    ev_split : EVSplit
    real_national_income_pct : float
        M3.
    tot_fisher_pct : float
        M4.
    rgdp_base_prices_pct : float
        M6.
    tariff_revenue_share_pct, effective_rate_dutiable_pct : float
        M5.
    pc_pe_pct, ge_minus_pe_pct : float
        M8: the PE unit-cost consumer price and ``100 [ln(1 + Pi^C) - ln(1 + Pi^{C,PE})]``.
    world_ev_pct : float
        ``100 sum_k EV_k / sum_k Y0_k``.
    bloc_ev_pct, partner_ev_pct : dict
        EV of requested blocs (``100 sum EV / sum Y0``) and partners (percent of own base GDP).
    metadata : dict
        Provenance (numeraire unit tag, wedge digest, model options).
    """

    country: str
    pc_over_w_pct: float
    ev_pct_base_gdp: float
    ev_split: EVSplit
    real_national_income_pct: float
    tot_fisher_pct: float
    rgdp_base_prices_pct: float
    tariff_revenue_share_pct: float
    effective_rate_dutiable_pct: float
    pc_pe_pct: float
    ge_minus_pe_pct: float
    world_ev_pct: float
    bloc_ev_pct: dict[str, float] = field(default_factory=dict)
    partner_ev_pct: dict[str, float] = field(default_factory=dict)
    metadata: dict[str, Any] = field(default_factory=dict)

    def headline(self) -> dict[str, float]:
        """The seven headline measures by name."""
        return {"pc_over_w_pct": self.pc_over_w_pct, "ev_pct_base_gdp": self.ev_pct_base_gdp,
                "tot_fisher_pct": self.tot_fisher_pct, "rgdp_base_prices_pct": self.rgdp_base_prices_pct,
                "tariff_revenue_share_pct": self.tariff_revenue_share_pct,
                "effective_rate_dutiable_pct": self.effective_rate_dutiable_pct,
                "real_national_income_pct": self.real_national_income_pct}

    def to_dict(self) -> dict[str, Any]:
        """Plain-JSON representation: headline, EV split, PE objects, world, bloc and partner EV, metadata."""
        return {"country": self.country, **self.headline(), "ev_split": self.ev_split.to_dict(),
                "pc_pe_pct": self.pc_pe_pct, "ge_minus_pe_pct": self.ge_minus_pe_pct,
                "world_ev_pct": self.world_ev_pct, "bloc_ev_pct": dict(self.bloc_ev_pct),
                "partner_ev_pct": dict(self.partner_ev_pct), "metadata": dict(self.metadata)}

    def to_dataframe(self) -> pd.DataFrame:
        """One-column table (the country) with one row per measure, split component, bloc and partner."""
        rows = [("Consumer prices relative to wages (%)", self.pc_over_w_pct),
                ("Equivalent variation (% of base GDP)", self.ev_pct_base_gdp),
                ("  real factor income (pp)", self.ev_split.factor_income_pp),
                ("  real net revenue (pp)", self.ev_split.revenue_pp),
                ("    tariff revenue (pp)", self.ev_split.tariff_revenue_pp),
                ("    other net taxes (pp)", self.ev_split.other_taxes_pp),
                ("  deficit and inventory transfer (pp)", self.ev_split.deficit_inventory_pp),
                ("Real national income (%)", self.real_national_income_pct),
                ("Fisher terms of trade, fob ex-duty (%)", self.tot_fisher_pct),
                ("Real GDP at base prices (%)", self.rgdp_base_prices_pct),
                ("Tariff revenue (% of GDP)", self.tariff_revenue_share_pct),
                ("Effective rate on dutiable imports (%)", self.effective_rate_dutiable_pct),
                ("PE unit-cost consumer price (%)", self.pc_pe_pct),
                ("GE minus PE relative-wage term (%)", self.ge_minus_pe_pct),
                ("World EV (% of world base GDP)", self.world_ev_pct)]
        rows += [(f"EV {name} (% of bloc base GDP)", v) for name, v in self.bloc_ev_pct.items()]
        rows += [(f"EV {code} (% of own base GDP)", v) for code, v in self.partner_ev_pct.items()]
        return pd.DataFrame(rows, columns=["Measure", self.country]).set_index("Measure")

    def summary(self) -> str:
        """One line with the headline measures of the country."""
        return (f"Condensed measures [{self.country}]: consumer prices relative to wages "
                f"{self.pc_over_w_pct:+.4f}%, EV {self.ev_pct_base_gdp:+.4f}% of base GDP, Fisher ToT "
                f"{self.tot_fisher_pct:+.4f}%, tariff revenue {self.tariff_revenue_share_pct:.3f}% of GDP "
                f"(effective rate {self.effective_rate_dutiable_pct:.2f}%); numeraire-free")

    def to_markdown(self, **kwargs: Any) -> str:
        """The measure table of :meth:`to_dataframe` as Markdown."""
        return df_to_markdown(self.to_dataframe(), **kwargs)

    def to_latex(self, **kwargs: Any) -> str:
        """The measure table of :meth:`to_dataframe` as a LaTeX tabular."""
        return df_to_latex(self.to_dataframe(), **kwargs)

    def to_typst(self, **kwargs: Any) -> str:
        """The measure table of :meth:`to_dataframe` as a Typst table."""
        return df_to_typst(self.to_dataframe(), **kwargs)


def compute_measures(
    calib: CondensedCalibration,
    wedges: TariffWedges,
    state: CondensedState,
    *,
    country: str | int,
    partners: Sequence[str] = (),
    blocs: Mapping[str, Sequence[str]] | None = None,
    inventories: str | None = None,
) -> CondensedMeasuresResult:
    """Gather M1-M8 for one country plus world, bloc and partner EV.

    Parameters
    ----------
    country : str or int
        The importing country the headline refers to.
    inventories : {"tariffed", "untariffed"}, optional
        Inventory treatment used for the dutiable base of the effective rate;
        read from ``state.metadata["inventories"]`` when omitted.
    partners : sequence of str
        Countries whose EV (percent of own base GDP) is reported.
    blocs : mapping name -> country codes, optional
        Blocs whose aggregate EV ``100 sum EV / sum Y0`` is reported; codes
        absent from the table are ignored, an empty bloc gives 0.
    """
    k = _country(calib, country)
    code = calib.country_codes[k]
    pc = 100.0 * consumer_price_relative_to_wages(calib, state, country=k)
    split = ev_split(calib, state, country=k)
    rni = 100.0 * real_national_income(calib, state, country=k)
    tot = 100.0 * tot_fisher(calib, state, country=k)
    rgdp = 100.0 * constant_price_rgdp(calib, state, country=k)
    tr_share, eff = tariff_revenue_and_effective_rate(calib, wedges, state, country=k, inventories=inventories)
    _, pi_pe = pe_unit_cost_bound(calib, wedges, country=k)
    ge_minus_pe = 100.0 * (np.log(1.0 + pc / 100.0) - np.log(1.0 + pi_pe))
    ev_units = _ev_all(calib, state)
    world = 100.0 * float(ev_units.sum() / calib.Y0.sum())
    bloc_out: dict[str, float] = {}
    for name, members in (blocs or {}).items():
        idx = [calib.index(c) for c in members if c in calib.country_codes]
        y0 = float(calib.Y0[idx].sum()) if idx else 0.0
        bloc_out[str(name)] = 100.0 * float(ev_units[idx].sum() / y0) if y0 > 0 else 0.0
    partner_out = {}
    for c in partners:
        j = calib.index(str(c))
        partner_out[str(c)] = 100.0 * float(ev_units[j] / calib.Y0[j])
    return CondensedMeasuresResult(
        country=code, pc_over_w_pct=pc, ev_pct_base_gdp=split.ev_pct_base_gdp, ev_split=split,
        real_national_income_pct=rni, tot_fisher_pct=tot, rgdp_base_prices_pct=rgdp,
        tariff_revenue_share_pct=tr_share, effective_rate_dutiable_pct=eff, pc_pe_pct=100.0 * pi_pe,
        ge_minus_pe_pct=float(ge_minus_pe), world_ev_pct=world, bloc_ev_pct=bloc_out, partner_ev_pct=partner_out,
        metadata={"unit": "numeraire-free (world factor income)", "model": "condensed_leontief_one_factor",
                  "wedges_sha256": wedges.sha256, **dict(state.metadata),
                  "effective_rate_inventories": _inventory_treatment(state, inventories),
                  "baseline_tariffs_separated": calib.baseline_tariffs is not None},
    )


__all__ = [
    "CondensedMeasuresResult",
    "EVSplit",
    "aggregation_gaps",
    "compute_measures",
    "constant_price_rgdp",
    "consumer_price_relative_to_wages",
    "ev_split",
    "pe_first_order_decomposition",
    "pe_unit_cost_bound",
    "real_national_income",
    "sector_incidence",
    "tariff_revenue_and_effective_rate",
    "tot_fisher",
]
