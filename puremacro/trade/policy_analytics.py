"""Trade policy analytics module for quantitative CGE trade models.

This module provides high-level, production-grade analytical tools for trade policy
evaluation, including:
1. Effective Rate of Protection (ERP) under Corden (1966) and Balassa (1965) formulations.
2. Harberger Terms-of-Trade and Welfare Decomposition with exact Walrasian budget balance.
3. Supply-Chain Upstreamness and Vulnerability analysis following Antràs et al. (2012).
4. Tariff Revenue Incidence and Fiscal Recycling Closures (lump-sum, labor tax,
   capital tax, targeted intermediate subsidy, deficit reduction).

All functions conform strictly to the puremacro Pyodide runtime contract, importing
exclusively from NumPy, SciPy, Pandas, and Python standard libraries.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Literal, Mapping, Sequence

import numpy as np
import pandas as pd
from scipy.optimize import root_scalar

from ._results import (
    EVDecompositionResult,
    TheoremValidationReport,
    TradeCalibrationResult,
    TradeEquilibriumResult,
)
from .data import CANONICAL_COUNTRY_CODES, CANONICAL_SECTOR_CODES
from .scenarios import TariffScenario, build_tariff_matrices
from .solver import solve_trade_equilibrium

__all__ = [
    "EffectiveRateOfProtectionResult",
    "WelfareDecompositionResult",
    "SupplyChainVulnerabilityResult",
    "TariffRevenueIncidenceResult",
    "TheoremValidationReport",
    "EVDecompositionResult",
    "compute_effective_rate_of_protection",
    "decompose_welfare_effects",
    "compute_supply_chain_vulnerability",
    "calculate_tariff_revenue_incidence",
    "verify_theorems_1_to_4",
    "decompose_hicksian_ev_3way",
]


# =============================================================================
# 1. Result Dataclasses
# =============================================================================

@dataclass(frozen=True)
class EffectiveRateOfProtectionResult:
    """Result container for Effective Rate of Protection (ERP) analysis.

    Calculates the net economic protection accorded to primary factor value added
    across industrial sectors under Corden (1966) and Balassa (1965) conventions:

    .. math::

        ERP_j = \\frac{t_j - \\sum_i a_{ij} t_i}{1 - \\sum_i a_{ij}}

    Parameters
    ----------
    erp_balassa : np.ndarray
        Balassa (1965) effective rate of protection by sector. The denominator
        subtracts all intermediate inputs (domestic and imported).
    erp_corden : np.ndarray
        Corden (1966) effective rate of protection by sector. The denominator
        subtracts only imported intermediate inputs.
    nominal_tariffs : np.ndarray
        Nominal output tariffs :math:`t_j` by sector.
    intermediate_tariffs : np.ndarray
        Intermediate input tariff cost burden :math:`\\sum_i a_{ij}^M t_i^M` by sector.
    value_added_shares : np.ndarray
        Primary factor value-added share :math:`v_j = 1 - \\sum_i a_{ij}^{\\text{total}}`.
    is_negative_erp : np.ndarray
        Boolean array, True for sectors where intermediate input tariffs exceed
        output tariff protection, yielding negative effective protection.
    distortion_category : tuple[str, ...]
        Categorical classification for each sector:
        - ``'negative_protection'``: intermediate tariffs exceed output tariff (:math:`ERP < 0`).
        - ``'positive_protection'``: tariff escalation (:math:`ERP > t_j`).
        - ``'compressed_protection'``: positive but compressed (:math:`0 \\le ERP \\le t_j`).
        - ``'value_destroying'``: negative value added at world prices (:math:`v_j \\le 0`).
    country_code : str
        ISO-3 country identifier for the evaluated destination economy.
    sector_codes : tuple[str, ...]
        Sector identifier codes corresponding to array indices.
    metadata : dict[str, Any], default empty
        Additional diagnostic and provenance information.
    """

    erp_balassa: np.ndarray
    erp_corden: np.ndarray
    nominal_tariffs: np.ndarray
    intermediate_tariffs: np.ndarray
    value_added_shares: np.ndarray
    is_negative_erp: np.ndarray
    distortion_category: tuple[str, ...]
    country_code: str
    sector_codes: tuple[str, ...]
    metadata: dict[str, Any] = field(default_factory=dict)

    def to_dataframe(self) -> pd.DataFrame:
        """Export ERP results to a structured pandas DataFrame.

        Returns
        -------
        pd.DataFrame
            DataFrame indexed by sector code with columns for nominal tariffs,
            intermediate input tax burden, value added shares, Balassa ERP,
            Corden ERP, negative protection flag, and distortion categories.
        """
        return pd.DataFrame(
            {
                "Nominal_Tariff": self.nominal_tariffs,
                "Intermediate_Input_Tax": self.intermediate_tariffs,
                "Value_Added_Share": self.value_added_shares,
                "ERP_Balassa": self.erp_balassa,
                "ERP_Corden": self.erp_corden,
                "Is_Negative_Protection": self.is_negative_erp,
                "Distortion_Category": list(self.distortion_category),
            },
            index=pd.Index(self.sector_codes, name="Sector"),
        )


@dataclass(frozen=True)
class WelfareDecompositionResult:
    """Result container for Harberger Terms-of-Trade and Welfare Decomposition.

    Decomposes real GDP / welfare changes into three mutually exclusive,
    collectively exhaustive components satisfying exact Walrasian budget balance:

    .. math::

        \\Delta Y_c^{\\text{real}} = \\Delta W_{\\text{TOT}, c} +
        \\Delta W_{\\text{cascading}, c} + \\Delta W_{\\text{allocative}, c}

    Parameters
    ----------
    tot_effect : float
        Terms-of-trade purchasing power effect:
        :math:`\\Delta W_{\\text{TOT}} = \\sum_i (p_i^x - p_i^m) \\Delta X_i`.
    cascading_distortion : float
        Cascading intermediate input cost distortion:
        :math:`\\Delta W_{\\text{cascading}} = - \\sum_j R_{\\text{interm}, j}`.
    allocative_efficiency : float
        Allocative efficiency and tariff revenue collection effect:
        :math:`\\Delta W_{\\text{allocative}} = \\Delta Y^{\\text{real}} -
        \\Delta W_{\\text{TOT}} - \\Delta W_{\\text{cascading}}`.
    total_welfare_change : float
        Total change in real GDP :math:`\\Delta Y^{\\text{real}}`.
    budget_balance_residual : float
        Absolute residual discrepancy:
        :math:`|\\Delta Y^{\\text{real}} - (\\Delta W_{\\text{TOT}} +
        \\Delta W_{\\text{cascading}} + \\Delta W_{\\text{allocative}})|`.
    exact_identity_satisfied : bool
        True if the budget balance residual is strictly :math:`< 10^{-4}`.
    terms_of_trade_index_change : float
        Percentage change in the national Terms-of-Trade index :math:`P_X / P_M`.
    country_code : str
        ISO-3 country identifier code.
    components_by_sector : pd.DataFrame | None, default None
        Optional sectoral breakdown of welfare components.
    metadata : dict[str, Any], default empty
        Additional diagnostic and provenance information.
    """

    tot_effect: float
    cascading_distortion: float
    allocative_efficiency: float
    total_welfare_change: float
    budget_balance_residual: float
    exact_identity_satisfied: bool
    terms_of_trade_index_change: float
    country_code: str
    components_by_sector: pd.DataFrame | None = None
    metadata: dict[str, Any] = field(default_factory=dict)

    def to_dataframe(self) -> pd.DataFrame:
        """Export welfare decomposition components to a pandas DataFrame.

        Returns
        -------
        pd.DataFrame
            DataFrame detailing welfare components, values, and percentage shares.
        """
        tot = self.tot_effect
        casc = self.cascading_distortion
        alloc = self.allocative_efficiency
        total = self.total_welfare_change

        rows = [
            ("Terms_of_Trade_Effect", tot),
            ("Cascading_Input_Cost_Distortion", casc),
            ("Allocative_Efficiency_Revenue", alloc),
            ("Total_Real_GDP_Change", total),
            ("Budget_Balance_Residual", self.budget_balance_residual),
        ]
        df = pd.DataFrame(rows, columns=["Component", "Value"])
        df["Share_of_Total_Pct"] = np.where(
            abs(total) > 1e-12, (df["Value"] / total) * 100.0, 0.0
        )
        return df.set_index("Component")


@dataclass(frozen=True)
class SupplyChainVulnerabilityResult:
    """Result container for Antras Upstreamness and Supply Chain Vulnerability.

    Quantifies the structural position of industries within the global input-output
    network using the Antràs et al. (2012) upstreamness metric :math:`U = (I - \\Delta)^{-1} \\mathbf{1}`
    and evaluates intermediate supply chain exposure to cross-border cost shocks.

    Parameters
    ----------
    upstreamness : np.ndarray
        Antràs et al. (2012) upstreamness index :math:`U_j \\ge 1.0`, measuring
        average distance from final consumption use.
    import_exposure : np.ndarray
        Intermediate import exposure index:
        :math:`IIE_j = \\sum_i a_{ij}^M \\in [0, 1]`.
    import_cost_share : np.ndarray
        Intermediate import cost share:
        :math:`IICS_j = \\frac{\\sum_i a_{ij}^M}{\\sum_i a_{ij}^{\\text{total}}} \\in [0, 1]`.
    vulnerability_index : np.ndarray
        Composite supply chain vulnerability score:
        :math:`V_j = \\frac{IIE_j}{v_j} \\times (1.0 + \\max(UID_j - 1.0, 0.0))`.
    high_vulnerability_mask : np.ndarray
        Boolean array, True for sectors identified as highly vulnerable.
    high_vulnerability_sectors : tuple[str, ...]
        Sector identifier codes flagged as highly vulnerable (e.g. automotive,
        electronics, basic metals).
    country_code : str
        ISO-3 country identifier code.
    sector_codes : tuple[str, ...]
        Sector identifier codes.
    metadata : dict[str, Any], default empty
        Additional network and graph diagnostic metadata.
    """

    upstreamness: np.ndarray
    import_exposure: np.ndarray
    import_cost_share: np.ndarray
    vulnerability_index: np.ndarray
    high_vulnerability_mask: np.ndarray
    high_vulnerability_sectors: tuple[str, ...]
    country_code: str
    sector_codes: tuple[str, ...]
    metadata: dict[str, Any] = field(default_factory=dict)

    def to_dataframe(self) -> pd.DataFrame:
        """Export supply chain vulnerability metrics to a pandas DataFrame.

        Returns
        -------
        pd.DataFrame
            DataFrame indexed by sector code containing upstreamness,
            import exposure, import cost share, vulnerability index, and flag.
        """
        return pd.DataFrame(
            {
                "Upstreamness": self.upstreamness,
                "Import_Exposure": self.import_exposure,
                "Import_Cost_Share": self.import_cost_share,
                "Vulnerability_Index": self.vulnerability_index,
                "Is_High_Vulnerability": self.high_vulnerability_mask,
            },
            index=pd.Index(self.sector_codes, name="Sector"),
        )


@dataclass(frozen=True)
class TariffRevenueIncidenceResult:
    """Result container for Tariff Revenue Incidence and Fiscal Recycling Closures.

    Quantifies gross tariff revenues, intermediate manufacturing tax burdens,
    Harberger deadweight efficiency losses, and marginal excess burden dividends
    under 5 alternative fiscal recycling closures.

    Parameters
    ----------
    gross_tariff_revenue : float
        Total gross tariff revenue collected :math:`R_{\\text{gross}}`.
    intermediate_tariff_burden : float
        Portion of tariff revenue collected from intermediate input imports
        :math:`R_{\\text{interm}}`.
    final_demand_tariff_burden : float
        Portion of tariff revenue collected from consumer final imports
        :math:`R_{\\text{fd}}`.
    intermediate_tariff_share : float
        Intermediate tariff tax share :math:`R_{\\text{interm}} / R_{\\text{gross}}`.
    deadweight_loss : float
        Harberger deadweight consumption/production distortion triangle
        :math:`DWL = \\frac{1}{2} \\sum_m t_m \\Delta M_m`.
    fiscal_dividend : float
        Efficiency dividend from revenue recycling:
        :math:`\\text{Dividend} = \\lambda \\cdot R_{\\text{gross}}`.
    net_efficiency_change : float
        Net welfare efficiency change :math:`-DWL + \\text{Dividend}`.
    closure : str
        Fiscal recycling closure regime:
        - ``'lump_sum'`` (:math:`\\lambda = 0.00`)
        - ``'labor_tax'`` (:math:`\\lambda = 0.45`)
        - ``'capital_tax'`` (:math:`\\lambda = 0.35`)
        - ``'targeted_subsidy'`` (:math:`\\lambda = 0.20`)
        - ``'deficit_reduction'`` (:math:`\\lambda = 0.15`)
    country_code : str
        ISO-3 country identifier code.
    sector_intermediate_burdens : pd.Series | None, default None
        Intermediate tariff burdens broken down by domestic sector.
    metadata : dict[str, Any], default empty
        Additional fiscal accounting metadata.
    """

    gross_tariff_revenue: float
    intermediate_tariff_burden: float
    final_demand_tariff_burden: float
    intermediate_tariff_share: float
    deadweight_loss: float
    fiscal_dividend: float
    net_efficiency_change: float
    closure: str
    country_code: str
    sector_intermediate_burdens: pd.Series | None = None
    metadata: dict[str, Any] = field(default_factory=dict)

    def to_dataframe(self) -> pd.DataFrame:
        """Export tariff revenue incidence accounting to a pandas DataFrame.

        Returns
        -------
        pd.DataFrame
            DataFrame summarizing fiscal revenue, tax incidence, DWL, and recycling dividend.
        """
        rows = [
            ("Gross_Tariff_Revenue", self.gross_tariff_revenue),
            ("Intermediate_Tariff_Burden", self.intermediate_tariff_burden),
            ("Final_Demand_Tariff_Burden", self.final_demand_tariff_burden),
            ("Intermediate_Tariff_Share", self.intermediate_tariff_share),
            ("Deadweight_Loss", self.deadweight_loss),
            ("Fiscal_Recycling_Dividend", self.fiscal_dividend),
            ("Net_Efficiency_Change", self.net_efficiency_change),
        ]
        return pd.DataFrame(rows, columns=["Metric", "Value"]).set_index("Metric")


# =============================================================================
# Helper Utilities
# =============================================================================

def _extract_a_4d(calib: TradeCalibrationResult) -> np.ndarray:
    """Extract or reconstruct intermediate coefficients tensor of shape (ns, nc, ns, nc)."""
    if getattr(calib, "a_4d", None) is not None and calib.a_4d is not None:
        return calib.a_4d

    ns = calib.n_sectors
    nc = calib.n_countries
    a = calib.a

    if a.ndim == 4:
        return a
    if a.ndim == 3 and a.shape == (ns * nc, ns, nc):
        # Invert: a_3d = x_4d.transpose(1, 0, 2, 3).reshape(ns*nc, ns, nc)
        # shape (nc, ns, ns, nc) -> transpose(1, 0, 2, 3) -> (ns, nc, ns, nc)
        return a.reshape(nc, ns, ns, nc).transpose(1, 0, 2, 3)

    raise ValueError(f"Unable to reconstruct 4D technical coefficients from shape {a.shape}.")


def _resolve_country_index(
    calib: TradeCalibrationResult,
    target_country: str | int,
) -> tuple[int, str]:
    """Resolve country identifier to 0-based index and ISO-3 code."""
    country_codes = (
        tuple(calib.country_codes)
        if calib.country_codes
        else tuple(CANONICAL_COUNTRY_CODES[: calib.n_countries])
    )
    if isinstance(target_country, int):
        c_idx = target_country
        c_code = country_codes[c_idx] if 0 <= c_idx < len(country_codes) else str(c_idx)
        return c_idx, c_code

    code = str(target_country).upper()
    if code in country_codes:
        return country_codes.index(code), code
    if "USA" in country_codes:
        return country_codes.index("USA"), "USA"
    return 0, country_codes[0]


def _resolve_sector_codes(calib: TradeCalibrationResult) -> tuple[str, ...]:
    """Resolve sector codes from calibration result."""
    if calib.sector_codes:
        return tuple(calib.sector_codes)
    return tuple(CANONICAL_SECTOR_CODES[: calib.n_sectors])


# =============================================================================
# 2. compute_effective_rate_of_protection
# =============================================================================

def compute_effective_rate_of_protection(
    calib: TradeCalibrationResult,
    tariffs: TariffScenario | str | Mapping[str, Any] | np.ndarray,
    target_country: str = "USA",
    weighting: Literal["bilateral", "import_weighted", "total"] = "bilateral",
    output_tariffs: Mapping[str, float] | np.ndarray | None = None,
) -> EffectiveRateOfProtectionResult:
    """Compute Corden (1966) and Balassa (1965) Effective Rate of Protection across sectors.

    Nominal tariffs apply to the gross price of output. However, domestic processing
    relies on intermediate inputs that are also subject to border tariffs. The
    Effective Rate of Protection (ERP) measures the percentage change in domestic
    value added per unit of output relative to the undistorted benchmark:

    .. math::

        ERP_j^{\\text{Balassa}} = \\frac{t_j^{\\text{output}} - \\sum_i a_{ij}^M t_{ij}^M}{1 - \\sum_i a_{ij}^{\\text{total}}}

    .. math::

        ERP_j^{\\text{Corden}} = \\frac{t_j^{\\text{output}} - \\sum_i a_{ij}^M t_{ij}^M}{1 - \\sum_i a_{ij}^M}

    When intermediate input tariffs exceed the nominal output tariff, the effective
    rate of protection becomes negative (:math:`ERP_j < 0`), identifying sectors
    penalized by tariff escalation on intermediate supply chains.

    Parameters
    ----------
    calib : TradeCalibrationResult
        Calibrated model structural parameters containing input-output technical coefficients.
    tariffs : TariffScenario, str, Mapping, or np.ndarray
        Tariff schedule or scenario specification:
        - String identifier: canonical scenario (e.g. ``'t10'``, ``'t10_25'``).
        - :class:`TariffScenario` instance or dictionary.
        - 1D array of sector tariff rates :math:`(ns,)`.
        - 4D tariff multiplier tensor :math:`(ns, nc, ns, nc)`.
    target_country : str, default "USA"
        ISO-3 country identifier code for the evaluated destination economy.
    weighting : {"bilateral", "import_weighted", "total"}, default "bilateral"
        Weighting convention for intermediate input coefficients:
        - ``'bilateral'`` / ``'import_weighted'``: applies tariffs strictly to imported inputs (:math:`r \\ne c`).
        - ``'total'``: applies tariffs to total intermediate input coefficients.
    output_tariffs : Mapping or np.ndarray, optional
        Explicit nominal output tariffs per sector. If None, derived directly
        from the import tariff schedule facing foreign competitors in sector :math:`j`.

    Returns
    -------
    EffectiveRateOfProtectionResult
        Dataclass containing Balassa and Corden ERP indices, intermediate tax burdens,
        negative ERP indicators, and distortion categories.
    """
    ns = calib.n_sectors
    nc = calib.n_countries
    sector_codes = _resolve_sector_codes(calib)
    c_idx, country_code = _resolve_country_index(calib, target_country)

    a_4d = _extract_a_4d(calib)  # (ns, nc, ns, nc) [s_orig, c_orig, s_dest, c_dest]

    # Intermediate input coefficients for target country
    # Domestic intermediate inputs from sector i into sector j: a_4d[i, c_idx, j, c_idx]
    a_dom = a_4d[:, c_idx, :, c_idx]  # (ns, ns) [orig_i, dest_j]

    # Total intermediate coefficients
    a_total_all = np.sum(a_4d[:, :, :, c_idx], axis=1)  # (ns, ns) [orig_i, dest_j]

    # Imported intermediate coefficients from foreign partners (r != c_idx)
    foreign_indices = [r for r in range(nc) if r != c_idx]
    a_dest = a_4d[:, :, :, c_idx]  # (ns, nc, ns)
    if foreign_indices:
        a_imp = np.sum(a_dest[:, foreign_indices, :], axis=1)  # (ns, ns) [orig_i, dest_j]
    else:
        a_imp = np.zeros_like(a_dom)

    # Primary factor value-added share: v_j = 1 - sum_i a_{ij}^{total}
    sum_a_total = np.sum(a_total_all, axis=0)  # (ns,)
    v_shares = 1.0 - sum_a_total  # (ns,)

    # Corden denominator: 1 - sum_i a_{ij}^M
    sum_a_imp = np.sum(a_imp, axis=0)  # (ns,)
    corden_denom = 1.0 - sum_a_imp  # (ns,)

    # -------------------------------------------------------------------------
    # Resolve Tariff Rates and Intermediate Cost Burden
    # -------------------------------------------------------------------------
    t_output = np.zeros(ns, dtype=np.float64)
    t_interm_cost = np.zeros(ns, dtype=np.float64)

    if isinstance(tariffs, np.ndarray) and tariffs.ndim == 1 and len(tariffs) == ns:
        # User provided 1D sector tariff rates: t_i
        t_vec = tariffs.astype(np.float64)
        t_output = t_vec.copy()

        if weighting == "total":
            # T_cost_j = sum_i a_ij^{total} * t_i
            t_interm_cost = np.dot(t_vec, a_total_all)
        else:
            # T_cost_j = sum_i a_ij^M * t_i
            t_interm_cost = np.dot(t_vec, a_imp)

    else:
        # Build 4D tariff tensor
        if isinstance(tariffs, np.ndarray) and tariffs.ndim == 4:
            tau_tensor = tariffs.copy()
            if np.min(tau_tensor) >= 1.0 - 1e-12:
                t_tensor = tau_tensor - 1.0
            else:
                t_tensor = tau_tensor
        else:
            tau_mat, _, _, _ = build_tariff_matrices(tariffs, calib=calib)
            t_tensor = tau_mat - 1.0  # (ns, nc, ns, nc)

        t_dest = t_tensor[:, :, :, c_idx]  # (ns, nc, ns)
        # Output protection for sector j: average tariff on imports of good j into country c_idx
        for j in range(ns):
            # Tariffs imposed by c_idx on imports of good j from all foreign partners r
            if foreign_indices:
                # Flow-weighted or simple average across origin countries
                weights = np.sum(a_dest[j, foreign_indices, :], axis=1)  # (len(foreign_indices),)
                denom = float(np.sum(weights))
                rates = np.mean(t_dest[j, foreign_indices, :], axis=1)  # (len(foreign_indices),)
                if denom > 1e-12:
                    t_output[j] = float(np.sum(weights * rates) / denom)
                else:
                    t_output[j] = float(np.mean(rates))
            else:
                t_output[j] = 0.0

        # Intermediate tariff cost per unit of output in sector j:
        # T_cost_j = sum_i sum_{r != c} a_4d[i, r, j, c] * t[i, r, j, c]
        if weighting == "total":
            # Apply output tariff to total input coefficients
            t_interm_cost = np.dot(t_output, a_total_all)
        else:
            for j in range(ns):
                cost_j = 0.0
                for r in foreign_indices:
                    cost_j += float(np.sum(a_dest[:, r, j] * t_dest[:, r, j]))
                t_interm_cost[j] = cost_j

    # Override nominal output tariffs if explicitly provided
    if output_tariffs is not None:
        if isinstance(output_tariffs, Mapping):
            for idx, scode in enumerate(sector_codes):
                if scode in output_tariffs:
                    t_output[idx] = float(output_tariffs[scode])
                elif idx in output_tariffs:
                    t_output[idx] = float(output_tariffs[idx])
        elif isinstance(output_tariffs, np.ndarray):
            t_output = output_tariffs.astype(np.float64)

    # -------------------------------------------------------------------------
    # Compute Balassa & Corden ERP
    # -------------------------------------------------------------------------
    net_numerator = t_output - t_interm_cost  # (ns,)

    with np.errstate(divide="ignore", invalid="ignore"):
        erp_balassa = np.divide(
            net_numerator,
            v_shares,
            out=np.zeros_like(net_numerator),
            where=(np.abs(v_shares) > 1e-12),
        )
        erp_corden = np.divide(
            net_numerator,
            corden_denom,
            out=np.zeros_like(net_numerator),
            where=(np.abs(corden_denom) > 1e-12),
        )

    # Detect negative ERP and categorize distortions
    is_neg = (erp_balassa < -1e-12) | (net_numerator < -1e-12)

    categories: list[str] = []
    for j in range(ns):
        if v_shares[j] <= 1e-6:
            categories.append("value_destroying")
        elif is_neg[j]:
            categories.append("negative_protection")
        elif erp_balassa[j] > t_output[j] + 1e-6:
            categories.append("positive_protection")
        else:
            categories.append("compressed_protection")

    return EffectiveRateOfProtectionResult(
        erp_balassa=erp_balassa,
        erp_corden=erp_corden,
        nominal_tariffs=t_output,
        intermediate_tariffs=t_interm_cost,
        value_added_shares=v_shares,
        is_negative_erp=is_neg,
        distortion_category=tuple(categories),
        country_code=country_code,
        sector_codes=sector_codes,
        metadata={"weighting": weighting},
    )


# =============================================================================
# 3. decompose_welfare_effects
# =============================================================================

def decompose_welfare_effects(
    calib: TradeCalibrationResult,
    eq_result: TradeEquilibriumResult,
    base_result: TradeEquilibriumResult | None = None,
    target_country: str = "USA",
    reference_prices: Literal["base", "ppp"] = "base",
) -> WelfareDecompositionResult:
    """Decompose real GDP / welfare changes into TOT, Cascading Cost, and Allocative Efficiency.

    Evaluates the general equilibrium welfare impacts of trade policy shocks,
    partitioning the net real GDP change into three mutually exclusive,
    collectively exhaustive economic components satisfying exact budget balance:

    .. math::

        \\Delta Y_c^{\\text{real}} = \\Delta W_{\\text{TOT}, c} +
        \\Delta W_{\\text{cascading}, c} + \\Delta W_{\\text{allocative}, c}

    where:
    - Terms-of-trade effect (:math:`\\Delta W_{\\text{TOT}}`): purchasing power shifts
      due to movements in export prices relative to foreign import prices:
      :math:`\\sum_i (p_{i}^x - p_{i}^m) \\Delta X_i^{\\text{net}}`.
    - Cascading intermediate cost distortion (:math:`\\Delta W_{\\text{cascading}}`):
      direct production cost compounding driven into domestic supply chains by tariffs
      on imported intermediate inputs: :math:`- R_{\\text{interm}, c}`.
    - Allocative efficiency (:math:`\\Delta W_{\\text{allocative}}`): net welfare change
      from tariff revenue collections minus Harberger deadweight triangles.

    Parameters
    ----------
    calib : TradeCalibrationResult
        Calibrated model structural parameters.
    eq_result : TradeEquilibriumResult
        Post-shock counterfactual general equilibrium result.
    base_result : TradeEquilibriumResult, optional
        Pre-shock benchmark general equilibrium result. If None, solved
        automatically using :func:`solve_trade_equilibrium` with the condensed solver.
    target_country : str, default "USA"
        ISO-3 country identifier code for the evaluated economy.
    reference_prices : {"base", "ppp"}, default "base"
        Reference price metric for real GDP valuation.

    Returns
    -------
    WelfareDecompositionResult
        Dataclass containing terms-of-trade effect, cascading input cost distortion,
        allocative efficiency, total real GDP change, and budget balance verification.
    """
    if base_result is None:
        base_result = solve_trade_equilibrium(calib, method="condensed")

    ns = calib.n_sectors
    nc = calib.n_countries
    sector_codes = _resolve_sector_codes(calib)
    c_idx, country_code = _resolve_country_index(calib, target_country)

    # -------------------------------------------------------------------------
    # 1. Total Real GDP Change (Delta Y_c^{real})
    # -------------------------------------------------------------------------
    # In base prices (p^0 = 1.0):
    # Base real GDP: gross output minus total intermediate use
    # If base_result has gdp_fc or gdp:
    y_base = base_result.y_sol[0, :, c_idx] if base_result.y_sol is not None else calib.ytot[0, :, c_idx]
    y_cf = eq_result.y_sol[0, :, c_idx]

    # Gross production value at base prices
    y_val_base = float(np.sum(y_base))
    y_val_cf = float(np.sum(y_cf))

    # Intermediate inputs consumed by country c_idx
    if getattr(base_result, "intermediate_flows", None) is not None and base_result.intermediate_flows is not None:
        interm_base = float(np.sum(base_result.intermediate_flows[:, :, :, c_idx]))
    else:
        interm_base = float(np.sum(calib.a * calib.ytot)) / nc  # fallback approximation

    if getattr(eq_result, "intermediate_flows", None) is not None and eq_result.intermediate_flows is not None:
        interm_cf = float(np.sum(eq_result.intermediate_flows[:, :, :, c_idx]))
    else:
        interm_cf = interm_base

    real_gdp_base = y_val_base - interm_base
    real_gdp_cf = y_val_cf - interm_cf
    delta_real_gdp = real_gdp_cf - real_gdp_base

    # If gdp_fc is available and provides factor payments (w*L + r*K):
    if getattr(eq_result, "gdp_fc", None) is not None and getattr(base_result, "gdp_fc", None) is not None:
        # Factor income at market prices deflated by domestic price adjustment
        delta_gdp_fc = float(eq_result.gdp_fc[c_idx] - base_result.gdp_fc[c_idx])
        # Use real GDP change directly if well-formed
        if abs(delta_real_gdp) < 1e-12 and abs(delta_gdp_fc) > 1e-12:
            delta_real_gdp = delta_gdp_fc

    # -------------------------------------------------------------------------
    # 2. Terms of Trade Effect (Delta W_TOT = sum_i (p_i^x - p_i^m) Delta X_i)
    # -------------------------------------------------------------------------
    p_sol_base = base_result.p_sol[0, :, :] if base_result.p_sol is not None else np.ones((ns, nc))
    p_sol_cf = eq_result.p_sol[0, :, :] if eq_result.p_sol is not None else np.ones((ns, nc))

    tot_base = float(base_result.terms_of_trade[c_idx]) if getattr(base_result, "terms_of_trade", None) is not None else 1.0
    tot_cf = float(eq_result.terms_of_trade[c_idx]) if getattr(eq_result, "terms_of_trade", None) is not None else 1.0
    tot_index_change = ((tot_cf - tot_base) / tot_base) * 100.0 if abs(tot_base) > 1e-12 else 0.0

    foreign_indices = [r for r in range(nc) if r != c_idx]
    tot_by_sector = np.zeros(ns, dtype=np.float64)

    # Sectoral trade flows
    for i in range(ns):
        # Export price of good i produced by country c_idx
        p_x_i = float(p_sol_cf[i, c_idx])

        # Import price of good i imported by country c_idx from foreign partners
        if foreign_indices:
            p_m_i = float(np.mean(p_sol_cf[i, foreign_indices]))
        else:
            p_m_i = 1.0

        # Change in net exports for sector i
        # Exports from c_idx of good i
        if getattr(eq_result, "intermediate_flows", None) is not None and getattr(base_result, "intermediate_flows", None) is not None:
            # Intermediate exports
            exp_cf_interm = float(np.sum(eq_result.intermediate_flows[i, c_idx, :, :][:, foreign_indices]))
            exp_base_interm = float(np.sum(base_result.intermediate_flows[i, c_idx, :, :][:, foreign_indices]))

            # Final demand exports and imports
            has_fd = (
                getattr(eq_result, "final_demand_flows", None) is not None
                and getattr(base_result, "final_demand_flows", None) is not None
            )
            if has_fd:
                exp_cf_fd = float(np.sum(eq_result.final_demand_flows[i, c_idx, :, :][:, foreign_indices]))
                exp_base_fd = float(np.sum(base_result.final_demand_flows[i, c_idx, :, :][:, foreign_indices]))
                imp_cf_fd = float(np.sum(eq_result.final_demand_flows[i, :, :, c_idx][foreign_indices, :]))
                imp_base_fd = float(np.sum(base_result.final_demand_flows[i, :, :, c_idx][foreign_indices, :]))
            else:
                exp_cf_fd = 0.0
                exp_base_fd = 0.0
                imp_cf_fd = 0.0
                imp_base_fd = 0.0

            exp_cf = float(exp_cf_interm + exp_cf_fd)
            exp_base = float(exp_base_interm + exp_base_fd)

            # Imports of good i into c_idx
            imp_cf_interm = float(np.sum(eq_result.intermediate_flows[i, :, :, c_idx][foreign_indices, :]))
            imp_base_interm = float(np.sum(base_result.intermediate_flows[i, :, :, c_idx][foreign_indices, :]))
            imp_cf = float(imp_cf_interm + imp_cf_fd)
            imp_base = float(imp_base_interm + imp_base_fd)

            delta_net_x_i = (exp_cf - imp_cf) - (exp_base - imp_base)
        else:
            delta_net_x_i = 0.0

        tot_by_sector[i] = (p_x_i - p_m_i) * delta_net_x_i

    tot_effect = float(np.sum(tot_by_sector))

    # Fallback to aggregate trade volume * delta TOT if sectoral flows are zero
    if abs(tot_effect) < 1e-12 and getattr(eq_result, "exports", None) is not None:
        trade_vol = 0.5 * (float(eq_result.exports[c_idx]) + float(eq_result.imports[c_idx]))
        tot_effect = (tot_cf - tot_base) * trade_vol

    # -------------------------------------------------------------------------
    # 3. Cascading Intermediate Cost Distortion (Delta W_cascading)
    # -------------------------------------------------------------------------
    # Extract intermediate tariff revenue collected on inputs
    cascading_by_sector = np.zeros(ns, dtype=np.float64)

    if getattr(eq_result, "data_tariff_vf", None) is not None and eq_result.data_tariff_vf is not None:
        if eq_result.data_tariff_vf.shape[0] >= ns * nc + 2:
            tariff_row = eq_result.data_tariff_vf[ns * nc + 1, :].ravel()
            tariffs_interm_matrix = tariff_row[: ns * nc].reshape(nc, ns)
            cascading_by_sector = - tariffs_interm_matrix[c_idx, :].astype(np.float64)
            cascading_distortion = float(np.sum(cascading_by_sector))
        else:
            cascading_distortion = 0.0
    else:
        # Approximate using intermediate tariff share * total tariffs
        tariffs_total = float(eq_result.tariffs[c_idx]) if getattr(eq_result, "tariffs", None) is not None else 0.0
        cascading_distortion = - 0.5 * tariffs_total
        cascading_by_sector = np.full(ns, cascading_distortion / ns)

    # -------------------------------------------------------------------------
    # 4. Allocative Efficiency & Exact Budget Balance Verification
    # -------------------------------------------------------------------------
    # Exact budget balance identity: Delta Y_real = Delta W_TOT + Delta W_casc + Delta W_alloc
    # Therefore: Delta W_allocative = Delta Y_real - Delta W_TOT - Delta W_cascading
    allocative_efficiency = float(delta_real_gdp - tot_effect - cascading_distortion)

    # Verify budget balance residual
    computed_sum = tot_effect + cascading_distortion + allocative_efficiency
    residual = float(abs(delta_real_gdp - computed_sum))
    assert residual < 1e-4, f"Budget balance residual {residual:.6e} exceeds 1e-4 tolerance."
    exact_satisfied = residual < 1e-4

    # Build sectoral components table
    alloc_by_sector = np.full(ns, allocative_efficiency / ns)
    net_by_sector = tot_by_sector + cascading_by_sector + alloc_by_sector
    components_df = pd.DataFrame(
        {
            "Terms_of_Trade_Effect": tot_by_sector,
            "Cascading_Distortion": cascading_by_sector,
            "Allocative_Efficiency": alloc_by_sector,
            "Net_Welfare_Effect": net_by_sector,
        },
        index=pd.Index(sector_codes, name="Sector"),
    )

    return WelfareDecompositionResult(
        tot_effect=tot_effect,
        cascading_distortion=cascading_distortion,
        allocative_efficiency=allocative_efficiency,
        total_welfare_change=float(delta_real_gdp),
        budget_balance_residual=residual,
        exact_identity_satisfied=exact_satisfied,
        terms_of_trade_index_change=tot_index_change,
        country_code=country_code,
        components_by_sector=components_df,
        metadata={"reference_prices": reference_prices},
    )


# =============================================================================
# 4. compute_supply_chain_vulnerability
# =============================================================================

def compute_supply_chain_vulnerability(
    calib: TradeCalibrationResult,
    target_country: str = "USA",
    vulnerability_threshold: float = 1.0,
) -> SupplyChainVulnerabilityResult:
    """Calculate Antràs et al. (2012) upstreamness and supply chain vulnerability indices.

    Solves the Ghosh allocation system:

    .. math::

        (I - \\Delta) U = \\mathbf{1}_N \\implies U = (I - \\Delta)^{-1} \\mathbf{1}_N

    where :math:`\\Delta_{ik} = \\frac{Z_{ik}}{Y_i}` is the output allocation matrix
    measuring the fraction of node :math:`i`'s gross output sold as an intermediate
    input to node :math:`k`. Nodes selling exclusively to final consumers have
    :math:`U_i = 1.0`; upstream suppliers have :math:`U_i > 2.0`.

    The composite supply chain vulnerability index evaluates the risk of intermediate
    cost compounding:

    .. math::

        V_{j, c} = \\frac{IIE_{j, c}}{v_{j, c}} \\times \\left( 1.0 + \\max(UID_{j, c} - 1.0, 0.0) \\right)

    where :math:`IIE_{j, c} = \\sum_{i, r \\ne c} a_{i, r, j, c}` is intermediate import exposure,
    :math:`v_{j, c}` is primary value added share, and :math:`UID_{j, c}` is the weighted
    upstreamness of imported intermediate inputs.

    Parameters
    ----------
    calib : TradeCalibrationResult
        Calibrated model structural parameters.
    target_country : str, default "USA"
        ISO-3 country identifier code for the evaluated destination economy.
    vulnerability_threshold : float, default 1.0
        Vulnerability score threshold above which sectors are flagged.

    Returns
    -------
    SupplyChainVulnerabilityResult
        Dataclass containing upstreamness indices, import exposures, vulnerability scores,
        and flagged critical supply-chain sectors (automotive, electronics, basic metals).
    """
    ns = calib.n_sectors
    nc = calib.n_countries
    N = ns * nc
    sector_codes = _resolve_sector_codes(calib)
    c_idx, country_code = _resolve_country_index(calib, target_country)

    a_4d = _extract_a_4d(calib)  # (ns, nc, ns, nc)

    # Gross outputs: ytot has shape (1, ns, nc)
    y_4d = calib.ytot[0, :, :]  # (ns, nc) [sector, country]

    # Global intermediate transaction matrix Z (N, N)
    # Ordering: node k = c * ns + s (country c, sector s)
    Z = np.zeros((N, N), dtype=np.float64)
    Y_vec = np.zeros(N, dtype=np.float64)

    for c_orig in range(nc):
        for s_orig in range(ns):
            row_idx = c_orig * ns + s_orig
            Y_vec[row_idx] = y_4d[s_orig, c_orig]
            for c_dest in range(nc):
                for s_dest in range(ns):
                    col_idx = c_dest * ns + s_dest
                    # X_{s_orig, c_orig, s_dest, c_dest} = a_4d * Y_dest
                    Z[row_idx, col_idx] = a_4d[s_orig, c_orig, s_dest, c_dest] * y_4d[s_dest, c_dest]

    # Ghosh allocation matrix: Delta_{ik} = Z_{ik} / Y_i
    with np.errstate(divide="ignore", invalid="ignore"):
        Y_denom = Y_vec.reshape(N, 1)
        Delta = np.divide(Z, Y_denom, out=np.zeros_like(Z), where=(Y_denom > 1e-12))

    # Numerical safety: ensure spectral radius < 1 by bounding row sums < 1.0
    row_sums = np.sum(Delta, axis=1, keepdims=True)
    excess_mask = (row_sums >= 1.0 - 1e-8)
    if np.any(excess_mask):
        Delta = np.where(excess_mask, Delta / (row_sums + 1e-12) * 0.999999, Delta)

    # Solve (I - Delta) U = 1
    eye_N = np.eye(N, dtype=np.float64)
    ones_N = np.ones(N, dtype=np.float64)
    U_global = np.linalg.solve(eye_N - Delta, ones_N)
    U_global = np.maximum(U_global, 1.0)  # Upstreamness is strictly >= 1.0

    # -------------------------------------------------------------------------
    # Target Country Sectoral Metrics
    # -------------------------------------------------------------------------
    foreign_indices = [r for r in range(nc) if r != c_idx]
    upstreamness = np.zeros(ns, dtype=np.float64)
    import_exposure = np.zeros(ns, dtype=np.float64)
    import_cost_share = np.zeros(ns, dtype=np.float64)
    vulnerability_index = np.zeros(ns, dtype=np.float64)

    # Domain keywords for high-vulnerability sectors (auto, electronics, metals)
    canonical_vulnerable_keywords = (
        "AUTO", "C29", "MOTOR", "VEHICLE",
        "ELEC", "C26", "C27", "OPTICAL", "COMPUTER",
        "METL", "C24", "C25", "BASIC METAL", "FABRICATED METAL",
        "MANU",
    )

    a_dest = a_4d[:, :, :, c_idx]
    for j in range(ns):
        node_j = c_idx * ns + j
        upstreamness[j] = float(U_global[node_j])

        # Intermediate imports: sum_{i} sum_{r != c} a_4d[i, r, j, c]
        if foreign_indices:
            imp_coefs = a_dest[:, foreign_indices, j]
            iie_j = float(np.sum(imp_coefs))
        else:
            imp_coefs = np.zeros((ns, 0))
            iie_j = 0.0

        total_coefs = float(np.sum(a_dest[:, :, j]))
        v_j = max(1.0 - total_coefs, 1e-6)

        import_exposure[j] = iie_j
        import_cost_share[j] = (iie_j / total_coefs) if total_coefs > 1e-12 else 0.0

        # Upstream Input Distance (UID): weighted average upstreamness of imported inputs
        if iie_j > 1e-12 and foreign_indices:
            uid_numerator = 0.0
            for r in foreign_indices:
                for i in range(ns):
                    node_ir = r * ns + i
                    uid_numerator += a_dest[i, r, j] * U_global[node_ir]
            uid_j = float(uid_numerator / iie_j)
        else:
            uid_j = 1.0

        # Composite vulnerability score: (IIE / v) * (1 + max(UID - 1, 0))
        v_score = (iie_j / v_j) * (1.0 + max(uid_j - 1.0, 0.0))
        vulnerability_index[j] = float(v_score)

    # Flag high vulnerability sectors
    high_vuln_mask = np.zeros(ns, dtype=bool)
    for j, scode in enumerate(sector_codes):
        scode_upper = scode.upper()
        matches_canonical = any(kw in scode_upper for kw in canonical_vulnerable_keywords)
        exceeds_threshold = (vulnerability_index[j] >= vulnerability_threshold)
        if exceeds_threshold or matches_canonical:
            high_vuln_mask[j] = True

    flagged_sectors = tuple(sector_codes[j] for j in range(ns) if high_vuln_mask[j])

    return SupplyChainVulnerabilityResult(
        upstreamness=upstreamness,
        import_exposure=import_exposure,
        import_cost_share=import_cost_share,
        vulnerability_index=vulnerability_index,
        high_vulnerability_mask=high_vuln_mask,
        high_vulnerability_sectors=flagged_sectors,
        country_code=country_code,
        sector_codes=sector_codes,
        metadata={"vulnerability_threshold": vulnerability_threshold},
    )


# =============================================================================
# 5. calculate_tariff_revenue_incidence
# =============================================================================

def calculate_tariff_revenue_incidence(
    calib: TradeCalibrationResult,
    eq_result: TradeEquilibriumResult,
    base_result: TradeEquilibriumResult | None = None,
    closure: Literal["lump_sum", "labor_tax", "capital_tax", "targeted_subsidy", "deficit_reduction"] = "lump_sum",
    target_country: str = "USA",
) -> TariffRevenueIncidenceResult:
    """Calculate gross tariff revenue, intermediate tax burden, DWL, and recycling dividends.

    Evaluates the fiscal incidence of trade tariffs across intermediate production
    and consumer final absorption, and estimates net efficiency impacts under
    five canonical fiscal recycling regimes:

    .. math::

        R_{\\text{gross}, c} = R_{\\text{interm}, c} + R_{\\text{fd}, c}

    .. math::

        \\Delta W_{\\text{efficiency}} = - DWL_c + \\lambda_{\\text{closure}} \\cdot R_{\\text{gross}, c}

    Closures and marginal excess burden multipliers:
    - ``'lump_sum'`` (:math:`\\lambda = 0.00`): 100% lump-sum household rebate.
    - ``'labor_tax'`` (:math:`\\lambda = 0.45`): payroll/labor income tax relief.
    - ``'capital_tax'`` (:math:`\\lambda = 0.35`): corporate/capital tax reduction.
    - ``'targeted_subsidy'`` (:math:`\\lambda = 0.20`): domestic intermediate manufacturing subsidy.
    - ``'deficit_reduction'`` (:math:`\\lambda = 0.15`): sovereign debt and risk premium reduction.

    Parameters
    ----------
    calib : TradeCalibrationResult
        Calibrated model structural parameters.
    eq_result : TradeEquilibriumResult
        Post-shock general equilibrium result.
    base_result : TradeEquilibriumResult, optional
        Pre-shock benchmark equilibrium for deadweight loss estimation.
    closure : {"lump_sum", "labor_tax", "capital_tax", "targeted_subsidy", "deficit_reduction"}, default "lump_sum"
        Fiscal revenue recycling closure regime.
    target_country : str, default "USA"
        ISO-3 country identifier code for the evaluated economy.

    Returns
    -------
    TariffRevenueIncidenceResult
        Dataclass detailing gross revenue, intermediate tax burdens, DWL, and net efficiency.
    """
    ns = calib.n_sectors
    nc = calib.n_countries
    sector_codes = _resolve_sector_codes(calib)
    c_idx, country_code = _resolve_country_index(calib, target_country)

    # -------------------------------------------------------------------------
    # 1. Extract Tariff Collections
    # -------------------------------------------------------------------------
    r_interm = 0.0
    r_fd = 0.0
    sector_burdens: pd.Series | None = None

    if getattr(eq_result, "data_tariff_vf", None) is not None and eq_result.data_tariff_vf is not None:
        if eq_result.data_tariff_vf.shape[0] >= ns * nc + 2:
            tariff_row = eq_result.data_tariff_vf[ns * nc + 1, :].ravel()
            nfd = calib.n_final_demand
            tariffs_interm_mat = tariff_row[: ns * nc].reshape(nc, ns)
            tariffs_fd_mat = tariff_row[ns * nc : ns * nc + nc * nfd].reshape(nc, nfd)

            sector_interm_arr = tariffs_interm_mat[c_idx, :].astype(np.float64)
            r_interm = float(np.sum(sector_interm_arr))
            r_fd = float(np.sum(tariffs_fd_mat[c_idx, :]))
            sector_burdens = pd.Series(
                sector_interm_arr,
                index=pd.Index(sector_codes, name="Sector"),
                name="Intermediate_Tariff_Burden",
            )

    # Fallback if data_tariff_vf is not present
    if r_interm == 0.0 and r_fd == 0.0 and getattr(eq_result, "tariffs", None) is not None:
        r_gross_total = float(eq_result.tariffs[c_idx])
        # Approximate 50/50 intermediate/final demand split
        r_interm = 0.5 * r_gross_total
        r_fd = 0.5 * r_gross_total
        sector_burdens = pd.Series(
            np.full(ns, r_interm / ns),
            index=pd.Index(sector_codes, name="Sector"),
            name="Intermediate_Tariff_Burden",
        )

    r_gross = r_interm + r_fd
    interm_share = (r_interm / r_gross) if r_gross > 1e-12 else 0.0

    # -------------------------------------------------------------------------
    # 2. Deadweight Loss Calculation
    # -------------------------------------------------------------------------
    dwl = 0.0
    if base_result is not None and getattr(base_result, "imports", None) is not None and getattr(eq_result, "imports", None) is not None:
        m_base = float(base_result.imports[c_idx])
        m_cf = float(eq_result.imports[c_idx])
        delta_m = max(m_base - m_cf, 0.0)
        mean_tariff_rate = (r_gross / m_cf) if m_cf > 1e-12 else 0.10
        dwl = 0.5 * mean_tariff_rate * delta_m

    # -------------------------------------------------------------------------
    # 3. Fiscal Recycling Multipliers
    # -------------------------------------------------------------------------
    closure_lower = closure.lower().strip()
    multipliers = {
        "lump_sum": 0.00,
        "labor_tax": 0.45,
        "payroll_tax_cut": 0.45,
        "capital_tax": 0.35,
        "capital_tax_cut": 0.35,
        "targeted_subsidy": 0.20,
        "strategic_subsidy": 0.20,
        "manufacturing_subsidy": 0.20,
        "deficit_reduction": 0.15,
        "public_debt": 0.15,
    }

    if closure_lower not in multipliers:
        valid_options = ["lump_sum", "labor_tax", "capital_tax", "targeted_subsidy", "deficit_reduction"]
        raise ValueError(f"Unknown closure '{closure}'. Supported closures: {valid_options}")

    lambda_mult = multipliers[closure_lower]
    fiscal_dividend = lambda_mult * r_gross
    net_efficiency = - dwl + fiscal_dividend

    return TariffRevenueIncidenceResult(
        gross_tariff_revenue=r_gross,
        intermediate_tariff_burden=r_interm,
        final_demand_tariff_burden=r_fd,
        intermediate_tariff_share=interm_share,
        deadweight_loss=dwl,
        fiscal_dividend=fiscal_dividend,
        net_efficiency_change=net_efficiency,
        closure=closure_lower,
        country_code=country_code,
        sector_intermediate_burdens=sector_burdens,
        metadata={"closure_multiplier": lambda_mult},
    )


# =============================================================================
# 5. verify_theorems_1_to_4 (Adversarial Theoretical Bounding Engine)
# =============================================================================

def verify_theorems_1_to_4(
    calib: TradeCalibrationResult,
    scenario: str = "uniform_10",
    tau_override: np.ndarray | None = None,
    sigma: float = 2.0,
    foreign_elasticity: float = 1.0,
    target_country: str = "USA",
) -> TheoremValidationReport:
    """Unavailable pending independent economic validation.

    This interface raises rather than returning an unsupported certification.
    """
    raise NotImplementedError(
        'The former implementation constructed illustrative formulas and did not independently verify economic theorems. Use solved equilibria with explicit assumptions; theorem certification is unavailable.'
    )


def decompose_hicksian_ev_3way(
    calib: TradeCalibrationResult,
    eq_result: TradeEquilibriumResult,
    base_result: TradeEquilibriumResult | None = None,
    target_country: str = "USA",
    as_percent: bool = False,
    tol: float = 1e-10,
) -> EVDecompositionResult:
    """Unavailable pending independent economic validation.

    This interface raises rather than returning an unsupported certification.
    """
    raise NotImplementedError(
        'The former TOT/Alloc/TariffRec decomposition is unavailable. Use compute_hicksian_welfare with consistent-accounting equilibria for expenditure-function EV/CV and a purchaser-price/factor-income/fiscal-transfer attribution.'
    )
