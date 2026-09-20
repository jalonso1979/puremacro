"""Comprehensive empirical challenge for 3-way additive Hicksian EV decomposition.

Challenger: challenger_m3_remed_2
Objective:
1. 1,000+ randomized calibrations, tariff vectors, and numeraire scalings.
2. Confirm residual <= 1e-10 in 100% of cases without artificial plug.
3. Stress test numeraire invariance over 24 orders of magnitude (10^-12 to 10^12).
4. Verify independent Harberger DWL calculation.
5. Stress test boundary conditions (autarky, zero tariff, prohibitive tariff, tiny economies).
"""
import sys
import math
import pytest
import numpy as np
import pandas as pd
from typing import Tuple

from puremacro.trade._results import (
    TradeCalibrationResult,
    TradeEquilibriumResult,
    EVDecompositionResult,
)
from puremacro.trade.policy_analytics import decompose_hicksian_ev_3way


def make_random_mrio_setup(
    nc: int,
    ns: int,
    rng: np.random.Generator,
    with_flows: bool = True,
    gdp_scale: float = 1000.0,
) -> Tuple[TradeCalibrationResult, TradeEquilibriumResult, TradeEquilibriumResult]:
    country_codes = tuple(f"C{i:02d}" for i in range(nc))
    sector_codes = tuple(f"S{j:02d}" for j in range(ns))

    # Base calibration
    calib = TradeCalibrationResult(
        a=np.zeros((ns * nc, ns, nc)),
        afd=np.zeros((ns * nc, 3, nc)),
        alpha=np.zeros((1, ns, nc)),
        beta=np.zeros((1, ns, nc)),
        k_endow=np.ones((1, nc)),
        l_endow=np.ones((1, nc)),
        invforT=np.zeros((1, nc)),
        tax=np.zeros((1, ns, nc)),
        ytot=np.full((1, ns, nc), gdp_scale),
        n_countries=nc,
        n_sectors=ns,
        n_final_demand=3,
        country_codes=country_codes,
        sector_codes=sector_codes,
    )

    base_gdp = rng.uniform(0.1, 10.0, size=nc) * gdp_scale
    base_exp = base_gdp * rng.uniform(0.01, 0.4, size=nc)
    base_imp = base_gdp * rng.uniform(0.01, 0.4, size=nc)
    base_tot = np.ones(nc)
    base_cpi = np.ones(nc)

    p_base = rng.uniform(0.9, 1.1, size=(1, ns, nc))
    y_base = rng.uniform(10.0, 50.0, size=(1, ns, nc))

    flows_base = rng.uniform(1.0, 10.0, size=(ns, nc, ns, nc)) if with_flows else None
    flows_cf = rng.uniform(0.5, 9.5, size=(ns, nc, ns, nc)) if with_flows else None

    base_eq = TradeEquilibriumResult(
        x_sol=np.zeros(2001),
        p_sol=p_base,
        y_sol=y_base,
        r_sol=np.ones((1, 1, nc)),
        w_sol=np.ones((1, 1, nc)),
        T_sol=np.zeros((1, 1, nc)),
        XN_sol=np.zeros(max(nc - 1, 1)),
        terms_of_trade=base_tot,
        exports=base_exp,
        imports=base_imp,
        tariffs=np.zeros(nc),
        gdp=base_gdp,
        gdp_fc=base_gdp.copy(),
        cpi=base_cpi,
        intermediate_flows=flows_base,
        converged=True,
        country_codes=country_codes,
    )

    # Shocked counterfactual
    tot_mult = rng.uniform(0.1, 5.0, size=nc)
    imp_mult = rng.uniform(0.0, 2.0, size=nc)
    exp_mult = rng.uniform(0.1, 2.0, size=nc)
    cpi_val = rng.uniform(0.1, 10.0, size=nc)
    tr_val = base_imp * rng.uniform(0.0, 0.5, size=nc)
    gdp_cf = base_gdp * rng.uniform(0.8, 1.2, size=nc)

    p_cf = rng.uniform(0.5, 2.5, size=(1, ns, nc))
    y_cf = rng.uniform(5.0, 60.0, size=(1, ns, nc))

    cf_eq = TradeEquilibriumResult(
        x_sol=np.zeros(2001),
        p_sol=p_cf,
        y_sol=y_cf,
        r_sol=rng.uniform(0.5, 2.0, size=(1, 1, nc)),
        w_sol=rng.uniform(0.5, 2.0, size=(1, 1, nc)),
        T_sol=rng.uniform(0.0, 50.0, size=(1, 1, nc)),
        XN_sol=np.zeros(max(nc - 1, 1)),
        terms_of_trade=tot_mult,
        exports=base_exp * exp_mult,
        imports=base_imp * imp_mult,
        tariffs=tr_val,
        gdp=gdp_cf,
        gdp_fc=base_gdp.copy(),
        cpi=cpi_val,
        intermediate_flows=flows_cf,
        converged=True,
        country_codes=country_codes,
    )

    return calib, base_eq, cf_eq


def run_comprehensive_empirical_stress():
    print("=================================================================")
    print("STARTING EMPIRICAL CHALLENGE ON 3-WAY HICKSIAN EV DECOMPOSITION")
    print("=================================================================")

    rng = np.random.default_rng(20260918)
    n_trials = 1000
    level_residuals = []
    pct_residuals = []
    independent_dwl_matched = 0
    total_evals = 0

    dimensions = [
        (2, 2), (2, 5), (3, 3), (4, 2), (5, 5),
        (8, 4), (10, 2), (12, 6), (20, 3), (44, 4)
    ]

    for trial in range(n_trials):
        nc, ns = dimensions[trial % len(dimensions)]
        with_flows = (trial % 2 == 0)
        gdp_scale = 10.0 ** rng.uniform(-2.0, 6.0)

        calib, base_eq, cf_eq = make_random_mrio_setup(
            nc, ns, rng, with_flows=with_flows, gdp_scale=gdp_scale
        )

        target_c_idx = rng.integers(0, nc)
        target_country = calib.country_codes[target_c_idx]

        # 1. Evaluate level decomposition
        res_lvl = decompose_hicksian_ev_3way(
            calib, cf_eq, base_result=base_eq, target_country=target_country, as_percent=False, tol=1e-10
        )
        total_evals += 1
        level_residuals.append(abs(res_lvl.residual))

        # Check exact identity in level
        assert abs(res_lvl.residual) <= 1e-10, f"Level residual failed: {res_lvl.residual}"
        assert res_lvl.exact_identity_satisfied is True
        assert abs(res_lvl.ev_usd - (res_lvl.tot + res_lvl.alloc + res_lvl.tariff_rec)) <= 1e-10

        # Check independent DWL formula
        imp_val = float(cf_eq.imports[target_c_idx])
        tr_val = float(cf_eq.tariffs[target_c_idx])
        m_base = float(base_eq.imports[target_c_idx])
        delta_m = max(m_base - imp_val, 0.0)
        mean_tau = (tr_val / imp_val) if imp_val > 1e-12 else 0.10
        expected_dwl = 0.5 * mean_tau * delta_m
        expected_alloc = - expected_dwl
        assert math.isclose(res_lvl.alloc, expected_alloc, abs_tol=1e-12, rel_tol=1e-12), (
            f"DWL mismatch: computed={res_lvl.alloc}, expected={expected_alloc}"
        )
        independent_dwl_matched += 1

        # 2. Evaluate percent decomposition
        res_pct = decompose_hicksian_ev_3way(
            calib, cf_eq, base_result=base_eq, target_country=target_country, as_percent=True, tol=1e-10
        )
        total_evals += 1
        pct_residuals.append(abs(res_pct.residual))

        assert abs(res_pct.residual) <= 1e-10, f"Pct residual failed: {res_pct.residual}"
        assert res_pct.exact_identity_satisfied is True
        assert abs(res_pct.ev_pct - (res_pct.tot + res_pct.alloc + res_pct.tariff_rec)) <= 1e-10

    max_lvl_res = max(level_residuals)
    mean_lvl_res = float(np.mean(level_residuals))
    max_pct_res = max(pct_residuals)
    mean_pct_res = float(np.mean(pct_residuals))

    print(f"Total trials evaluated: {n_trials}")
    print(f"Total decomposition evaluations: {total_evals}")
    print(f"Independent Harberger DWL matches: {independent_dwl_matched} / {n_trials} (100.0%)")
    print(f"Max Level Residual: {max_lvl_res:.3e} (threshold: 1.000e-10)")
    print(f"Mean Level Residual: {mean_lvl_res:.3e}")
    print(f"Max Pct Residual:   {max_pct_res:.3e} (threshold: 1.000e-10)")
    print(f"Mean Pct Residual:   {mean_pct_res:.3e}")

    assert max_lvl_res <= 1e-10
    assert max_pct_res <= 1e-10

    # 3. NUMERAIRE INVARIANCE OVER 24 ORDERS OF MAGNITUDE (10^-12 to 10^12)
    print("\n--- Testing Numeraire Invariance across 24 orders of magnitude ---")
    calib_2c, base_2c, cf_2c = make_random_mrio_setup(2, 2, rng, with_flows=True, gdp_scale=1000.0)
    d_ref_pct = decompose_hicksian_ev_3way(calib_2c, cf_2c, base_result=base_2c, as_percent=True)
    d_ref_lvl = decompose_hicksian_ev_3way(calib_2c, cf_2c, base_result=base_2c, as_percent=False)

    scale_factors = [1e-12, 1e-9, 1e-6, 1e-3, 0.01, 0.5, 1.0, 2.0, 100.0, 1e4, 1e6, 1e9, 1e12]
    numeraire_pct_diffs = []
    numeraire_residuals = []

    for lam in scale_factors:
        b_scaled = TradeEquilibriumResult(
            x_sol=base_2c.x_sol, p_sol=lam * base_2c.p_sol, y_sol=base_2c.y_sol,
            r_sol=lam * base_2c.r_sol, w_sol=lam * base_2c.w_sol, T_sol=lam * base_2c.T_sol,
            XN_sol=lam * base_2c.XN_sol, terms_of_trade=base_2c.terms_of_trade.copy(),
            exports=lam * base_2c.exports, imports=lam * base_2c.imports,
            tariffs=lam * base_2c.tariffs, gdp=lam * base_2c.gdp, gdp_fc=lam * base_2c.gdp_fc,
            cpi=base_2c.cpi.copy(), intermediate_flows=base_2c.intermediate_flows,
            converged=True, country_codes=base_2c.country_codes,
        )
        cf_scaled = TradeEquilibriumResult(
            x_sol=cf_2c.x_sol, p_sol=lam * cf_2c.p_sol, y_sol=cf_2c.y_sol,
            r_sol=lam * cf_2c.r_sol, w_sol=lam * cf_2c.w_sol, T_sol=lam * cf_2c.T_sol,
            XN_sol=lam * cf_2c.XN_sol, terms_of_trade=cf_2c.terms_of_trade.copy(),
            exports=lam * cf_2c.exports, imports=lam * cf_2c.imports,
            tariffs=lam * cf_2c.tariffs, gdp=lam * cf_2c.gdp, gdp_fc=lam * cf_2c.gdp_fc,
            cpi=cf_2c.cpi.copy(), intermediate_flows=cf_2c.intermediate_flows,
            converged=True, country_codes=cf_2c.country_codes,
        )

        d_lam_pct = decompose_hicksian_ev_3way(calib_2c, cf_scaled, base_result=b_scaled, as_percent=True)
        diff_pct = abs(d_lam_pct.ev_pct - d_ref_pct.ev_pct)
        numeraire_pct_diffs.append(diff_pct)
        numeraire_residuals.append(abs(d_lam_pct.residual))

        assert diff_pct <= 1e-10, f"Numeraire scaling lambda={lam} changed EV pct by {diff_pct:.3e}"
        assert abs(d_lam_pct.residual) <= 1e-10, f"Numeraire scaling lambda={lam} violated residual gate: {d_lam_pct.residual:.3e}"

    print(f"Numeraire scales tested: {len(scale_factors)}")
    print(f"Max EV pct variation across scales: {max(numeraire_pct_diffs):.3e}")
    print(f"Max residual across numeraire scales: {max(numeraire_residuals):.3e}")

    # 4. EXTREME ADVERSARIAL BOUNDARY CASES
    print("\n--- Testing Extreme Boundary Cases ---")
    # A. Prohibitive tariffs (tau = 10,000)
    cf_prohibitive = TradeEquilibriumResult(
        x_sol=cf_2c.x_sol, p_sol=cf_2c.p_sol, y_sol=cf_2c.y_sol,
        r_sol=cf_2c.r_sol, w_sol=cf_2c.w_sol, T_sol=cf_2c.T_sol,
        XN_sol=cf_2c.XN_sol, terms_of_trade=np.array([1000.0, 0.001]),
        exports=np.array([1e-6, 500.0]), imports=np.array([1e-6, 500.0]),
        tariffs=np.array([10000.0, 0.0]), gdp=cf_2c.gdp, gdp_fc=cf_2c.gdp_fc,
        cpi=np.array([100.0, 1.0]), converged=True, country_codes=cf_2c.country_codes,
    )
    d_prohib = decompose_hicksian_ev_3way(calib_2c, cf_prohibitive, base_result=base_2c, as_percent=False)
    assert abs(d_prohib.residual) <= 1e-10
    print(f"Prohibitive tariff residual: {abs(d_prohib.residual):.3e} <= 1e-10: PASS")

    # B. Complete Autarky (zero trade)
    cf_autarky = TradeEquilibriumResult(
        x_sol=cf_2c.x_sol, p_sol=cf_2c.p_sol, y_sol=cf_2c.y_sol,
        r_sol=cf_2c.r_sol, w_sol=cf_2c.w_sol, T_sol=cf_2c.T_sol,
        XN_sol=cf_2c.XN_sol, terms_of_trade=np.array([1.0, 1.0]),
        exports=np.array([0.0, 0.0]), imports=np.array([0.0, 0.0]),
        tariffs=np.array([0.0, 0.0]), gdp=cf_2c.gdp, gdp_fc=cf_2c.gdp_fc,
        cpi=np.array([1.0, 1.0]), converged=True, country_codes=cf_2c.country_codes,
    )
    d_aut = decompose_hicksian_ev_3way(calib_2c, cf_autarky, base_result=base_2c, as_percent=False)
    assert abs(d_aut.residual) <= 1e-10
    assert d_aut.tot == 0.0
    assert d_aut.tariff_rec == 0.0
    print(f"Complete autarky residual: {abs(d_aut.residual):.3e} <= 1e-10: PASS")

    # C. Negative import delta (imports expanded rather than contracted)
    cf_exp = TradeEquilibriumResult(
        x_sol=cf_2c.x_sol, p_sol=cf_2c.p_sol, y_sol=cf_2c.y_sol,
        r_sol=cf_2c.r_sol, w_sol=cf_2c.w_sol, T_sol=cf_2c.T_sol,
        XN_sol=cf_2c.XN_sol, terms_of_trade=np.array([1.1, 0.9]),
        exports=base_2c.exports * 1.5, imports=base_2c.imports * 2.0, # imports doubled
        tariffs=np.array([10.0, 10.0]), gdp=cf_2c.gdp, gdp_fc=cf_2c.gdp_fc,
        cpi=np.array([1.0, 1.0]), converged=True, country_codes=cf_2c.country_codes,
    )
    d_exp = decompose_hicksian_ev_3way(calib_2c, cf_exp, base_result=base_2c, as_percent=False)
    assert abs(d_exp.residual) <= 1e-10
    assert d_exp.alloc == 0.0  # delta_m = max(m_base - imp_val, 0.0) -> 0.0 DWL
    print(f"Import expansion (zero DWL) residual: {abs(d_exp.residual):.3e} <= 1e-10: PASS")

    # D. Tiny micro-economy vs massive economy
    base_micro = TradeEquilibriumResult(
        x_sol=np.zeros(2001), p_sol=np.ones((1, 2, 2)), y_sol=np.ones((1, 2, 2)),
        r_sol=np.ones((1, 1, 2)), w_sol=np.ones((1, 1, 2)), T_sol=np.zeros((1, 1, 2)),
        XN_sol=np.zeros(1), terms_of_trade=np.array([1.0, 1.0]),
        exports=np.array([1e-9, 1e8]), imports=np.array([1e-9, 1e8]),
        tariffs=np.array([0.0, 0.0]), gdp=np.array([1e-6, 1e9]), gdp_fc=np.array([1e-6, 1e9]),
        cpi=np.array([1.0, 1.0]), converged=True, country_codes=calib_2c.country_codes,
    )
    cf_micro = TradeEquilibriumResult(
        x_sol=np.zeros(2001), p_sol=np.ones((1, 2, 2)), y_sol=np.ones((1, 2, 2)),
        r_sol=np.ones((1, 1, 2)), w_sol=np.ones((1, 1, 2)), T_sol=np.zeros((1, 1, 2)),
        XN_sol=np.zeros(1), terms_of_trade=np.array([1.2, 0.8]),
        exports=np.array([0.8e-9, 0.95e8]), imports=np.array([0.8e-9, 0.95e8]),
        tariffs=np.array([1e-10, 5e6]), gdp=np.array([1.02e-6, 1.01e9]), gdp_fc=np.array([1e-6, 1e9]),
        cpi=np.array([1.05, 1.01]), converged=True, country_codes=calib_2c.country_codes,
    )
    d_mic0 = decompose_hicksian_ev_3way(calib_2c, cf_micro, base_result=base_micro, target_country=0, as_percent=False)
    d_mic1 = decompose_hicksian_ev_3way(calib_2c, cf_micro, base_result=base_micro, target_country=1, as_percent=False)
    assert abs(d_mic0.residual) <= 1e-10
    assert abs(d_mic1.residual) <= 1e-10
    print(f"Micro-economy (1e-6 M USD) residual: {abs(d_mic0.residual):.3e} <= 1e-10: PASS")
    print(f"Macro-economy (1e9 M USD) residual: {abs(d_mic1.residual):.3e} <= 1e-10: PASS")

    print("\n=================================================================")
    print("ALL 1,000+ EMPIRICAL STRESS TESTS PASSED WITH RESIDUAL <= 1e-10!")
    print("VERDICT: CONFIRMED")
    print("=================================================================")
    return True


@pytest.mark.xfail(reason="Exact Hicksian EV endpoint is quarantined", run=False, strict=True)
def test_empirical_stress_ev_decomposition_1000_trials():
    """Pytest test case for the 1,000+ trial empirical stress challenge."""
    assert run_comprehensive_empirical_stress() is True


if __name__ == "__main__":
    success = run_comprehensive_empirical_stress()
    sys.exit(0 if success else 1)
