"""Adversarial stress-test suite for DSGE formal rank identification criteria.

Comprehensive empirical verification covering:
1. Identifiable 3-shock New Keynesian model (full rank 9/9, finite condition numbers, empty null spaces).
2. Unidentifiable 1-shock New Keynesian model (rank deficiency 5/6, exact Taylor collinearity 0.8480 phi_pi + 0.5300 phi_y = 0).
3. Planted redundancies: product model, sum model, asymmetric product (rank 1/2, null vector [1/sqrt(2), -1/sqrt(2)]).
4. Near-singular parameter configurations and collinear parameter pair warnings (R^2 > 0.95, cond > 1e4).
5. Publication presentation methods: .rank_scorecard(), .summary(), .to_markdown(), .to_latex(), .to_typst().
6. Adversarial edge cases: unused parameter, single parameter, underdetermined moments (m < n), zero shock variance.
"""
from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

import puremacro.dsge as dsge
from puremacro.dsge.dynare import build_dynare


# =============================================================================
# Helper Model Factories
# =============================================================================

def _create_3shock_nk_model(params: dict[str, float] | None = None) -> dsge.LinearModel:
    """Canonical 3-equation New Keynesian model with 3 exogenous AR(1) shocks."""
    def nk_eqs(lead, curr, lag, shocks, p):
        return [
            curr.y - lead.y + (1.0 / p.sigma) * (curr.r - lead.pi) - curr.g,
            curr.pi - 0.99 * lead.pi - p.kappa * curr.y - curr.u,
            curr.r - p.phi_pi * curr.pi - p.phi_y * curr.y - shocks.eps_r,
            curr.g - p.rho_g * lag.g - shocks.eps_g,
            curr.u - p.rho_u * lag.u - shocks.eps_u,
        ]

    p_dict = dict(
        sigma=1.0,
        kappa=0.1,
        phi_pi=1.5,
        phi_y=0.125,
        rho_g=0.8,
        rho_u=0.5,
    )
    if params is not None:
        p_dict.update(params)

    return build_dynare(
        nk_eqs,
        variables=["y", "pi", "r", "g", "u"],
        shocks=["eps_r", "eps_g", "eps_u"],
        params=p_dict,
        states=["g", "u"],
        guess=dict(y=0.0, pi=0.0, r=0.0, g=0.0, u=0.0),
    )


def _create_1shock_nk_model(params: dict[str, float] | None = None) -> dsge.LinearModel:
    """Canonical 3-equation New Keynesian model with only 1 shock (cost-push shock eps_u)."""
    def nk_1shock_eqs(lead, curr, lag, shocks, p):
        return [
            curr.y - lead.y + (1.0 / p.sigma) * (curr.r - lead.pi),
            curr.pi - 0.99 * lead.pi - p.kappa * curr.y - curr.u,
            curr.r - p.phi_pi * curr.pi - p.phi_y * curr.y,
            curr.u - p.rho_u * lag.u - shocks.eps_u,
        ]

    p_dict = dict(
        sigma=1.0,
        kappa=0.1,
        phi_pi=1.5,
        phi_y=0.125,
        rho_u=0.5,
    )
    if params is not None:
        p_dict.update(params)

    return build_dynare(
        nk_1shock_eqs,
        variables=["y", "pi", "r", "u"],
        shocks=["eps_u"],
        params=p_dict,
        states=["u"],
        guess=dict(y=0.0, pi=0.0, r=0.0, u=0.0),
    )


# =============================================================================
# Adversarial Test Suite
# =============================================================================

def test_identifiable_3shock_nk_rigorous_criteria():
    """Stress-test 1: Identifiable 3-shock NK model full rank 9/9 across all 4 criteria."""
    m = _create_3shock_nk_model()
    p_names = ["sigma", "kappa", "phi_pi", "phi_y", "rho_g", "rho_u", "SE_eps_r", "SE_eps_g", "SE_eps_u"]

    # Baseline identification run
    res = dsge.identification(m, params=p_names, varobs=["y", "pi", "r"], lags=2, n_freq=16)

    # 1. Full rank assertions
    assert res.is_identified is True
    assert res.rank_deficient is False
    assert res.j1_rank == 9
    assert res.j2_rank == 9
    assert res.jh_rank == 9
    assert res.js_rank == 9

    assert res.j1_n_params == 9
    assert res.j2_n_params == 9
    assert res.jh_n_params == 9
    assert res.js_n_params == 9

    # 2. Condition numbers must be strictly finite, > 1.0, and well within ill-conditioning threshold (< 1e4)
    for cond_name, cond_val in [
        ("J1", res.j1_condition_number),
        ("J2", res.j2_condition_number),
        ("JH", res.jh_condition_number),
        ("JS", res.js_condition_number),
    ]:
        assert np.isfinite(cond_val), f"{cond_name} condition number is not finite: {cond_val}"
        assert cond_val > 1.0, f"{cond_name} condition number <= 1.0: {cond_val}"
        assert cond_val < 1000.0, f"{cond_name} condition number too large: {cond_val}"

    # 3. Null spaces must be completely empty (shape 0 x 9)
    assert res.j1_null_space.shape == (0, 9)
    assert res.j2_null_space.shape == (0, 9)
    assert res.jh_null_space.shape == (0, 9)
    assert res.js_null_space.shape == (0, 9)

    assert len(res.j1_null_combinations) == 0
    assert len(res.j2_null_combinations) == 0
    assert len(res.jh_null_combinations) == 0
    assert len(res.js_null_combinations) == 0

    # 4. Invariance to grid type and frequency density
    for freq_type in ["uniform", "gauss_legendre"]:
        for n_f in [8, 16, 32]:
            res_grid = dsge.identification(m, params=p_names, varobs=["y", "pi", "r"], n_freq=n_f, freq_type=freq_type)
            assert res_grid.jh_rank == 9
            assert res_grid.js_rank == 9
            assert np.isfinite(res_grid.jh_condition_number)
            assert np.isfinite(res_grid.js_condition_number)

    # 5. Invariance to autocovariance lag depth
    for p_lag in [1, 2, 4]:
        res_lag = dsge.identification(m, params=p_names, varobs=["y", "pi", "r"], lags=p_lag)
        assert res_lag.j2_rank == 9
        assert np.isfinite(res_lag.j2_condition_number)


def test_unidentifiable_1shock_nk_taylor_collinearity_6params():
    """Stress-test 2: Unidentifiable 1-shock NK model (6 params) verifies 5/6 rank and exact Taylor collinearity."""
    m = _create_1shock_nk_model()
    p_names = ["sigma", "kappa", "phi_pi", "phi_y", "rho_u", "SE_eps_u"]
    res = dsge.identification(m, params=p_names, varobs=["y", "pi", "r", "u"], lags=2, n_freq=16)

    # 1. Rank deficiency across criteria
    assert res.is_identified is False
    assert res.rank_deficient is True
    assert res.j1_rank == 5
    assert res.jh_rank == 5
    assert res.j2_rank == 5
    assert res.js_rank == 5
    assert res.j1_n_params == 6
    assert res.jh_n_params == 6

    # 2. Condition numbers must be infinite
    assert np.isinf(res.j1_condition_number)
    assert np.isinf(res.jh_condition_number)
    assert np.isinf(res.j2_condition_number)
    assert np.isinf(res.js_condition_number)

    # 3. Null space shape: exactly 1 vector in 6-dimensional parameter space
    assert res.j1_null_space.shape == (1, 6)
    assert res.jh_null_space.shape == (1, 6)

    # 4. Verify exact Taylor rule collinearity direction: 0.8480 * phi_pi + 0.5300 * phi_y = 0
    assert len(res.j1_null_combinations) == 1
    assert "0.8480 * phi_pi + 0.5300 * phi_y = 0" in res.j1_null_combinations[0]

    assert len(res.jh_null_combinations) == 1
    assert "0.8480 * phi_pi + 0.5300 * phi_y = 0" in res.jh_null_combinations[0]

    # Verify vector components numerically
    v_j1 = res.j1_null_space[0]
    idx_pi = p_names.index("phi_pi")
    idx_y = p_names.index("phi_y")

    assert v_j1[idx_pi] == pytest.approx(0.8480, abs=5e-4)
    assert v_j1[idx_y] == pytest.approx(0.5300, abs=5e-4)

    # All other parameter components in null vector must be zero within numerical precision (< 1e-10)
    other_indices = [i for i in range(6) if i not in (idx_pi, idx_y)]
    for idx in other_indices:
        assert abs(v_j1[idx]) < 1e-10

    # Same check on Komunjer-Ng JH null vector
    v_jh = res.jh_null_space[0]
    assert v_jh[idx_pi] == pytest.approx(0.8480, abs=5e-4)
    assert v_jh[idx_y] == pytest.approx(0.5300, abs=5e-4)
    for idx in other_indices:
        assert abs(v_jh[idx]) < 1e-10

    # Verify normalization: 0.8480^2 + 0.5300^2 == 1.0
    norm_sq = v_j1[idx_pi]**2 + v_j1[idx_y]**2
    assert norm_sq == pytest.approx(1.0, abs=1e-5)


def test_unidentifiable_1shock_nk_taylor_collinearity_5params():
    """Stress-test 3: Unidentifiable 1-shock NK model (5 params, standard observables) recovers Taylor collinearity."""
    m = _create_1shock_nk_model()
    p_names = ["sigma", "kappa", "phi_pi", "phi_y", "rho_u"]
    res = dsge.identification(m, params=p_names, varobs=["y", "pi", "r"], lags=2)

    assert res.j1_rank == 4
    assert res.jh_rank == 4
    assert res.j1_n_params == 5

    assert len(res.j1_null_combinations) == 1
    assert "0.8480 * phi_pi + 0.5300 * phi_y = 0" in res.j1_null_combinations[0]


def test_planted_redundancies_product_and_sum():
    """Stress-test 4: Planted redundancies (product, sum, asymmetric) recover analytical null spaces."""
    # 1. Product model: y_t = (theta1 * theta2) * y_{t-1} + eps_t
    def eqs_prod(xp, x, e, p):
        return [xp.y - (p.theta1 * p.theta2) * x.y - e.eps]

    m_prod = dsge.build(
        eqs_prod,
        variables=["y"],
        states=["y"],
        shocks=["eps"],
        params=dict(theta1=0.7, theta2=0.7),
        guess=dict(y=0.0),
        linearize="level",
    )
    res_prod = dsge.identification(m_prod, params=["theta1", "theta2"], varobs=["y"])

    assert res_prod.j1_rank == 1
    assert res_prod.j2_rank == 1
    assert res_prod.jh_rank == 1
    assert res_prod.js_rank == 1

    assert np.isinf(res_prod.j1_condition_number)
    assert np.isinf(res_prod.jh_condition_number)

    expected_null = np.array([[1.0 / np.sqrt(2.0), -1.0 / np.sqrt(2.0)]])
    assert np.allclose(res_prod.j1_null_space, expected_null, atol=1e-5)
    assert np.allclose(res_prod.j2_null_space, expected_null, atol=1e-5)
    assert np.allclose(res_prod.jh_null_space, expected_null, atol=1e-5)
    assert np.allclose(res_prod.js_null_space, expected_null, atol=1e-5)

    assert res_prod.j1_null_combinations[0] == "0.7071 * theta1 - 0.7071 * theta2 = 0"

    # 2. Sum model: y_t = (alpha + beta) * y_{t-1} + eps_t
    def eqs_sum(xp, x, e, p):
        return [xp.y - (p.alpha + p.beta) * x.y - e.eps]

    m_sum = dsge.build(
        eqs_sum,
        variables=["y"],
        states=["y"],
        shocks=["eps"],
        params=dict(alpha=0.35, beta=0.35),
        guess=dict(y=0.0),
        linearize="level",
    )
    res_sum = dsge.identification(m_sum, params=["alpha", "beta"], varobs=["y"])

    assert res_sum.j1_rank == 1
    assert res_sum.j2_rank == 1
    assert res_sum.jh_rank == 1
    assert res_sum.js_rank == 1

    assert np.allclose(res_sum.j1_null_space, expected_null, atol=1e-5)
    assert np.allclose(res_sum.jh_null_space, expected_null, atol=1e-5)
    assert np.allclose(res_sum.js_null_space, expected_null, atol=1e-5)

    # 3. Asymmetric product challenge: y_t = (theta1^2 * theta2) * y_{t-1} + eps_t
    # At theta1=0.5, theta2=1.0: gradient is [2*theta1*theta2, theta1^2] = [1.0, 0.25]
    # Orthogonal null vector is [1/sqrt(17), -4/sqrt(17)] = [0.2425356, -0.9701425]
    def eqs_asym(xp, x, e, p):
        return [xp.y - (p.theta1**2 * p.theta2) * x.y - e.eps]

    m_asym = dsge.build(
        eqs_asym,
        variables=["y"],
        states=["y"],
        shocks=["eps"],
        params=dict(theta1=0.5, theta2=1.0),
        guess=dict(y=0.0),
        linearize="level",
    )
    res_asym = dsge.identification(m_asym, params=["theta1", "theta2"], varobs=["y"])

    assert res_asym.j1_rank == 1
    assert res_asym.jh_rank == 1
    expected_asym_null = np.array([[1.0 / np.sqrt(17.0), -4.0 / np.sqrt(17.0)]])
    assert np.allclose(res_asym.j1_null_space, expected_asym_null, atol=1e-3)
    assert "0.2425 * theta1 - 0.9701 * theta2 = 0" in res_asym.j1_null_combinations[0]


def test_near_singular_configurations_and_collinear_warnings():
    """Stress-test 5: Near-singular configurations trigger R^2 > 0.95 and condition number warnings."""
    # 1. Product model triggers R^2 = 1.0 collinear pair warnings
    def eqs_prod(xp, x, e, p):
        return [xp.y - (p.theta1 * p.theta2) * x.y - e.eps]

    m_prod = dsge.build(
        eqs_prod,
        variables=["y"],
        states=["y"],
        shocks=["eps"],
        params=dict(theta1=0.7, theta2=0.7),
        guess=dict(y=0.0),
        linearize="level",
    )
    res_prod = dsge.identification(m_prod, params=["theta1", "theta2"], varobs=["y"])

    assert len(res_prod.collinear_pairs) > 0
    p1, p2, r2, crit = res_prod.collinear_pairs[0]
    assert {p1, p2} == {"theta1", "theta2"}
    assert r2 > 0.999

    warn_text = " ".join(res_prod.warnings)
    assert "rank deficient" in warn_text
    assert "infinite condition number" in warn_text
    assert "High parameter collinearity" in warn_text

    # 2. Near-unit-root model triggers ill-conditioned J2 warning
    def eqs_ar1(xp, x, e, p):
        return [xp.y - p.rho * x.y - e.eps]

    m_near = dsge.build(
        eqs_ar1,
        variables=["y"],
        states=["y"],
        shocks=["eps"],
        params=dict(rho=0.9999),
        guess=dict(y=0.0),
        linearize="level",
    )
    res_near = dsge.identification(m_near, params=["rho", "SE_eps"], varobs=["y"])
    near_warns = " ".join(res_near.warnings)
    assert "Near-singular condition number in J2" in near_warns
    assert "High parameter collinearity between 'rho' and 'SE_eps'" in near_warns

    # 3. Near determinacy boundary in Taylor rule (phi_pi = 1.001, phi_y = 0.0)
    m_bd = _create_3shock_nk_model(params=dict(phi_pi=1.001, phi_y=0.0))
    res_bd = dsge.identification(m_bd, params=["phi_pi", "phi_y"], varobs=["y", "pi", "r"])
    bd_warns = " ".join(res_bd.warnings)
    assert "High parameter collinearity between 'phi_pi' and 'phi_y'" in bd_warns


def test_publication_presentation_suite():
    """Stress-test 6: Verifies .rank_scorecard(), .summary(), .to_markdown(), .to_latex(), .to_typst()."""
    m = _create_3shock_nk_model()
    p_names = ["sigma", "kappa", "phi_pi", "phi_y", "rho_g", "rho_u", "SE_eps_r", "SE_eps_g", "SE_eps_u"]
    res = dsge.identification(m, params=p_names, varobs=["y", "pi", "r"], lags=2)

    # 1. Scorecard DataFrame
    sc = res.rank_scorecard()
    assert isinstance(sc, pd.DataFrame)
    assert list(sc.index) == ["J1", "J2", "JH", "JS"]
    assert list(sc.columns) == ["criterion", "rank", "total", "deficiency", "condition_number", "status"]
    assert np.all(sc["rank"] == 9)
    assert np.all(sc["total"] == 9)
    assert np.all(sc["deficiency"] == 0)
    assert np.all(sc["status"] == "FULL RANK")

    # 2. Summary text
    summ = res.summary()
    assert isinstance(summ, str)
    assert "PARAMETER IDENTIFICATION ANALYSIS" in summ
    assert "Overall status         : IDENTIFIED" in summ
    assert "RANK CRITERIA SCORECARD" in summ
    assert "None (All parameters locally identified)" in summ
    assert "PARAMETER IDENTIFICATION SUMMARY" in summ

    # 3. Markdown exports
    md_sc = res.to_markdown(table="scorecard")
    assert isinstance(md_sc, str)
    assert "|" in md_sc
    assert "J1 (Iskrev Solution)" in md_sc
    assert "FULL RANK" in md_sc

    md_params = res.to_markdown(table="parameters")
    assert "|" in md_params
    assert "sigma" in md_params
    assert "phi_pi" in md_params

    # 4. LaTeX exports
    ltx_sc = res.to_latex(table="scorecard")
    assert isinstance(ltx_sc, str)
    assert r"\begin{tabular}" in ltx_sc
    assert r"\end{tabular}" in ltx_sc
    assert "J1 (Iskrev Solution)" in ltx_sc

    ltx_params = res.to_latex(table="parameters")
    assert r"\begin{tabular}" in ltx_params
    assert r"\end{tabular}" in ltx_params
    assert "sigma" in ltx_params

    # 5. Typst exports
    typ_sc = res.to_typst(table="scorecard")
    assert isinstance(typ_sc, str)
    assert "#table(" in typ_sc
    assert "[J1]" in typ_sc

    typ_params = res.to_typst(table="parameters")
    assert "#table(" in typ_params
    assert "[sigma]" in typ_params


def test_adversarial_edge_cases():
    """Stress-test 7: Edge cases: unused parameter, single parameter, underdetermined moments."""
    # 1. Unused parameter: must have sensitivity 0 and null combination 1.0000 * dummy = 0
    def eqs_dummy(xp, x, e, p):
        return [xp.y - p.rho * x.y - e.eps]

    m_dum = dsge.build(
        eqs_dummy,
        variables=["y"],
        states=["y"],
        shocks=["eps"],
        params=dict(rho=0.7, dummy=1.0),
        guess=dict(y=0.0),
        linearize="level",
    )
    res_dum = dsge.identification(m_dum, params=["rho", "dummy"], varobs=["y"])
    assert res_dum.j1_rank == 1
    assert res_dum.j1_n_params == 2
    assert "1.0000 * dummy = 0" in res_dum.j1_null_combinations[0]
    assert res_dum.strength.loc["dummy", "sensitivity"] == 0.0

    # 2. Single parameter model (no division by zero in collinearity)
    res_single = dsge.identification(m_dum, params=["rho"], varobs=["y"])
    assert res_single.j1_rank == 1
    assert res_single.j1_n_params == 1
    assert res_single.j1_condition_number == 1.0
    assert res_single.j1_collinearity.loc["rho", "r2"] == 0.0

    # 3. Underdetermined moments: m < n (lags=0 gives only 1 moment Gamma_0 for 3 parameters)
    res_und = dsge.identification(m_dum, params=["rho", "SE_eps", "ME_y"], varobs=["y"], lags=0)
    assert res_und.j2_rank == 1
    assert res_und.j2_n_params == 3
    assert res_und.j2_null_space.shape == (2, 3)  # Null space dimension is n - rank = 3 - 1 = 2
    assert len(res_und.j2_null_combinations) == 2

    # 4. Custom frequency array with single frequency node
    res_single_freq = dsge.identification(m_dum, params=["rho"], varobs=["y"], frequencies=[1.5])
    assert res_single_freq.jh_rank == 1
    assert res_single_freq.js_rank == 1
