"""Default-suite assertion: every registered replication case reproduces its
paper's published result (offline cases only; live-fetch cases carry
@pytest.mark.replication and are deselected here).
"""
import pytest

from puremacro.replication import run_all

_RESULTS = [r for r in run_all() if not r.network]


@pytest.mark.parametrize("res", _RESULTS, ids=[r.id for r in _RESULTS])
def test_case_reproduces_paper(res):
    assert res.passed, f"{res.id} failed (margin={res.margin}); error={res.error}"


def test_aiyagari_cases_present():
    ids = {r.id for r in run_all(family="ha_dsge_causal")}
    assert {
        "ha_dsge_causal.aiyagari_precautionary_wedge",
        "ha_dsge_causal.aiyagari_r_decreasing_in_sigma",
        "ha_dsge_causal.aiyagari_r_decreasing_in_rho",
    } <= ids


def test_huggett_case_present_and_passes():
    results = {r.id: r for r in run_all(family="ha_dsge_causal")}
    r = results.get("ha_dsge_causal.huggett_precautionary_riskfree_rate")
    assert r is not None, "Huggett case missing"
    assert r.target_kind == "sign" and r.passed, f"margin={r.margin}; error={r.error}"


def test_sw07_case_present_and_passes():
    results = {r.id: r for r in run_all(family="ha_dsge_causal")}
    r = results.get("ha_dsge_causal.sw07_seven_shocks_seven_observables")
    assert r is not None, "SW07 case missing"
    assert r.target_kind == "point" and r.passed, f"margin={r.margin}; error={r.error}"


def test_okun_case_present_and_passes():
    results = {r.id: r for r in run_all(family="stylized_facts")}
    r = results.get("stylized_facts.okun_law_fred")
    assert r is not None, "Okun case missing"
    assert r.target_kind == "correlation" and r.passed, f"margin={r.margin}; error={r.error}"


def test_dsge_estimation_cases_present_and_pass():
    results = {r.id: r for r in run_all(family="dsge_estimation")}
    expected_ids = {
        "dsge_estimation.sw07_log_posterior_at_mode",
        "dsge_estimation.sw07_laplace_marginal_data_density",
        "dsge_estimation.sw07_harmonic_mean_mdd_consistency",
        "dsge_estimation.sw07_structural_parameters_mode",
    }
    assert expected_ids <= set(results), f"Missing cases: {expected_ids - set(results)}"
    for cid in expected_ids:
        r = results[cid]
        assert r.passed, f"{cid} failed: margin={r.margin}, error={r.error}"


def test_regression_cases_present_and_pass():
    results = {r.id: r for r in run_all(family="regression")}
    expected_ids = {
        "regression.card1995_iv_vs_ols",
        "regression.long_ervin2000_hc_hierarchy",
        "regression.mroz1987_logit_participation",
        "regression.romer_romer_tax_multiplier_ols",
    }
    assert expected_ids <= set(results), f"Missing cases: {expected_ids - set(results)}"
    for cid in expected_ids:
        r = results[cid]
        assert r.passed, f"{cid} failed: margin={r.margin}, error={r.error}"

