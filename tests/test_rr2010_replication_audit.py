"""Independent regression-design checks beyond the frozen published oracle."""
import numpy as np
import pandas as pd
import pytest
from numpy.testing import assert_allclose

from puremacro.replication.romer_romer_2010 import estimate_rr2010_baseline


@pytest.fixture
def known_experiment():
    """Construct a full-rank experiment with exactly known OLS coefficients.

    Residuals are projected off the design, so the target coefficients are
    known without fitting the production estimator or using its outputs.
    Tax observations before 1950 remain nonzero and essential to lag twelve.
    """
    rng = np.random.default_rng(5820)
    dates = pd.period_range("1947Q1", "2007Q4", freq="Q", name="quarter")
    tax = rng.normal(size=len(dates))
    rows = []
    for quarter in range(12, len(dates)):
        rows.append([1., *tax[quarter - np.arange(13)]])
    design = np.asarray(rows)
    coefficients = np.r_[.62, np.linspace(-.3, .21, 13)]
    q, r = np.linalg.qr(design, mode="reduced")
    disturbance = rng.normal(scale=.1, size=len(design))
    disturbance -= q @ (q.T @ disturbance)
    growth = np.r_[np.zeros(12), design @ coefficients + disturbance]
    nominal = np.linspace(150., 1500., len(dates))
    frame = pd.DataFrame({"gdp": 50 * np.exp(np.cumsum(growth) / 100),
                          "nomgdp": nominal, "defic": tax * nominal / 100,
                          "longr": np.zeros(len(dates))}, index=dates)
    frame.attrs["is_synthetic"] = True
    inverse_r = np.linalg.solve(r, np.eye(14))
    covariance = disturbance @ disturbance / 218 * inverse_r @ inverse_r.T
    return frame, coefficients, covariance


def test_known_experiment_recovers_all_lags_and_exact_conventional_covariance(known_experiment):
    frame, coefficients, covariance = known_experiment
    actual = estimate_rr2010_baseline(frame)
    assert_allclose(actual.coefficients, coefficients, rtol=1e-11, atol=1e-13)
    assert_allclose(actual.coefficient_covariance, covariance, rtol=1e-10, atol=1e-14)
    expected_irf = np.cumsum(coefficients[1:])
    # Explicit double sums audit every cross-horizon covariance and exclude
    # the intercept, independently of the production accumulation matrix.
    expected_covariance = np.array([
        [sum(covariance[i, j] for i in range(1, h + 2) for j in range(1, k + 2))
         for k in range(13)] for h in range(13)
    ])
    assert_allclose(actual.irf, expected_irf, atol=1e-12)
    assert_allclose(actual.irf_covariance, expected_covariance, rtol=1e-10, atol=1e-14)
    assert actual.nobs == 232 and actual.df_resid == 218
    assert actual.metadata["is_synthetic"] is True
    assert actual.metadata["input_origin"] == "caller-supplied sensitivity data"


def test_rebasing_output_and_currency_does_not_change_estimates_or_uncertainty(known_experiment):
    frame, _, _ = known_experiment
    baseline = estimate_rr2010_baseline(frame)
    rescaled = frame.copy()
    rescaled["gdp"] *= 350.
    rescaled[["nomgdp", "defic", "longr"]] *= 1000.
    changed = estimate_rr2010_baseline(rescaled)
    for name in ("coefficients", "coefficient_covariance", "irf", "irf_covariance",
                 "standard_errors", "t_statistics"):
        assert_allclose(getattr(changed, name), getattr(baseline, name), rtol=2e-10, atol=1e-11)


def test_gdp_growth_before_sample_is_excluded_but_presample_tax_is_used(known_experiment):
    frame, _, _ = known_experiment
    baseline = estimate_rr2010_baseline(frame)
    changed = frame.copy()
    # 1949Q4 GDP is needed for the first growth observation. Earlier GDP
    # levels must be irrelevant in the no-output-controls specification.
    changed.loc[:"1949Q3", "gdp"] *= np.linspace(.4, 2., 11)
    result = estimate_rr2010_baseline(changed)
    assert_allclose(result.coefficients, baseline.coefficients, atol=1e-14)
    assert_allclose(result.coefficient_covariance, baseline.coefficient_covariance, atol=1e-14)
    # The 1947Q1 shock enters exactly the first sample row as tax lag twelve.
    changed = frame.copy()
    changed.loc["1947Q1", "defic"] += changed.loc["1947Q1", "nomgdp"] * .02
    result = estimate_rr2010_baseline(changed)
    assert np.linalg.norm(result.coefficients - baseline.coefficients) > 1e-3


def test_absent_tax_variation_cannot_produce_published_style_inference(known_experiment):
    frame, _, _ = known_experiment
    frame[["defic", "longr"]] = 0.
    with pytest.raises((ValueError, np.linalg.LinAlgError)):
        estimate_rr2010_baseline(frame)


def test_real_published_t_discrepancy_cannot_be_hidden_by_software_parity():
    from puremacro.validation import research_benchmarks, run_research_benchmarks

    cases = [case for case in research_benchmarks() if case.id.startswith("rr2010_")]
    report = run_research_benchmarks(cases)
    results = {result.id: result for result in report.results}
    assert results["rr2010_baseline_software"].passed
    printed = results["rr2010_published_peak"]
    assert not printed.passed and not report.passed
    assert printed.metrics["response_h10"]["passed"]
    assert printed.metrics["trough_horizon"]["passed"]
    assert not printed.metrics["t_h10"]["passed"]
    assert .005 < printed.metrics["t_h10"]["max_abs_error"] < .00501
    assert printed.provenance["atol"] == .005 and printed.provenance["rtol"] == 0
    exported = report.to_dict()
    assert exported["passed"] is False
    failed_case = next(case for case in exported["cases"] if case["id"] == printed.id)
    assert failed_case["metrics"]["t_h10"]["passed"] is False
