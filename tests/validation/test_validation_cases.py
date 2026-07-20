"""Default-suite assertion: every registered validation case passes.

Pure (no statsmodels): PACKAGE cases compare to frozen goldens; ANALYTICAL/
SCIPY/INTERNAL compute live. The live-package drift-guard is in
test_reference_drift.py (``-m reference``).
"""
import pytest

from puremacro.validation import run_all

_RESULTS = run_all()


@pytest.mark.parametrize("res", _RESULTS, ids=[r.id for r in _RESULTS])
def test_case_passes(res):
    assert res.passed, f"{res.id} failed (margin={res.max_margin}); error={res.error}"


def test_var_internal_cases_present():
    ids = {r.id for r in run_all(subsystem="var")}
    assert {"var.fevd_sums_to_one", "var.stability_iff_spectral_radius"} <= ids


def test_var_cholesky_package_case_passes_against_golden():
    results = {r.id: r for r in run_all(subsystem="var")}
    r = results.get("var.cholesky_irf_vs_statsmodels")
    assert r is not None, "Cholesky PACKAGE case missing"
    assert r.mechanism == "package" and r.tol == "tight"
    assert r.passed, f"margin={r.max_margin}; error={r.error}"
