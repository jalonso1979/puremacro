"""CI-only drift-guard: recompute the LIVE statsmodels reference and assert the
frozen golden in goldens/inference.json is still faithful. This is the single
place statsmodels is touched for the inference subsystem. Run via:
    pytest -m reference
"""
import numpy as np
import pytest

pytestmark = pytest.mark.reference

statsmodels = pytest.importorskip("statsmodels")


def test_inference_hac_golden_matches_live_statsmodels():
    import statsmodels.api as sm

    from puremacro.validation._goldens import load_golden
    from puremacro.validation.cases_inference import hac_demo_data

    d = hac_demo_data()
    res = sm.OLS(d["y"], d["X"]).fit(
        cov_type="HAC", cov_kwds={"maxlags": int(d["lags"])}
    )
    live = np.asarray(res.bse, dtype=float)

    # Both PACKAGE cases (ols_hac and the standalone newey_west alias) share the
    # same statsmodels HAC bse reference.
    # If this fails, statsmodels changed its output — regenerate the golden:
    #   python tools/gen_validation_goldens_inference.py
    for key in ("inference:ols_hac_vs_statsmodels", "inference:newey_west_vs_statsmodels"):
        golden = np.asarray(load_golden(key)["se"], dtype=float)
        np.testing.assert_allclose(live, golden, rtol=1e-6, atol=1e-9)
