"""CI-only drift-guard: recompute the LIVE statsmodels reference and assert the
frozen golden in goldens/var.json is still faithful. This is the single place
statsmodels is touched for validation. Run via: pytest -m reference
"""
import numpy as np
import pytest

pytestmark = pytest.mark.reference

statsmodels = pytest.importorskip("statsmodels")


def test_var_cholesky_golden_matches_live_statsmodels():
    import pandas as pd
    from statsmodels.tsa.api import VAR

    from puremacro.validation._fixtures import var_demo_data
    from puremacro.validation._goldens import load_golden

    d = var_demo_data()
    res = VAR(pd.DataFrame(d["Y"])).fit(d["p"])
    live = np.asarray(res.irf(d["horizon"]).orth_irfs, dtype=float)

    golden = np.asarray(load_golden("var:cholesky_irf_vs_statsmodels")["irf"], dtype=float)
    # If this fails, statsmodels changed its output — regenerate the golden:
    #   python tools/gen_validation_goldens_var.py
    np.testing.assert_allclose(live, golden, rtol=1e-6, atol=1e-9)
