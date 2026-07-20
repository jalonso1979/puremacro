"""CI-only drift-guard: recompute the LIVE statsmodels / linearmodels
references and assert the frozen goldens in goldens/lp.json are still faithful.
This is the single place those packages are touched for the lp subsystem.
Run via: pytest -m reference
"""
import numpy as np
import pytest

pytestmark = pytest.mark.reference

statsmodels = pytest.importorskip("statsmodels")
linearmodels = pytest.importorskip("linearmodels")


def test_lp_goldens_match_live_references():
    from tools.gen_validation_goldens_lp import (
        _ref_lp_hac,
        _ref_lp_iv,
        _ref_panel_lp,
    )

    from puremacro.validation.cases_lp import (
        LP_HORIZONS,
        LP_N_LAGS,
        lp_iv_demo,
        lp_panel_demo,
        lp_ts_demo,
    )
    from puremacro.validation._goldens import load_golden

    H, NL = LP_HORIZONS, LP_N_LAGS

    # If any of these fail, the reference package changed its output —
    # regenerate the golden: python tools/gen_validation_goldens_lp.py
    b_hac, se_hac = _ref_lp_hac(lp_ts_demo(), "y", "x", H, NL)
    np.testing.assert_allclose(
        b_hac,
        np.asarray(load_golden("lp:jorda_hac_beta_vs_statsmodels")["beta"], float),
        rtol=1e-9,
        atol=1e-12,
    )
    np.testing.assert_allclose(
        se_hac,
        np.asarray(load_golden("lp:jorda_hac_se_vs_statsmodels")["se"], float),
        rtol=1e-9,
        atol=1e-12,
    )

    b_iv = _ref_lp_iv(lp_iv_demo(), "y", "x", "z", H, NL)
    np.testing.assert_allclose(
        b_iv,
        np.asarray(load_golden("lp:iv_beta_vs_linearmodels_iv2sls")["beta"], float),
        rtol=1e-9,
        atol=1e-12,
    )

    b_panel = _ref_panel_lp(lp_panel_demo(), "y", "x", H, NL)
    np.testing.assert_allclose(
        b_panel,
        np.asarray(
            load_golden("lp:panel_two_way_fe_beta_vs_linearmodels")["beta"], float
        ),
        rtol=1e-9,
        atol=1e-12,
    )
